"""CLI Typer — commandes Rune (édition complète).

Différences vs le prototype léger :
- Utilise Hippocampe Lythea complet (42 000 lignes de cognition)
- Wrappé par RuneCortex qui ajoute AutoSkill, FailureMemory,
  TieredRetriever, SubAgent, Cron
- Backend modèle : HFModelWrapper de Lythea (avec hooks PyTorch, steering,
  mémoire SDM/MHN/KG/Chroma) au lieu du MockBackend

Commandes :
    rune chat            # mode interactif console
    rune run "task"      # exécute une mission (subagent)
    rune serve           # démarre l'API HTTP (FastAPI Lythea)
    rune skills list     # liste les skills
    rune skills show ID  # affiche un skill
    rune skills archive ID
    rune cron list       # liste les tâches cron
    rune cron add ...    # ajoute une tâche
    rune cron run ID     # exécute une tâche immédiatement
    rune cron start      # démarre le scheduler en arrière-plan
    rune status          # affiche le statut global
    rune consolidate     # force un microsleep
    rune deep-sleep      # force un deep sleep
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rune",
    help="Agent IA cognitif local, headless, open-weights only",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
skills_app = typer.Typer(help="Gestion des compétences auto-apprises")
cron_app = typer.Typer(help="Gestion des tâches cron")
app.add_typer(skills_app, name="skills")
app.add_typer(cron_app, name="cron")

console = Console()
log = logging.getLogger("rune.cli")


def _build_rune():
    """Construit RuneCortex complet (Hippocampe Lythea + extensions).

    Cette fonction est lourde — elle instancie Hippocampe avec SDM, MHN,
    KG, Chroma, model, etc. À n'appeler qu'une fois par processus.
    """
    from rune.cortex_ext.integration import RuneCortex
    from rune.server.app import LytheaApp

    # LytheaApp construit Hippocampe + tous les subsystems
    lythea_app = LytheaApp()
    rune = RuneCortex(lythea_app.hippocampe)
    return rune, lythea_app


def _build_lightweight():
    """Version légère pour les commandes qui ne nécessitent pas Hippocampe.

    Utilise seulement AutoSkillStore + FailureMemory sans Hippocampe.
    Pratique pour `skills list` ou `cron list` sans boot complet.
    """
    from rune.memory.auto_skill import AutoSkillStore
    from rune.memory.failure_memory import FailureMemory
    from rune.agents.cron import CronScheduler
    from pathlib import Path

    skills = AutoSkillStore(storage_dir=Path("data/skills"))
    failures = FailureMemory(storage_dir=Path("data/failures"))
    cron = CronScheduler(storage_dir=Path("data/cron"))
    return skills, failures, cron


# ── Commandes ─────────────────────────────────────────────────────────


@app.command()
def chat():
    """Mode interactif console — nécessite le boot complet."""
    from rune.channels.console import ConsoleChannel
    from rune.channels.base import OutgoingMessage

    console.print("[bold cyan]Rune[/] — boot en cours…", style="cyan")
    rune, _ = _build_rune()
    console.print("[green]Prêt.[/] Tape /quit pour sortir.\n")

    def handler(msg):
        # process_message est un generator (SSE events) — on cumule
        # les tokens partial + on récupère le texte final depuis done.
        # Les events "error" non-fatals sont loggés mais n'interrompent
        # pas le flux (Lythea peut émettre des erreurs sur web search,
        # reasoning, etc. tout en continuant à générer).
        final_text = ""
        partial_chunks = []
        errors = []
        for event in rune.process_message(msg.text, history=[]):
            event_type = event.get("type", "")
            if event_type == "partial":
                # Lythea stream les tokens via le champ "token" ou "text"
                chunk = event.get("token") or event.get("text") or event.get("chunk", "")
                if chunk:
                    partial_chunks.append(chunk)
            elif event_type == "done":
                # Le event done peut contenir le texte complet, ou pas
                final_text = event.get("text") or event.get("content") or ""
            elif event_type == "error":
                # Logge mais n'interrompt pas — Lythea peut continuer après
                err_msg = (
                    event.get("message")
                    or event.get("error")
                    or event.get("detail")
                    or event.get("reason")
                    or str(event)
                )
                errors.append(err_msg)
        # Si done n'avait pas le texte, on reconstruit depuis les partials
        if not final_text and partial_chunks:
            final_text = "".join(partial_chunks)
        if not final_text:
            if errors:
                return OutgoingMessage(text=f"(pas de réponse — erreurs: {' | '.join(errors[:2])})")
            return OutgoingMessage(text="(pas de réponse)")
        return OutgoingMessage(text=final_text)

    channel = ConsoleChannel(message_handler=handler)
    channel.start()


@app.command()
def run(
    task: str = typer.Argument(..., help="La mission à exécuter"),
    context: str = typer.Option("", "--context", "-c", help="Contexte additionnel"),
    timeout: float = typer.Option(120.0, "--timeout", "-t", help="Timeout en secondes"),
    lightweight: bool = typer.Option(
        False, "--lightweight",
        help="Utilise le SubAgentSpawner standalone (sans boot Hippocampe)",
    ),
):
    """Exécute une mission ponctuelle via subagent."""
    if lightweight:
        from rune.agents.subagent import SubAgentConfig, SubAgentSpawner
        spawner = SubAgentSpawner(SubAgentConfig(timeout_sec=timeout))
        result = spawner.run(task=task, context=context)
        console.print_json(data=result.as_dict())
    else:
        console.print("[cyan]Boot Hippocampe…[/]")
        rune, _ = _build_rune()
        result = rune.run_subagent(task=task, context=context, timeout=timeout)
        console.print_json(data=result)


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Host API"),
    port: Optional[int] = typer.Option(None, "--port", help="Port API"),
):
    """Démarre l'API HTTP Lythea (FastAPI complet).

    Gère proprement le cas où le port est déjà pris (au lieu de segfaulter
    sur le shutdown uvicorn). Si le port est occupé, affiche un message
    clair et les commandes pour libérer le port.
    """
    import os
    import signal
    import socket
    import sys
    import uvicorn
    from rune.server.app import create_app

    api_host = host or "0.0.0.0"
    api_port = port or 7860

    # ── Vérifie que le port est libre avant de lancer uvicorn ────────
    # Évite le segfault uvicorn sur shutdown après bind error.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((api_host, api_port))
    except OSError as exc:
        if "Address already in use" in str(exc) or exc.errno == 98:
            console.print(
                f"[red]Erreur : le port {api_port} est déjà utilisé.[/]\n\n"
                f"Libère le port avec :\n"
                f"  [cyan]fuser -k {api_port}/tcp[/]\n"
                f"  ou\n"
                f"  [cyan]pkill -9 -f 'rune serve'[/]\n\n"
                f"Puis relance : [cyan]rune serve --port {api_port}[/]"
            )
            raise typer.Exit(1)
        raise

    api_app = create_app()
    console.print(
        f"[bold green]Rune API[/] sur http://{api_host}:{api_port}"
    )

    # ── Gestion propre du Ctrl+C et signaux ──────────────────────────
    # Évite le segfault uvicorn sur interruption.
    def _signal_handler(signum, frame):
        console.print("\n[yellow]Arrêt en cours…[/]")
        # uvicorn gère le SIGINT/SIGTERM lui-même via son lifespan
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        uvicorn.run(api_app, host=api_host, port=api_port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Arrêté.[/]")
    except SystemExit:
        pass
    except Exception as exc:
        console.print(f"[red]Erreur serveur : {exc}[/]")
        raise typer.Exit(1)


@app.command()
def status():
    """Affiche le statut global (version légère sans boot complet)."""
    skills, failures, cron = _build_lightweight()
    data = {
        "skills": skills.stats(),
        "failures": failures.stats(),
        "cron": cron.status(),
    }
    console.print_json(data=data)


# ── Dashboard ─────────────────────────────────────────────────────────


@app.command()
def dashboard(
    api: str = typer.Option("http://localhost:7860", "--api", help="URL API Rune"),
    refresh: float = typer.Option(0.5, "--refresh", help="Intervalle refresh (s)"),
):
    """Dashboard ASCII live — affiche mission, subagents, mémoire, blackboard."""
    from rune.cli.dashboard import run_dashboard
    run_dashboard(api_url=api, refresh_sec=refresh, mock=False)


@app.command()
def demo_dashboard():
    """Dashboard ASCII live avec données mockées (sans serveur)."""
    from rune.cli.dashboard import run_dashboard
    run_dashboard(mock=True)


# ── Skills ────────────────────────────────────────────────────────────


@skills_app.command("list")
def skills_list():
    """Liste les skills actifs."""
    skills, _, _ = _build_lightweight()
    active = skills.active()

    table = Table(title="Compétences auto-apprises")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Trigger", style="white")
    table.add_column("Succès", justify="right", style="green")
    table.add_column("Échecs", justify="right", style="red")
    table.add_column("Confiance", justify="right")
    table.add_column("Fiable?", justify="center")

    for s in active:
        table.add_row(
            s.skill_id,
            s.trigger[:50] + ("…" if len(s.trigger) > 50 else ""),
            str(s.success_count),
            str(s.failure_count),
            f"{s.confidence:.2f}",
            "✓" if s.is_reliable() else "✗",
        )
    console.print(table)
    console.print(f"\n[dim]{skills.stats()}[/]")


@skills_app.command("show")
def skills_show(skill_id: str):
    """Affiche le détail d'un skill."""
    skills, _, _ = _build_lightweight()
    skill = skills.get(skill_id)
    if skill is None:
        console.print(f"[red]Skill {skill_id} introuvable[/]")
        raise typer.Exit(1)
    console.print_json(data=skill.as_dict())


@skills_app.command("archive")
def skills_archive(skill_id: str):
    """Archive un skill."""
    skills, _, _ = _build_lightweight()
    skills.archive(skill_id)
    console.print(f"[green]Skill {skill_id} archivé[/]")


@skills_app.command("compose")
def skills_compose(
    skill_ids: list[str] = typer.Argument(..., help="IDs des skills à composer (2 minimum)"),
    strategy: str = typer.Option(
        "sequential", "--strategy", "-s",
        help="Stratégie : sequential | parallel | conditional | pipeline",
    ),
    trigger: str = typer.Option("", "--trigger", "-t", help="Trigger personnalisé"),
    force: bool = typer.Option(False, "--force", help="Compose même si skills non fiables"),
):
    """Compose plusieurs skills en une nouvelle."""
    skills, _, _ = _build_lightweight()
    result = skills.compose(
        skill_ids=skill_ids,
        strategy=strategy,
        composed_trigger=trigger or None,
        force=force,
    )
    console.print_json(data=result)


@skills_app.command("candidates")
def skills_candidates(
    max_pairs: int = typer.Option(5, "--max", "-n", help="Nombre max de paires"),
):
    """Trouve des paires de skills composables."""
    skills, _, _ = _build_lightweight()
    pairs = skills.find_composable_candidates(max_pairs=max_pairs)
    if not pairs:
        console.print("[dim]Aucune paire composable trouvée.[/]")
        return

    table = Table(title="Paires de skills composables")
    table.add_column("Skill A", style="cyan")
    table.add_column("Skill B", style="cyan")
    table.add_column("Potential", justify="right", style="green")

    for p in pairs:
        table.add_row(
            p["skill_a"]["id"][:20],
            p["skill_b"]["id"][:20],
            f"{p['potential']:.2f}",
        )
    console.print(table)


# ── Cron ──────────────────────────────────────────────────────────────


@cron_app.command("list")
def cron_list():
    """Liste les tâches cron."""
    _, _, cron = _build_lightweight()
    tasks = cron.list_tasks()

    table = Table(title="Tâches cron")
    table.add_column("ID", style="cyan")
    table.add_column("Schedule", style="yellow")
    table.add_column("Action", style="white")
    table.add_column("Enabled", justify="center")
    table.add_column("Runs", justify="right")
    table.add_column("Succès", justify="right", style="green")

    for t in tasks:
        table.add_row(
            t.task_id,
            t.schedule,
            t.action[:40] + ("…" if len(t.action) > 40 else ""),
            "✓" if t.enabled else "✗",
            str(t.run_count),
            str(t.success_count),
        )
    console.print(table)


@cron_app.command("add")
def cron_add(
    task_id: str = typer.Option(..., "--id", help="ID de la tâche"),
    schedule: str = typer.Option(..., "--schedule", help="Schedule (cron unix ou every:Ns)"),
    action: str = typer.Option(..., "--action", help="Mission à exécuter"),
):
    """Ajoute une tâche cron."""
    from rune.agents.cron import CronTask
    _, _, cron = _build_lightweight()
    task = CronTask(task_id=task_id, schedule=schedule, action=action)
    cron.add_task(task)
    console.print(f"[green]Tâche {task_id} ajoutée[/]")


@cron_app.command("run")
def cron_run(task_id: str):
    """Exécute une tâche cron immédiatement."""
    _, _, cron = _build_lightweight()
    result = cron.run_task_now(task_id)
    console.print_json(data=result.as_dict())


@cron_app.command("start")
def cron_start():
    """Démarre le scheduler cron en arrière-plan (bloquant)."""
    _, _, cron = _build_lightweight()
    cron.start()
    console.print("[green]Scheduler cron démarré. Ctrl+C pour arrêter.[/]")
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        cron.stop()
        console.print("\n[yellow]Arrêté.[/]")


# ── Consolidation ─────────────────────────────────────────────────────


@app.command()
def consolidate():
    """Force un microsleep (nécessite le boot complet)."""
    console.print("[cyan]Boot Hippocampe…[/]")
    rune, _ = _build_rune()
    result = rune.consolidation_scheduler.maybe_microsleep(force=True)
    console.print_json(data=result or {"status": "skipped"})


@app.command("deep-sleep")
def deep_sleep():
    """Force un deep sleep (nécessite le boot complet)."""
    console.print("[cyan]Boot Hippocampe…[/]")
    rune, _ = _build_rune()
    result = rune.consolidation_scheduler.deep_sleep()
    console.print_json(data=result)


if __name__ == "__main__":
    app()
