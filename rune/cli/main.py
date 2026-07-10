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

# Charge .env dans os.environ AVANT tout le reste. La CLI ne passe pas
# par run.py (qui appelle bootstrap_env), donc sans ça les variables du
# .env lues via os.getenv (ex: LYTHEA_WEB_PROVIDER pour le provider web,
# les clés API Tavily/Serper) resteraient invisibles — mettre
# LYTHEA_WEB_PROVIDER=auto dans .env n'aurait aucun effet en `rune chat`.
from rune.env import bootstrap_env as _bootstrap_env

_bootstrap_env()

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

    Important : LytheaApp.__init__ construit déjà un RuneCortex interne
    et monkey-patche hippocampe.process_message. On réutilise ce RuneCortex
    au lieu d'en créer un second (évite le double traitement AutoSkill /
    FailureMemory par message, et la double boucle de consolidation).
    """
    from rune.server.app import LytheaApp

    lythea_app = LytheaApp()

    # RuneCortex déjà construit et monkey-patché par LytheaApp.__init__
    rune = lythea_app.rune_cortex
    if rune is None:
        # Fallback : RuneCortex n'a pas pu s'initialiser (import error).
        # On en crée un minimal pour que le CLI reste utilisable.
        from rune.cortex_ext.integration import RuneCortex
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
def chat(
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model ID HuggingFace à charger (ex: Qwen/Qwen2.5-7B-Instruct). "
             "Défaut : RUNE_DEFAULT_MODEL, sinon LYTHEA_MODEL_ID, sinon le "
             "modèle par défaut du catalogue.",
    ),
    no_model: bool = typer.Option(
        False, "--no-model",
        help="Ne charge aucun modèle (chat impossible, utile pour debug boot).",
    ),
    searxng: bool = typer.Option(
        False, "--searxng",
        help="Lance un SearXNG local pour une recherche web fiable "
             "(clone + install au 1er run, ~1-2 min). Sans cette option, "
             "un SearXNG local déjà lancé est détecté et utilisé "
             "automatiquement ; sinon la recherche web retombe sur DDG.",
    ),
):
    """Mode interactif console — nécessite le boot complet.

    Charge automatiquement un modèle au démarrage (sauf --no-model), car
    sans modèle en VRAM le chat renvoie « Aucun modèle chargé ».
    """
    import os
    from rune.channels.console import ConsoleChannel
    from rune.channels.base import OutgoingMessage

    console.print("[bold cyan]Rune[/] — boot en cours…", style="cyan")

    # ── Recherche web : détecte/lance un SearXNG local ────────────────
    # Les instances SearXNG publiques sont rate-limitées par Google
    # (« All SearXNG instances failed »). launch.sh lance un SearXNG
    # local, mais pas rune chat — on reproduit ça ici. Sans --searxng,
    # on se contente de détecter une instance déjà lancée ; avec, on la
    # bootstrap. Dans tous les cas, s'il n'y en a pas, la chaîne web
    # retombe sur DDG (mode auto).
    from rune.env import ensure_local_searxng
    _sx = ensure_local_searxng(autostart=searxng)
    if _sx:
        # Garde le mode composite (auto) : SearXNG d'abord, DDG en repli.
        os.environ.setdefault("LYTHEA_WEB_PROVIDER", "auto")

    rune, lythea_app = _build_rune()

    # ── Chargement du modèle ──────────────────────────────────────────
    # _build_rune() construit Hippocampe mais NE charge PAS de modèle
    # (l'autoload du boot ne s'applique qu'au serveur). En CLI, on charge
    # ici, sinon process_message échoue avec le code "no_model".
    if not no_model:
        from rune.config import DEFAULT_MODEL
        model_id = (
            model
            or os.environ.get("RUNE_DEFAULT_MODEL")
            or os.environ.get("LYTHEA_MODEL_ID")
            or os.environ.get("LYTHEA_DEFAULT_MODEL")
            or DEFAULT_MODEL
        )
        if lythea_app.model.is_loaded:
            console.print(f"[green]Modèle déjà chargé : {lythea_app.model.model_id}[/]")
        else:
            console.print(f"[cyan]Chargement du modèle {model_id}…[/] "
                          f"[dim](peut prendre 1-3 min au 1er lancement)[/]")
            try:
                lythea_app.model.load(model_id)
                console.print(f"[green]Modèle chargé : {model_id}[/]")
            except Exception as exc:
                console.print(
                    f"[red]Échec du chargement de {model_id} : {exc}[/]\n"
                    f"[yellow]Le chat va démarrer mais ne pourra pas répondre. "
                    f"Essaie un autre modèle avec [cyan]--model <id>[/].[/]"
                )
    else:
        console.print("[yellow]--no-model : aucun modèle chargé, le chat ne "
                      "pourra pas répondre.[/]")

    # ── Silence les barres de progression du chat ─────────────────────
    # Le modèle principal est chargé (barre utile ci-dessus). À partir
    # d'ici, les modèles auxiliaires (GLiNER, cross-encoder, ONNX Chroma)
    # se chargent paresseusement à chaque message et polluent la console
    # de barres « Loading weights / Fetching files ». On les coupe une
    # fois le chat prêt : tqdm + HF hub silencieux, et on filtre le
    # warning HF_TOKEN qui se répète.
    import os as _os
    import warnings as _warnings
    _os.environ["TQDM_DISABLE"] = "1"
    _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from huggingface_hub.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        import huggingface_hub as _hfh
        _hfh.utils.disable_progress_bars()
    except Exception:
        pass
    _warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
    _warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

    console.print("[green]Prêt.[/] Tape /quit pour sortir.\n")

    def handler(msg):
        # process_message est un generator qui yield des events au format
        # {"type": <type>, "data": {...}}. Les champs utiles sont DANS
        # event["data"], pas à plat :
        #   - partial : data["text"]        (token/chunk streamé)
        #   - done    : data["final_text"]  (réponse complète)
        #   - error   : data["message"]     (+ data["code"])
        # Les events "error" non-fatals sont collectés mais n'interrompent
        # pas le flux (Lythea peut émettre des erreurs sur web search,
        # reasoning, etc. tout en continuant à générer).
        final_text = ""
        partial_chunks = []
        errors = []
        for event in rune.process_message(msg.text, history=[]):
            event_type = event.get("type", "")
            data = event.get("data", {})
            if not isinstance(data, dict):
                data = {}
            if event_type == "partial":
                # Le token streamé est dans data["text"]. Chaque partial
                # porte le texte propre cumulé — on garde le dernier plutôt
                # que de concaténer (sinon duplication).
                chunk = data.get("text") or data.get("token") or data.get("chunk", "")
                if chunk:
                    partial_chunks.append(chunk)
            elif event_type == "done":
                # La réponse complète est dans data["final_text"].
                final_text = data.get("final_text") or data.get("text") or ""
            elif event_type == "error":
                # data["message"] + data["code"] (ex: "no_model").
                err_msg = (
                    data.get("message")
                    or data.get("error")
                    or data.get("detail")
                    or data.get("reason")
                    or str(event)
                )
                code = data.get("code")
                if code:
                    err_msg = f"{err_msg} [{code}]"
                errors.append(err_msg)
        # Les partials portent le texte cumulé : le dernier est la réponse
        # la plus complète. Si "done" n'a rien donné, on prend ce dernier
        # partial plutôt que de tout concaténer (ce qui dupliquerait).
        if not final_text and partial_chunks:
            final_text = partial_chunks[-1]
        if not final_text:
            if errors:
                return OutgoingMessage(
                    text=f"(pas de réponse — erreurs: {' | '.join(errors[:2])})"
                )
            return OutgoingMessage(text="(pas de réponse)")
        return OutgoingMessage(text=final_text)

    channel = ConsoleChannel(message_handler=handler)
    channel.start()


@app.command()
def run(
    task: str = typer.Argument(..., help="La mission à exécuter"),
    context: str = typer.Option("", "--context", "-c", help="Contexte additionnel"),
    timeout: float = typer.Option(120.0, "--timeout", "-t", help="Timeout en secondes"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model ID HuggingFace pour le subagent (ex: Qwen/Qwen2.5-7B-Instruct). "
             "Défaut : RUNE_DEFAULT_MODEL / LYTHEA_MODEL_ID. En mode "
             "--lightweight SANS modèle, le subagent utilise un MockBackend "
             "(réponses génériques, ne calcule rien de réel).",
    ),
    lightweight: bool = typer.Option(
        False, "--lightweight",
        help="Utilise le SubAgentSpawner standalone (sans boot Hippocampe)",
    ),
):
    """Exécute une mission ponctuelle via subagent."""
    import os
    # Résout le modèle : --model, sinon env. Sans modèle en --lightweight,
    # le subagent tombe sur MockBackend (réponses templates, pas de vrai
    # calcul) — c'est ce qui donne des réponses génériques du type "C'est
    # une question intéressante…". On avertit clairement dans ce cas.
    effective_model = (
        model
        or os.environ.get("RUNE_DEFAULT_MODEL")
        or os.environ.get("LYTHEA_MODEL_ID")
        or os.environ.get("LYTHEA_DEFAULT_MODEL")
    )
    if lightweight:
        from rune.agents.subagent import SubAgentConfig, SubAgentSpawner
        if not effective_model:
            console.print(
                "[yellow]⚠ Aucun modèle spécifié en mode --lightweight : "
                "le subagent va utiliser un MockBackend (réponses génériques, "
                "aucun calcul réel).[/]\n"
                "[yellow]  Ajoute [cyan]--model Qwen/Qwen2.5-7B-Instruct[/] pour "
                "un vrai modèle.[/]"
            )
        spawner = SubAgentSpawner(SubAgentConfig(timeout_sec=timeout))
        result = spawner.run(
            task=task, context=context, model_id=effective_model
        )
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
