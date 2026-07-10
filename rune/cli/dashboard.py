"""Rune Dashboard — interface ASCII live avec rich.live + rich.layout.

Affiche en temps réel pendant l'exécution :
- Le banner ASCII Rune en haut
- La mission courante + phase + modèle Trinity actif
- Les sous-agents et leur statut (WAIT/RUN/DONE/ERROR)
- Le blackboard (sections + wins/fails/notes)
- La mémoire (SDM/MHN/KG/Chroma/Skills/Failures)
- La métacognition (confidence, doubt, Brier)
- Les skills actifs (top 3 par utility)
- Les derniers logs

Usage :
    rune dashboard                    # connecte à http://localhost:7860
    rune dashboard --api http://...:7860  # API custom
    rune demo-dashboard              # démo avec mock data (sans serveur)

Le dashboard poll l'API REST toutes les 0.5s pour récupérer le status.
Si l'API est down, affiche un message d'erreur mais continue à tourner
(reconnect automatique).
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Any

import httpx
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group

log = logging.getLogger("rune.cli.dashboard")


# ── Banner ASCII (fourni par l'utilisateur) ──────────────────────────
BANNER = r"""
  ____    _   _   _   _   _____       _       ____   _____   _   _   _____
 |  _ \  | | | | | \ | | | ____|     / \     / ___| | ____| | \ | | |_   _|
 | |_) | | | | | |  \| | |  _|      / _ \   |  _  |  _|   |  \| |   | |
 |  _ <  | |_| | | |\  | | |___    / ___ \  | |_| | | |___  | |\  |   | |
 |_| \_\  \___/  |_| \_| |_____|  /_/   \_\  \____| |_____| |_| \_|  |_|
"""


def render_banner() -> Text:
    """Retourne le banner ASCII en objet rich.Text (style cyan)."""
    return Text(BANNER, style="bold cyan")


# ── Panels ───────────────────────────────────────────────────────────


def render_header(status: dict[str, Any]) -> Panel:
    """Panneau d'en-tête : banner + ligne de statut global."""
    trinity = status.get("trinity", {})
    trinity_line = ""
    if trinity.get("enabled"):
        worker = trinity.get("roles", {}).get("worker", {}).get("model_id", "?")
        trinity_line = f"Trinity: ON (Worker: {worker})"
    else:
        trinity_line = "Trinity: OFF (single-model)"

    now = time.strftime("%H:%M:%S")
    title = f"Rune v0.1.0 — {trinity_line} — {now}"

    return Panel(
        Align.left(Text(BANNER, style="bold cyan")),
        title=title,
        border_style="cyan",
        padding=(0, 1),
    )


def render_mission(status: dict[str, Any]) -> Panel:
    """Panneau mission courante."""
    mission = status.get("current_mission", {})
    if not mission:
        content = Text("(aucune mission en cours)", style="dim italic")
    else:
        task = mission.get("task", "?")
        phase = mission.get("phase", "?")
        model = mission.get("model", "?")
        elapsed = mission.get("elapsed_sec", 0)
        tokens = mission.get("tokens", 0)

        lines = [
            Text(f"Task: {task}", style="white"),
            Text(f"Phase: {phase} → {model}", style="yellow"),
            Text(f"Elapsed: {elapsed:.1f}s | Tokens: {tokens}", style="dim"),
        ]
        content = Group(*lines)

    return Panel(content, title="Mission courante", border_style="blue")


def render_subagents(status: dict[str, Any]) -> Panel:
    """Panneau sous-agents."""
    subagents = status.get("subagents", [])
    if not subagents:
        content = Text("(aucun sous-agent)", style="dim italic")
    else:
        table = Table(show_header=True, header_style="bold", expand=True, box=None)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Section", style="white")
        table.add_column("Statut", justify="center", width=6)
        table.add_column("Modèle", style="dim")

        for i, sa in enumerate(subagents, 1):
            sa_status = sa.get("status", "?")
            style = {
                "RUN": "bold yellow",
                "DONE": "green",
                "ERROR": "red",
                "WAIT": "dim",
            }.get(sa_status, "white")
            table.add_row(
                str(i),
                sa.get("section", "?")[:20],
                Text(sa_status, style=style),
                sa.get("model", "?")[:30],
            )
        content = table

    return Panel(content, title="Subagents", border_style="magenta")


def render_blackboard(status: dict[str, Any]) -> Panel:
    """Panneau blackboard."""
    bb = status.get("blackboard", {})
    if not bb or not bb.get("sections"):
        content = Text("(pas de blackboard)", style="dim italic")
    else:
        table = Table(show_header=True, header_style="bold", expand=True, box=None)
        table.add_column("Section", style="cyan")
        table.add_column("Wins", justify="right", style="green", width=5)
        table.add_column("Fails", justify="right", style="red", width=5)
        table.add_column("Notes", justify="right", style="yellow", width=5)
        table.add_column("Statut", style="white", width=10)

        contract = bb.get("contract", "")
        if contract:
            table.add_row("_contract", "-", "-", "-", Text(contract[:40], style="dim italic"))

        for name, sec in bb.get("sections", {}).items():
            table.add_row(
                name[:20],
                str(len(sec.get("wins", []))),
                str(len(sec.get("fails", []))),
                str(len(sec.get("notes", []))),
                sec.get("status", "?"),
            )
        content = table

    return Panel(content, title="Blackboard", border_style="green")


def render_memory(status: dict[str, Any]) -> Panel:
    """Panneau mémoire."""
    mem = status.get("memory", {})
    if not mem:
        content = Text("(mémoire indisponible)", style="dim italic")
    else:
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Key", style="cyan", width=12)
        table.add_column("Value", style="white")
        table.add_row("SDM", str(mem.get("sdm_count", "?")))
        table.add_row("MHN", str(mem.get("mhn_count", "?")))
        table.add_row("KG", str(mem.get("kg_entities", "?")))
        table.add_row("Chroma", str(mem.get("chroma_count", "?")))
        table.add_row("Skills", str(mem.get("skills_count", "?")))
        table.add_row("Failures", str(mem.get("failures_count", "?")))
        content = table

    return Panel(content, title="Mémoire", border_style="blue")


def render_metacognition(status: dict[str, Any]) -> Panel:
    """Panneau métacognition."""
    meta = status.get("metacognition", {})
    if not meta:
        content = Text("(métacognition indisponible)", style="dim italic")
    else:
        confidence = meta.get("confidence_label", "?")
        doubt = meta.get("doubt_index", 0)
        brier = meta.get("calibration_score", 0)
        recommend_web = meta.get("recommend_web", False)

        # Couleur selon confidence
        conf_style = {
            "très_certaine": "bold green",
            "certaine": "green",
            "incertaine": "yellow",
            "très_incertaine": "bold red",
        }.get(confidence, "white")

        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Key", style="cyan", width=14)
        table.add_column("Value")
        table.add_row("Confidence", Text(confidence, style=conf_style))
        table.add_row("Doubt", f"{doubt:.3f}")
        table.add_row("Brier score", f"{brier:.3f}")
        table.add_row(
            "Recommend web",
            Text("yes" if recommend_web else "no",
                 style="yellow" if recommend_web else "dim"),
        )
        content = table

    return Panel(content, title="Métacognition", border_style="yellow")


def render_skills(status: dict[str, Any]) -> Panel:
    """Panneau skills actifs (top 3)."""
    skills = status.get("skills", [])
    if not skills:
        content = Text("(aucun skill appris)", style="dim italic")
    else:
        table = Table(show_header=True, header_style="bold", expand=True, box=None)
        table.add_column("ID", style="cyan", width=14)
        table.add_column("Trigger", style="white")
        table.add_column("Succès", justify="right", style="green", width=6)
        table.add_column("Conf.", justify="right", width=6)

        for skill in skills[:3]:
            table.add_row(
                skill.get("id", "?")[:12],
                skill.get("trigger", "?")[:30],
                str(skill.get("success_count", 0)),
                f"{skill.get('confidence', 0):.2f}",
            )
        content = table

    return Panel(content, title="Skills actifs", border_style="magenta")


def render_logs(status: dict[str, Any]) -> Panel:
    """Panneau logs (derniers 5)."""
    logs = status.get("logs", [])
    if not logs:
        content = Text("(aucun log)", style="dim italic")
    else:
        lines = []
        for log_entry in logs[:5]:
            ts = log_entry.get("ts", "?")
            level = log_entry.get("level", "?")
            msg = log_entry.get("msg", "?")
            level_style = {
                "ERROR": "red",
                "WARNING": "yellow",
                "INFO": "white",
                "DEBUG": "dim",
            }.get(level, "white")
            lines.append(
                Text(f"{ts} ", style="dim")
                + Text(f"[{level}] ", style=level_style)
                + Text(msg[:80])
            )
        content = Group(*lines)

    return Panel(content, title="Logs", border_style="blue")


# ── Layout assemblage ────────────────────────────────────────────────


def build_layout(status: dict[str, Any]) -> Layout:
    """Assemble tous les panels dans un layout grid."""
    layout = Layout()

    # Ligne 1 : banner (taille fixe)
    layout.split_column(
        Layout(name="header", size=8),
        Layout(name="body", ratio=1),
    )

    # Body : 3 rangées
    layout["body"].split_column(
        Layout(name="row1", size=8),  # mission
        Layout(name="row2", ratio=1),  # subagents + blackboard
        Layout(name="row3", ratio=1),  # memory + metacog + skills
        Layout(name="row4", size=8),  # logs
    )

    # Row 2 : subagents | blackboard
    layout["row2"].split_row(
        Layout(name="subagents", ratio=1),
        Layout(name="blackboard", ratio=1),
    )

    # Row 3 : memory | metacog | skills
    layout["row3"].split_row(
        Layout(name="memory", ratio=1),
        Layout(name="metacog", ratio=1),
        Layout(name="skills", ratio=1),
    )

    # Remplissage
    layout["header"].update(render_header(status))
    layout["row1"].update(render_mission(status))
    layout["row2"]["subagents"].update(render_subagents(status))
    layout["row2"]["blackboard"].update(render_blackboard(status))
    layout["row3"]["memory"].update(render_memory(status))
    layout["row3"]["metacog"].update(render_metacognition(status))
    layout["row3"]["skills"].update(render_skills(status))
    layout["row4"].update(render_logs(status))

    return layout


# ── Status fetcher ───────────────────────────────────────────────────


def fetch_status(
    api_url: str = "http://localhost:7860",
    timeout: float = 2.0,
    data_dir: str = "data",
) -> dict[str, Any]:
    """Récupère le status depuis l'API Lythea + lit les fichiers Rune.

    Sources :
    - API Lythea : /api/health, /api/boot/status, /api/memory/status
    - Fichiers Rune : data/skills/skills.json, data/failures/failures.json,
      data/cron/tasks.json

    Si l'API est down, on garde quand même les données fichiers.
    """
    import json
    from pathlib import Path

    status: dict[str, Any] = {"error": None}

    # ── API Lythea ──────────────────────────────────────────────────
    try:
        with httpx.Client(timeout=timeout) as client:
            # Healthcheck
            try:
                r = client.get(f"{api_url}/api/health")
                if r.status_code == 200:
                    health = r.json()
                    status["backend"] = health.get("backend", "?")
                    status["model_loaded"] = health.get("model_loaded", False)
                    status["vram_free_gb"] = health.get("vram_free_gb", 0)
                    status["model_id"] = health.get("model_id")
                else:
                    status["error"] = f"HTTP {r.status_code} on /api/health"
            except Exception as exc:
                # Erreur de connexion typique : le serveur n'est pas
                # lancé. "Cannot assign requested address" (Errno 99) est
                # cryptique — on ajoute l'action attendue.
                msg = str(exc)
                if "Cannot assign requested address" in msg or "Connection refused" in msg:
                    status["error"] = (
                        f"Serveur injoignable sur {api_url} — lance "
                        f"'rune serve' dans un autre terminal, ou passe "
                        f"--api <url> si le serveur tourne ailleurs."
                    )
                else:
                    status["error"] = f"API down: {msg}"

            # Boot status
            try:
                r = client.get(f"{api_url}/api/boot/status")
                if r.status_code == 200:
                    boot = r.json()
                    components = boot.get("components", {})
                    status["trinity"] = {
                        "enabled": components.get("trinity") == "ok",
                    }
                    status["boot_components"] = components
            except Exception:
                pass

            # Mémoire Lythea (SDM/MHN/KG/Chroma)
            # L'API retourne des dicts imbriqués :
            #   {"sdm": {"active_rows": N, ...}, "mhn": {"stored": N, ...}, ...}
            # On normalise en clés plates pour render_memory().
            try:
                r = client.get(f"{api_url}/api/memory/status")
                if r.status_code == 200:
                    raw = r.json()
                    status["memory"] = {
                        "sdm_count": raw.get("sdm", {}).get("active_rows", "?"),
                        "mhn_count": raw.get("mhn", {}).get("stored", "?"),
                        "kg_entities": raw.get("kg", {}).get("entities", "?"),
                        "chroma_count": raw.get("chroma", {}).get("count", "?"),
                        # skills_count et failures_count injectés depuis
                        # les fichiers JSON Rune ci-dessous
                        "skills_count": 0,
                        "failures_count": 0,
                    }
            except Exception:
                pass

    except Exception as exc:
        status["error"] = str(exc)

    # ── Fichiers Rune (skills, failures, cron) ──────────────────────
    # Ces données ne sont pas dans l'API Lythea — on lit directement
    # les fichiers JSON persistés par RuneCortex.
    data_path = Path(data_dir)

    # Skills
    try:
        skills_file = data_path / "skills" / "skills.json"
        if skills_file.exists():
            with skills_file.open() as f:
                skills_data = json.load(f)
            skills_list = skills_data.get("skills", [])
            # Format pour le dashboard
            status["skills"] = [
                {
                    "id": s.get("skill_id", "?"),
                    "trigger": s.get("trigger", "?"),
                    "success_count": s.get("success_count", 0),
                    "confidence": s.get("confidence", 0),
                    "is_reliable": _is_skill_reliable(s),
                }
                for s in skills_list
                if not s.get("archived", False)
            ]
        else:
            status["skills"] = []
    except Exception:
        status["skills"] = []

    # Failures
    try:
        failures_file = data_path / "failures" / "failures.json"
        if failures_file.exists():
            with failures_file.open() as f:
                failures_data = json.load(f)
            status["failures_count"] = len(failures_data.get("failures", []))
        else:
            status["failures_count"] = 0
    except Exception:
        status["failures_count"] = 0

    # Injecte skills_count et failures_count dans le dict mémoire
    # (render_memory() lit depuis status["memory"], pas depuis status["skills"])
    if "memory" not in status:
        status["memory"] = {}
    status["memory"]["skills_count"] = len(status.get("skills", []))
    status["memory"]["failures_count"] = status.get("failures_count", 0)

    # Cron tasks
    try:
        cron_file = data_path / "cron" / "tasks.json"
        if cron_file.exists():
            with cron_file.open() as f:
                cron_data = json.load(f)
            status["cron"] = {
                "running": False,
                "task_count": len(cron_data.get("tasks", [])),
                "enabled_count": sum(
                    1 for t in cron_data.get("tasks", []) if t.get("enabled", True)
                ),
            }
        else:
            status["cron"] = {"running": False, "task_count": 0, "enabled_count": 0}
    except Exception:
        status["cron"] = {"running": False, "task_count": 0}

    return status


def _is_skill_reliable(skill: dict) -> bool:
    """Reproduit Skill.is_reliable() depuis un dict."""
    total = skill.get("success_count", 0) + skill.get("failure_count", 0)
    if total < 2:
        return False
    if skill.get("failure_count", 0) / total > 0.3:
        return False
    return True


def get_mock_status() -> dict[str, Any]:
    """Status mocké pour la démo (sans serveur)."""
    return {
        "error": None,
        "backend": "transformers",
        "model_loaded": True,
        "trinity": {
            "enabled": True,
            "roles": {
                "worker": {"model_id": "Qwen/Qwen2.5-7B-Instruct", "loaded": True},
                "thinker": {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "loaded": True},
                "critic": {"model_id": "Qwen/Qwen2.5-3B-Instruct", "loaded": True},
            },
        },
        "current_mission": {
            "task": "Débugge la fonction fibonacci",
            "phase": "execution",
            "model": "Worker (Qwen2.5-7B-Instruct)",
            "elapsed_sec": 12.3,
            "tokens": 234,
        },
        "subagents": [
            {"section": "debug_agent", "status": "RUN", "model": "Qwen2.5-7B-Instruct"},
            {"section": "test_agent", "status": "DONE", "model": "Qwen2.5-7B-Instruct"},
            {"section": "docs_agent", "status": "WAIT", "model": "Qwen2.5-7B-Instruct"},
        ],
        "blackboard": {
            "contract": "Débugge fibonacci et écris les tests",
            "sections": {
                "debug_agent": {"wins": ["Résultat: fix appliqué"], "fails": [], "notes": ["subagent running"], "status": "running"},
                "test_agent": {"wins": ["Résultat: tests pass"], "fails": [{"what": "test 1", "why": "import manquant"}], "notes": ["3 tests passent"], "status": "done"},
            },
        },
        "memory": {
            "sdm_count": 234,
            "mhn_count": 89,
            "kg_entities": 45,
            "chroma_count": 1024,
            "skills_count": 7,
            "failures_count": 3,
        },
        "metacognition": {
            "confidence_label": "certaine",
            "doubt_index": 0.18,
            "calibration_score": 0.21,
            "recommend_web": False,
        },
        "skills": [
            {"id": "skill_abc", "trigger": "Calculer fibonacci", "success_count": 3, "confidence": 0.85},
            {"id": "skill_def", "trigger": "Débugger Python", "success_count": 5, "confidence": 0.92},
            {"id": "skill_ghi", "trigger": "Écrire tests pytest", "success_count": 2, "confidence": 0.78},
        ],
        "logs": [
            {"ts": "18:42:15", "level": "INFO", "msg": "Blackboard updated (section=debug_agent)"},
            {"ts": "18:42:14", "level": "INFO", "msg": "Trinity routing: reasoning → Thinker"},
            {"ts": "18:42:10", "level": "INFO", "msg": "HotContext built: 3 chunks, 1 skill, 0 anti-pat"},
            {"ts": "18:42:05", "level": "INFO", "msg": "SubAgent #1 started (debug_agent)"},
            {"ts": "18:42:00", "level": "INFO", "msg": "Mission started: Débugge fibonacci"},
        ],
    }


# ── Entrée principale ────────────────────────────────────────────────


def run_dashboard(
    api_url: str = "http://localhost:7860",
    refresh_sec: float = 0.5,
    mock: bool = False,
    data_dir: str = "data",
) -> None:
    """Lance le dashboard ASCII live.

    Parameters
    ----------
    api_url : str
        URL de l'API REST Rune (par défaut http://localhost:7860).
    refresh_sec : float
        Intervalle de refresh en secondes (défaut 0.5s).
    mock : bool
        Si True, utilise des données mockées (pour démo sans serveur).
    data_dir : str
        Chemin vers le dossier data/ de Rune (pour lire skills.json,
        failures.json, cron/tasks.json directement).
    """
    print("\n  Démarrage du dashboard Rune...\n", file=sys.stderr)
    time.sleep(0.3)

    with Live(
        build_layout(get_mock_status() if mock else {"error": "connecting"}),
        refresh_per_second=2,
        screen=True,
    ) as live:
        while True:
            try:
                if mock:
                    status = get_mock_status()
                else:
                    status = fetch_status(api_url, data_dir=data_dir)

                if status.get("error"):
                    # Affiche l'erreur dans le panel header
                    status["current_mission"] = {
                        "task": f"[ERREUR] {status['error']}",
                        "phase": "error",
                        "model": "—",
                        "elapsed_sec": 0,
                        "tokens": 0,
                    }

                live.update(build_layout(status))
                time.sleep(refresh_sec)

            except KeyboardInterrupt:
                break
            except Exception as exc:
                # Affiche l'erreur mais continue
                error_status = {
                    "error": str(exc),
                    "current_mission": {
                        "task": f"[ERREUR] {exc}",
                        "phase": "error",
                        "model": "—",
                        "elapsed_sec": 0,
                        "tokens": 0,
                    },
                }
                live.update(build_layout(error_status))
                time.sleep(refresh_sec)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rune dashboard")
    parser.add_argument("--api", default="http://localhost:7860", help="API URL")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--refresh", type=float, default=0.5, help="Refresh sec")
    args = parser.parse_args()
    run_dashboard(api_url=args.api, refresh_sec=args.refresh, mock=args.mock)
