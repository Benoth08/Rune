"""Rune Dashboard — thème « Claude Code ».

Une vue alternative du dashboard Rune, dans le langage visuel de Claude
Code : accent orange coral, puces ``⏺`` pour les actions, sorties
indentées ``⎿``, liste de todos avec cases ``☒``/``☐``, et une ligne de
statut compacte en bas.

Ce module réutilise ``fetch_status`` / ``get_mock_status`` du dashboard
standard (rune.cli.dashboard) — il ne fait que RE-RENDRE les mêmes
données dans un autre style. Rien de la logique de collecte n'est
dupliqué.

Usage :
    rune dashboard-cc              # connecte http://localhost:7860
    rune dashboard-cc --mock       # démo sans serveur
"""
from __future__ import annotations

import sys
import time
from typing import Any

from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.box import ROUNDED

from rune.cli.dashboard import fetch_status, get_mock_status


# ── Palette « Claude Code » ──────────────────────────────────────────
# Orange coral (« book cloth ») en accent principal, sur fond terminal.
CC_ORANGE = "#D97757"       # accent principal (la teinte Claude)
CC_ORANGE_D = "#CC785C"     # variante plus profonde (bordures)
CC_TEXT = "#E8E4DD"         # texte clair, légèrement chaud
CC_DIM = "#8A857C"          # secondaire (gris chaud)
CC_GREEN = "#7FB069"        # succès
CC_RED = "#D9776C"          # échec (rouge coral, cohérent avec la palette)
CC_YELLOW = "#E0B85F"       # attention

SPARK = "✻"   # la « marque » Claude (astérisque six branches)
DOT = "⏺"     # puce d'action
SUB = "⎿"     # continuation / sortie sous une action
CHECK = "☒"   # todo faite
BOX = "☐"     # todo à faire


# ── Bandeau d'accueil ────────────────────────────────────────────────

def render_welcome(status: dict[str, Any]) -> Panel:
    """Boîte d'accueil façon Claude Code : ✻ + infos de session."""
    model_id = status.get("model_id")
    loaded = status.get("model_loaded", False)
    model_short = model_id.split("/")[-1] if model_id else "aucun modèle"

    grid = Table.grid(padding=(0, 0))
    grid.add_column()

    title = Text()
    title.append(f"{SPARK} ", style=f"bold {CC_ORANGE}")
    title.append("Welcome to ", style=CC_TEXT)
    title.append("Rune", style=f"bold {CC_ORANGE}")
    grid.add_row(title)
    grid.add_row(Text(""))

    sub = Text()
    sub.append("  agent cognitif local", style=CC_DIM)
    grid.add_row(sub)

    mline = Text()
    mline.append("  modèle: ", style=CC_DIM)
    mline.append(model_short, style=CC_TEXT if loaded else CC_RED)
    grid.add_row(mline)

    # Trinity si actif
    if status.get("trinity", {}).get("enabled"):
        t = Text()
        t.append("  trinity: ", style=CC_DIM)
        t.append("Thinker · Worker · Critic", style=CC_ORANGE)
        grid.add_row(t)

    return Panel(
        grid,
        border_style=CC_ORANGE_D,
        box=ROUNDED,
        padding=(0, 2),
    )


# ── Flux d'actions (le cœur du style Claude Code) ────────────────────

def _event_to_lines(e: dict) -> list[Text]:
    """Transforme un event Rune en lignes façon Claude Code :

        ⏺ write_file(fibonacci.py)
          ⎿  Wrote 210 bytes
    """
    etype = e.get("type", "?")
    tool = e.get("tool", "")
    hint = e.get("hint", "")
    lines: list[Text] = []

    if etype == "tool_call":
        head = Text()
        head.append(f"{DOT} ", style=f"bold {CC_ORANGE}")
        arg = f"({hint})" if hint else "()"
        head.append(f"{tool}", style=CC_TEXT)
        head.append(arg, style=CC_DIM)
        lines.append(head)

    elif etype == "tool_result":
        ok = e.get("ok", False)
        sub = Text()
        sub.append(f"  {SUB}  ", style=CC_DIM)
        if ok:
            sub.append(hint or "ok", style=CC_GREEN)
        else:
            sub.append(hint or "échec", style=CC_RED)
        lines.append(sub)

    elif etype in ("agent_warning", "critique"):
        t = Text()
        t.append(f"{DOT} ", style=f"bold {CC_YELLOW}")
        label = "correction" if etype == "agent_warning" else "critique"
        t.append(label, style=CC_YELLOW)
        if hint:
            t.append(f"  {hint}", style=CC_DIM)
        lines.append(t)

    elif etype == "synthesis":
        t = Text()
        t.append(f"{DOT} ", style=f"bold {CC_ORANGE}")
        t.append("synthèse", style=CC_TEXT)
        lines.append(t)

    elif etype == "lesson_learned":
        t = Text()
        t.append(f"{DOT} ", style=f"bold {CC_ORANGE}")
        t.append("leçon apprise", style=CC_ORANGE)
        if hint:
            sub = Text()
            sub.append(f"  {SUB}  ", style=CC_DIM)
            sub.append(hint, style=CC_DIM)
            lines.append(t)
            lines.append(sub)
            return lines
        lines.append(t)

    elif etype == "plan":
        t = Text()
        t.append(f"{DOT} ", style=f"bold {CC_ORANGE}")
        t.append("Plan", style=CC_TEXT)
        if hint:
            t.append(f"  {hint}", style=CC_DIM)
        lines.append(t)

    elif etype == "run_start":
        t = Text()
        t.append(f"{SPARK} ", style=f"bold {CC_ORANGE}")
        t.append("Mission démarrée", style=CC_DIM)
        lines.append(t)

    elif etype in ("run_done", "run_stopped"):
        t = Text()
        done_ok = etype == "run_done"
        t.append(f"{DOT} ", style=f"bold {CC_GREEN if done_ok else CC_RED}")
        t.append("Terminé" if done_ok else "Arrêté",
                 style=CC_GREEN if done_ok else CC_RED)
        lines.append(t)

    return lines


def render_flow(status: dict[str, Any]) -> Panel:
    """Le flux d'actions de l'agent, façon Claude Code."""
    events = status.get("recent_events", [])
    if not events:
        return Panel(
            Align.center(
                Text(f"{SPARK} en attente d'une mission…", style=CC_DIM),
                vertical="middle"),
            border_style=CC_ORANGE_D, box=ROUNDED, padding=(1, 2),
            title=Text("agent", style=CC_DIM),
        )
    blocks: list[Text] = []
    for e in list(events)[-16:]:
        blocks.extend(_event_to_lines(e))
    return Panel(
        Group(*blocks),
        border_style=CC_ORANGE_D, box=ROUNDED, padding=(1, 2),
        title=Text("agent", style=CC_DIM),
    )


# ── Todos (mission + blackboard → checklist) ─────────────────────────

def render_todos(status: dict[str, Any]) -> Panel:
    """Liste de todos façon Claude Code, dérivée du blackboard.

    Chaque section du blackboard devient une ligne : cochée si son
    statut est terminé, sinon à faire. Les wins/fails sont résumés.
    """
    bb = status.get("blackboard", {})
    mission = status.get("current_mission", {})
    sections = bb.get("sections", {}) if bb else {}

    grid = Table.grid(padding=(0, 0))
    grid.add_column()

    head = Text()
    head.append(f"{DOT} ", style=f"bold {CC_ORANGE}")
    head.append("Todos", style=CC_TEXT)
    grid.add_row(head)

    if not sections:
        # Pas de blackboard : montrer au moins la mission courante
        if mission:
            done = mission.get("done", False)
            line = Text()
            line.append(f"  {CHECK if done else BOX}  ", style=CC_GREEN if done else CC_DIM)
            line.append(mission.get("name", "mission") or "mission",
                        style=CC_TEXT if not done else CC_DIM)
            grid.add_row(line)
        else:
            grid.add_row(Text(f"  {SUB}  aucune tâche", style=CC_DIM))
        return Panel(grid, border_style=CC_ORANGE_D, box=ROUNDED,
                     padding=(0, 2), title=Text("todos", style=CC_DIM))

    for name, sec in sections.items():
        st = sec.get("status", "")
        done = "termin" in st or "done" in st
        w = len(sec.get("wins", []))
        f = len(sec.get("fails", []))
        line = Text()
        line.append(f"  {CHECK if done else BOX}  ",
                    style=CC_GREEN if done else CC_ORANGE)
        line.append(name, style=CC_DIM if done else CC_TEXT)
        meta = f"  {w}✓"
        if f:
            meta += f" {f}✗"
        line.append(meta, style=CC_DIM)
        grid.add_row(line)

    return Panel(grid, border_style=CC_ORANGE_D, box=ROUNDED,
                 padding=(0, 2), title=Text("todos", style=CC_DIM))


# ── Ligne de statut (bas, façon Claude Code) ─────────────────────────

def render_statusline(status: dict[str, Any]) -> Panel:
    """Ligne de statut compacte : modèle · VRAM · mission · hints."""
    model_id = status.get("model_id")
    model_short = model_id.split("/")[-1] if model_id else "—"
    free = status.get("vram_free_gb", 0) or 0
    total = status.get("vram_total_gb", 0) or 0
    used = max(0.0, total - free) if total else 0

    mission = status.get("current_mission", {})
    running = mission and not mission.get("done", True)

    line = Text()
    line.append(f" {SPARK} ", style=CC_ORANGE)
    line.append(model_short, style=CC_TEXT)
    if total:
        line.append("  ·  ", style=CC_DIM)
        line.append(f"{used:.1f}/{total:.0f} GB", style=CC_DIM)
    line.append("  ·  ", style=CC_DIM)
    if running:
        elapsed = mission.get("elapsed_sec", 0)
        line.append(f"{SPARK} en cours {elapsed:.0f}s", style=CC_ORANGE)
        line.append("  (esc to interrupt)", style=CC_DIM)
    else:
        line.append("prêt", style=CC_GREEN)
    now = time.strftime("%H:%M")
    line.append(f"   {now}", style=CC_DIM)

    return Panel(line, border_style=CC_ORANGE_D, box=ROUNDED, padding=(0, 1))


# ── Assemblage ───────────────────────────────────────────────────────

def build_cc_layout(status: dict[str, Any]) -> Layout:
    """Assemble la vue Claude Code."""
    layout = Layout()
    layout.split_column(
        Layout(name="welcome", size=8),
        Layout(name="body", ratio=1),
        Layout(name="status", size=3),
    )
    layout["body"].split_row(
        Layout(name="flow", ratio=2),
        Layout(name="todos", ratio=1),
    )
    layout["welcome"].update(render_welcome(status))
    layout["body"]["flow"].update(render_flow(status))
    layout["body"]["todos"].update(render_todos(status))
    layout["status"].update(render_statusline(status))
    return layout


def run_cc_dashboard(
    api_url: str = "http://localhost:7860",
    refresh_sec: float = 0.5,
    mock: bool = False,
    data_dir: str = "data",
) -> None:
    """Lance le dashboard live thème Claude Code."""
    print(f"\n  {SPARK} Démarrage du dashboard Rune (thème Claude Code)...\n",
          file=sys.stderr)
    time.sleep(0.3)

    with Live(
        build_cc_layout(get_mock_status() if mock else {"error": "connecting"}),
        refresh_per_second=2,
        screen=True,
    ) as live:
        while True:
            try:
                if mock:
                    status = get_mock_status()
                else:
                    status = fetch_status(api_url, data_dir=data_dir)
                live.update(build_cc_layout(status))
                time.sleep(refresh_sec)
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                err = {"error": str(exc), "recent_events": [], "blackboard": {},
                       "current_mission": {}}
                live.update(build_cc_layout(err))
                time.sleep(refresh_sec)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Rune dashboard (Claude Code theme)")
    p.add_argument("--api", default="http://localhost:7860")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    run_cc_dashboard(api_url=args.api, mock=args.mock)
