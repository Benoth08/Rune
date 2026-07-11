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
from rich.box import ROUNDED, HEAVY, MINIMAL

log = logging.getLogger("rune.cli.dashboard")


# ── Palette cohérente ────────────────────────────────────────────────
# Une identité visuelle par domaine : l'œil apprend à lire le tableau.
C_INFRA = "cyan"          # infrastructure (modèle, VRAM, boot)
C_OK = "green"            # succès, actif, sain
C_WARN = "yellow"         # attention, en cours, incertain
C_ERR = "red"             # échec, erreur, critique
C_LEARN = "magenta"       # apprentissage (skills, leçons)
C_MEM = "blue"            # mémoire
C_DIM = "grey50"          # secondaire
C_ACCENT = "bright_cyan"  # accents, titres


def _bar(value: float, maximum: float, width: int = 20,
         *, lo=C_OK, mid=C_WARN, hi=C_ERR, reverse: bool = False) -> Text:
    """Barre de progression colorée (jauge). La couleur suit le taux de
    remplissage : vert → jaune → rouge (ou l'inverse si reverse).

    Utilisé pour la VRAM, les scores, la progression d'une mission.
    """
    if maximum <= 0:
        maximum = 1.0
    frac = max(0.0, min(1.0, value / maximum))
    filled = int(round(frac * width))
    # Couleur selon le taux (VRAM pleine = rouge ; score haut = vert).
    ratio = (1.0 - frac) if reverse else frac
    if ratio < 0.5:
        color = lo
    elif ratio < 0.8:
        color = mid
    else:
        color = hi
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * (width - filled), style=C_DIM)
    return bar


def _gauge_line(label: str, value: float, maximum: float, unit: str = "",
                width: int = 18, **kw) -> Text:
    """Une ligne « label [████░░░░] valeur/max unit »."""
    line = Text()
    line.append(f"{label:<10}", style=C_DIM)
    line.append_text(_bar(value, maximum, width, **kw))
    txt = f" {value:.1f}"
    if maximum and maximum != value:
        txt += f"/{maximum:.0f}"
    if unit:
        txt += f" {unit}"
    line.append(txt, style="white")
    return line


def _pill(text: str, style: str) -> Text:
    """Une pastille colorée ●."""
    t = Text()
    t.append("● ", style=style)
    t.append(text, style=style)
    return t


def _sparkbar(counts: list[int], width_each: int = 1) -> Text:
    """Mini histogramme vertical à partir de compteurs (sparkline)."""
    if not counts:
        return Text("", style=C_DIM)
    blocks = "▁▂▃▄▅▆▇█"
    mx = max(counts) or 1
    t = Text()
    for c in counts:
        idx = int(round((c / mx) * (len(blocks) - 1)))
        t.append(blocks[idx] * width_each, style=C_ACCENT)
    return t


# ── Banner ASCII ─────────────────────────────────────────────────────
BANNER = r"""
  ____    _   _   _   _   _____
 |  _ \  | | | | | \ | | | ____|
 | |_) | | | | | |  \| | |  _|
 |  _ <  | |_| | | |\  | | |___
 |_| \_\  \___/  |_| \_| |_____|
"""


def render_banner() -> Text:
    """Retourne le banner ASCII en objet rich.Text (style cyan)."""
    return Text(BANNER, style="bold cyan")


# ── Panels ───────────────────────────────────────────────────────────


def render_header(status: dict[str, Any], started: float = 0.0) -> Panel:
    """En-tête : banner + bandeau de contrôle (modèle, VRAM, Trinity, uptime)."""
    # Colonne gauche : banner + tagline
    left = Table.grid()
    left.add_column()
    left.add_row(Text(BANNER, style=f"bold {C_INFRA}"))
    left.add_row(Text("  agent cognitif local · v0.1.1", style=C_DIM))

    # Colonne droite : statut système en lignes compactes
    right = Table.grid(padding=(0, 1))
    right.add_column(justify="left")

    # Modèle chargé — pastille
    model_id = status.get("model_id")
    loaded = status.get("model_loaded", False)
    if loaded and model_id:
        short = model_id.split("/")[-1]
        right.add_row(_pill(f"Modèle : {short}", C_OK))
    else:
        right.add_row(_pill("Aucun modèle chargé", C_ERR))

    # VRAM — jauge (rouge si pleine)
    free = status.get("vram_free_gb", 0) or 0
    total = status.get("vram_total_gb", 0) or 0
    if total > 0:
        used = max(0.0, total - free)
        right.add_row(_gauge_line("VRAM", used, total, "Go", width=16, reverse=True))

    # Trinity — état + rôles
    trinity = status.get("trinity", {})
    if trinity.get("enabled"):
        right.add_row(_pill("Trinity : ON (Thinker+Worker+Critic)", C_LEARN))
    else:
        right.add_row(_pill("Trinity : OFF (single-model)", C_DIM))

    # Boot — pastille selon l'état des composants
    comps = status.get("boot_components", {})
    if comps:
        n_ok = sum(1 for v in comps.values() if v == "ok")
        n_tot = len(comps)
        bstyle = C_OK if n_ok == n_tot else (C_WARN if n_ok else C_ERR)
        right.add_row(_pill(f"Boot : {n_ok}/{n_tot} composants", bstyle))

    # Uptime + horloge
    now = time.strftime("%H:%M:%S")
    up = ""
    if started:
        secs = int(time.time() - started)
        up = f"  ·  uptime {secs//60}m{secs%60:02d}s"
    right.add_row(Text(f"⏱  {now}{up}", style=C_DIM))

    # Assemblage deux colonnes
    grid = Table.grid(expand=True)
    grid.add_column(ratio=2)
    grid.add_column(ratio=3)
    grid.add_row(left, right)

    return Panel(
        grid,
        title=Text("RUNE", style=f"bold {C_ACCENT}"),
        subtitle=Text("cognitive agent · live", style=C_DIM),
        border_style=C_INFRA,
        box=HEAVY,
        padding=(0, 1),
    )


def render_mission(status: dict[str, Any]) -> Panel:
    """Mission courante : nom, statut, durée, progression, compteurs."""
    mission = status.get("current_mission", {})
    events = status.get("recent_events", [])
    if not mission:
        return Panel(
            Align.center(Text("En attente d'une mission…", style="dim italic"),
                         vertical="middle"),
            title="Mission", border_style=C_DIM, box=ROUNDED,
        )

    name = mission.get("name", "") or "—"
    task = mission.get("task", "?")
    elapsed = mission.get("elapsed_sec", 0)
    done = mission.get("done", False)

    # Compteurs d'actions depuis le fil d'events
    n_tools = sum(1 for e in events if e.get("type") == "tool_call")
    n_ok = sum(1 for e in events if e.get("type") == "tool_result" and e.get("ok"))
    n_ko = sum(1 for e in events if e.get("type") == "tool_result" and not e.get("ok"))
    n_lesson = sum(1 for e in events if e.get("type") == "lesson_learned")

    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column()

    # Ligne titre : pastille statut + nom
    head = Text()
    if done:
        head.append("● ", style=C_OK)
        head.append(name, style=f"bold {C_ACCENT}")
        head.append("   terminée", style=C_OK)
    else:
        head.append("● ", style=C_WARN)
        head.append(name, style=f"bold {C_ACCENT}")
        head.append("   en cours", style=C_WARN)
    head.append(f"   ·   {elapsed:.1f}s", style=C_DIM)
    grid.add_row(head)

    # Tâche (tronquée)
    grid.add_row(Text(task[:110] + ("…" if len(task) > 110 else ""), style="white"))

    # Ligne compteurs : outils / réussis / échoués / leçons
    counters = Text()
    counters.append(f"⚙ {n_tools} actions", style=C_INFRA)
    counters.append("    ")
    counters.append(f"✓ {n_ok}", style=C_OK)
    counters.append("   ")
    counters.append(f"✗ {n_ko}", style=C_ERR if n_ko else C_DIM)
    if n_lesson:
        counters.append("    ")
        counters.append(f"🎓 {n_lesson} leçon(s)", style=C_LEARN)
    grid.add_row(counters)

    border = C_OK if done else C_WARN
    return Panel(grid, title="Mission courante", border_style=border, box=ROUNDED)


def render_subagents(status: dict[str, Any]) -> Panel:
    """Sous-agents et leur statut, avec pastilles colorées."""
    subagents = status.get("subagents", [])
    if not subagents:
        return Panel(
            Align.center(Text("Aucun sous-agent actif", style="dim italic"),
                         vertical="middle"),
            title="Sous-agents", border_style=C_DIM, box=ROUNDED,
        )
    table = Table(show_header=True, header_style=f"bold {C_DIM}", expand=True, box=None)
    table.add_column("#", style=C_INFRA, width=3)
    table.add_column("Section", style="white")
    table.add_column("Statut", justify="left", width=8)
    table.add_column("Modèle", style=C_DIM)
    _pillmap = {
        "RUN": ("en cours", C_WARN), "DONE": ("fini", C_OK),
        "ERROR": ("erreur", C_ERR), "WAIT": ("attente", C_DIM),
    }
    for i, sa in enumerate(subagents, 1):
        lbl, st = _pillmap.get(sa.get("status", "?"), ("?", "white"))
        table.add_row(str(i), sa.get("section", "?")[:20],
                      _pill(lbl, st), sa.get("model", "?")[:24])
    return Panel(table, title="Sous-agents", border_style=C_LEARN, box=ROUNDED)


def render_blackboard(status: dict[str, Any]) -> Panel:
    """Blackboard : sections avec mini-barres wins/fails visuelles."""
    bb = status.get("blackboard", {})
    if not bb or not bb.get("sections"):
        return Panel(
            Align.center(Text("Pas encore de tableau noir", style="dim italic"),
                         vertical="middle"),
            title="Blackboard", border_style=C_DIM, box=ROUNDED,
        )

    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(style=C_INFRA, no_wrap=True)   # section
    grid.add_column()                               # barres
    grid.add_column(justify="right", style=C_DIM)   # statut

    contract = bb.get("contract", "")
    if contract:
        grid.add_row(Text("contrat", style=C_DIM),
                     Text(contract[:44], style="dim italic"), "")

    sections = bb.get("sections", {})
    # Échelle commune pour comparer les sections entre elles
    max_items = 1
    for sec in sections.values():
        max_items = max(max_items, len(sec.get("wins", [])) + len(sec.get("fails", [])))

    for name, sec in sections.items():
        w = len(sec.get("wins", []))
        f = len(sec.get("fails", []))
        n = len(sec.get("notes", []))
        st = sec.get("status", "?")
        # Barre : segment vert (wins) + segment rouge (fails)
        bar = Text()
        seg_w = int(round((w / max_items) * 14)) if max_items else 0
        seg_f = int(round((f / max_items) * 14)) if max_items else 0
        bar.append("█" * seg_w, style=C_OK)
        bar.append("█" * seg_f, style=C_ERR)
        bar.append("·" * max(0, 14 - seg_w - seg_f), style=C_DIM)
        bar.append(f"  {w}✓ {f}✗", style=C_DIM)
        if n:
            bar.append(f" {n}📝", style=C_WARN)
        ststyle = C_OK if "termin" in st or "done" in st else C_WARN
        grid.add_row(name[:16], bar, Text(st[:10], style=ststyle))

    return Panel(grid, title="Blackboard", border_style=C_OK, box=ROUNDED)


def render_memory(status: dict[str, Any]) -> Panel:
    """Mémoire : compteurs avec distinction actif (KG/Chroma) / dormant (SDM)."""
    mem = status.get("memory", {})
    if not mem:
        return Panel(
            Align.center(Text("Mémoire indisponible", style="dim italic"),
                         vertical="middle"),
            title="Mémoire", border_style=C_DIM, box=ROUNDED,
        )
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(style=C_MEM, no_wrap=True, width=9)
    grid.add_column(justify="right", style="white", width=7)
    grid.add_column(style=C_DIM)   # étiquette actif/dormant

    def _row(label, val, tag, tagstyle):
        grid.add_row(label, str(val), Text(tag, style=tagstyle))

    # Actifs (utilisés aujourd'hui)
    _row("KG", mem.get("kg_entities", "?"), "actif", C_OK)
    _row("Chroma", mem.get("chroma_count", "?"), "actif", C_OK)
    _row("MHN", mem.get("mhn_count", "?"), "câblé", C_INFRA)
    # Dormant
    _row("SDM", mem.get("sdm_count", "?"), "dormant", C_DIM)
    # Apprentissage
    _row("Skills", mem.get("skills_count", "?"), "appris", C_LEARN)
    _row("Échecs", mem.get("failures_count", "?"), "", C_DIM)

    return Panel(grid, title="Mémoire", border_style=C_MEM, box=ROUNDED)


def render_metacognition(status: dict[str, Any]) -> Panel:
    """Métacognition : confidence + jauges doute/calibration."""
    meta = status.get("metacognition", {})
    if not meta or not meta.get("available", True):
        return Panel(
            Align.center(Text("En attente d'un échange…", style="dim italic"),
                         vertical="middle"),
            title="Métacognition", border_style=C_DIM, box=ROUNDED,
        )
    confidence = meta.get("confidence_label", "?")
    doubt = meta.get("doubt_index", 0) or 0
    brier = meta.get("calibration_score", 0) or 0
    recommend_web = meta.get("recommend_web", False)

    conf_style = {
        "très_certaine": f"bold {C_OK}", "certaine": C_OK,
        "incertaine": C_WARN, "très_incertaine": f"bold {C_ERR}",
    }.get(confidence, "white")

    grid = Table.grid(padding=(0, 0), expand=True)
    grid.add_column()
    grid.add_row(_pill(f"Confiance : {confidence}", conf_style))
    # Jauge de doute (haut = rouge)
    grid.add_row(_gauge_line("Doute", doubt, 1.0, "", width=14, reverse=True))
    # Jauge de calibration Brier (bas = bon → reverse pour vert quand bas)
    grid.add_row(_gauge_line("Brier", brier, 1.0, "", width=14, reverse=True))
    web = Text()
    web.append("Web conseillé : ", style=C_DIM)
    web.append("oui" if recommend_web else "non",
               style=C_WARN if recommend_web else C_DIM)
    grid.add_row(web)

    return Panel(grid, title="Métacognition", border_style=C_WARN, box=ROUNDED)


def render_cognitive(status: dict[str, Any]) -> Panel:
    """Cycle cognitif : planification, predictive coding, inhibition, phases.

    Lit /api/config/v4 (v4_status). Affiche l'état actif/dormant de chaque
    phase + les données riches quand elles existent (goal actif, gating,
    stats d'inhibition).
    """
    cog = status.get("cognitive", {})
    if not cog:
        return Panel(
            Align.center(Text("État cognitif indisponible", style="dim italic"),
                         vertical="middle"),
            title="Cognitif", border_style=C_DIM, box=ROUNDED,
        )

    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(no_wrap=True)

    def _phase(name: str, block: dict, extra: Text | None = None):
        on = bool(block.get("enabled"))
        line = Text()
        line.append("● ", style=C_OK if on else C_DIM)
        line.append(f"{name} ", style="white" if on else C_DIM)
        line.append("actif" if on else "dormant",
                    style=C_OK if on else C_DIM)
        grid.add_row(line)
        if extra is not None and on:
            grid.add_row(extra)

    # Planification — avec goal actif si présent
    planning = cog.get("planning", {})
    goal = planning.get("active_goal")
    extra = None
    if goal:
        cur = goal.get("current_step", 0)
        n = goal.get("n_steps", 0)
        desc = (goal.get("description", "") or "")[:34]
        e = Text()
        e.append("    ", style=C_DIM)
        e.append_text(_bar(cur, max(n, 1), 10))
        e.append(f" {cur}/{n}  {desc}", style=C_DIM)
        extra = e
    _phase("Planification", planning, extra)

    # Predictive coding — dernière décision de gating
    pc = cog.get("predictive_coding", {})
    extra = None
    last = pc.get("last_decision")
    if last and isinstance(last, dict):
        err = last.get("error") or last.get("prediction_error") or last.get("surprise")
        if err is not None:
            e = Text()
            e.append("    erreur préd. ", style=C_DIM)
            try:
                e.append_text(_bar(float(err), 1.0, 10, reverse=True))
                e.append(f" {float(err):.2f}", style=C_DIM)
            except (TypeError, ValueError):
                e.append(str(err)[:20], style=C_DIM)
            extra = e
    _phase("Predictive coding", pc, extra)

    # Inhibition — stats
    inh = cog.get("inhibition", {})
    extra = None
    stats = inh.get("stats")
    if stats and isinstance(stats, dict):
        blocked = stats.get("blocked", stats.get("n_blocked", 0))
        reformed = stats.get("reformulated", stats.get("n_reformulated", 0))
        e = Text()
        e.append(f"    {blocked} bloquées · {reformed} reformulées", style=C_DIM)
        extra = e
    _phase("Inhibition", inh, extra)

    # Délibération / Réflexion — on/off (pas de trace détaillée stockée)
    _phase("Délibération", cog.get("deliberation", {"enabled": cog.get("metacognition", {}).get("enabled", False)}))
    _phase("Métacognition", cog.get("metacognition", {}))

    return Panel(grid, title="Cognitif · cycle", border_style=C_ACCENT, box=ROUNDED)


def render_skills(status: dict[str, Any]) -> Panel:
    """Skills appris (top 3) avec jauge de confiance."""
    skills = status.get("skills", [])
    if not skills:
        return Panel(
            Align.center(Text("Aucun skill appris", style="dim italic"),
                         vertical="middle"),
            title="Skills", border_style=C_DIM, box=ROUNDED,
        )
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(style="white")            # trigger
    grid.add_column(justify="right", style=C_OK, width=4)  # succès
    grid.add_column(width=10)                 # jauge conf
    for s in skills[:4]:
        trig = s.get("trigger", "?")
        conf = s.get("confidence", 0) or 0
        grid.add_row(
            trig[:24] + ("…" if len(trig) > 24 else ""),
            f"{s.get('success_count', 0)}✓",
            _bar(conf, 1.0, 8),
        )
    return Panel(grid, title="Skills appris", border_style=C_LEARN, box=ROUNDED)


def render_logs(status: dict[str, Any]) -> Panel:
    """Panneau « fil des actions de l'agent » (live).

    Affiche les derniers événements de la mission en cours (write_file,
    run_command, warnings, synthèse…) avec une icône par type et un
    court résumé. C'est la fenêtre lisible sur ce que fait l'agent, en
    remplacement du flux SSE brut.
    """
    events = status.get("recent_events", [])
    if not events:
        return Panel(
            Align.center(
                Text("En attente d'actions — lance une mission",
                     style="dim italic"), vertical="middle"),
            title="Actions de l'agent · live", border_style=C_DIM, box=ROUNDED,
        )
    _icon = {
        "run_start": ("▶", C_INFRA), "plan": ("≣", C_INFRA),
        "tool_call": ("→", "white"), "tool_result": ("✓", C_OK),
        "agent_warning": ("⚠", C_WARN), "critique": ("✎", C_WARN),
        "deliberation": ("…", C_DIM), "synthesis": ("★", C_LEARN),
        "lesson_learned": ("🎓", C_LEARN), "run_done": ("■", C_OK),
        "run_stopped": ("■", C_ERR),
    }
    _labels = {
        "lesson_learned": "leçon apprise", "synthesis": "synthèse",
        "run_done": "terminé", "run_stopped": "arrêté",
        "run_start": "démarrage", "agent_warning": "alerte",
        "critique": "critique", "plan": "plan",
    }
    lines = []
    for e in list(events)[-13:]:
        etype = e.get("type", "?")
        icon, style = _icon.get(etype, ("·", "white"))
        if etype == "tool_result" and not e.get("ok", False):
            icon, style = "✗", C_ERR
        t = e.get("t", 0)
        tool = e.get("tool", "")
        hint = e.get("hint", "")
        label = tool or _labels.get(etype, etype)
        line = Text(f"{t:>5.1f}s ", style=C_DIM)
        line.append(f"{icon} ", style=style)
        line.append(f"{label:<13}", style=style)
        if hint:
            line.append(f" {hint}", style=C_DIM)
        lines.append(line)
    return Panel(Group(*lines), title="Actions de l'agent · live",
                 border_style=C_INFRA, box=ROUNDED)


# ── Layout assemblage ────────────────────────────────────────────────


def build_layout(status: dict[str, Any], started: float = 0.0) -> Layout:
    """Assemble tous les panels dans un layout grid."""
    layout = Layout()

    # Bandeau + corps
    layout.split_column(
        Layout(name="header", size=9),
        Layout(name="body", ratio=1),
    )

    # Corps : mission (fin), puis contexte, puis le fil d'actions (large)
    layout["body"].split_column(
        Layout(name="row1", size=7),   # mission (compacte)
        Layout(name="row2", size=9),   # subagents + blackboard
        Layout(name="row3", size=8),   # memory + metacog + skills
        Layout(name="row_cog", size=9),  # cognitif (cycle complet)
        Layout(name="row4", ratio=1),  # actions — la vedette, prend le reste
    )

    layout["row2"].split_row(
        Layout(name="subagents", ratio=1),
        Layout(name="blackboard", ratio=2),   # blackboard plus large
    )
    layout["row3"].split_row(
        Layout(name="memory", ratio=1),
        Layout(name="metacog", ratio=1),
        Layout(name="skills", ratio=1),
    )

    layout["header"].update(render_header(status, started))
    layout["row1"].update(render_mission(status))
    layout["row2"]["subagents"].update(render_subagents(status))
    layout["row2"]["blackboard"].update(render_blackboard(status))
    layout["row3"]["memory"].update(render_memory(status))
    layout["row3"]["metacog"].update(render_metacognition(status))
    layout["row3"]["skills"].update(render_skills(status))
    layout["row_cog"].update(render_cognitive(status))
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

            # Statut agentique live (mission courante + fil des actions
            # + blackboard). Endpoint dédié qui lit le registre des runs
            # de l'orchestrateur sans le construire (pas de 503).
            try:
                r = client.get(f"{api_url}/api/agent/status")
                if r.status_code == 200:
                    ag = r.json()
                    status["current_mission"] = ag.get("current_mission", {})
                    status["recent_events"] = ag.get("recent_events", [])
                    status["blackboard"] = ag.get("blackboard", {})
            except Exception:
                pass

            # Métacognition (dernière décision : confidence/doute/Brier).
            try:
                r = client.get(f"{api_url}/api/metacognition/status")
                if r.status_code == 200:
                    status["metacognition"] = r.json()
            except Exception:
                pass

            # État cognitif complet (planning, predictive coding,
            # inhibition, délibération/réflexion on/off) via /api/config/v4.
            try:
                r = client.get(f"{api_url}/api/config/v4")
                if r.status_code == 200:
                    status["cognitive"] = r.json()
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
        "vram_free_gb": 11.2,
        "vram_total_gb": 24.0,
        "model_loaded": True,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "boot_components": {"model": "ok", "memory": "ok", "mcp": "ok", "trinity": "ok"},
        "current_mission": {
            "name": "Débogage Fibonacci",
            "task": "Débugge la fonction fibonacci et écris les tests pytest",
            "elapsed_sec": 42.3,
            "done": False,
        },
        "recent_events": [
            {"t": 2.1, "type": "run_start"},
            {"t": 3.4, "type": "plan", "hint": "3 étapes"},
            {"t": 5.0, "type": "tool_call", "tool": "write_file", "hint": "fibonacci.py"},
            {"t": 6.2, "type": "tool_result", "tool": "write_file", "ok": True, "hint": '{"size": 210}'},
            {"t": 8.1, "type": "tool_call", "tool": "write_file", "hint": "test_fibonacci.py"},
            {"t": 9.0, "type": "tool_result", "tool": "write_file", "ok": True, "hint": '{"size": 180}'},
            {"t": 11.5, "type": "tool_call", "tool": "run_tests", "hint": ""},
            {"t": 14.2, "type": "tool_result", "tool": "run_tests", "ok": False, "hint": "1 failed: import"},
            {"t": 16.0, "type": "agent_warning", "hint": "correction de l'import manquant"},
            {"t": 18.3, "type": "tool_call", "tool": "edit_file", "hint": "fibonacci.py"},
            {"t": 20.1, "type": "tool_result", "tool": "run_tests", "ok": True, "hint": "3 passed"},
            {"t": 22.0, "type": "synthesis", "hint": "synthèse produite"},
            {"t": 23.1, "type": "lesson_learned", "hint": "Débugger une fonction récursive Python"},
        ],
        "subagents": [
            {"section": "debug_agent", "status": "RUN", "model": "Qwen2.5-7B-Instruct"},
            {"section": "test_agent", "status": "DONE", "model": "Qwen2.5-7B-Instruct"},
            {"section": "docs_agent", "status": "WAIT", "model": "Qwen2.5-7B-Instruct"},
        ],
        "blackboard": {
            "contract": "Débugge fibonacci et écris les tests",
            "sections": {
                "debug_agent": {"wins": ["fix appliqué", "import corrigé"], "fails": [], "notes": ["running"], "status": "running"},
                "test_agent": {"wins": ["tests pass"], "fails": [{"what": "test 1", "why": "import"}], "notes": ["3 tests"], "status": "done"},
            },
        },
        "memory": {
            "sdm_count": 0,
            "mhn_count": 89,
            "kg_entities": 45,
            "chroma_count": 1024,
            "skills_count": 7,
            "failures_count": 3,
        },
        "metacognition": {
            "available": True,
            "confidence_label": "certaine",
            "doubt_index": 0.18,
            "calibration_score": 0.21,
            "recommend_web": False,
        },
        "cognitive": {
            "planning": {"enabled": True, "active_goal": {
                "description": "Débugger et tester fibonacci", "current_step": 2, "n_steps": 4}},
            "predictive_coding": {"enabled": True, "last_decision": {"error": 0.34}},
            "inhibition": {"enabled": True, "stats": {"blocked": 2, "reformulated": 1}},
            "deliberation": {"enabled": True},
            "metacognition": {"enabled": True},
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
    _started = time.time()

    with Live(
        build_layout(get_mock_status() if mock else {"error": "connecting"}, _started),
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

                live.update(build_layout(status, _started))
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
                live.update(build_layout(error_status, _started))
                time.sleep(refresh_sec)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rune dashboard")
    parser.add_argument("--api", default="http://localhost:7860", help="API URL")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--refresh", type=float, default=0.5, help="Refresh sec")
    args = parser.parse_args()
    run_dashboard(api_url=args.api, refresh_sec=args.refresh, mock=args.mock)
