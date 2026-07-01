"""Tableau noir : ownership, anti-amnésie, rendu borné, persistance."""

import json
from pathlib import Path

from rune.agentic.blackboard import MissionBlackboard, _clip


def test_ownership_and_render_own_section():
    bb = MissionBlackboard()
    bb.ensure("agent", goal="valider des emails")
    bb.record_fail("agent", "regex naive \\d+", why="rejette user@x.com valide")
    bb.record_win("agent", "parse simple OK")
    block = bb.render_for("agent", peers=False)
    assert "TA section (agent)" in block
    assert "DÉJÀ ESSAYÉ ET ÉCHOUÉ" in block
    assert "rejette user@x.com" in block
    assert "parse simple OK" in block


def test_record_fail_dedupes_exact_repeat():
    bb = MissionBlackboard()
    bb.record_fail("agent", "même chose", why="même raison")
    bb.record_fail("agent", "même chose", why="même raison")   # doublon exact
    bb.record_fail("agent", "autre", why="autre raison")
    assert len(bb.sections["agent"].fails) == 2


def test_caps_bound_the_lists():
    bb = MissionBlackboard()
    for i in range(20):
        bb.record_fail("agent", f"essai {i}", why=f"raison {i}")
    assert len(bb.sections["agent"].fails) == 8       # _MAX_FAILS
    # garde les plus récents
    whats = [f["what"] for f in bb.sections["agent"].fails]
    assert "essai 19" in whats and "essai 0" not in whats


def test_clip_collapses_whitespace_and_truncates():
    assert _clip("  a   b\n\tc  ") == "a b c"
    assert _clip("x" * 500).endswith("…")
    assert len(_clip("x" * 500)) <= 240


def test_peers_are_read_only_summary():
    bb = MissionBlackboard()
    bb.ensure("auth", goal="login")
    bb.ensure("db", goal="stockage")
    bb.set_interface("db", "expose get(key)/set(key,val)")
    bb.note("db", "utilisez set() avant get(), pas d'accès direct")
    bb.record_fail("db", "sqlite :memory: partagé", why="threads séparés")
    block = bb.render_for("auth")                    # peers=True par défaut
    assert "section db" in block
    assert "expose get(key)" in block
    assert "utilisez set()" in block
    assert "sqlite :memory:" in block
    # la section auth est la sienne, pas listée comme peer
    assert block.count("section auth") == 0


def test_save_load_round_trip(tmp_path: Path):
    p = tmp_path / "blackboard.json"
    bb = MissionBlackboard.load(p)                   # absent → vide
    bb.ensure("agent", goal="t")
    bb.record_fail("agent", "x", why="y")
    bb.set_contract("interface: f(x)->bool")
    bb.save()
    assert p.exists()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["contract"].startswith("interface")
    bb2 = MissionBlackboard.load(p)
    assert bb2.contract.startswith("interface")
    assert bb2.sections["agent"].fails == [{"what": "x", "why": "y"}]


def test_load_missing_file_is_empty():
    bb = MissionBlackboard.load("/nonexistent/dir/blackboard.json")
    assert bb.sections == {}
    assert bb.contract == ""
