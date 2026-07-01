"""Suivi de progression généraliste (remplace red_streak code-centré)."""

from rune.agentic.progress import ProgressTracker


def test_code_progress_resets_stall():
    p = ProgressTracker(kind="code")
    assert p.update_code(5, passed=False) is True    # 1er rouge = baseline
    assert p.stalled == 0
    assert p.update_code(2, passed=False) is True     # 5→2 = progrès
    assert p.stalled == 0
    assert p.update_code(2, passed=False) is False    # stagne
    assert p.stalled == 1


def test_code_stall_accumulates_despite_no_consecutiveness():
    # Le point clé : la stagnation s'accumule même si on n'a pas de rouges
    # "consécutifs" au sens strict — chaque tour sans progrès compte.
    p = ProgressTracker(kind="code")
    p.update_code(3, passed=False)   # baseline
    p.update_code(3, passed=False)   # stall 1
    p.update_code(3, passed=False)   # stall 2
    assert p.should_enrich is True   # palier 2 atteint
    p.update_code(3, passed=False)   # stall 3
    p.update_code(3, passed=False)   # stall 4
    assert p.should_alternate is True


def test_code_pass_is_progress():
    p = ProgressTracker(kind="code")
    p.update_code(1, passed=False)
    assert p.update_code(0, passed=True) is True
    assert p.stalled == 0


def test_text_progress_on_growth():
    p = ProgressTracker(kind="redaction")
    assert p.update_text(100) is True
    assert p.update_text(110) is False   # +10 seulement → pas assez
    assert p.update_text(200) is True    # croissance nette
    assert p.stalled == 0


def test_research_progress_on_new_sources():
    p = ProgressTracker(kind="recherche")
    assert p.update_research(2) is True
    assert p.update_research(2) is False  # aucune nouvelle source
    assert p.update_research(5) is True


def test_escalation_ladder_thresholds():
    p = ProgressTracker(kind="code")
    p.update_code(3, passed=False)  # baseline, stall 0
    for _ in range(2):
        p.update_code(3, passed=False)
    assert p.should_enrich and not p.should_alternate
    for _ in range(2):
        p.update_code(3, passed=False)
    assert p.should_alternate
    for _ in range(2):
        p.update_code(3, passed=False)
    assert p.should_decompose
    p.update_code(3, passed=False)
    assert p.should_stop
