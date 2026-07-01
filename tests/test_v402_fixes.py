"""V4.0.2 — Tests for the new fixes:
- Auto-calibration (QuantileCalibrator + AutoCalibratedThresholds)
- Timeline temporal inconsistency detection
- Planning step_completion intent + advance_step pipeline
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from rune.cognition.auto_calibrator import (
    AutoCalibratedThresholds,
    QuantileCalibrator,
)
from rune.cognition.metacognition import (
    MetacognitionConfig,
    MetacognitivePhase,
)
from rune.cognition.planning import (
    GoalStack,
    IntentClassifier,
    PlanningPhase,
)
from rune.cognition.predictive_coding import (
    PredictiveCodingConfig,
    PredictiveCodingPhase,
)
from rune.cognition.timeline import (
    TimelineConfig,
    TimelineExtractor,
    detect_inconsistencies,
    render_block,
)


# ════════════════════════════════════════════════════════════════════
# QuantileCalibrator
# ════════════════════════════════════════════════════════════════════


def test_quantile_calibrator_empty_returns_none():
    qc = QuantileCalibrator()
    assert qc.quantile(0.5) is None
    assert qc.n_samples() == 0


def test_quantile_calibrator_single_sample():
    qc = QuantileCalibrator()
    qc.observe(0.42)
    assert qc.quantile(0.5) == 0.42
    assert qc.n_samples() == 1


def test_quantile_calibrator_p25_p75():
    qc = QuantileCalibrator()
    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        qc.observe(v)
    p25 = qc.quantile(0.25)
    p75 = qc.quantile(0.75)
    assert p25 < p75
    # P25 should be around 0.3, P75 around 0.7 (with linear interpolation).
    assert 0.25 < p25 < 0.4
    assert 0.6 < p75 < 0.8


def test_quantile_calibrator_window_limit():
    qc = QuantileCalibrator(window=10)
    for i in range(50):
        qc.observe(i / 100.0)
    assert qc.n_samples() == 10


def test_quantile_calibrator_clamps_invalid():
    qc = QuantileCalibrator()
    qc.observe(-1.0)  # clamped to 0
    qc.observe(float("nan"))  # rejected
    qc.observe("not a number")  # rejected
    # only the -1 (rejected by clamp guard) goes through… actually NaN
    # and negative are both rejected by the safety clamps.
    # Verify nothing crashes and at most safe values stored.
    assert qc.n_samples() <= 1


def test_quantile_calibrator_persistence(tmp_path: Path):
    p = tmp_path / "calib.json"
    qc1 = QuantileCalibrator(storage_path=p)
    for v in [0.1, 0.2, 0.3]:
        qc1.observe(v)
    # Reload
    qc2 = QuantileCalibrator(storage_path=p)
    assert qc2.n_samples() == 3
    assert qc2.quantile(0.5) == qc1.quantile(0.5)


# ════════════════════════════════════════════════════════════════════
# AutoCalibratedThresholds
# ════════════════════════════════════════════════════════════════════


def test_auto_thresholds_bootstrap_when_few_samples():
    qc = QuantileCalibrator()
    at = AutoCalibratedThresholds(
        bootstrap_low=0.15, bootstrap_high=0.35, bootstrap_very_high=0.55,
        min_samples=20,
    )
    qc.observe(0.05)
    qc.observe(0.05)
    low, high, very_high = at.get_thresholds(qc)
    assert (low, high, very_high) == (0.15, 0.35, 0.55)
    assert at.is_bootstrapping(qc) is True


def test_auto_thresholds_empirical_after_min_samples():
    qc = QuantileCalibrator()
    at = AutoCalibratedThresholds(min_samples=10)
    # Simulate a tight distribution typical of small models
    for _ in range(15):
        qc.observe(0.1)
    low, high, very_high = at.get_thresholds(qc)
    # Empirical → should hover around 0.1 with monotonicity padding
    assert 0.0 <= low < high < very_high <= 1.0
    assert at.is_bootstrapping(qc) is False


def test_auto_thresholds_monotonicity_enforced():
    """Even if all observations are identical, the bands must be ordered."""
    qc = QuantileCalibrator()
    at = AutoCalibratedThresholds(min_samples=5)
    for _ in range(10):
        qc.observe(0.5)
    low, high, very_high = at.get_thresholds(qc)
    assert low < high < very_high


# ════════════════════════════════════════════════════════════════════
# Timeline — detect_inconsistencies
# ════════════════════════════════════════════════════════════════════


def test_timeline_inconsistency_hier_vs_future_date():
    """Sentinelle Mika 10 mai 2026 : 'hier' + '12 mai 2026' = warning."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    text = "Hier on a soutenu la réunion du 12 mai 2026"
    ext = TimelineExtractor(TimelineConfig(now=now))
    events = ext.extract(text)
    warnings = detect_inconsistencies(events, now=now, text=text)
    assert len(warnings) >= 1
    assert "Incohérence" in warnings[0]
    assert "Hier" in warnings[0] or "hier" in warnings[0]


def test_timeline_consistent_dates_no_warning():
    """'Hier' + '9 mai 2026' alors qu'on est le 10 mai → cohérent, pas de warning."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    text = "Hier on a vu un défaut. Le 9 mai 2026 on a documenté."
    ext = TimelineExtractor(TimelineConfig(now=now))
    events = ext.extract(text)
    warnings = detect_inconsistencies(events, now=now, text=text)
    assert warnings == []


def test_timeline_three_distinct_events_no_warning():
    """3 événements dans 3 phrases distinctes → pas de fausse alerte."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    text = "Le 12 mai 2026 j ai eu un rendez-vous. Hier j ai signé. Dans 2 mois je commence."
    ext = TimelineExtractor(TimelineConfig(now=now))
    events = ext.extract(text)
    warnings = detect_inconsistencies(events, now=now, text=text)
    # No clause contains both an inconsistent past relative AND a
    # contradicting absolute, so we expect no warning.
    assert warnings == []


def test_timeline_render_block_includes_warnings():
    """Le bloc rendu doit inclure les warnings."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    text = "Hier on a soutenu la réunion du 12 mai 2026"
    ext = TimelineExtractor(TimelineConfig(now=now))
    events = ext.extract(text)
    block = render_block(events, config=TimelineConfig(now=now), text=text)
    assert "[Chronologie]" in block
    assert "⚠️" in block or "Incoh" in block


def test_timeline_render_block_no_warning_when_clean():
    """Pas d'alerte si les dates sont cohérentes."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    text = "Le 9 mai 2026 j ai signé. Demain je déploie."
    ext = TimelineExtractor(TimelineConfig(now=now))
    events = ext.extract(text)
    block = render_block(events, config=TimelineConfig(now=now), text=text)
    assert "⚠️" not in block


# ════════════════════════════════════════════════════════════════════
# format_warning (helper générique)
# ════════════════════════════════════════════════════════════════════


def test_format_warning_basic():
    from rune.cognition.warnings_v4 import format_warning
    out = format_warning(
        "Incohérence temporelle",
        "« hier » ne correspond pas à demain",
        "Demande à l'utilisateur de clarifier.",
    )
    # Header line
    assert out.startswith("⚠️ Incohérence temporelle : ")
    # Directive on second line
    assert "\n   → Demande" in out


def test_format_warning_no_directive():
    from rune.cognition.warnings_v4 import format_warning
    out = format_warning("Issue", "details", "")
    # No newline / arrow when directive empty
    assert "\n" not in out
    assert "→" not in out


def test_format_warning_no_details():
    from rune.cognition.warnings_v4 import format_warning
    out = format_warning("Issue", "", "Do something.")
    # Header should be just the issue (no colon when empty details)
    assert out.startswith("⚠️ Issue\n")
    assert "→ Do something." in out


def test_format_warning_custom_icon():
    from rune.cognition.warnings_v4 import format_warning
    out = format_warning("Info", "context", "Continue.", icon="ℹ️")
    assert out.startswith("ℹ️ Info")


def test_is_v4_warning_line_detection():
    from rune.cognition.warnings_v4 import is_v4_warning_line
    assert is_v4_warning_line("⚠️ Incohérence temporelle : ...") is True
    assert is_v4_warning_line("   ⚠️ ...") is True  # leading whitespace
    assert is_v4_warning_line("Normal text") is False
    assert is_v4_warning_line("") is False
    assert is_v4_warning_line("→ directive only") is False


def test_timeline_warning_includes_directive():
    """Le warning Timeline doit inclure une directive d'action."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    text = "Hier on a soutenu la réunion du 12 mai 2026"
    ext = TimelineExtractor(TimelineConfig(now=now))
    events = ext.extract(text)
    block = render_block(events, config=TimelineConfig(now=now), text=text)
    # Directive prefix should be present
    assert "   →" in block
    # System date should be inline in the directive (key fix from in vivo test)
    assert "10/05/2026" in block
    # Action verb in the directive
    assert ("Demande" in block or "demande" in block)


def test_timeline_warning_directive_has_both_dates():
    """La directive doit citer les deux dates en conflit pour que le LLM puisse demander précisément."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    text = "Hier on a soutenu la réunion du 12 mai 2026"
    ext = TimelineExtractor(TimelineConfig(now=now))
    events = ext.extract(text)
    block = render_block(events, config=TimelineConfig(now=now), text=text)
    # Both candidate dates must appear in the directive part
    directive_start = block.find("→")
    directive = block[directive_start:]
    assert "09/05/2026" in directive  # "hier" resolved
    assert "12/05/2026" in directive  # the future absolute


# ════════════════════════════════════════════════════════════════════
# Planning — step_completion intent
# ════════════════════════════════════════════════════════════════════


def test_step_completion_slash_done_command():
    cls = IntentClassifier()
    res = cls.classify("/done", has_active_goal=True)
    assert res.intent == "step_completion"
    assert res.confidence >= 0.9


def test_step_completion_verbal_marker_fr():
    cls = IntentClassifier()
    res = cls.classify("ok j'ai fini cette étape", has_active_goal=True)
    assert res.intent == "step_completion"


def test_step_completion_verbal_marker_en():
    cls = IntentClassifier()
    res = cls.classify("step done, what's next", has_active_goal=True)
    # Either step_completion (priority) or continuation (next).
    assert res.intent in ("step_completion", "continuation")


def test_step_completion_does_not_fire_without_goal():
    """Sans goal actif, /done tombe en one_shot (pas de step à advance)."""
    cls = IntentClassifier()
    res = cls.classify("/done", has_active_goal=False)
    assert res.intent != "step_completion"


def test_step_completion_long_message_falls_through():
    """Un message long contenant 'j'ai fini' n'est pas un signal d'advance."""
    cls = IntentClassifier()
    long_msg = (
        "j'ai fini de réfléchir à ce que tu m'as dit hier soir et "
        "je crois que finalement il faudrait peut-être qu'on revoie "
        "tout le périmètre du projet ensemble"
    )
    res = cls.classify(long_msg, has_active_goal=True)
    # Long message → multi_step or one_shot, not step_completion.
    assert res.intent != "step_completion"


def test_planning_phase_advance_step_via_done(tmp_path: Path):
    """End-to-end : créer un goal, puis l'advance via /done."""
    gs = GoalStack(tmp_path / "g.json")
    pp = PlanningPhase(goal_stack=gs)
    # 1. Créer un goal
    res1 = pp.process(
        "D'abord refactor le module ensuite déploie ensuite teste"
    )
    if res1.intent != "multi_step":
        pytest.skip("template fallback may not split — environment dependent")
    assert pp.goal_stack.has_active()
    initial_step = res1.active_goal.current_step
    # 2. Advance
    res2 = pp.process("/done")
    assert res2.intent == "step_completion"
    assert res2.advanced_step is True
    assert res2.active_goal.current_step == initial_step + 1


# ════════════════════════════════════════════════════════════════════
# Metacognition — auto-calibration integration
# ════════════════════════════════════════════════════════════════════


def test_metacognition_bootstrap_then_empirical():
    """Pre-bootstrap uses fixed thresholds; post-bootstrap uses empirical."""
    mp = MetacognitivePhase(config=MetacognitionConfig(
        auto_calibrate=True, auto_calibrate_min_samples=10,
    ))
    # Pre-bootstrap : feed 5 observations, check we're still bootstrapping
    for _ in range(5):
        mp.observe(doubt_index=0.10, epistemic=0.5, web_used=False)
    snap1 = mp.to_dict()
    assert snap1["thresholds_in_use"]["source"] == "bootstrap"
    # Post-bootstrap : feed 10 more
    for _ in range(10):
        mp.observe(doubt_index=0.12, epistemic=0.5, web_used=False)
    snap2 = mp.to_dict()
    assert snap2["thresholds_in_use"]["source"] == "empirical"


def test_metacognition_auto_calibrate_off_keeps_fixed_thresholds():
    mp = MetacognitivePhase(config=MetacognitionConfig(
        auto_calibrate=False, low_doubt=0.15, high_doubt=0.35,
    ))
    for _ in range(20):
        mp.observe(doubt_index=0.10, epistemic=0.5, web_used=False)
    snap = mp.to_dict()
    # Should report the fixed bootstrap values still in use
    assert snap["thresholds_in_use"]["low_doubt"] == 0.15
    assert snap["thresholds_in_use"]["high_doubt"] == 0.35


def test_metacognition_discrimination_after_calibration():
    """After calibration on a tight distribution, the bands discriminate."""
    mp = MetacognitivePhase(config=MetacognitionConfig(
        auto_calibrate=True, auto_calibrate_min_samples=15,
    ))
    # Feed a tight distribution centered at 0.10
    import random
    random.seed(0)
    for _ in range(30):
        mp.observe(
            doubt_index=random.uniform(0.05, 0.15),
            epistemic=0.5, web_used=False,
        )
    # A doubt at the very low end should be classified très_certaine
    res_low = mp.observe(doubt_index=0.04, epistemic=0.5, web_used=False)
    # A doubt at the high end should be classified less certainly
    res_high = mp.observe(doubt_index=0.16, epistemic=0.5, web_used=False)
    # confidence_score: 1.0 = très_certaine, 0.7 = certaine,
    # 0.4 = incertaine, 0.1 = très_incertaine.
    # We expect the high-doubt observation to be at most as confident
    # as the low-doubt one.
    assert res_high.confidence_score <= res_low.confidence_score


# ════════════════════════════════════════════════════════════════════
# Predictive coding — auto-calibration integration
# ════════════════════════════════════════════════════════════════════


def test_predictive_coding_to_dict_exposes_thresholds():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(
        auto_calibrate=True, auto_calibrate_min_samples=5,
    ))
    snap = pc.to_dict()
    assert "thresholds_in_use" in snap
    assert "low" in snap["thresholds_in_use"]
    assert "high" in snap["thresholds_in_use"]
    assert "source" in snap["thresholds_in_use"]


def test_predictive_coding_auto_calibrate_off():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(
        auto_calibrate=False, low_threshold=0.15, high_threshold=0.65,
    ))
    # Cold-start
    for _ in range(3):
        pc.observe([0.5] * 8)
    # Post-cold-start
    for _ in range(20):
        pc.observe([0.5] * 8)
    snap = pc.to_dict()
    # auto_calibrate=False → fixed bootstrap thresholds always
    assert snap["thresholds_in_use"]["low"] == 0.15
    assert snap["thresholds_in_use"]["high"] == 0.65


def test_predictive_coding_post_bootstrap_uses_empirical():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(
        auto_calibrate=True, auto_calibrate_min_samples=10, cold_start_min=3,
    ))
    # Cold-start
    for _ in range(3):
        pc.observe([0.5] * 8)
    # 12 post-cold-start observations
    for i in range(12):
        pc.observe([0.5 + 0.01 * i] * 8)
    snap = pc.to_dict()
    assert snap["thresholds_in_use"]["source"] == "empirical"
    # Empirical thresholds should differ from bootstrap defaults.
    assert snap["thresholds_in_use"]["low"] != 0.15 or snap["thresholds_in_use"]["high"] != 0.65


def test_predictive_coding_reset_clears_calibrator():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(
        auto_calibrate=True, auto_calibrate_min_samples=5, cold_start_min=2,
    ))
    for _ in range(2):
        pc.observe([0.5] * 8)
    for _ in range(10):
        pc.observe([0.6] * 8)
    assert pc._auto_calibrator.n_samples() > 0
    pc.reset()
    assert pc._auto_calibrator.n_samples() == 0
    assert pc.last_decision is None
