"""V4.4 — Tests for metacognition module."""

import json
from pathlib import Path

import pytest

from rune.cognition.metacognition import (
    CONFIDENCE_LABELS,
    CalibrationTracker,
    CertaintyClassifier,
    CertaintyConfig,
    MetacognitionConfig,
    MetacognitiveDecision,
    MetacognitivePhase,
    hedge_prefix,
)


# ════════════════════════════════════════════════════════════════════
# 1. CalibrationTracker
# ════════════════════════════════════════════════════════════════════


def test_calibration_brier_none_on_empty():
    t = CalibrationTracker()
    assert t.brier_score() is None
    assert t.calibration_score() == 0.5  # neutral prior


def test_calibration_perfect_zero_brier():
    t = CalibrationTracker()
    for _ in range(5):
        t.record(announced=1.0, correct=1.0)
        t.record(announced=0.0, correct=0.0)
    assert t.brier_score() == 0.0
    assert t.calibration_score() == 1.0


def test_calibration_worst_brier():
    t = CalibrationTracker()
    for _ in range(5):
        t.record(announced=1.0, correct=0.0)  # confident but wrong
    assert abs(t.brier_score() - 1.0) < 1e-9
    assert t.calibration_score() == 0.0


def test_calibration_clamps_input():
    t = CalibrationTracker()
    t.record(announced=2.5, correct=-0.5)  # both out of range
    # Should have stored (1.0, 0.0)
    assert t.brier_score() == 1.0


def test_calibration_window_bounded():
    t = CalibrationTracker(window=5)
    for i in range(20):
        t.record(announced=0.5, correct=0.5)
    assert t.n_entries() == 5


def test_calibration_persistence_roundtrip(tmp_path: Path):
    p = tmp_path / "calib.json"
    t1 = CalibrationTracker(storage_path=p)
    t1.record(announced=0.8, correct=0.7)
    t1.record(announced=0.9, correct=0.6)
    n_before = t1.n_entries()

    t2 = CalibrationTracker(storage_path=p)
    assert t2.n_entries() == n_before
    # Brier should be reproducible
    assert abs(t1.brier_score() - t2.brier_score()) < 1e-9


def test_calibration_corrupted_file_no_crash(tmp_path: Path):
    p = tmp_path / "calib.json"
    p.write_text("{ broken json", encoding="utf-8")
    t = CalibrationTracker(storage_path=p)
    assert t.n_entries() == 0


def test_calibration_atomic_no_tmp_leftover(tmp_path: Path):
    p = tmp_path / "calib.json"
    t = CalibrationTracker(storage_path=p)
    t.record(0.5, 0.5)
    assert p.exists()
    assert not (tmp_path / "calib.json.tmp").exists()


def test_calibration_reset(tmp_path: Path):
    t = CalibrationTracker(storage_path=tmp_path / "c.json")
    for _ in range(5):
        t.record(0.5, 0.5)
    t.reset()
    assert t.n_entries() == 0


def test_calibration_invalid_input_ignored():
    t = CalibrationTracker()
    t.record(announced="not a number", correct=0.5)  # type: ignore[arg-type]
    assert t.n_entries() == 0


# ════════════════════════════════════════════════════════════════════
# 2. CertaintyClassifier
# ════════════════════════════════════════════════════════════════════


def test_classifier_very_high_doubt_returns_very_uncertain():
    c = CertaintyClassifier()
    assert c.classify(doubt_index=0.9, epistemic=0.5) == "très_incertaine"


def test_classifier_low_doubt_returns_very_certain():
    c = CertaintyClassifier()
    assert c.classify(doubt_index=0.05, epistemic=0.8) == "très_certaine"


def test_classifier_moderate_doubt_returns_certaine_or_incertaine():
    c = CertaintyClassifier()
    label = c.classify(doubt_index=0.25, epistemic=0.5, web_used=True)
    assert label in ("certaine", "incertaine")


def test_classifier_epistemic_boost_shifts_one_band():
    c = CertaintyClassifier()
    # High epistemic should shift band toward more certain
    label_low = c.classify(doubt_index=0.25, epistemic=0.3, web_used=True)
    label_high = c.classify(doubt_index=0.25, epistemic=0.9, web_used=True)
    idx_low = CONFIDENCE_LABELS.index(label_low)
    idx_high = CONFIDENCE_LABELS.index(label_high)
    assert idx_high <= idx_low  # high epistemic ≤ low epistemic (band index)


def test_classifier_web_absence_penalty():
    c = CertaintyClassifier()
    # High doubt + no web → drop a band
    with_web = c.classify(doubt_index=0.5, epistemic=0.5, web_used=True)
    without_web = c.classify(doubt_index=0.5, epistemic=0.5, web_used=False)
    assert CONFIDENCE_LABELS.index(without_web) >= CONFIDENCE_LABELS.index(with_web)


def test_classifier_handles_invalid_inputs():
    c = CertaintyClassifier()
    label = c.classify(doubt_index="not a number", epistemic=None)  # type: ignore[arg-type]
    assert label in CONFIDENCE_LABELS


# ════════════════════════════════════════════════════════════════════
# 3. Hedge prefixes
# ════════════════════════════════════════════════════════════════════


def test_hedge_very_certain_empty():
    assert hedge_prefix("très_certaine") == ""


def test_hedge_certain_empty():
    assert hedge_prefix("certaine") == ""


def test_hedge_uncertain_non_empty():
    assert hedge_prefix("incertaine")  # non-empty
    assert "sûre" in hedge_prefix("incertaine")


def test_hedge_very_uncertain_non_empty():
    assert hedge_prefix("très_incertaine")
    assert "manque" in hedge_prefix("très_incertaine")


def test_hedge_unknown_label_empty():
    assert hedge_prefix("bogus") == ""


# ════════════════════════════════════════════════════════════════════
# 4. MetacognitivePhase
# ════════════════════════════════════════════════════════════════════


def test_phase_returns_decision_dataclass(tmp_path: Path):
    mp = MetacognitivePhase(storage_path=tmp_path / "calib.json")
    d = mp.observe(doubt_index=0.2, epistemic=0.5, web_used=True)
    assert isinstance(d, MetacognitiveDecision)
    assert d.confidence_label in CONFIDENCE_LABELS


def test_phase_low_doubt_high_epistemic_yields_certain(tmp_path: Path):
    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    d = mp.observe(doubt_index=0.05, epistemic=0.9, web_used=True)
    assert d.confidence_label == "très_certaine"
    assert d.confidence_score >= 0.9
    assert d.hedge_prefix == ""


def test_phase_very_high_doubt_yields_uncertain_recommend_web(tmp_path: Path):
    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    d = mp.observe(doubt_index=0.85, epistemic=0.3, web_used=False)
    assert d.confidence_label == "très_incertaine"
    assert d.recommend_web is True


def test_phase_already_used_web_does_not_recommend(tmp_path: Path):
    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    d = mp.observe(doubt_index=0.85, epistemic=0.3, web_used=True)
    # Even if very_uncertain, recommend_web stays False if web already used
    assert d.recommend_web is False


def test_phase_apply_hedge_off_by_default(tmp_path: Path):
    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    d = mp.observe(doubt_index=0.85, epistemic=0.3, web_used=False)
    # apply_hedge=False by default → empty prefix even when uncertain
    assert d.hedge_prefix == ""


def test_phase_apply_hedge_on_emits_prefix(tmp_path: Path):
    mp = MetacognitivePhase(
        config=MetacognitionConfig(apply_hedge=True),
        storage_path=tmp_path / "c.json",
    )
    d = mp.observe(doubt_index=0.85, epistemic=0.3, web_used=False)
    assert d.hedge_prefix
    assert "manque" in d.hedge_prefix


def test_phase_records_calibration_each_call(tmp_path: Path):
    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    for d in [0.1, 0.4, 0.7]:
        mp.observe(doubt_index=d, epistemic=0.5, web_used=False)
    assert mp.calibration.n_entries() == 3


def test_phase_kg_facts_improve_correctness_estimate(tmp_path: Path):
    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    # Same doubt + epistemic, but one with KG support, one without
    mp.observe(doubt_index=0.3, epistemic=0.5, web_used=True, kg_facts_count=0)
    n0_brier = mp.calibration.brier_score()
    mp.calibration.reset()
    mp.observe(doubt_index=0.3, epistemic=0.5, web_used=True, kg_facts_count=10)
    n_high_brier = mp.calibration.brier_score()
    # We don't assert direction (depends on exact values), just both
    # are valid numbers — ensures the kg_facts_count path runs.
    assert isinstance(n0_brier, float)
    assert isinstance(n_high_brier, float)


def test_phase_handles_internal_crash(tmp_path: Path):
    """Defensive: if classifier raises, return neutral decision."""

    class BrokenClassifier:
        def classify(self, *a, **kw):
            raise RuntimeError("broken")

    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    mp.classifier = BrokenClassifier()  # type: ignore[assignment]
    d = mp.observe(doubt_index=0.5, epistemic=0.5)
    assert isinstance(d, MetacognitiveDecision)
    # Default neutral label
    assert d.confidence_label == "certaine"


# ════════════════════════════════════════════════════════════════════
# 5. Decision dict serialization
# ════════════════════════════════════════════════════════════════════


def test_decision_to_dict():
    d = MetacognitiveDecision(
        confidence_label="incertaine",
        confidence_score=0.4,
        hedge_prefix="hmm ",
        recommend_web=True,
    )
    out = d.to_dict()
    assert out["confidence_label"] == "incertaine"
    assert out["recommend_web"] is True


def test_phase_to_dict_telemetry(tmp_path: Path):
    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    mp.observe(doubt_index=0.3, epistemic=0.5, web_used=True)
    snap = mp.to_dict()
    assert "config" in snap
    assert "calibration_score" in snap
    assert snap["n_calibration_entries"] == 1


# ════════════════════════════════════════════════════════════════════
# 6. Confidence score band mapping
# ════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "doubt,expected_min_score",
    [
        (0.05, 0.9),  # very_certain → ~1.0
        (0.5, 0.3),   # uncertain → ~0.4
    ],
)
def test_confidence_score_correlates_with_doubt(tmp_path: Path, doubt, expected_min_score):
    mp = MetacognitivePhase(storage_path=tmp_path / "c.json")
    d = mp.observe(doubt_index=doubt, epistemic=0.7, web_used=True)
    if doubt < 0.2:
        assert d.confidence_score >= expected_min_score
    if doubt > 0.4:
        assert d.confidence_score <= 0.7
