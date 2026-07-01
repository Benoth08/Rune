"""V4.2 — Tests for predictive_coding module."""

import math

import pytest

from rune.cognition.predictive_coding import (
    GatingDecision,
    PredictiveCodingConfig,
    PredictiveCodingPhase,
    _ema_predict,
    _l2_norm,
    cosine_distance,
)


# ════════════════════════════════════════════════════════════════════
# 1. Math helpers
# ════════════════════════════════════════════════════════════════════


def test_l2_norm_basic():
    assert abs(_l2_norm([3.0, 4.0]) - 5.0) < 1e-9


def test_l2_norm_zero():
    assert _l2_norm([0.0, 0.0, 0.0]) == 0.0


def test_l2_norm_empty():
    assert _l2_norm([]) == 0.0


def test_cosine_distance_identical():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_distance_orthogonal():
    assert abs(cosine_distance([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-9


def test_cosine_distance_opposite():
    # cos = -1 → distance = 2
    assert abs(cosine_distance([1.0, 0.0], [-1.0, 0.0]) - 2.0) < 1e-9


def test_cosine_distance_zero_vector():
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_distance_mismatched_lengths():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0, 0.0]) == 1.0


def test_cosine_distance_empty():
    assert cosine_distance([], [1.0]) == 1.0


def test_ema_predict_single_history():
    out = _ema_predict([[1.0, 2.0]], decay=0.6)
    assert out == [1.0, 2.0]


def test_ema_predict_recency_weighted():
    """Most recent vector should dominate."""
    history = [[10.0, 0.0], [0.0, 10.0]]  # old, new
    out = _ema_predict(history, decay=0.5)
    # weights: [0.5, 1.0] → normalized [1/3, 2/3]
    # out = 1/3·[10,0] + 2/3·[0,10] = [3.33, 6.67]
    assert out[1] > out[0]  # second component dominates


def test_ema_predict_empty_history():
    assert _ema_predict([], decay=0.5) == []


# ════════════════════════════════════════════════════════════════════
# 2. PredictiveCodingPhase — cold start
# ════════════════════════════════════════════════════════════════════


def test_cold_start_returns_full():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(cold_start_min=3))
    for i in range(2):
        d = pc.observe([1.0, 0.0])
        assert d.mode == "full"
        assert "cold-start" in d.reason


def test_no_embedding_returns_full():
    pc = PredictiveCodingPhase()
    d = pc.observe(None)
    assert d.mode == "full"
    d = pc.observe([])
    assert d.mode == "full"


def test_non_numeric_embedding_returns_full():
    pc = PredictiveCodingPhase()
    d = pc.observe(["not", "a", "vector"])  # type: ignore[list-item]
    assert d.mode == "full"
    assert "non-numeric" in d.reason


# ════════════════════════════════════════════════════════════════════
# 3. PredictiveCodingPhase — gating modes
# ════════════════════════════════════════════════════════════════════


def test_low_power_when_repeating():
    """Same embedding 4 times → after cold-start, surprise≈0 → low_power."""
    pc = PredictiveCodingPhase(
        PredictiveCodingConfig(cold_start_min=2, low_threshold=0.05)
    )
    vec = [1.0, 0.0, 0.0]
    pc.observe(vec)  # cold 1/2
    pc.observe(vec)  # cold 2/2
    d = pc.observe(vec)  # post cold-start, perfect prediction
    assert d.mode == "low_power"


def test_high_when_orthogonal_jump():
    """Orthogonal jump after stable history → high mode."""
    pc = PredictiveCodingPhase(
        PredictiveCodingConfig(cold_start_min=2, high_threshold=0.5)
    )
    pc.observe([1.0, 0.0, 0.0])
    pc.observe([1.0, 0.0, 0.0])
    pc.observe([1.0, 0.0, 0.0])  # post cold-start, low surprise
    d = pc.observe([0.0, 1.0, 0.0])  # orthogonal — high surprise
    assert d.mode == "high"


def test_full_for_intermediate_distance():
    pc = PredictiveCodingPhase(
        PredictiveCodingConfig(
            cold_start_min=2, low_threshold=0.05, high_threshold=0.95
        )
    )
    pc.observe([1.0, 0.0])
    pc.observe([1.0, 0.0])
    # Half-rotated → cosine ≈ 0.7, distance ≈ 0.3
    d = pc.observe([0.7, 0.7])
    assert d.mode == "full"


# ════════════════════════════════════════════════════════════════════
# 4. Confidence + error fields
# ════════════════════════════════════════════════════════════════════


def test_confidence_capped():
    pc = PredictiveCodingPhase(
        PredictiveCodingConfig(cold_start_min=2, confidence_cap=0.85)
    )
    pc.observe([1.0, 0.0])
    pc.observe([1.0, 0.0])
    d = pc.observe([1.0, 0.0])
    assert d.confidence <= 0.85


def test_error_in_valid_range():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(cold_start_min=1))
    pc.observe([1.0, 0.0])
    d = pc.observe([0.0, 1.0])
    assert 0.0 <= d.error <= 2.0


def test_decision_has_reason_string():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(cold_start_min=1))
    d = pc.observe([1.0, 0.0])
    assert d.reason  # non-empty


# ════════════════════════════════════════════════════════════════════
# 5. State management
# ════════════════════════════════════════════════════════════════════


def test_history_bounded_by_history_size():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(history_size=4))
    for _ in range(20):
        pc.observe([1.0, 0.0])
    assert len(pc._history) == 4


def test_n_observations_counts():
    pc = PredictiveCodingPhase()
    for _ in range(5):
        pc.observe([1.0, 0.0])
    assert pc.n_observations == 5


def test_last_decision_cached():
    pc = PredictiveCodingPhase()
    d = pc.observe([1.0, 0.0])
    assert pc.last_decision is d


def test_reset_clears_state():
    pc = PredictiveCodingPhase()
    for _ in range(5):
        pc.observe([1.0, 0.0])
    pc.reset()
    assert pc.n_observations == 0
    assert pc.last_decision is None
    assert len(pc._history) == 0


# ════════════════════════════════════════════════════════════════════
# 6. GatingDecision serialization
# ════════════════════════════════════════════════════════════════════


def test_gating_decision_to_dict():
    d = GatingDecision(mode="low_power", confidence=0.7, error=0.1, reason="ok")
    out = d.to_dict()
    assert out["mode"] == "low_power"
    assert out["error"] == 0.1


# ════════════════════════════════════════════════════════════════════
# 7. Defensive: mismatched dimensions across calls
# ════════════════════════════════════════════════════════════════════


def test_mismatched_embedding_dims_no_crash():
    pc = PredictiveCodingPhase(PredictiveCodingConfig(cold_start_min=2))
    pc.observe([1.0, 0.0, 0.0])
    pc.observe([1.0, 0.0, 0.0])
    # Now feed a different dim — must not crash
    d = pc.observe([1.0, 0.0])
    assert isinstance(d, GatingDecision)
    assert d.mode == "full"


# ════════════════════════════════════════════════════════════════════
# 8. Determinism
# ════════════════════════════════════════════════════════════════════


def test_deterministic_for_same_inputs():
    pc1 = PredictiveCodingPhase(PredictiveCodingConfig(cold_start_min=2))
    pc2 = PredictiveCodingPhase(PredictiveCodingConfig(cold_start_min=2))
    seq = [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    decisions1 = [pc1.observe(v) for v in seq]
    decisions2 = [pc2.observe(v) for v in seq]
    for d1, d2 in zip(decisions1, decisions2):
        assert d1.mode == d2.mode
        assert abs(d1.error - d2.error) < 1e-9
