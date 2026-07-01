"""V4.1 — Tests for microsleep affect-modulated consolidation.

Sections
--------
1. Sentinel : V3.9.4 record_event(surprise) signature still works.
2. V4.1 affect flagging : threshold, no-op when disabled, FIFO bound.
3. clear_affect_flags drains.
4. record_event handles invalid kwargs gracefully.
5. ConsolidationPhase.record_event new kwargs forward to tracker.

Note: replay boost integration is exercised in higher-level
torch-dependent tests (test_microsleep.py with torch); pure-Python
tests focus on flagging mechanics.
"""

import pytest

from rune.microsleep import MicrosleepConfig, RippleTracker


# ════════════════════════════════════════════════════════════════════
# 1. Sentinel — V3 signature backward compat
# ════════════════════════════════════════════════════════════════════


def test_v41_old_record_event_signature_still_works():
    """SENTINEL: V3.9.4 callers using `record_event(0.7)` must keep working."""
    cfg = MicrosleepConfig()
    tracker = RippleTracker(cfg)
    # 1-arg positional, exactly as in V3.9.4 callers
    tracker.record_event(0.7)
    tracker.record_event(0.7)
    # Below threshold
    tracker.record_event(0.1)
    # Counter should reflect only the 2 above-threshold events
    assert tracker._counter == 2


def test_v41_old_signature_ripple_threshold_unchanged():
    """V3 ripple-trigger semantics unchanged when V4.1 is off."""
    cfg = MicrosleepConfig(ripple_trigger_count=3, ripple_surprise_threshold=0.5)
    tracker = RippleTracker(cfg)
    for _ in range(2):
        tracker.record_event(0.7)
    assert tracker.should_ripple() is False
    tracker.record_event(0.7)
    assert tracker.should_ripple() is True


# ════════════════════════════════════════════════════════════════════
# 2. V4.1 default-off — affect kwargs are no-ops
# ════════════════════════════════════════════════════════════════════


def test_v41_disabled_by_default_no_flagging():
    """When affect_modulates=False (default), passing affect kwargs is a no-op."""
    cfg = MicrosleepConfig()  # affect_modulates defaults to False
    tracker = RippleTracker(cfg)
    tracker.record_event(
        0.7,
        affect_intensity=0.9,
        affect_arousal=0.9,
        last_pattern_idx=42,
    )
    assert tracker.affect_flagged_count() == 0
    assert tracker.is_affect_flagged(42) is False


# ════════════════════════════════════════════════════════════════════
# 3. V4.1 enabled — flagging threshold
# ════════════════════════════════════════════════════════════════════


def test_v41_high_arousal_flags_pattern():
    cfg = MicrosleepConfig(
        affect_modulates=True,
        affect_ripple_arousal_threshold=0.5,
    )
    tracker = RippleTracker(cfg)
    tracker.record_event(0.7, affect_arousal=0.8, last_pattern_idx=42)
    assert tracker.is_affect_flagged(42) is True
    assert tracker.affect_flagged_count() == 1


def test_v41_low_arousal_does_not_flag():
    cfg = MicrosleepConfig(
        affect_modulates=True,
        affect_ripple_arousal_threshold=0.6,
    )
    tracker = RippleTracker(cfg)
    tracker.record_event(0.7, affect_arousal=0.3, last_pattern_idx=42)
    assert tracker.affect_flagged_count() == 0


def test_v41_missing_pattern_idx_does_not_flag():
    cfg = MicrosleepConfig(
        affect_modulates=True,
        affect_ripple_arousal_threshold=0.5,
    )
    tracker = RippleTracker(cfg)
    tracker.record_event(0.7, affect_arousal=0.9, last_pattern_idx=None)
    assert tracker.affect_flagged_count() == 0
    tracker.record_event(0.7, affect_arousal=0.9, last_pattern_idx=-1)
    assert tracker.affect_flagged_count() == 0


# ════════════════════════════════════════════════════════════════════
# 4. FIFO bound (64)
# ════════════════════════════════════════════════════════════════════


def test_v41_fifo_bound_64():
    cfg = MicrosleepConfig(
        affect_modulates=True,
        affect_ripple_arousal_threshold=0.0,
    )
    tracker = RippleTracker(cfg)
    for i in range(80):
        tracker.record_event(0.7, affect_arousal=0.9, last_pattern_idx=i)
    # FIFO bound at 64 — oldest entries dropped
    assert tracker.affect_flagged_count() == 64
    # Entries 0-15 should be gone, 16-79 present
    assert tracker.is_affect_flagged(0) is False
    assert tracker.is_affect_flagged(15) is False
    assert tracker.is_affect_flagged(79) is True


# ════════════════════════════════════════════════════════════════════
# 5. Drain behavior
# ════════════════════════════════════════════════════════════════════


def test_v41_clear_affect_flags_drains():
    cfg = MicrosleepConfig(affect_modulates=True, affect_ripple_arousal_threshold=0.5)
    tracker = RippleTracker(cfg)
    for i in range(3):
        tracker.record_event(0.7, affect_arousal=0.9, last_pattern_idx=i)
    drained = tracker.clear_affect_flags()
    assert drained == [0, 1, 2]
    assert tracker.affect_flagged_count() == 0


def test_v41_clear_on_empty_returns_empty():
    cfg = MicrosleepConfig(affect_modulates=True)
    tracker = RippleTracker(cfg)
    assert tracker.clear_affect_flags() == []


# ════════════════════════════════════════════════════════════════════
# 6. Defensive: invalid kwargs don't crash
# ════════════════════════════════════════════════════════════════════


def test_v41_non_numeric_arousal_does_not_crash():
    cfg = MicrosleepConfig(affect_modulates=True, affect_ripple_arousal_threshold=0.5)
    tracker = RippleTracker(cfg)
    tracker.record_event(0.7, affect_arousal="not a number", last_pattern_idx=1)  # type: ignore[arg-type]
    assert tracker.affect_flagged_count() == 0


# ════════════════════════════════════════════════════════════════════
# 7. ConsolidationPhase — kwargs forward
# ════════════════════════════════════════════════════════════════════


def test_v41_consolidation_phase_record_event_forwards_kwargs():
    """ConsolidationPhase.record_event should pass V4.1 kwargs through."""
    captured: dict = {}

    class FakeManager:
        def record_event(self, surprise, **kwargs):
            captured["surprise"] = surprise
            captured["kwargs"] = kwargs

    # Build a minimal ConsolidationPhase wrapping the fake manager.
    from rune.cognition.consolidation import ConsolidationPhase

    cp = ConsolidationPhase.__new__(ConsolidationPhase)
    cp._microsleep_manager = FakeManager()  # type: ignore[attr-defined]
    cp.record_event(
        0.8,
        affect_intensity=0.5,
        affect_arousal=0.7,
        last_pattern_idx=42,
    )
    assert captured["surprise"] == 0.8
    assert captured["kwargs"]["affect_intensity"] == 0.5
    assert captured["kwargs"]["affect_arousal"] == 0.7
    assert captured["kwargs"]["last_pattern_idx"] == 42


def test_v41_consolidation_phase_v3_signature_still_works():
    """SENTINEL: ConsolidationPhase.record_event(0.8) still valid."""
    captured: dict = {}

    class FakeManager:
        def record_event(self, surprise, **kwargs):
            captured["surprise"] = surprise

    from rune.cognition.consolidation import ConsolidationPhase

    cp = ConsolidationPhase.__new__(ConsolidationPhase)
    cp._microsleep_manager = FakeManager()  # type: ignore[attr-defined]
    cp.record_event(0.8)  # V3 signature
    assert captured["surprise"] == 0.8


# ════════════════════════════════════════════════════════════════════
# 8. Multiple flags + same idx
# ════════════════════════════════════════════════════════════════════


def test_v41_repeated_flag_same_idx_accumulates():
    """Same pattern flagged twice → appears twice (helps boost saturation)."""
    cfg = MicrosleepConfig(affect_modulates=True, affect_ripple_arousal_threshold=0.0)
    tracker = RippleTracker(cfg)
    tracker.record_event(0.7, affect_arousal=0.9, last_pattern_idx=42)
    tracker.record_event(0.7, affect_arousal=0.9, last_pattern_idx=42)
    assert tracker.affect_flagged_count() == 2
    assert tracker.is_affect_flagged(42) is True
