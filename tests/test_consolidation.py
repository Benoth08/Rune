"""Unit tests for :class:`lythea.cognition.consolidation.ConsolidationPhase`.

The phase is a thin orchestration layer over the actual consolidation
engine (:class:`MicrosleepManager`) — what we test here is the
threading, locking, persistence, and bookkeeping logic, all
mocked out from the heavy cognitive engine.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch", reason="contents.norm needs tensors")

from rune.cognition.consolidation import ConsolidationPhase  # noqa: E402


def _make_phase():
    """Build a ConsolidationPhase with all dependencies mocked.

    The SDM mock has a ``contents`` attribute that supports ``.norm(dim=1) > 0``
    so the active-rows count in the post-microsleep log doesn't crash.
    """
    sdm = MagicMock()
    # contents.norm(dim=1) returns a tensor of zeros — easy to .sum() on
    contents_norm = torch.zeros(8)
    sdm.contents.norm.return_value = contents_norm

    mhn = MagicMock()
    mhn.n_stored = 0

    kg = MagicMock()
    kg.entities = {}

    git = MagicMock()
    msm = MagicMock()
    msm.consolidate.return_value = {
        "ripple_active": False,
        "replay_sequences": 0,
        "compressed_to_chroma": 0,
    }

    phase = ConsolidationPhase(
        sdm=sdm, mhn=mhn, kg=kg, git=git, microsleep_manager=msm,
    )
    return phase, {"sdm": sdm, "mhn": mhn, "kg": kg, "git": git, "msm": msm}


# ── Public API ─────────────────────────────────────────────────────────

def test_initial_last_microsleep_set_to_now():
    phase, _ = _make_phase()
    assert phase.last_microsleep_ts <= time.time()
    assert phase.last_microsleep_ts > time.time() - 5


def test_record_event_forwards_to_manager():
    phase, m = _make_phase()
    phase.record_event(0.42)
    # V4.1: signature additive — kwargs avec defaults are always
    # passed through. The 1-arg V3 caller still works (the surprise
    # value is preserved); only the kwargs are new.
    m["msm"].record_event.assert_called_once_with(
        0.42,
        affect_intensity=0.0,
        affect_arousal=0.0,
        last_pattern_idx=None,
    )


def test_record_event_swallows_failure():
    """A broken ripple counter must not raise into the orchestrator."""
    phase, m = _make_phase()
    m["msm"].record_event.side_effect = RuntimeError("boom")
    # Must not raise
    phase.record_event(0.5)


def test_bind_exchange_counter_used_in_log():
    """The exchange counter getter is read during _run_microsleep."""
    phase, m = _make_phase()
    phase.bind_exchange_counter(lambda: 42)
    phase._run_microsleep()
    # Just verify it ran without raising and stats were consumed
    m["msm"].consolidate.assert_called_once()


def test_maybe_trigger_at_interval():
    """maybe_trigger fires when count is a multiple of MICROSLEEP_INTERVAL."""
    from rune.config import MICROSLEEP_INTERVAL
    phase, _ = _make_phase()
    triggered = []
    # Patch trigger_microsleep to a recorder
    phase.trigger_microsleep = lambda: triggered.append(True)

    phase.maybe_trigger_after_exchange(MICROSLEEP_INTERVAL - 1)
    assert triggered == []
    phase.maybe_trigger_after_exchange(MICROSLEEP_INTERVAL)
    assert triggered == [True]
    phase.maybe_trigger_after_exchange(MICROSLEEP_INTERVAL * 2)
    assert triggered == [True, True]


def test_maybe_trigger_at_zero_does_not_fire():
    """Count 0 must NOT trigger — we don't want a microsleep at startup."""
    phase, _ = _make_phase()
    triggered = []
    phase.trigger_microsleep = lambda: triggered.append(True)
    phase.maybe_trigger_after_exchange(0)
    assert triggered == []


# ── Microsleep cycle ───────────────────────────────────────────────────

def test_run_microsleep_persists_and_pushes():
    phase, m = _make_phase()
    phase._run_microsleep()
    m["sdm"].save.assert_called_once()
    m["mhn"].save.assert_called_once()
    m["kg"].cleanup_pending.assert_called_once()
    m["kg"].promote_pending.assert_called_once()
    m["kg"].save.assert_called_once()
    m["git"].push_async.assert_called_once()


def test_run_microsleep_updates_last_ts():
    phase, _ = _make_phase()
    initial = phase.last_microsleep_ts
    time.sleep(0.01)
    phase._run_microsleep()
    assert phase.last_microsleep_ts > initial


def test_run_microsleep_calls_consolidate_with_settings():
    from rune.config import MICROSLEEP_BOOST, MICROSLEEP_REHEARSE_K
    phase, m = _make_phase()
    phase._run_microsleep()
    call = m["msm"].consolidate.call_args
    assert call.kwargs["rehearse_top_k"] == MICROSLEEP_REHEARSE_K
    assert call.kwargs["rehearse_boost"] == MICROSLEEP_BOOST


def test_run_microsleep_failure_clears_pending():
    """If consolidate raises, pending flag must still be cleared."""
    phase, m = _make_phase()
    m["msm"].consolidate.side_effect = RuntimeError("boom")
    # Set the flag like trigger_microsleep would
    phase._microsleep_pending = True
    phase._run_microsleep()
    assert phase._microsleep_pending is False


def test_anti_stacking_returns_when_locked():
    """If the lock is already held, _run_microsleep must return immediately."""
    phase, m = _make_phase()
    phase._microsleep_lock.acquire()
    try:
        phase._microsleep_pending = True
        phase._run_microsleep()
        # consolidate must NOT have been called — we returned early
        m["msm"].consolidate.assert_not_called()
        # pending must still be cleared so the next trigger can fire
        assert phase._microsleep_pending is False
    finally:
        phase._microsleep_lock.release()


def test_trigger_microsleep_no_double_start():
    """Two trigger_microsleep calls in a row must not start two threads."""
    phase, m = _make_phase()
    # Make consolidate slow so the first thread is still running
    import threading
    started = threading.Event()
    blocker = threading.Event()
    def slow_consolidate(**_):
        started.set()
        blocker.wait(timeout=2)
        return {"replay_sequences": 0}
    m["msm"].consolidate.side_effect = slow_consolidate

    phase.trigger_microsleep()
    started.wait(timeout=2)            # first thread is in
    phase.trigger_microsleep()         # second call — should be a no-op
    blocker.set()                       # unblock first thread
    # Wait a bit for the thread to wind down
    time.sleep(0.05)
    # consolidate was called exactly once
    assert m["msm"].consolidate.call_count == 1


# ── Inactivity timer ───────────────────────────────────────────────────

def test_reset_inactivity_timer_creates_timer():
    phase, _ = _make_phase()
    phase.reset_inactivity_timer()
    assert phase._inactivity_timer is not None
    assert phase._inactivity_timer.daemon is True
    phase.cancel_inactivity_timer()


def test_cancel_inactivity_timer_clears():
    phase, _ = _make_phase()
    phase.reset_inactivity_timer()
    phase.cancel_inactivity_timer()
    assert phase._inactivity_timer is None


def test_reset_cancels_previous_timer():
    """Calling reset twice replaces the timer cleanly."""
    phase, _ = _make_phase()
    phase.reset_inactivity_timer()
    first = phase._inactivity_timer
    phase.reset_inactivity_timer()
    second = phase._inactivity_timer
    assert first is not second
    phase.cancel_inactivity_timer()


# ── Deep sleep ─────────────────────────────────────────────────────────

def test_deep_sleep_prunes_persists_flushes_pushes():
    phase, m = _make_phase()
    msg = phase.deep_sleep()
    m["sdm"].prune.assert_called_once()
    m["sdm"].save.assert_called_once()
    m["mhn"].save.assert_called_once()
    m["kg"].save.assert_called_once()
    m["sdm"].flush.assert_called_once()
    # MHN must NOT be flushed during deep sleep
    m["mhn"].clear.assert_not_called()
    m["git"].push_async.assert_called_once()
    assert "terminée" in msg


def test_deep_sleep_uses_aggressive_prune_threshold():
    phase, m = _make_phase()
    phase.deep_sleep()
    call = m["sdm"].prune.call_args
    assert call.kwargs.get("threshold") == 0.5


# ── Failure isolation across calls ─────────────────────────────────────

def test_record_event_with_non_numeric_logs_and_continues():
    """A non-numeric surprise value must not raise."""
    phase, m = _make_phase()
    # Simulate a string value sneaking in (defensive — should not happen)
    m["msm"].record_event.side_effect = lambda v: float(v)  # will fail
    # Must not raise
    phase.record_event("not a number")  # type: ignore
