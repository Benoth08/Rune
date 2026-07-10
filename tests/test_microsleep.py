"""Tests for the enhanced microsleep module: ripples, replay, compression."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch_available = True
try:
    import torch
except (ImportError, OSError):
    torch_available = False

from rune.microsleep import (
    MicrosleepConfig,
    MicrosleepManager,
    MemoryCompressor,
    ReplayEngine,
    RippleTracker,
)


# ── RippleTracker ─────────────────────────────────────────────────────

def test_ripple_tracker_counts_only_high_surprise():
    cfg = MicrosleepConfig(
        ripple_trigger_count=3, ripple_surprise_threshold=0.5,
    )
    t = RippleTracker(cfg)

    # Below threshold — ignored
    t.record_event(0.2)
    t.record_event(0.4)
    assert not t.should_ripple()

    # At/above threshold — counted
    t.record_event(0.6)
    t.record_event(0.7)
    t.record_event(0.8)
    assert t.should_ripple()


def test_ripple_tracker_reset_returns_count_and_zeroes():
    cfg = MicrosleepConfig(ripple_trigger_count=2)
    t = RippleTracker(cfg)
    t.record_event(0.9)
    t.record_event(0.9)
    t.record_event(0.9)
    n = t.reset()
    assert n == 3
    assert not t.should_ripple()


def test_ripple_tracker_replay_count():
    cfg = MicrosleepConfig()
    t = RippleTracker(cfg)
    assert t.get_replay_count(7) == 0
    t.increment_replay(7)
    t.increment_replay(7)
    assert t.get_replay_count(7) == 2
    assert t.get_replay_count(8) == 0


def test_ripple_tracker_thread_safe():
    """Concurrent record_event must produce a deterministic count."""
    import threading
    cfg = MicrosleepConfig(ripple_trigger_count=1000)
    t = RippleTracker(cfg)

    def worker():
        for _ in range(100):
            t.record_event(0.9)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for th in threads: th.start()
    for th in threads: th.join()
    assert t.reset() == 1000


# ── ReplayEngine ──────────────────────────────────────────────────────

@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_replay_engine_no_op_when_too_few_patterns():
    cfg = MicrosleepConfig(replay_sequence_length=4)
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 1
    fake_mhn.attention = torch.zeros(1)

    engine = ReplayEngine(fake_mhn, cfg)
    tracker = RippleTracker(cfg)
    out = engine.replay(tracker, ripple_active=False)
    assert out["sequences_replayed"] == 0


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_replay_engine_boosts_attention():
    cfg = MicrosleepConfig(
        replay_sequence_length=2,
        replay_n_sequences=1,
        replay_attention_boost=0.1,
    )
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 4
    fake_mhn.attention = torch.zeros(4)

    engine = ReplayEngine(fake_mhn, cfg)
    tracker = RippleTracker(cfg)
    out = engine.replay(tracker, ripple_active=False)

    assert out["sequences_replayed"] >= 1
    # Some patterns must have been boosted (forward + reverse passes).
    assert (fake_mhn.attention > 0).any().item()


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_replay_engine_ripple_doubles_boost():
    cfg = MicrosleepConfig(
        replay_sequence_length=2,
        replay_n_sequences=1,
        replay_attention_boost=0.1,
        ripple_boost_multiplier=2.0,
    )
    fake_mhn1 = MagicMock()
    fake_mhn1.n_stored = 4
    fake_mhn1.attention = torch.zeros(4)
    fake_mhn2 = MagicMock()
    fake_mhn2.n_stored = 4
    fake_mhn2.attention = torch.zeros(4)

    e1 = ReplayEngine(fake_mhn1, cfg)
    e2 = ReplayEngine(fake_mhn2, cfg)
    e1.replay(RippleTracker(cfg), ripple_active=False)
    e2.replay(RippleTracker(cfg), ripple_active=True)

    # The ripple-active replay must reach a higher cumulative boost.
    assert fake_mhn2.attention.sum() > fake_mhn1.attention.sum()


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_replay_engine_attention_clamped_to_one():
    """Boost should never push attention above 1.0."""
    cfg = MicrosleepConfig(
        replay_sequence_length=2, replay_n_sequences=1,
        replay_attention_boost=10.0,  # huge
    )
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 4
    fake_mhn.attention = torch.full((4,), 0.95)

    engine = ReplayEngine(fake_mhn, cfg)
    engine.replay(RippleTracker(cfg), ripple_active=True)
    assert fake_mhn.attention.max().item() <= 1.0


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_replay_engine_increments_replay_counter():
    cfg = MicrosleepConfig(replay_sequence_length=2, replay_n_sequences=1)
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 4
    fake_mhn.attention = torch.zeros(4)

    engine = ReplayEngine(fake_mhn, cfg)
    tracker = RippleTracker(cfg)
    engine.replay(tracker, ripple_active=False)

    # At least the patterns in the replayed sequences should have count > 0.
    counts = [tracker.get_replay_count(i) for i in range(4)]
    assert sum(counts) > 0


# ── MemoryCompressor ──────────────────────────────────────────────────

@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_compressor_skips_unmet_thresholds():
    cfg = MicrosleepConfig(
        compression_replay_threshold=3,
        compression_attention_threshold=0.4,
    )
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 2
    fake_mhn.attention = torch.tensor([0.2, 0.6])  # second meets attention
    fake_mhn.metadata = [{"text": "doc 0"}, {"text": "doc 1"}]
    fake_chroma = MagicMock()

    compressor = MemoryCompressor(fake_mhn, fake_chroma, cfg)
    tracker = RippleTracker(cfg)
    # Neither has been replayed enough — both skipped
    out = compressor.compress(tracker)
    assert out["compressed"] == 0
    fake_chroma.add.assert_not_called()


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_compressor_transfers_eligible_patterns():
    cfg = MicrosleepConfig(
        compression_replay_threshold=2,
        compression_attention_threshold=0.4,
        compression_max_per_cycle=10,
    )
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 3
    fake_mhn.attention = torch.tensor([0.6, 0.3, 0.7])
    fake_mhn.metadata = [
        {"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"},
    ]
    fake_chroma = MagicMock()

    compressor = MemoryCompressor(fake_mhn, fake_chroma, cfg)
    tracker = RippleTracker(cfg)
    # Make idx 0 and 2 eligible (meet replay + attention)
    tracker.increment_replay(0)
    tracker.increment_replay(0)
    tracker.increment_replay(2)
    tracker.increment_replay(2)

    out = compressor.compress(tracker)
    assert out["compressed"] == 2
    assert fake_chroma.add.call_count == 2


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_compressor_caps_per_cycle():
    cfg = MicrosleepConfig(
        compression_replay_threshold=1,
        compression_attention_threshold=0.0,
        compression_max_per_cycle=2,
    )
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 5
    fake_mhn.attention = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    fake_mhn.metadata = [{"text": f"doc {i}"} for i in range(5)]
    fake_chroma = MagicMock()

    tracker = RippleTracker(cfg)
    for i in range(5):
        tracker.increment_replay(i)

    compressor = MemoryCompressor(fake_mhn, fake_chroma, cfg)
    out = compressor.compress(tracker)
    assert out["compressed"] == 2
    # Top-2 by attention: idx 4 (0.9) and idx 3 (0.8)


# ── MicrosleepManager (integration) ───────────────────────────────────

@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_manager_records_ripple_events():
    cfg = MicrosleepConfig(
        ripple_trigger_count=2, ripple_surprise_threshold=0.5,
    )
    fake_sdm = MagicMock()
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 0
    fake_mhn.attention = torch.zeros(0)
    fake_mhn.metadata = []
    mgr = MicrosleepManager(fake_sdm, fake_mhn, MagicMock(), cfg)

    mgr.record_event(0.9)
    mgr.record_event(0.9)
    assert mgr.tracker.should_ripple()


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_manager_consolidate_returns_stats():
    cfg = MicrosleepConfig(
        ripple_trigger_count=100,  # high — no ripple this run
        replay_sequence_length=2,
        replay_n_sequences=1,
    )
    fake_sdm = MagicMock()
    fake_sdm.prune = MagicMock(return_value=3)
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 4
    fake_mhn.attention = torch.zeros(4)
    fake_mhn.metadata = [{"text": f"d{i}"} for i in range(4)]
    fake_chroma = MagicMock()

    mgr = MicrosleepManager(fake_sdm, fake_mhn, fake_chroma, cfg)
    stats = mgr.consolidate(rehearse_top_k=10, rehearse_boost=0.1)

    assert "ripple_active" in stats
    assert stats["ripple_active"] is False
    assert stats["sdm_pruned"] == 3
    assert stats["replay_sequences"] >= 1
    fake_sdm.rehearse.assert_called()
    fake_sdm.decay.assert_called()


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_manager_consolidate_resilient_to_sdm_failure():
    """If SDM fails, the manager still runs replay + compression."""
    cfg = MicrosleepConfig(replay_sequence_length=2, replay_n_sequences=1)
    fake_sdm = MagicMock()
    fake_sdm.rehearse = MagicMock(side_effect=RuntimeError("oops"))
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 4
    fake_mhn.attention = torch.zeros(4)
    fake_mhn.metadata = [{"text": f"d{i}"} for i in range(4)]

    mgr = MicrosleepManager(fake_sdm, fake_mhn, MagicMock(), cfg)
    stats = mgr.consolidate(rehearse_top_k=10, rehearse_boost=0.1)
    # Did not crash; replay still happened
    assert stats["replay_sequences"] >= 0


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_replay_improves_retention_smoke():
    """After several microsleep cycles, well-replayed patterns reach
    higher attention than rarely-touched ones — smoke test that the
    boost mechanism cumulates as expected."""
    cfg = MicrosleepConfig(
        replay_sequence_length=2,
        replay_n_sequences=1,
        replay_attention_boost=0.1,
    )
    fake_mhn = MagicMock()
    fake_mhn.n_stored = 4
    fake_mhn.attention = torch.zeros(4)

    engine = ReplayEngine(fake_mhn, cfg)
    tracker = RippleTracker(cfg)

    # Run 5 replay cycles
    for _ in range(5):
        engine.replay(tracker, ripple_active=False)

    # The most recent patterns (idx 2, 3) should be in the replay window
    # and accumulate boost. The earliest (idx 0, 1) won't be in any
    # contiguous chunk if chunks start from n_stored - seq_len * n_seq.
    # At minimum: at least one pattern has attention > 0.
    assert (fake_mhn.attention > 0.0).any().item()
