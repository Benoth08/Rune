"""Tests for Modern Hopfield Network."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from rune.memory.mhn import ModernHopfieldNetwork


def test_store_and_retrieve():
    """Stored pattern should be retrievable."""
    mhn = ModernHopfieldNetwork(max_patterns=16, dim=64, beta=8.0)
    emb = torch.randn(64)
    mhn.store(emb, "test message")

    results = mhn.retrieve(emb, top_k=1, min_attention=0.01)
    assert len(results) == 1
    assert results[0]["text"] == "test message"


def test_fifo_eviction():
    """Ring buffer should evict oldest pattern when full."""
    mhn = ModernHopfieldNetwork(max_patterns=4, dim=32, beta=8.0)

    for i in range(6):
        emb = torch.randn(32)
        mhn.store(emb, f"msg_{i}")

    assert mhn.n_stored == 4
    # The oldest two (msg_0, msg_1) should have been evicted
    texts = [t for t in mhn.texts if t is not None]
    assert "msg_0" not in texts
    assert "msg_1" not in texts


def test_clear():
    """Clear should reset all state."""
    mhn = ModernHopfieldNetwork(max_patterns=16, dim=32)
    mhn.store(torch.randn(32), "hello")
    mhn.clear()
    assert mhn.n_stored == 0
    results = mhn.retrieve(torch.randn(32), top_k=1, min_attention=0.01)
    assert len(results) == 0


def test_retrieve_empty():
    """Retrieval on empty memory should return empty list."""
    mhn = ModernHopfieldNetwork(max_patterns=8, dim=32)
    results = mhn.retrieve(torch.randn(32))
    assert results == []


def test_persist_and_load():
    """State should survive save/load cycle."""
    mhn = ModernHopfieldNetwork(max_patterns=8, dim=32)
    emb = torch.randn(32)
    mhn.store(emb, "persistent msg")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mhn.pt"
        mhn.save(path)

        mhn2 = ModernHopfieldNetwork(max_patterns=8, dim=32)
        mhn2.load_state(path)

        assert mhn2.n_stored == 1
        results = mhn2.retrieve(emb, top_k=1, min_attention=0.01)
        assert len(results) == 1
        assert results[0]["text"] == "persistent msg"


def test_min_attention_filter():
    """Low-attention results should be filtered out."""
    mhn = ModernHopfieldNetwork(max_patterns=16, dim=32, beta=8.0)

    target = torch.randn(32)
    mhn.store(target, "relevant")

    # Store many distractors
    for i in range(10):
        mhn.store(torch.randn(32), f"noise_{i}")

    results = mhn.retrieve(target, top_k=5, min_attention=0.5)
    # Should return at most the strongly matching one
    assert all(r["attention"] >= 0.5 for r in results)


def test_energy_empty_is_max():
    """Empty MHN should return maximum surprise (1.0)."""
    mhn = ModernHopfieldNetwork(max_patterns=8, dim=32)
    energy = mhn.energy(torch.randn(32))
    assert energy == 1.0


def test_energy_familiar_is_low():
    """Querying a stored pattern should give low energy (low surprise)."""
    mhn = ModernHopfieldNetwork(max_patterns=16, dim=64, beta=8.0)
    target = torch.randn(64)
    mhn.store(target, "familiar")

    energy = mhn.energy(target)
    assert energy < 0.3


def test_energy_novel_is_high():
    """Querying an unseen pattern should give high energy."""
    mhn = ModernHopfieldNetwork(max_patterns=16, dim=64, beta=8.0)
    stored = torch.randn(64)
    mhn.store(stored, "stored")

    novel = torch.randn(64)
    energy = mhn.energy(novel)
    assert energy > 0.5
