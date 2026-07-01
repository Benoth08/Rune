"""Tests for the Hippocampe cognitive orchestrator."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import chromadb
import torch

from rune.git_sync import GitSync
from rune.hippocampe import Hippocampe
from rune.memory.kg import KnowledgeGraphStore
from rune.memory.mhn import ModernHopfieldNetwork
from rune.memory.sdm import SparseDistributedMemory


def _make_hippocampe(tmp_dir: str) -> Hippocampe:
    """Create a Hippocampe with mocked model and in-memory Chroma."""
    model = MagicMock()
    model.is_loaded = True
    model.model_id = "test-model"
    model.hidden_dim = 64
    model.analyze_input.return_value = {
        "token_entropies": [0.3, 0.5, 0.1],
        "latent_states": torch.randn(3, 64),
        "mean_entropy": 0.3,
        "vocab_size": 32000,
    }

    sdm = SparseDistributedMemory(dim=64, rows=128, k=4)
    mhn = ModernHopfieldNetwork(max_patterns=16, dim=32)

    client = chromadb.Client()
    coll = client.get_or_create_collection("test_memory")

    tmp = Path(tmp_dir)
    kg = KnowledgeGraphStore(persist_dir=tmp / "kg")
    git = GitSync(tmp / "git")

    return Hippocampe(
        model=model,
        sdm=sdm,
        mhn=mhn,
        chroma_collection=coll,
        git=git,
        kg=kg,
    )


def test_reset_session():
    with tempfile.TemporaryDirectory() as tmp:
        h = _make_hippocampe(tmp)
        # Write something to SDM
        pattern = torch.sign(torch.randn(1, 64))
        h.sdm.write(pattern, pattern, strength=2.0)
        h.mhn.store(torch.randn(32), "test")

        h.reset_session()

        assert h.sdm.contents.abs().sum().item() == 0
        assert h.mhn.n_stored == 0
        assert h.exchange_count == 0


def test_memory_status():
    with tempfile.TemporaryDirectory() as tmp:
        h = _make_hippocampe(tmp)
        status = h.memory_status()
        assert "sdm" in status
        assert "mhn" in status
        assert "kg" in status
        assert "chroma" in status
        assert status["sdm"]["total_rows"] == 128


def test_deep_sleep():
    with tempfile.TemporaryDirectory() as tmp:
        h = _make_hippocampe(tmp)
        pattern = torch.sign(torch.randn(1, 64))
        h.sdm.write(pattern, pattern, strength=2.0)
        h.mhn.store(torch.randn(32), "episode")

        msg = h.deep_sleep()

        assert "terminée" in msg
        assert h.sdm.contents.abs().sum().item() == 0
        # MHN should be preserved after deep sleep
        assert h.mhn.n_stored == 1


def test_phase_a_nonsalient_skips():
    with tempfile.TemporaryDirectory() as tmp:
        h = _make_hippocampe(tmp)
        result = h._phase_a_learn("ok")
        assert not result["salient"]
        assert result["surprise"]["global"] == 0.0


def test_reasoning_label_disambiguation():
    """The Réflexion debug label distinguishes thinking-native from two-pass.

    Three mutually-exclusive states:
      - Thinking model → label says toggle is ignored, even if it's on.
      - Non-thinking + toggle on → label says two-pass is active.
      - Non-thinking + toggle off → label says disabled.

    This was added after a UI confusion where users saw a reasoning
    panel from a Qwen3-Thinking <think> block and thought the toggle
    was broken. The label now makes the situation explicit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        h = _make_hippocampe(tmp)

        # 1. Thinking model — toggle ignored regardless of value.
        h.model.is_thinking = True
        h.reasoning_enabled = True
        assert "thinking natif" in h._reasoning_label()
        h.reasoning_enabled = False
        assert "thinking natif" in h._reasoning_label()

        # 2. Non-thinking + toggle on — two-pass active.
        h.model.is_thinking = False
        h.reasoning_enabled = True
        assert h._reasoning_label() == "two-pass activé"

        # 3. Non-thinking + toggle off — disabled.
        h.reasoning_enabled = False
        assert h._reasoning_label() == "désactivée"
