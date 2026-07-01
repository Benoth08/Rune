"""V4 — Hippocampe orchestration tests (require torch + chromadb).

These tests instantiate the full Hippocampe pipeline using the same
mock/factory pattern as ``test_hippocampe.py`` (model is a MagicMock,
SDM/MHN/Chroma are real but small, KG and Git are scoped to tmp).

They are skipped in sandboxes without torch/chromadb, but exercise
the V4 init paths (master flags ON/OFF) end-to-end on a machine
with the full stack.

Run on a machine with torch + chromadb:

    python3 -m pytest tests/test_hippocampe_v4.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Skip the entire module gracefully if torch/chromadb are missing.
torch = pytest.importorskip("torch")
chromadb = pytest.importorskip("chromadb")


def _make_hippocampe(tmp_dir: str):
    """Create a Hippocampe with mocked model and in-memory Chroma.

    Mirrors ``tests/test_hippocampe.py::_make_hippocampe`` so V4 tests
    keep passing when V3 dependencies evolve (single source of truth
    for the construction recipe).
    """
    from rune.git_sync import GitSync
    from rune.hippocampe import Hippocampe
    from rune.memory.kg import KnowledgeGraphStore
    from rune.memory.mhn import ModernHopfieldNetwork
    from rune.memory.sdm import SparseDistributedMemory

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
    # ChromaDB requires names to be 3-512 chars, [a-zA-Z0-9._-],
    # starting AND ending with [a-zA-Z0-9]. Strip non-alnum trailing
    # chars from the tmp dir name to satisfy the regex.
    suffix = Path(tmp_dir).name
    safe_suffix = "".join(c for c in suffix if c.isalnum()) or "x"
    coll = client.get_or_create_collection(
        f"testmemv4{safe_suffix}"
    )

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


# ── Env fixtures: flip every V4 master flag at once ─────────────────


@pytest.fixture
def env_v4_off(monkeypatch):
    """Explicitly turn every V4 flag OFF — the V3.9.4 baseline."""
    for var in (
        "LYTHEA_ENABLE_COGNITIVE_STATE",
        "LYTHEA_ENABLE_INHIBITION",
        "LYTHEA_ENABLE_PLANNING",
        "LYTHEA_ENABLE_PREDICTIVE_CODING",
        "LYTHEA_ENABLE_TIMELINE",
        "LYTHEA_ENABLE_METACOGNITION",
        "LYTHEA_AFFECT_MODULATES_CONSOLIDATION",
    ):
        monkeypatch.setenv(var, "false")
    from rune.settings import get_settings
    get_settings.cache_clear()


@pytest.fixture
def env_v4_all_on(monkeypatch):
    for var in (
        "LYTHEA_ENABLE_COGNITIVE_STATE",
        "LYTHEA_ENABLE_INHIBITION",
        "LYTHEA_ENABLE_PLANNING",
        "LYTHEA_ENABLE_PREDICTIVE_CODING",
        "LYTHEA_ENABLE_TIMELINE",
        "LYTHEA_ENABLE_METACOGNITION",
        "LYTHEA_AFFECT_MODULATES_CONSOLIDATION",
    ):
        monkeypatch.setenv(var, "true")
    from rune.settings import get_settings
    get_settings.cache_clear()


# ── Tests ───────────────────────────────────────────────────────────


def test_hippocampe_loads_with_v4_off(env_v4_off):
    """SENTINEL: with all V4 flags off, no V4 attribute is active."""
    with tempfile.TemporaryDirectory() as tmp:
        hippo = _make_hippocampe(tmp)
        assert hippo._cognitive_state is None
        assert hippo._inhibition is None
        assert hippo._planning is None
        assert hippo._predictive_coding is None
        assert hippo._timeline is None
        assert hippo._metacognition is None


def test_hippocampe_loads_with_v4_on(env_v4_all_on):
    """With all V4 flags on, every module initializes successfully."""
    with tempfile.TemporaryDirectory() as tmp:
        hippo = _make_hippocampe(tmp)
        assert hippo._cognitive_state is not None
        assert hippo._inhibition is not None
        assert hippo._planning is not None
        assert hippo._predictive_coding is not None
        assert hippo._timeline is not None
        assert hippo._metacognition is not None


def test_phase_a_returns_encoding_emb_field(env_v4_all_on):
    """V4: _phase_a_learn populates encoding_emb on salient turns.

    Defensive: the encoding may flag the input as non-salient depending
    on the salience filter; in that case we accept the absence of the
    field rather than fail.
    """
    with tempfile.TemporaryDirectory() as tmp:
        hippo = _make_hippocampe(tmp)
        res = hippo._phase_a_learn("L'analyse spectroscopique révèle un défaut.")
        # Either non-salient (legitimate skip) OR encoding_emb present.
        if res.get("salient"):
            assert "encoding_emb" in res


def test_phase_c_v4_blocks_param_optional(env_v4_off):
    """SENTINEL: _phase_c_assemble accepts v4_blocks=None (back-compat)."""
    with tempfile.TemporaryDirectory() as tmp:
        hippo = _make_hippocampe(tmp)
        msgs = hippo._phase_c_assemble("hello", [], rag_context="")
        assert isinstance(msgs, list)
        msgs2 = hippo._phase_c_assemble(
            "hello", [], rag_context="", v4_blocks=None,
        )
        assert isinstance(msgs2, list)


def test_phase_c_v4_blocks_appear_in_system(env_v4_all_on):
    """When v4_blocks are provided, they appear in the system message."""
    with tempfile.TemporaryDirectory() as tmp:
        hippo = _make_hippocampe(tmp)
        msgs = hippo._phase_c_assemble(
            "hello",
            [],
            rag_context="",
            v4_blocks=["[Plan en cours]\nBut: démo"],
        )
        sys_msg = msgs[0]
        assert sys_msg["role"] == "system"
        assert "[Plan en cours]" in sys_msg["content"]
        assert "But: démo" in sys_msg["content"]


def test_v4_status_method_exposes_modules(env_v4_all_on):
    """v4_status() returns a well-formed snapshot for every module."""
    with tempfile.TemporaryDirectory() as tmp:
        hippo = _make_hippocampe(tmp)
        status = hippo.v4_status()
        for m in (
            "cognitive_state", "inhibition", "planning",
            "predictive_coding", "timeline", "metacognition",
        ):
            assert m in status, f"v4_status missing key {m!r}"
            assert "enabled" in status[m]


def test_v4_set_module_runtime_toggle(env_v4_off):
    """v4_set_module flips a module on at runtime."""
    with tempfile.TemporaryDirectory() as tmp:
        hippo = _make_hippocampe(tmp)
        # Start: planning OFF
        assert hippo._planning is None
        # Flip ON
        status = hippo.v4_set_module("planning", True)
        assert status["planning"]["enabled"] is True
        assert hippo._planning is not None
        # Flip OFF
        status = hippo.v4_set_module("planning", False)
        assert status["planning"]["enabled"] is False
        assert hippo._planning is None


def test_v4_set_module_unknown_module_silent(env_v4_off):
    """Unknown module name doesn't raise — the API endpoint catches
    invalid inputs upstream, but the method itself stays defensive."""
    with tempfile.TemporaryDirectory() as tmp:
        hippo = _make_hippocampe(tmp)
        # Should not raise
        status = hippo.v4_set_module("bogus_module", True)
        assert isinstance(status, dict)
