"""Unit tests for :class:`lythea.cognition.surprise.SurprisePhase`.

Each of the four signals is tested in isolation, plus the
JSON-shape contract via ``as_dict``, plus the post-generation
doubt + epistemic mapping. All backend objects are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch", reason="surprise paths use tensors")

from rune.cognition.surprise import (  # noqa: E402
    DOUBT_FACT_MAX,
    DOUBT_INTUITION_MAX,
    EPISTEMIC_FACT,
    EPISTEMIC_HYPOTHESIS,
    EPISTEMIC_INTUITION,
    SurprisePhase,
    SurpriseSignals,
)


def _make_phase(
    *,
    mhn_energy: float | None = 0.5,
    sdm_cos: float | None = 0.7,
    retriever_score: float | None = None,
    model_loaded: bool = True,
) -> tuple[SurprisePhase, dict[str, MagicMock]]:
    """Build a SurprisePhase with controlled mocks for each signal."""
    sdm = MagicMock()
    if sdm_cos is not None:
        # Mock project + read so cosine_similarity returns sdm_cos.
        # We rig it the simplest way: vec and prior are colinear with
        # a scaling that yields the requested cosine.
        v = torch.tensor([1.0, 0.0, 0.0, 0.0])
        sdm.project.return_value = v
        # Rotate prior so cos(v, prior) = sdm_cos (use 2D for simplicity)
        import math
        angle = math.acos(max(min(sdm_cos, 1.0), -1.0))
        prior = torch.tensor([math.cos(angle), math.sin(angle), 0.0, 0.0])
        sdm.read.return_value = prior

    mhn = MagicMock()
    if mhn_energy is not None:
        mhn.energy.return_value = mhn_energy

    model = MagicMock()
    model.is_loaded = model_loaded
    model.model_id = "test-model"
    model.hidden_dim = 4

    retriever = None
    if retriever_score is not None:
        retriever = MagicMock()
        retriever.search.return_value = [{"score": retriever_score}]

    phase = SurprisePhase(sdm=sdm, mhn=mhn, model=model, retriever=retriever)
    return phase, {"sdm": sdm, "mhn": mhn, "model": model, "retriever": retriever}


# ── as_dict contract ───────────────────────────────────────────────────

def test_as_dict_keys_match_legacy_contract():
    """The frontend / Storage rely on this exact set of keys.

    Note ``"global"`` is the bare key (not ``global_``) — it must
    survive the dataclass-to-dict translation."""
    s = SurpriseSignals(
        structural=0.5, episodic=0.6, predictive=0.7,
        chroma_discount=0.1, composite=0.55, global_=0.495,
    )
    d = s.as_dict()
    assert set(d.keys()) == {
        "structural", "episodic", "predictive",
        "chroma_discount", "composite", "global",
    }
    # rounding to 3 decimals
    assert d["global"] == 0.495
    assert d["structural"] == 0.5


def test_as_dict_clips_global_at_one():
    """Even if S_composite × (1-discount) > 1 numerically, the
    JSON shape must clip — Storage uses this for SDM strength."""
    s = SurpriseSignals(global_=1.7)
    assert s.as_dict()["global"] == 1.0


# ── Structural signal ──────────────────────────────────────────────────

def test_structural_clipped_at_one():
    phase, _ = _make_phase()
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=2.5, model_latent=None,
    )
    assert sig.structural == 1.0


def test_structural_passthrough_below_one():
    phase, _ = _make_phase()
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.42, model_latent=None,
    )
    assert sig.structural == pytest.approx(0.42)


# ── Episodic signal ────────────────────────────────────────────────────

def test_episodic_uses_mhn_energy():
    phase, m = _make_phase(mhn_energy=0.33)
    sig = phase.compute(
        text="x", gliner_emb=torch.zeros(8),
        structural_entropy=0.5, model_latent=None,
    )
    assert sig.episodic == pytest.approx(0.33)
    m["mhn"].energy.assert_called_once()


def test_episodic_neutral_when_no_embedding():
    """No GLiNER embedding → fall back to neutral 1.0 (= maximally novel)."""
    phase, m = _make_phase()
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5, model_latent=None,
    )
    assert sig.episodic == 1.0
    m["mhn"].energy.assert_not_called()


def test_episodic_neutral_on_mhn_failure():
    phase, m = _make_phase()
    m["mhn"].energy.side_effect = RuntimeError("dim mismatch")
    sig = phase.compute(
        text="x", gliner_emb=torch.zeros(8),
        structural_entropy=0.5, model_latent=None,
    )
    assert sig.episodic == 1.0


# ── Predictive signal ──────────────────────────────────────────────────

def test_predictive_neutral_when_model_unloaded():
    phase, _ = _make_phase(model_loaded=False)
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5, model_latent=torch.randn(4),
    )
    assert sig.predictive == 1.0


def test_predictive_neutral_when_no_latent():
    phase, _ = _make_phase()
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5, model_latent=None,
    )
    assert sig.predictive == 1.0


def test_predictive_is_one_minus_cosine():
    """When prior == latent (cos=1), predictive surprise = 0.
    When prior is orthogonal (cos=0), predictive = 1."""
    # cos = 1 → predictive = 0
    phase, _ = _make_phase(sdm_cos=1.0)
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5,
        model_latent=torch.randn(4),
    )
    assert sig.predictive == pytest.approx(0.0, abs=1e-5)

    # cos ≈ 0 → predictive ≈ 1
    phase2, _ = _make_phase(sdm_cos=0.0)
    sig2 = phase2.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5,
        model_latent=torch.randn(4),
    )
    assert sig2.predictive == pytest.approx(1.0, abs=1e-5)


def test_predictive_handles_1d_latent():
    """A 1-D latent must be unsqueezed to 2-D before SDM project."""
    phase, m = _make_phase(sdm_cos=0.5)
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5,
        model_latent=torch.randn(4),  # shape (4,) — 1D
    )
    # Must not raise; the SDM project mock should have been called once
    assert m["sdm"].project.call_count == 1


# ── Chroma discount ────────────────────────────────────────────────────

def test_chroma_discount_zero_without_retriever():
    phase, _ = _make_phase(retriever_score=None)
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5, model_latent=None,
    )
    assert sig.chroma_discount == 0.0


def test_chroma_discount_capped_at_095():
    phase, _ = _make_phase(retriever_score=0.99)
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5, model_latent=None,
    )
    assert sig.chroma_discount == 0.95


def test_chroma_discount_clamped_above_zero():
    """Scores can come back negative from cosine — clamp to 0."""
    phase, _ = _make_phase(retriever_score=-0.2)
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5, model_latent=None,
    )
    assert sig.chroma_discount == 0.0


def test_chroma_discount_prefers_rerank_score():
    """When the retriever returns a rerank_score, it takes priority."""
    phase, m = _make_phase()
    retriever = MagicMock()
    retriever.search.return_value = [
        {"score": 0.2, "rerank_score": 0.8},
    ]
    phase.retriever = retriever
    sig = phase.compute(
        text="x", gliner_emb=None,
        structural_entropy=0.5, model_latent=None,
    )
    assert sig.chroma_discount == 0.8


def test_chroma_discount_uses_no_rerank():
    """Surprise hot path must call search with rerank=False."""
    phase, _ = _make_phase(retriever_score=0.3)
    phase.compute(
        text="hello", gliner_emb=None,
        structural_entropy=0.5, model_latent=None,
    )
    call = phase.retriever.search.call_args
    assert call.kwargs.get("rerank") is False
    assert call.kwargs.get("n") == 1


# ── Composite formula ──────────────────────────────────────────────────

def test_composite_global_formula():
    """Spot-check the additive blend × (1 - discount) shape.

    With weights from config, computing exact values is brittle —
    but invariants must hold:
      * composite is in [0, sum(weights)]
      * global is composite × (1 - discount)
    """
    phase, _ = _make_phase(
        mhn_energy=0.2, sdm_cos=0.5, retriever_score=0.4,
    )
    sig = phase.compute(
        text="x", gliner_emb=torch.zeros(8),
        structural_entropy=0.6,
        model_latent=torch.randn(4),
    )
    expected_global = sig.composite * (1.0 - sig.chroma_discount)
    assert sig.global_ == pytest.approx(expected_global, abs=1e-6)


def test_one_signal_failure_does_not_kill_others():
    """MHN fails → episodic neutral, but structural and Chroma survive."""
    phase, m = _make_phase(retriever_score=0.3)
    m["mhn"].energy.side_effect = RuntimeError("boom")
    sig = phase.compute(
        text="x", gliner_emb=torch.zeros(8),
        structural_entropy=0.4, model_latent=None,
    )
    assert sig.episodic == 1.0           # neutral fallback
    assert sig.structural == 0.4          # unaffected
    assert sig.chroma_discount == 0.3     # unaffected


# ── Doubt index — output side ──────────────────────────────────────────

def test_doubt_empty_entropies_returns_fact():
    doubt, label = SurprisePhase.doubt_from_entropies([], threshold=0.5)
    assert doubt == 0.0
    assert label == EPISTEMIC_FACT


def test_doubt_low_entropies_label_fact():
    """Mean entropy 0.1, threshold 0.5 → doubt=0.1/0.5=0.2 < 0.3 → fact."""
    doubt, label = SurprisePhase.doubt_from_entropies(
        [0.1, 0.1, 0.1], threshold=0.5,
    )
    assert doubt < DOUBT_FACT_MAX
    assert label == EPISTEMIC_FACT


def test_doubt_mid_entropies_label_intuition():
    """Mean 0.3 / threshold 0.5 → 0.6 → intuition."""
    doubt, label = SurprisePhase.doubt_from_entropies(
        [0.3, 0.3, 0.3], threshold=0.5,
    )
    assert DOUBT_FACT_MAX <= doubt < DOUBT_INTUITION_MAX
    assert label == EPISTEMIC_INTUITION


def test_doubt_high_entropies_label_hypothesis():
    """Mean 0.5 / threshold 0.5 → 1.0 → hypothesis."""
    doubt, label = SurprisePhase.doubt_from_entropies(
        [0.5, 0.5, 0.5], threshold=0.5,
    )
    assert doubt >= DOUBT_INTUITION_MAX
    assert label == EPISTEMIC_HYPOTHESIS


def test_doubt_threshold_floor_protects_division():
    """A near-zero threshold must not blow up the divisor."""
    doubt, _ = SurprisePhase.doubt_from_entropies(
        [0.5], threshold=0.0,
    )
    # With threshold floor of 0.1, divisor = max(1*0.1, 0.01) = 0.1
    # → doubt = 0.5 / 0.1 = 5.0 (very high but finite)
    assert doubt == pytest.approx(5.0)


def test_doubt_legacy_formula_reproduction():
    """Tripwire test: the *exact* legacy numerical formula.

    The original code was:
        doubt = sum(ents) / max(len(ents)*max(thr, 0.1), 0.01)
    Reproducing it byte-identically here so any drift in
    SurprisePhase.doubt_from_entropies will be detected.
    """
    entropies = [0.2, 0.4, 0.1, 0.6]
    threshold = 0.7
    legacy = sum(entropies) / max(
        len(entropies) * max(threshold, 0.1), 0.01,
    )
    new, _ = SurprisePhase.doubt_from_entropies(entropies, threshold)
    assert new == pytest.approx(legacy)
