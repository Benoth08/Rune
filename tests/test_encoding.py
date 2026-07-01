"""Unit tests for :class:`lythea.cognition.encoding.EncodingPhase`.

These tests use mocks for the model, entity extractor, and salience
filter so they run without torch / GLiNER installed. The phase is
pure orchestration logic and must be testable in isolation.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# torch is needed only for the tests that exercise the latent path.
# We skip those if it isn't installed, but the rest still run.
torch = pytest.importorskip("torch", reason="latent paths need tensor I/O")

from rune.cognition.encoding import (  # noqa: E402
    ENTITY_NOISE,
    EncodingPhase,
    EncodingResult,
)


def _passing_salience() -> MagicMock:
    s = MagicMock()
    s.evaluate.return_value = SimpleNamespace(passed=True)
    return s


def _failing_salience() -> MagicMock:
    s = MagicMock()
    s.evaluate.return_value = SimpleNamespace(passed=False)
    return s


def _model_loaded(hidden_dim: int = 16, n_tokens: int = 4) -> MagicMock:
    m = MagicMock()
    m.is_loaded = True
    m.hidden_dim = hidden_dim
    m.model_id = "test-model"
    m.analyze_input.return_value = {
        "token_entropies": [0.1, 0.5, 0.3, 0.2],
        "latent_states": torch.randn(n_tokens, hidden_dim),
        "mean_entropy": 0.275,
        "vocab_size": 1000,
    }
    return m


def _model_unloaded() -> MagicMock:
    m = MagicMock()
    m.is_loaded = False
    return m


# ── Salience gate ──────────────────────────────────────────────────────

def test_non_salient_short_circuits():
    """When salience fails, the phase must return an empty result and
    must NOT call analyze_input or extract — it's a noise gate."""
    extractor = MagicMock()
    extractor.encode.return_value = torch.zeros(8)
    model = _model_loaded()

    phase = EncodingPhase(
        model=model,
        entity_extractor=extractor,
        salience=_failing_salience(),
    )
    result = phase.encode("ok")

    assert isinstance(result, EncodingResult)
    assert result.salient is False
    assert result.latents is None
    assert result.token_entropies is None
    assert result.mean_latent is None
    assert result.raw_entities == []
    # GLiNER encode is called BEFORE salience (cf. docstring) so this
    # is fine. But analyze_input and extract must be skipped.
    model.analyze_input.assert_not_called()
    extractor.extract.assert_not_called()


def test_has_images_bypasses_salience_filter():
    """An image turn is always saillant, even if the text stub alone
    would be filtered out by the salience cascade.

    Without this whitelist, "Décris cette image" lands in the noise
    bucket and the resulting exchange (image + Taëlys's description)
    is never archived to Chroma — losing what is actually a content-
    rich turn. With ``has_images=True`` we override that verdict.
    """
    extractor = MagicMock()
    extractor.encode.return_value = torch.zeros(8)
    extractor.extract.return_value = []
    model = _model_loaded()
    salience = _failing_salience()

    phase = EncodingPhase(
        model=model,
        entity_extractor=extractor,
        salience=salience,
    )
    # Salience would normally reject "ok" as noise...
    rejected = phase.encode("ok")
    assert rejected.salient is False

    # ...but with has_images=True, salience is bypassed entirely.
    accepted = phase.encode("ok", has_images=True)
    assert accepted.salient is True
    # The full pipeline ran — analyze_input was reached.
    model.analyze_input.assert_called()


def test_has_images_does_not_call_salience_evaluate():
    """When ``has_images=True`` the cascade is short-circuited: we
    don't even run :meth:`salience.evaluate`. This matters because
    evaluate has side effects (it writes to the dedup memory)."""
    extractor = MagicMock()
    extractor.encode.return_value = torch.zeros(8)
    extractor.extract.return_value = []
    model = _model_loaded()
    salience = _passing_salience()  # would accept normally

    phase = EncodingPhase(
        model=model,
        entity_extractor=extractor,
        salience=salience,
    )
    phase.encode("anything", has_images=True)

    salience.evaluate.assert_not_called()


# ── Latent encoding ────────────────────────────────────────────────────

def test_salient_returns_full_encoding():
    extractor = MagicMock()
    extractor.encode.return_value = torch.zeros(8)
    extractor.extract.return_value = []
    model = _model_loaded(hidden_dim=16, n_tokens=4)

    phase = EncodingPhase(
        model=model,
        entity_extractor=extractor,
        salience=_passing_salience(),
    )
    result = phase.encode("Mika travaille à Aix-en-Provence")

    assert result.salient is True
    assert result.structural_entropy == pytest.approx(0.275)
    assert result.latents is not None and result.latents.shape == (4, 16)
    assert result.token_entropies == [0.1, 0.5, 0.3, 0.2]
    assert result.mean_latent is not None and result.mean_latent.shape == (16,)
    assert result.gliner_emb is not None


def test_model_unloaded_falls_back_to_default_entropy():
    """When the model is not loaded, structural entropy defaults to
    0.5 and latents are None — same contract as the original
    ``_phase_a_learn``."""
    extractor = MagicMock()
    extractor.encode.return_value = torch.zeros(8)
    extractor.extract.return_value = []

    phase = EncodingPhase(
        model=_model_unloaded(),
        entity_extractor=extractor,
        salience=_passing_salience(),
    )
    result = phase.encode("question intéressante")

    assert result.salient is True
    assert result.structural_entropy == 0.5
    assert result.latents is None
    assert result.token_entropies is None
    assert result.mean_latent is None


def test_analyze_input_failure_falls_back_silently():
    """If analyze_input raises (OOM, tokenizer mismatch...), the
    phase logs and returns the same fallback as if the model wasn't
    loaded. Salience must still pass and entities still extract."""
    extractor = MagicMock()
    extractor.encode.return_value = torch.zeros(8)
    extractor.extract.return_value = [
        {"text": "Aix", "label": "location", "score": 0.9},
    ]
    model = MagicMock()
    model.is_loaded = True
    model.analyze_input.side_effect = RuntimeError("simulated OOM")

    phase = EncodingPhase(
        model=model,
        entity_extractor=extractor,
        salience=_passing_salience(),
    )
    result = phase.encode("test")

    assert result.salient is True
    assert result.structural_entropy == 0.5
    assert result.latents is None
    # Entity extraction must still have happened.
    assert len(result.raw_entities) == 1


# ── Entity filtering ───────────────────────────────────────────────────

def test_entity_noise_filtered():
    """Pronouns, articles, and generic nouns must be dropped before
    they ever leave the encoding phase."""
    extractor = MagicMock()
    extractor.encode.return_value = torch.zeros(8)
    extractor.extract.return_value = [
        {"text": "Je", "label": "person", "score": 0.99},          # noise
        {"text": " ça ", "label": "thing", "score": 0.5},          # noise (stripped)
        {"text": "PROJET", "label": "project", "score": 0.7},      # noise (lower)
        {"text": "Mika", "label": "person", "score": 0.95},        # keep
        {"text": "Aix-en-Provence", "label": "location", "score": 0.9},  # keep
    ]
    phase = EncodingPhase(
        model=_model_unloaded(),
        entity_extractor=extractor,
        salience=_passing_salience(),
    )
    result = phase.encode("any")

    kept = [e["text"] for e in result.raw_entities]
    assert kept == ["Mika", "Aix-en-Provence"]


def test_short_entity_filtered():
    """Single-character hits (often artefacts) must be dropped."""
    extractor = MagicMock()
    extractor.encode.return_value = torch.zeros(8)
    extractor.extract.return_value = [
        {"text": "X", "label": "person", "score": 0.5},
        {"text": "  ", "label": "thing", "score": 0.5},
        {"text": "ai", "label": "skill", "score": 0.5},  # 2 chars → keep
    ]
    phase = EncodingPhase(
        model=_model_unloaded(),
        entity_extractor=extractor,
        salience=_passing_salience(),
    )
    result = phase.encode("any")
    assert [e["text"] for e in result.raw_entities] == ["ai"]


def test_no_entity_extractor_returns_empty():
    """When no GLiNER is wired, the phase must still run (salience
    will get None as the embedding) and return zero entities."""
    phase = EncodingPhase(
        model=_model_unloaded(),
        entity_extractor=None,
        salience=_passing_salience(),
    )
    result = phase.encode("any")
    assert result.raw_entities == []
    assert result.gliner_emb is None


# ── Stop list contract ─────────────────────────────────────────────────

def test_entity_noise_contract():
    """The noise set is part of the public contract — duplicate it
    here as a tripwire so any change is forced to update tests."""
    must_be_noise = {"je", "tu", "ça", "projet", "i", "you", "they"}
    assert must_be_noise.issubset(ENTITY_NOISE)
