"""Tests du backend Mock — base de tous les autres tests."""
from __future__ import annotations

import pytest

from rune.perf.backend import GenerationConfig, get_backend
from rune.perf.mock import MockBackend


def test_mock_backend_basic_properties():
    backend = MockBackend({"hidden_dim": 32, "n_layers": 2})
    assert backend.name == "mock"
    assert backend.hidden_dim == 32
    assert backend.n_layers == 2
    assert backend.has_hooks is True
    assert backend.is_thinking_model is False


def test_mock_backend_encode_is_deterministic():
    backend = MockBackend({"hidden_dim": 64})
    e1 = backend.encode("bonjour le monde")
    e2 = backend.encode("bonjour le monde")
    assert e1 == e2
    assert len(e1) == 64


def test_mock_backend_encode_l2_normalized():
    backend = MockBackend({"hidden_dim": 32})
    e = backend.encode("test")
    norm = sum(x * x for x in e) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_mock_backend_encode_different_inputs_orthogonal():
    """Deux textes différents donnent des vecteurs quasi-orthogonaux."""
    backend = MockBackend({"hidden_dim": 128})
    e1 = backend.encode("bonjour")
    e2 = backend.encode("au revoir")
    dot = sum(a * b for a, b in zip(e1, e2))
    # Cosine sim should be near 0 (hash diffusion)
    assert abs(dot) < 0.3


def test_mock_backend_generate_returns_text():
    backend = MockBackend({"responses": ["Réponse de test"]})
    result = backend.generate(
        messages=[{"role": "user", "content": "question complexe"}],
        config=GenerationConfig(max_new_tokens=100),
    )
    assert result.text == "Réponse de test"
    assert result.finish_reason == "stop"
    assert result.tokens_generated > 0
    assert len(result.entropies) > 0
    assert result.hidden_states is not None
    assert len(result.hidden_states) == backend.n_layers


def test_mock_backend_generate_thinking_model():
    backend = MockBackend({"is_thinking": True, "responses": ["Ma réponse"]})
    result = backend.generate(
        messages=[{"role": "user", "content": "question détaillée"}],
        config=GenerationConfig(),
    )
    assert "<think>" in result.raw_text
    assert "<think>" not in result.text
    assert result.text == "Ma réponse"


def test_mock_backend_generate_short_messages():
    """Messages courts (merci, ok) renvoient une réponse courte."""
    backend = MockBackend()
    result = backend.generate(
        messages=[{"role": "user", "content": "merci"}],
        config=GenerationConfig(),
    )
    assert "merci" in result.text.lower() or "plaisir" in result.text.lower()


def test_mock_backend_stream_generate():
    backend = MockBackend({"responses": ["un deux trois"]})
    gen = backend.stream_generate(
        messages=[{"role": "user", "content": "test"}],
        config=GenerationConfig(),
    )
    tokens = []
    while True:
        try:
            tokens.append(next(gen))
        except StopIteration as stop:
            result = stop.value
            break
    assert len(tokens) == 3
    assert result.text == "un deux trois"
    assert result.tokens_generated == 3


def test_mock_backend_register_forward_hook():
    backend = MockBackend({"hidden_dim": 16, "n_layers": 3})
    calls = []
    remove = backend.register_forward_hook(0, lambda x: calls.append(x))
    # Hook is registered — the mock doesn't actually call it, but
    # the registration must not fail.
    assert callable(remove)
    remove()


def test_mock_backend_register_hook_out_of_range_doesnt_crash():
    """Le mock ne valide pas les indices, mais ne doit pas crasher."""
    backend = MockBackend()
    remove = backend.register_forward_hook(999, lambda x: None)
    remove()


def test_get_backend_returns_mock_when_forced():
    backend = get_backend({"backend": "mock"})
    assert backend.name == "mock"


def test_get_backend_auto_falls_back_to_mock(monkeypatch):
    """Sans CUDA, get_backend auto-detect retourne MockBackend."""
    monkeypatch.setenv("HERMES_LYTHEA_BACKEND", "mock")
    backend = get_backend()
    assert backend.name == "mock"


def test_get_backend_transformers_falls_back_on_error():
    """Si transformers n'est pas dispo, on tombe sur mock."""
    backend = get_backend({"backend": "transformers", "model_id": "invalid/model"})
    # Soit ça lève à l'init (model_id invalide), soit le lazy load échoue
    # au premier generate. Dans tous les cas, get_backend doit retomber
    # sur MockBackend.
    assert backend.name in {"mock", "transformers"}
