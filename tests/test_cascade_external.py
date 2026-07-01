"""Tests cascade extérieure — Gemini + Claude + factory."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from rune.external.anthropic_client import (
    ClaudeClient,
    ClaudeClientError,
    ClaudeQuotaExceededError,
    ClaudeUnauthorizedError,
    detect_api_key,
    mask_api_key,
)
from rune.external.factory import (
    cascade_status,
    get_cascade_client,
)


# ── Anthropic client — helpers ────────────────────────────────────────


def test_mask_api_key_full():
    assert mask_api_key("sk-ant-abc123def456") == "sk-ant-a...f456"


def test_mask_api_key_short():
    assert mask_api_key("short") == "***"


def test_mask_api_key_none():
    assert mask_api_key(None) == "(not set)"


def test_detect_api_key_valid():
    assert detect_api_key("sk-ant-something") == "sk-ant-something"


def test_detect_api_key_invalid():
    assert detect_api_key("AIzaSy-test") is None
    assert detect_api_key("") is None
    assert detect_api_key(None) is None


# ── Anthropic client — init ───────────────────────────────────────────


def test_claude_client_init_with_key():
    client = ClaudeClient(api_key="sk-ant-test123")
    assert client.is_configured() is True
    assert client.api_key == "sk-ant-test123"


def test_claude_client_init_without_key():
    client = ClaudeClient(api_key=None)
    assert client.is_configured() is False


def test_claude_client_init_with_invalid_key():
    """Une clé qui ne commence pas par 'sk-ant-' n'est pas configurée."""
    client = ClaudeClient(api_key="invalid-key")
    assert client.is_configured() is False


def test_claude_client_default_model():
    client = ClaudeClient(api_key="sk-ant-test")
    assert "claude" in client.model


def test_claude_client_custom_model():
    client = ClaudeClient(
        api_key="sk-ant-test",
        model="claude-3-opus-20240229",
    )
    assert client.model == "claude-3-opus-20240229"


# ── Anthropic client — quota ──────────────────────────────────────────


def test_claude_client_quota_status():
    client = ClaudeClient(api_key="sk-ant-test", daily_limit=100)
    status = client.quota_status()
    assert status["daily_limit"] == 100
    assert status["daily_used"] == 0
    assert status["daily_remaining"] == 100


def test_claude_client_quota_decrements():
    """Le quota local décrémente à chaque check_and_increment."""
    client = ClaudeClient(api_key="sk-ant-test", daily_limit=2)
    assert client._daily_quota.check_and_increment() is True
    assert client._daily_quota.check_and_increment() is True
    # 3e appel doit échouer
    assert client._daily_quota.check_and_increment() is False


# ── Anthropic client — generate (mocked) ──────────────────────────────


def test_claude_client_generate_not_configured_raises():
    """Sans clé API, generate lève ClaudeUnauthorizedError."""
    client = ClaudeClient(api_key=None)
    with pytest.raises(ClaudeUnauthorizedError):
        client.generate(
            system_prompt="test",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_claude_client_generate_quota_exceeded():
    """Quota local dépassé → ClaudeQuotaExceededError."""
    client = ClaudeClient(api_key="sk-ant-test", daily_limit=1)
    # Force quota exhausted
    client._daily_quota.check_and_increment()  # utilise le seul appel
    with pytest.raises(ClaudeQuotaExceededError):
        client.generate(
            system_prompt="test",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_claude_client_test_connection_failure():
    """test_connection retourne False si l'API n'est pas joignable."""
    client = ClaudeClient(api_key="sk-ant-test")
    # Mock generate pour simuler un échec
    client.generate = MagicMock(side_effect=ClaudeClientError("network error"))
    assert client.test_connection() is False


# ── Factory ───────────────────────────────────────────────────────────


def test_factory_disabled_by_default(monkeypatch):
    """Sans RUNE_ENABLE_CASCADE, get_cascade_client retourne None."""
    monkeypatch.delenv("RUNE_ENABLE_CASCADE", raising=False)
    assert get_cascade_client() is None


def test_factory_enabled_but_no_provider_key(monkeypatch):
    """Si activé mais pas de clé API, retourne None avec warning."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "claude")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_cascade_client() is None


def test_factory_claude_with_key(monkeypatch):
    """Si activé + ANTHROPIC_API_KEY set → retourne ClaudeClient."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123456789")
    client = get_cascade_client()
    assert client is not None
    assert client.is_configured() is True


def test_factory_gemini_with_key(monkeypatch):
    """Si activé + GOOGLE_API_KEY set → retourne GeminiClient."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "gemini")
    # Clé Gemini au bon format (AIzaSy + 33 chars)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy" + "a" * 33)
    client = get_cascade_client()
    # GeminiClient peut être retourné (selon imports disponibles)
    assert client is not None


def test_factory_unknown_provider(monkeypatch):
    """Provider inconnu → retourne None avec warning."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "unknown_provider")
    assert get_cascade_client() is None


def test_factory_invalid_claude_key_format(monkeypatch):
    """Clé Anthropic avec mauvais format → retourne None."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid-key-format")
    assert get_cascade_client() is None


def test_factory_openai_not_implemented(monkeypatch):
    """OpenAI provider pas encore implémenté → None avec warning."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "openai")
    assert get_cascade_client() is None


# ── cascade_status ────────────────────────────────────────────────────


def test_cascade_status_disabled(monkeypatch):
    """Status quand cascade désactivée."""
    monkeypatch.delenv("RUNE_ENABLE_CASCADE", raising=False)
    status = cascade_status()
    assert status["enabled"] is False
    assert status["provider"] is None
    assert status["configured"] is False


def test_cascade_status_enabled_claude(monkeypatch):
    """Status quand cascade activée avec Claude."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123456789")
    status = cascade_status()
    assert status["enabled"] is True
    assert status["provider"] == "claude"
    assert status["configured"] is True
    assert "claude" in status["model"]
    # La clé ne doit jamais être dans le status
    assert "sk-ant" not in str(status)


def test_cascade_status_enabled_gemini(monkeypatch):
    """Status quand cascade activée avec Gemini."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy" + "a" * 33)
    status = cascade_status()
    assert status["enabled"] is True
    assert status["provider"] == "gemini"
    assert status["configured"] is True


def test_cascade_status_no_api_key_leak(monkeypatch):
    """Le status ne doit jamais contenir la clé API en clair."""
    monkeypatch.setenv("RUNE_ENABLE_CASCADE", "true")
    monkeypatch.setenv("RUNE_CASCADE_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret-key-123456")
    status = cascade_status()
    status_str = str(status)
    assert "very-secret" not in status_str
    assert "sk-ant-very" not in status_str


# ── Intégration avec RuneCortex ───────────────────────────────────────


def test_cascade_client_protocol_compatible():
    """ClaudeClient et GeminiClient respectent le Protocol CascadeClient."""
    from rune.external.anthropic_client import ClaudeClient

    claude = ClaudeClient(api_key="sk-ant-test123456789")
    assert hasattr(claude, "is_configured")
    assert hasattr(claude, "generate")
    assert hasattr(claude, "test_connection")
    # Protocol OK — pas besoin d'instance GeminiClient réelle ici
