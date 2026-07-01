"""Factory pour les clients LLM distants (cascade extérieure).

Permet de switcher entre Gemini et Claude (et futur OpenAI/Mistral)
sans modifier le code cascade. Le provider est choisi via la variable
d'environnement RUNE_CASCADE_PROVIDER (défaut: gemini).

Providers supportés
-------------------
- ``gemini``   : Google Gemini (cf. external/gemini_client.py)
- ``claude``   : Anthropic Claude (cf. external/anthropic_client.py)
- (futur) ``openai``   : OpenAI GPT-4 / GPT-4o
- (futur) ``mistral``  : Mistral La Plateforme

Désactivé par défaut
--------------------
La cascade extérieure est **désactivée par défaut**. Pour activer :

    RUNE_ENABLE_CASCADE=true
    RUNE_CASCADE_PROVIDER=claude   # ou gemini
    ANTHROPIC_API_KEY=sk-ant-...   # ou GOOGLE_API_KEY=AIzaSy...

Usage
-----
    from rune.external.factory import get_cascade_client, CascadeProvider

    client = get_cascade_client()  # retourne GeminiClient ou ClaudeClient
    if client is None:
        # Cascade désactivée — fallback local
        ...
    else:
        response = client.generate(system_prompt, messages)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Protocol

log = logging.getLogger("rune.external.factory")


class CascadeClient(Protocol):
    """Interface commune aux clients cascade (Gemini, Claude, …)."""

    def is_configured(self) -> bool: ...

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model: str | None = None,
    ) -> Any: ...

    def test_connection(self) -> bool: ...


def get_cascade_client() -> CascadeClient | None:
    """Retourne le client cascade configuré, ou None si désactivé.

    Logique
    -------
    1. Si RUNE_ENABLE_CASCADE != true → retourne None
    2. Sinon, lit RUNE_CASCADE_PROVIDER (défaut: gemini)
    3. Instancie le client correspondant
    4. Si le client n'est pas configuré (clé manquante) → log warning + None

    Returns
    -------
    CascadeClient | None
        Le client, ou None si cascade désactivée ou mal configurée.
    """
    enabled = os.environ.get("RUNE_ENABLE_CASCADE", "false").lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
        return None

    provider = os.environ.get("RUNE_CASCADE_PROVIDER", "gemini").lower()

    if provider == "gemini":
        return _build_gemini_client()
    elif provider == "claude":
        return _build_claude_client()
    elif provider == "openai":
        log.warning(
            "OpenAI cascade not yet implemented — falling back to None. "
            "Set RUNE_CASCADE_PROVIDER=gemini or claude."
        )
        return None
    elif provider == "mistral":
        log.warning(
            "Mistral cascade not yet implemented — falling back to None. "
            "Set RUNE_CASCADE_PROVIDER=gemini or claude."
        )
        return None
    else:
        log.warning(
            "Unknown RUNE_CASCADE_PROVIDER=%r — supported: gemini, claude. "
            "Cascade disabled.",
            provider,
        )
        return None


def _build_gemini_client() -> CascadeClient | None:
    """Construit un GeminiClient si GOOGLE_API_KEY est set."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
        "LYTHEA_CASCADE_GEMINI_KEY"
    )
    if not api_key:
        log.warning(
            "Cascade provider=gemini but GOOGLE_API_KEY not set. "
            "Cascade disabled."
        )
        return None

    try:
        from .gemini_client import GeminiClient
        model = os.environ.get("LYTHEA_CASCADE_GEMINI_MODEL", "gemini-3.5-flash")
        client = GeminiClient(api_key=api_key, model=model)
        log.info(
            "Cascade enabled: Gemini (model=%s, key=%s)",
            model, _mask_key(api_key),
        )
        return client  # type: ignore[return-value]
    except Exception as exc:
        log.warning("Failed to build GeminiClient: %s — cascade disabled", exc)
        return None


def _build_claude_client() -> CascadeClient | None:
    """Construit un ClaudeClient si ANTHROPIC_API_KEY est set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning(
            "Cascade provider=claude but ANTHROPIC_API_KEY not set. "
            "Cascade disabled."
        )
        return None

    try:
        from .anthropic_client import ClaudeClient, detect_api_key
        if not detect_api_key(api_key):
            log.warning(
                "ANTHROPIC_API_KEY has invalid format (expected 'sk-ant-...'). "
                "Cascade disabled."
            )
            return None

        model = os.environ.get(
            "RUNE_CASCADE_CLAUDE_MODEL",
            "claude-sonnet-4-20250514",
        )
        client = ClaudeClient(api_key=api_key, model=model)
        log.info(
            "Cascade enabled: Claude (model=%s, key=%s)",
            model, _mask_key(api_key),
        )
        return client  # type: ignore[return-value]
    except Exception as exc:
        log.warning("Failed to build ClaudeClient: %s — cascade disabled", exc)
        return None


def _mask_key(key: str) -> str:
    """Masque une clé API pour log."""
    if not key or len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def cascade_status() -> dict[str, Any]:
    """Snapshot de l'état cascade pour /status.

    N'inclut jamais la clé API en clair.
    """
    enabled = os.environ.get("RUNE_ENABLE_CASCADE", "false").lower() in {
        "1", "true", "yes", "on",
    }
    provider = os.environ.get("RUNE_CASCADE_PROVIDER", "gemini") if enabled else None

    client = get_cascade_client()
    # is_configured() peut ne pas exister selon le client (GeminiClient
    # n'a pas cette méthode). On fallback sur "client is not None".
    if client is not None and hasattr(client, "is_configured"):
        try:
            configured = bool(client.is_configured())
        except Exception:
            configured = True
    else:
        configured = client is not None

    # Récupère le quota si dispo
    quota: dict[str, Any] = {}
    if client is not None and hasattr(client, "quota_status"):
        try:
            quota = client.quota_status()  # type: ignore[attr-defined]
        except Exception:
            pass

    return {
        "enabled": enabled and configured,
        "provider": provider,
        "configured": configured,
        "model": _get_cascade_model(provider),
        "quota": quota,
    }


def _get_cascade_model(provider: str | None) -> str:
    """Retourne le modèle configuré pour le provider."""
    if provider == "gemini":
        return os.environ.get("LYTHEA_CASCADE_GEMINI_MODEL", "gemini-3.5-flash")
    if provider == "claude":
        return os.environ.get(
            "RUNE_CASCADE_CLAUDE_MODEL", "claude-sonnet-4-20250514"
        )
    return "unknown"
