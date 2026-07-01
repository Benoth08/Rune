"""Anthropic Claude API client — pour la cascade extérieure optionnelle.

Architecture identique à GeminiClient (cf. external/gemini_client.py) :
- Pas de vendor SDK, on utilise httpx directement
- Quota counter local (rate limiting côté client)
- Secrets jamais leakés (mask_api_key)
- Retry avec backoff exponentiel
- Interface sync (la cascade est sync)
- Désactivé par défaut — activé via RUNE_ENABLE_CASCADE=true + ANTHROPIC_API_KEY

Usage
-----
    from rune.external.anthropic_client import ClaudeClient

    client = ClaudeClient(api_key="sk-ant-...")
    response = client.generate(
        system_prompt="Tu es Rune...",
        messages=[{"role": "user", "content": "Bonjour"}],
    )
    print(response.text)

Différences vs GeminiClient
---------------------------
- Endpoint : https://api.anthropic.com/v1/messages
- Auth : header `x-api-key` + `anthropic-version: 2023-06-01`
- Format messages : {"role": "user"|"assistant", "content": str}
  (même format que OpenAI/Gemini — pratique)
- System prompt : champ séparé `system` (pas dans messages)
- Modèles : claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022,
  claude-3-opus-20240229, etc.

Modèles recommandés (juin 2026)
-------------------------------
- Claude Sonnet 4 : équilibre vitesse/qualité, bon pour le draft
- Claude Haiku 3.5 : rapide et économique, bon pour les drafts courts
- Claude Opus 4 : le plus capable, pour les drafts complexes
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("rune.external.anthropic")


# ── Constants ────────────────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT_SEC = 60.0
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5  # secondes

# Quota local (best-effort — Anthropic n'expose pas de free tier fixe)
DEFAULT_DAILY_LIMIT = 1000  # conservateur
DEFAULT_PER_MINUTE_LIMIT = 50


# ── Exceptions ───────────────────────────────────────────────────────


class ClaudeClientError(Exception):
    """Base exception."""


class ClaudeUnauthorizedError(ClaudeClientError):
    """API key rejected (401/403)."""


class ClaudeQuotaExceededError(ClaudeClientError):
    """Quota exhausted (429)."""


class ClaudeTransientError(ClaudeClientError):
    """Network or 5xx — retry."""


class ClaudeResponseError(ClaudeClientError):
    """Malformed response."""


# ── Helpers ──────────────────────────────────────────────────────────


def mask_api_key(key: str | None) -> str:
    """Masque une clé API pour affichage (ex: 'sk-ant-...3xYz')."""
    if not key:
        return "(not set)"
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def detect_api_key(key: str | None) -> str | None:
    """Détecte si une clé est au format Anthropic.

    Anthropic keys commencent par 'sk-ant-'.
    """
    if not key:
        return None
    if key.startswith("sk-ant-"):
        return key
    return None


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class ClaudeResponse:
    """Réponse de l'API Anthropic."""
    text: str = ""
    model: str = ""
    finish_reason: str = "stop"  # stop | length | tool_use | error
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_sec: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


# ── Quota trackers ───────────────────────────────────────────────────


class _DailyQuota:
    """Compteur de quota quotidien (best-effort, reset à minuit local)."""

    def __init__(self, daily_limit: int = DEFAULT_DAILY_LIMIT) -> None:
        self.daily_limit = daily_limit
        self._count = 0
        self._reset_day = time.localtime().tm_yday

    def check_and_increment(self) -> bool:
        """Retourne True si on peut faire un appel, False sinon."""
        today = time.localtime().tm_yday
        if today != self._reset_day:
            self._count = 0
            self._reset_day = today
        if self._count >= self.daily_limit:
            return False
        self._count += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self._count)

    @property
    def used(self) -> int:
        return self._count


class _RateLimiter:
    """Rate limiter par minute (best-effort)."""

    def __init__(self, per_minute: int = DEFAULT_PER_MINUTE_LIMIT) -> None:
        self.per_minute = per_minute
        self._timestamps: list[float] = []

    def check(self) -> bool:
        """Retourne True si on peut faire un appel maintenant."""
        now = time.time()
        # Nettoie les timestamps > 60s
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.per_minute:
            return False
        self._timestamps.append(now)
        return True


# ── ClaudeClient ─────────────────────────────────────────────────────


class ClaudeClient:
    """Client Anthropic Claude API.

    Parameters
    ----------
    api_key : str | None
        Clé API Anthropic (sk-ant-...). Si None, lit ANTHROPIC_API_KEY.
    model : str
        Modèle par défaut. Default: claude-sonnet-4-20250514.
    daily_limit : int
        Quota local quotidien (best-effort).
    per_minute_limit : int
        Rate limit par minute (best-effort).
    timeout_sec : float
        Timeout HTTP par requête.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        daily_limit: int = DEFAULT_DAILY_LIMIT,
        per_minute_limit: int = DEFAULT_PER_MINUTE_LIMIT,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.timeout_sec = timeout_sec

        if not self.api_key:
            log.warning(
                "ClaudeClient: no API key set. Set ANTHROPIC_API_KEY env var."
            )

        self._daily_quota = _DailyQuota(daily_limit)
        self._rate_limiter = _RateLimiter(per_minute_limit)
        self._client: httpx.Client | None = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_sec)
        return self._client

    def is_configured(self) -> bool:
        """True si la clé API est set."""
        return bool(self.api_key and detect_api_key(self.api_key))

    def quota_status(self) -> dict[str, Any]:
        """Snapshot du quota local."""
        return {
            "daily_used": self._daily_quota.used,
            "daily_remaining": self._daily_quota.remaining,
            "daily_limit": self._daily_quota.daily_limit,
            "per_minute_limit": self._rate_limiter.per_minute,
        }

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        model: str | None = None,
    ) -> ClaudeResponse:
        """Send a chat-style request to Claude and return the response.

        Parameters
        ----------
        system_prompt : str
            System instruction (séparé des messages chez Anthropic).
        messages : list[dict]
            Chat history [{"role": "user"|"assistant", "content": str}].
        max_tokens : int
            Hard cap on generated tokens.
        temperature : float
            Sampling temperature.
        model : str | None
            Override le modèle par défaut.

        Raises
        ------
        ClaudeUnauthorizedError
            API key rejected.
        ClaudeQuotaExceededError
            Quota local ou serveur dépassé.
        ClaudeTransientError
            Network/5xx après retries.
        ClaudeClientError
            Autre erreur.
        """
        if not self.is_configured():
            raise ClaudeUnauthorizedError(
                "ANTHROPIC_API_KEY not set or invalid format"
            )

        if not self._daily_quota.check_and_increment():
            raise ClaudeQuotaExceededError(
                f"Daily local quota exhausted ({self._daily_quota.daily_limit} req/day)"
            )

        if not self._rate_limiter.check():
            raise ClaudeQuotaExceededError(
                f"Per-minute rate limit exceeded ({self._rate_limiter.per_minute} req/min)"
            )

        effective_model = model or self.model
        payload = {
            "model": effective_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": messages,
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

        start = time.time()
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                client = self._ensure_client()
                resp = client.post(
                    ANTHROPIC_API_URL,
                    json=payload,
                    headers=headers,
                )

                if resp.status_code == 200:
                    return self._parse_response(resp, effective_model, start)

                if resp.status_code in (401, 403):
                    raise ClaudeUnauthorizedError(
                        f"API key rejected ({resp.status_code}): {resp.text[:200]}"
                    )
                if resp.status_code == 429:
                    raise ClaudeQuotaExceededError(
                        f"Server-side rate limit (429): {resp.text[:200]}"
                    )
                if 500 <= resp.status_code < 600:
                    # Transient — retry
                    last_exc = ClaudeTransientError(
                        f"Server error {resp.status_code}: {resp.text[:200]}"
                    )
                    log.warning(
                        "Claude transient error (attempt %d/%d): %s",
                        attempt + 1, MAX_RETRIES, last_exc,
                    )
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                    continue

                # Autre 4xx
                raise ClaudeClientError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )

            except httpx.TimeoutException as exc:
                last_exc = ClaudeTransientError(f"Timeout: {exc}")
                log.warning(
                    "Claude timeout (attempt %d/%d): %s",
                    attempt + 1, MAX_RETRIES, exc,
                )
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            except httpx.NetworkError as exc:
                last_exc = ClaudeTransientError(f"Network error: {exc}")
                log.warning(
                    "Claude network error (attempt %d/%d): %s",
                    attempt + 1, MAX_RETRIES, exc,
                )
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue

        # Tous les retries ont échoué
        if last_exc:
            raise last_exc
        raise ClaudeTransientError("All retries failed")

    def test_connection(self) -> bool:
        """Teste la connexion avec un prompt minimal.

        Retourne True si l'API répond correctement. Consomme 1 quota unit.
        """
        try:
            resp = self.generate(
                system_prompt="You are a test assistant. Reply with 'OK'.",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
                temperature=0.0,
            )
            return resp.finish_reason != "error"
        except Exception as exc:
            log.warning("Claude test_connection failed: %s", exc)
            return False

    def close(self) -> None:
        """Ferme le client HTTP."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── Internes ──────────────────────────────────────────────────────

    def _parse_response(
        self,
        resp: httpx.Response,
        model: str,
        start: float,
    ) -> ClaudeResponse:
        """Parse la réponse JSON d'Anthropic."""
        try:
            data = resp.json()
        except Exception as exc:
            raise ClaudeResponseError(
                f"Invalid JSON response: {exc}"
            ) from exc

        # Format Anthropic :
        # {
        #   "content": [{"type": "text", "text": "..."}],
        #   "model": "claude-...",
        #   "stop_reason": "end_turn" | "max_tokens" | ...,
        #   "usage": {"input_tokens": N, "output_tokens": N}
        # }
        content_blocks = data.get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ]
        text = "".join(text_parts)

        stop_reason = data.get("stop_reason", "end_turn")
        if stop_reason == "max_tokens":
            finish = "length"
        elif stop_reason == "end_turn":
            finish = "stop"
        elif stop_reason == "tool_use":
            finish = "tool_use"
        else:
            finish = "stop"

        usage = data.get("usage", {})

        return ClaudeResponse(
            text=text,
            model=data.get("model", model),
            finish_reason=finish,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            elapsed_sec=time.time() - start,
            raw=data,
        )
