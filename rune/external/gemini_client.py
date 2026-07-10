"""Google Gemini API client (V3.9 cascade).

A small, dependency-light wrapper around the Gemini ``generateContent``
REST endpoint, designed for the draft-then-refine cascade in
:mod:`rune.cognition.cascade`.

Design constraints
------------------

1. **No vendor SDK.** We use ``httpx`` directly to keep the dependency
   footprint small and avoid the larger ``google-generativeai`` package
   which pulls in proto / grpc machinery we don't need.

2. **Free-tier friendly.** A local quota counter tracks daily usage so
   the UI can warn before hitting the 1500 req/day limit. The counter
   resets at local midnight (best-effort — it's a hint, not a contract).

3. **Secrets never leak.** :func:`mask_api_key` is the single entry
   point for any code that wants to show the key for debug/logging.

4. **Network failures are recoverable.** Three retry attempts with
   exponential backoff. After that, the cascade falls back to the
   local model (handled in :mod:`rune.cognition.cascade`).

5. **Synchronous interface.** Lythéa's cognition pipeline is sync.
   Async would force colored functions everywhere for marginal gain
   on a single-user workload.

Usage
-----

>>> client = GeminiClient(api_key="AIzaSy...", model="gemini-3.5-flash")
>>> response = client.generate(
...     system_prompt="You are Rune...",
...     messages=[{"role": "user", "content": "Bonjour"}],
...     max_tokens=500,
... )
>>> print(response.text)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 0.5  # seconds, doubles each retry

# Free tier daily quota (Flash family — gemini-3.5-flash et al. — as of
# 2026-06, AI Studio, no card). Soft warning threshold; Google enforces
# the real limit.
FREE_TIER_DAILY_QUOTA = 1500

# Free tier per-minute rate limit. Empirically observed (V3.9.4
# prod test 2026-05-04): Gemini 2.5 Flash returns 429 after ~10
# successive requests in 60 seconds. We throttle one below that to
# leave headroom for retries and never trigger the server-side 429.
FREE_TIER_PER_MINUTE_QUOTA = 8

# Valid Gemini API key format. Google keys start with "AIzaSy" and are
# 39 characters long. We do a syntactic check at boot so a typo in
# .env produces a clear error rather than a 401 at first call.
_API_KEY_PATTERN = re.compile(r"^AIzaSy[A-Za-z0-9_\-]{33}$")


# ── Exceptions ─────────────────────────────────────────────────────────


class GeminiClientError(Exception):
    """Base class for Gemini API errors raised by this client.

    Catch this to fall back gracefully to local generation. Subclasses
    distinguish between recoverable conditions (rate limit, transient
    network) and permanent ones (bad key, model not found).
    """


class GeminiUnauthorizedError(GeminiClientError):
    """API key rejected. Permanent — don't retry, ask the user."""


class GeminiQuotaExceededError(GeminiClientError):
    """Daily quota exhausted. Local-only mode until midnight."""


class GeminiTransientError(GeminiClientError):
    """Network glitch or 5xx. Retry-able."""


# ── Helpers ────────────────────────────────────────────────────────────


def mask_api_key(key: str | None) -> str:
    """Return a safe representation of an API key for logs and UI.

    Only the last 4 characters are shown. ``None`` and short strings
    return ``"<missing>"`` and ``"<invalid>"`` respectively.

    >>> mask_api_key("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567")
    '...4567'
    >>> mask_api_key(None)
    '<missing>'
    """
    if key is None:
        return "<missing>"
    if not isinstance(key, str) or len(key) < 8:
        return "<invalid>"
    return f"...{key[-4:]}"


def validate_api_key_format(key: str | None) -> bool:
    """Cheap syntactic check on a Google API key.

    Returns ``True`` for keys matching ``AIzaSy[35-char alnum/_/-]``,
    ``False`` for anything else (including ``None``).

    Note: this does NOT verify the key is active or has billing enabled
    — only the format. A live check requires :meth:`GeminiClient.test`.
    """
    if not isinstance(key, str):
        return False
    return bool(_API_KEY_PATTERN.match(key))


# ── Quota tracker ──────────────────────────────────────────────────────


class _DailyQuotaTracker:
    """Local counter for free-tier requests.

    This is a best-effort hint: Google enforces the real quota
    server-side. Use this to display a warning in the UI before the
    actual 429.

    Thread-safe so streaming and background calls don't clobber each
    other's increments.
    """

    def __init__(self, daily_limit: int = FREE_TIER_DAILY_QUOTA) -> None:
        self._limit = int(daily_limit)
        self._lock = threading.Lock()
        self._date = date.today()
        self._count = 0

    def _maybe_reset(self) -> None:
        # Caller must hold the lock.
        today = date.today()
        if today != self._date:
            self._date = today
            self._count = 0

    def increment(self) -> int:
        with self._lock:
            self._maybe_reset()
            self._count += 1
            return self._count

    def used(self) -> int:
        with self._lock:
            self._maybe_reset()
            return self._count

    def remaining(self) -> int:
        return max(0, self._limit - self.used())

    def is_exhausted(self) -> bool:
        return self.used() >= self._limit

    def reset(self) -> None:
        # Test helper.
        with self._lock:
            self._date = date.today()
            self._count = 0


class _PerMinuteRateLimiter:
    """Sliding-window rate limiter for short-term Gemini quota.

    Added in V3.9.4 after empirical observation: Gemini 2.5 Flash
    returns HTTP 429 ("quota exceeded for metric:
    generativelanguage.googleapis.com/generate_content_free_tier_requests")
    after ~10 requests within 60 seconds, even when daily quota is
    barely used. The daily counter alone doesn't catch this.

    We track timestamps of successful requests in a deque, prune
    anything older than 60 seconds, and refuse new requests when the
    window is full. The caller catches :class:`GeminiQuotaExceededError`
    and falls back to the local model.

    This is a *client-side* throttle. Google's actual quota is still
    enforced server-side; we just avoid hitting it.
    """

    def __init__(self, per_minute_limit: int = FREE_TIER_PER_MINUTE_QUOTA) -> None:
        self._limit = int(per_minute_limit)
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def _prune(self, now: float) -> None:
        # Caller must hold the lock.
        cutoff = now - 60.0
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def check_and_record(self) -> tuple[bool, int]:
        """Atomically check if a slot is available and reserve it.

        Returns ``(allowed, current_count)``. When ``allowed`` is False,
        the caller should raise :class:`GeminiQuotaExceededError`.
        """
        with self._lock:
            now = time.time()
            self._prune(now)
            if len(self._timestamps) >= self._limit:
                return (False, len(self._timestamps))
            self._timestamps.append(now)
            return (True, len(self._timestamps))

    def used_in_window(self) -> int:
        with self._lock:
            self._prune(time.time())
            return len(self._timestamps)

    def seconds_until_next_slot(self) -> float:
        """How long the caller would need to wait before the next slot
        opens. Returns 0 if a slot is currently available.

        Useful for surfacing in the UI: "Gemini saturé, prochain slot
        dans ~12s".
        """
        with self._lock:
            now = time.time()
            self._prune(now)
            if len(self._timestamps) < self._limit:
                return 0.0
            # The oldest timestamp will fall out of the window in
            # (oldest + 60) - now seconds.
            oldest = min(self._timestamps)
            return max(0.0, (oldest + 60.0) - now)

    def reset(self) -> None:
        # Test helper.
        with self._lock:
            self._timestamps = []


# ── Response dataclass ─────────────────────────────────────────────────


@dataclass
class GeminiResponse:
    """Result of a successful :meth:`GeminiClient.generate` call."""

    text: str
    """The generated text content (concatenation of all text parts)."""

    finish_reason: str | None
    """Why generation stopped (``STOP``, ``MAX_TOKENS``, ``SAFETY``…)."""

    usage_input_tokens: int
    """Tokens billed as input. ``0`` if not reported by the API."""

    usage_output_tokens: int
    """Tokens billed as output. ``0`` if not reported by the API."""

    model: str
    """The exact model id that responded."""

    raw: dict[str, Any]
    """The full JSON response, for debug. Do not parse this in
    application code — use the typed fields above."""


# ── Main client ────────────────────────────────────────────────────────


class GeminiClient:
    """Synchronous Gemini API client with retry and quota tracking.

    Parameters
    ----------
    api_key:
        The Google AI Studio API key. Validated for format at init.
    model:
        Default model id (e.g. ``"gemini-3.5-flash"``). Can be overridden
        per call via :meth:`generate`'s ``model`` parameter.
    timeout:
        Per-request timeout in seconds. Default: 30s.
    max_retries:
        Number of retry attempts for transient failures. Default: 3.
    backoff_base:
        Base for exponential backoff (in seconds). Default: 0.5.
    daily_limit:
        Soft daily quota for the local tracker. Default: 1500
        (Gemini Flash free tier).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        daily_limit: int = FREE_TIER_DAILY_QUOTA,
        per_minute_limit: int = FREE_TIER_PER_MINUTE_QUOTA,
    ) -> None:
        if not HTTPX_AVAILABLE:
            raise GeminiClientError(
                "httpx is not installed. Run: pip install httpx"
            )
        if not validate_api_key_format(api_key):
            raise GeminiClientError(
                f"Invalid Google API key format ({mask_api_key(api_key)}). "
                "Expected pattern: AIzaSy + 33 chars. Get a key from "
                "https://aistudio.google.com/app/apikey"
            )
        self._api_key = api_key
        self._model = model
        self._timeout = float(timeout)
        self._max_retries = int(max_retries)
        self._backoff_base = float(backoff_base)
        self._quota = _DailyQuotaTracker(daily_limit=daily_limit)
        # V3.9.4: client-side throttle to prevent free-tier 429s on
        # short bursts. Server-side quota is still authoritative.
        self._rate_limiter = _PerMinuteRateLimiter(
            per_minute_limit=per_minute_limit,
        )

        log.info(
            "GeminiClient initialised — model=%s key=%s daily=%d per_min=%d",
            model, mask_api_key(api_key), daily_limit, per_minute_limit,
        )

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._model

    @property
    def quota_used(self) -> int:
        return self._quota.used()

    @property
    def quota_remaining(self) -> int:
        return self._quota.remaining()

    @property
    def per_minute_used(self) -> int:
        """How many requests the client has made in the past 60s.

        Useful for surfacing a "Gemini saturé bientôt" warning in
        the UI before the throttle actually triggers.
        """
        return self._rate_limiter.used_in_window()

    @property
    def per_minute_seconds_until_slot(self) -> float:
        """Seconds the caller would have to wait before the next slot.

        Returns 0.0 if a slot is currently available.
        """
        return self._rate_limiter.seconds_until_next_slot()

    def reset_quota_counter(self) -> None:
        """Test helper. Don't call in production code."""
        self._quota.reset()
        self._rate_limiter.reset()

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model: str | None = None,
    ) -> GeminiResponse:
        """Send a chat-style request to Gemini and return the response.

        Parameters
        ----------
        system_prompt:
            The system instruction. Lythéa's full SYSTEM_PROMPT plus
            the assembled context blocks (identity, KG, episodic, …)
            should be passed here.
        messages:
            Chat history as a list of ``{"role": "user"|"assistant",
            "content": str}`` dicts. Empty list is allowed but unusual
            (the generation will be driven entirely by the system
            prompt).
        max_tokens:
            Hard cap on generated tokens. Default: 1024.
        temperature:
            Sampling temperature. Default: 0.7.
        model:
            Override the default model for this single call.

        Raises
        ------
        GeminiQuotaExceededError
            Daily local quota or server-side quota exhausted.
        GeminiUnauthorizedError
            API key rejected (401/403).
        GeminiTransientError
            Network or 5xx after all retries. The cascade should fall
            back to the local model.
        GeminiClientError
            Any other unexpected error (4xx, malformed response, …).
        """
        if self._quota.is_exhausted():
            raise GeminiQuotaExceededError(
                f"Local daily quota exhausted ({self._quota.used()} reqs). "
                "Resets at midnight."
            )

        # V3.9.4: per-minute throttle. Reserve a slot atomically; if
        # the window is full, surface the wait time to the caller so
        # the cascade can fall back gracefully instead of waiting.
        allowed, in_window = self._rate_limiter.check_and_record()
        if not allowed:
            wait_s = self._rate_limiter.seconds_until_next_slot()
            raise GeminiQuotaExceededError(
                f"Per-minute rate limit reached ({in_window} reqs in 60s). "
                f"Next slot in ~{wait_s:.0f}s."
            )

        target_model = model or self._model
        url = (
            f"{GEMINI_API_BASE}/models/{target_model}:generateContent"
            f"?key={self._api_key}"
        )
        body = self._build_request_body(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Increment the local counter BEFORE the call — even failed
        # calls count toward the free tier.
        used = self._quota.increment()

        try:
            response_json = self._post_with_retry(url, body)
        except (GeminiQuotaExceededError, GeminiUnauthorizedError):
            # Bubble up unchanged.
            raise
        except GeminiTransientError as exc:
            log.warning("Gemini transient failure after retries: %s", exc)
            raise
        except Exception as exc:  # pragma: no cover  - defensive
            log.exception("Gemini call failed (unexpected): %s", exc)
            raise GeminiClientError(f"Unexpected error: {exc}") from exc

        log.debug(
            "Gemini OK (req #%d today, model=%s, payload=%d chars)",
            used, target_model, len(str(response_json)),
        )
        return self._parse_response(response_json, model=target_model)

    def test_connection(self) -> bool:
        """Send a minimal request to verify the key works.

        Returns ``True`` if the API replies with a 200, ``False`` if
        the key is rejected or any other error occurs.

        Note: this consumes 1 quota unit on the free tier.
        """
        try:
            self.generate(
                system_prompt="You are a test assistant.",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
                temperature=0.0,
            )
            return True
        except Exception as exc:
            log.warning("Gemini connection test failed: %s", exc)
            return False

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_request_body(
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Translate Lythéa's chat format to Gemini's REST schema.

        Gemini uses ``contents`` (list of turns) plus an optional
        ``systemInstruction``. Roles in ``contents`` are ``user`` and
        ``model`` (not ``assistant``).
        """
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if not text:
                continue
            gemini_role = "user" if role == "user" else "model"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": text}],
            })

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        if system_prompt:
            body["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }
        return body

    def _post_with_retry(
        self, url: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the POST with exponential-backoff retry.

        Retries on:
          - Network errors (ConnectError, ReadTimeout, …)
          - HTTP 500/502/503/504
          - HTTP 429 (rate limit) — but only if we haven't hit our
            local quota yet, otherwise we raise immediately.

        Does NOT retry on:
          - HTTP 400/401/403/404 (programming errors or bad key)
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout) as http:
                    resp = http.post(url, json=body)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._backoff_base * (2 ** attempt)
                    log.info(
                        "Gemini network retry %d/%d after %.2fs (err=%s)",
                        attempt + 1, self._max_retries, delay, exc,
                    )
                    time.sleep(delay)
                    continue
                raise GeminiTransientError(f"Network error: {exc}") from exc

            # We have an HTTP response — interpret status code.
            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (401, 403):
                msg = self._extract_error_message(resp)
                raise GeminiUnauthorizedError(
                    f"API key rejected (HTTP {resp.status_code}): {msg}"
                )

            if resp.status_code == 429:
                # Rate-limited / quota exhausted server-side.
                msg = self._extract_error_message(resp)
                raise GeminiQuotaExceededError(
                    f"Server-side quota exceeded: {msg}"
                )

            if 500 <= resp.status_code < 600:
                last_exc = GeminiTransientError(
                    f"Server error HTTP {resp.status_code}: "
                    f"{self._extract_error_message(resp)}"
                )
                if attempt < self._max_retries:
                    delay = self._backoff_base * (2 ** attempt)
                    log.info(
                        "Gemini 5xx retry %d/%d after %.2fs (status=%d)",
                        attempt + 1, self._max_retries, delay,
                        resp.status_code,
                    )
                    time.sleep(delay)
                    continue
                raise last_exc

            # Other 4xx — programming error, do not retry.
            raise GeminiClientError(
                f"HTTP {resp.status_code}: "
                f"{self._extract_error_message(resp)}"
            )

        # Should never reach here but defensive.
        if last_exc is not None:
            raise GeminiTransientError(str(last_exc))
        raise GeminiClientError("Unknown error during retry loop")

    @staticmethod
    def _extract_error_message(resp: "httpx.Response") -> str:
        """Best-effort extraction of the human error message."""
        try:
            data = resp.json()
        except Exception:
            return resp.text[:200]
        err = data.get("error", {}) if isinstance(data, dict) else {}
        msg = err.get("message", "")
        if not msg:
            return resp.text[:200]
        return str(msg)[:300]

    @staticmethod
    def _parse_response(
        data: dict[str, Any], model: str
    ) -> GeminiResponse:
        """Parse the Gemini JSON response into our typed dataclass."""
        candidates = data.get("candidates") or []
        if not candidates:
            # Could happen if the response was filtered by safety.
            block_reason = (
                data.get("promptFeedback", {})
                .get("blockReason", "no candidates")
            )
            raise GeminiClientError(
                f"Empty response from Gemini: {block_reason}"
            )

        first = candidates[0]
        finish_reason = first.get("finishReason")
        content = first.get("content", {}) or {}
        parts = content.get("parts", []) or []
        text = "".join(
            p.get("text", "") for p in parts if isinstance(p, dict)
        )

        usage = data.get("usageMetadata", {}) or {}
        return GeminiResponse(
            text=text,
            finish_reason=finish_reason,
            usage_input_tokens=int(usage.get("promptTokenCount", 0) or 0),
            usage_output_tokens=int(
                usage.get("candidatesTokenCount", 0) or 0
            ),
            model=model,
            raw=data,
        )
