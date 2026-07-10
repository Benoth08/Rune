"""Unit tests for :mod:`rune.external.gemini_client`.

These tests use ``unittest.mock`` to stub out ``httpx.Client`` so we
never make real network calls. Run normally in CI / on the pod —
no API key, no internet required.

Test groups
-----------

* ``test_mask_api_key_*`` — log-safe key formatting
* ``test_validate_format_*`` — boot-time syntactic check
* ``test_quota_*`` — daily counter behaviour
* ``test_generate_*`` — happy paths and error mapping (mocked HTTP)
* ``test_request_body_*`` — REST schema translation
* ``test_response_parsing_*`` — JSON → dataclass
* ``test_retry_*`` — exponential backoff on transient failures
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module if httpx is unavailable. The Gemini client
# requires it at instantiation time, but the helper functions below
# don't, so we import them eagerly and import the client lazily
# inside fixtures.
httpx = pytest.importorskip("httpx")

from rune.external.gemini_client import (
    FREE_TIER_DAILY_QUOTA,
    GeminiClient,
    GeminiClientError,
    GeminiQuotaExceededError,
    GeminiTransientError,
    GeminiUnauthorizedError,
    GeminiResponse,
    _DailyQuotaTracker,
    mask_api_key,
    validate_api_key_format,
)


VALID_KEY = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"  # 39 chars, valid pattern


# ── Helpers ────────────────────────────────────────────────────────────


def _make_response(
    status_code: int = 200,
    json_body: dict | None = None,
    text_body: str = "",
):
    """Build a stub httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if json_body is not None:
        resp.json.return_value = json_body
        resp.text = json.dumps(json_body)
    else:
        resp.json.side_effect = ValueError("no body")
        resp.text = text_body
    return resp


def _make_gemini_success_body(text: str = "Bonjour Mika.") -> dict:
    return {
        "candidates": [{
            "content": {"parts": [{"text": text}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 42,
            "candidatesTokenCount": 7,
        },
    }


# ── Helper function tests ──────────────────────────────────────────────


class TestMaskApiKey:
    def test_none_returns_missing(self):
        assert mask_api_key(None) == "<missing>"

    def test_too_short_returns_invalid(self):
        assert mask_api_key("abc") == "<invalid>"

    def test_non_string_returns_invalid(self):
        assert mask_api_key(12345) == "<invalid>"  # type: ignore

    def test_normal_key_shows_only_last_4(self):
        assert mask_api_key(VALID_KEY) == "...4567"

    def test_does_not_leak_full_key(self):
        # Sanity: the masked output must not contain the prefix.
        masked = mask_api_key(VALID_KEY)
        assert "AIzaSy" not in masked
        assert VALID_KEY[:30] not in masked


class TestValidateFormat:
    def test_valid_key(self):
        assert validate_api_key_format(VALID_KEY) is True

    def test_wrong_prefix(self):
        assert validate_api_key_format("BIzaSy" + "x" * 33) is False

    def test_too_short(self):
        assert validate_api_key_format("AIzaSy123") is False

    def test_too_long(self):
        assert validate_api_key_format(VALID_KEY + "extra") is False

    def test_invalid_chars(self):
        bad = "AIzaSy" + "!" * 33  # ! is not allowed
        assert validate_api_key_format(bad) is False

    def test_none(self):
        assert validate_api_key_format(None) is False

    def test_non_string(self):
        assert validate_api_key_format(12345) is False  # type: ignore

    def test_underscores_and_dashes_allowed(self):
        # Real-life keys can contain _ and -
        good = "AIzaSyABC_DEF-GHI_JKL_MNO_PQR_STU_VWX_Y"  # 39 chars
        assert validate_api_key_format(good) is True


# ── Quota tracker tests ────────────────────────────────────────────────


class TestQuotaTracker:
    def test_starts_at_zero(self):
        q = _DailyQuotaTracker()
        assert q.used() == 0
        assert q.remaining() == FREE_TIER_DAILY_QUOTA

    def test_increment(self):
        q = _DailyQuotaTracker()
        assert q.increment() == 1
        assert q.increment() == 2
        assert q.used() == 2

    def test_remaining_decreases(self):
        q = _DailyQuotaTracker(daily_limit=10)
        for _ in range(3):
            q.increment()
        assert q.remaining() == 7

    def test_exhausted(self):
        q = _DailyQuotaTracker(daily_limit=2)
        assert not q.is_exhausted()
        q.increment()
        q.increment()
        assert q.is_exhausted()
        assert q.remaining() == 0

    def test_reset_for_tests(self):
        q = _DailyQuotaTracker()
        q.increment()
        q.increment()
        q.reset()
        assert q.used() == 0


# ── Constructor tests ──────────────────────────────────────────────────


class TestGeminiClientInit:
    def test_rejects_invalid_key_format(self):
        with pytest.raises(GeminiClientError, match="Invalid Google API key"):
            GeminiClient(api_key="not-a-key")

    def test_rejects_none_key(self):
        with pytest.raises(GeminiClientError):
            GeminiClient(api_key=None)  # type: ignore

    def test_accepts_valid_key(self):
        client = GeminiClient(api_key=VALID_KEY)
        assert client.model == "gemini-3.5-flash"

    def test_custom_model(self):
        client = GeminiClient(api_key=VALID_KEY, model="gemini-2.5-flash-lite")
        assert client.model == "gemini-2.5-flash-lite"

    def test_custom_quota_limit(self):
        client = GeminiClient(api_key=VALID_KEY, daily_limit=100)
        assert client.quota_remaining == 100


# ── Request body construction tests ────────────────────────────────────


class TestRequestBody:
    def test_basic_user_message(self):
        body = GeminiClient._build_request_body(
            system_prompt="You are Rune.",
            messages=[{"role": "user", "content": "Bonjour"}],
            max_tokens=500,
            temperature=0.6,
        )
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "Bonjour"}]}
        ]
        assert body["systemInstruction"]["parts"][0]["text"] == "You are Rune."
        assert body["generationConfig"]["temperature"] == 0.6
        assert body["generationConfig"]["maxOutputTokens"] == 500

    def test_assistant_role_mapped_to_model(self):
        body = GeminiClient._build_request_body(
            system_prompt="",
            messages=[
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
                {"role": "user", "content": "Q2"},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        roles = [c["role"] for c in body["contents"]]
        assert roles == ["user", "model", "user"]

    def test_empty_content_skipped(self):
        body = GeminiClient._build_request_body(
            system_prompt="",
            messages=[
                {"role": "user", "content": ""},
                {"role": "user", "content": "real"},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        assert len(body["contents"]) == 1
        assert body["contents"][0]["parts"][0]["text"] == "real"

    def test_no_system_prompt_omits_field(self):
        body = GeminiClient._build_request_body(
            system_prompt="",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=100,
            temperature=0.5,
        )
        assert "systemInstruction" not in body


# ── Response parsing tests ─────────────────────────────────────────────


class TestResponseParsing:
    def test_basic_response(self):
        data = _make_gemini_success_body("Hello")
        resp = GeminiClient._parse_response(data, model="gemini-2.5-flash")
        assert resp.text == "Hello"
        assert resp.finish_reason == "STOP"
        assert resp.usage_input_tokens == 42
        assert resp.usage_output_tokens == 7
        assert resp.model == "gemini-2.5-flash"

    def test_concatenates_multiple_text_parts(self):
        data = {
            "candidates": [{
                "content": {"parts": [
                    {"text": "Hello "},
                    {"text": "world"},
                ]},
                "finishReason": "STOP",
            }],
        }
        resp = GeminiClient._parse_response(data, model="m")
        assert resp.text == "Hello world"

    def test_missing_usage_defaults_to_zero(self):
        data = {
            "candidates": [{
                "content": {"parts": [{"text": "x"}]},
                "finishReason": "STOP",
            }],
        }
        resp = GeminiClient._parse_response(data, model="m")
        assert resp.usage_input_tokens == 0
        assert resp.usage_output_tokens == 0

    def test_empty_candidates_raises(self):
        data = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
        with pytest.raises(GeminiClientError, match="Empty response"):
            GeminiClient._parse_response(data, model="m")

    def test_no_candidates_key_raises(self):
        with pytest.raises(GeminiClientError):
            GeminiClient._parse_response({}, model="m")


# ── End-to-end generation tests (mocked HTTP) ──────────────────────────


def _mock_httpx_client(response_or_responses):
    """Build a context-manager mock that returns the given response(s)."""
    cm = MagicMock()
    cm.__enter__.return_value = cm
    cm.__exit__.return_value = None
    if isinstance(response_or_responses, list):
        cm.post.side_effect = response_or_responses
    else:
        cm.post.return_value = response_or_responses
    return cm


class TestGenerate:
    def test_happy_path(self):
        client = GeminiClient(api_key=VALID_KEY)
        success_resp = _make_response(
            status_code=200,
            json_body=_make_gemini_success_body("Bonjour Mika."),
        )
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = _mock_httpx_client(success_resp)
            result = client.generate(
                system_prompt="You are Rune.",
                messages=[{"role": "user", "content": "Bonjour"}],
                max_tokens=100,
                temperature=0.7,
            )
        assert result.text == "Bonjour Mika."
        assert result.finish_reason == "STOP"
        assert client.quota_used == 1

    def test_url_contains_api_key(self):
        """The key MUST be passed as ?key= URL parameter for v1beta REST."""
        client = GeminiClient(api_key=VALID_KEY)
        success_resp = _make_response(
            status_code=200,
            json_body=_make_gemini_success_body("ok"),
        )
        captured = {}
        cm = _mock_httpx_client(success_resp)

        def fake_post(url, json=None):
            captured["url"] = url
            return success_resp
        cm.post.side_effect = fake_post

        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = cm
            client.generate(
                system_prompt="",
                messages=[{"role": "user", "content": "x"}],
            )

        assert f"key={VALID_KEY}" in captured["url"]
        # Le client construit l'URL avec DEFAULT_MODEL (gemini-3.5-flash
        # depuis la mise à jour du catalogue Gemini). Voir DEFAULT_MODEL
        # dans rune.external.gemini_client.
        assert "gemini-3.5-flash" in captured["url"]

    def test_401_raises_unauthorized(self):
        client = GeminiClient(api_key=VALID_KEY)
        bad = _make_response(
            status_code=401,
            json_body={"error": {"message": "API key invalid"}},
        )
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = _mock_httpx_client(bad)
            with pytest.raises(GeminiUnauthorizedError):
                client.generate(
                    system_prompt="",
                    messages=[{"role": "user", "content": "x"}],
                )

    def test_403_raises_unauthorized(self):
        client = GeminiClient(api_key=VALID_KEY)
        bad = _make_response(
            status_code=403,
            json_body={"error": {"message": "permission denied"}},
        )
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = _mock_httpx_client(bad)
            with pytest.raises(GeminiUnauthorizedError):
                client.generate(
                    system_prompt="",
                    messages=[{"role": "user", "content": "x"}],
                )

    def test_429_raises_quota_exceeded(self):
        client = GeminiClient(api_key=VALID_KEY)
        bad = _make_response(
            status_code=429,
            json_body={"error": {"message": "Quota exceeded"}},
        )
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = _mock_httpx_client(bad)
            with pytest.raises(GeminiQuotaExceededError):
                client.generate(
                    system_prompt="",
                    messages=[{"role": "user", "content": "x"}],
                )

    def test_local_quota_blocks_before_call(self):
        client = GeminiClient(api_key=VALID_KEY, daily_limit=1)
        # Exhaust the local counter without a real call.
        client._quota.increment()
        assert client.quota_remaining == 0
        with pytest.raises(GeminiQuotaExceededError, match="Local daily"):
            client.generate(
                system_prompt="",
                messages=[{"role": "user", "content": "x"}],
            )

    def test_400_raises_generic_error(self):
        client = GeminiClient(api_key=VALID_KEY)
        bad = _make_response(
            status_code=400,
            json_body={"error": {"message": "bad request"}},
        )
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = _mock_httpx_client(bad)
            with pytest.raises(GeminiClientError, match="HTTP 400"):
                client.generate(
                    system_prompt="",
                    messages=[{"role": "user", "content": "x"}],
                )


# ── Retry tests ────────────────────────────────────────────────────────


class TestRetry:
    def test_retries_on_500_then_succeeds(self):
        client = GeminiClient(
            api_key=VALID_KEY, max_retries=2, backoff_base=0.001
        )
        responses = [
            _make_response(500, json_body={"error": {"message": "oops"}}),
            _make_response(
                200, json_body=_make_gemini_success_body("recovered")
            ),
        ]
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = _mock_httpx_client(responses)
            result = client.generate(
                system_prompt="",
                messages=[{"role": "user", "content": "x"}],
            )
        assert result.text == "recovered"

    def test_gives_up_after_max_retries(self):
        client = GeminiClient(
            api_key=VALID_KEY, max_retries=1, backoff_base=0.001
        )
        bad = _make_response(503, json_body={"error": {"message": "down"}})
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = _mock_httpx_client([bad, bad])
            with pytest.raises(GeminiTransientError):
                client.generate(
                    system_prompt="",
                    messages=[{"role": "user", "content": "x"}],
                )

    def test_does_not_retry_on_401(self):
        client = GeminiClient(
            api_key=VALID_KEY, max_retries=3, backoff_base=0.001
        )
        bad = _make_response(401, json_body={"error": {"message": "no"}})
        cm = _mock_httpx_client(bad)
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = cm
            with pytest.raises(GeminiUnauthorizedError):
                client.generate(
                    system_prompt="",
                    messages=[{"role": "user", "content": "x"}],
                )
        # post() called exactly once — no retry on auth errors.
        assert cm.post.call_count == 1

    def test_retries_on_network_error(self):
        client = GeminiClient(
            api_key=VALID_KEY, max_retries=2, backoff_base=0.001
        )
        success_resp = _make_response(
            200, json_body=_make_gemini_success_body("ok after net error")
        )

        cm = MagicMock()
        cm.__enter__.return_value = cm
        cm.__exit__.return_value = None
        cm.post.side_effect = [
            httpx.ConnectError("connection refused"),
            success_resp,
        ]
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            cli.return_value = cm
            result = client.generate(
                system_prompt="",
                messages=[{"role": "user", "content": "x"}],
            )
        assert result.text == "ok after net error"


# ── V3.9.4 per-minute rate limiter ─────────────────────────────────────


class TestPerMinuteRateLimiter:
    """Tests for the V3.9.4 client-side rate limiter.

    Added after empirical observation: Gemini 2.5 Flash returns 429
    after ~10 successive requests within 60 seconds on the free tier,
    even when daily quota is barely used. The sliding-window limiter
    catches this client-side and surfaces a clear error to the cascade.
    """

    def test_starts_empty(self):
        from rune.external.gemini_client import _PerMinuteRateLimiter
        rl = _PerMinuteRateLimiter(per_minute_limit=8)
        assert rl.used_in_window() == 0
        assert rl.seconds_until_next_slot() == 0.0

    def test_allows_until_limit(self):
        from rune.external.gemini_client import _PerMinuteRateLimiter
        rl = _PerMinuteRateLimiter(per_minute_limit=3)
        for _ in range(3):
            allowed, count = rl.check_and_record()
            assert allowed is True
        assert rl.used_in_window() == 3

    def test_blocks_at_limit(self):
        from rune.external.gemini_client import _PerMinuteRateLimiter
        rl = _PerMinuteRateLimiter(per_minute_limit=2)
        rl.check_and_record()
        rl.check_and_record()
        allowed, count = rl.check_and_record()
        assert allowed is False
        assert count == 2

    def test_seconds_until_slot_when_full(self):
        from rune.external.gemini_client import _PerMinuteRateLimiter
        rl = _PerMinuteRateLimiter(per_minute_limit=1)
        rl.check_and_record()
        # Window is full — should return a positive wait time
        wait = rl.seconds_until_next_slot()
        assert wait > 0
        assert wait <= 60.0

    def test_old_entries_pruned(self):
        """Entries older than 60s should not count toward the limit."""
        from rune.external.gemini_client import _PerMinuteRateLimiter
        import time as time_mod
        rl = _PerMinuteRateLimiter(per_minute_limit=2)
        # Manually push old timestamps (61 seconds ago)
        old = time_mod.time() - 61
        rl._timestamps = [old, old]
        # New requests should be allowed (old entries pruned)
        allowed, count = rl.check_and_record()
        assert allowed is True
        assert count == 1

    def test_reset_clears_state(self):
        from rune.external.gemini_client import _PerMinuteRateLimiter
        rl = _PerMinuteRateLimiter(per_minute_limit=3)
        rl.check_and_record()
        rl.check_and_record()
        rl.reset()
        assert rl.used_in_window() == 0


class TestPerMinuteIntegratedInClient:
    """Tests verifying the rate limiter is wired up in GeminiClient."""

    def test_client_exposes_per_minute_used(self):
        from unittest.mock import patch, MagicMock
        client = GeminiClient(api_key=VALID_KEY, per_minute_limit=5)
        assert client.per_minute_used == 0

    def test_client_exposes_seconds_until_slot(self):
        client = GeminiClient(api_key=VALID_KEY)
        assert client.per_minute_seconds_until_slot == 0.0

    def test_per_minute_block_raises_quota_error(self):
        """When the per-minute window is full, generate() must raise
        GeminiQuotaExceededError with 'Per-minute' in the message
        (distinct from the daily-quota error)."""
        from unittest.mock import patch, MagicMock
        client = GeminiClient(api_key=VALID_KEY, per_minute_limit=1)

        # Pre-fill the per-minute window
        client._rate_limiter.check_and_record()

        # The next call should be blocked client-side without making
        # any network call.
        with patch("rune.external.gemini_client.httpx.Client") as cli:
            with pytest.raises(GeminiQuotaExceededError, match="Per-minute"):
                client.generate(
                    system_prompt="",
                    messages=[{"role": "user", "content": "x"}],
                )
            # No network call should have been attempted
            cli.assert_not_called()
