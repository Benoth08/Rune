"""Integration tests for the V3.9 cascade in :class:`Hippocampe`.

These tests verify the wiring without exercising the whole streaming
generation pipeline. They use mocks for the Gemini client and the
local model so they run without torch and without network.

What's covered
--------------

* Cascade is None when ``enable_cascade=False`` — V3 path preserved
* Cascade is built when settings allow + a valid key is in env
* Cascade is None when the key is malformed (no crash, just disabled)
* Cascade is None when ``GOOGLE_API_KEY`` is missing
* :meth:`Hippocampe.cascade_status` never leaks the full API key
* ``cascade_enabled`` boolean reflects the readiness state
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# torch needed for Hippocampe init in standard tests; we skip the whole
# module in sandbox without torch.
try:
    import torch as _torch_check  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torch not available or broken CUDA", allow_module_level=True)
pytest.importorskip("httpx", reason="cascade needs httpx")


VALID_KEY = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"


# ── Helpers ────────────────────────────────────────────────────────────


def _make_minimal_settings(
    enable_cascade: bool = False,
    google_api_key: str | None = None,
    cascade_gemini_model: str = "gemini-2.5-flash",
    cascade_synthesis_threshold_tokens: int = 50,
    cascade_synthesis_max_tokens: int = 120,
    cascade_gemini_max_tokens: int = 800,
    cascade_gemini_temperature: float = 0.7,
    cascade_daily_quota_hint: int = 1500,
):
    """Build a SimpleNamespace mimicking the cascade-relevant fields
    of LytheaSettings. We only need the subset _build_cascade_if_enabled
    reads — anything else is irrelevant."""
    return SimpleNamespace(
        enable_cascade=enable_cascade,
        google_api_key=google_api_key,
        cascade_gemini_model=cascade_gemini_model,
        cascade_synthesis_threshold_tokens=cascade_synthesis_threshold_tokens,
        cascade_synthesis_max_tokens=cascade_synthesis_max_tokens,
        cascade_gemini_max_tokens=cascade_gemini_max_tokens,
        cascade_gemini_temperature=cascade_gemini_temperature,
        cascade_daily_quota_hint=cascade_daily_quota_hint,
    )


class _FakeHippocampe:
    """Minimal subclass-shaped stand-in. We test the helper methods
    directly without doing the full Hippocampe.__init__ which pulls in
    SDM/MHN/KG/torch. The helpers we want to test (``_build_cascade_if_enabled``,
    ``cascade_enabled``, ``cascade_status``, ``_generate_local_blocking``)
    only depend on a small surface."""

    def __init__(self, model_loaded: bool = True):
        # Stub the model interface used by _generate_local_blocking.
        self.model = SimpleNamespace(
            is_loaded=model_loaded,
            stream_generate=lambda *a, **k: iter([{"text": "stub"}]),
        )
        self.sampling_profile = SimpleNamespace()
        self._cascade = None
        # Clé API saisie à chaud (UI) — RAM only. Le vrai Hippocampe la
        # définit dans __init__ ; le mock doit la reproduire car
        # _build_cascade_if_enabled la lit (self._cascade_key_override
        # or settings.google_api_key).
        self._cascade_key_override = None

    # Bind real methods from Hippocampe so we test the actual code.
    from rune.hippocampe import Hippocampe
    _build_cascade_if_enabled = Hippocampe._build_cascade_if_enabled
    _generate_local_blocking = Hippocampe._generate_local_blocking
    cascade_enabled = Hippocampe.cascade_enabled
    cascade_status = Hippocampe.cascade_status


# ── Build path ─────────────────────────────────────────────────────────


class TestCascadeBuilder:
    def test_disabled_returns_none(self):
        h = _FakeHippocampe()
        s = _make_minimal_settings(enable_cascade=False)
        result = h._build_cascade_if_enabled(s)
        assert result is None

    def test_enabled_without_key_returns_none(self):
        h = _FakeHippocampe()
        s = _make_minimal_settings(enable_cascade=True, google_api_key=None)
        result = h._build_cascade_if_enabled(s)
        assert result is None

    def test_enabled_with_invalid_key_returns_none(self):
        """Bad key format must NOT crash — log warning, return None,
        let Rune run in V3 mode."""
        h = _FakeHippocampe()
        s = _make_minimal_settings(
            enable_cascade=True, google_api_key="not-a-valid-key"
        )
        result = h._build_cascade_if_enabled(s)
        assert result is None

    def test_enabled_with_good_key_builds_cascade(self):
        h = _FakeHippocampe()
        s = _make_minimal_settings(
            enable_cascade=True, google_api_key=VALID_KEY
        )
        result = h._build_cascade_if_enabled(s)
        assert result is not None
        assert result.is_enabled

    def test_enabled_with_empty_string_key_returns_none(self):
        h = _FakeHippocampe()
        s = _make_minimal_settings(enable_cascade=True, google_api_key="")
        result = h._build_cascade_if_enabled(s)
        assert result is None


# ── Status endpoint contract ──────────────────────────────────────────


class TestCascadeStatus:
    def _patched_settings(self, **kw):
        """Patch rune.settings.get_settings to return our fake."""
        s = _make_minimal_settings(**kw)
        return patch("rune.settings.get_settings", return_value=s)

    def test_status_disabled_has_required_fields(self):
        h = _FakeHippocampe()
        h._cascade = None
        with self._patched_settings(enable_cascade=False):
            status = h.cascade_status()
        assert status["enabled"] is False
        assert status["reason"] == "disabled"
        assert "api_key_masked" in status
        # No key set → masked as <missing>
        assert status["api_key_masked"] == "<missing>"
        assert status["quota_used"] == 0

    def test_status_no_api_key_reason(self):
        h = _FakeHippocampe()
        h._cascade = None
        with self._patched_settings(
            enable_cascade=True, google_api_key=None
        ):
            status = h.cascade_status()
        assert status["reason"] == "no_api_key"

    def test_status_ready_never_leaks_full_key(self):
        h = _FakeHippocampe()
        s = _make_minimal_settings(
            enable_cascade=True, google_api_key=VALID_KEY,
        )
        h._cascade = h._build_cascade_if_enabled(s)
        assert h._cascade is not None

        with self._patched_settings(
            enable_cascade=True, google_api_key=VALID_KEY,
        ):
            status = h.cascade_status()

        assert status["enabled"] is True
        assert status["reason"] == "ready"
        # Key must be masked — only last 4 chars shown.
        assert status["api_key_masked"] == "...4567"
        assert VALID_KEY not in str(status)
        assert "AIzaSy" not in status["api_key_masked"]

    def test_status_ready_exposes_quota_counter(self):
        h = _FakeHippocampe()
        s = _make_minimal_settings(
            enable_cascade=True, google_api_key=VALID_KEY,
        )
        h._cascade = h._build_cascade_if_enabled(s)

        with self._patched_settings(
            enable_cascade=True, google_api_key=VALID_KEY,
        ):
            status = h.cascade_status()

        assert "quota_used" in status
        assert "quota_remaining" in status
        assert status["quota_used"] == 0
        assert status["quota_remaining"] == 1500


# ── cascade_enabled property ──────────────────────────────────────────


class TestCascadeEnabledProperty:
    def test_no_cascade_is_disabled(self):
        h = _FakeHippocampe()
        h._cascade = None
        # Property descriptor needs explicit __get__ when bound this way
        # since we monkeypatch it from a real class.
        assert _FakeHippocampe.cascade_enabled.fget(h) is False

    def test_built_cascade_is_enabled(self):
        h = _FakeHippocampe()
        s = _make_minimal_settings(
            enable_cascade=True, google_api_key=VALID_KEY,
        )
        h._cascade = h._build_cascade_if_enabled(s)
        assert _FakeHippocampe.cascade_enabled.fget(h) is True
