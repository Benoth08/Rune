"""Tests for the GET config endpoints (entropy, web-mode).

These endpoints exist so the UI can fetch the live backend state at
load time instead of relying on HTML hardcoded defaults. Without them,
the slider/dropdown displays values that may be stale, and clicking
"Save" pushes the displayed (stale) value back to the backend —
silently overriding the actual setting.

The functions are tested by invoking them directly with a mock
``request`` rather than spinning up a full TestClient: the routes are
trivial single-line readers, the rest is FastAPI plumbing that doesn't
need to be exercised here.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The routes module transitively imports lythea.model which requires
# torch. In the sandbox without torch, skip the whole module — on the
# real pod torch is always available so these tests run normally.
pytest.importorskip("torch", reason="lythea.server.routes imports model which needs torch")


def _make_request(entropy_threshold: float = 0.2, web_mode: str = "auto"):
    """Build a fake request whose ``app.state.lythea`` exposes the
    minimal surface our two GET endpoints touch.
    """
    web_policy = SimpleNamespace(mode=web_mode)
    hippocampe = SimpleNamespace(
        entropy_threshold=entropy_threshold,
        web_policy=web_policy,
    )
    lythea_app = SimpleNamespace(hippocampe=hippocampe)
    state = SimpleNamespace(lythea=lythea_app)
    fake_app = SimpleNamespace(state=state)
    return SimpleNamespace(app=fake_app)


def test_get_entropy_returns_backend_value():
    """The GET endpoint must reflect the backend's *live* threshold,
    not a hardcoded constant. Tested with a non-default value to make
    sure we're really reading from hippocampe and not echoing a const.
    """
    from rune.server.routes import get_entropy

    request = _make_request(entropy_threshold=0.37)
    result = asyncio.run(get_entropy(request))

    assert result == {"threshold": 0.37}


def test_get_entropy_after_post_reflects_new_value():
    """Round-trip check: simulate a POST mutation by directly setting
    the attribute, then GET — the new value must come back. This is
    the property the UI relies on after a user clicks Save."""
    from rune.server.routes import get_entropy

    request = _make_request(entropy_threshold=0.5)
    # Simulate the POST handler having mutated the attribute.
    request.app.state.lythea.hippocampe.entropy_threshold = 0.15
    result = asyncio.run(get_entropy(request))

    assert result == {"threshold": 0.15}


def test_get_web_mode_returns_backend_value():
    """Same property for web-mode: GET must mirror the live state."""
    from rune.server.routes import get_web_mode

    for mode in ("off", "auto", "always"):
        request = _make_request(web_mode=mode)
        result = asyncio.run(get_web_mode(request))
        assert result == {"mode": mode}, f"web-mode={mode} not echoed"


# ── V3.9 cascade endpoint ─────────────────────────────────────────────


def _make_cascade_request(cascade_status: dict):
    """Build a fake request whose hippocampe.cascade_status() returns
    the given dict. Used to test get_cascade_config without spinning
    up a full server."""
    hippocampe = SimpleNamespace(cascade_status=lambda: cascade_status)
    lythea_app = SimpleNamespace(hippocampe=hippocampe)
    state = SimpleNamespace(lythea=lythea_app)
    fake_app = SimpleNamespace(state=state)
    return SimpleNamespace(app=fake_app)


def test_cascade_endpoint_disabled_state():
    """When the cascade is off, the endpoint surfaces enabled=False
    and the reason. The api_key_masked field must be present (even
    when no key is set) so the UI can display '<missing>'."""
    from rune.server.routes import get_cascade_config

    request = _make_cascade_request({
        "enabled": False,
        "reason": "disabled",
        "model": "gemini-2.5-flash",
        "api_key_masked": "<missing>",
        "quota_used": 0,
        "quota_remaining": 0,
    })
    payload = asyncio.run(get_cascade_config(request))
    assert payload["enabled"] is False
    assert payload["reason"] == "disabled"
    assert payload["api_key_masked"] == "<missing>"


def test_cascade_endpoint_ready_state():
    """When the cascade is configured and ready, the endpoint returns
    the model id, a MASKED key (never the full string), and the
    quota counters."""
    from rune.server.routes import get_cascade_config

    request = _make_cascade_request({
        "enabled": True,
        "reason": "ready",
        "model": "gemini-2.5-flash",
        "api_key_masked": "...4567",
        "quota_used": 12,
        "quota_remaining": 1488,
        "synthesis_threshold_tokens": 50,
        "synthesis_max_tokens": 120,
    })
    payload = asyncio.run(get_cascade_config(request))
    assert payload["enabled"] is True
    assert payload["model"] == "gemini-2.5-flash"
    assert payload["quota_used"] == 12
    # CRITICAL: full key must NEVER appear, only the mask.
    assert payload["api_key_masked"].startswith("...")
    assert "AIzaSy" not in payload["api_key_masked"]


def test_cascade_endpoint_no_api_key_reason():
    """If enable_cascade=True but the key is missing, the reason
    must be ``no_api_key`` so the UI can display a clear message."""
    from rune.server.routes import get_cascade_config

    request = _make_cascade_request({
        "enabled": False,
        "reason": "no_api_key",
        "model": "gemini-2.5-flash",
        "api_key_masked": "<missing>",
        "quota_used": 0,
        "quota_remaining": 0,
    })
    payload = asyncio.run(get_cascade_config(request))
    assert payload["reason"] == "no_api_key"


# ── V3.9.4 cascade toggle endpoint ────────────────────────────────────


def _make_toggle_request(cascade_status_returns: list[dict], hippocampe_extra=None):
    """Build a fake request whose hippocampe.cascade_status() returns
    each given dict on successive calls. The toggle endpoint calls
    cascade_status() at the end so we need at least one entry."""
    state = {"calls": 0}

    def cascade_status():
        i = min(state["calls"], len(cascade_status_returns) - 1)
        state["calls"] += 1
        return cascade_status_returns[i]

    hippocampe = SimpleNamespace(
        cascade_status=cascade_status,
        cascade_enabled=cascade_status_returns[0].get("enabled", False),
        _cascade=None if not cascade_status_returns[0].get("enabled") else object(),
    )
    if hippocampe_extra:
        for k, v in hippocampe_extra.items():
            setattr(hippocampe, k, v)

    lythea_app = SimpleNamespace(hippocampe=hippocampe)
    state_obj = SimpleNamespace(lythea=lythea_app)
    fake_app = SimpleNamespace(state=state_obj)

    request = SimpleNamespace(app=fake_app)

    async def json():
        return {}

    request.json = json
    return request


def test_cascade_toggle_endpoint_is_no_op_when_already_correct():
    """If the user requests enabled=True and cascade is already enabled,
    the endpoint returns the current status without rebuilding."""
    from rune.server.routes import toggle_cascade

    enabled_status = {
        "enabled": True, "reason": "ready", "model": "gemini-2.5-flash",
        "api_key_masked": "...4567", "quota_used": 5, "quota_remaining": 1495,
    }

    request = _make_toggle_request([enabled_status, enabled_status])

    async def json():
        return {"enabled": True}
    request.json = json

    result = asyncio.run(toggle_cascade(request))
    assert result["enabled"] is True


def test_cascade_toggle_endpoint_disables():
    """Toggling from enabled to disabled should set _cascade to None."""
    from rune.server.routes import toggle_cascade

    enabled = {
        "enabled": True, "reason": "ready", "model": "gemini-2.5-flash",
        "api_key_masked": "...4567", "quota_used": 5, "quota_remaining": 1495,
    }
    disabled = {
        "enabled": False, "reason": "disabled", "model": "gemini-2.5-flash",
        "api_key_masked": "...4567", "quota_used": 0, "quota_remaining": 0,
    }

    # First call returns enabled (initial), second returns disabled (after toggle)
    request = _make_toggle_request([enabled, disabled])

    async def json():
        return {"enabled": False}
    request.json = json

    result = asyncio.run(toggle_cascade(request))
    # After disabling, _cascade should be None
    assert request.app.state.lythea.hippocampe._cascade is None


def test_cascade_toggle_endpoint_handles_empty_body():
    """Empty body means flip the current state."""
    from rune.server.routes import toggle_cascade

    enabled = {"enabled": True, "reason": "ready", "model": "x",
               "api_key_masked": "...x", "quota_used": 0, "quota_remaining": 0}
    disabled = {"enabled": False, "reason": "disabled", "model": "x",
                "api_key_masked": "...x", "quota_used": 0, "quota_remaining": 0}

    request = _make_toggle_request([enabled, disabled])

    async def json():
        raise ValueError("no body")
    request.json = json

    # Should not crash on empty body — flips current (True) to False
    result = asyncio.run(toggle_cascade(request))
    assert request.app.state.lythea.hippocampe._cascade is None
