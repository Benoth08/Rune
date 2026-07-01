"""Tests for the authentication middleware."""
from __future__ import annotations

import pytest

from rune.server.auth import (
    AuthMiddleware,
    LOOPBACK_HOSTS,
    _extract_bearer,
    _is_cloudflare_tunneled,
    _is_loopback,
    auth_banner,
)


# ── Pure helpers ──────────────────────────────────────────────────────

def test_loopback_recognises_standard_hosts():
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("::1")
    assert _is_loopback("localhost")


def test_loopback_rejects_external():
    assert not _is_loopback("8.8.8.8")
    assert not _is_loopback("2a01:cb00:::abcd")
    assert not _is_loopback("")
    assert not _is_loopback(None)


def test_loopback_strips_brackets():
    """IPv6 with brackets like [::1] should still match."""
    assert _is_loopback("[::1]")


def test_extract_bearer_from_header():
    """Helper: build a fake Request-like object with .headers."""
    class FakeRequest:
        def __init__(self, header):
            self.headers = {"authorization": header} if header else {}

    assert _extract_bearer(FakeRequest("Bearer abc123")) == "abc123"
    assert _extract_bearer(FakeRequest("bearer abc123")) == "abc123"  # case
    assert _extract_bearer(FakeRequest("")) is None
    assert _extract_bearer(FakeRequest(None)) is None
    assert _extract_bearer(FakeRequest("Basic abc")) is None  # wrong scheme
    assert _extract_bearer(FakeRequest("Bearer ")) is None  # empty


def test_cloudflare_detection_positive():
    """Headers cf-connecting-ip or cf-ray indicate a tunneled request."""
    class FakeRequest:
        def __init__(self, headers):
            self.headers = headers

    assert _is_cloudflare_tunneled(FakeRequest({"cf-connecting-ip": "8.8.8.8"}))
    assert _is_cloudflare_tunneled(FakeRequest({"cf-ray": "abc-DEF"}))
    assert _is_cloudflare_tunneled(FakeRequest({
        "cf-connecting-ip": "8.8.8.8", "cf-ray": "x",
    }))


def test_cloudflare_detection_negative():
    """Without CF headers we don't claim tunneled."""
    class FakeRequest:
        def __init__(self, headers):
            self.headers = headers

    assert not _is_cloudflare_tunneled(FakeRequest({}))
    assert not _is_cloudflare_tunneled(FakeRequest({"x-forwarded-for": "8.8.8.8"}))


def test_cloudflare_paranoia_loopback_cf_ip():
    """If cf-connecting-ip is itself loopback, treat as not tunneled.

    This guards against poorly-configured upstream proxies that
    mirror our local IP into the header.
    """
    class FakeRequest:
        def __init__(self, headers):
            self.headers = headers

    assert not _is_cloudflare_tunneled(FakeRequest({"cf-connecting-ip": "127.0.0.1"}))


# ── Banner ────────────────────────────────────────────────────────────

def test_banner_open_mode():
    msg = auth_banner("", strict=False)
    assert "OPEN" in msg
    assert "local" in msg.lower()


def test_banner_normal_mode():
    msg = auth_banner("xyz", strict=False)
    assert "Loopback" in msg
    assert "127.0.0.1" in msg


def test_banner_strict_mode():
    msg = auth_banner("xyz", strict=True)
    assert "Strict" in msg


# ── Integration tests via TestClient ──────────────────────────────────

@pytest.fixture
def make_app():
    """Build a minimal FastAPI app with AuthMiddleware to test routing."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    def _make(token: str = "", strict: bool = False, public=()):
        app = FastAPI()

        @app.get("/api/protected")
        def protected():
            return {"ok": True}

        @app.get("/api/boot/status")
        def status():
            return {"ready": False}

        @app.get("/")
        def index():
            return {"page": "home"}

        app.add_middleware(
            AuthMiddleware,
            expected_token=token, strict=strict, public_paths=public,
        )
        return TestClient(app)
    return _make


def test_no_token_means_open(make_app):
    client = make_app(token="")
    r = client.get("/api/protected")
    assert r.status_code == 200


def test_loopback_bypasses_when_token_set(make_app):
    """A real 127.0.0.1 client should bypass auth.

    Note: FastAPI TestClient sets client.host="testclient", not a real
    loopback IP, so we cannot test this end-to-end via the client. We
    test the helper directly instead, and trust the integration via
    other tests.
    """
    # The unit-level guarantee is in test_loopback_recognises_standard_hosts.
    # Here we just verify the middleware *would* bypass via a mocked request.
    from unittest.mock import MagicMock, AsyncMock

    middleware = AuthMiddleware(
        app=None, expected_token="secret", strict=False, public_paths=(),
    )
    fake_request = MagicMock()
    fake_request.url.path = "/api/protected"
    fake_request.headers = {}
    fake_request.client.host = "127.0.0.1"

    call_next = AsyncMock(return_value="OK_RESPONSE")

    import asyncio
    result = asyncio.run(middleware.dispatch(fake_request, call_next))
    assert result == "OK_RESPONSE"
    call_next.assert_called_once()


def test_strict_mode_requires_token_even_on_loopback(make_app):
    """In strict mode, the token is required regardless of origin.

    We test via a mocked loopback request to be explicit (TestClient
    doesn't simulate real 127.0.0.1 — see test_loopback_bypasses_*).
    """
    from unittest.mock import MagicMock, AsyncMock
    import asyncio

    middleware = AuthMiddleware(
        app=None, expected_token="secret", strict=True, public_paths=(),
    )

    # Loopback + no token: still rejected because strict
    fake = MagicMock()
    fake.url.path = "/api/protected"
    fake.headers = {}
    fake.client.host = "127.0.0.1"
    call_next = AsyncMock()
    res = asyncio.run(middleware.dispatch(fake, call_next))
    assert res.status_code == 401
    call_next.assert_not_called()

    # Loopback + valid token: pass
    fake2 = MagicMock()
    fake2.url.path = "/api/protected"
    fake2.headers = {"authorization": "Bearer secret"}
    fake2.client.host = "127.0.0.1"
    call_next2 = AsyncMock(return_value="OK")
    res2 = asyncio.run(middleware.dispatch(fake2, call_next2))
    assert res2 == "OK"


def test_cloudflare_header_forces_auth(make_app):
    client = make_app(token="secret")
    # Even from "loopback", the CF header forces auth check
    r = client.get("/api/protected", headers={"cf-connecting-ip": "8.8.8.8"})
    assert r.status_code == 401
    r = client.get("/api/protected", headers={
        "cf-connecting-ip": "8.8.8.8",
        "Authorization": "Bearer secret",
    })
    assert r.status_code == 200


def test_public_paths_bypass_auth(make_app):
    client = make_app(token="secret", strict=True, public=("/api/boot/status",))
    r = client.get("/api/boot/status")
    assert r.status_code == 200
    # Other routes still locked
    r = client.get("/api/protected")
    assert r.status_code == 401


def test_static_paths_pass_through(make_app):
    """Non-/api/ paths must always pass (they need to load the JS that
    will then prompt for the token)."""
    client = make_app(token="secret", strict=True)
    r = client.get("/")
    assert r.status_code == 200


def test_invalid_bearer_format(make_app):
    client = make_app(token="secret", strict=True)
    r = client.get("/api/protected", headers={"Authorization": "secret"})  # no Bearer
    assert r.status_code == 401
