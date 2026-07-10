"""Authentication middleware — bearer token with smart loopback bypass.

Policy
------
The middleware decides whether a given request must present a valid
``Authorization: Bearer <token>`` header.

Decision matrix:

+----------------------+-------------------+-------------------------+
| auth_token defined?  | request origin    | auth required?          |
+======================+===================+=========================+
| No                   | any               | No (open dev mode)      |
| Yes (strict mode)    | any               | Yes                     |
| Yes                  | Cloudflare tunnel | Yes (always)            |
| Yes                  | loopback          | No (local dev bypass)   |
| Yes                  | other             | Yes                     |
+----------------------+-------------------+-------------------------+

Cloudflare detection
--------------------
A naive ``request.client.host`` check is not enough: Cloudflare tunnel
forwards the request via a local TCP connection, so the client IP would
appear as 127.0.0.1 and an attacker could bypass auth. We therefore
treat the request as "Cloudflare-tunneled" if any of the following
headers are present:

- ``cf-connecting-ip`` — Cloudflare's standard "real client IP" header
- ``cf-ray`` — Cloudflare's request tracing ID

Both are set by Cloudflare's edge before reaching our backend. They
cannot be spoofed by the end client (they would be overwritten or
rejected by CF). If the request reaches us with one of these headers
*and* a non-loopback ``cf-connecting-ip``, we know it came through
the tunnel.

Note that we don't try to parse other reverse proxies (nginx, traefik,
ngrok, etc.) — for those, the safe path is "if not loopback then
require token", which is exactly what the matrix above does.
"""
from __future__ import annotations

import logging
import secrets
from typing import Iterable

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("rune.server.auth")


# ── IP/host helpers ───────────────────────────────────────────────────

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback(host: str | None) -> bool:
    """True if the given host is a loopback address."""
    if not host:
        return False
    # Strip IPv6 zone and port artefacts ("[::1]:7860" → "::1")
    host = host.strip().lstrip("[").split("]")[0]
    return host in LOOPBACK_HOSTS


def _is_cloudflare_tunneled(request: Request) -> bool:
    """True if the request looks like it came through a Cloudflare tunnel.

    We look for headers that Cloudflare's edge sets and that should not
    appear on direct loopback requests. Header names are case-insensitive
    in HTTP and Starlette's ``Headers`` already normalises them.
    """
    h = request.headers
    cf_ip = h.get("cf-connecting-ip", "").strip()
    cf_ray = h.get("cf-ray", "").strip()
    if not cf_ip and not cf_ray:
        return False
    # If cf-connecting-ip exists, also reject the case where it's a
    # loopback (paranoia: some proxies forward 127.0.0.1 from upstream).
    if cf_ip and _is_loopback(cf_ip):
        return False
    return True


# ── Token extraction ──────────────────────────────────────────────────

def _extract_bearer(request: Request) -> str | None:
    """Extract a bearer token from the Authorization header.

    Returns the raw token string, or None if absent/malformed.
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    token = header[len("Bearer "):].strip()
    return token or None


# ── Middleware ────────────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer-token authentication with loopback bypass.

    Parameters
    ----------
    expected_token : str
        The configured token. Empty string disables auth entirely.
    strict : bool
        If True, require the token even for loopback requests.
    public_paths : iterable of str, optional
        Paths that are always reachable without auth (e.g. the splash
        screen polling endpoint). They still pass through the boot gate.
    """

    def __init__(
        self,
        app,
        expected_token: str,
        strict: bool = False,
        public_paths: Iterable[str] = (),
    ) -> None:
        super().__init__(app)
        self.expected_token = expected_token
        self.strict = strict
        self.public_paths = frozenset(public_paths)

    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. No token configured → open mode, let everything through.
        if not self.expected_token:
            return await call_next(request)

        path = request.url.path

        # 2. Static assets and the SPA root never require auth — they
        # contain no sensitive data and the JS itself fetches /api/*
        # endpoints which DO require auth. Letting / through means the
        # browser can load the splash and prompt for the token.
        if not path.startswith("/api/"):
            return await call_next(request)

        # 3. Always-public API paths (typically /api/boot/status, used
        # by the splash screen before the user even has a chance to
        # enter their token).
        if path in self.public_paths:
            return await call_next(request)

        # 4. Cloudflare-tunneled requests ALWAYS require the token.
        if _is_cloudflare_tunneled(request):
            return await self._check_or_reject(request, call_next, "cloudflare")

        # 5. Strict mode requires the token regardless of origin.
        if self.strict:
            return await self._check_or_reject(request, call_next, "strict")

        # 6. Loopback bypass.
        client_host = request.client.host if request.client else None
        if _is_loopback(client_host):
            return await call_next(request)

        # 7. Anything else: require the token.
        return await self._check_or_reject(request, call_next, "external")

    async def _check_or_reject(
        self, request: Request, call_next, reason: str,
    ) -> Response:
        provided = _extract_bearer(request)
        if not provided:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "missing_token",
                    "detail": "Authorization: Bearer <token> required.",
                    "reason": reason,
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Constant-time compare to avoid timing attacks.
        if not secrets.compare_digest(provided, self.expected_token):
            log.warning("Auth rejected (%s) for %s", reason, request.url.path)
            return JSONResponse(
                status_code=401,
                content={
                    "error": "invalid_token",
                    "detail": "Bearer token does not match.",
                    "reason": reason,
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


# ── Banner helper ─────────────────────────────────────────────────────

def auth_banner(token: str, strict: bool) -> str:
    """Return a single-line banner describing the active auth policy.

    Used at startup so the operator immediately sees whether the
    server is open, gated, or strict.
    """
    if not token:
        return (
            "[auth] No LYTHEA_AUTH_TOKEN set. "
            "Server is OPEN — only safe for purely local use."
        )
    if strict:
        return (
            "[auth] Strict mode (LYTHEA_AUTH_STRICT=1). "
            "Token required for ALL requests, including loopback."
        )
    return (
        "[auth] Token configured. "
        "Loopback (127.0.0.1) requests bypass auth. "
        "External and Cloudflare-tunneled requests require Bearer token."
    )
