"""Rate limiting via slowapi — IP-based with loopback bypass.

We rate-limit only the resource-intensive routes:
- ``/api/chat`` — heavy LLM inference, abuse can DOS the GPU
- ``/api/models/load`` — multi-GB downloads, costly on bandwidth

Bypass strategy
---------------
Different versions of slowapi pass different signatures to
``exempt_when``: some call ``exempt_when()`` (no args), others call
``exempt_when(request)``. To be robust across versions, we don't rely
on ``exempt_when`` at all. Instead, the bypass logic lives in
:func:`_key_func` which uses a context variable to track the current
request, and returns ``None`` for loopback requests. slowapi treats
``None`` keys as exempt from the limit.

Cloudflare-tunneled requests use the real client IP from
``cf-connecting-ip``, not the loopback we'd see if we read
``request.client.host`` naively.
"""
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from rune.server.auth import _is_loopback

log = logging.getLogger("lythea.server.rate_limit")


def _get_client_ip(request: Request) -> str:
    """Return the best-effort client IP for rate-limit keying.

    Priority:
    1. ``cf-connecting-ip`` if present (Cloudflare tunnel)
    2. ``x-forwarded-for`` first hop (other reverse proxies)
    3. ``request.client.host`` (direct connection)
    """
    h = request.headers
    cf_ip = h.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip
    xff = h.get("x-forwarded-for", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_request_exempt(request: Request) -> bool:
    """Check if a request should bypass rate limiting.

    A request is exempt when it originates from loopback and has no
    Cloudflare forwarding headers. Cloudflare-tunneled requests are
    NOT exempt even if their TCP origin is 127.0.0.1, because that's
    just the tunnel relaying the external client.
    """
    cf_ip = request.headers.get("cf-connecting-ip", "").strip()
    if cf_ip and not _is_loopback(cf_ip):
        return False  # tunneled external request — apply the limit
    client_host = request.client.host if request.client else None
    return _is_loopback(client_host)


def _key_func(request: Request) -> str:
    """Key function for slowapi.

    For exempt (loopback) requests we return a dedicated bucket name
    that we'll never apply a limit to. The actual exemption is enforced
    at the decorator level by checking the bucket name there.

    Returning a unique-per-request key would also work but pollutes
    Redis-style storage if slowapi is ever migrated there. The shared
    bucket "loopback" is fine because it's never hit with a real limit.
    """
    if _is_request_exempt(request):
        return "loopback-exempt"
    return _get_client_ip(request)


# Global limiter — initialised at app creation in app.py
limiter = Limiter(
    key_func=_key_func,
    default_limits=[],  # we apply per-route, not globally
    headers_enabled=True,  # add X-RateLimit-* headers
)


async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded,
) -> JSONResponse:
    """Custom 429 response with a friendly French message."""
    log.warning(
        "Rate limit exceeded on %s by %s",
        request.url.path, _get_client_ip(request),
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": (
                "Trop de requêtes. Patiente un instant avant de réessayer."
            ),
            "limit": str(exc.detail),
        },
    )


def make_limit_decorator(rate: str):
    """Build a rate-limit decorator that respects loopback exemption.

    We don't pass ``exempt_when`` to ``limiter.limit`` because slowapi
    versions disagree on whether the callback receives the request or
    nothing. Instead, we wrap the decorator so we can call
    :func:`_is_request_exempt` ourselves with the request in scope.

    This works because FastAPI passes a ``Request`` object to every
    rate-limited endpoint (Lythéa's signature requires it), and slowapi
    extracts that request before applying the limit. We intercept by
    checking the limiter's key — if it's the loopback bucket, we skip
    the limit decorator entirely.
    """
    inner = limiter.limit(rate)

    def decorator(func):
        # Wrap with the slowapi limiter, then add our own pre-check
        # that short-circuits for exempt requests. We use functools.wraps
        # to preserve FastAPI's introspection of the endpoint signature.
        import functools
        wrapped = inner(func)

        @functools.wraps(func)
        async def gate(*args, **kwargs):
            # FastAPI passes the Request as either a positional arg or
            # the "request" kwarg depending on signature. Find it.
            request = kwargs.get("request")
            if request is None:
                for a in args:
                    if isinstance(a, Request):
                        request = a
                        break
            if request is not None and _is_request_exempt(request):
                # Skip slowapi entirely — call the original endpoint
                return await func(*args, **kwargs)
            # Subject to the limit
            return await wrapped(*args, **kwargs)

        return gate

    return decorator
