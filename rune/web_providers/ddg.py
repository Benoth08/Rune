"""DuckDuckGo provider — kept as fallback when SearXNG is unreachable.

This is a thin wrapper around the ``ddgs`` library that Lythéa used
prior to V4.0.3. The library scrapes DuckDuckGo's HTML, which is
fragile and rate-limited, so we prefer SearXNG when available. But
DDG remains a useful no-config fallback for offline / air-gapped
deployments where SearXNG isn't reachable either.
"""

from __future__ import annotations

import logging

from rune.web_providers.base import SearchResult

log = logging.getLogger("lythea.web_providers.ddg")


class DdgProvider:
    """DuckDuckGo fallback provider via the ``ddgs`` library."""

    name = "ddg"

    def __init__(self) -> None:
        self._ddg = None
        self._availability_checked = False
        self._available = False

    def is_available(self) -> bool:
        if self._availability_checked:
            return self._available
        try:
            from ddgs import DDGS  # noqa: F401

            self._available = True
        except ImportError:
            log.warning(
                "ddgs library missing — pip install ddgs to enable the "
                "DuckDuckGo fallback (SearXNG remains available)."
            )
            self._available = False
        self._availability_checked = True
        return self._available

    def _ensure_client(self) -> bool:
        if self._ddg is not None:
            return True
        if not self.is_available():
            return False
        try:
            from ddgs import DDGS

            self._ddg = DDGS()
            return True
        except Exception:
            log.warning("DDGS init failed", exc_info=True)
            return False

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if not self._ensure_client():
            return []
        try:
            results = list(self._ddg.text(query, max_results=max_results))
        except Exception as exc:
            log.warning("DDG search failed: %s", exc)
            return []
        # ddgs returns dicts already in {title, body, href} format —
        # just defensive-copy to typed dicts.
        out: list[SearchResult] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            out.append({
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "href": r.get("href", ""),
            })
        return out
