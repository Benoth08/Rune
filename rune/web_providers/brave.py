"""Brave Search API provider — high-quality independent search.

Brave Search is an independent search index (not Google/Bing rebrand)
with a clean JSON API. Free tier: 2000 queries/month, no credit card.

Sign up & API key
-----------------
1. https://brave.com/search/api/
2. Create a "Data for Search" subscription (free tier)
3. Copy the API key
4. Set ``BRAVE_API_KEY`` env var (or in ``.env``)

This provider is added at the top of the default composite chain so
Brave answers first when available. If the key is missing or quota is
exhausted, the chain transparently falls back to SearXNG then DDG.

Configuration via env vars
--------------------------
- ``BRAVE_API_KEY`` (required to enable the provider)
- ``BRAVE_SEARCH_COUNTRY`` (optional, default "FR")
- ``BRAVE_SEARCH_LANG`` (optional, default "fr")
- ``BRAVE_SAFESEARCH`` (optional: "off", "moderate", "strict"; default "off")

Format conversion
-----------------
Brave returns ``{"web": {"results": [{"title", "description", "url", ...}]}}``.
We map ``description`` → ``body`` and ``url`` → ``href`` to match the
canonical SearchResult format.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from rune.web_providers.base import SearchResult

log = logging.getLogger("rune.web_providers.brave")


# Brave's official API endpoint.
BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

# HTTP timeout — Brave is fast (~500ms typically), so keep conservative.
DEFAULT_TIMEOUT_SEC = 8.0


class BraveProvider:
    """Brave Search API provider.

    Parameters
    ----------
    api_key : str | None
        Brave API key. If None, reads from ``BRAVE_API_KEY`` env var
        each call (so .env reloads work transparently).
    timeout : float
        HTTP timeout per request.
    country : str
        ISO 3166-1 alpha-2 country code biasing results. Default "FR".
    language : str
        UI language code biasing snippets/dates. Default "fr".
    safesearch : str
        "off", "moderate", or "strict". Default "off".
    """

    name = "brave"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        country: str = "FR",
        language: str = "fr",
        safesearch: str = "off",
    ) -> None:
        self._api_key_override = (api_key or "").strip() or None
        self.timeout = timeout
        self.country = country
        self.language = language
        self.safesearch = safesearch if safesearch in (
            "off", "moderate", "strict"
        ) else "off"
        self._availability_checked: bool = False
        self._available: bool = False

    # ── Public API ──────────────────────────────────────────────────

    def _resolve_api_key(self) -> str | None:
        """Read the API key fresh each call so .env reloads work.

        Priority: constructor override → env var. Empty string treated
        as None.
        """
        if self._api_key_override:
            return self._api_key_override
        key = (os.getenv("BRAVE_API_KEY", "") or "").strip()
        return key or None

    def is_available(self) -> bool:
        """True if a Brave API key is configured.

        We don't probe the network here — the actual quota / endpoint
        check happens at search time, where failure falls through to
        the next provider in the chain.
        """
        # Cache the boolean availability but re-check the key existence
        # each call to support runtime .env edits.
        return self._resolve_api_key() is not None

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """One round-trip to Brave's web search endpoint.

        Returns
        -------
        list[SearchResult]
            Empty list on any failure (missing key, HTTP error, malformed
            response, quota exhausted). The orchestrator then falls
            through to SearXNG / DDG.
        """
        if not query or not query.strip():
            return []
        api_key = self._resolve_api_key()
        if not api_key:
            log.debug("BRAVE_API_KEY missing — skipping Brave provider")
            return []

        # Brave caps queries at 400 chars; be defensive.
        query = query.strip()[:380]
        params = {
            "q": query,
            "count": str(max(1, min(20, int(max_results)))),
            "country": self.country,
            "search_lang": self.language,
            "safesearch": self.safesearch,
            # Exclude features that bloat the response and we don't use.
            "result_filter": "web",
            # Spell-check disabled — our triggers already build the query.
            "spellcheck": "0",
        }
        full_url = f"{BRAVE_API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            full_url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",  # avoid gzip decode issues
                "X-Subscription-Token": api_key,
                "User-Agent": "Lythea/4.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    log.warning(
                        "Brave HTTP %s for query=%r", resp.status, query[:60],
                    )
                    return []
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # 401 = bad key, 422 = invalid params, 429 = quota.
            # All non-fatal — let the chain fall through.
            log.warning(
                "Brave HTTPError %s for query=%r (%s)",
                exc.code, query[:60], exc.reason,
            )
            return []
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.debug("Brave network error: %s", exc)
            return []
        except Exception:
            log.warning("Brave unexpected error", exc_info=True)
            return []

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            log.warning("Brave returned malformed JSON")
            return []

        web = payload.get("web") if isinstance(payload, dict) else None
        raw_results = (
            web.get("results") if isinstance(web, dict) else None
        ) or []

        out: list[SearchResult] = []
        for r in raw_results[:max_results]:
            if not isinstance(r, dict):
                continue
            title = r.get("title", "") or ""
            # Brave uses 'description' for the snippet; some entries
            # have 'extra_snippets' (a list) which we concatenate.
            body = r.get("description", "") or ""
            extras = r.get("extra_snippets")
            if isinstance(extras, list) and extras:
                body = (body + " " + " ".join(
                    str(e) for e in extras if e
                )).strip()
            href = r.get("url", "") or ""
            if not (title or body):
                continue
            out.append({
                "title": title,
                "body": body,
                "href": href,
            })
        return out
