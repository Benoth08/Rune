"""Serper Google Search API provider — high-volume Google access.

Serper (https://serper.dev) is a managed proxy in front of Google
Search. It returns Google's native JSON (organic results, knowledge
graph, answer box, news vertical) with low latency and a generous
free tier.

Why Serper for Lythéa
---------------------
- Free tier: 2500 requests/month (~83/day), no credit card required.
- Latency ~300ms (lowest of any provider in the chain).
- Quality identical to google.com (it IS Google, just proxied).
- Returns ``knowledgeGraph`` (Wikipedia summary, key facts) and
  ``answerBox`` (featured snippets) when available — both are RAG gold.
- Stable JSON schema since 2023.

Configuration
-------------
- ``SERPER_API_KEY`` env var: required. Get one at https://serper.dev
  — free tier needs only an email.
- ``SERPER_COUNTRY`` env var: optional, ISO country code (default "fr").
- ``SERPER_LANG`` env var: optional, language hint (default "fr").

Failure modes
-------------
- No API key → ``is_available()`` returns False, factory falls back.
- HTTP error / timeout → empty list.
- 401 / 403 / 429 → logged at warning level + empty list.

Format conversion
-----------------
Serper returns multiple top-level keys (``organic``, ``answerBox``,
``knowledgeGraph``, ``peopleAlsoAsk``, ``relatedSearches``). We:
1. Promote ``knowledgeGraph`` and ``answerBox`` as the first results
   (they're typically the most informative).
2. Append ``organic`` results in order.
3. Map ``link`` → ``href``, ``snippet`` → ``body``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from rune.web_providers.base import SearchResult

log = logging.getLogger("lythea.web_providers.serper")


# Serper's official search endpoint.
SERPER_API_URL = "https://google.serper.dev/search"

# HTTP timeout — Serper is fast (~300ms) so 6s is plenty.
DEFAULT_TIMEOUT_SEC = 6.0


class SerperProvider:
    """Serper Google Search API provider.

    Parameters
    ----------
    api_key : str | None
        Serper API key. If None, reads from ``SERPER_API_KEY`` env var
        each call (so .env reloads work transparently).
    timeout : float
        HTTP timeout per request, seconds.
    country : str
        ISO 3166 country code (``gl`` param). Default "fr".
    language : str
        UI language code (``hl`` param). Default "fr".
    """

    name = "serper"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        country: str = "fr",
        language: str = "fr",
    ) -> None:
        self._api_key_override = (api_key or "").strip() or None
        self.timeout = timeout
        self.country = country.lower()
        self.language = language.lower()

    # ── Public API ──────────────────────────────────────────────────

    def _resolve_api_key(self) -> str | None:
        """Read the API key fresh each call so .env reloads work."""
        if self._api_key_override:
            return self._api_key_override
        key = (os.getenv("SERPER_API_KEY", "") or "").strip()
        return key or None

    def is_available(self) -> bool:
        """True if a Serper API key is configured."""
        return self._resolve_api_key() is not None

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """One round-trip to Serper's search endpoint.

        Returns
        -------
        list[SearchResult]
            Empty list on any failure. The orchestrator falls through.
        """
        if not query or not query.strip():
            return []
        api_key = self._resolve_api_key()
        if not api_key:
            log.debug("SERPER_API_KEY missing — skipping Serper provider")
            return []

        query = query.strip()[:400]
        max_n = max(1, min(20, int(max_results or 5)))

        body = {
            "q": query,
            "num": max_n,
            "gl": self.country,
            "hl": self.language,
        }
        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            SERPER_API_URL,
            data=data,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Lythea/4.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    log.warning(
                        "Serper HTTP %s for query=%r",
                        resp.status, query[:60],
                    )
                    return []
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # 401 = bad key, 402 = payment required, 429 = quota.
            log.warning(
                "Serper HTTPError %s for query=%r (%s)",
                exc.code, query[:60], exc.reason,
            )
            return []
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.debug("Serper network error: %s", exc)
            return []
        except Exception:
            log.warning("Serper unexpected error", exc_info=True)
            return []

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            log.warning("Serper returned malformed JSON")
            return []

        return self._extract_results(payload, max_n)

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _extract_results(
        payload: dict,
        max_results: int,
    ) -> list[SearchResult]:
        """Convert Serper payload to canonical SearchResult dicts.

        Promotion order:
        1. ``knowledgeGraph`` (Wikipedia-style summary, when present).
        2. ``answerBox`` (featured snippet at top of Google results).
        3. ``organic`` (the 10 blue links).

        Each is mapped to ``{title, body, href}``.
        """
        out: list[SearchResult] = []

        if not isinstance(payload, dict):
            return out

        # 1. Knowledge graph — most condensed info.
        kg = payload.get("knowledgeGraph")
        if isinstance(kg, dict):
            title = kg.get("title", "") or ""
            desc = kg.get("description", "") or ""
            # Include type / attributes if present (e.g. "Personne née en …").
            type_ = kg.get("type", "") or ""
            if type_:
                desc = f"({type_}) {desc}" if desc else f"({type_})"
            href = kg.get("descriptionLink", "") or kg.get("website", "") or ""
            if title or desc:
                out.append({
                    "title": title or "Knowledge Graph",
                    "body": desc,
                    "href": href,
                })

        # 2. Answer box — featured snippet.
        ab = payload.get("answerBox")
        if isinstance(ab, dict):
            title = ab.get("title", "") or ab.get("source", "") or "Answer"
            body = (
                ab.get("answer", "")
                or ab.get("snippet", "")
                or ab.get("snippetHighlighted", "")
                or ""
            )
            href = ab.get("link", "") or ""
            if title or body:
                out.append({
                    "title": str(title),
                    "body": str(body),
                    "href": str(href),
                })

        # 3. Organic results.
        organic = payload.get("organic") or []
        if isinstance(organic, list):
            for r in organic:
                if not isinstance(r, dict):
                    continue
                title = r.get("title", "") or ""
                body = r.get("snippet", "") or ""
                href = r.get("link", "") or ""
                if not (title or body):
                    continue
                # Include date hint when present (recency signal).
                date = r.get("date", "") or ""
                if date:
                    body = f"[{date}] {body}" if body else f"[{date}]"
                out.append({
                    "title": title,
                    "body": body,
                    "href": href,
                })
                if len(out) >= max_results + 2:  # +2 for KG / AB headers
                    break
        return out
