"""Tavily AI Search provider — designed for LLM/RAG pipelines.

Tavily (https://tavily.com) is a search API built specifically for AI
agents. Its key differentiator vs Brave/Serper/DDG: in addition to a
list of result snippets, Tavily returns a pre-synthesised ``answer``
field — a 2-3 sentence summary of the most relevant findings. This
saves LLM tokens since the model doesn't need to digest 5 raw
snippets to extract the answer.

Why Tavily for Lythéa
---------------------
- Free tier: 1000 requests/month, no credit card required.
- Returns ``answer`` (already-synthesised summary) + ``results``
  (classic snippets). Both are useful for RAG augmentation.
- Latency 1-2s (longer than Brave because it does the synthesis).
- Privacy-respecting (no user profiling).

Configuration
-------------
- ``TAVILY_API_KEY`` env var: required. Get one at https://tavily.com
  — free tier needs only an email.

Failure modes
-------------
- No API key → ``is_available()`` returns False, factory falls back
  to the next provider in the composite chain.
- HTTP error / timeout → empty list (factory falls back).
- 401 (bad key) / 429 (quota exhausted) → logged + empty list.

Format conversion
-----------------
Tavily returns ``{"answer": str, "results": [{"title", "content",
"url", "score", ...}]}``. We:
1. Prepend the ``answer`` as a synthetic first result (title=
   "Synthèse Tavily", body=answer) so the LLM sees it first.
2. Map subsequent results: ``content`` → ``body``, ``url`` → ``href``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from rune.web_providers.base import SearchResult

log = logging.getLogger("lythea.web_providers.tavily")


# Tavily's official search endpoint.
TAVILY_API_URL = "https://api.tavily.com/search"

# HTTP timeout — Tavily synthesises so it's slower (~1.5s typical).
DEFAULT_TIMEOUT_SEC = 12.0


class TavilyProvider:
    """Tavily Search API provider.

    Parameters
    ----------
    api_key : str | None
        Tavily API key. If None, reads from ``TAVILY_API_KEY`` env var
        each call (so .env reloads work transparently).
    timeout : float
        HTTP timeout per request, seconds.
    search_depth : str
        ``"basic"`` (default, ~1s) or ``"advanced"`` (deeper crawl,
        ~3s). Advanced costs 2 credits per call but returns richer
        context — useful for technical / niche queries.
    include_answer : bool
        If True (default), Tavily synthesises a 2-3 sentence answer
        from the top results and includes it as a synthetic first
        ``SearchResult``. Disable for raw snippets only.
    max_results : int
        Default upper bound — overridden per call by ``search()``.
    """

    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        search_depth: str = "basic",
        include_answer: bool = True,
    ) -> None:
        self._api_key_override = (api_key or "").strip() or None
        self.timeout = timeout
        self.search_depth = (
            search_depth if search_depth in ("basic", "advanced") else "basic"
        )
        self.include_answer = bool(include_answer)

    # ── Public API ──────────────────────────────────────────────────

    def _resolve_api_key(self) -> str | None:
        """Read the API key fresh each call so .env reloads work.

        Priority: constructor override → env var. Empty string treated
        as None.
        """
        if self._api_key_override:
            return self._api_key_override
        key = (os.getenv("TAVILY_API_KEY", "") or "").strip()
        return key or None

    def is_available(self) -> bool:
        """True if a Tavily API key is configured.

        We don't probe the network here — actual auth / quota errors
        surface at search time and fall through to the next backend.
        """
        return self._resolve_api_key() is not None

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """One round-trip to Tavily's search endpoint.

        Returns
        -------
        list[SearchResult]
            Empty list on any failure (missing key, HTTP error,
            malformed response, quota exhausted). The orchestrator
            then falls through to the next provider.
        """
        if not query or not query.strip():
            return []
        api_key = self._resolve_api_key()
        if not api_key:
            log.debug("TAVILY_API_KEY missing — skipping Tavily provider")
            return []

        # Tavily is happier with shorter queries (it does NLP rewriting).
        query = query.strip()[:400]
        max_n = max(1, min(20, int(max_results or 5)))

        body = {
            "api_key": api_key,
            "query": query,
            "search_depth": self.search_depth,
            "include_answer": self.include_answer,
            "max_results": max_n,
            # We don't need images / domains / raw_content for now.
            "include_images": False,
            "include_raw_content": False,
        }
        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            TAVILY_API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Lythea/4.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    log.warning(
                        "Tavily HTTP %s for query=%r",
                        resp.status, query[:60],
                    )
                    return []
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # 401 = bad key, 402 = payment required, 429 = rate-limit.
            log.warning(
                "Tavily HTTPError %s for query=%r (%s)",
                exc.code, query[:60], exc.reason,
            )
            return []
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.debug("Tavily network error: %s", exc)
            return []
        except Exception:
            log.warning("Tavily unexpected error", exc_info=True)
            return []

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            log.warning("Tavily returned malformed JSON")
            return []

        return self._extract_results(payload, max_n)

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _extract_results(
        payload: dict,
        max_results: int,
    ) -> list[SearchResult]:
        """Convert Tavily payload to canonical SearchResult dicts.

        Structure expected:

            {
              "query": "...",
              "answer": "Synthèse en 2-3 phrases.",  # optional
              "results": [
                {
                  "title": str,
                  "url": str,
                  "content": str,    # snippet
                  "score": float,    # relevance score
                  ...
                }
              ]
            }

        We synthesise a first result from ``answer`` so the LLM sees
        the digest before the raw snippets. This is the main value-
        add of Tavily over Brave/Serper.
        """
        out: list[SearchResult] = []

        if not isinstance(payload, dict):
            return out

        # 1. Synthesised answer (Tavily's killer feature).
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            out.append({
                "title": "Synthèse",
                "body": answer.strip(),
                "href": "",
            })

        # 2. Raw results.
        raw_results = payload.get("results") or []
        if not isinstance(raw_results, list):
            return out

        for r in raw_results:
            if not isinstance(r, dict):
                continue
            title = r.get("title", "") or ""
            body = r.get("content", "") or ""
            href = r.get("url", "") or ""
            if not (title or body):
                continue
            out.append({
                "title": title,
                "body": body,
                "href": href,
            })
            if len(out) >= max_results + 1:  # +1 for the answer header
                break
        return out
