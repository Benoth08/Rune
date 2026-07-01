"""Composite provider + factory.

The composite tries providers in order, returning the first non-empty
result set. This gives transparent failover.

Default chain (V4.0.3+): **Tavily → Serper → Brave → SearXNG → DDG**

- **Tavily** first when ``TAVILY_API_KEY`` set: LLM-optimised search
  with synthesised answer (1000 req/mois free).
- **Serper** next when ``SERPER_API_KEY`` set: Google Search API
  (2500 req/mois free, fastest).
- **Brave** when ``BRAVE_API_KEY`` set: independent index (kept for
  backward-compat, free tier was removed late 2026 but paid plans
  still work).
- **SearXNG** open-source meta-search aggregator (public instances).
- **DDG** legacy fallback (DuckDuckGo Instant Answer, limited
  quality but always available).

Each provider whose ``is_available()`` returns False is silently
skipped — no error, no log spam.

Public entry points:
- :func:`get_default_provider` — returns the composite chain.
- :func:`get_provider` — returns a single provider by name.
- :func:`list_providers` — registry introspection.
"""

from __future__ import annotations

import logging
import os

from rune.web_providers.base import SearchResult, WebSearchProvider
from rune.web_providers.brave import BraveProvider
from rune.web_providers.ddg import DdgProvider
from rune.web_providers.searxng import SearxngProvider
from rune.web_providers.serper import SerperProvider
from rune.web_providers.tavily import TavilyProvider

log = logging.getLogger("lythea.web_providers.factory")


_REGISTRY: dict[str, type] = {
    "tavily": TavilyProvider,
    "serper": SerperProvider,
    "brave": BraveProvider,
    "searxng": SearxngProvider,
    "ddg": DdgProvider,
}


class CompositeProvider:
    """Try providers in order, return the first non-empty result set.

    Logs which provider answered so users can audit reliability.
    """

    name = "composite"

    def __init__(self, providers: list[WebSearchProvider]) -> None:
        if not providers:
            raise ValueError("composite requires at least one provider")
        self.providers = providers

    def is_available(self) -> bool:
        return any(p.is_available() for p in self.providers)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        for p in self.providers:
            if not p.is_available():
                continue
            try:
                results = p.search(query, max_results=max_results)
            except Exception:
                log.warning(
                    "Provider %s raised during search", p.name,
                    exc_info=True,
                )
                continue
            if results:
                log.debug(
                    "web search answered by %s (%d results)",
                    p.name, len(results),
                )
                return results
        log.info("All web providers returned empty for query=%r", query[:60])
        return []


def list_providers() -> list[str]:
    """Return the names of all known single providers."""
    return sorted(_REGISTRY.keys())


def get_provider(name: str) -> WebSearchProvider:
    """Instantiate a single provider by name. Raises KeyError if unknown."""
    cls = _REGISTRY[name]
    return cls()


def get_default_provider() -> WebSearchProvider:
    """Return the configured default — usually the composite chain.

    Configuration via env vars (read each call so .env reloads work):

    - ``WEB_SEARCH_PROVIDER`` :
        - ``auto`` (default) → composite
          Tavily → Serper → Brave → SearXNG → DDG
        - ``tavily`` → Tavily only (requires ``TAVILY_API_KEY``)
        - ``serper`` → Serper only (requires ``SERPER_API_KEY``)
        - ``brave`` → Brave only (requires ``BRAVE_API_KEY``)
        - ``searxng`` → SearXNG only
        - ``ddg`` → DDG only
    - ``TAVILY_API_KEY`` : enable Tavily provider (free 1000/mois at
      https://tavily.com).
    - ``SERPER_API_KEY`` : enable Serper provider (free 2500/mois at
      https://serper.dev).
    - ``BRAVE_API_KEY`` : enable Brave provider (paid only since late
      2026 at https://brave.com/search/api/).
    - ``SEARXNG_INSTANCE_URL`` : override SearXNG instance URL.

    Each provider whose ``is_available()`` returns False is silently
    skipped — the chain still works with whatever providers ARE
    available.

    Var names: ``LYTHEA_WEB_PROVIDER`` and ``LYTHEA_SEARXNG_INSTANCE_URL``
    are also accepted (preferred — the ``LYTHEA_`` prefix matches the
    rest of the project's env conventions).
    """
    # Accept both prefixed and unprefixed names. LYTHEA_* takes
    # precedence since it matches the project's naming convention,
    # but bare names remain supported for back-compat with older
    # deployments and 3rd-party tooling.
    choice = (
        os.getenv("LYTHEA_WEB_PROVIDER")
        or os.getenv("WEB_SEARCH_PROVIDER")
        or "auto"
    ).strip().lower()
    instance_url = (
        os.getenv("LYTHEA_SEARXNG_INSTANCE_URL")
        or os.getenv("SEARXNG_INSTANCE_URL")
        or ""
    ).strip() or None

    if choice == "tavily":
        return TavilyProvider()
    if choice == "serper":
        return SerperProvider()
    if choice == "brave":
        return BraveProvider()
    if choice == "searxng":
        return SearxngProvider(instance_url=instance_url)
    if choice == "ddg":
        return DdgProvider()
    # auto / unknown → full composite chain.
    return CompositeProvider([
        TavilyProvider(),
        SerperProvider(),
        BraveProvider(),
        SearxngProvider(instance_url=instance_url),
        DdgProvider(),
    ])
