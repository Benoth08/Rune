"""V4.0.3 — Pluggable web search providers.

Architecture
------------
Lythéa historically called DuckDuckGo's unofficial library directly.
This package abstracts that behind a :class:`WebSearchProvider`
Protocol so we can plug richer / more reliable backends without
touching the orchestration code (``lythea/web.py``).

Available providers
-------------------
- **SearXNG** (primary) — open-source meta search engine that
  aggregates Google / Bing / Brave / Wikipedia / …. No API key.
  Either points at a public instance or your own self-hosted one.
- **DuckDuckGo** (fallback) — kept as fallback when SearXNG is
  unreachable. Same library as before (``ddgs``).

Output format
-------------
All providers return ``list[dict]`` with keys ``{title, body, href}``,
matching the legacy DDG output so the rest of Lythéa (web.py,
hippocampe.py, RAG injection) is unchanged.

Selection
---------
``get_provider(name)`` returns a provider. ``get_default_provider()``
returns a composite that tries SearXNG first and falls back to DDG.
Configuration via env vars in ``lythea/config.py`` / ``.env``:

- ``WEB_SEARCH_PROVIDER`` : ``auto`` (default) | ``searxng`` | ``ddg``
- ``SEARXNG_INSTANCE_URL`` : URL of the SearXNG instance to use
  (default: a hardcoded list of known-good public instances is
  tried in order).
"""

from __future__ import annotations

from rune.web_providers.base import WebSearchProvider, SearchResult
from rune.web_providers.factory import (
    get_default_provider,
    get_provider,
    list_providers,
)

__all__ = [
    "WebSearchProvider",
    "SearchResult",
    "get_default_provider",
    "get_provider",
    "list_providers",
]
