"""Protocol defining a web search provider.

Any object implementing :meth:`search` with the right signature can
be used as a Lythéa web search backend. The return type is a list of
plain dicts (not dataclasses) to match what the rest of Lythéa
already expects.
"""

from __future__ import annotations

from typing import Protocol, TypedDict


class SearchResult(TypedDict, total=False):
    """One search hit. ``total=False`` because some providers omit
    fields (e.g. DDG occasionally returns no body for image results)."""

    title: str
    body: str
    href: str


class WebSearchProvider(Protocol):
    """Plug-in for the web search agent.

    Implementations must be **stateless** at call boundaries (cache OK,
    but no per-request state that would leak between calls). They
    should also be **safe by default**: never raise — return an empty
    list on any failure and log the reason.
    """

    name: str
    """Human-readable identifier (also used in logs and config)."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Perform a single search round.

        Parameters
        ----------
        query : str
            Search query as built by ``WebTriggerPolicy.build_search_query``.
        max_results : int, default 5
            Upper bound on results returned. Providers may return fewer.

        Returns
        -------
        list[SearchResult]
            Each dict has at minimum ``title`` and ``body`` keys. ``href``
            is included when the provider supplies it.
        """
        ...

    def is_available(self) -> bool:
        """Quick availability probe.

        Used by composite providers to decide whether to attempt a
        call or skip to a fallback. Should be cheap (no network round
        trip after first call — cache the result).
        """
        ...
