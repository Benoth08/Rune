"""Tests for the pluggable web search providers.

Network is mocked everywhere — these tests run offline.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from rune.web_providers import (
    get_default_provider,
    get_provider,
    list_providers,
)
from rune.web_providers.base import WebSearchProvider
from rune.web_providers.ddg import DdgProvider
from rune.web_providers.factory import CompositeProvider
from rune.web_providers.searxng import SearxngProvider


# ════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════


def test_list_providers_returns_at_least_searxng_and_ddg():
    names = list_providers()
    assert "searxng" in names
    assert "ddg" in names


def test_get_provider_by_name():
    p = get_provider("searxng")
    assert isinstance(p, SearxngProvider)
    assert p.name == "searxng"


def test_get_provider_unknown_raises():
    with pytest.raises(KeyError):
        get_provider("unknown_provider")


def test_get_default_provider_returns_composite_by_default(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    p = get_default_provider()
    assert isinstance(p, CompositeProvider)


def test_get_default_provider_searxng_only_when_configured(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "searxng")
    p = get_default_provider()
    assert isinstance(p, SearxngProvider)


def test_get_default_provider_ddg_only_when_configured(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "ddg")
    p = get_default_provider()
    assert isinstance(p, DdgProvider)


def test_get_default_provider_unknown_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "bogus")
    p = get_default_provider()
    assert isinstance(p, CompositeProvider)


# ════════════════════════════════════════════════════════════════════
# SearXNG provider
# ════════════════════════════════════════════════════════════════════


def test_searxng_empty_query_returns_empty():
    p = SearxngProvider(instance_url="https://example.org")
    assert p.search("") == []
    assert p.search("   ") == []


def test_searxng_uses_configured_instance_only(monkeypatch):
    """When instance_url is set, pool size is 1."""
    p = SearxngProvider(instance_url="https://my-instance.org/")
    # Trailing slash stripped
    assert p.instance_url == "https://my-instance.org"
    assert p._instances_pool == ["https://my-instance.org"]


def test_searxng_falls_through_to_next_instance_on_failure():
    """If instance #1 raises, try instance #2."""
    p = SearxngProvider()
    p._instances_pool = ["https://bad", "https://good"]
    # Pin the order — search() shuffles by default for load balancing
    # across public instances. The test wants deterministic order.
    p.instance_url = "https://bad"  # disables shuffle path

    call_count = {"n": 0}

    def fake_search_one(instance_url, query, max_results):
        call_count["n"] += 1
        if instance_url == "https://bad":
            raise RuntimeError("HTTP 503")
        return [{"title": "ok", "body": "from good", "href": "https://x"}]

    with patch.object(p, "_search_one", side_effect=fake_search_one):
        results = p.search("test query")
    assert len(results) == 1
    assert results[0]["body"] == "from good"
    assert call_count["n"] == 2


def test_searxng_caches_known_good_instance():
    p = SearxngProvider()
    p._instances_pool = ["https://a", "https://b"]

    def fake_search_one(instance_url, query, max_results):
        if instance_url == "https://a":
            raise RuntimeError("503")
        return [{"title": "t", "body": "b", "href": "h"}]

    with patch.object(p, "_search_one", side_effect=fake_search_one):
        p.search("q1")
    # After one successful search via b, _known_good should be b
    assert p._known_good == "https://b"


def test_searxng_returns_empty_when_all_instances_fail():
    p = SearxngProvider()
    p._instances_pool = ["https://a", "https://b"]
    with patch.object(p, "_search_one", side_effect=RuntimeError("fail")):
        assert p.search("q") == []


def test_searxng_parses_response_format():
    """Mock the HTTP call and verify mapping content→body, url→href."""
    p = SearxngProvider(instance_url="https://test")

    fake_payload = json.dumps({
        "results": [
            {"title": "T1", "content": "C1", "url": "U1"},
            {"title": "T2", "content": "C2", "url": "U2"},
        ]
    }).encode("utf-8")

    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = fake_payload
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda self, *a: None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        results = p.search("test")

    assert len(results) == 2
    assert results[0] == {"title": "T1", "body": "C1", "href": "U1"}
    assert results[1] == {"title": "T2", "body": "C2", "href": "U2"}


def test_searxng_drops_empty_results():
    """Entries with no title AND no body are filtered out."""
    p = SearxngProvider(instance_url="https://test")

    fake_payload = json.dumps({
        "results": [
            {"title": "good", "content": "body", "url": "u"},
            {"title": "", "content": "", "url": "u"},  # both empty → drop
        ]
    }).encode("utf-8")

    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = fake_payload
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda self, *a: None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        results = p.search("test")

    assert len(results) == 1
    assert results[0]["title"] == "good"


def test_searxng_is_available_when_pool_non_empty():
    p = SearxngProvider(instance_url="https://x")
    assert p.is_available() is True


# ════════════════════════════════════════════════════════════════════
# DDG provider
# ════════════════════════════════════════════════════════════════════


def test_ddg_handles_missing_library(monkeypatch):
    p = DdgProvider()

    def fake_import(name, *args, **kwargs):
        if name == "ddgs":
            raise ImportError("not installed")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        assert p.is_available() is False
        assert p.search("test") == []


def test_ddg_empty_query_returns_empty():
    p = DdgProvider()
    assert p.search("") == []


# ════════════════════════════════════════════════════════════════════
# Composite provider
# ════════════════════════════════════════════════════════════════════


class _FakeProvider:
    def __init__(self, name, available=True, results=None, raises=False):
        self.name = name
        self._available = available
        self._results = results or []
        self._raises = raises
        self.calls = 0

    def is_available(self):
        return self._available

    def search(self, query, max_results=5):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return list(self._results)


def test_composite_uses_first_available():
    a = _FakeProvider("a", results=[{"title": "A", "body": "from a", "href": ""}])
    b = _FakeProvider("b", results=[{"title": "B", "body": "from b", "href": ""}])
    comp = CompositeProvider([a, b])
    results = comp.search("test")
    assert results[0]["body"] == "from a"
    assert a.calls == 1
    assert b.calls == 0  # not called — a succeeded


def test_composite_falls_back_when_first_empty():
    a = _FakeProvider("a", results=[])
    b = _FakeProvider("b", results=[{"title": "B", "body": "from b", "href": ""}])
    comp = CompositeProvider([a, b])
    results = comp.search("test")
    assert results[0]["body"] == "from b"
    assert a.calls == 1
    assert b.calls == 1


def test_composite_skips_unavailable():
    a = _FakeProvider("a", available=False, results=[{"title": "X", "body": "", "href": ""}])
    b = _FakeProvider("b", results=[{"title": "B", "body": "from b", "href": ""}])
    comp = CompositeProvider([a, b])
    results = comp.search("test")
    assert results[0]["body"] == "from b"
    assert a.calls == 0  # skipped
    assert b.calls == 1


def test_composite_swallows_exceptions():
    a = _FakeProvider("a", raises=True)
    b = _FakeProvider("b", results=[{"title": "B", "body": "from b", "href": ""}])
    comp = CompositeProvider([a, b])
    results = comp.search("test")
    assert results[0]["body"] == "from b"


def test_composite_all_empty_returns_empty():
    a = _FakeProvider("a", results=[])
    b = _FakeProvider("b", results=[])
    comp = CompositeProvider([a, b])
    assert comp.search("test") == []


def test_composite_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        CompositeProvider([])


# ════════════════════════════════════════════════════════════════════
# Integration with WebAgent
# ════════════════════════════════════════════════════════════════════


def test_webagent_uses_injected_provider():
    """A WebAgent with a custom provider doesn't touch DDG."""
    from rune.web import WebAgent

    fake = _FakeProvider("test", results=[
        {"title": "T", "body": "body", "href": "h"},
    ])
    agent = WebAgent(provider=fake)
    results = agent.search("query")
    assert len(results) == 1
    assert fake.calls == 1


def test_webagent_handles_provider_failure():
    from rune.web import WebAgent

    fake = _FakeProvider("test", raises=True)
    agent = WebAgent(provider=fake)
    # Should not raise — returns empty list.
    assert agent.search("query") == []


# ════════════════════════════════════════════════════════════════════
# Brave provider (V4.0.3+)
# ════════════════════════════════════════════════════════════════════


def test_brave_unavailable_without_key(monkeypatch):
    from rune.web_providers.brave import BraveProvider

    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    p = BraveProvider()
    assert p.is_available() is False
    # Should also return [] from search() — no network call attempted.
    assert p.search("test") == []


def test_brave_available_with_key(monkeypatch):
    from rune.web_providers.brave import BraveProvider

    monkeypatch.setenv("BRAVE_API_KEY", "fake-key-12345")
    p = BraveProvider()
    assert p.is_available() is True


def test_brave_constructor_override_beats_env(monkeypatch):
    from rune.web_providers.brave import BraveProvider

    monkeypatch.setenv("BRAVE_API_KEY", "env-key")
    p = BraveProvider(api_key="ctor-key")
    # is_available reads from internal override, not env
    assert p._resolve_api_key() == "ctor-key"


def test_brave_safesearch_validation():
    from rune.web_providers.brave import BraveProvider

    # Unknown value falls back to "off"
    p = BraveProvider(safesearch="weird")
    assert p.safesearch == "off"
    p = BraveProvider(safesearch="strict")
    assert p.safesearch == "strict"


def test_brave_empty_query_returns_empty(monkeypatch):
    from rune.web_providers.brave import BraveProvider

    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    p = BraveProvider()
    assert p.search("") == []
    assert p.search("   ") == []


def test_brave_extracts_results_from_payload(monkeypatch):
    """Mock the HTTP layer and verify payload → SearchResult conversion."""
    from unittest.mock import patch, MagicMock
    import json
    from rune.web_providers.brave import BraveProvider

    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    p = BraveProvider()

    fake_payload = {
        "web": {
            "results": [
                {
                    "title": "Article 1",
                    "description": "Description 1",
                    "url": "https://example.com/1",
                },
                {
                    "title": "Article 2",
                    "description": "Description 2",
                    "url": "https://example.com/2",
                    "extra_snippets": ["snippet A", "snippet B"],
                },
            ]
        }
    }
    raw = json.dumps(fake_payload).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = p.search("test query", max_results=5)

    assert len(results) == 2
    assert results[0]["title"] == "Article 1"
    assert results[0]["body"] == "Description 1"
    assert results[0]["href"] == "https://example.com/1"
    # extra_snippets should be appended to body
    assert "snippet A" in results[1]["body"]
    assert "snippet B" in results[1]["body"]


def test_brave_returns_empty_on_http_error(monkeypatch):
    """401 / 429 / 500 from Brave should fall through cleanly."""
    from unittest.mock import patch
    from urllib.error import HTTPError
    from rune.web_providers.brave import BraveProvider

    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    p = BraveProvider()

    error = HTTPError(
        url="https://api.search.brave.com", code=429,
        msg="Too Many Requests", hdrs={}, fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=error):
        results = p.search("test")
    assert results == []


def test_brave_handles_malformed_json(monkeypatch):
    from unittest.mock import patch, MagicMock
    from rune.web_providers.brave import BraveProvider

    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    p = BraveProvider()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"this is not JSON {{"
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = p.search("test")
    assert results == []


# ════════════════════════════════════════════════════════════════════
# Composite chain with Brave in pole position
# ════════════════════════════════════════════════════════════════════


def test_default_chain_includes_brave_first(monkeypatch):
    """V4.0.3: Brave is in the chain (not necessarily first since
    Tavily/Serper were added). Kept for backward-compat — the new
    canonical assertion is in ``test_default_chain_full_order``.
    """
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    from rune.web_providers.factory import get_default_provider

    prov = get_default_provider()
    assert hasattr(prov, "providers")
    names = [p.name for p in prov.providers]
    # Brave must be present in the auto chain.
    assert "brave" in names
    # And the chain ends with the no-config fallbacks.
    assert names[-2:] == ["searxng", "ddg"]


def test_brave_skipped_when_unavailable_in_chain(monkeypatch):
    """No BRAVE_API_KEY → composite should skip Brave silently."""
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    from rune.web_providers.factory import CompositeProvider
    from rune.web_providers.brave import BraveProvider

    brave = BraveProvider()  # no key → not available
    fake_fallback = _FakeProvider("fallback", results=[
        {"title": "from fallback", "body": "ok", "href": "h"},
    ])
    chain = CompositeProvider([brave, fake_fallback])
    results = chain.search("test")
    # Brave is skipped, fallback answers.
    assert len(results) == 1
    assert results[0]["title"] == "from fallback"


def test_provider_choice_brave(monkeypatch):
    """WEB_SEARCH_PROVIDER=brave → only Brave is used."""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    from rune.web_providers.factory import get_default_provider

    prov = get_default_provider()
    assert prov.name == "brave"


# ════════════════════════════════════════════════════════════════════
# Tavily provider (V4.0.3+)
# ════════════════════════════════════════════════════════════════════


def test_tavily_unavailable_without_key(monkeypatch):
    from rune.web_providers.tavily import TavilyProvider

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    p = TavilyProvider()
    assert p.is_available() is False
    assert p.search("test") == []


def test_tavily_available_with_key(monkeypatch):
    from rune.web_providers.tavily import TavilyProvider

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")
    p = TavilyProvider()
    assert p.is_available() is True


def test_tavily_extracts_answer_and_results(monkeypatch):
    """Tavily's killer feature: synthesised answer prepended."""
    from unittest.mock import patch, MagicMock
    import json as _json
    from rune.web_providers.tavily import TavilyProvider

    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    p = TavilyProvider()

    fake_payload = {
        "query": "test",
        "answer": "Voici un résumé synthétisé.",
        "results": [
            {
                "title": "Source 1",
                "content": "Detail from source 1",
                "url": "https://a.example",
                "score": 0.95,
            },
            {
                "title": "Source 2",
                "content": "Detail from source 2",
                "url": "https://b.example",
                "score": 0.81,
            },
        ],
    }
    raw = _json.dumps(fake_payload).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = p.search("query", max_results=5)

    # First result must be the synthesised answer.
    assert len(results) >= 1
    assert results[0]["title"] == "Synthèse"
    assert "résumé synthétisé" in results[0]["body"]
    # Subsequent results are the raw snippets.
    assert any(r["title"] == "Source 1" for r in results)
    assert any(r["href"] == "https://b.example" for r in results)


def test_tavily_no_answer_still_returns_results(monkeypatch):
    """If Tavily doesn't include an answer, we still surface results."""
    from unittest.mock import patch, MagicMock
    import json as _json
    from rune.web_providers.tavily import TavilyProvider

    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    p = TavilyProvider()

    fake_payload = {
        "query": "test",
        "results": [
            {"title": "T", "content": "C", "url": "https://x"},
        ],
    }
    raw = _json.dumps(fake_payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = p.search("q")
    assert len(results) == 1
    assert results[0]["title"] == "T"
    # No "Synthèse" prepended.
    assert results[0]["title"] != "Synthèse"


def test_tavily_handles_http_429(monkeypatch):
    """Quota exhausted → empty list, no raise."""
    from unittest.mock import patch
    from urllib.error import HTTPError
    from rune.web_providers.tavily import TavilyProvider

    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    p = TavilyProvider()
    error = HTTPError(
        url="https://api.tavily.com", code=429,
        msg="Too Many Requests", hdrs={}, fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=error):
        assert p.search("test") == []


def test_tavily_search_depth_validation():
    from rune.web_providers.tavily import TavilyProvider

    # Unknown value → "basic"
    p = TavilyProvider(search_depth="weird")
    assert p.search_depth == "basic"
    p = TavilyProvider(search_depth="advanced")
    assert p.search_depth == "advanced"


# ════════════════════════════════════════════════════════════════════
# Serper provider (V4.0.3+)
# ════════════════════════════════════════════════════════════════════


def test_serper_unavailable_without_key(monkeypatch):
    from rune.web_providers.serper import SerperProvider

    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    p = SerperProvider()
    assert p.is_available() is False
    assert p.search("test") == []


def test_serper_available_with_key(monkeypatch):
    from rune.web_providers.serper import SerperProvider

    monkeypatch.setenv("SERPER_API_KEY", "fake-serper-key")
    p = SerperProvider()
    assert p.is_available() is True


def test_serper_promotes_knowledge_graph(monkeypatch):
    """Knowledge graph should appear as first result."""
    from unittest.mock import patch, MagicMock
    import json as _json
    from rune.web_providers.serper import SerperProvider

    monkeypatch.setenv("SERPER_API_KEY", "fake")
    p = SerperProvider()

    fake_payload = {
        "knowledgeGraph": {
            "title": "Albert Einstein",
            "type": "Physicien",
            "description": "Physicien théoricien allemand.",
            "descriptionLink": "https://wikipedia.org/wiki/Einstein",
        },
        "organic": [
            {
                "title": "Wikipedia",
                "snippet": "Article complet sur Einstein",
                "link": "https://wikipedia.org/wiki/Einstein",
                "date": "2024",
            },
        ],
    }
    raw = _json.dumps(fake_payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = p.search("Einstein")

    # KG should be first.
    assert results[0]["title"] == "Albert Einstein"
    assert "Physicien" in results[0]["body"]
    # Then organic.
    assert any(r["title"] == "Wikipedia" for r in results)


def test_serper_promotes_answer_box(monkeypatch):
    from unittest.mock import patch, MagicMock
    import json as _json
    from rune.web_providers.serper import SerperProvider

    monkeypatch.setenv("SERPER_API_KEY", "fake")
    p = SerperProvider()

    fake_payload = {
        "answerBox": {
            "title": "Capitale de la France",
            "answer": "Paris",
            "link": "https://wikipedia.org/wiki/Paris",
        },
    }
    raw = _json.dumps(fake_payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = p.search("capitale france")
    assert results[0]["body"] == "Paris"


def test_serper_handles_http_error(monkeypatch):
    from unittest.mock import patch
    from urllib.error import HTTPError
    from rune.web_providers.serper import SerperProvider

    monkeypatch.setenv("SERPER_API_KEY", "fake")
    p = SerperProvider()
    error = HTTPError(
        url="https://google.serper.dev", code=401,
        msg="Unauthorized", hdrs={}, fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=error):
        assert p.search("test") == []


def test_serper_empty_query_returns_empty(monkeypatch):
    from rune.web_providers.serper import SerperProvider

    monkeypatch.setenv("SERPER_API_KEY", "fake")
    p = SerperProvider()
    assert p.search("") == []
    assert p.search("   ") == []


# ════════════════════════════════════════════════════════════════════
# Composite chain V4.0.3+ : Tavily → Serper → Brave → SearXNG → DDG
# ════════════════════════════════════════════════════════════════════


def test_default_chain_full_order(monkeypatch):
    """V4.0.3+: auto chain is Tavily → Serper → Brave → SearXNG → DDG."""
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    from rune.web_providers.factory import get_default_provider

    prov = get_default_provider()
    assert hasattr(prov, "providers")
    names = [p.name for p in prov.providers]
    assert names == ["tavily", "serper", "brave", "searxng", "ddg"]


def test_chain_skips_unavailable_providers(monkeypatch):
    """No keys at all → composite skips paid providers, lands on DDG."""
    for k in ("TAVILY_API_KEY", "SERPER_API_KEY", "BRAVE_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    from rune.web_providers.factory import (
        CompositeProvider, get_default_provider,
    )

    chain = get_default_provider()
    assert isinstance(chain, CompositeProvider)
    # None of tavily/serper/brave is available.
    assert chain.providers[0].is_available() is False  # tavily
    assert chain.providers[1].is_available() is False  # serper
    assert chain.providers[2].is_available() is False  # brave


def test_provider_choice_tavily(monkeypatch):
    """WEB_SEARCH_PROVIDER=tavily → only Tavily."""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    from rune.web_providers.factory import get_default_provider

    prov = get_default_provider()
    assert prov.name == "tavily"


def test_provider_choice_serper(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "serper")
    monkeypatch.setenv("SERPER_API_KEY", "fake")
    from rune.web_providers.factory import get_default_provider

    prov = get_default_provider()
    assert prov.name == "serper"


def test_registry_lists_all_5_providers():
    from rune.web_providers.factory import list_providers

    names = list_providers()
    assert set(names) == {"tavily", "serper", "brave", "searxng", "ddg"}
