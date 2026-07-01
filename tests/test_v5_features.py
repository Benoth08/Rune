"""Suite de tests pour les nouveautés V5 / V5.1 / V5.2.

Couvre les briques ajoutées pendant les sessions des 16-17 mai 2026 :

V5.0 — Routage hybride web
   • web_classifier (slow-path LLM binaire OUI/NON)
   • looks_like_question (heuristique question vs casual)
   • Cache LRU + parsing robuste
   • Tags /web et /noweb

V5.1 — Multi-tool router
   • semantic_router (structure)
   • tool_dispatcher (slow-path JSON multi-classes)
   • python_executor (sandbox subprocess + timeout)

V5.2 — CRAG + GraphRAG communities
   • crag.evaluate_retrieval (classification CORRECT/AMBIGUOUS/INCORRECT)
   • crag.evaluate_and_rescue (rewrite + retry)
   • graph_communities.detect_communities (3 backends en cascade)
   • graph_communities.summarise_communities (LLM summaries)
   • graph_communities.render_community_context (injection prompt)
"""

from __future__ import annotations

import pytest


# ── Mocks partagés ─────────────────────────────────────────────────────


class FakeLLM:
    """Mock LLM avec réponses programmables par mots-clés dans la query."""

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.is_loaded = True
        self.call_count = 0
        self.last_messages: list = []

    def complete_sync(
        self, messages, max_new_tokens=32, temperature=0.3, timeout=None,
    ):
        self.call_count += 1
        self.last_messages = messages
        content = messages[-1]["content"] if messages else ""
        for trigger, response in self.responses.items():
            if trigger.lower() in content.lower():
                return response
        return ""


class FakeRetriever:
    def __init__(self, results_by_query: dict | None = None):
        self.results_by_query = results_by_query or {}
        self.default_results: list = []
        self.call_log: list = []

    def search(self, query, n=5, rerank=True):
        self.call_log.append(query)
        for trigger, results in self.results_by_query.items():
            if trigger.lower() in query.lower():
                return results
        return self.default_results


# ═══════════════════════════════════════════════════════════════════════
# V5.0 — Web classifier
# ═══════════════════════════════════════════════════════════════════════


class TestWebClassifierV5:

    def test_looks_like_question_positive(self):
        from rune.cognition.web_classifier import looks_like_question
        for msg in [
            "Quel est le prix de l'iPhone ?",
            "Comment fonctionne X",
            "Combien coûte un mug ?",
            "Who won the World Cup?",
            "Je voudrais savoir quand sort GTA 6",
            "Dis-moi tout sur Roland Garros",
            "I want to know about AI",
        ]:
            assert looks_like_question(msg), f"Devrait matcher : {msg!r}"

    def test_looks_like_question_negative(self):
        from rune.cognition.web_classifier import looks_like_question
        for msg in ["Merci !", "Bonjour", "ok", "", "   "]:
            assert not looks_like_question(msg), f"Ne devrait pas matcher : {msg!r}"

    def test_parse_classifier_response(self):
        from rune.cognition.web_classifier import _parse_classifier_response
        cases = [
            ("OUI : prix volatile", True),
            ("NON: concept stable", False),
            ("YES - news today", True),
            ("NO", False),
        ]
        for raw, expected_verdict in cases:
            result = _parse_classifier_response(raw)
            assert result is not None
            assert result[0] == expected_verdict, f"Cas {raw!r}"
        assert _parse_classifier_response("") is None

    def test_classifier_cache(self):
        from rune.cognition.web_classifier import (
            should_search_via_llm, clear_cache,
        )
        clear_cache()
        llm = FakeLLM({
            "prix": "OUI : prix volatile",
            "fonctionne": "NON : concept stable",
        })
        d1, _ = should_search_via_llm("Quel est le prix de l'iPhone ?", llm)
        d2, _ = should_search_via_llm("Quel est le prix de l'iPhone ?", llm)
        d3, _ = should_search_via_llm("Comment fonctionne un transformer ?", llm)
        assert d1 is True
        assert d2 is True
        assert d3 is False
        assert llm.call_count == 2

    def test_classifier_skip_non_question(self):
        from rune.cognition.web_classifier import (
            should_search_via_llm, clear_cache,
        )
        clear_cache()
        llm = FakeLLM()
        result = should_search_via_llm("Merci !", llm)
        assert result == (False, "not_a_question")
        assert llm.call_count == 0


class TestWebTagsV5:

    def test_web_tag_forces_search(self):
        from rune.web import WebTriggerPolicy
        p = WebTriggerPolicy(mode="auto")
        trig, reason = p.should_search("/web Recommande-moi un mug")
        assert trig is True
        assert "manual /web" in reason

    def test_noweb_tag_blocks_search(self):
        from rune.web import WebTriggerPolicy
        p = WebTriggerPolicy(mode="auto")
        trig1, _ = p.should_search("Recommande-moi un modèle NER")
        assert trig1 is True
        trig2, reason2 = p.should_search("/noweb Recommande-moi un modèle NER")
        assert trig2 is False
        assert "manual /noweb" in reason2

    def test_noweb_priority_over_web(self):
        from rune.web import WebTriggerPolicy
        p = WebTriggerPolicy(mode="auto")
        trig, reason = p.should_search("/web /noweb question")
        assert trig is False
        assert "manual /noweb" in reason

    def test_website_not_mangled(self):
        from rune.web import WebTriggerPolicy
        p = WebTriggerPolicy(mode="auto")
        _, reason = p.should_search("Quel est ton /website préféré ?")
        assert "manual /web tag" not in reason


# ═══════════════════════════════════════════════════════════════════════
# V5.1 — Semantic router
# ═══════════════════════════════════════════════════════════════════════


class TestSemanticRouterStructure:

    def test_routes_defined(self):
        from rune.cognition.semantic_router import ROUTES
        names = {r.name for r in ROUTES}
        assert names == {"web", "python", "none"}

    def test_route_examples_coverage(self):
        from rune.cognition.semantic_router import ROUTES
        for route in ROUTES:
            assert len(route.examples) >= 20, (
                f"Route {route.name}: {len(route.examples)} exemples (< 20)"
            )

    def test_thresholds_sensible(self):
        from rune.cognition.semantic_router import ROUTES
        for route in ROUTES:
            assert 0.3 <= route.threshold <= 0.8


# ═══════════════════════════════════════════════════════════════════════
# V5.1 — Tool dispatcher
# ═══════════════════════════════════════════════════════════════════════


class TestToolDispatcherV51:

    def test_parse_strict_json(self):
        from rune.cognition.tool_dispatcher import _parse_dispatcher_response
        assert _parse_dispatcher_response(
            '{"tool": "web", "reason": "actu"}'
        ) == ("web", "actu")

    def test_parse_json_with_blabla(self):
        from rune.cognition.tool_dispatcher import _parse_dispatcher_response
        raw = 'Je pense {"tool":"python","reason":"calcul"} oui'
        assert _parse_dispatcher_response(raw) == ("python", "calcul")

    def test_parse_regex_fallback(self):
        from rune.cognition.tool_dispatcher import _parse_dispatcher_response
        raw = 'la réponse est "tool": "none" voilà'
        result = _parse_dispatcher_response(raw)
        assert result is not None and result[0] == "none"

    def test_parse_invalid_tool_rejected(self):
        from rune.cognition.tool_dispatcher import _parse_dispatcher_response
        result = _parse_dispatcher_response(
            '{"tool": "delete_all_files", "reason": "evil"}'
        )
        assert result is None

    def test_parse_garbage(self):
        from rune.cognition.tool_dispatcher import _parse_dispatcher_response
        assert _parse_dispatcher_response("") is None

    def test_dispatch_cache(self):
        from rune.cognition.tool_dispatcher import (
            dispatch_via_llm, clear_cache,
        )
        clear_cache()
        llm = FakeLLM({"prix": '{"tool":"web","reason":"prix"}'})
        dispatch_via_llm("Quel prix ?", llm)
        dispatch_via_llm("Quel prix ?", llm)
        assert llm.call_count == 1


# ═══════════════════════════════════════════════════════════════════════
# V5.1 — Python executor
# ═══════════════════════════════════════════════════════════════════════


class TestPythonExecutorV51:

    def test_simple_calc(self):
        from rune.tools.python_executor import run
        r = run("print(17 * 23 + 89 / 11)")
        assert r["ok"] is True
        assert "399.09" in r["stdout"]

    def test_empty_code(self):
        from rune.tools.python_executor import run
        r = run("")
        assert r["ok"] is False
        assert r["error"] == "empty_code"

    def test_runtime_error_captured(self):
        from rune.tools.python_executor import run
        r = run("undefined_variable")
        assert r["ok"] is False
        assert "NameError" in r["stderr"]

    def test_timeout(self):
        from rune.tools.python_executor import run
        r = run("while True: pass", timeout=1.0)
        assert r["ok"] is False
        assert "timeout" in r["error"]

    def test_format_result(self):
        from rune.tools.python_executor import run, format_result
        r = run("for i in range(3): print(i)")
        formatted = format_result(r)
        assert "Exécution réussie" in formatted
        assert "0" in formatted and "1" in formatted and "2" in formatted

    def test_clean_env_no_secrets(self):
        from rune.tools.python_executor import run
        r = run(
            "import os; "
            "print('OPENAI_API_KEY' in os.environ, "
            "'ANTHROPIC_API_KEY' in os.environ)"
        )
        assert r["ok"]
        assert "False False" in r["stdout"]


# ═══════════════════════════════════════════════════════════════════════
# V5.2 — CRAG
# ═══════════════════════════════════════════════════════════════════════


class TestCRAGEvaluate:

    def test_empty(self):
        from rune.cognition.crag import evaluate_retrieval, RetrievalStatus
        v = evaluate_retrieval("q", [])
        assert v.status == RetrievalStatus.EMPTY

    def test_correct(self):
        from rune.cognition.crag import evaluate_retrieval, RetrievalStatus
        v = evaluate_retrieval("q", [{"rerank_score": 0.95}])
        assert v.status == RetrievalStatus.CORRECT

    def test_correct_at_threshold(self):
        from rune.cognition.crag import evaluate_retrieval, RetrievalStatus
        v = evaluate_retrieval("q", [{"rerank_score": 0.7}])
        assert v.status == RetrievalStatus.CORRECT

    def test_ambiguous(self):
        from rune.cognition.crag import evaluate_retrieval, RetrievalStatus
        v = evaluate_retrieval("q", [{"rerank_score": 0.5}])
        assert v.status == RetrievalStatus.AMBIGUOUS

    def test_incorrect(self):
        from rune.cognition.crag import evaluate_retrieval, RetrievalStatus
        v = evaluate_retrieval("q", [{"rerank_score": 0.1}])
        assert v.status == RetrievalStatus.INCORRECT

    def test_score_fallback(self):
        from rune.cognition.crag import evaluate_retrieval, RetrievalStatus
        v = evaluate_retrieval("q", [{"score": 0.8}])
        assert v.status == RetrievalStatus.CORRECT

    def test_uses_max_score(self):
        from rune.cognition.crag import evaluate_retrieval, RetrievalStatus
        chunks = [
            {"rerank_score": 0.2},
            {"rerank_score": 0.95},
            {"rerank_score": 0.4},
        ]
        v = evaluate_retrieval("q", chunks)
        assert v.top_score == 0.95


class TestCRAGRescue:

    def test_no_rescue_if_correct(self):
        from rune.cognition.crag import evaluate_and_rescue
        llm = FakeLLM({"x": "reformulation"})
        v = evaluate_and_rescue(
            "q", [{"rerank_score": 0.9}], FakeRetriever(),
            llm=llm, enable_rewrite=True,
        )
        assert v.rewritten_query is None
        assert llm.call_count == 0

    def test_no_rescue_if_incorrect(self):
        from rune.cognition.crag import evaluate_and_rescue
        llm = FakeLLM({"x": "X"})
        v = evaluate_and_rescue(
            "q", [{"rerank_score": 0.1}], FakeRetriever(),
            llm=llm, enable_rewrite=True,
        )
        assert v.rewritten_query is None

    def test_rescue_succeeds(self):
        from rune.cognition.crag import evaluate_and_rescue, RetrievalStatus
        llm = FakeLLM({"trucs ML": "machine learning algorithmes"})
        retriever = FakeRetriever({
            "machine learning": [{"rerank_score": 0.85}],
        })
        v = evaluate_and_rescue(
            "trucs ML", [{"rerank_score": 0.5}], retriever,
            llm=llm, enable_rewrite=True,
        )
        assert v.top_score == 0.85
        assert v.rewritten_query == "machine learning algorithmes"

    def test_rescue_declines_when_worse(self):
        from rune.cognition.crag import evaluate_and_rescue
        llm = FakeLLM({"trucs ML": "machine learning"})
        retriever = FakeRetriever({"machine learning": [{"rerank_score": 0.2}]})
        v = evaluate_and_rescue(
            "trucs ML", [{"rerank_score": 0.5}], retriever,
            llm=llm, enable_rewrite=True,
        )
        assert v.top_score == 0.5
        assert v.rewritten_query == "machine learning"

    def test_cognitive_items(self):
        from rune.cognition.crag import (
            evaluate_retrieval, cognitive_item_for,
        )
        v = evaluate_retrieval("q", [{"rerank_score": 0.9}])
        assert cognitive_item_for(v) is None
        v = evaluate_retrieval("q", [{"rerank_score": 0.5}])
        assert "partiellement" in cognitive_item_for(v)
        v = evaluate_retrieval("q", [{"rerank_score": 0.1}])
        assert "Rien de vraiment" in cognitive_item_for(v)


# ═══════════════════════════════════════════════════════════════════════
# V5.2 — GraphRAG communities
# ═══════════════════════════════════════════════════════════════════════


def _make_fake_kg():
    from dataclasses import dataclass

    @dataclass
    class FakeEnt:
        entity_id: str
        value: str
        type: str = "concept"

    @dataclass
    class FakeRel:
        subject_id: str
        object_id: str
        predicate: str = "related_to"

    entities = {
        "e1": FakeEnt("e1", "Marie", "person"),
        "e2": FakeEnt("e2", "Paul", "person"),
        "e3": FakeEnt("e3", "Léa", "person"),
        "e4": FakeEnt("e4", "Lythéa", "project"),
        "e5": FakeEnt("e5", "Taëlys", "project"),
        "e6": FakeEnt("e6", "RunPod", "infra"),
    }
    relations = [
        FakeRel("e1", "e2"), FakeRel("e2", "e3"), FakeRel("e1", "e3"),
        FakeRel("e4", "e5"), FakeRel("e5", "e6"), FakeRel("e4", "e6"),
        FakeRel("e1", "e4"),
    ]
    return entities, relations


class TestGraphCommunitiesDetection:

    def test_empty_kg(self):
        from rune.cognition.graph_communities import detect_communities
        assert detect_communities({}, []) == []

    def test_no_relations(self):
        from rune.cognition.graph_communities import detect_communities
        entities, _ = _make_fake_kg()
        assert detect_communities(entities, []) == []

    def test_basic_detection(self):
        from rune.cognition.graph_communities import detect_communities
        entities, relations = _make_fake_kg()
        communities = detect_communities(entities, relations)
        assert len(communities) >= 1
        total = sum(c.size for c in communities)
        assert total == 6

    def test_min_size_filter(self):
        from rune.cognition.graph_communities import detect_communities
        entities, relations = _make_fake_kg()
        assert detect_communities(
            entities, relations, min_community_size=100,
        ) == []

    def test_sorted_by_size(self):
        from rune.cognition.graph_communities import detect_communities
        entities, relations = _make_fake_kg()
        communities = detect_communities(entities, relations)
        if len(communities) > 1:
            sizes = [c.size for c in communities]
            assert sizes == sorted(sizes, reverse=True)


class TestGraphCommunitiesSummarise:

    def test_summarise_basic(self):
        from rune.cognition.graph_communities import (
            detect_communities, summarise_communities,
        )
        entities, relations = _make_fake_kg()
        communities = detect_communities(entities, relations)
        llm = FakeLLM({
            "Lythéa": "Projets tech",
            "Marie": "Famille",
        })
        summarise_communities(communities, entities, llm, max_communities=5)
        with_summary = [c for c in communities if c.summary]
        assert len(with_summary) >= 1

    def test_summarise_skips_already_done(self):
        from rune.cognition.graph_communities import (
            detect_communities, summarise_communities,
        )
        entities, relations = _make_fake_kg()
        communities = detect_communities(entities, relations)
        for c in communities:
            c.summary = "déjà fait"
        llm = FakeLLM()
        summarise_communities(communities, entities, llm, max_communities=5)
        assert llm.call_count == 0


class TestGraphCommunitiesRender:

    def test_render_basic(self):
        from rune.cognition.graph_communities import (
            detect_communities, render_community_context,
        )
        entities, relations = _make_fake_kg()
        communities = detect_communities(entities, relations)
        for c in communities:
            c.summary = f"Cluster {c.community_id}"
        block = render_community_context(communities, entities)
        assert "Thématiques de mémoire" in block

    def test_render_empty(self):
        from rune.cognition.graph_communities import render_community_context
        assert render_community_context([], {}) == ""

    def test_render_truncates(self):
        from rune.cognition.graph_communities import (
            detect_communities, render_community_context,
        )
        entities, relations = _make_fake_kg()
        communities = detect_communities(entities, relations)
        for c in communities:
            c.summary = "x" * 5000
        block = render_community_context(
            communities, entities, max_chars=300,
        )
        assert len(block) <= 320


# ═══════════════════════════════════════════════════════════════════════
# Intégration end-to-end
# ═══════════════════════════════════════════════════════════════════════


class TestE2EIntegration:

    def test_crag_full_path(self):
        """AMBIGUOUS → rewrite → CORRECT."""
        from rune.cognition.crag import evaluate_and_rescue, RetrievalStatus
        llm = FakeLLM({
            "le truc dont": "anti-confabulation pattern",
        })
        retriever = FakeRetriever({
            "anti-confabulation": [{"rerank_score": 0.82}],
        })
        v = evaluate_and_rescue(
            "le truc dont on parlait", [{"rerank_score": 0.55}],
            retriever, llm=llm, enable_rewrite=True,
        )
        assert v.status == RetrievalStatus.CORRECT
        assert v.top_score >= 0.82

    def test_dispatcher_python_route(self):
        """Question calcul → dispatcher choisit python."""
        from rune.cognition.tool_dispatcher import (
            dispatch_via_llm, clear_cache,
        )
        clear_cache()
        llm = FakeLLM({
            "calcul": '{"tool": "python", "reason": "calc"}',
        })
        tool, _ = dispatch_via_llm("Calcule 17*23", llm)
        assert tool == "python"

    def test_full_kg_lifecycle(self):
        """KG vide → ajout entités → détection communautés."""
        from rune.cognition.graph_communities import (
            detect_communities, summarise_communities, render_community_context,
        )
        entities, relations = _make_fake_kg()
        communities = detect_communities(entities, relations)
        assert len(communities) >= 1

        llm = FakeLLM({
            "Marie": "Famille",
            "Lythéa": "Projets",
        })
        summarise_communities(communities, entities, llm)

        block = render_community_context(communities, entities)
        assert "Thématiques" in block


# ═══════════════════════════════════════════════════════════════════════
# Hygiène : reset des caches entre tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def reset_caches():
    yield
    try:
        from rune.cognition.web_classifier import clear_cache as wc
        wc()
    except Exception:
        pass
    try:
        from rune.cognition.tool_dispatcher import clear_cache as tc
        tc()
    except Exception:
        pass
