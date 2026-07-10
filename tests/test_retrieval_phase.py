"""Unit tests for :class:`rune.cognition.retrieval.RetrievalPhase`.

Mocks the four backends. The KG mock matches the public surface of
:class:`KnowledgeGraphStore` (``entities`` dict, ``relations`` dict,
``query_by_question``). Entities and relations are simple
namespace objects with the attributes used by the phase.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch", reason="some recall paths use tensors")

from rune.cognition.retrieval import (  # noqa: E402
    IDENTITY_FRESHNESS_MIN_GAP_SEC,
    RECALL_TEXT_MAX_CHARS,
    SECTION_SEPARATOR,
    UNCOVERED_ENTITIES_LIMIT,
    RetrievalContext,
    RetrievalPhase,
)


# ── KG fixture builders ────────────────────────────────────────────────

def _entity(eid: str, value: str, etype: str, last_seen: float = 0.0):
    return SimpleNamespace(
        entity_id=eid, value=value, type=etype, last_seen=last_seen,
    )


def _relation(subj: str, pred: str, obj: str):
    return SimpleNamespace(subject_id=subj, predicate=pred, object_id=obj)


def _make_kg(entities: list, relations: list, facts_for_query: list[str] | None = None):
    """Build a KG mock with the exact shape RetrievalPhase reads."""
    kg = MagicMock()
    kg.entities = {e.entity_id: e for e in entities}
    kg.relations = {f"r{i}": r for i, r in enumerate(relations)}
    kg.query_by_question.return_value = facts_for_query or []
    return kg


def _make_phase(
    *,
    kg=None,
    extractor=None,
    mhn_results=None,
    chroma_results=None,
    encode_fail: bool = False,
):
    if kg is None:
        kg = _make_kg([], [])

    if extractor is None:
        extractor = MagicMock()
        if encode_fail:
            extractor.encode.side_effect = RuntimeError("fail")
        else:
            extractor.encode.return_value = torch.zeros(8)
        extractor.extract.return_value = []

    mhn = MagicMock()
    mhn.retrieve.return_value = mhn_results or []

    retriever = MagicMock()
    retriever.search.return_value = chroma_results or []

    return RetrievalPhase(
        kg=kg, mhn=mhn, entity_extractor=extractor,
        hybrid_retriever=retriever,
    )


# ── KG identity narrative ──────────────────────────────────────────────

def test_identity_empty_when_no_entities():
    phase = _make_phase(kg=_make_kg([], []))
    assert phase.kg_identity_summary() == ""


def test_identity_with_no_persons_returns_uncovered_only():
    """Non-person entities with no relations → fallback section only.

    The narrative still has a header (``[Identité de ton interlocuteur
    — pour mémoire]``) so the prompt block is not empty."""
    kg = _make_kg(
        entities=[_entity("e1", "Aix", "location"),
                  _entity("e2", "RunPod", "product")],
        relations=[],
    )
    phase = _make_phase(kg=kg)
    summary = phase.kg_identity_summary()
    assert summary != ""
    assert "Autres informations connues" in summary
    assert "Aix (location)" in summary
    assert "RunPod (product)" in summary


def test_identity_with_person_and_relations():
    """Person + 2 relations → narrative line with both facts in FR."""
    kg = _make_kg(
        entities=[
            _entity("p1", "Mika", "person", last_seen=0.0),
            _entity("e1", "Aix", "location"),
            _entity("e2", "Anthropic", "organization"),
        ],
        relations=[
            _relation("p1", "vit_à", "e1"),
            _relation("p1", "travaille_chez", "e2"),
        ],
    )
    phase = _make_phase(kg=kg)
    summary = phase.kg_identity_summary()
    assert "Ton interlocuteur s'appelle Mika" in summary
    assert "vit à Aix" in summary
    assert "travaille chez Anthropic" in summary
    # No "Autres informations" because both objects are covered by relations.
    assert "Autres informations" not in summary


def test_identity_block_uses_descriptive_not_imperative_wording():
    """The identity block must guide the LLM towards mentioning these
    facts only when relevant — not towards reciting them at every turn.

    Bug observed in prod with Mistral-7B: the previous wording ("Tu DOIS
    utiliser ces informations dans ta réponse") made the model open
    every reply with "Salut Mika, tu es bien à Aix-en-Provence, tu
    travailles chez Anthropic..." regardless of the question. Verifies
    that the block now uses descriptive language.
    """
    kg = _make_kg(
        entities=[_entity("p1", "Sophie", "person", last_seen=0.0)],
        relations=[],
    )
    summary = _make_phase(kg=kg).kg_identity_summary()
    # The old imperative formulation must be gone.
    assert "Tu DOIS" not in summary
    assert "INFORMATIONS VÉRIFIÉES" not in summary
    # The new descriptive language must be present.
    assert "pour mémoire" in summary.lower()
    # The "don't recite at every turn" guidance must be in the block.
    assert "à chaque message" in summary or "récit" in summary.lower()


def test_identity_unknown_predicate_passes_through():
    """A predicate not in _PREDICATE_FR must be rendered as-is."""
    kg = _make_kg(
        entities=[
            _entity("p1", "Mika", "person"),
            _entity("e1", "Quelque chose", "thing"),
        ],
        relations=[_relation("p1", "experimental_pred", "e1")],
    )
    phase = _make_phase(kg=kg)
    summary = phase.kg_identity_summary()
    assert "experimental_pred Quelque chose" in summary


def test_identity_freshness_suffix_omitted_under_5min():
    """last_seen <5 min ago → no '(dernière mention…)' suffix."""
    recent = time.time() - 60  # 1 min ago
    kg = _make_kg(
        entities=[_entity("p1", "Mika", "person", last_seen=recent)],
        relations=[],
    )
    summary = _make_phase(kg=kg).kg_identity_summary()
    assert "dernière mention" not in summary


def test_identity_freshness_suffix_added_above_5min():
    """last_seen ≫5 min ago → suffix present."""
    old = time.time() - (IDENTITY_FRESHNESS_MIN_GAP_SEC + 600)
    kg = _make_kg(
        entities=[_entity("p1", "Mika", "person", last_seen=old)],
        relations=[],
    )
    summary = _make_phase(kg=kg).kg_identity_summary()
    assert "dernière mention" in summary


def test_identity_uncovered_capped_at_limit():
    """More than UNCOVERED_ENTITIES_LIMIT uncovered entities → truncated."""
    n = UNCOVERED_ENTITIES_LIMIT + 3
    entities = [_entity(f"e{i}", f"v{i}", "thing") for i in range(n)]
    kg = _make_kg(entities=entities, relations=[])
    summary = _make_phase(kg=kg).kg_identity_summary()
    # Count occurrences of "v" + digit in the listing
    listed = sum(1 for i in range(n) if f"v{i}" in summary)
    assert listed == UNCOVERED_ENTITIES_LIMIT


def test_identity_skips_person_from_uncovered():
    """Persons without relations get their own line, not the
    fallback list."""
    kg = _make_kg(
        entities=[
            _entity("p1", "Mika", "person"),
            _entity("e1", "Aix", "location"),
        ],
        relations=[],
    )
    summary = _make_phase(kg=kg).kg_identity_summary()
    assert "Ton interlocuteur s'appelle Mika" in summary
    # Person must NOT appear in the "Autres informations" section
    autre_line = [l for l in summary.split("\n") if "Autres informations" in l]
    assert len(autre_line) == 1
    assert "Mika" not in autre_line[0]


# ── KG facts section ───────────────────────────────────────────────────

def test_kg_facts_section_appended_when_facts_exist():
    kg = _make_kg(
        entities=[],
        relations=[],
        facts_for_query=["Mika vit à Aix", "Mika travaille chez Anthropic"],
    )
    phase = _make_phase(kg=kg)
    ctx = phase.gather("Où vit Mika ?")
    facts_section = next(s for s in ctx.sections if s.startswith("[Faits connus"))
    assert "• Mika vit à Aix" in facts_section
    assert "• Mika travaille chez Anthropic" in facts_section


def test_kg_facts_section_omitted_when_empty():
    phase = _make_phase()  # no facts in default KG
    ctx = phase.gather("anything")
    assert not any(s.startswith("[Faits connus") for s in ctx.sections)


def test_kg_facts_skipped_without_extractor():
    """Without an entity extractor, no facts query is attempted."""
    kg = _make_kg(
        entities=[], relations=[],
        facts_for_query=["should not appear"],
    )
    phase = RetrievalPhase(
        kg=kg, mhn=MagicMock(),
        entity_extractor=None, hybrid_retriever=None,
    )
    ctx = phase.gather("anything")
    assert not any("Faits connus" in s for s in ctx.sections)
    kg.query_by_question.assert_not_called()


# ── MHN episodic section ───────────────────────────────────────────────

def test_episodic_section_with_results():
    mhn_results = [
        {"text": "souvenir 1", "timestamp": time.time() - 60},
        {"text": "souvenir 2", "timestamp": time.time() - 3600},
    ]
    phase = _make_phase(mhn_results=mhn_results)
    ctx = phase.gather("salut")
    epis = next(s for s in ctx.sections if s.startswith("[Mémoire épisodique"))
    assert "souvenir 1" in epis
    assert "souvenir 2" in epis
    # The thought should fire once
    assert any("épisodique" in t for t in ctx.thoughts)


def test_episodic_text_truncated_to_recall_max():
    long_text = "x" * 1000
    phase = _make_phase(mhn_results=[
        {"text": long_text, "timestamp": time.time()},
    ])
    ctx = phase.gather("q")
    epis = next(s for s in ctx.sections if s.startswith("[Mémoire épisodique"))
    # The injected snippet must be ≤ RECALL_TEXT_MAX_CHARS chars long
    # (plus the freshness annotation which is short).
    body = epis.replace("[Mémoire épisodique — pour information]\n• ", "")
    # Strip any trailing freshness annotation in parens
    head = body.split(" (")[0]
    assert len(head) <= RECALL_TEXT_MAX_CHARS


def test_episodic_calls_mhn_with_correct_top_k_and_attention():
    from rune.cognition.retrieval import MHN_TOP_K, MHN_MIN_ATTENTION
    phase = _make_phase(mhn_results=[])
    phase.gather("q")
    call = phase.mhn.retrieve.call_args
    assert call.kwargs["top_k"] == MHN_TOP_K
    assert call.kwargs["min_attention"] == MHN_MIN_ATTENTION


def test_episodic_skipped_when_extractor_returns_none():
    extractor = MagicMock()
    extractor.encode.return_value = None
    extractor.extract.return_value = []
    phase = RetrievalPhase(
        kg=_make_kg([], []), mhn=MagicMock(),
        entity_extractor=extractor, hybrid_retriever=None,
    )
    ctx = phase.gather("q")
    assert not any("épisodique" in s for s in ctx.sections)
    phase.mhn.retrieve.assert_not_called()


def test_episodic_skipped_on_encode_failure():
    """An exception in extractor.encode must not crash the gather."""
    phase = _make_phase(encode_fail=True)
    ctx = phase.gather("q")
    assert not any("épisodique" in s for s in ctx.sections)


# ── Chroma semantic section ────────────────────────────────────────────

def test_semantic_section_with_results():
    chroma_results = [
        {"document": "doc 1", "metadata": {"ts": time.time() - 60}},
        {"document": "doc 2", "metadata": {"ts": time.time() - 3600}},
    ]
    phase = _make_phase(chroma_results=chroma_results)
    ctx = phase.gather("query")
    sem = next(s for s in ctx.sections if s.startswith("[Mémoire sémantique"))
    assert "doc 1" in sem
    assert "doc 2" in sem
    assert any("long-terme" in t for t in ctx.thoughts)


def test_semantic_uses_rerank_true():
    """Retrieval (unlike Surprise) MUST rerank — accuracy beats latency."""
    phase = _make_phase(chroma_results=[])
    phase.gather("q")
    call = phase.hybrid_retriever.search.call_args
    assert call.kwargs.get("rerank") is True


def test_semantic_handles_missing_metadata():
    """Some documents may not have ts metadata — must not crash."""
    phase = _make_phase(chroma_results=[
        {"document": "no metadata"},  # no 'metadata' key at all
        {"document": "empty meta", "metadata": None},
    ])
    ctx = phase.gather("q")
    # Both should appear; no exception
    sem = next(s for s in ctx.sections if s.startswith("[Mémoire sémantique"))
    assert "no metadata" in sem
    assert "empty meta" in sem


def test_semantic_skipped_without_retriever():
    phase = RetrievalPhase(
        kg=_make_kg([], []), mhn=MagicMock(),
        entity_extractor=MagicMock(),
        hybrid_retriever=None,
    )
    ctx = phase.gather("q")
    assert not any("sémantique" in s for s in ctx.sections)


# ── Composition / RetrievalContext.render ──────────────────────────────

def test_render_empty_returns_empty_string():
    ctx = RetrievalContext()
    assert ctx.render() == ""


def test_render_joins_with_section_separator():
    ctx = RetrievalContext(sections=["A", "B", "C"])
    rendered = ctx.render()
    assert SECTION_SEPARATOR in rendered
    # Should split back to the original sections
    parts = rendered.split(SECTION_SEPARATOR)
    assert parts == ["A", "B", "C"]


def test_gather_full_pipeline():
    """All four sources fire — context contains 4 sections."""
    kg = _make_kg(
        entities=[
            _entity("p1", "Mika", "person"),
            _entity("e1", "Aix", "location"),
        ],
        relations=[_relation("p1", "vit_à", "e1")],
        facts_for_query=["Mika vit à Aix"],
    )
    phase = _make_phase(
        kg=kg,
        mhn_results=[{"text": "souvenir", "timestamp": time.time()}],
        chroma_results=[{"document": "doc archive",
                         "metadata": {"ts": time.time()}}],
    )
    ctx = phase.gather("Où vit Mika ?")
    section_kinds = [s.split("\n")[0] for s in ctx.sections]
    # Identity has a different first line — check by substring
    assert any("Identité" in s for s in section_kinds)
    assert any("Faits connus" in s for s in section_kinds)
    assert any("Mémoire épisodique" in s for s in section_kinds)
    assert any("Mémoire sémantique" in s for s in section_kinds)
    assert len(ctx.sections) == 5
    # Two thoughts (épisodique + long-terme)
    assert len(ctx.thoughts) == 2


def test_gather_one_source_failure_does_not_kill_others():
    """Chroma raises → identity + episodic still get through."""
    kg = _make_kg(
        entities=[_entity("p1", "Mika", "person")],
        relations=[],
    )
    phase = _make_phase(
        kg=kg,
        mhn_results=[{"text": "ep", "timestamp": time.time()}],
    )
    phase.hybrid_retriever.search.side_effect = RuntimeError("chroma down")
    ctx = phase.gather("q")
    # Identity + episodic still present, semantic missing
    assert any("Identité" in s for s in ctx.sections)
    assert any("épisodique" in s for s in ctx.sections)
    assert not any("sémantique" in s for s in ctx.sections)


def test_gather_all_empty_yields_empty_render():
    """No KG, no extractor, no retriever → empty context renders as ''."""
    phase = RetrievalPhase(
        kg=_make_kg([], []),
        mhn=MagicMock(),
        entity_extractor=None,
        hybrid_retriever=None,
    )
    ctx = phase.gather("anything")
    assert ctx.render() == ""
    assert ctx.sections == []
    assert ctx.thoughts == []
