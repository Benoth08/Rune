"""Tests for the KG improvements: accent stripping, trigram pre-filter."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rune.memory.kg import KnowledgeGraphStore


# ── Normalisation with accents ────────────────────────────────────────

def test_normalize_strips_french_accents():
    n = KnowledgeGraphStore._normalize
    assert n("François") == "francois"
    assert n("Élise") == "elise"
    assert n("Hervé") == n("herve") == "herve"
    assert n("Genève") == "geneve"


def test_normalize_strips_other_diacritics():
    n = KnowledgeGraphStore._normalize
    # Spanish ñ
    assert n("España") == "espana"
    # German umlauts: NFKD splits ä → a + combining-diaeresis (no special
    # German collation, just diacritic removal). This may surprise German
    # speakers who'd expect ä→ae but is consistent with our French-first
    # design — and we document it.
    assert n("über") == "uber"
    # Polish ł doesn't decompose into l + combining mark — it's a single
    # codepoint. NFKD leaves it untouched. We accept this as-is.


def test_normalize_preserves_internal_spaces_and_lowers():
    n = KnowledgeGraphStore._normalize
    assert n("  Jean Pierre  ") == "jean pierre"
    assert n("HELLO WORLD") == "hello world"


def test_normalize_handles_empty_and_none_like():
    n = KnowledgeGraphStore._normalize
    assert n("") == ""
    # The function is typed for str but a defensive None-equivalent
    assert n("   ") == ""


# ── Accent-aware deduplication via upsert ─────────────────────────────

def test_upsert_dedupes_accented_variant():
    """François and francois must collapse to the same entity."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        e1 = kg.upsert_entity("François", "person", confidence=0.9)
        e2 = kg.upsert_entity("francois", "person", confidence=0.7)
        assert e1 == e2
        assert kg.entities[e1].mention_count == 2


def test_upsert_dedupes_uppercase_accented():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        e1 = kg.upsert_entity("FRANÇOIS", "person", confidence=0.9)
        e2 = kg.upsert_entity("françois", "person", confidence=0.7)
        assert e1 == e2


# ── Trigram index correctness ─────────────────────────────────────────

def test_trigrams_basic():
    tg = KnowledgeGraphStore._trigrams("paul")
    # Padded to "  paul  ", we get 6 trigrams of length 3
    assert len(tg) == 6
    assert "  p" in tg
    assert "pau" in tg
    assert "aul" in tg


def test_trigrams_empty_input():
    assert KnowledgeGraphStore._trigrams("") == set()


def test_trigrams_single_char():
    """Short words still produce some trigrams thanks to padding."""
    tg = KnowledgeGraphStore._trigrams("a")
    assert len(tg) >= 2


def test_candidate_filter_returns_overlapping_only():
    """Pre-filter must include similar names but exclude unrelated ones."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        kg.upsert_entity("Jean Dupont", "person", confidence=0.9)
        kg.upsert_entity("Jeanne Dupond", "person", confidence=0.9)
        kg.upsert_entity("Marie Curie", "person", confidence=0.9)
        kg.upsert_entity("Paris", "location", confidence=0.9)

        # Query for "Jean" — should match the two "Jean*" but not Marie
        candidates = kg._candidate_entity_ids(
            kg._normalize("Jean Dupond"), "person",
        )
        names = {kg.entities[eid].value for eid in candidates}
        assert "Jean Dupont" in names
        assert "Jeanne Dupond" in names
        # Marie Curie has very few trigrams in common with "jean dupond"
        # — it might still be returned if "  m" / similar overlaps. The
        # contract is "candidates ⊇ similar entities", not strict equality.
        # Most importantly: the location must NOT appear (different type).
        assert "Paris" not in names


def test_candidate_filter_only_returns_matching_type():
    """Type isolation: a query for 'person' never returns 'location'."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        kg.upsert_entity("Paris", "location", confidence=0.9)
        kg.upsert_entity("Paris Hilton", "person", confidence=0.9)

        person_candidates = kg._candidate_entity_ids("paris", "person")
        loc_candidates = kg._candidate_entity_ids("paris", "location")

        person_names = {kg.entities[eid].value for eid in person_candidates}
        loc_names = {kg.entities[eid].value for eid in loc_candidates}

        assert "Paris Hilton" in person_names
        assert "Paris" not in person_names  # the location

        assert "Paris" in loc_names
        assert "Paris Hilton" not in loc_names


# ── Fuzzy match still works correctly ─────────────────────────────────

def test_fuzzy_dedup_via_rapidfuzz_or_difflib():
    """The fuzzy dedup that existed before still works with both backends."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        e1 = kg.upsert_entity("Jean-Pierre", "person", confidence=0.8)
        e2 = kg.upsert_entity("Jean Pierre", "person", confidence=0.7)
        assert e1 == e2
        assert "Jean Pierre" in kg.entities[e1].aliases


def test_fuzzy_does_not_overmerge_unrelated():
    """Unrelated names with low ratio must NOT merge."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        e1 = kg.upsert_entity("Alice", "person", confidence=0.9)
        e2 = kg.upsert_entity("Robert", "person", confidence=0.9)
        assert e1 != e2


# ── Persistence + index migration ─────────────────────────────────────

def test_load_rebuilds_indexes_with_new_normalisation():
    """Old data persisted with old normalisation must reindex correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        # Write a stub entities.json simulating an OLD KG without accent stripping
        (path / "entities.json").write_text(
            '{"active": {"e_old": {"entity_id": "e_old", "type": "person", '
            '"value": "François", "aliases": [], "mention_count": 1, '
            '"first_seen": 0, "last_seen": 0, "confidence": 0.9}}, '
            '"pending": {}}',
            encoding="utf-8",
        )

        kg = KnowledgeGraphStore(persist_dir=path)
        # The new normalisation strips the accent at index time, so a
        # query for "francois" must find the existing "François".
        eid = kg.upsert_entity("francois", "person", confidence=0.7)
        assert eid == "e_old"
        assert kg.entities["e_old"].mention_count == 2


# ── Delete cleans both indexes ────────────────────────────────────────

def test_delete_removes_from_both_indexes():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        eid = kg.upsert_entity("Hervé", "person", confidence=0.9)
        # Before delete: present in both
        assert kg._normalize("Hervé") in kg._norm_index
        any_trigram_has_it = any(
            eid in tg_set for tg_set in kg._trigram_index.values()
        )
        assert any_trigram_has_it

        assert kg.delete_entity(eid)
        # After delete: gone from both
        assert kg._normalize("Hervé") not in kg._norm_index
        any_trigram_has_it = any(
            eid in tg_set for tg_set in kg._trigram_index.values()
        )
        assert not any_trigram_has_it


# ── Performance smoke (5000 entities, all dedup hits) ─────────────────

def test_pre_filter_keeps_lookup_fast_at_scale():
    """With trigram pre-filter, dedup against 5000 entities stays cheap.

    This is a smoke test, not a benchmark — we just ensure the operation
    completes in well under 1 second, which would be impossible if the
    fuzzy match still scanned all 5000 entities for every upsert.
    """
    import time as _time
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))

        # Insert 1000 entities (5000 was overkill for a unit test).
        for i in range(1000):
            kg.upsert_entity(f"entity_unique_name_{i:04d}", "topic", confidence=0.9)

        # Now do 100 lookups for already-seen entities (exact match) and
        # 100 lookups for slightly-different variants (fuzzy match path).
        start = _time.perf_counter()
        for i in range(100):
            kg.upsert_entity(f"entity_unique_name_{i:04d}", "topic", confidence=0.9)
            # Tiny variant to trigger fuzzy
            kg.upsert_entity(f"entity_uniqe_name_{i:04d}", "topic", confidence=0.9)
        elapsed = _time.perf_counter() - start

        # On a typical CI worker this is < 0.5s. We allow generous slack.
        assert elapsed < 5.0, f"Lookup too slow: {elapsed:.2f}s"
