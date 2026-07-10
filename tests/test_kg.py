"""Tests for the Knowledge Graph store."""
from __future__ import annotations

import tempfile
from pathlib import Path

from rune.memory.kg import KnowledgeGraphStore


def test_upsert_new_entity():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        eid = kg.upsert_entity("Jean Dupont", "person", confidence=0.8)
        assert eid in kg.entities
        assert kg.entities[eid].value == "Jean Dupont"
        assert kg.entities[eid].mention_count == 1


def test_upsert_exact_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        eid1 = kg.upsert_entity("Paris", "location", confidence=0.8)
        eid2 = kg.upsert_entity("paris", "location", confidence=0.6)
        assert eid1 == eid2
        assert kg.entities[eid1].mention_count == 2


def test_upsert_fuzzy_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        eid1 = kg.upsert_entity("Jean-Pierre", "person", confidence=0.8)
        eid2 = kg.upsert_entity("Jean Pierre", "person", confidence=0.7)
        assert eid1 == eid2
        assert "Jean Pierre" in kg.entities[eid1].aliases


def test_pending_below_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        eid = kg.upsert_entity("Maybe", "topic", confidence=0.2)
        assert eid not in kg.entities
        assert eid in kg.pending


def test_add_relation():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        e1 = kg.upsert_entity("Mika", "person", confidence=0.9)
        e2 = kg.upsert_entity("Rune", "project", confidence=0.9)
        rid = kg.add_relation(e1, "works_on", e2, confidence=0.8)
        assert rid in kg.relations


def test_query_by_question():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        e1 = kg.upsert_entity("Mika", "person", confidence=0.9)
        e2 = kg.upsert_entity("SP3H", "organization", confidence=0.9)
        kg.add_relation(e1, "worked_at", e2)

        facts = kg.query_by_question(
            "Où a travaillé Mika ?",
            [{"text": "Mika", "label": "person", "score": 0.9}],
        )
        assert len(facts) > 0
        assert any("Mika" in f for f in facts)


def test_cleanup_pending():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        eid = kg.upsert_entity("ephemeral", "topic", confidence=0.2)
        # Force expire
        kg.pending[eid].last_seen = 0
        cleaned = kg.cleanup_pending()
        assert cleaned == 1
        assert eid not in kg.pending


def test_persist_and_reload():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        kg.upsert_entity("Test Entity", "topic", confidence=0.9)
        kg.save()

        kg2 = KnowledgeGraphStore(persist_dir=Path(tmp))
        assert len(kg2.entities) == 1


def test_delete_entity():
    with tempfile.TemporaryDirectory() as tmp:
        kg = KnowledgeGraphStore(persist_dir=Path(tmp))
        eid = kg.upsert_entity("ToDelete", "person", confidence=0.9)
        assert kg.delete_entity(eid)
        assert eid not in kg.entities
