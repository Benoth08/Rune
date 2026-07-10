"""Critical safety facts must bypass the `pending` purgatory and surface
as vital constraints. (config → settings needs pydantic; rapidfuzz for the KG.)"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("rapidfuzz")


def _kg(tmp_path):
    from rune.memory.kg import KnowledgeGraphStore
    return KnowledgeGraphStore(persist_dir=tmp_path)


def test_critical_type_bypasses_pending_at_low_score(tmp_path):
    kg = _kg(tmp_path)
    eid = kg.upsert_entity("arachides", "allergy", confidence=0.20)   # < 0.5
    assert eid in kg.entities          # active despite weak score
    assert eid not in kg.pending
    assert any("arachides" in c for c in kg.critical_constraints())


def test_non_critical_low_score_still_pending(tmp_path):
    kg = _kg(tmp_path)
    eid = kg.upsert_entity("un sujet anodin", "topic", confidence=0.20)
    assert eid in kg.pending           # ordinary facts unchanged
    assert eid not in kg.entities


def test_critical_constraints_lists_all_safety_types(tmp_path):
    kg = _kg(tmp_path)
    kg.upsert_entity("sans gluten", "dietary_restriction", confidence=0.1)
    kg.upsert_entity("asthme", "medical_condition", confidence=0.1)
    blob = " ".join(kg.critical_constraints())
    assert "sans gluten" in blob and "asthme" in blob


def test_merge_entity_passes_dedupes_keeping_best_score():
    from rune.memory.kg import merge_entity_passes
    passes = [
        [{"text": "Framatome", "label": "organization", "score": 0.6}],
        [{"text": "Framatome", "label": "institution", "score": 0.9}],  # higher
        [{"text": "Aix", "label": "location", "score": 0.7}],
    ]
    out = {e["text"]: e for e in merge_entity_passes(passes)}
    assert out["Framatome"]["label"] == "institution"   # best score wins
    assert out["Framatome"]["score"] == 0.9
    assert "Aix" in out


def test_merge_entity_passes_sensitive_filter():
    from rune.memory.kg import merge_entity_passes
    passes = [[{"text": "catholique", "label": "belief", "score": 0.8},
               {"text": "jazz", "label": "preference", "score": 0.8}]]
    on = {e["text"] for e in merge_entity_passes(passes, capture_sensitive=True)}
    off = {e["text"] for e in merge_entity_passes(passes, capture_sensitive=False)}
    assert "catholique" in on and "jazz" in on
    assert "catholique" not in off and "jazz" in off    # sensitive dropped


def test_taxonomy_groups_well_formed():
    from rune.config import (
        GLINER_LABEL_GROUPS, GLINER_LABELS, CRITICAL_ENTITY_TYPES,
    )
    # every pass stays in GLiNER's precision sweet spot
    assert all(len(g) <= 13 for g in GLINER_LABEL_GROUPS)
    # flat list is the dedup union
    flat = [l for g in GLINER_LABEL_GROUPS for l in g]
    assert set(GLINER_LABELS) == set(flat)
    assert len(GLINER_LABELS) == len(set(flat))          # no dup leaks
    # every critical type is actually present in the taxonomy
    assert CRITICAL_ENTITY_TYPES <= set(GLINER_LABELS)
    assert "topic" in GLINER_LABELS                       # catch-all kept


def test_catalog_has_coder_14b():
    from rune.config import CATALOG
    spec = CATALOG.get("Qwen/Qwen2.5-Coder-14B-Instruct")
    assert spec is not None, "Coder-14B absent du catalogue"
    assert spec.quant_4bit is True
    # Un second modèle Coder doit rester dans le catalogue (le 7B).
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in CATALOG
