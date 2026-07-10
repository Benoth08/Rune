"""Unit tests for :class:`rune.cognition.storage.StoragePhase`.

Two write moments are exercised:

* ``write_active`` — SDM token-by-token + KG promotion + relations.
* ``archive_exchange`` — Chroma exchange document + MHN store.

Both are tested in isolation with mocks. Failure-isolation is a key
contract: a Chroma failure must not block the MHN store, and a SDM
failure must not block KG promotion.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch", reason="latent paths need tensor I/O")

from rune.cognition.storage import (  # noqa: E402
    _RELATION_PREDICATES,
    StoragePhase,
)


def _make_storage(
    *,
    extractor: MagicMock | None = None,
    model_loaded: bool = True,
) -> tuple[StoragePhase, dict[str, MagicMock]]:
    """Build a StoragePhase with all backends mocked.

    Returns the phase plus a dict of every backend mock so tests can
    assert calls against them.
    """
    sdm = MagicMock()
    sdm.project.side_effect = lambda lat, model_id, hidden_dim: torch.zeros(8)
    mhn = MagicMock()
    kg = MagicMock()
    # KG returns a deterministic id per call so co-occurrence linking
    # produces predictable subjects/objects.
    kg.upsert_entity.side_effect = [f"e_{i}" for i in range(20)]
    chroma = MagicMock()
    model = MagicMock()
    model.is_loaded = model_loaded
    model.model_id = "test-model"
    model.hidden_dim = 16

    if extractor is None:
        extractor = MagicMock()
        extractor.encode.return_value = torch.zeros(8)

    phase = StoragePhase(
        sdm=sdm, mhn=mhn, kg=kg, chroma=chroma,
        model=model, entity_extractor=extractor,
    )
    return phase, {"sdm": sdm, "mhn": mhn, "kg": kg, "chroma": chroma,
                   "model": model, "extractor": extractor}


# ── write_active : SDM ─────────────────────────────────────────────────

def test_write_active_writes_one_per_above_floor_token():
    """Tokens with entropy ≥ 0.05 → SDM write. Tokens below → skipped."""
    phase, m = _make_storage()
    latents = torch.randn(5, 16)
    # 3 tokens above floor, 2 below
    entropies = [0.5, 0.01, 0.3, 0.04, 0.2]
    out = phase.write_active(
        latents=latents, token_entropies=entropies,
        raw_entities=[], s_global=0.5,
    )
    assert out["sdm_written"] == 3
    assert m["sdm"].write.call_count == 3


def test_write_active_skips_when_model_not_loaded():
    phase, m = _make_storage(model_loaded=False)
    out = phase.write_active(
        latents=torch.randn(3, 16),
        token_entropies=[0.3, 0.4, 0.5],
        raw_entities=[], s_global=0.5,
    )
    assert out["sdm_written"] == 0
    m["sdm"].write.assert_not_called()


def test_write_active_skips_when_latents_none():
    phase, m = _make_storage()
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=[], s_global=0.5,
    )
    assert out["sdm_written"] == 0
    m["sdm"].write.assert_not_called()


def test_write_active_sdm_strength_capped():
    """Strength must clip at 3.0 even when surprise × entropy explodes."""
    phase, m = _make_storage()
    latents = torch.randn(1, 16)
    # s_global=1.0, ent=1.0 → 1.0*1.0*5.0 = 5.0 → cap to 3.0
    phase.write_active(
        latents=latents, token_entropies=[1.0],
        raw_entities=[], s_global=1.0,
    )
    _, _, strength = m["sdm"].write.call_args[0]
    assert strength == pytest.approx(3.0)


def test_write_active_sdm_failure_does_not_propagate():
    """A SDM exception must be swallowed; the loop logs and stops."""
    phase, m = _make_storage()
    m["sdm"].project.side_effect = RuntimeError("boom")
    out = phase.write_active(
        latents=torch.randn(2, 16), token_entropies=[0.3, 0.4],
        raw_entities=[], s_global=0.5,
    )
    # No tokens make it through because project fails first
    assert out["sdm_written"] == 0


# ── write_active : KG promotion + relations ────────────────────────────

def test_write_active_promotes_entities():
    phase, m = _make_storage()
    entities = [
        {"text": "Mika", "label": "person", "score": 0.95},
        {"text": "Aix", "label": "location", "score": 0.9},
    ]
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["entities_promoted"] == 2
    assert m["kg"].upsert_entity.call_count == 2


def test_write_active_links_person_location_with_vit_a():
    """Person + location in same utterance → 'vit_à' relation."""
    phase, m = _make_storage()
    entities = [
        {"text": "Mika", "label": "person", "score": 0.95},
        {"text": "Aix", "label": "location", "score": 0.9},
    ]
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["relations_added"] == 1
    args = m["kg"].add_relation.call_args
    # signature: add_relation(subject_id, predicate, object_id, confidence=...)
    assert args.args[1] == "vit_à"


def test_write_active_no_relation_when_no_person():
    """Without any person anchor, no relation is created."""
    phase, m = _make_storage()
    entities = [
        {"text": "Aix", "label": "location", "score": 0.9},
        {"text": "RunPod", "label": "product", "score": 0.8},
    ]
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["relations_added"] == 0
    m["kg"].add_relation.assert_not_called()


def test_write_active_links_all_typed_co_occurrences():
    """One person + one of each predicate type → one relation each."""
    phase, m = _make_storage()
    entities = [{"text": "Mika", "label": "person", "score": 0.95}]
    for label in _RELATION_PREDICATES:
        entities.append({"text": label.title(), "label": label, "score": 0.8})

    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["relations_added"] == len(_RELATION_PREDICATES)


def test_write_active_unknown_entity_type_not_linked():
    """Entity of a type not in the predicate map → ignored for linking."""
    phase, m = _make_storage()
    entities = [
        {"text": "Mika", "label": "person", "score": 0.95},
        {"text": "Stuff", "label": "unobtanium", "score": 0.8},  # unknown type
    ]
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["relations_added"] == 0


# ── Coreference fallback (fix #4) ──────────────────────────────────────

def _kg_with_recent_person(eid: str, value: str, age_seconds: float):
    """Build a MagicMock KG whose ``entities`` dict holds a person seen
    ``age_seconds`` ago. The returned mock can be plugged via
    ``phase.kg = ...``."""
    import time as _time
    from types import SimpleNamespace
    kg = MagicMock()
    kg.entities = {
        eid: SimpleNamespace(
            type="person",
            value=value,
            last_seen=_time.time() - age_seconds,
        ),
    }
    return kg


def test_coreference_anchor_used_when_no_person_extracted():
    """When the user writes "Je travaille chez X" without naming
    themselves, GLiNER emits no person — but the most-recent person
    in the KG should be used as the implicit anchor and the relation
    should still be created."""
    phase, m = _make_storage()
    # Replace the bare MagicMock kg with one that has a recent Mika.
    phase.kg = _kg_with_recent_person("e_mika", "Mika", age_seconds=60)
    # Simulate "Je travaille chez Anthropic" — no person extracted,
    # only the org. The fallback should kick in.
    entities = [
        {"text": "Anthropic", "label": "organization", "score": 0.9},
    ]
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["relations_added"] == 1
    args = phase.kg.add_relation.call_args
    # Anchor should be Mika (the inferred coreference target).
    assert args.args[0] == "e_mika"
    assert args.args[1] == "travaille_chez"
    # Confidence should match the inferred-fallback default (0.6).
    assert args.kwargs.get("confidence") == pytest.approx(0.6)


def test_coreference_anchor_skipped_outside_window():
    """If the most-recent person was seen too long ago, the fallback
    must not fire — we assume the user has moved on."""
    phase, m = _make_storage()
    # Mika was seen 2 hours ago — beyond the 30-min default window.
    phase.kg = _kg_with_recent_person("e_mika", "Mika", age_seconds=7200)
    entities = [
        {"text": "Anthropic", "label": "organization", "score": 0.9},
    ]
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["relations_added"] == 0
    phase.kg.add_relation.assert_not_called()


def test_coreference_anchor_skipped_when_no_person_in_kg():
    """If the KG has no person at all, the fallback returns gracefully."""
    phase, m = _make_storage()
    kg = MagicMock()
    kg.entities = {}  # empty KG
    phase.kg = kg
    entities = [
        {"text": "Anthropic", "label": "organization", "score": 0.9},
    ]
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["relations_added"] == 0
    kg.add_relation.assert_not_called()


def test_coreference_does_not_alter_direct_path():
    """When a person IS extracted in the message, the fallback is not
    triggered — the direct path is used and confidence stays at the
    historical 0.6 (not the inferred-fallback value, even if the same)."""
    phase, m = _make_storage()
    # Even if the KG also has a stale Mika, the fresh extraction wins.
    phase.kg = _kg_with_recent_person("e_old", "Stale", age_seconds=120)
    # Reset add_relation mock since we replaced the kg.
    entities = [
        {"text": "Mika", "label": "person", "score": 0.95},
        {"text": "Anthropic", "label": "organization", "score": 0.9},
    ]
    # upsert_entity returns "e_0" for Mika and "e_1" for Anthropic
    # (deterministic mock side_effect from _make_storage).
    phase.kg.upsert_entity.side_effect = [f"e_{i}" for i in range(20)]
    out = phase.write_active(
        latents=None, token_entropies=None,
        raw_entities=entities, s_global=0.5,
    )
    assert out["relations_added"] == 1
    args = phase.kg.add_relation.call_args
    # Anchor should be the freshly-extracted Mika ("e_0"), not the
    # stale KG entry ("e_old").
    assert args.args[0] == "e_0"


# ── archive_exchange ───────────────────────────────────────────────────

def test_archive_writes_to_chroma_and_mhn():
    phase, m = _make_storage()
    ok = phase.archive_exchange(
        query="Tu connais Mika ?",
        response="Oui, c'est toi.",
        entities=[{"text": "Mika", "label": "person", "score": 0.9}],
        surprise={"global": 0.42},
        doubt_index=0.1,
        epistemic="fait",
    )
    assert ok is True
    assert m["chroma"].add.call_count == 1
    assert m["mhn"].store.call_count == 1


def test_archive_chroma_payload_shape():
    """Chroma must receive a single doc, single id, single metadata."""
    phase, m = _make_storage()
    phase.archive_exchange(
        query="q", response="r",
        entities=[{"text": "X", "label": "t", "score": 1.0}],
        surprise={"global": 0.5},
        doubt_index=0.2, epistemic="intuition",
    )
    kwargs = m["chroma"].add.call_args.kwargs
    assert len(kwargs["documents"]) == 1
    assert len(kwargs["ids"]) == 1
    assert len(kwargs["metadatas"]) == 1
    meta = kwargs["metadatas"][0]
    assert meta["type"] == "exchange"
    assert meta["doubt_index"] == 0.2
    assert meta["epistemic"] == "intuition"
    assert meta["surprise"] == 0.5
    assert meta["atoms_count"] == 1
    assert kwargs["ids"][0].startswith("ex_")
    assert "Q: q" in kwargs["documents"][0]
    assert "R: r" in kwargs["documents"][0]
    assert "[Atoms: X]" in kwargs["documents"][0]


def test_archive_mhn_uses_input_embedding():
    """MHN key must be the embedding of the QUERY, not the response.

    Recall comes in via the next user utterance, so the key must
    come from the same distribution as the recall probe.
    """
    phase, m = _make_storage()
    phase.archive_exchange(
        query="hello world", response="bonjour",
        entities=[], surprise={"global": 0.3},
        doubt_index=0.0, epistemic="fait",
    )
    # extractor.encode must have been called with the query
    m["extractor"].encode.assert_called_with("hello world")


def test_archive_mhn_value_truncated_to_300():
    phase, m = _make_storage()
    phase.archive_exchange(
        query="q", response="x" * 1000,
        entities=[], surprise={"global": 0.3},
        doubt_index=0.0, epistemic="fait",
    )
    _, value = m["mhn"].store.call_args.args
    assert len(value) <= 300


def test_archive_chroma_failure_does_not_block_mhn():
    """Failure isolation: Chroma down → MHN still writes."""
    phase, m = _make_storage()
    m["chroma"].add.side_effect = RuntimeError("chroma down")
    ok = phase.archive_exchange(
        query="q", response="r",
        entities=[], surprise={"global": 0.3},
        doubt_index=0.0, epistemic="fait",
    )
    assert ok is True            # MHN succeeded
    m["mhn"].store.assert_called_once()


def test_archive_mhn_failure_does_not_block_chroma():
    """Failure isolation: MHN raises → Chroma still went through."""
    phase, m = _make_storage()
    m["mhn"].store.side_effect = RuntimeError("dim mismatch")
    ok = phase.archive_exchange(
        query="q", response="r",
        entities=[], surprise={"global": 0.3},
        doubt_index=0.0, epistemic="fait",
    )
    assert ok is True            # Chroma succeeded
    m["chroma"].add.assert_called_once()


def test_archive_no_extractor_skips_mhn_keeps_chroma():
    extractor = None
    phase = StoragePhase(
        sdm=MagicMock(), mhn=MagicMock(), kg=MagicMock(),
        chroma=MagicMock(),
        model=MagicMock(is_loaded=True, model_id="m", hidden_dim=8),
        entity_extractor=extractor,
    )
    ok = phase.archive_exchange(
        query="q", response="r",
        entities=[], surprise={"global": 0.3},
        doubt_index=0.0, epistemic="fait",
    )
    assert ok is True
    phase.mhn.store.assert_not_called()
    phase.chroma.add.assert_called_once()


def test_archive_extractor_returns_none_skips_mhn():
    """If GLiNER returns None (e.g. very short text), skip MHN store."""
    phase, m = _make_storage()
    m["extractor"].encode.return_value = None
    phase.archive_exchange(
        query="q", response="r",
        entities=[], surprise={"global": 0.3},
        doubt_index=0.0, epistemic="fait",
    )
    m["mhn"].store.assert_not_called()
    m["chroma"].add.assert_called_once()


def test_archive_both_fail_returns_false():
    phase, m = _make_storage()
    m["chroma"].add.side_effect = RuntimeError("a")
    m["mhn"].store.side_effect = RuntimeError("b")
    ok = phase.archive_exchange(
        query="q", response="r",
        entities=[], surprise={"global": 0.3},
        doubt_index=0.0, epistemic="fait",
    )
    assert ok is False
