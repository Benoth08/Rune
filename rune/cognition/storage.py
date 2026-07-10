"""Storage phase — writing into the 4-memory system.

Biological inspiration
----------------------
After encoding (entorhinal), CA3 of the hippocampus writes a
sparse pattern that binds the cortical features into a recallable
trace. This is the "online write" — fast, content-addressable,
and weighted by salience. The Lythéa equivalent is the SDM
token-by-token write, scaled by composite surprise.

Two distinct write moments coexist:

* **Active write** (during Phase A, before generation): the
  *input* is committed. SDM gets the LLM latents weighted by
  ``S_global × per_token_entropy``. The KG promotes named entities
  and infers co-occurrence relations between persons and other
  typed entities.

* **Archive** (post-generation): the *exchange* (Q+R+atoms) is
  committed to long-term storage. Chroma gets the textual
  document, MHN gets the GLiNER embedding of the query as the
  recall key with the document text as the bound value.

The split mirrors the original ``hippocampe.py`` design: writes
are not deferred to consolidation — only **transfers to neocortex**
(Chroma compression) happen during microsleep.

Design notes
------------
- Every backend call is wrapped: a single failing memory must
  never poison the others. We log at WARNING and continue.
- KG relation predicates are owned by this module (the
  ``_RELATION_PREDICATES`` table). If you add a new entity type to
  GLiNER, mapping it here is a one-line change.
- The phase is stateful only in that it holds references to the
  memory backends. It does not cache anything.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import torch

log = logging.getLogger("rune.cognition.storage")


# Mapping from co-occurring entity type → predicate, for auto-relations
# anchored on a "person" entity. Identical to the inline dict that lived
# in the original ``_phase_a_learn``.
_RELATION_PREDICATES: dict[str, str] = {
    "location": "vit_à",
    "organization": "travaille_chez",
    "project": "travaille_sur",
    "product": "utilise",
    "role": "est",
    "skill": "maîtrise",
}

# SDM per-token write thresholds and scaling — preserved from the
# original active-write loop. Lower than 0.05 entropy = quasi-certain
# token (function words, punctuation) → not worth the SDM slot.
_SDM_ENTROPY_FLOOR: float = 0.05
_SDM_STRENGTH_GAIN: float = 5.0
_SDM_STRENGTH_CAP: float = 3.0

# MHN value truncation — keep recall payload bounded so we don't blow
# memory for very long exchanges. Same 300-char cap as the original.
_MHN_VALUE_MAX_CHARS: int = 300


class StoragePhase:
    """Write the encoded input + the generated exchange to memory.

    Parameters
    ----------
    sdm
        :class:`SparseDistributedMemory`. Must expose ``project``
        and ``write``.
    mhn
        :class:`ModernHopfieldNetwork`. Must expose ``store``.
    kg
        :class:`KnowledgeGraphStore`. Must expose ``upsert_entity``
        and ``add_relation``.
    chroma
        ChromaDB collection. Must expose ``add``.
    model
        :class:`HFModelWrapper`. Used only for ``model_id`` and
        ``hidden_dim`` when projecting SDM keys; never invoked.
    entity_extractor
        :class:`EntityExtractor`. Used to compute the MHN recall key
        (GLiNER embedding of the query) at archive time.
    """

    def __init__(
        self,
        sdm: Any,
        mhn: Any,
        kg: Any,
        chroma: Any,
        model: Any,
        entity_extractor: Any | None,
    ) -> None:
        self.sdm = sdm
        self.mhn = mhn
        self.kg = kg
        self.chroma = chroma
        self.model = model
        self.entity_extractor = entity_extractor

    # ── Active write — during Phase A ──────────────────────────────────

    def write_active(
        self,
        latents: torch.Tensor | None,
        token_entropies: list[float] | None,
        raw_entities: list[dict[str, Any]],
        s_global: float,
    ) -> dict[str, int]:
        """Commit the input to SDM + KG using the encoding outputs.

        Parameters
        ----------
        latents
            Per-token hidden states from
            :class:`~rune.cognition.encoding.EncodingResult`.
            ``None`` → SDM write skipped.
        token_entropies
            Per-token entropies aligned with ``latents``. ``None`` →
            SDM write skipped.
        raw_entities
            Filtered NER hits from the encoding phase. They are
            promoted to the KG and used to seed co-occurrence
            relations.
        s_global
            Composite surprise (``surprise.global``). Weights the
            SDM write strength.

        Returns
        -------
        dict
            ``{sdm_written, entities_promoted, relations_added}``.
            Useful for tests and debug telemetry; ignored by the
            normal pipeline.
        """
        sdm_written = self._write_sdm(latents, token_entropies, s_global)
        entity_ids = self._promote_entities(raw_entities)
        relations_added = self._link_co_occurrences(entity_ids)
        return {
            "sdm_written": sdm_written,
            "entities_promoted": len(entity_ids),
            "relations_added": relations_added,
        }

    # ── Archive — post-generation ──────────────────────────────────────

    def archive_exchange(
        self,
        query: str,
        response: str,
        entities: list[dict[str, Any]],
        surprise: dict[str, float],
        doubt_index: float,
        epistemic: str,
    ) -> bool:
        """Archive a completed (Q, R, atoms) exchange to Chroma + MHN.

        The caller is responsible for deciding *whether* to archive
        (e.g. skip pure questions, skip non-salient inputs). This
        method always archives if invoked, but each backend write
        is independently wrapped — a Chroma failure does not block
        the MHN store, and vice versa.

        Returns
        -------
        bool
            ``True`` if at least one backend accepted the write,
            ``False`` if everything failed (purely informational —
            the pipeline should not branch on this).
        """
        atoms = [e["text"] for e in entities]
        doc = f"Q: {query}\nR: {response}\n[Atoms: {', '.join(atoms)}]"

        chroma_ok = self._archive_chroma(
            doc, surprise, doubt_index, epistemic, len(atoms),
        )
        mhn_ok = self._archive_mhn(query, doc)
        return chroma_ok or mhn_ok

    # ── Internals ──────────────────────────────────────────────────────

    def _write_sdm(
        self,
        latents: torch.Tensor | None,
        token_entropies: list[float] | None,
        s_global: float,
    ) -> int:
        """SDM token-by-token write, weighted by S_global × entropy.

        Returns the count of tokens actually written (after the
        entropy-floor filter). On any failure logs and returns 0.
        """
        if latents is None or token_entropies is None or not getattr(
            self.model, "is_loaded", False
        ):
            log.info(
                "SDM write skipped: latents=%s, token_ents=%s",
                "None" if latents is None else f"shape={latents.shape}",
                "None" if token_entropies is None else f"len={len(token_entropies)}",
            )
            return 0

        sdm_written = 0
        try:
            model_id = self.model.model_id or ""
            for ent, lat in zip(token_entropies, latents):
                if ent < _SDM_ENTROPY_FLOOR:
                    continue
                strength = min(s_global * ent * _SDM_STRENGTH_GAIN, _SDM_STRENGTH_CAP)
                vec = self.sdm.project(
                    lat, model_id=model_id, hidden_dim=self.model.hidden_dim,
                )
                self.sdm.write(vec, vec, strength)
                sdm_written += 1
            log.info(
                "SDM write: %d tokens written (of %d)",
                sdm_written, len(token_entropies),
            )
        except Exception as exc:
            log.warning("Phase A SDM write failed: %s", exc)
        return sdm_written

    def _promote_entities(
        self, raw_entities: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """Upsert each entity into the KG. Returns ``[(entity_id, type)]``."""
        entity_ids: list[tuple[str, str]] = []
        for ent in raw_entities:
            try:
                eid = self.kg.upsert_entity(
                    value=ent["text"],
                    entity_type=ent["label"],
                    confidence=ent["score"],
                )
                entity_ids.append((eid, ent["label"]))
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("KG upsert failed for %r: %s", ent.get("text"), exc)
        return entity_ids

    def _link_co_occurrences(
        self, entity_ids: list[tuple[str, str]],
    ) -> int:
        """Auto-link co-occurring entities anchored on persons.

        For each ``person`` entity in the same utterance, scan the
        other entities and create a typed relation when the predicate
        map has an entry for the object's type.

        Coreference fallback
        --------------------
        When the user writes a "Je..." sentence (e.g. "Je travaille
        chez Anthropic") without naming themselves, GLiNER does not
        emit a person entity. Without a fallback, every such sentence
        contributes zero relations to the KG — but it clearly contains
        new information about the user.

        The fallback: if no person was extracted in this utterance,
        look up the most-recently-mentioned person in the KG (subject
        to a freshness window) and use them as the implicit anchor.
        Relations created via this path are tagged with a separate
        confidence (``coreference_inferred_confidence``, default 0.6)
        so they can be filtered out later if desired.

        Returns the number of relations actually added.
        """
        added = 0
        persons = [(eid, t) for eid, t in entity_ids if t == "person"]

        # Coreference fallback path — only used when no person was
        # extracted explicitly. Returns ``None`` if no eligible
        # interlocutor is in the KG within the freshness window.
        inferred_anchor: str | None = None
        anchor_confidence: float = 0.6
        if not persons:
            inferred_anchor, anchor_confidence = (
                self._find_coreference_anchor()
            )
            if inferred_anchor is None:
                return 0
            persons = [(inferred_anchor, "person")]

        # The actual linking loop — identical for direct and inferred
        # paths, except for the confidence tag.
        for p_id, _ in persons:
            link_confidence = (
                anchor_confidence if p_id == inferred_anchor else 0.6
            )
            for o_id, o_type in entity_ids:
                if o_id == p_id:
                    continue
                pred = _RELATION_PREDICATES.get(o_type)
                if not pred:
                    continue
                try:
                    self.kg.add_relation(
                        p_id, pred, o_id, confidence=link_confidence,
                    )
                    added += 1
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning("KG relation %s failed: %s", pred, exc)
        return added

    def _find_coreference_anchor(self) -> tuple[str | None, float]:
        """Return ``(entity_id, confidence)`` for the implicit "Je" anchor.

        Strategy: pick the person entity with the most recent
        ``last_seen`` timestamp, provided it falls within the
        configured freshness window (defaults to 30 minutes). Beyond
        the window we assume the user has moved on conversationally
        and we don't infer the relation.

        Returns ``(None, 0.0)`` if no eligible anchor exists.
        """
        # Lazy import — keeps the cognition layer free of any direct
        # settings dependency at import time. Cached at module level
        # via ``lru_cache`` in :func:`get_settings`, so calls are cheap.
        from rune.settings import get_settings
        s = get_settings()
        window = s.coreference_window_sec
        confidence = s.coreference_inferred_confidence

        now = time.time()
        best_id: str | None = None
        best_ts: float = 0.0
        for eid, ent in self.kg.entities.items():
            if ent.type != "person":
                continue
            if not ent.last_seen:
                continue
            if (now - ent.last_seen) > window:
                continue
            if ent.last_seen > best_ts:
                best_ts = ent.last_seen
                best_id = eid

        if best_id is None:
            return (None, 0.0)
        return (best_id, confidence)

    def _archive_chroma(
        self,
        doc: str,
        surprise: dict[str, float],
        doubt_index: float,
        epistemic: str,
        atoms_count: int,
    ) -> bool:
        """Add the exchange document to Chroma with full metadata.

        The id pattern ``ex_{ms_epoch}`` is preserved verbatim from
        the original code — UI / debug paths may grep on it.
        """
        try:
            self.chroma.add(
                documents=[doc],
                ids=[f"ex_{int(time.time() * 1000)}"],
                metadatas=[{
                    "type": "exchange",
                    "ts": time.time(),
                    "doubt_index": doubt_index,
                    "epistemic": epistemic,
                    "surprise": surprise.get("global", 0),
                    "atoms_count": atoms_count,
                    # V5.9 rétention : created_ts figé à l'écriture,
                    # last_access_ts rafraîchi à chaque rappel (retrieval).
                    "created_ts": time.time(),
                    "last_access_ts": time.time(),
                }],
            )
            return True
        except Exception as exc:
            log.warning("Chroma archive failed: %s", exc)
            return False

    def _archive_mhn(self, query: str, doc: str) -> bool:
        """Bind GLiNER(query) → doc[:300] in the MHN.

        We use the **input** embedding (not the response) because
        that is the key the MHN will be queried with at retrieval
        time — energy matching is symmetric only when the keys come
        from the same distribution.
        """
        if self.entity_extractor is None:
            return False
        try:
            gliner_emb_input = self.entity_extractor.encode(query)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("GLiNER encode for MHN store failed: %s", exc)
            return False
        if gliner_emb_input is None:
            return False
        try:
            self.mhn.store(gliner_emb_input, doc[:_MHN_VALUE_MAX_CHARS])
            return True
        except Exception as exc:
            log.warning("MHN store failed: %s", exc)
            return False
