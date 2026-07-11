"""Knowledge Graph — persistent identity memory via GLiNER NER.

Extracts entities and relations from text, stores them with fuzzy
deduplication, and provides targeted fact retrieval.

Performance and matching
------------------------
Entity deduplication uses a two-stage approach:

1. **Exact match** on a normalised form of the value, via a hash index
   (``_norm_index``). O(1) lookup. Normalisation strips whitespace,
   lower-cases, AND removes accents (so "François" and "francois"
   match).

2. **Fuzzy match** on same-type entities only. To keep this scalable
   beyond a few thousand entities, we maintain a trigram inverted
   index (``_trigram_index``): each entity contributes its set of
   character-trigrams to a ``trigram → set of entity_ids`` map.
   At lookup time we only score candidates that share at least one
   trigram with the query. This typically reduces the candidate
   set from O(n) to O(k) where k ≈ 5–20.

3. The fuzzy score itself uses ``rapidfuzz.fuzz.ratio`` when
   ``rapidfuzz`` is installed (50× faster than ``difflib.SequenceMatcher``
   on short strings) with a transparent fallback to ``difflib``.

Migration
---------
Old persisted KGs (pre-accent-stripping) are migrated transparently.
At load, ``_rebuild_index`` regenerates both ``_norm_index`` and
``_trigram_index`` from current entities, so any new normalisation
rule applies to historical data without manual migration.
"""
from __future__ import annotations

import json
import logging
import os
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Fuzzy matcher: prefer rapidfuzz (much faster) but fall back to the
# stdlib difflib so the project still works if rapidfuzz isn't installed.
try:
    from rapidfuzz import fuzz as _rapidfuzz
    _FUZZ_BACKEND = "rapidfuzz"

    def _ratio(a: str, b: str) -> float:
        # rapidfuzz returns 0..100; normalise to 0..1 to match difflib API
        return _rapidfuzz.ratio(a, b) / 100.0
except ImportError:
    import difflib as _difflib
    _FUZZ_BACKEND = "difflib"

    def _ratio(a: str, b: str) -> float:
        return _difflib.SequenceMatcher(None, a, b).ratio()

from rune.config import (
    CRITICAL_ENTITY_TYPES,
    GLINER_LABEL_GROUPS,
    GLINER_LABELS,
    GLINER_MODEL,
    KG_ACTIVE_THRESHOLD,
    KG_DIR,
    KG_FUZZY_THRESHOLD,
    KG_PENDING_TTL_HOURS,
    SENSITIVE_ENTITY_TYPES,
)

log = logging.getLogger("rune.memory.kg")


def merge_entity_passes(passes: list, capture_sensitive: bool = True) -> list[dict]:
    """Merge per-pass GLiNER results into one deduped entity list.

    Multi-pass extraction runs the model once per thematic label group; a span
    may surface in several passes (different labels). We dedupe by lowercased
    text, keeping the highest-scoring label, and optionally drop GDPR-sensitive
    types. Pure function → unit-testable without the model."""
    merged: dict[str, dict] = {}
    for ents in passes:
        for e in ents or []:
            key = (e.get("text") or "").strip().lower()
            if not key:
                continue
            if not capture_sensitive and e.get("label") in SENSITIVE_ENTITY_TYPES:
                continue
            prev = merged.get(key)
            if prev is None or e.get("score", 0.0) > prev.get("score", 0.0):
                merged[key] = {"text": e["text"], "label": e["label"],
                               "score": e["score"]}
    return list(merged.values())


@dataclass
class KGEntity:
    """A named entity in the knowledge graph."""

    entity_id: str
    type: str
    value: str
    aliases: list[str] = field(default_factory=list)
    mention_count: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    confidence: float = 0.5


@dataclass
class KGRelation:
    """A relation between entities or values."""

    rel_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 0.5
    source_doc: str = ""
    created_ts: float = 0.0  # V5.9 — départage par récence
    superseded: bool = False  # V5.9 — fait périmé (maj) → ignoré au retrieval


# V5.9 — prédicats fonctionnels (mono-valués) : un nouveau fait périme
# l'ancien (changement d'employeur, de ville). Les autres (projets,
# produits, compétences) sont multi-valués et s'accumulent.
_FUNCTIONAL_PREDICATES = frozenset({"travaille_chez", "vit_à"})


class EntityExtractor:
    """GLiNER-based entity extraction with shared DeBERTa backbone.

    The backbone is also used as an embedder (768-dim) for the MHN.
    """

    def __init__(self, model_name: str = GLINER_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        # Profil de labels : lu depuis les settings (défaut "chat"). Le
        # profil détermine quels groupes de labels GLiNER sont utilisés —
        # conversationnels ("chat") ou techniques ("agent"). L'agent peut
        # basculer à chaud via set_label_profile() pendant une mission.
        from rune.config import gliner_label_groups
        try:
            from rune.settings import get_settings
            self._profile = get_settings().kg_label_profile or "chat"
        except Exception:  # noqa: BLE001
            self._profile = "chat"
        self._label_groups = [list(g) for g in gliner_label_groups(self._profile)]
        self._labels = list(dict.fromkeys(
            lbl for g in self._label_groups for lbl in g))
        self._st_model = None
        self._st_tried = False

        # Bounded cache for encode() — many call sites encode the same
        # query during a single turn (Phase A salience, Phase B retrieval,
        # post-generation MHN store). The cache size is configurable;
        # 0 disables.
        from rune.cache import BoundedCache
        from rune.settings import get_settings
        cache_size = get_settings().embed_cache_size
        self._encode_cache: BoundedCache = BoundedCache(
            max_size=cache_size, name="entity_extractor.encode",
        )

    def set_label_profile(self, profile: str) -> None:
        """Bascule le profil de labels GLiNER à chaud (idempotent).

        Utilisé par l'agent pour passer en profil technique le temps d'une
        mission, puis revenir. Ne recharge PAS le modèle (GLiNER prend les
        labels à chaque appel de extract()), donc c'est instantané.
        """
        from rune.config import gliner_label_groups
        p = (profile or "chat").lower()
        if p == getattr(self, "_profile", None):
            return
        self._profile = p
        self._label_groups = [list(g) for g in gliner_label_groups(p)]
        self._labels = list(dict.fromkeys(
            lbl for g in self._label_groups for lbl in g))

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Mémoïse l'échec : sans gliner installé, _model reste None et on
        # retenterait (+ reloggerait) à CHAQUE appel — plusieurs fois par
        # message. On ne tente qu'une fois, on logge une fois, puis on
        # abandonne silencieusement (le KG se remplit alors moins, mais
        # le pipeline continue).
        if getattr(self, "_load_failed", False):
            return
        try:
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(self._model_name)
            log.info("GLiNER loaded: %s", self._model_name)
        except Exception as exc:
            self._load_failed = True
            # Message distingue "pas installé" de "installé mais échoue au
            # chargement" (ex: tokenizer SentencePiece nécessitant le
            # paquet sentencepiece, ou conflit de version transformers).
            import importlib.util
            if importlib.util.find_spec("gliner") is None:
                hint = "paquet absent — `pip install gliner`"
            else:
                hint = f"installé mais chargement échoué : {exc}"
            log.warning(
                "GLiNER indisponible (%s) — extraction d'entités désactivée "
                "pour cette session (%s).",
                type(exc).__name__, hint,
            )

    def extract(self, text: str) -> list[dict]:
        """Extract entities from text.

        Returns
        -------
        list[dict]
            Each dict: ``{text, label, score}``.
        """
        self._ensure_loaded()
        if self._model is None:
            return []

        # Multi-pass: one GLiNER call per thematic group (≤~12 labels) keeps the
        # uni-encoder in its precision sweet spot; results merged + deduped.
        try:
            from rune.settings import get_settings
            capture_sensitive = bool(get_settings().kg_capture_sensitive)
        except Exception:  # noqa: BLE001
            capture_sensitive = True
        passes = []
        for group in self._label_groups:
            try:
                passes.append(
                    self._model.predict_entities(text, group, threshold=0.3))
            except Exception as exc:  # noqa: BLE001
                log.warning("GLiNER extraction error (group): %s", exc)
        return merge_entity_passes(passes, capture_sensitive)

    def encode(self, text: str) -> "torch.Tensor | None":
        """Encode text to 768-dim embedding (cached).

        Tries sentence-transformers first (robust), then GLiNER backbone,
        then falls back to deterministic hash embedding. Results are
        memoised in a bounded LRU cache so repeated calls within the
        cognitive cycle (Phase A → Phase B → post-gen) don't re-compute.
        """
        # Cache layer — see BoundedCache.get_or_compute. Returning None
        # from compute() means "don't cache" (transient failure).
        cached = self._encode_cache.get(text)
        if cached is not None:
            return cached
        value = self._encode_uncached(text)
        if value is not None:
            self._encode_cache.put(text, value)
        return value

    def _encode_uncached(self, text: str) -> "torch.Tensor | None":
        """Actual encoding logic (no cache). Public ``encode`` wraps this."""
        import torch

        # Try sentence-transformers (most reliable)
        if self._st_model is None and not self._st_tried:
            self._st_tried = True
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                )
                log.info("SentenceTransformer loaded for encoding")
            except Exception:
                pass

        if self._st_model is not None:
            try:
                emb = self._st_model.encode(text, convert_to_tensor=True)
                # Pad or project to 768 if needed
                if emb.shape[0] != 768:
                    padded = torch.zeros(768)
                    padded[:emb.shape[0]] = emb[:768]
                    return padded.cpu()
                return emb.cpu()
            except Exception:
                pass

        return self._hash_embed(text)

    def cache_stats(self) -> dict:
        """Return embedding cache statistics."""
        return self._encode_cache.stats()

    @staticmethod
    def _hash_embed(text: str, dim: int = 768) -> "torch.Tensor":
        """Deterministic pseudo-embedding from text hash (fallback)."""
        import hashlib
        import torch
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(dim, generator=gen)
        return emb / emb.norm().clamp(min=1e-8)


class KnowledgeGraphStore:
    """Persistent knowledge graph with fuzzy entity deduplication.

    Parameters
    ----------
    persist_dir : Path
        Directory for JSON persistence.
    """

    def __init__(self, persist_dir: Path = KG_DIR) -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._entity_path = persist_dir / "entities.json"
        self._relation_path = persist_dir / "relations.json"
        # V5.2 — GraphRAG communities are detected + summarised during
        # microsleep, then cached here for cheap retrieval-time access.
        self._communities_path = persist_dir / "communities.json"

        self.entities: dict[str, KGEntity] = {}
        self.relations: dict[str, KGRelation] = {}
        self.pending: dict[str, KGEntity] = {}
        # V5.2 — Detected communities (list[Community]). Empty until
        # the first microsleep with enough entities triggers detection.
        self.communities: list = []

        # Normalized value → entity_id index for fast exact match
        self._norm_index: dict[str, str] = {}
        # Trigram inverted index: trigram → set of entity_ids that contain it.
        # Used to pre-filter fuzzy match candidates (O(n) → O(k)).
        self._trigram_index: dict[str, set[str]] = {}

        self._load()
        log.info("KG loaded with fuzzy backend: %s", _FUZZ_BACKEND)

    # ── V5.8.1 — Clear en place (préserve les références externes) ────

    def clear_in_place(self) -> dict[str, int]:
        """Vide toutes les structures internes sans casser les références.

        Critique pour le wipe : si on réinstancie un nouveau KG via
        ``KnowledgeGraphStore()``, tous les modules qui ont reçu le KG
        au moment de l'init (StoragePhase, RetrievalPhase, ReasoningGen,
        DeepReasoning, etc.) gardent une **référence figée vers
        l'ancien KG**. Résultat : le storage écrit dans l'ancien, le
        retrieval lit l'ancien, le compteur (qui pointe vers le nouveau)
        affiche 0. Bug observé in vivo en V5.8.1.

        ``clear_in_place()`` vide les dicts/listes sur place. Toutes les
        références externes pointent toujours sur la même instance, qui
        est maintenant vide. Comportement attendu : count=0 partout +
        écritures futures repartent d'un état propre.

        Returns
        -------
        dict[str, int]
            Compteurs avant clear (pour reporting de wipe).
        """
        counts = {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "pending": len(self.pending),
            "communities": len(self.communities),
            "norm_index": len(self._norm_index),
            "trigram_index": len(self._trigram_index),
        }
        self.entities.clear()
        self.relations.clear()
        self.pending.clear()
        self.communities.clear()
        self._norm_index.clear()
        self._trigram_index.clear()
        log.info(
            "KG cleared in-place: %d entities, %d relations, "
            "%d pending, %d communities removed",
            counts["entities"], counts["relations"],
            counts["pending"], counts["communities"],
        )
        return counts

    # ── Normalize for matching ─────────────────────────────────────────

    @staticmethod
    def _normalize(value: str) -> str:
        """Lower-case, strip whitespace, remove accents.

        We use NFKD decomposition to split accented characters into
        base+combining-mark, then drop combining marks. This makes
        "François", "francois" and "FRANÇOIS" all collapse to the
        same normalised form.
        """
        if not value:
            return ""
        # NFKD: decompose composed characters into base + combining marks
        # (e.g. 'é' → 'e' + U+0301). Then we filter out combining marks
        # (Unicode category Mn — "Mark, nonspacing").
        decomposed = unicodedata.normalize("NFKD", value)
        stripped = "".join(
            ch for ch in decomposed
            if not unicodedata.combining(ch)
        )
        return stripped.strip().lower()

    @staticmethod
    def _trigrams(value: str) -> set[str]:
        """Return the set of character trigrams of a normalised value.

        We pad with two leading and two trailing spaces so short words
        and word boundaries get reasonable trigram coverage. Example::

            "paul" → {"  p", " pa", "pau", "aul", "ul ", "l  "}

        For values shorter than 3 chars we degrade gracefully to whatever
        the padding produces.
        """
        if not value:
            return set()
        padded = "  " + value + "  "
        return {padded[i:i + 3] for i in range(len(padded) - 2)}

    def _index_entity(self, entity_id: str, value: str) -> None:
        """Add a value to both the norm index and the trigram index."""
        norm = self._normalize(value)
        if not norm:
            return
        self._norm_index[norm] = entity_id
        for tg in self._trigrams(norm):
            self._trigram_index.setdefault(tg, set()).add(entity_id)

    def _unindex_entity(self, entity_id: str) -> None:
        """Remove an entity from both indexes (used in delete)."""
        # Drop from norm_index by value
        norms_to_drop = [
            n for n, eid in self._norm_index.items() if eid == entity_id
        ]
        for n in norms_to_drop:
            self._norm_index.pop(n, None)
        # Drop from trigram_index — iterate the trigram values for this entity
        for tg_set in self._trigram_index.values():
            tg_set.discard(entity_id)

    def _rebuild_index(self) -> None:
        """Rebuild both indexes from scratch.

        Called on load (so historical data picks up any new normalisation
        rule like accent stripping) and after bulk modifications.
        """
        self._norm_index.clear()
        self._trigram_index.clear()
        for eid, ent in self.entities.items():
            self._index_entity(eid, ent.value)
            for alias in ent.aliases:
                self._index_entity(eid, alias)

    def _candidate_entity_ids(self, query_norm: str, entity_type: str) -> set[str]:
        """Return entity_ids that share at least one trigram with the query.

        Restricts to the requested entity_type (since our fuzzy match
        is type-scoped). If the query has no trigrams (e.g. very short
        value), falls back to all same-type entities so we don't lose
        recall on edge cases.
        """
        query_tgs = self._trigrams(query_norm)
        if not query_tgs:
            return {
                eid for eid, ent in self.entities.items()
                if ent.type == entity_type
            }
        candidates: set[str] = set()
        for tg in query_tgs:
            if tg in self._trigram_index:
                candidates.update(self._trigram_index[tg])
        # Filter by entity_type without rescanning all entities
        return {
            eid for eid in candidates
            if eid in self.entities and self.entities[eid].type == entity_type
        }

    # ── Upsert with optimized fuzzy match ──────────────────────────────

    def upsert_entity(
        self,
        value: str,
        entity_type: str,
        confidence: float = 0.5,
        source: str = "",
    ) -> str:
        """Insert or merge an entity with fuzzy deduplication.

        Returns
        -------
        str
            The entity_id (existing or new).
        """
        now = time.time()
        norm = self._normalize(value)

        # Fast exact match on normalised value or alias
        if norm and norm in self._norm_index:
            eid = self._norm_index[norm]
            ent = self.entities[eid]
            ent.mention_count += 1
            ent.last_seen = now
            ent.confidence = min(1.0, ent.confidence + 0.05)
            return eid

        # Fuzzy match on same-type entities only, pre-filtered by trigrams.
        # This makes the fuzzy pass O(k) where k is the number of entities
        # sharing at least one trigram with the query — typically 5–20
        # rather than the full N entities.
        best_score = 0.0
        best_id: str | None = None
        for eid in self._candidate_entity_ids(norm, entity_type):
            ent = self.entities[eid]
            ratio = _ratio(norm, self._normalize(ent.value))
            if ratio > best_score:
                best_score = ratio
                best_id = eid

        if best_score >= KG_FUZZY_THRESHOLD and best_id is not None:
            ent = self.entities[best_id]
            if norm not in [self._normalize(a) for a in ent.aliases]:
                ent.aliases.append(value)
                self._index_entity(best_id, value)
            ent.mention_count += 1
            ent.last_seen = now
            ent.confidence = min(1.0, ent.confidence + 0.05)
            return best_id

        # New entity
        eid = f"e_{uuid.uuid4().hex[:8]}"
        entity = KGEntity(
            entity_id=eid,
            type=entity_type,
            value=value,
            first_seen=now,
            last_seen=now,
            confidence=confidence,
        )

        # Safety-relevant facts (allergy, medical_condition, …) must never be
        # stranded in `pending` on a weak GLiNER score — they bypass the
        # threshold and go straight to active.
        if confidence >= KG_ACTIVE_THRESHOLD or entity_type in CRITICAL_ENTITY_TYPES:
            self.entities[eid] = entity
            self._index_entity(eid, value)
        else:
            self.pending[eid] = entity

        return eid

    def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float = 0.5,
        source: str = "",
    ) -> str:
        """Add a relation between entities."""
        now = time.time()
        # V5.9 — dédoublonnage exact : même (sujet, prédicat, objet) → on
        # rafraîchit l'existante au lieu de créer un doublon.
        for r in self.relations.values():
            if (not r.superseded and r.subject_id == subject_id
                    and r.predicate == predicate and r.object_id == object_id):
                r.created_ts = now
                r.confidence = min(1.0, r.confidence + 0.05)
                return r.rel_id
        # V5.9 — supersession des prédicats fonctionnels : un nouveau fait
        # périme l'ancien (ex : changement d'employeur / de ville).
        if predicate in _FUNCTIONAL_PREDICATES:
            for r in self.relations.values():
                if (not r.superseded and r.subject_id == subject_id
                        and r.predicate == predicate):
                    r.superseded = True
        rid = f"r_{uuid.uuid4().hex[:8]}"
        self.relations[rid] = KGRelation(
            rel_id=rid,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            source_doc=source,
            created_ts=now,
        )
        return rid

    # ── Query ──────────────────────────────────────────────────────────

    def query_by_question(
        self,
        question: str,
        extracted_entities: list[dict],
        max_facts: int = 5,
    ) -> list[str]:
        """Retrieve relevant facts for a question.

        Parameters
        ----------
        question : str
            The user question.
        extracted_entities : list[dict]
            Entities extracted from the question via GLiNER.
        max_facts : int
            Maximum facts to return.

        Returns
        -------
        list[str]
            Human-readable fact strings.
        """
        entity_ids: set[str] = set()

        for ext in extracted_entities:
            norm = self._normalize(ext["text"])
            if norm in self._norm_index:
                entity_ids.add(self._norm_index[norm])
            else:
                for eid, ent in self.entities.items():
                    if norm in self._normalize(ent.value):
                        entity_ids.add(eid)
                        break

        facts: list[str] = []
        for rid, rel in self.relations.items():
            if rel.superseded:  # V5.9 — on ignore les faits périmés
                continue
            if rel.subject_id in entity_ids or rel.object_id in entity_ids:
                subj = self.entities.get(rel.subject_id)
                obj = self.entities.get(rel.object_id)
                s_name = subj.value if subj else rel.subject_id
                o_name = obj.value if obj else rel.object_id
                facts.append(f"{s_name} {rel.predicate} {o_name}")

        for eid in entity_ids:
            ent = self.entities.get(eid)
            if ent:
                # NOTE: ``mention_count`` is intentionally NOT exposed in
                # the LLM prompt. It is a system counter used internally
                # for scoring + UI, but >5B models tend to misread
                # "(person, vu 19x)" as narrative content (e.g. "tu y
                # vis depuis 19 minutes" — observed on Qwen2.5-7B).
                # Keep the type tag only; freshness is conveyed via the
                # [Identité] block when relevant.
                facts.append(f"{ent.value} ({ent.type})")

        return facts[:max_facts]

    def get_all_entities(self) -> list[dict]:
        """Return all active entities as dicts."""
        return [asdict(e) for e in self.entities.values()]

    def critical_constraints(self) -> list[str]:
        """Active safety-relevant facts (allergy, medical_condition, …) as
        ``"value (type)"`` strings, for the always-on vital-constraints block.
        Most-recently-seen first."""
        crit = [e for e in self.entities.values()
                if e.type in CRITICAL_ENTITY_TYPES]
        crit.sort(key=lambda e: getattr(e, "last_seen", 0.0), reverse=True)
        return [f"{e.value} ({e.type})" for e in crit]

    # ── Pending cleanup ────────────────────────────────────────────────

    def cleanup_pending(self) -> int:
        """Purge pending entities older than TTL."""
        now = time.time()
        ttl = KG_PENDING_TTL_HOURS * 3600
        expired = [
            eid for eid, ent in self.pending.items()
            if now - ent.last_seen > ttl
        ]
        for eid in expired:
            del self.pending[eid]
        return len(expired)

    def promote_pending(self) -> int:
        """Promote pending entities that now exceed the active threshold."""
        promoted = []
        for eid, ent in list(self.pending.items()):
            if ent.confidence >= KG_ACTIVE_THRESHOLD or ent.type in CRITICAL_ENTITY_TYPES:
                self.entities[eid] = ent
                # Index both value and any aliases the entity may have
                # accumulated while pending.
                self._index_entity(eid, ent.value)
                for alias in ent.aliases:
                    self._index_entity(eid, alias)
                promoted.append(eid)
        for eid in promoted:
            del self.pending[eid]
        return len(promoted)

    # ── Persistence ────────────────────────────────────────────────────

    def save(self) -> None:
        """Save entities and relations atomically."""
        self._save_json(self._entity_path, {
            "active": {eid: asdict(e) for eid, e in self.entities.items()},
            "pending": {eid: asdict(e) for eid, e in self.pending.items()},
        })
        self._save_json(self._relation_path, {
            rid: asdict(r) for rid, r in self.relations.items()
        })
        # V5.2 — Persist detected communities for next session.
        # Communities are recomputed each microsleep ; persisting them
        # lets the system start with the previous cycle's clusters
        # before the first new microsleep finishes.
        if self.communities:
            try:
                payload = [
                    {
                        "community_id": c.community_id,
                        "entity_ids": list(c.entity_ids),
                        "summary": c.summary,
                        "size": c.size,
                        "density": c.density,
                        "detected_at": c.detected_at,
                    }
                    for c in self.communities
                ]
                self._save_json(self._communities_path, {"communities": payload})
            except Exception as exc:
                log.warning("Community save failed: %s", exc)

    def _load(self) -> None:
        """Load state from JSON files."""
        if self._entity_path.exists():
            try:
                data = json.loads(self._entity_path.read_text("utf-8"))
                for eid, d in data.get("active", {}).items():
                    self.entities[eid] = KGEntity(**d)
                for eid, d in data.get("pending", {}).items():
                    self.pending[eid] = KGEntity(**d)
                self._rebuild_index()
            except Exception as exc:
                log.warning("KG entities load failed: %s", exc)

        if self._relation_path.exists():
            try:
                data = json.loads(self._relation_path.read_text("utf-8"))
                for rid, d in data.items():
                    self.relations[rid] = KGRelation(**d)
            except Exception as exc:
                log.warning("KG relations load failed: %s", exc)

        # V5.2 — Load persisted communities from previous session.
        if self._communities_path.exists():
            try:
                from rune.cognition.graph_communities import Community
                data = json.loads(self._communities_path.read_text("utf-8"))
                for payload in data.get("communities", []):
                    self.communities.append(Community(
                        community_id=payload["community_id"],
                        entity_ids=payload["entity_ids"],
                        summary=payload.get("summary", ""),
                        size=payload.get("size", len(payload["entity_ids"])),
                        density=payload.get("density", 0.0),
                        detected_at=payload.get("detected_at", 0.0),
                    ))
                log.info("Loaded %d persisted communities", len(self.communities))
            except Exception as exc:
                log.warning("Community load failed: %s", exc)

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        """Atomic JSON write."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(path))

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relations."""
        if entity_id not in self.entities:
            return False
        self.entities.pop(entity_id)
        # Drop from both norm and trigram indexes
        self._unindex_entity(entity_id)

        to_remove = [
            rid for rid, r in self.relations.items()
            if r.subject_id == entity_id or r.object_id == entity_id
        ]
        for rid in to_remove:
            del self.relations[rid]
        return True
