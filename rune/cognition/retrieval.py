"""Retrieval phase — assemble RAG context from KG + MHN + Chroma.

Biological inspiration
----------------------
This is the *recall* counterpart of Storage. Where Storage writes
sparse codes during encoding, Retrieval reactivates them when a
new query arrives. In rodents, CA3 pattern completion lets a
partial cue (e.g. a familiar smell) drive the reinstatement of
a full memory trace. We approximate this with three parallel
recall paths, each tuned for a different *kind* of memory:

1. **Identity (KG)** — a narrative summary of the persistent,
   relational view of the user. Always injected, even when the
   query is unrelated, because the model needs to know *who* it
   is talking to. Cognitive analogue: declarative semantic
   memory about persons.

2. **Episodic (MHN)** — Hopfield-attention recall of recent
   exchanges keyed by GLiNER embedding. The same key used at
   storage time. Cognitive analogue: hippocampal episodic recall.

3. **Semantic (Chroma hybrid)** — BM25 + dense + RRF + cross-
   encoder rerank, the heavy lifter for old / consolidated
   memories. Reranking is *enabled* here (unlike the surprise
   discount path which uses ``rerank=False``) because we need
   precision when constructing the prompt. Cognitive analogue:
   neocortical semantic memory after consolidation.

The three sections are joined by a fixed separator and become
the ``[Mémoire contextuelle]`` block of the system prompt.

Each retrieved item is annotated with a freshness tag
("il y a 3 jours", "à l'instant", ...) so the model can weight
the recall by recency. The freshness suffix on the identity line
is gated to ≥5 min to keep the prompt clean across rapid turns.

Design notes
------------
- ``kg_identity_summary`` is exposed as a public method, not just
  used internally, because the two-pass reasoning prompt
  (:meth:`Hippocampe._generate_reasoning`) needs the same identity
  context but skips the rest of the RAG.
- The phase has no internal state — every call is self-contained
  and reads the live memory backends through their references.
- Failures on any single backend log and fall through; we never
  let a Chroma timeout block the KG identity injection.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rune.temporal import annotate_with_freshness, humanise_delta

log = logging.getLogger("lythea.cognition.retrieval")


# Section separator inside the [Mémoire contextuelle] prompt block.
# UTF-8 box-drawing chars survive tokenisation cleanly across modern
# tokenizers (GPT-2, Llama, Qwen) and visually mark section breaks
# better than plain dashes.
SECTION_SEPARATOR: str = "\n═══\n"

# Recall budgets — keep these tight, the prompt budget in Phase C
# trims aggressively if exceeded. Empirically 3 hits per source is
# the sweet spot between coverage and noise.
MHN_TOP_K: int = 3
MHN_MIN_ATTENTION: float = 0.15
# CHROMA_TOP_N : combien de chunks le RAG remonte après cross-encoder
# rerank. Historiquement 3 — trop conservateur pour les questions
# ouvertes (« résume-moi ce que tu sais sur X ») où on a besoin de
# voir plusieurs facettes du sujet. Augmenté à 10 — aligné sur ce que
# font Claude/GPT en interne (top-k 10-20 selon les sources publiques).
# Le cross-encoder threshold (cross_encoder_min_score, défaut 0.20)
# filtre toujours les chunks faiblement pertinents, donc on n'ajoute
# pas de bruit pour autant.
CHROMA_TOP_N: int = 10

# Per-recall content cap. Long episodes / documents get truncated
# before injection. 200 chars ≈ 50 tokens — enough to identify the
# memory, short enough to not blow the context.
RECALL_TEXT_MAX_CHARS: int = 200

# Below this gap we omit the freshness suffix on the identity line.
# "à l'instant" on every turn pollutes the prompt without adding
# information. 5 minutes is the threshold from the original code.
IDENTITY_FRESHNESS_MIN_GAP_SEC: float = 300.0

# Number of "uncovered" entities (not subject/object of any relation)
# we list under "Autres informations connues" as a fallback.
UNCOVERED_ENTITIES_LIMIT: int = 5

# Predicate translation: KG stores French slugs (vit_à, travaille_chez,
# ...) but the prompt needs natural French. Symmetric to the
# storage-side _RELATION_PREDICATES table.
_PREDICATE_FR: dict[str, str] = {
    "vit_à": "vit à",
    "travaille_chez": "travaille chez",
    "travaille_sur": "travaille sur",
    "utilise": "utilise",
    "est": "est",
    "maîtrise": "maîtrise",
}


@dataclass
class RetrievalContext:
    """Output of the Retrieval phase.

    Attributes
    ----------
    sections
        Already-formatted section strings (identity, facts,
        episodic, semantic). Empty list when nothing was found.
    thoughts
        Cognitive trace fragments to surface in the UI ("Hmm, ça
        me rappelle quelque chose…"). Pure UX — they are not
        injected into the prompt.
    """

    sections: list[str] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Join sections with the canonical separator.

        Returns ``""`` when there are no sections, which Phase C
        relies on to skip the ``[Mémoire contextuelle]`` block
        entirely.
        """
        if not self.sections:
            return ""
        return SECTION_SEPARATOR.join(self.sections)


class RetrievalPhase:
    """Compose KG identity + KG facts + MHN episodic + Chroma semantic.

    Parameters
    ----------
    kg
        :class:`KnowledgeGraphStore`. Read-only access for identity
        narrative and ``query_by_question`` facts.
    mhn
        :class:`ModernHopfieldNetwork`. Used for episodic recall.
    entity_extractor
        :class:`EntityExtractor`. Used to encode the query for MHN
        and to extract entities for KG fact lookup. ``None`` =
        no MHN recall, no KG facts (identity still works).
    hybrid_retriever
        :class:`HybridRetriever` over Chroma. ``None`` = semantic
        section is skipped.
    """

    def __init__(
        self,
        kg: Any,
        mhn: Any,
        entity_extractor: Any | None,
        hybrid_retriever: Any | None,
        llm_for_crag: Any | None = None,
    ) -> None:
        self.kg = kg
        self.mhn = mhn
        self.entity_extractor = entity_extractor
        self.hybrid_retriever = hybrid_retriever
        # V5.2 — LLM reference for CRAG query rewriting. Stored as
        # private attribute so _gather_semantic can pick it up
        # without changing every internal call signature.
        self._crag_llm = llm_for_crag

    # ── Public API ─────────────────────────────────────────────────────

    def gather(self, query: str) -> RetrievalContext:
        """Run the four recall paths and assemble the context.

        Each path is tried independently. A failure on one (e.g.
        Chroma timeout) does not poison the others.
        """
        ctx = RetrievalContext()

        # 1. KG identity — always attempted, always sectioned at top.
        identity = self.kg_identity_summary()
        if identity:
            ctx.sections.append(identity)
            log.info(
                "KG identity injected: %d chars, %d entities, %d relations",
                len(identity), len(self.kg.entities), len(self.kg.relations),
            )
        else:
            log.info("KG identity empty — no entities known")

        # 2. KG query-specific facts.
        self._gather_kg_facts(query, ctx)

        # 2bis. V5.2 — GraphRAG communities (thematic context).
        # Cheap : reads pre-computed clusters from kg.communities.
        self._gather_kg_communities(query, ctx)

        # 3. MHN episodic.
        self._gather_episodic(query, ctx)

        # 4. Chroma semantic (with rerank).
        self._gather_semantic(query, ctx)

        return ctx

    def kg_identity_summary(self) -> str:
        """Build the narrative identity string from the KG.

        Public because :meth:`Hippocampe._generate_reasoning` needs
        identity context for the two-pass reasoning prompt without
        running the rest of the RAG.

        Returns ``""`` when no entities are known yet — Phase C
        and the reasoning pass both rely on the empty-string
        sentinel to skip injection.
        """
        if not self.kg or not self.kg.entities:
            return ""

        lines = self._build_person_lines()
        self._append_uncovered_entities(lines)

        if not lines:
            return ""

        return (
            "[Identité de ton interlocuteur — pour mémoire]\n"
            "Ces faits sont vérifiés mais n'ont pas à être récités à chaque "
            "message. Utilise-les seulement quand la question les concerne "
            "directement, ou pour la première interaction de la session.\n"
            + "\n".join(lines)
        )

    # ── Identity narrative ─────────────────────────────────────────────

    def _build_person_lines(self) -> list[str]:
        """One line per person entity, with their relations rendered FR."""
        lines: list[str] = []
        persons = [e for e in self.kg.entities.values() if e.type == "person"]
        for person in persons:
            facts = self._person_facts(person)
            freshness = self._freshness_suffix(person)
            if facts:
                lines.append(
                    f"Ton interlocuteur s'appelle {person.value}{freshness}. "
                    f"Il/Elle {', '.join(facts)}."
                )
            else:
                lines.append(
                    f"Ton interlocuteur s'appelle {person.value}{freshness}."
                )
        return lines

    def _person_facts(self, person: Any) -> list[str]:
        """Render every relation where ``person`` is the subject in French."""
        out: list[str] = []
        for rel in self.kg.relations.values():
            if rel.subject_id != person.entity_id:
                continue
            obj = self.kg.entities.get(rel.object_id)
            if obj is None:
                continue
            pred_fr = _PREDICATE_FR.get(rel.predicate, rel.predicate)
            out.append(f"{pred_fr} {obj.value}")
        return out

    def _freshness_suffix(self, person: Any) -> str:
        """Return ``" (dernière mention : il y a X)"`` or ``""``.

        The ≥5 min gate avoids the prompt being decorated with
        "à l'instant" on every turn during an active session.
        """
        if not person.last_seen:
            return ""
        gap = time.time() - person.last_seen
        if gap <= IDENTITY_FRESHNESS_MIN_GAP_SEC:
            return ""
        return f" (dernière mention : {humanise_delta(gap)})"

    def _append_uncovered_entities(self, lines: list[str]) -> None:
        """Tail-line listing entities that have no relation yet.

        Excludes ``person``-type entities (they get their own line).
        Capped at :data:`UNCOVERED_ENTITIES_LIMIT` to keep the
        prompt bounded.
        """
        covered_ids: set[str] = set()
        for rel in self.kg.relations.values():
            covered_ids.add(rel.subject_id)
            covered_ids.add(rel.object_id)

        uncovered = [
            e for e in self.kg.entities.values()
            if e.entity_id not in covered_ids and e.type != "person"
        ]
        if not uncovered:
            return
        extras = [
            f"{e.value} ({e.type})"
            for e in uncovered[:UNCOVERED_ENTITIES_LIMIT]
        ]
        lines.append(f"Autres informations connues : {', '.join(extras)}")

    # ── KG facts ───────────────────────────────────────────────────────

    def _gather_kg_facts(self, query: str, ctx: RetrievalContext) -> None:
        """Append a ``[Faits connus]`` section if KG matches the query."""
        if not (self.entity_extractor and self.kg):
            return
        try:
            extracted = self.entity_extractor.extract(query)
            facts = self.kg.query_by_question(query, extracted)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("KG facts lookup failed: %s", exc)
            return
        if facts:
            ctx.sections.append(
                "[Faits connus — pour information]\n"
                + "\n".join(f"• {f}" for f in facts)
            )

    # ── KG communities (V5.2 GraphRAG) ────────────────────────────────

    def _gather_kg_communities(self, query: str, ctx: RetrievalContext) -> None:
        """Append a thematic-context section from GraphRAG communities.

        Strategy : if the query mentions known entities, surface the
        communities those entities belong to (their thematic context).
        If no specific entities match, fall back to the top-N largest
        communities to give the LLM a general thematic map of the user's
        knowledge graph.

        Cheap at retrieval time : communities are pre-computed in the
        consolidation phase, this method just reads ``self.kg.communities``
        and selects the relevant ones.
        """
        if not self.kg or not getattr(self.kg, "communities", None):
            return
        if not self.entity_extractor:
            # Without entity extraction we can't focus on relevant
            # communities, so we silently skip rather than dumping
            # the whole community map.
            return

        try:
            from rune.cognition.graph_communities import render_community_context
        except ImportError:
            return

        # Find which KG entities are mentioned in the query.
        try:
            extracted = self.entity_extractor.extract(query)
        except Exception as exc:
            log.warning("Community focus extract failed: %s", exc)
            extracted = []

        # Map extracted strings → KG entity_ids (fuzzy via normalised value)
        focus_ids: set[str] = set()
        for item in (extracted or []):
            val = (item.get("text") or item.get("value") or "").strip()
            if not val:
                continue
            norm = self.kg._normalize(val) if hasattr(self.kg, "_normalize") else val.lower()
            eid = self.kg._norm_index.get(norm) if hasattr(self.kg, "_norm_index") else None
            if eid:
                focus_ids.add(eid)

        block = render_community_context(
            self.kg.communities,
            self.kg.entities,
            focus_entity_ids=focus_ids if focus_ids else None,
            max_communities=3,
        )
        if block:
            ctx.sections.append(block)
            log.info(
                "KG communities injected: %d total, focus=%d entities, "
                "block_size=%d chars",
                len(self.kg.communities), len(focus_ids), len(block),
            )

    # ── MHN episodic ───────────────────────────────────────────────────

    def _gather_episodic(self, query: str, ctx: RetrievalContext) -> None:
        """Append a ``[Mémoire épisodique]`` section from the MHN."""
        if self.entity_extractor is None:
            return
        try:
            gliner_emb = self.entity_extractor.encode(query)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("GLiNER encode for MHN recall failed: %s", exc)
            return
        if gliner_emb is None:
            return
        try:
            results = self.mhn.retrieve(
                gliner_emb,
                top_k=MHN_TOP_K,
                min_attention=MHN_MIN_ATTENTION,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("MHN retrieve failed: %s", exc)
            return
        if not results:
            return

        ctx.thoughts.append(
            "💭 *Hmm, ça me rappelle quelque chose dans ma mémoire épisodique…*"
        )
        pairs = [
            (r["text"][:RECALL_TEXT_MAX_CHARS], r.get("timestamp", 0.0))
            for r in results
        ]
        annotated = annotate_with_freshness(pairs)
        # Light usage-hint header — same descriptive style as the
        # post-fix-#10 identity block. Avoids the pre-fix injunctive
        # pattern that triggers verbatim recitation on >5B models.
        ctx.sections.append(
            "[Mémoire épisodique — pour information]\n"
            + "\n".join(f"• {e}" for e in annotated)
        )

    # ── Chroma semantic ────────────────────────────────────────────────

    def _gather_semantic(self, query: str, ctx: RetrievalContext) -> None:
        """Append a ``[Mémoire sémantique]`` section from Chroma.

        Reranking is *enabled* here — unlike the surprise discount
        path which intentionally skips it for speed. We are
        building the actual prompt, accuracy beats latency.

        V5.2 — Corrective RAG (CRAG) integration. After the initial
        hybrid search + rerank, we evaluate retrieval quality via
        :func:`lythea.cognition.crag.evaluate_and_rescue`. The result
        is a CRAGVerdict with status (CORRECT/AMBIGUOUS/INCORRECT/
        EMPTY) that drives both the chunk selection (rescued if
        AMBIGUOUS and rewrite improved scores) and the cognitive
        items shown to the user (transparency on retrieval quality).
        """
        if not self.hybrid_retriever:
            return
        try:
            results = self.hybrid_retriever.search(query, n=CHROMA_TOP_N, rerank=True)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Chroma hybrid search failed: %s", exc)
            return

        # V5.2 — CRAG evaluation. LLM passed via the retriever's stored
        # reference if available ; falling back to evaluate-only if not.
        llm_for_rewrite = getattr(self, "_crag_llm", None)
        crag_msg_shown = False
        try:
            from rune.cognition.crag import (
                evaluate_and_rescue, cognitive_item_for,
            )
            from rune.settings import get_settings
            settings = get_settings()
            verdict = evaluate_and_rescue(
                query, results, self.hybrid_retriever,
                llm=llm_for_rewrite,
                correct_threshold=settings.crag_correct_threshold,
                ambiguous_threshold=settings.crag_ambiguous_threshold,
                enable_rewrite=settings.crag_enable_rewrite and llm_for_rewrite is not None,
            )
            # Replace results with the (possibly rescued) chunks.
            results = verdict.chunks

            # Surface the verdict to the user when meaningful.
            cog_msg = cognitive_item_for(verdict)
            if cog_msg:
                ctx.thoughts.append(cog_msg)
                crag_msg_shown = True

            log.info(
                "CRAG verdict: status=%s top=%.2f n=%d%s",
                verdict.status.value, verdict.top_score, verdict.n_chunks,
                " (rewritten=True)" if verdict.rewritten_query else "",
            )
        except Exception as exc:
            # CRAG must never break the retrieval pipeline. If it
            # crashes, fall back to the legacy behaviour (use what
            # the retriever returned without evaluation).
            log.warning("CRAG evaluation failed, using raw results: %s", exc)

        if not results:
            return

        # V5.6 — Ne pas annoncer "j'ai trouvé des souvenirs pertinents" si
        # CRAG a déjà émis un verdict : INCORRECT produirait deux pensées
        # contradictoires ("rien de pertinent" + "trouvé"), AMBIGUOUS une
        # redondance. Le message CRAG décrit déjà l'issue de la récupération
        # long-terme de façon plus précise. CORRECT (cog_msg=None) laisse
        # passer la pensée générique ci-dessous.
        if not crag_msg_shown:
            ctx.thoughts.append(
                "💭 *J'ai trouvé des souvenirs pertinents dans ma mémoire long-terme…*"
            )
        pairs = [
            (
                r["document"][:RECALL_TEXT_MAX_CHARS],
                (r.get("metadata") or {}).get("ts", 0.0),
            )
            for r in results
        ]
        annotated = annotate_with_freshness(pairs)
        # Light usage-hint header — see _gather_mhn for rationale.
        ctx.sections.append(
            "[Mémoire sémantique — pour information]\n"
            + "\n".join(f"• {e}" for e in annotated)
        )
