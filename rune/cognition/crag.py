"""Corrective RAG (CRAG) — V5.2 retrieval evaluator.

Adds an evaluation layer on top of the hybrid retriever : instead of
silently filtering out low-score chunks (V5 behaviour), we classify
the retrieval quality into one of three buckets and act accordingly :

- CORRECT : top rerank_score ≥ correct_threshold (default 0.7).
  The retrieval is trustworthy as-is, use the chunks directly.

- AMBIGUOUS : ambiguous_threshold ≤ top < correct_threshold (default
  0.3-0.7). The retrieval found *something* but not strongly aligned.
  We rewrite the query via LLM (one shot, short prompt) and retry
  retrieval once. If the second pass scores better, use it ; else
  keep the original.

- INCORRECT : top < ambiguous_threshold (default 0.3). The retrieval
  has nothing useful. We surface a cognitive item to the user
  ("mémoire long-terme peu pertinente, je vais répondre depuis ce
  que je sais") and the caller decides if web fallback is needed
  (already handled by V5.1 routing — CRAG doesn't loop into web).

Why this matters
----------------
In V5 logs we routinely saw lines like :
    Cross-encoder rerank: 7 → 0 (threshold 0.20, top=0.04)
…which silently produced an empty RAG context. The LLM then answered
from its parametric memory only, with no signal that the memory was
queried but empty. CRAG fixes this by :
1. Making the evaluation explicit (status flag in metadata).
2. Trying to rescue ambiguous cases via query rewriting.
3. Communicating clearly to the user what happened.

Reference : "Corrective Retrieval Augmented Generation", Yan et al.,
ICLR 2024. We implement a simplified, single-pass version without
the external knowledge expansion (web fallback is handled elsewhere
in our architecture via the V5.1 router).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

log = logging.getLogger("lythea.cognition.crag")


# ── Status enum ────────────────────────────────────────────────────────


class RetrievalStatus(str, Enum):
    """Classification of retrieval quality.

    Strings (not auto enum) so the status survives JSON serialisation
    and log inspection without obscure ``RetrievalStatus.CORRECT: 1``
    repr noise.
    """

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"
    EMPTY = "empty"  # No chunks at all (corpus empty or query unmatched)


@dataclass
class CRAGVerdict:
    """Output of the CRAG evaluation step."""

    status: RetrievalStatus
    top_score: float
    n_chunks: int
    # Original chunks (may be empty if INCORRECT/EMPTY)
    chunks: list[dict] = field(default_factory=list)
    # If we rewrote and retried, the rewritten query (else None)
    rewritten_query: str | None = None
    # Short human reason for logs/UI
    reason: str = ""


# ── LLM interface ──────────────────────────────────────────────────────


class LLMCompleter(Protocol):
    """Same minimal interface as web_classifier / tool_dispatcher."""

    def complete_sync(
        self,
        messages: list[dict],
        max_new_tokens: int = 32,
        timeout: float | None = None,
    ) -> str: ...


# ── Query rewriting ────────────────────────────────────────────────────


_REWRITE_SYSTEM_PROMPT = (
    "Tu reformules une requête pour améliorer la recherche dans une base "
    "vectorielle. La requête originale n'a pas donné de bons résultats. "
    "Ta reformulation doit :\n"
    "- Garder le sens et l'intention\n"
    "- Ajouter des synonymes / termes proches si utile\n"
    "- Retirer les mots vides ou marqueurs conversationnels "
    "(« dis-moi », « peux-tu », etc.)\n"
    "- Rester en 1 phrase concise (10-20 mots max)\n"
    "\n"
    "Réponds UNIQUEMENT par la requête reformulée, rien d'autre. "
    "Pas de préambule, pas d'explication, pas de guillemets."
)


def _rewrite_query(
    original_query: str,
    llm: LLMCompleter,
    timeout: float = 3.0,
) -> str | None:
    """Ask the LLM to rephrase the query for better retrieval.

    Returns the rewritten query (single line), or None on failure.
    Caller should fall back to the original query in that case.
    """
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Requête originale : {original_query.strip()}"},
    ]
    try:
        raw = llm.complete_sync(
            messages, max_new_tokens=48, temperature=0.3, timeout=timeout,
        )
    except Exception as exc:
        log.warning("CRAG rewrite LLM failed: %s", exc)
        return None

    if not raw or not raw.strip():
        return None

    # Sanitise : take first line, strip quotes/preamble.
    line = raw.strip().splitlines()[0].strip()
    # Common LLM preambles to strip
    for prefix in (
        "Reformulation :", "Reformulation:", "Voici :", "Voici:",
        "Réponse :", "Réponse:", "Query:", "Query :",
    ):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix):].strip()
    # Strip surrounding quotes
    line = re.sub(r'^["\'`]+|["\'`]+$', "", line).strip()

    # Sanity : reject if too short, too long, or identical to original
    if len(line) < 3 or len(line) > 200:
        return None
    if line.lower().strip() == original_query.lower().strip():
        return None

    return line


# ── Main CRAG evaluator ────────────────────────────────────────────────


def evaluate_retrieval(
    query: str,
    chunks: list[dict],
    *,
    correct_threshold: float = 0.7,
    ambiguous_threshold: float = 0.3,
    score_key: str = "rerank_score",
) -> CRAGVerdict:
    """Classify retrieval quality without rewriting.

    This is the pure evaluation step — separated from the retry logic
    so it can be unit-tested in isolation and reused without the LLM
    dependency.

    Parameters
    ----------
    query : str
        Original user query (used only for logging context).
    chunks : list[dict]
        Output from HybridRetriever.search(). Expected to contain a
        ``rerank_score`` (preferred) or ``score`` field.
    correct_threshold : float
        Top score above which we consider the retrieval reliable.
        Default 0.7 — cross-encoder scores are sigmoidal-ish and 0.7
        is a strong match in practice.
    ambiguous_threshold : float
        Top score above which we'll try to rescue via rewrite.
        Below this, retrieval is too poor to be useful.
    score_key : str
        Which dict key to read the per-chunk score from. ``rerank_score``
        comes from the cross-encoder, ``score`` is the RRF fallback.
    """
    if not chunks:
        return CRAGVerdict(
            status=RetrievalStatus.EMPTY,
            top_score=0.0,
            n_chunks=0,
            reason="no_chunks_returned",
        )

    # Get the best score available. Fall back from rerank_score to
    # plain score if the rerank step was skipped.
    def _score_of(c: dict) -> float:
        if score_key in c:
            return float(c[score_key])
        return float(c.get("score", 0.0))

    top_score = max((_score_of(c) for c in chunks), default=0.0)
    n_chunks = len(chunks)

    if top_score >= correct_threshold:
        status = RetrievalStatus.CORRECT
        reason = f"top={top_score:.2f}≥{correct_threshold}"
    elif top_score >= ambiguous_threshold:
        status = RetrievalStatus.AMBIGUOUS
        reason = f"top={top_score:.2f} in [{ambiguous_threshold},{correct_threshold})"
    else:
        status = RetrievalStatus.INCORRECT
        reason = f"top={top_score:.2f}<{ambiguous_threshold}"

    return CRAGVerdict(
        status=status,
        top_score=top_score,
        n_chunks=n_chunks,
        chunks=chunks,
        reason=reason,
    )


def evaluate_and_rescue(
    query: str,
    chunks: list[dict],
    retriever,
    llm: LLMCompleter | None = None,
    *,
    correct_threshold: float = 0.7,
    ambiguous_threshold: float = 0.3,
    enable_rewrite: bool = True,
    rewrite_timeout: float = 3.0,
    score_key: str = "rerank_score",
) -> CRAGVerdict:
    """Full CRAG : evaluate, and if AMBIGUOUS try rewriting once.

    On AMBIGUOUS, we ask the LLM to rephrase the query and re-run
    the retriever. We pick whichever result set has the highest top
    score. We never go more than one rewrite (no infinite loops).

    Parameters
    ----------
    query : str
        Original user query.
    chunks : list[dict]
        First-pass retrieval results.
    retriever
        Object with a ``.search(query, n=..., rerank=...)`` method
        returning list[dict]. Typically a HybridRetriever instance.
    llm : LLMCompleter | None
        If None or ``enable_rewrite=False``, skips rewrite (returns
        same as ``evaluate_retrieval`` without rescue).
    """
    verdict = evaluate_retrieval(
        query, chunks,
        correct_threshold=correct_threshold,
        ambiguous_threshold=ambiguous_threshold,
        score_key=score_key,
    )

    # Only rescue if AMBIGUOUS — CORRECT is already good, INCORRECT
    # / EMPTY suggests the corpus simply lacks the answer and a rewrite
    # won't conjure new content out of thin air.
    if verdict.status != RetrievalStatus.AMBIGUOUS:
        return verdict
    if not enable_rewrite or llm is None:
        return verdict

    rewritten = _rewrite_query(query, llm, timeout=rewrite_timeout)
    if not rewritten:
        return verdict

    log.info("CRAG rewrite: %r → %r", query[:60], rewritten[:60])
    try:
        new_chunks = retriever.search(rewritten, rerank=True)
    except Exception as exc:
        log.warning("CRAG retry retrieval failed: %s", exc)
        return verdict

    new_verdict = evaluate_retrieval(
        rewritten, new_chunks,
        correct_threshold=correct_threshold,
        ambiguous_threshold=ambiguous_threshold,
        score_key=score_key,
    )
    new_verdict.rewritten_query = rewritten

    # Keep whichever has the higher top_score. Avoids the rewrite
    # making things worse (e.g. lost nuance, over-broadening).
    if new_verdict.top_score > verdict.top_score:
        log.info(
            "CRAG rescue successful: %.2f → %.2f (status %s → %s)",
            verdict.top_score, new_verdict.top_score,
            verdict.status.value, new_verdict.status.value,
        )
        return new_verdict
    else:
        log.info(
            "CRAG rescue declined: rewrite gave %.2f, original %.2f kept",
            new_verdict.top_score, verdict.top_score,
        )
        # Preserve the rewritten_query field on the original verdict
        # so observability stays intact.
        verdict.rewritten_query = rewritten
        return verdict


# ── UI message helpers ────────────────────────────────────────────────


def cognitive_item_for(verdict: CRAGVerdict) -> str | None:
    """Return a user-facing cognitive item describing the CRAG outcome.

    Used by hippocampe to surface what happened in the cognitive
    activity block. Returns None for the trivial CORRECT case (no
    point cluttering UI when retrieval just worked).
    """
    if verdict.status == RetrievalStatus.CORRECT:
        return None  # silently use the chunks
    if verdict.status == RetrievalStatus.AMBIGUOUS:
        if verdict.rewritten_query and verdict.top_score > 0.7:
            return (
                "🔄 *Première recherche imprécise, j'ai reformulé "
                "et trouvé mieux.*"
            )
        return (
            "🟡 *Souvenirs partiellement pertinents — j'utilise ce "
            "que j'ai mais avec prudence.*"
        )
    if verdict.status == RetrievalStatus.INCORRECT:
        return (
            "⚠️ *Rien de vraiment pertinent en mémoire long-terme — "
            "je vais répondre depuis ce que je sais.*"
        )
    if verdict.status == RetrievalStatus.EMPTY:
        return None  # gather() already announces empty memory elsewhere
    return None
