"""Hybrid retrieval over Chroma — BM25 + dense + RRF fusion.

Optional cross-encoder reranking with lazy loading. Can fallback to
cosine rerank via GLiNER embedder to avoid loading an extra model.
"""
from __future__ import annotations

import logging
from typing import Any

from rune.config import RETRIEVAL_RRF_K, RETRIEVAL_RERANK_TOP, RETRIEVAL_TOP_N

log = logging.getLogger("rune.memory.retrieval")


class HybridRetriever:
    """Retrieves from ChromaDB using dense + sparse fusion.

    Parameters
    ----------
    collection : Any
        ChromaDB collection handle.
    embedder : callable, optional
        Embedding function for cosine rerank fallback.
    use_cross_encoder : bool
        Whether to try loading a cross-encoder for reranking.
    """

    def __init__(
        self,
        collection: Any,
        embedder: Any | None = None,
        use_cross_encoder: bool = False,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.use_cross_encoder = use_cross_encoder
        self._bm25 = None
        self._bm25_doc_count = 0
        self._cross_encoder = None
        self._cross_encoder_loaded = False

    # ── BM25 index ─────────────────────────────────────────────────────

    def _maybe_rebuild_bm25(self) -> None:
        """Lazily rebuild BM25 index when document count changes."""
        try:
            count = self.collection.count()
        except Exception:
            return

        if count == self._bm25_doc_count and self._bm25 is not None:
            return

        try:
            from rank_bm25 import BM25Okapi
            results = self.collection.get(include=["documents"])
            docs = results.get("documents") or []
            if not docs:
                return
            tokenized = [d.lower().split() for d in docs]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_ids = results.get("ids", [])
            self._bm25_docs = docs
            self._bm25_doc_count = count
        except ImportError:
            log.debug("rank_bm25 not available, using dense-only retrieval")
        except Exception as exc:
            log.warning("BM25 rebuild failed: %s", exc)

    # ── Cross-encoder rerank (lazy load) ───────────────────────────────

    def _get_cross_encoder(self) -> Any | None:
        """Load cross-encoder on first use.

        Tries the user-configured model first (LYTHEA_CROSS_ENCODER_MODEL),
        then falls back through a chain of well-known multilingual
        rerankers. Loaded once and cached for the process lifetime.
        """
        if self._cross_encoder_loaded:
            return self._cross_encoder

        self._cross_encoder_loaded = True
        if not self.use_cross_encoder:
            return None

        # User preference first, then well-known fallbacks. Deduplicate
        # in case the user set the same as a default.
        from rune.settings import get_settings
        preferred = get_settings().cross_encoder_model
        chain = [preferred, "BAAI/bge-reranker-v2-m3",
                 "BAAI/bge-reranker-base",
                 "cross-encoder/ms-marco-MiniLM-L-6-v2"]
        seen: set[str] = set()
        models: list[str] = []
        for m in chain:
            if m and m not in seen:
                seen.add(m)
                models.append(m)

        for model_name in models:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(model_name)
                log.info("Cross-encoder loaded: %s", model_name)
                return self._cross_encoder
            except Exception as exc:
                log.debug("Cross-encoder %s failed: %s", model_name, exc)

        log.info("No cross-encoder available, using cosine rerank fallback")
        return None

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = RETRIEVAL_RERANK_TOP,
        rerank: bool = True,
    ) -> list[dict]:
        """Hybrid search with RRF fusion.

        Parameters
        ----------
        query : str
            Search query.
        n : int
            Number of final results.
        rerank : bool
            Whether to apply reranking.

        Returns
        -------
        list[dict]
            Sorted results with ``{id, document, metadata, score}``.
        """
        if self.collection.count() == 0:
            return []

        # Dense retrieval via Chroma
        try:
            dense_results = self.collection.query(
                query_texts=[query],
                n_results=min(RETRIEVAL_TOP_N, self.collection.count()),
            )
        except Exception as exc:
            log.warning("Chroma query failed: %s", exc)
            return []

        dense_ids = dense_results.get("ids", [[]])[0]
        dense_docs = dense_results.get("documents", [[]])[0]
        dense_metas = dense_results.get("metadatas", [[]])[0]

        # BM25 sparse retrieval
        self._maybe_rebuild_bm25()
        sparse_ranking: dict[str, int] = {}
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query.lower().split())
            sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, idx in enumerate(sorted_idx[:RETRIEVAL_TOP_N]):
                if idx < len(self._bm25_ids):
                    sparse_ranking[self._bm25_ids[idx]] = rank

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, (did, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
            rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (RETRIEVAL_RRF_K + rank)
            doc_map[did] = {"id": did, "document": doc, "metadata": meta or {}}

        for did, rank in sparse_ranking.items():
            rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (RETRIEVAL_RRF_K + rank)
            if did not in doc_map and self._bm25_docs:
                idx = self._bm25_ids.index(did) if did in self._bm25_ids else -1
                if idx >= 0:
                    doc_map[did] = {
                        "id": did,
                        "document": self._bm25_docs[idx],
                        "metadata": {},
                    }

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        candidates = []
        for did in sorted_ids[:RETRIEVAL_TOP_N]:
            if did in doc_map:
                entry = doc_map[did]
                entry["score"] = rrf_scores[did]
                candidates.append(entry)

        # Surface the fusion stats so operators can see what the
        # hybrid retriever is doing per query (helpful for debugging
        # "memory miss" complaints).
        log.info(
            "Hybrid search: dense=%d, bm25=%d, fused=%d (query=%r)",
            len(dense_ids), len(sparse_ranking), len(candidates),
            query[:60] + ("…" if len(query) > 60 else ""),
        )

        if not candidates:
            return []

        # Reranking
        if rerank and candidates:
            candidates = self._rerank(query, candidates, n)
        else:
            candidates = candidates[:n]

        return candidates

    def _rerank(self, query: str, candidates: list[dict], n: int) -> list[dict]:
        """Rerank candidates using cross-encoder or cosine fallback."""
        ce = self._get_cross_encoder()

        if ce is not None:
            pairs = [(query, c["document"]) for c in candidates]
            try:
                from rune.settings import get_settings
                min_score = get_settings().cross_encoder_min_score
                scores = ce.predict(pairs)
                for i, s in enumerate(scores):
                    candidates[i]["rerank_score"] = float(s)
                candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
                kept = [
                    c for c in candidates[:n]
                    if c.get("rerank_score", 0) >= min_score
                ]
                # Visibility: how many survived the threshold?
                # Showing top score helps tune CROSS_ENCODER_MIN_SCORE.
                top_score = max((c.get("rerank_score", 0) for c in candidates), default=0.0)
                log.info(
                    "Cross-encoder rerank: %d → %d (threshold %.2f, top=%.2f)",
                    len(candidates), len(kept), min_score, top_score,
                )
                return kept
            except Exception as exc:
                log.warning("Cross-encoder rerank failed: %s", exc)

        # Cosine fallback using embedder
        if self.embedder is not None:
            try:
                import torch
                q_emb = self.embedder(query)
                if q_emb is not None:
                    q_emb = q_emb.view(1, -1)
                    for c in candidates:
                        d_emb = self.embedder(c["document"])
                        if d_emb is not None:
                            cos = torch.nn.functional.cosine_similarity(
                                q_emb, d_emb.view(1, -1),
                            ).item()
                            c["rerank_score"] = cos
                    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            except Exception as exc:
                log.debug("Cosine rerank failed: %s", exc)

        return candidates[:n]
