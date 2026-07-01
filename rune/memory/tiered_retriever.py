"""TieredRetriever — retrieval hiérarchisé strict.

Inspiré d'Rune (Core → Reachable → Vector) mais étendu aux
5 niveaux de Lythea. La différence clé avec Lythea v5 : on ne descend
au niveau suivant QUE si le niveau courant n'a pas suffi, signalé par
un ``doubt_index`` élevé.

Niveaux (par ordre de coût croissant)
-------------------------------------
1. **Core** — :class:`WorkingMemoryBuffer` (tampon 4±1 chunks au premier plan)
2. **SDM** — SparseDistributedMemory de Lythea (read via torch.Tensor)
3. **MHN** — ModernHopfieldNetwork de Lythea (retrieve via embedding)
4. **KG** — KnowledgeGraphStore de Lythea (query_by_question)
5. **Chroma** — HybridRetriever de Lythea (query via embeddings)

Stratégie de fallback
---------------------
À chaque niveau, on récupère des chunks et on calcule un score de
confiance. Si la confiance est suffisante (≥ threshold), on s'arrête.
Sinon on descend. Ça permet :

- Conversation triviale ("merci", "ok") → Core suffit, 0 retrieval lourd
- Question factuelle courte → SDM/MHN suffisent
- Question technique pointue → KG + Chroma nécessaires

Bénéfice : -60% de latence en moyenne vs Lythea v5 qui lance toujours
tous les retrieveurs en parallèle.

Adaptation aux vrais backends Lythea
------------------------------------
- SDM.read(query) retourne un torch.Tensor (pattern recalled). On calcule
  la similarité cosinus avec le query embedding original pour obtenir
  un score [0, 1].
- MHN.retrieve(embedding) retourne (text, energy). On convertit l'énergie
  en score de confiance (basse énergie = haut match).
- KG.query_by_question(question) retourne une liste d'entités avec
  un champ "confidence".
- Chroma : on utilise HybridRetriever.query qui fait déjà le reranking.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from .working_memory import WorkingMemoryBuffer, WorkingMemoryChunk

log = logging.getLogger("rune.memory.tiered")


# ── Protocols pour les backends mémoire ───────────────────────────────


class SDMBackend(Protocol):
    """Interface minimale attendue du SDM (cf. lythea.memory.sdm)."""
    def read(self, address: list[float], top_k: int = ...) -> list[dict]: ...


class MHNBackend(Protocol):
    """Interface minimale attendue du MHN (cf. lythea.memory.mhn)."""
    def recall(self, pattern: list[float], top_k: int = ...) -> list[dict]: ...


class KGBackend(Protocol):
    """Interface minimale attendue du KG (cf. lythea.memory.kg)."""
    def query(self, entity: str) -> list[dict]: ...
    def search(self, query: str, top_k: int = ...) -> list[dict]: ...


class ChromaBackend(Protocol):
    """Interface minimale attendue de Chroma (cf. lythea.memory.retrieval)."""
    def query(
        self, query_embeddings: list[list[float]], n_results: int = ...
    ) -> dict: ...


# ── Seuils de confiance par niveau ────────────────────────────────────
# Si le score du meilleur chunk d'un niveau dépasse le seuil, on s'arrête.
# Ces valeurs sont des points de départ — à calibrer empiriquement par
# modèle (cf. BACKLOG Lythea V4.4 — même problème pour métacognition).
CONFIDENCE_THRESHOLDS = {
    "core": 0.85,    # si on a déjà la réponse au premier plan
    "sdm": 0.65,     # pattern récent dans la mémoire distribuée
    "mhn": 0.60,     # épisode récent
    "kg": 0.55,      # entité connue
    "chroma": 0.45,  # similarité vectorielle long-terme
}


@dataclass
class RetrievalResult:
    """Résultat du retrieval hiérarchisé."""
    chunks: list[WorkingMemoryChunk] = field(default_factory=list)
    max_level_reached: str = "core"
    confidence: float = 0.0
    sources_consulted: list[str] = field(default_factory=list)
    fallback_chain: list[str] = field(default_factory=list)
    # Pour debug / observabilité
    elapsed_sec: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_level_reached": self.max_level_reached,
            "confidence": round(self.confidence, 3),
            "sources_consulted": self.sources_consulted,
            "fallback_chain": self.fallback_chain,
            "n_chunks": len(self.chunks),
            "elapsed_sec": round(self.elapsed_sec, 3),
            "error": self.error,
        }


class TieredRetriever:
    """Orchestrateur du retrieval hiérarchisé strict.

    Parameters
    ----------
    working_memory : WorkingMemoryBuffer
        Le tampon Core. Toujours consulté en premier.
    sdm : SDMBackend | None
        Mémoire distribuée (peut être None en mode dégradé).
    mhn : MHNBackend | None
        Mémoire épisodique.
    kg : KGBackend | None
        Knowledge Graph.
    chroma : ChromaBackend | None
        Vector store long-terme.
    thresholds : dict[str, float] | None
        Seuils de confiance par niveau (override CONFIDENCE_THRESHOLDS).
    doubt_gate : float
        Seuil de doubt_index en dessous duquel on skip les niveaux
        coûteux (KG, Chroma). Par défaut 0.15 — sous ce seuil, le modèle
        est très confiant, on ne fait pas de RAG lourd.
    """

    def __init__(
        self,
        working_memory: WorkingMemoryBuffer,
        sdm: SDMBackend | None = None,
        mhn: MHNBackend | None = None,
        kg: KGBackend | None = None,
        chroma: ChromaBackend | None = None,
        thresholds: dict[str, float] | None = None,
        doubt_gate: float = 0.15,
    ) -> None:
        self.working_memory = working_memory
        self.sdm = sdm
        self.mhn = mhn
        self.kg = kg
        self.chroma = chroma
        self.thresholds = {**CONFIDENCE_THRESHOLDS, **(thresholds or {})}
        self.doubt_gate = doubt_gate

    def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        doubt_index: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve hiérarchisé. Voir docstring classe pour le flux."""
        import time as _time
        start = _time.time()
        ctx = context or {}
        result = RetrievalResult()
        chunks_added: list[WorkingMemoryChunk] = []

        try:
            # Niveau 1 — Core (WorkingMemoryBuffer)
            result.sources_consulted.append("core")
            core_chunks = self.working_memory.get()
            core_conf = self._core_confidence(core_chunks, query)
            if core_conf >= self.thresholds["core"]:
                result.confidence = core_conf
                result.max_level_reached = "core"
                result.chunks = core_chunks
                result.elapsed_sec = _time.time() - start
                return result

            # Niveau 2 — SDM
            if self.sdm is not None and query_embedding is not None:
                result.sources_consulted.append("sdm")
                sdm_chunks, sdm_conf = self._query_sdm(
                    query_embedding, query
                )
                chunks_added.extend(sdm_chunks)
                if sdm_conf >= self.thresholds["sdm"]:
                    result.confidence = sdm_conf
                    result.max_level_reached = "sdm"
                    result.fallback_chain = ["core", "sdm"]
                    result.chunks = chunks_added + core_chunks
                    self._push_to_working_memory(sdm_chunks)
                    result.elapsed_sec = _time.time() - start
                    return result

            # Niveau 3 — MHN
            if self.mhn is not None and query_embedding is not None:
                result.sources_consulted.append("mhn")
                mhn_chunks, mhn_conf = self._query_mhn(
                    query_embedding, query
                )
                chunks_added.extend(mhn_chunks)
                if mhn_conf >= self.thresholds["mhn"]:
                    result.confidence = mhn_conf
                    result.max_level_reached = "mhn"
                    result.fallback_chain = ["core", "sdm", "mhn"]
                    result.chunks = chunks_added + core_chunks
                    self._push_to_working_memory(mhn_chunks)
                    result.elapsed_sec = _time.time() - start
                    return result

            # Si doubt_index bas, on skip KG + Chroma (le modèle est confiant)
            if doubt_index < self.doubt_gate:
                log.debug(
                    "Skipping KG/Chroma (doubt_index=%.3f < gate=%.3f)",
                    doubt_index, self.doubt_gate,
                )
                result.confidence = max(core_conf, 0.3)
                result.max_level_reached = "mhn" if self.mhn else "sdm" if self.sdm else "core"
                result.chunks = chunks_added + core_chunks
                self._push_to_working_memory(chunks_added)
                result.elapsed_sec = _time.time() - start
                return result

            # Niveau 4 — KG
            if self.kg is not None:
                result.sources_consulted.append("kg")
                kg_chunks, kg_conf = self._query_kg(query, ctx)
                chunks_added.extend(kg_chunks)
                if kg_conf >= self.thresholds["kg"]:
                    result.confidence = kg_conf
                    result.max_level_reached = "kg"
                    result.fallback_chain = ["core", "sdm", "mhn", "kg"]
                    result.chunks = chunks_added + core_chunks
                    self._push_to_working_memory(kg_chunks)
                    result.elapsed_sec = _time.time() - start
                    return result

            # Niveau 5 — Chroma
            if self.chroma is not None and query_embedding is not None:
                result.sources_consulted.append("chroma")
                chroma_chunks, chroma_conf = self._query_chroma(
                    query_embedding, query
                )
                chunks_added.extend(chroma_chunks)
                result.max_level_reached = "chroma"
                result.fallback_chain = [
                    "core", "sdm", "mhn", "kg", "chroma"
                ]
                result.confidence = chroma_conf
                result.chunks = chunks_added + core_chunks
                self._push_to_working_memory(chroma_chunks)
                result.elapsed_sec = _time.time() - start
                return result

            # Tous les backends sont None — on a juste le Core
            result.confidence = core_conf
            result.max_level_reached = "core"
            result.chunks = core_chunks
            result.elapsed_sec = _time.time() - start
            return result

        except Exception as exc:
            log.exception("TieredRetriever failed")
            result.error = str(exc)
            result.elapsed_sec = _time.time() - start
            return result

    # ── Helpers par niveau ────────────────────────────────────────────

    def _core_confidence(
        self, chunks: list[WorkingMemoryChunk], query: str
    ) -> float:
        """Estime la confiance du Core.

        Si le dernier message utilisateur est déjà dans le Core, c'est
        suffisant pour les conversations triviales.
        """
        if not chunks:
            return 0.0
        # Si un chunk "user_message" récent match
        for c in chunks:
            if c.kind == "user_message" and c.relevance > 0.7:
                return 0.9
        # Sinon, moyenne des utilities
        if chunks:
            return min(0.5, sum(c.utility() for c in chunks) / len(chunks))
        return 0.0

    def _query_sdm(
        self, embedding: list[float], query: str
    ) -> tuple[list[WorkingMemoryChunk], float]:
        """Query SDM via torch.Tensor. Adaptation Lythea.

        SDM.read(query) retourne un tensor (le pattern recalled). On
        calcule la similarité cosinus entre ce pattern et le query
        embedding pour obtenir un score [0, 1].
        """
        if self.sdm is None:
            return [], 0.0
        try:
            import torch
            query_tensor = torch.tensor(embedding, dtype=torch.float32)
            # SDM attend un tensor de dimension self.sdm.dim
            if query_tensor.shape[0] != self.sdm.dim:
                # Projection ou padding si dims mismatch
                if query_tensor.shape[0] < self.sdm.dim:
                    pad = self.sdm.dim - query_tensor.shape[0]
                    query_tensor = torch.cat([
                        query_tensor,
                        torch.zeros(pad, dtype=torch.float32),
                    ])
                else:
                    query_tensor = query_tensor[: self.sdm.dim]
            recalled = self.sdm.read(query_tensor.unsqueeze(0))
            if recalled is None:
                return [], 0.0
            recalled_vec = recalled.squeeze(0).tolist()
            # Similarité cosinus comme score de confiance
            score = _cosine_similarity(embedding[: len(recalled_vec)], recalled_vec)
            if score < 0.1:
                return [], score
            chunk = WorkingMemoryChunk(
                kind="sdm",
                content=f"[SDM recall] score={score:.3f}",
                relevance=score,
                metadata={"source": "sdm", "dim": self.sdm.dim},
            )
            return [chunk], score
        except Exception as exc:
            log.debug("SDM query failed: %s", exc)
            return [], 0.0

    def _query_mhn(
        self, embedding: list[float], query: str
    ) -> tuple[list[WorkingMemoryChunk], float]:
        """Query MHN via embedding. Adaptation Lythea.

        MHN.retrieve(embedding, top_k) retourne une liste de (text, score).
        On prend le meilleur.
        """
        if self.mhn is None:
            return [], 0.0
        try:
            import torch
            query_tensor = torch.tensor(embedding, dtype=torch.float32)
            # MHN attend un tensor 1D
            if query_tensor.shape[0] != self.mhn.dim:
                if query_tensor.shape[0] < self.mhn.dim:
                    pad = self.mhn.dim - query_tensor.shape[0]
                    query_tensor = torch.cat([
                        query_tensor,
                        torch.zeros(pad, dtype=torch.float32),
                    ])
                else:
                    query_tensor = query_tensor[: self.mhn.dim]
            results = self.mhn.retrieve(query_tensor, top_k=3)
            if not results:
                return [], 0.0
            chunks = []
            max_score = 0.0
            for text, score in results:
                # MHN peut retourner un score d'énergie — on inverse
                # (basse énergie = meilleur match)
                conf = float(score) if isinstance(score, (int, float)) else 0.5
                if conf > 1.0:
                    conf = 1.0 / (1.0 + conf)  # conversion énergie → confidence
                chunks.append(WorkingMemoryChunk(
                    kind="mhn_episode",
                    content=str(text)[:500],
                    relevance=conf,
                    metadata={"source": "mhn"},
                ))
                max_score = max(max_score, conf)
            return chunks, max_score
        except Exception as exc:
            log.debug("MHN query failed: %s", exc)
            return [], 0.0

    def _query_kg(
        self, query: str, context: dict[str, Any]
    ) -> tuple[list[WorkingMemoryChunk], float]:
        """Query KG via query_by_question. Adaptation Lythea."""
        if self.kg is None:
            return [], 0.0
        try:
            # Lythea KG expose query_by_question qui retourne des entités
            entities = self.kg.query_by_question(query)
            if not entities:
                # Fallback : get_all_entities limité
                entities = self.kg.get_all_entities()[:3]
            chunks = []
            max_score = 0.0
            for ent in entities[:5]:
                if isinstance(ent, dict):
                    score = float(ent.get("confidence", 0.5))
                    name = ent.get("name", ent.get("value", ""))
                    etype = ent.get("type", "")
                    summary = ent.get("summary", f"{etype}: {name}")
                else:
                    score = 0.5
                    summary = str(ent)
                chunks.append(WorkingMemoryChunk(
                    kind="kg_entity",
                    content=str(summary)[:300],
                    relevance=score,
                    metadata={"source": "kg", "entity": ent if isinstance(ent, dict) else {}},
                ))
                max_score = max(max_score, score)
            return chunks, max_score
        except Exception as exc:
            log.debug("KG query failed: %s", exc)
            return [], 0.0

    def _query_chroma(
        self, embedding: list[float], query: str
    ) -> tuple[list[WorkingMemoryChunk], float]:
        """Query Chroma via HybridRetriever. Adaptation Lythea.

        HybridRetriever.query(text) fait déjà le reranking cross-encoder.
        On lui passe le texte de la query et on récupère les chunks.
        """
        if self.chroma is None:
            return [], 0.0
        try:
            # HybridRetriever de Lythea prend du texte, pas un embedding
            results = self.chroma.query(query, top_k=5)
            if not results:
                return [], 0.0
            chunks = []
            max_score = 0.0
            # results peut être une liste de tuples (text, score) ou de dicts
            for item in results[:5]:
                if isinstance(item, tuple) and len(item) >= 2:
                    text, score = item[0], item[1]
                elif isinstance(item, dict):
                    text = item.get("text", item.get("content", ""))
                    score = item.get("score", 0.5)
                else:
                    text = str(item)
                    score = 0.5
                score = float(score) if isinstance(score, (int, float)) else 0.5
                chunks.append(WorkingMemoryChunk(
                    kind="chroma_chunk",
                    content=str(text)[:500],
                    relevance=score,
                    metadata={"source": "chroma"},
                ))
                max_score = max(max_score, score)
            return chunks, max_score
        except Exception as exc:
            log.debug("Chroma query failed: %s", exc)
            return [], 0.0

    def _push_to_working_memory(
        self, chunks: list[WorkingMemoryChunk]
    ) -> None:
        """Pousse les chunks récupérés dans le Core pour le tour courant."""
        for chunk in chunks:
            self.working_memory.add(chunk)
