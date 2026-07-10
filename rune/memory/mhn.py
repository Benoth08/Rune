"""Modern Hopfield Network — Ramsauer et al. 2020.

Episodic memory for exact phrase recall via softmax attention with high β.
Uses a ring buffer for O(1) FIFO eviction instead of torch.roll.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch

from rune.config import MHN_BETA, MHN_DIM, MHN_MAX_PATTERNS

log = logging.getLogger("rune.memory.mhn")


class ModernHopfieldNetwork:
    """Session-scoped episodic memory using Modern Hopfield attention.

    Parameters
    ----------
    max_patterns : int
        Maximum number of stored patterns (FIFO ring buffer).
    dim : int
        Embedding dimension (768 = DeBERTa-v3 from GLiNER).
    beta : float
        Inverse temperature for softmax attention.
    device : str
        Torch device.
    """

    def __init__(
        self,
        max_patterns: int = MHN_MAX_PATTERNS,
        dim: int = MHN_DIM,
        beta: float = MHN_BETA,
        device: str = "cpu",
    ) -> None:
        self.max_patterns = max_patterns
        self.dim = dim
        self.beta = beta
        self.device = device

        self.W = torch.zeros(max_patterns, dim, device=device)
        self.texts: list[str | None] = [None] * max_patterns
        self.timestamps: list[float] = [0.0] * max_patterns

        # Ring buffer state
        self._head = 0
        self._count = 0

    @property
    def n_stored(self) -> int:
        return self._count

    # ── Store ──────────────────────────────────────────────────────────

    def store(self, embedding: torch.Tensor, text: str) -> None:
        """Store a pattern at the ring buffer head.

        Parameters
        ----------
        embedding : torch.Tensor
            Pattern vector of shape ``[dim]`` or ``[1, dim]``.
        text : str
            Associated text for retrieval.
        """
        emb = embedding.view(-1).to(self.device)
        if emb.shape[0] != self.dim:
            log.warning("MHN dim mismatch: got %d, expected %d", emb.shape[0], self.dim)
            return

        idx = self._head
        self.W[idx] = emb
        self.texts[idx] = text
        self.timestamps[idx] = time.time()

        self._head = (self._head + 1) % self.max_patterns
        self._count = min(self._count + 1, self.max_patterns)

    # ── Retrieve ───────────────────────────────────────────────────────

    def energy(self, query: torch.Tensor) -> float:
        """Compute Hopfield energy for a query — episodic surprise signal.

        Uses max cosine similarity against stored patterns rather than
        softmax attention, which degenerates when n_stored is small.

        High energy = no stored pattern matches = novel information.
        Low energy = strong match = familiar information.

        Parameters
        ----------
        query : torch.Tensor
            Query vector ``[dim]`` or ``[1, dim]``.

        Returns
        -------
        float
            Normalized energy in ``[0, 1]``. 1.0 = maximally novel.
        """
        if self._count == 0:
            return 1.0

        q = query.view(1, -1).to(self.device)
        active = self.W[:self._count]

        # Cosine similarity against all stored patterns
        q_norm = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
        a_norm = active / active.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cosines = (a_norm @ q_norm.T).squeeze(1)

        # Max cosine = best match; high = familiar, low = novel
        max_cos = cosines.max().item()

        # Map to surprise: cosine 1.0 → energy 0.0, cosine 0.0 → energy 1.0
        return float(max(0.0, 1.0 - max(max_cos, 0.0)))

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 3,
        min_attention: float = 0.15,
    ) -> list[dict]:
        """Retrieve top-k patterns via softmax attention.

        Parameters
        ----------
        query : torch.Tensor
            Query vector ``[dim]`` or ``[1, dim]``.
        top_k : int
            Number of results to return.
        min_attention : float
            Minimum attention weight to include.

        Returns
        -------
        list[dict]
            Each dict: ``{text, attention, timestamp, index}``.
        """
        if self._count == 0:
            return []

        q = query.view(1, -1).to(self.device)
        n = self._count
        active = self.W[:n]

        # Softmax attention: attention = softmax(β · W @ q^T)
        scores = self.beta * (active @ q.T).squeeze(1)
        attention = torch.softmax(scores, dim=0)

        k = min(top_k, n)
        top_vals, top_idx = attention.topk(k)

        results = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if val < min_attention:
                continue
            text = self.texts[idx]
            if text is None:
                continue
            results.append({
                "text": text,
                "attention": val,
                "timestamp": self.timestamps[idx],
                "index": idx,
            })

        return results

    # ── Clear ──────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset all stored patterns (session switch)."""
        self.W.zero_()
        self.texts = [None] * self.max_patterns
        self.timestamps = [0.0] * self.max_patterns
        self._head = 0
        self._count = 0

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save state atomically."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "W": self.W[:self._count].cpu(),
            "texts": self.texts[:self._count],
            "timestamps": self.timestamps[:self._count],
            "head": self._head,
            "count": self._count,
        }, str(tmp))
        os.replace(str(tmp), str(path))

    def load_state(self, path: Path) -> None:
        """Load state from disk."""
        if not path.exists():
            return
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        count = state["count"]
        w = state["W"]
        if w.shape[1] != self.dim:
            log.warning("MHN dim mismatch on load, starting fresh")
            return
        n = min(count, self.max_patterns)
        self.W[:n] = w[:n].to(self.device)
        self.texts[:n] = state["texts"][:n]
        self.timestamps[:n] = state["timestamps"][:n]
        self._head = state.get("head", n % self.max_patterns)
        self._count = n
