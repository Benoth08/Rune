"""Sparse Distributed Memory — Kanerva (1988) with VSA operations.

Working memory using bipolar hyper-vectors (±1), Hebbian activation
tracking, and model-aware projection matrices.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

from rune.config import SDM_DECAY, SDM_DIM, SDM_K, SDM_PRUNE_THRESHOLD, SDM_ROWS

log = logging.getLogger("rune.memory.sdm")

_ADDR_SEED = 1_234_567  # deterministic seed for the bipolar address matrix


class SparseDistributedMemory:
    """Session-scoped sparse distributed memory with VSA binding/bundling.

    Parameters
    ----------
    dim : int
        Dimension of hyper-vectors.
    rows : int
        Number of address rows.
    k : int
        Number of top-k neighbours for read/write.
    decay_factor : float
        Base exponential decay rate.
    device : str
        Torch device for tensors.
    """

    def __init__(
        self,
        dim: int = SDM_DIM,
        rows: int = SDM_ROWS,
        k: int = SDM_K,
        decay_factor: float = SDM_DECAY,
        device: str = "cpu",
    ) -> None:
        self.dim = dim
        self.rows = rows
        self.k = k
        self.decay_factor = decay_factor
        self.device = device

        # Fixed random bipolar addresses — SEEDED so a fresh instance is
        # reproducible, and persisted in save()/load_state() so the
        # address↔content-row mapping survives a restart. (Before: unseeded
        # randint + not persisted → reads hit wrong rows after reload =
        # silently corrupted working memory.)
        _g = torch.Generator().manual_seed(_ADDR_SEED)
        self.addresses = (
            (torch.randint(0, 2, (rows, dim), generator=_g) * 2 - 1)
            .float().to(device)
        )
        # Accumulator content
        self.contents = torch.zeros(rows, dim, device=device)
        # Hebbian activation counters
        self.activations = torch.zeros(rows, device=device)

        # Projection matrix: maps hidden_dim → sdm_dim (lazy init)
        self._projection: torch.Tensor | None = None
        self._proj_source: str | None = None

    # ── Projection with model tracking ─────────────────────────────────

    def _projection_key(self, model_id: str, hidden_dim: int) -> str:
        """Hash to track which model generated the projection matrix."""
        return hashlib.md5(f"{model_id}:{hidden_dim}".encode()).hexdigest()[:12]

    def project(self, latent: torch.Tensor, model_id: str = "", hidden_dim: int = 0) -> torch.Tensor:
        """Project a latent vector from model space to SDM space.

        Parameters
        ----------
        latent : torch.Tensor
            Shape ``[1, hidden_dim]`` or ``[hidden_dim]``.
        model_id : str
            Model identifier for projection matrix tracking.
        hidden_dim : int
            Explicit hidden dim (used if latent has wrong shape).

        Returns
        -------
        torch.Tensor
            Bipolar vector of shape ``[1, dim]``.
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        h = hidden_dim or latent.shape[-1]
        key = self._projection_key(model_id, h)

        if self._projection is None or self._proj_source != key:
            if self._proj_source is not None:
                log.warning("Model changed (%s → %s), regenerating projection matrix", self._proj_source, key)
            self._projection = torch.randn(h, self.dim, device=self.device) / (h ** 0.5)
            self._proj_source = key

        projected = latent.to(self.device).float() @ self._projection
        return torch.sign(projected)

    # ── Write ──────────────────────────────────────────────────────────

    def write(self, address: torch.Tensor, content: torch.Tensor, strength: float = 1.0) -> None:
        """Write content to the k nearest address rows.

        Parameters
        ----------
        address : torch.Tensor
            Query address ``[1, dim]``.
        content : torch.Tensor
            Content to store ``[1, dim]``.
        strength : float
            Write strength (clamped to [0, 3]).
        """
        strength = max(0.0, min(strength, 3.0))
        addr = address.view(1, -1).to(self.device)
        cont = content.view(1, -1).to(self.device)

        # Cosine similarity
        norms_a = self.addresses.norm(dim=1, keepdim=True).clamp(min=1e-8)
        norms_q = addr.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cosines = (self.addresses / norms_a) @ (addr / norms_q).T
        cosines = cosines.squeeze(1)

        _, topk_idx = cosines.topk(self.k)

        self.contents[topk_idx] += cont * strength
        self.activations[topk_idx] += strength

    # ── Read ───────────────────────────────────────────────────────────

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from memory using softmax-weighted top-k neighbours.

        Parameters
        ----------
        query : torch.Tensor
            Query address ``[1, dim]``.

        Returns
        -------
        torch.Tensor
            Recalled bipolar vector ``[1, dim]``.
        """
        q = query.view(1, -1).to(self.device)

        norms_a = self.addresses.norm(dim=1, keepdim=True).clamp(min=1e-8)
        norms_q = q.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cosines = (self.addresses / norms_a) @ (q / norms_q).T
        cosines = cosines.squeeze(1)

        _, topk_idx = cosines.topk(self.k)
        weights = torch.softmax(cosines[topk_idx], dim=0)
        recalled = (weights.unsqueeze(1) * self.contents[topk_idx]).sum(dim=0, keepdim=True)

        return torch.sign(recalled)

    # ── Hebbian decay ──────────────────────────────────────────────────

    def decay(self, factor: float | None = None) -> None:
        """Apply Hebbian decay — active rows decay slower.

        Parameters
        ----------
        factor : float, optional
            Base decay factor. Defaults to ``self.decay_factor``.
        """
        f = factor if factor is not None else self.decay_factor
        per_row = f ** (1.0 / (1.0 + torch.log1p(self.activations)))
        self.contents *= per_row.unsqueeze(1)
        self.activations *= 0.98

    # ── Prune ──────────────────────────────────────────────────────────

    def prune(self, threshold: float = SDM_PRUNE_THRESHOLD) -> int:
        """Zero-out rows with norm below threshold.

        Returns
        -------
        int
            Number of pruned rows.
        """
        norms = self.contents.norm(dim=1)
        weak = norms < threshold
        count = weak.sum().item()
        if count > 0:
            self.contents[weak] = 0
            self.activations[weak] = 0
        return int(count)

    # ── Rehearsal (sleep replay) ───────────────────────────────────────

    def rehearse(self, top_k: int = 16, boost: float = 0.15) -> None:
        """Boost the most active rows (sleep replay, Buzsáki)."""
        if self.activations.sum() < 1e-6:
            return
        k = min(top_k, self.rows)
        _, active_idx = self.activations.topk(k)
        self.contents[active_idx] *= (1.0 + boost)

    # ── Flush ──────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Reset all content and activations (session switch)."""
        self.contents.zero_()
        self.activations.zero_()

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save state to disk atomically."""
        import os
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "contents": self.contents.cpu(),
            "activations": self.activations.cpu(),
            "addresses": self.addresses.cpu(),
            "projection": self._projection.cpu() if self._projection is not None else None,
            "proj_source": self._proj_source,
        }, str(tmp))
        os.replace(str(tmp), str(path))

    def load_state(self, path: Path) -> None:
        """Load state from disk."""
        if not path.exists():
            return
        state = torch.load(str(path), map_location=self.device, weights_only=True)
        loaded_contents = state["contents"]
        if loaded_contents.shape == self.contents.shape:
            self.contents = loaded_contents.to(self.device)
            self.activations = state["activations"].to(self.device)
            # Restore the address matrix that these contents were written
            # under — otherwise top-k addressing selects the wrong rows.
            loaded_addr = state.get("addresses")
            if loaded_addr is not None and loaded_addr.shape == self.addresses.shape:
                self.addresses = loaded_addr.to(self.device)
            elif loaded_addr is None:
                log.warning("SDM save predates address persistence; addressing "
                            "may not match pre-existing contents (one-time)")
        else:
            log.warning("SDM shape mismatch on load, starting fresh")

        if state.get("projection") is not None:
            self._projection = state["projection"].to(self.device)
            self._proj_source = state.get("proj_source")

    # ── VSA operations ─────────────────────────────────────────────────

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hadamard product binding."""
        return a * b

    @staticmethod
    def bundle(*vectors: torch.Tensor) -> torch.Tensor:
        """Majority-rule bundling (signed sum)."""
        return torch.sign(sum(vectors))


# Alias rétrocompatible. L'API historique Lythea exposait la classe sous
# le nom court ``SDM`` ; certains modules et tests l'importent encore ainsi.
# On garde l'alias pour ne pas casser ces imports.
SDM = SparseDistributedMemory
