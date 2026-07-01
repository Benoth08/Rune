"""Surprise phase — composite biomimetic surprise + doubt calibration.

Biological inspiration
----------------------
Surprise is what tells the hippocampus *whether* to bother
encoding an event in the first place. In rodents, novel locations
trigger sharp-wave ripples and dopaminergic bursts that gate
plasticity in CA1; familiar locations trigger almost nothing.
This module is Lythéa's gating signal — its output is consumed
by Storage (SDM write strength) and by Consolidation (ripple
trigger counter).

The composite surprise blends four signals, each capturing a
different "kind" of novelty:

1. **Structural surprise** — mean per-token entropy of the user
   utterance under the LLM. High entropy = the language model
   itself didn't see this coming. Cognitive analogue: cortical
   prediction error.

2. **Episodic surprise** — Hopfield energy of the GLiNER
   embedding under the MHN. High energy = the embedding is far
   from any stored pattern, i.e. no episodic match. Cognitive
   analogue: hippocampal CA3 mismatch.

3. **Predictive surprise** — cosine *distance* between the SDM
   read of the model latent (the "prior") and the latent itself
   (the "posterior"). High distance = the SDM did not anticipate
   this representation. Cognitive analogue: top-down prediction
   error from prior beliefs.

4. **Chroma discount** — semantic similarity to the closest
   long-term memory in Chroma. High similarity *reduces* surprise
   multiplicatively — we already know about this, no need to
   re-encode. Cognitive analogue: neocortical consolidation
   amortizer.

The composite is::

    S_composite = w₁·S_structural + w₂·S_episodic + w₃·S_predictive
    S_global    = S_composite × (1 - discount_chroma)

Weights live in :mod:`lythea.config` so calibration is data-driven.

The phase also computes the **doubt index** post-generation, from
the per-token entropies of the LLM's *output*. This is the dual
of structural surprise but on the response side — it grades the
model's own confidence in what it just produced::

    doubt = mean(out_entropies) / max(threshold, 0.1)

with epistemic labels: ``fait`` (<0.3), ``intuition`` (<0.8),
``hypothese`` (≥0.8).

Design notes
------------
- Every signal is independently wrapped: a torch RuntimeError on
  one of them must not mask the others. We default failed signals
  to their *neutral* value (1.0 for surprise components, 0.0 for
  the discount) — i.e. "no information, assume novel".
- The dataclass exposes ``as_dict()`` whose JSON-serialised form
  is byte-identical to the original ``_compute_surprise`` return,
  preserving the contract for the frontend and Storage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from rune.config import (
    SURPRISE_W_EPISODIC,
    SURPRISE_W_PREDICTIVE,
    SURPRISE_W_STRUCTURAL,
)

log = logging.getLogger("lythea.cognition.surprise")


# Doubt-index thresholds. Below 0.3 the model is essentially
# certain — we tag the output as fact. Above 0.8 the model is
# struggling — we tag it as hypothesis. The middle band gets
# "intuition" which is what we display to the user as "I think
# but I'm not sure". Keep these in sync with the frontend badge
# colours.
DOUBT_FACT_MAX: float = 0.3
DOUBT_INTUITION_MAX: float = 0.8

EPISTEMIC_FACT: str = "fait"
EPISTEMIC_INTUITION: str = "intuition"
EPISTEMIC_HYPOTHESIS: str = "hypothese"

# Floor for the threshold normalisation in doubt computation.
# Without it, a tiny entropy_threshold would make the divisor
# vanish and doubt explode. 0.1 gives us "at most 10× the threshold
# can read as full doubt".
_DOUBT_THRESHOLD_FLOOR: float = 0.1
_DOUBT_DIVISOR_FLOOR: float = 0.01

# Chroma discount upper bound — even if a perfect match exists,
# we never fully zero out the surprise. The 0.95 cap preserves a
# minimal residual gating signal for the downstream phases.
_CHROMA_DISCOUNT_MAX: float = 0.95


@dataclass
class SurpriseSignals:
    """Composite biomimetic surprise — output of :class:`SurprisePhase`.

    Attributes
    ----------
    structural, episodic, predictive
        The three additive surprise components, each in [0, 1].
        Default to their *neutral* values when the corresponding
        signal could not be computed.
    chroma_discount
        Multiplicative amortizer in [0, 0.95]. ``0`` means no
        long-term match was found.
    composite
        Weighted additive blend before the Chroma amortizer.
    global_
        Final, JSON-clipped surprise in [0, 1]. The trailing
        underscore avoids clashing with Python's ``global``
        keyword. JSON serialisation through :meth:`as_dict` uses
        the bare key ``"global"``.
    """

    structural: float = 1.0
    episodic: float = 1.0
    predictive: float = 1.0
    chroma_discount: float = 0.0
    composite: float = 0.0
    global_: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Serialise to the frontend-facing JSON shape.

        The key order and rounding are byte-identical to the
        original ``_compute_surprise`` return so existing
        consumers (frontend doubt badge, Storage metadata)
        keep working without changes.
        """
        return {
            "structural": round(self.structural, 3),
            "episodic": round(self.episodic, 3),
            "predictive": round(self.predictive, 3),
            "chroma_discount": round(self.chroma_discount, 3),
            "composite": round(self.composite, 3),
            "global": round(min(self.global_, 1.0), 3),
        }


class SurprisePhase:
    """Compute composite surprise + doubt index.

    Parameters
    ----------
    sdm
        :class:`SparseDistributedMemory`. Used by the predictive
        signal: project the latent, read it back, measure the
        cosine distance.
    mhn
        :class:`ModernHopfieldNetwork`. Used by the episodic
        signal: Hopfield energy of the GLiNER embedding.
    model
        :class:`HFModelWrapper`. Read for ``model_id``,
        ``hidden_dim``, ``is_loaded``. Never invoked.
    retriever
        :class:`HybridRetriever` or ``None``. Used by the Chroma
        discount: top-1 search on the long-term store. ``None``
        → discount stays at 0.
    """

    def __init__(
        self,
        sdm: Any,
        mhn: Any,
        model: Any,
        retriever: Any | None,
    ) -> None:
        self.sdm = sdm
        self.mhn = mhn
        self.model = model
        self.retriever = retriever

    # ── Composite surprise ─────────────────────────────────────────────

    def compute(
        self,
        text: str,
        gliner_emb: torch.Tensor | None,
        structural_entropy: float,
        model_latent: torch.Tensor | None,
    ) -> SurpriseSignals:
        """Run the four signals and combine them.

        Each signal is computed in its own try-block so a failure
        on one (e.g. SDM dim mismatch) cannot poison the others.
        """
        s_structural = self._signal_structural(structural_entropy)
        s_episodic = self._signal_episodic(gliner_emb)
        s_predictive = self._signal_predictive(model_latent)
        discount_chroma = self._signal_chroma_discount(text)

        s_composite = (
            SURPRISE_W_STRUCTURAL * s_structural
            + SURPRISE_W_EPISODIC * s_episodic
            + SURPRISE_W_PREDICTIVE * s_predictive
        )
        s_global = s_composite * (1.0 - discount_chroma)

        return SurpriseSignals(
            structural=s_structural,
            episodic=s_episodic,
            predictive=s_predictive,
            chroma_discount=discount_chroma,
            composite=s_composite,
            global_=s_global,
        )

    # ── Doubt index — post-generation ──────────────────────────────────

    @staticmethod
    def doubt_from_entropies(
        entropies: list[float],
        threshold: float,
    ) -> tuple[float, str]:
        """Compute output-side doubt from per-token entropies.

        Parameters
        ----------
        entropies
            Per-token entropies of the LLM's output stream.
        threshold
            ``self.entropy_threshold`` from the orchestrator. Acts
            as the normaliser: outputs that hover near the threshold
            score around 1.0; well-confident outputs trend to 0.

        Returns
        -------
        tuple[float, str]
            ``(doubt_index, epistemic_label)``.
        """
        if not entropies:
            return 0.0, EPISTEMIC_FACT

        divisor = max(
            len(entropies) * max(threshold, _DOUBT_THRESHOLD_FLOOR),
            _DOUBT_DIVISOR_FLOOR,
        )
        doubt = sum(entropies) / divisor

        if doubt < DOUBT_FACT_MAX:
            label = EPISTEMIC_FACT
        elif doubt < DOUBT_INTUITION_MAX:
            label = EPISTEMIC_INTUITION
        else:
            label = EPISTEMIC_HYPOTHESIS
        return doubt, label

    # ── Internals — one method per signal ──────────────────────────────

    def _signal_structural(self, structural_entropy: float) -> float:
        """LLM input entropy, clipped to [0, 1]."""
        return min(structural_entropy, 1.0)

    def _signal_episodic(self, gliner_emb: torch.Tensor | None) -> float:
        """Hopfield energy under the MHN.

        Neutral value 1.0 is returned when no embedding is provided
        or the MHN call fails — i.e. "we cannot say it's familiar,
        therefore assume novel".
        """
        if gliner_emb is None:
            return 1.0
        try:
            return float(self.mhn.energy(gliner_emb))
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("MHN energy failed: %s", exc)
            return 1.0

    def _signal_predictive(self, model_latent: torch.Tensor | None) -> float:
        """SDM prior vs latent posterior, as cosine distance.

        This is the only signal that re-uses the *same projection*
        as the Storage SDM write (``sdm.project`` with the LLM
        ``model_id`` + ``hidden_dim``). Consistency here matters:
        if the projection differed, the predictive signal would
        be measuring noise rather than prediction error.
        """
        if model_latent is None or not getattr(self.model, "is_loaded", False):
            return 1.0
        try:
            model_id = self.model.model_id or ""
            latent_2d = (
                model_latent.unsqueeze(0)
                if model_latent.dim() == 1
                else model_latent
            )
            vec = self.sdm.project(
                latent_2d,
                model_id=model_id,
                hidden_dim=self.model.hidden_dim,
            )
            prior = self.sdm.read(vec)
            cos = torch.nn.functional.cosine_similarity(
                vec.view(1, -1).float(),
                prior.view(1, -1).float(),
            ).item()
            return max(0.0, 1.0 - cos)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Predictive signal failed: %s", exc)
            return 1.0

    def _signal_chroma_discount(self, text: str) -> float:
        """Top-1 semantic similarity to the long-term store, capped at 0.95.

        We deliberately call ``rerank=False`` here — the discount
        only needs a coarse "is this familiar?" signal, and we
        want this hot path to be cheap. Reranking is reserved for
        the Retrieval phase.
        """
        if not self.retriever:
            return 0.0
        try:
            results = self.retriever.search(text, n=1, rerank=False)
            if not results:
                return 0.0
            score = results[0].get("rerank_score", results[0].get("score", 0))
            return min(max(float(score), 0.0), _CHROMA_DISCOUNT_MAX)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Chroma discount failed: %s", exc)
            return 0.0
