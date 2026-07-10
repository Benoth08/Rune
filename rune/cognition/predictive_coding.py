"""V4.2 — Predictive coding: Friston-style cortical prediction.

Inspiration biologique
----------------------
Théorie du codage prédictif (Karl Friston). Le cerveau ne traite pas
chaque stimulus de zéro : il compare l'observation à une prédiction
descendante et n'alloue de ressources qu'à l'erreur de prédiction.

Application
-----------
Lythéa maintient une prédiction (EMA) de l'embedding du prochain
message utilisateur. À chaque tour :
- Calcule l'embedding observé.
- Cosine distance entre prédit et observé = "surprise".
- Si surprise faible → mode "low_power" (réponse rapide, pas de RAG).
- Si surprise normale → "full".
- Si surprise très haute → "high" (force RAG + plus de tokens).

Position dans le cycle cognitif
-------------------------------
Hook A.3 (Phase A, après learn → l'embedding `encoding_emb` est exposé) :
    decision = predictive_coding.observe(encoding_emb)
    → GatingDecision(mode, confidence, error)

Hook B (Phase B, gating) :
    Si pc_apply_gating ET decision.mode == "low_power" :
        Supprimer le web search non explicitement demandé.

Design contracts
----------------
1. Cold-start : observations < cold_start_min → toujours "full".
2. Crash interne → GatingDecision(mode="full", reason="error").
3. State non persistant (utile pour 1 seule session).
4. Pure Python — pas de torch, pas de numpy.
5. Embeddings entrants : list[float], len arbitraire mais stable.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

log = logging.getLogger("rune.cognition.predictive_coding")


# ── Constants ────────────────────────────────────────────────────────

GATING_MODES = ("low_power", "full", "high")


# ════════════════════════════════════════════════════════════════════
# Math helpers — pure Python, no numpy.
# ════════════════════════════════════════════════════════════════════


def _l2_norm(v: Sequence[float]) -> float:
    """L2 norm of a vector. 0 for empty/all-zeros."""
    if not v:
        return 0.0
    s = 0.0
    for x in v:
        try:
            s += float(x) * float(x)
        except (TypeError, ValueError):
            return 0.0
    return math.sqrt(s)


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance = 1 - cos_sim, clamped to [0, 2].

    Edge cases:
    - empty/null vector → returns 1.0 (max uncertainty).
    - mismatched lengths → returns 1.0.
    - either zero-norm → returns 1.0.
    """
    if not a or not b or len(a) != len(b):
        return 1.0
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    try:
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
    except (TypeError, ValueError):
        return 1.0
    sim = dot / (na * nb)
    # Numerical safety
    sim = max(-1.0, min(1.0, sim))
    return max(0.0, min(2.0, 1.0 - sim))


def _ema_predict(history: Sequence[Sequence[float]], decay: float) -> list[float]:
    """Exponential weighted average over a history of embeddings.

    Most recent has highest weight. `decay ∈ [0, 1]` controls how fast
    older entries are forgotten:
        weight_i = decay^(n-1-i) for i in 0..n-1
    """
    if not history:
        return []
    n = len(history)
    dim = len(history[-1])
    if dim == 0:
        return []

    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_w = sum(weights)
    if total_w <= 0.0:
        return list(history[-1])  # fallback to last

    out = [0.0] * dim
    for i, vec in enumerate(history):
        if len(vec) != dim:
            continue  # skip mismatched
        w = weights[i] / total_w
        for j, x in enumerate(vec):
            try:
                out[j] += w * float(x)
            except (TypeError, ValueError):
                continue
    return out


# ════════════════════════════════════════════════════════════════════
# Config + Decision dataclasses
# ════════════════════════════════════════════════════════════════════


@dataclass
class PredictiveCodingConfig:
    history_size: int = 8
    cold_start_min: int = 3
    ema_decay: float = 0.6
    low_threshold: float = 0.15
    high_threshold: float = 0.65
    confidence_cap: float = 0.85
    gating_w_sdm: float = 0.0  # V5.9 — poids du signal SDM dans l'erreur de gating (0 = EMA seule)
    # V4.0.2 — auto-calibration of error thresholds.
    # When ``auto_calibrate=True`` (default), the module observes its
    # own ``error`` distribution after cold-start. Once
    # ``auto_calibrate_min_samples`` errors are collected, the
    # ``low_threshold`` becomes the empirical P25 and ``high_threshold``
    # the empirical P75 of observations. This makes the module
    # discriminative on any LLM regardless of intrinsic embedding
    # geometry. Set to False to keep the fixed bootstrap values.
    auto_calibrate: bool = True
    auto_calibrate_min_samples: int = 20
    auto_calibrate_window: int = 200


@dataclass
class GatingDecision:
    mode: str = "full"
    confidence: float = 0.5
    error: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "confidence": self.confidence,
            "error": self.error,
            "reason": self.reason,
        }


# ════════════════════════════════════════════════════════════════════
# PredictiveCodingPhase
# ════════════════════════════════════════════════════════════════════


class PredictiveCodingPhase:
    """Maintain an EMA prediction of the next observed embedding.

    Lifetime: a single Hippocampe instance keeps one PredictiveCodingPhase.
    State (history) is in-memory only; not persisted across restarts.
    """

    def __init__(self, config: PredictiveCodingConfig | None = None):
        self.config = config or PredictiveCodingConfig()
        self._history: deque[list[float]] = deque(maxlen=self.config.history_size)
        # Running stats for reporting / debug
        self.n_observations = 0
        self.last_decision: GatingDecision | None = None
        # V4.0.2 — auto-calibrator. We track only the post-cold-start
        # `error` values (cold-start errors are 0.0 by convention and
        # would skew the distribution).
        from rune.cognition.auto_calibrator import (
            AutoCalibratedThresholds, QuantileCalibrator,
        )
        self._auto_calibrator = QuantileCalibrator(
            window=self.config.auto_calibrate_window,
            storage_path=None,  # in-memory only — distribution is per-session
        )
        self._auto_thresholds = AutoCalibratedThresholds(
            bootstrap_low=self.config.low_threshold,
            bootstrap_high=self.config.high_threshold,
            bootstrap_very_high=self.config.high_threshold,  # not used here
            min_samples=self.config.auto_calibrate_min_samples,
            p_low=0.25,
            p_high=0.75,
        )

    def _current_thresholds(self) -> tuple[float, float]:
        """Return (low, high) effective thresholds (auto or bootstrap)."""
        if not self.config.auto_calibrate:
            return self.config.low_threshold, self.config.high_threshold
        low, high, _ = self._auto_thresholds.get_thresholds(
            self._auto_calibrator,
        )
        return low, high

    def is_bootstrapping(self) -> bool:
        """True while the auto-calibrator hasn't collected enough errors."""
        return (
            self.config.auto_calibrate
            and self._auto_thresholds.is_bootstrapping(self._auto_calibrator)
        )

    def observe(self, embedding: Sequence[float] | None, sdm_error: float | None = None) -> GatingDecision:
        """Compare `embedding` with the EMA prediction; classify mode.

        Pipeline
        --------
        1. Defensive validation : non-empty, finite values.
        2. If observations < cold_start_min: append, return "full".
        3. Predict from history (excluding the new one).
        4. error = cosine_distance(predicted, observed).
        5. mode :
            error < low_threshold  → low_power
            error > high_threshold → high
            else                   → full
        6. confidence = min(confidence_cap, 0.5 + 0.5·|error - 0.4|)
           (high confidence at extremes, low in the middle).
        7. Append observation to history.
        8. Cache last_decision for telemetry.

        V5.9 — ``sdm_error`` (optionnel) : distance de lecture SDM. Si
        fourni ET ``gating_w_sdm > 0``, il est mélangé à l'erreur EMA
        pour décider le mode. Sert de juge d'ablation pour trancher si
        le signal SDM apporte de la variance discriminante.
        """
        try:
            return self._observe_inner(embedding, sdm_error)
        except Exception:
            log.warning("PredictiveCodingPhase.observe crashed", exc_info=True)
            return GatingDecision(
                mode="full",
                confidence=0.5,
                error=0.0,
                reason="error: internal crash, falling back to full",
            )

    def _observe_inner(self, embedding: Sequence[float] | None, sdm_error: float | None = None) -> GatingDecision:
        # 1. Validate
        if embedding is None or len(embedding) == 0:
            decision = GatingDecision(
                mode="full",
                confidence=0.3,
                error=0.0,
                reason="no embedding provided",
            )
            self.last_decision = decision
            return decision

        # Convert to list of floats once, defensively.
        try:
            obs = [float(x) for x in embedding]
        except (TypeError, ValueError):
            decision = GatingDecision(
                mode="full",
                confidence=0.3,
                error=0.0,
                reason="embedding contained non-numeric values",
            )
            self.last_decision = decision
            return decision

        # 2. Cold start
        if self.n_observations < self.config.cold_start_min:
            self._history.append(obs)
            self.n_observations += 1
            decision = GatingDecision(
                mode="full",
                confidence=0.4,
                error=0.0,
                reason=f"cold-start ({self.n_observations}/{self.config.cold_start_min})",
            )
            self.last_decision = decision
            return decision

        # 3. Predict from current history (BEFORE appending obs)
        predicted = _ema_predict(list(self._history), self.config.ema_decay)
        if not predicted or len(predicted) != len(obs):
            # Predictor unable (shouldn't happen post cold-start) — fall to full.
            self._history.append(obs)
            self.n_observations += 1
            decision = GatingDecision(
                mode="full",
                confidence=0.3,
                error=0.0,
                reason="predictor returned mismatched dimension",
            )
            self.last_decision = decision
            return decision

        # 4. Error (EMA) + injection SDM (V5.9)
        error_ema = cosine_distance(predicted, obs)
        w = float(getattr(self.config, "gating_w_sdm", 0.0) or 0.0)
        if sdm_error is not None and w > 0.0:
            try:
                error = (1.0 - w) * error_ema + w * float(sdm_error)
            except (TypeError, ValueError):
                error = error_ema
        else:
            error = error_ema

        # V4.0.2 — Feed the auto-calibrator BEFORE deciding so the
        # current observation also informs its own band post-bootstrap.
        # Cold-start errors are not fed (they're 0.0 by convention and
        # would distort the empirical distribution).
        try:
            self._auto_calibrator.observe(float(error))
        except Exception:
            pass
        low_thr, high_thr = self._current_thresholds()

        # 5. Mode
        if error < low_thr:
            mode = "low_power"
            reason = f"low surprise ({error:.3f} < {low_thr:.3f})"
        elif error > high_thr:
            mode = "high"
            reason = f"high surprise ({error:.3f} > {high_thr:.3f})"
        else:
            mode = "full"
            reason = f"normal surprise ({error:.3f})"

        # Annotate reason with calibration source for debugging.
        if self.config.auto_calibrate:
            src = "bootstrap" if self.is_bootstrapping() else "empirical"
            reason = f"{reason} [{src}]"

        # V5.9 — logging de divergence SDM (juge de l'ablation). On
        # recalcule le mode qu'aurait donné l'EMA seule et on logge quand
        # le signal SDM fait basculer la décision : c'est la mesure
        # directe de « la SDM change-t-elle quelque chose ? ».
        if sdm_error is not None and w > 0.0:
            if error_ema < low_thr:
                _mode_ema = "low_power"
            elif error_ema > high_thr:
                _mode_ema = "high"
            else:
                _mode_ema = "full"
            if _mode_ema != mode:
                log.info(
                    "PC gating divergence: EMA=%.3f->%s | +SDM(err=%.3f,w=%.2f)=%.3f->%s",
                    error_ema, _mode_ema, float(sdm_error), w, error, mode,
                )

        # 6. Confidence — peaks at extremes, dips in middle.
        # |error - 0.4| has max ~1.0 (at error=0 or error=1.4 capped to 1.0)
        center_dist = abs(error - 0.4)
        confidence = min(self.config.confidence_cap, 0.5 + 0.5 * min(1.0, center_dist))

        # 7. Append observation
        self._history.append(obs)
        self.n_observations += 1

        decision = GatingDecision(
            mode=mode,
            confidence=confidence,
            error=error,
            reason=reason,
        )
        self.last_decision = decision
        return decision

    def reset(self) -> None:
        self._history.clear()
        self.n_observations = 0
        self.last_decision = None
        # Reset auto-calibrator so a new session starts with a fresh
        # distribution (each conversation may have a different topic
        # mix and thus a different error distribution).
        try:
            self._auto_calibrator.reset()
        except Exception:
            pass

    def to_dict(self) -> dict:
        """Snapshot for telemetry / UI."""
        low, high = self._current_thresholds()
        is_bootstrap = self.is_bootstrapping()
        return {
            "n_observations": self.n_observations,
            "thresholds_in_use": {
                "low": round(low, 4),
                "high": round(high, 4),
                "source": "bootstrap" if is_bootstrap else "empirical",
            },
            "auto_calibrator": self._auto_calibrator.to_dict(),
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
        }
