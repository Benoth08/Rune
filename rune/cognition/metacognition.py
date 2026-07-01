"""V4.4 — Metacognition: self-monitoring of certainty + calibration.

Inspiration biologique
----------------------
Cortex préfrontal médian (mPFC) + cortex cingulaire antérieur dorsal.
Évalue la qualité de ses propres décisions en cours d'exécution :
"suis-je en train de me tromper ? mon niveau de confiance est-il
calibré ?".

Position dans le cycle cognitif
-------------------------------
Hook M (Phase E, après strip_reasoning final, AVANT inhibition) :
    decision = metacognition.observe(
        doubt_index, epistemic, web_used, kg_facts_count
    )
    → MetacognitiveDecision(
        confidence_label, hedge_prefix, recommend_web, calibration_score
    )

Architecture
------------
3 sous-systèmes :

1. CalibrationTracker — historique de paires
   (confidence_announced, was_correct). Fenêtre glissante 100 entrées.
   Calcule un Brier-score-like. Sans label de vérité explicite, on
   approxime "was_correct" = (doubt_index final < 0.5).

2. CertaintyClassifier — règles déterministes mappant
   (doubt_index, epistemic, web_used) → label ∈
   {"très_certaine", "certaine", "incertaine", "très_incertaine"}.

3. HedgeGenerator — préfixes verbaux modulant la verbosité :
   très_certaine → ""
   certaine     → ""
   incertaine   → "Je ne suis pas totalement sûre, mais "
   très_incertaine → "C'est une zone où je manque d'éléments — "

Design contracts
----------------
1. Pure Python, pas de torch / numpy.
2. Crash interne → MetacognitiveDecision neutre (jamais raise).
3. Persistance JSON atomique pour la calibration (cumulative entre sessions).
4. Le hook ne MUTE jamais le texte — il SUGGÈRE un préfixe.
   L'application du préfixe est la responsabilité du caller (et reste
   opt-in via une sub-flag pour ne pas surprendre).
5. Web recommendation : si très_incertaine ET web non utilisé →
   recommend_web=True. Le caller décide quoi en faire.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("lythea.cognition.metacognition")


CONFIDENCE_LABELS = ("très_certaine", "certaine", "incertaine", "très_incertaine")


# ════════════════════════════════════════════════════════════════════
# CalibrationTracker — running Brier-style score
# ════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationEntry:
    announced: float  # ∈ [0, 1] confidence announced at the time
    correct: float    # ∈ [0, 1], 1.0 if turn judged successful
    timestamp: float


class CalibrationTracker:
    """Sliding-window calibration tracker (last 100 entries by default).

    Brier score = mean((announced - correct)^2). Lower is better.
    A perfectly calibrated agent has Brier ≈ variance(correct).

    Persistence: JSON atomic write (.tmp + replace) to a single file.
    Tolerates missing/corrupted files (returns to empty history).
    """

    def __init__(self, storage_path: Path | None = None, window: int = 100):
        self.window = max(1, int(window))
        self._entries: deque[CalibrationEntry] = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._storage_path = storage_path
        if storage_path is not None:
            try:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                log.warning("calibration storage_dir create failed", exc_info=True)
            self._load()

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", []) if isinstance(data, dict) else []
            for e in entries[-self.window:]:
                if not isinstance(e, dict):
                    continue
                try:
                    self._entries.append(CalibrationEntry(
                        announced=float(e.get("announced", 0.5)),
                        correct=float(e.get("correct", 0.5)),
                        timestamp=float(e.get("timestamp", time.time())),
                    ))
                except (TypeError, ValueError):
                    continue
        except Exception:
            log.warning("CalibrationTracker._load failed", exc_info=True)

    def _save_locked(self) -> None:
        if self._storage_path is None:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 1,
                        "entries": [
                            {
                                "announced": e.announced,
                                "correct": e.correct,
                                "timestamp": e.timestamp,
                            }
                            for e in self._entries
                        ],
                    },
                    f,
                    ensure_ascii=False,
                )
            tmp.replace(self._storage_path)
        except Exception:
            log.warning("CalibrationTracker._save_locked failed", exc_info=True)

    def record(self, announced: float, correct: float) -> None:
        """Append a new (announced, correct) pair. Both clamped to [0, 1]."""
        try:
            a = max(0.0, min(1.0, float(announced)))
            c = max(0.0, min(1.0, float(correct)))
        except (TypeError, ValueError):
            return
        with self._lock:
            self._entries.append(CalibrationEntry(
                announced=a, correct=c, timestamp=time.time(),
            ))
            self._save_locked()

    def brier_score(self) -> float | None:
        """Mean squared error between announced and correct.

        Returns None when the history is empty so callers can
        distinguish "no data" from "perfectly calibrated".
        """
        with self._lock:
            if not self._entries:
                return None
            errs = [(e.announced - e.correct) ** 2 for e in self._entries]
            return sum(errs) / len(errs)

    def calibration_score(self) -> float:
        """Inverse Brier in [0, 1]. Higher = better calibrated."""
        b = self.brier_score()
        if b is None:
            return 0.5  # neutral prior
        # Brier max for binary correct ∈ {0,1} and announced ∈ [0,1] is 1.
        return max(0.0, min(1.0, 1.0 - b))

    def n_entries(self) -> int:
        with self._lock:
            return len(self._entries)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._save_locked()


# ════════════════════════════════════════════════════════════════════
# CertaintyClassifier — deterministic rules
# ════════════════════════════════════════════════════════════════════


@dataclass
class CertaintyConfig:
    """Thresholds for the 4-band certainty classifier.

    Defaults are tuned for V3.9.4 doubt_index ranges observed in
    production (typically 0.05–0.7, with 0.3+ correlating with
    user-flagged inaccuracies).
    """

    very_high_doubt: float = 0.55      # > → "très_incertaine"
    high_doubt: float = 0.35           # > → "incertaine"
    low_doubt: float = 0.15            # < → "très_certaine" (with epistemic)
    epistemic_boost_threshold: float = 0.7  # epistemic ≥ this → 1 band more confident
    web_penalty_threshold: bool = True  # if web not used + high doubt → drop a band


class CertaintyClassifier:
    """Map (doubt_index, epistemic, web_used) → certainty label."""

    def __init__(self, config: CertaintyConfig | None = None):
        self.config = config or CertaintyConfig()

    def classify(
        self,
        doubt_index: float,
        epistemic: float = 0.5,
        web_used: bool = False,
    ) -> str:
        try:
            d = max(0.0, min(1.0, float(doubt_index)))
        except (TypeError, ValueError):
            d = 0.5
        try:
            e = max(0.0, min(1.0, float(epistemic)))
        except (TypeError, ValueError):
            e = 0.5

        # Base band from doubt
        if d > self.config.very_high_doubt:
            base_idx = 3  # très_incertaine
        elif d > self.config.high_doubt:
            base_idx = 2  # incertaine
        elif d > self.config.low_doubt:
            base_idx = 1  # certaine
        else:
            base_idx = 0  # très_certaine

        # Epistemic boost: high epistemic shifts one band toward "certain".
        if e >= self.config.epistemic_boost_threshold and base_idx > 0:
            base_idx -= 1

        # Web absence on high doubt: shift one band toward "incertain".
        if (
            self.config.web_penalty_threshold
            and not web_used
            and d > self.config.high_doubt
            and base_idx < 3
        ):
            base_idx += 1

        return CONFIDENCE_LABELS[base_idx]


# ════════════════════════════════════════════════════════════════════
# HedgeGenerator — verbal prefix per certainty band
# ════════════════════════════════════════════════════════════════════


_HEDGE_PREFIXES: dict[str, str] = {
    "très_certaine": "",
    "certaine": "",
    "incertaine": "Je ne suis pas totalement sûre, mais ",
    "très_incertaine": "C'est une zone où je manque d'éléments — ",
}


def hedge_prefix(label: str) -> str:
    """Return the hedge prefix for a certainty label, or '' on unknown."""
    return _HEDGE_PREFIXES.get(label, "")


# ════════════════════════════════════════════════════════════════════
# MetacognitivePhase — top-level orchestrator
# ════════════════════════════════════════════════════════════════════


@dataclass
class MetacognitiveDecision:
    confidence_label: str = "certaine"
    confidence_score: float = 0.5  # in [0, 1], higher = more certain
    hedge_prefix: str = ""
    recommend_web: bool = False
    calibration_score: float = 0.5  # rolling self-calibration ∈ [0,1]
    n_calibration_entries: int = 0

    def to_dict(self) -> dict:
        return {
            "confidence_label": self.confidence_label,
            "confidence_score": self.confidence_score,
            "hedge_prefix": self.hedge_prefix,
            "recommend_web": self.recommend_web,
            "calibration_score": self.calibration_score,
            "n_calibration_entries": self.n_calibration_entries,
        }


@dataclass
class MetacognitionConfig:
    """Configuration for the full metacognition phase.

    Bootstrap thresholds (``very_high_doubt``, ``high_doubt``,
    ``low_doubt``) are used while the auto-calibrator collects
    observations. After ``auto_calibrate_min_samples`` doubt_index
    observations, thresholds switch to empirical P25/P75/P90 of the
    observed distribution — making the module model-agnostic.

    Set ``auto_calibrate=False`` to keep the fixed bootstrap thresholds
    forever (useful for tests or when you want deterministic behaviour).
    """

    very_high_doubt: float = 0.55
    high_doubt: float = 0.35
    low_doubt: float = 0.15
    epistemic_boost_threshold: float = 0.7
    apply_hedge: bool = False  # opt-in: prepend hedge to final_text
    recommend_web_threshold: str = "très_incertaine"  # band → recommend
    calibration_window: int = 100
    # V4.0.2: auto-calibration of doubt thresholds.
    auto_calibrate: bool = True
    auto_calibrate_min_samples: int = 30
    auto_calibrate_window: int = 200


class MetacognitivePhase:
    """Coordinate calibration + classifier + hedge generation.

    V4.0.2: integrates a quantile-based auto-calibrator. The classifier
    thresholds (low/high/very_high doubt) start at the configured
    bootstrap values, then auto-adjust to the empirical P25/P75/P90 of
    observed doubt_index values once enough samples are collected.
    This makes the module work consistently across models with very
    different intrinsic doubt scales (Qwen 3B vs Qwen 7B vs Llama vs
    GPT-4-class).
    """

    def __init__(
        self,
        config: MetacognitionConfig | None = None,
        storage_path: Path | None = None,
        auto_calib_storage_path: Path | None = None,
    ):
        self.config = config or MetacognitionConfig()
        self.calibration = CalibrationTracker(
            storage_path=storage_path,
            window=self.config.calibration_window,
        )
        # V4.0.2 auto-calibrator. Persisted alongside the calibration
        # tracker so bootstrap completes faster after restart.
        from rune.cognition.auto_calibrator import (
            AutoCalibratedThresholds, QuantileCalibrator,
        )
        self._auto_calibrator = QuantileCalibrator(
            window=self.config.auto_calibrate_window,
            storage_path=auto_calib_storage_path,
        )
        self._auto_thresholds = AutoCalibratedThresholds(
            bootstrap_low=self.config.low_doubt,
            bootstrap_high=self.config.high_doubt,
            bootstrap_very_high=self.config.very_high_doubt,
            min_samples=self.config.auto_calibrate_min_samples,
        )
        self.classifier = CertaintyClassifier(
            CertaintyConfig(
                very_high_doubt=self.config.very_high_doubt,
                high_doubt=self.config.high_doubt,
                low_doubt=self.config.low_doubt,
                epistemic_boost_threshold=self.config.epistemic_boost_threshold,
            )
        )

    def _refresh_classifier_thresholds(self) -> None:
        """Sync the classifier's thresholds with the auto-calibrator.

        Called at every observe() when ``auto_calibrate`` is on. After
        ``min_samples`` observations, the thresholds become empirical
        quantiles of the observed doubt_index distribution.
        """
        if not self.config.auto_calibrate:
            return
        low, high, very_high = self._auto_thresholds.get_thresholds(
            self._auto_calibrator,
        )
        self.classifier.config.low_doubt = low
        self.classifier.config.high_doubt = high
        self.classifier.config.very_high_doubt = very_high

    def observe(
        self,
        doubt_index: float,
        epistemic: float = 0.5,
        web_used: bool = False,
        kg_facts_count: int = 0,
    ) -> MetacognitiveDecision:
        """Compute the metacognitive decision for the current turn.

        Pipeline
        --------
        1. Classify certainty.
        2. Build hedge prefix.
        3. Determine if web search would help.
        4. Record (announced=1-doubt, correct=heuristic) for calibration.
        5. Read current rolling calibration score for telemetry.
        """
        try:
            return self._observe_inner(
                doubt_index, epistemic, web_used, kg_facts_count,
            )
        except Exception:
            log.warning("MetacognitivePhase.observe crashed", exc_info=True)
            return MetacognitiveDecision()

    def _observe_inner(
        self,
        doubt_index: float,
        epistemic: float,
        web_used: bool,
        kg_facts_count: int,
    ) -> MetacognitiveDecision:
        # V4.0.2: feed the auto-calibrator BEFORE classifying so the
        # current observation also informs its own band when post-bootstrap.
        # Pre-bootstrap, this is a no-op since the bootstrap thresholds
        # are still in effect.
        try:
            d_clamped = max(0.0, min(1.0, float(doubt_index)))
            self._auto_calibrator.observe(d_clamped)
            self._refresh_classifier_thresholds()
        except (TypeError, ValueError):
            pass
        except Exception:
            log.warning(
                "auto-calibrator update failed", exc_info=True,
            )

        label = self.classifier.classify(doubt_index, epistemic, web_used)

        # Confidence score: higher = more certain. Mirror band index.
        band_idx = CONFIDENCE_LABELS.index(label)
        # 4 bands → score = 1.0, 0.7, 0.4, 0.1
        score_map = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
        confidence_score = score_map[band_idx]

        prefix = hedge_prefix(label) if self.config.apply_hedge else ""

        # Recommend web if certainty falls at or below the threshold band
        # AND web wasn't already used.
        threshold_idx = CONFIDENCE_LABELS.index(
            self.config.recommend_web_threshold
            if self.config.recommend_web_threshold in CONFIDENCE_LABELS
            else "très_incertaine"
        )
        recommend_web = (band_idx >= threshold_idx) and (not web_used)

        # Calibration record. Without ground truth we approximate:
        # "announced" = 1 - doubt_index (the model's own self-confidence)
        # "correct"   = heuristic blend of low doubt + high epistemic +
        #               having KG facts to back the answer.
        try:
            d = max(0.0, min(1.0, float(doubt_index)))
            e = max(0.0, min(1.0, float(epistemic)))
        except (TypeError, ValueError):
            d, e = 0.5, 0.5
        announced = 1.0 - d
        kg_factor = 0.0 if kg_facts_count <= 0 else min(1.0, kg_facts_count / 5.0)
        correct = 0.5 * (1.0 - d) + 0.3 * e + 0.2 * kg_factor
        correct = max(0.0, min(1.0, correct))

        self.calibration.record(announced=announced, correct=correct)

        return MetacognitiveDecision(
            confidence_label=label,
            confidence_score=confidence_score,
            hedge_prefix=prefix,
            recommend_web=recommend_web,
            calibration_score=self.calibration.calibration_score(),
            n_calibration_entries=self.calibration.n_entries(),
        )

    def to_dict(self) -> dict:
        """Snapshot for telemetry / UI."""
        is_bootstrapping = (
            self._auto_thresholds.is_bootstrapping(self._auto_calibrator)
            if self.config.auto_calibrate else False
        )
        # Current effective thresholds — either bootstrap or empirical.
        current_low = self.classifier.config.low_doubt
        current_high = self.classifier.config.high_doubt
        current_very_high = self.classifier.config.very_high_doubt
        return {
            "config": {
                "very_high_doubt": self.config.very_high_doubt,
                "high_doubt": self.config.high_doubt,
                "low_doubt": self.config.low_doubt,
                "apply_hedge": self.config.apply_hedge,
                "auto_calibrate": self.config.auto_calibrate,
            },
            "thresholds_in_use": {
                "low_doubt": round(current_low, 4),
                "high_doubt": round(current_high, 4),
                "very_high_doubt": round(current_very_high, 4),
                "source": "bootstrap" if is_bootstrapping else "empirical",
            },
            "auto_calibrator": self._auto_calibrator.to_dict(),
            "calibration_score": self.calibration.calibration_score(),
            "n_calibration_entries": self.calibration.n_entries(),
        }
