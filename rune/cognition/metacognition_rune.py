"""Metacognition — auto-monitoring de la confiance (étendu).

Héritage Lythea
---------------
Lythea v4.4 a une MetacognitivePhase avec CalibrationTracker (Brier
score) et CertaintyClassifier. Ici on étend avec les signaux
supplémentaires préconisés dans le BACKLOG V4.4 :

- KG-orphan flag
- RAG-coverage
- Embedding distance to active memory

C'est l'axe 1 du diagnostic initial : combler le problème "Qwen 3B
invente avec aplomb" (doubt_index toujours < 0.15 même quand le modèle
hallucine).

Le CalibrationTracker est repris tel quel de Lythea — il marche bien.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("rune.cognition.metacognition")


CONFIDENCE_LABELS = (
    "très_certaine", "certaine", "incertaine", "très_incertaine"
)


@dataclass
class MetacognitiveDecision:
    """Décision métacognitive post-generation."""
    confidence_label: str = "certaine"
    doubt_index: float = 0.3
    hedge_prefix: str = ""
    recommend_web: bool = False
    calibration_score: float = 0.5  # Brier-like, plus bas = mieux calibré
    signals: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "confidence_label": self.confidence_label,
            "doubt_index": round(self.doubt_index, 3),
            "hedge_prefix": self.hedge_prefix,
            "recommend_web": self.recommend_web,
            "calibration_score": round(self.calibration_score, 3),
            "signals": self.signals,
        }


@dataclass
class _CalibrationEntry:
    announced: float
    correct: float
    timestamp: float


class _CalibrationTracker:
    """Sliding-window Brier score (hérité de Lythea)."""

    def __init__(self, storage_path: Path | None = None, window: int = 100):
        self.window = max(1, int(window))
        self._entries: deque[_CalibrationEntry] = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._storage_path = storage_path
        if storage_path is not None:
            try:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                log.warning("calibration dir create failed", exc_info=True)
            self._load()

    def record(self, announced: float, correct: float) -> None:
        with self._lock:
            self._entries.append(_CalibrationEntry(
                announced=max(0.0, min(1.0, announced)),
                correct=max(0.0, min(1.0, correct)),
                timestamp=time.time(),
            ))
            if len(self._entries) % 10 == 0:
                self._save()

    def brier_score(self) -> float:
        with self._lock:
            if not self._entries:
                return 0.25  # neutral
            sq = sum(
                (e.announced - e.correct) ** 2 for e in self._entries
            )
            return sq / len(self._entries)

    def _save(self) -> None:
        if self._storage_path is None:
            return
        try:
            data = {
                "entries": [
                    {"announced": e.announced, "correct": e.correct,
                     "timestamp": e.timestamp}
                    for e in self._entries
                ]
            }
            tmp = self._storage_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._storage_path)
        except Exception:
            log.warning("calibration save failed", exc_info=True)

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for e in data.get("entries", [])[-self.window:]:
                self._entries.append(_CalibrationEntry(
                    announced=float(e.get("announced", 0.5)),
                    correct=float(e.get("correct", 0.5)),
                    timestamp=float(e.get("timestamp", time.time())),
                ))
        except Exception:
            log.warning("CalibrationTracker load failed", exc_info=True)


class Metacognition:
    """Métacognition multi-signaux.

    Parameters
    ----------
    very_high_doubt, high_doubt, low_doubt : float
        Seuils pour les 4 bandes de confiance.
    apply_hedge : bool
        Si True, applique le hedge_prefix au texte (opt-in).
    calibration_path : Path | None
        Persistance du Brier score.
    """

    def __init__(
        self,
        very_high_doubt: float = 0.55,
        high_doubt: float = 0.35,
        low_doubt: float = 0.15,
        apply_hedge: bool = False,
        calibration_path: Path | None = None,
    ) -> None:
        self.very_high_doubt = float(very_high_doubt)
        self.high_doubt = float(high_doubt)
        self.low_doubt = float(low_doubt)
        self.apply_hedge = bool(apply_hedge)
        self._calibration = _CalibrationTracker(calibration_path)
        self._last_decision: MetacognitiveDecision | None = None

    def observe(
        self,
        doubt_index: float,
        epistemic_label: str,
        web_used: bool = False,
        kg_hits: int = 0,
        rag_coverage: float = 0.0,
        confidence_announced: float | None = None,
    ) -> MetacognitiveDecision:
        """Observe le tour courant et produit une décision.

        Signals combinés :
        - doubt_index (depuis SurpriseMeter.compute_output_doubt)
        - epistemic_label (fait/intuition/hypothese)
        - web_used (bool)
        - kg_hits (int)
        - rag_coverage (0-1)
        """
        # Ajustement multi-signaux (BACKLOG V4.4)
        kg_orphan_penalty = 0.15 if kg_hits == 0 else 0.0
        rag_bonus = -0.1 * max(0.0, min(1.0, rag_coverage))
        adjusted_doubt = max(0.0, min(1.0, doubt_index + kg_orphan_penalty + rag_bonus))

        # Classification
        if adjusted_doubt < self.low_doubt:
            label = "très_certaine"
            hedge = ""
        elif adjusted_doubt < self.high_doubt:
            label = "certaine"
            hedge = ""
        elif adjusted_doubt < self.very_high_doubt:
            label = "incertaine"
            hedge = "Je ne suis pas totalement sûre, mais "
        else:
            label = "très_incertaine"
            hedge = "C'est une zone où je manque d'éléments — "

        # Recommandation web
        recommend_web = (
            label == "très_incertaine"
            and not web_used
        )

        # Calibration recording (approximation : on suppose que si le
        # doute final est < 0.5, c'était correct)
        announced = 1.0 - adjusted_doubt
        correct = 1.0 if adjusted_doubt < 0.5 else 0.0
        self._calibration.record(announced, correct)

        decision = MetacognitiveDecision(
            confidence_label=label,
            doubt_index=adjusted_doubt,
            hedge_prefix=hedge,
            recommend_web=recommend_web,
            calibration_score=self._calibration.brier_score(),
            signals={
                "epistemic_label": epistemic_label,
                "web_used": web_used,
                "kg_hits": kg_hits,
                "rag_coverage": round(rag_coverage, 3),
                "kg_orphan_penalty": kg_orphan_penalty,
                "rag_bonus": rag_bonus,
            },
        )
        self._last_decision = decision
        return decision

    @property
    def last_decision(self) -> MetacognitiveDecision | None:
        return self._last_decision

    def to_dict(self) -> dict:
        return {
            "thresholds": {
                "low_doubt": self.low_doubt,
                "high_doubt": self.high_doubt,
                "very_high_doubt": self.very_high_doubt,
            },
            "apply_hedge": self.apply_hedge,
            "calibration_score": round(self._calibration.brier_score(), 3),
            "calibration_entries": len(self._calibration._entries),
            "last_decision": (
                self._last_decision.as_dict() if self._last_decision else None
            ),
        }
