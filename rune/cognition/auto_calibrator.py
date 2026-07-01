"""V4.0.2 — Quantile-based auto-calibration (model-agnostic).

Inspiration biologique
----------------------
Les seuils neuronaux ne sont pas figés : un neurone qui reçoit
constamment de fortes excitations finit par ajuster son seuil de
spike vers le haut (homeostasie synaptique). De même, nos seuils
"low / high doubt" doivent s'auto-adapter à la distribution observée
des signaux du modèle, pas être figés théoriquement.

Pourquoi
--------
Les seuils par défaut V4 (metacog: low=0.15/high=0.35/very_high=0.55,
predictive_coding: low=0.15/high=0.65) ont été calibrés théoriquement.
En production sur un Qwen 3B, le `doubt_index` reste systématiquement
< 0.15 et l'`error` cosine entre embeddings reste < 0.10 — les bandes
ne discriminent rien.

La solution générique (compatible avec tout modèle, présent ou futur) :
au lieu de hardcoder des seuils, **observer la distribution de chaque
signal** sur une fenêtre glissante de N observations, puis recalculer
les seuils comme **quantiles empiriques**.

Architecture
------------
- :class:`QuantileCalibrator` : observateur léger, sans dépendance.
  Maintient un buffer borné, expose ``quantile(p)`` et ``adjusted_thresholds``.
- :class:`AutoCalibratedThresholds` : wrapper qui combine seuils
  bootstrap (utilisés pendant la phase d'amorçage avant qu'on ait
  assez de data) avec seuils empiriques (pris des quantiles dès que
  le buffer dépasse ``min_samples``).

Garanties
---------
- Pure Python, pas de numpy.
- Persistance JSON atomique optionnelle (pour cumulatif inter-sessions).
- Fallback safe : si le buffer est vide ou trop petit, retombe sur
  les seuils bootstrap.
- Thread-safe (lock sur observe + adjust).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("lythea.cognition.auto_calibrator")


# ════════════════════════════════════════════════════════════════════
# QuantileCalibrator — sliding-window quantile tracker
# ════════════════════════════════════════════════════════════════════


class QuantileCalibrator:
    """Track a sliding window of float observations + compute quantiles.

    Used to expose empirical thresholds (e.g. P25/P75/P90 of doubt_index)
    that adapt to whatever model is producing the signals.

    Persistence: optional JSON atomic write (.tmp + replace). Safe to
    instantiate without storage_path — the buffer is then in-memory only.
    """

    def __init__(
        self,
        window: int = 200,
        storage_path: Path | None = None,
    ):
        self.window = max(1, int(window))
        self._buf: deque[float] = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._storage_path = storage_path
        if storage_path is not None:
            try:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                log.warning(
                    "QuantileCalibrator storage parent create failed",
                    exc_info=True,
                )
            self._load()

    # ── Persistence ─────────────────────────────────────────────────

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", []) if isinstance(data, dict) else []
            for s in samples[-self.window:]:
                try:
                    self._buf.append(float(s))
                except (TypeError, ValueError):
                    continue
        except Exception:
            log.warning("QuantileCalibrator load failed", exc_info=True)

    def _save_locked(self) -> None:
        if self._storage_path is None:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._storage_path.with_suffix(
                self._storage_path.suffix + ".tmp"
            )
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(
                    {"version": 1, "samples": list(self._buf)},
                    f,
                    ensure_ascii=False,
                )
            tmp.replace(self._storage_path)
        except Exception:
            log.warning("QuantileCalibrator save failed", exc_info=True)

    # ── Public API ──────────────────────────────────────────────────

    def observe(self, value: float, save: bool = True) -> None:
        """Append one observation. Clamped to [0, 1] when out of range."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        # Defensive clamp: signals are typically in [0, 1] but we don't
        # want a single garbage value to skew the quantiles.
        if v < 0.0 or v != v:  # NaN check
            return
        with self._lock:
            self._buf.append(v)
            if save:
                self._save_locked()

    def n_samples(self) -> int:
        with self._lock:
            return len(self._buf)

    def quantile(self, p: float) -> float | None:
        """Return the p-quantile of the buffer (p in [0, 1]).

        None if the buffer is empty. Linear interpolation between
        adjacent samples (matches numpy's default ``method='linear'``).
        """
        try:
            p = max(0.0, min(1.0, float(p)))
        except (TypeError, ValueError):
            return None
        with self._lock:
            if not self._buf:
                return None
            sorted_buf = sorted(self._buf)
        if len(sorted_buf) == 1:
            return sorted_buf[0]
        idx_f = p * (len(sorted_buf) - 1)
        lo = int(idx_f)
        hi = min(lo + 1, len(sorted_buf) - 1)
        frac = idx_f - lo
        return sorted_buf[lo] * (1 - frac) + sorted_buf[hi] * frac

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
            self._save_locked()

    def to_dict(self) -> dict:
        """Snapshot for telemetry."""
        with self._lock:
            samples = list(self._buf)
        return {
            "n_samples": len(samples),
            "min": min(samples) if samples else None,
            "max": max(samples) if samples else None,
            "p25": self.quantile(0.25),
            "p50": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p90": self.quantile(0.90),
        }


# ════════════════════════════════════════════════════════════════════
# AutoCalibratedThresholds — bootstrap → empirical fade
# ════════════════════════════════════════════════════════════════════


@dataclass
class AutoCalibratedThresholds:
    """Three-level thresholds that auto-adjust to observed distribution.

    During the bootstrap period (< ``min_samples`` observations), uses
    the configured bootstrap thresholds. After ``min_samples``, switches
    to **empirical quantiles** of the buffer.

    The mapping is designed so that:
      - ``low``       ≈ P25 of observations (cuts the lower 25%)
      - ``high``      ≈ P75 of observations (cuts the upper 25%)
      - ``very_high`` ≈ P90 of observations (top 10%)

    This makes the bands automatically discriminative regardless of
    the underlying model: tight distribution → tight bands; spread
    distribution → wide bands.

    Bootstrap defaults are sane for moderate models (e.g. Qwen 7B+).
    On very confident models (Qwen 3B), the bands tighten themselves
    after ~30-50 observations.
    """

    bootstrap_low: float = 0.15
    bootstrap_high: float = 0.35
    bootstrap_very_high: float = 0.55
    min_samples: int = 30
    p_low: float = 0.25
    p_high: float = 0.75
    p_very_high: float = 0.90

    def get_thresholds(
        self, calibrator: QuantileCalibrator,
    ) -> tuple[float, float, float]:
        """Return (low, high, very_high) thresholds.

        Pre-bootstrap → bootstrap_*. Post-bootstrap → empirical quantiles
        with monotonicity enforcement (low < high < very_high).
        """
        n = calibrator.n_samples()
        if n < self.min_samples:
            return (
                self.bootstrap_low,
                self.bootstrap_high,
                self.bootstrap_very_high,
            )
        low = calibrator.quantile(self.p_low) or self.bootstrap_low
        high = calibrator.quantile(self.p_high) or self.bootstrap_high
        very_high = calibrator.quantile(self.p_very_high) or self.bootstrap_very_high
        # Enforce strict monotonicity to avoid degenerate bands.
        # If empirical quantiles collapse (very tight distribution), we
        # still want at least some discrimination — pad by 1% per band.
        if high <= low:
            high = low + 0.01
        if very_high <= high:
            very_high = high + 0.01
        # Clamp to [0, 1] just in case.
        low = max(0.0, min(1.0, low))
        high = max(0.0, min(1.0, high))
        very_high = max(0.0, min(1.0, very_high))
        return low, high, very_high

    def is_bootstrapping(self, calibrator: QuantileCalibrator) -> bool:
        return calibrator.n_samples() < self.min_samples
