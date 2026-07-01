"""ConsolidationScheduler — microsleep + deep sleep différés.

Héritage Lythea
---------------
Lythea a ConsolidationPhase (489 lignes) + MicrosleepManager qui font
ripples + replay + compression Chroma + git sync. Ici on fournit un
scheduler léger qui déclenche la consolidation de façon opportuniste :

- Microsleep toutes les N exchanges (défaut 5)
- Microsleep après N secondes d'inactivité (défaut 300s)
- Deep sleep sur appel explicite (CLI / API)

Le scheduler ne fait pas lui-même la consolidation (ça demande un vrai
SDM/MHN/Chroma). Il délègue à des callbacks configurables. Si les
callbacks sont None, il logge juste — utile pour les tests et le
bootstrap.

Couplage cron (à venir)
-----------------------
Quand une tâche cron se termine, on appelle ``trigger_post_cron_microsleep``
qui force un microsleep pour consolider les patterns appris pendant la
tâche. C'est ce qui rend Rune meilleur qu'Rune (qui n'a pas
de consolidation entre les runs cron).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("rune.cognition.consolidation")


@dataclass
class ConsolidationConfig:
    """Config du scheduler."""
    microsleep_interval: int = 5  # toutes les N exchanges
    microsleep_inactivity_sec: float = 300.0  # 5 min
    deep_sleep_prune_threshold: float = 0.5
    # Callbacks (None = no-op, juste log)
    on_microsleep: Callable[[], dict] | None = None
    on_deep_sleep: Callable[[], dict] | None = None


@dataclass
class ConsolidationStats:
    """Stats d'exécution."""
    microsleep_count: int = 0
    deep_sleep_count: int = 0
    last_microsleep_ts: float = 0.0
    last_deep_sleep_ts: float = 0.0
    last_microsleep_result: dict = field(default_factory=dict)
    last_deep_sleep_result: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "microsleep_count": self.microsleep_count,
            "deep_sleep_count": self.deep_sleep_count,
            "last_microsleep_ts": self.last_microsleep_ts,
            "last_deep_sleep_ts": self.last_deep_sleep_ts,
            "seconds_since_last_microsleep": (
                time.time() - self.last_microsleep_ts
                if self.last_microsleep_ts > 0 else -1
            ),
            "last_microsleep_result": self.last_microsleep_result,
            "last_deep_sleep_result": self.last_deep_sleep_result,
        }


class ConsolidationScheduler:
    """Scheduler microsleep + deep sleep.

    Thread-safe. Le microsleep opportuniste est déclenché par
    ``maybe_microsleep()`` appelé après chaque tour cognitif. Le deep
    sleep est explicite (CLI/API).

    Parameters
    ----------
    config : ConsolidationConfig
        Seuils et callbacks.
    exchange_counter : Callable[[], int]
        Fonction qui retourne le nombre d'échanges courant (pour
        déclencher le microsleep opportuniste).
    """

    def __init__(
        self,
        config: ConsolidationConfig,
        exchange_counter: Callable[[], int] | None = None,
    ) -> None:
        self.config = config
        self.exchange_counter = exchange_counter or (lambda: 0)
        self.stats = ConsolidationStats()
        self._lock = threading.Lock()
        self._last_check_ts: float = time.time()
        self._last_seen_exchange: int = 0

    def maybe_microsleep(self, force: bool = False) -> dict | None:
        """Déclenche un microsleep si les conditions sont remplies.

        Conditions (au moins une) :
        - ``force=True`` (appel explicite)
        - N exchanges depuis le dernier microsleep
        - Inactivité ≥ microsleep_inactivity_sec

        Retourne le résultat du callback (ou None si pas déclenché).
        """
        with self._lock:
            now = time.time()
            exchanges = self.exchange_counter()
            exchanges_since = exchanges - self._last_seen_exchange
            idle_sec = now - self._last_check_ts

            should_trigger = force or (
                exchanges_since >= self.config.microsleep_interval
                and exchanges_since > 0
            ) or (
                idle_sec >= self.config.microsleep_inactivity_sec
                and exchanges_since > 0
            )

            if not should_trigger:
                return None

            self._last_seen_exchange = exchanges
            self._last_check_ts = now

        # Exécution hors lock (le callback peut être long)
        return self._run_microsleep()

    def trigger_post_cron_microsleep(self) -> dict:
        """Force un microsleep après une tâche cron (toujours).

        C'est l'innovation différenciante : on consolide systématiquement
        après chaque tâche de fond pour ancrer les patterns appris.
        """
        log.info("Triggering post-cron microsleep")
        return self._run_microsleep()

    def deep_sleep(self) -> dict:
        """Déclenche un deep sleep (explicite)."""
        with self._lock:
            self.stats.deep_sleep_count += 1
            self.stats.last_deep_sleep_ts = time.time()

        if self.config.on_deep_sleep is None:
            result = {"status": "skipped", "reason": "no callback"}
        else:
            try:
                result = self.config.on_deep_sleep() or {}
            except Exception as exc:
                log.exception("Deep sleep callback failed")
                result = {"status": "error", "error": str(exc)}

        with self._lock:
            self.stats.last_deep_sleep_result = result
        return result

    def status(self) -> dict:
        return {
            "config": {
                "microsleep_interval": self.config.microsleep_interval,
                "microsleep_inactivity_sec": self.config.microsleep_inactivity_sec,
            },
            "stats": self.stats.as_dict(),
        }

    # ── Internes ──────────────────────────────────────────────────────

    def _run_microsleep(self) -> dict:
        """Exécute un microsleep via le callback configuré."""
        start = time.time()
        with self._lock:
            self.stats.microsleep_count += 1
            self.stats.last_microsleep_ts = time.time()

        if self.config.on_microsleep is None:
            result = {"status": "skipped", "reason": "no callback"}
        else:
            try:
                result = self.config.on_microsleep() or {}
            except Exception as exc:
                log.exception("Microsleep callback failed")
                result = {"status": "error", "error": str(exc)}

        result["elapsed_sec"] = time.time() - start
        with self._lock:
            self.stats.last_microsleep_result = result
        log.info("Microsleep done in %.2fs: %s", result["elapsed_sec"], result.get("status"))
        return result
