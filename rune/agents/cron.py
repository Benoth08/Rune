"""CronScheduler — tâches de fond avec consolidation post-run.

Inspirations
------------
- **OpenClaw** : cron de fond pour scraping/audit/routines.
- **Rune** : automatisation de fond autonome.
- **Lythea** : microsleep/deep sleep pour consolidation.

L'innovation différenciante : après chaque exécution de tâche cron, on
déclenche un microsleep pour consolider les patterns appris. Rune est
amnésique entre les runs ; Rune apprend.

Format cron
-----------
On supporte deux types de schedules :

1. **APScheduler cron** (unix-like) : ``*/30 * * * *`` (toutes les 30 min)
2. **Interval** : ``every:300s`` (toutes les 300 secondes)

Les tâches sont persistées dans ``data/cron/tasks.json``. Au boot, le
scheduler recharge les tâches et reprend.

Une tâche cron typique :

    {
        "task_id": "cron_veille_tech",
        "schedule": "0 9 * * *",  # tous les jours à 9h
        "action": "Recherche les dernières news sur la crypto post-quantique",
        "use_subagent": true,
        "enabled": true,
        "last_run_ts": 0,
        "last_result": null,
        "run_count": 0,
        "success_count": 0
    }
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("rune.agents.cron")


@dataclass
class CronTask:
    """Une tâche cron."""
    task_id: str
    schedule: str  # unix cron ou "every:Ns"
    action: str  # description de la mission
    use_subagent: bool = True
    enabled: bool = True
    last_run_ts: float = 0.0
    last_result: dict = field(default_factory=dict)
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CronRunResult:
    """Résultat d'une exécution cron."""
    task_id: str
    status: str = "ok"  # ok | error | skipped
    result: str = ""
    elapsed_sec: float = 0.0
    consolidated: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class CronScheduler:
    """Scheduler de tâches de fond.

    Parameters
    ----------
    storage_dir : Path
        Répertoire de persistance des tâches.
    subagent_runner : Callable[[str, str], dict] | None
        Fonction qui exécute une tâche via subagent. Signature :
        (task: str, context: str) -> dict (status, result, error).
        Si None, on utilise un runner local simple (MockBackend).
    consolidation_trigger : Callable[[], dict] | None
        Callback appelé après chaque exécution réussie pour déclencher
        un microsleep. C'est le hook de consolidation post-run.
    """

    def __init__(
        self,
        storage_dir: Path | str = "data/cron",
        subagent_runner: Callable[[str, str], dict] | None = None,
        consolidation_trigger: Callable[[], dict] | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.storage_dir / "tasks.json"
        self.tasks: dict[str, CronTask] = {}
        self._subagent_runner = subagent_runner
        self._consolidation_trigger = consolidation_trigger
        self._scheduler_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._load()

    # ── API publique ──────────────────────────────────────────────────

    def add_task(self, task: CronTask) -> CronTask:
        """Ajoute ou met à jour une tâche."""
        with self._lock:
            self.tasks[task.task_id] = task
            self._save()
        log.info("Cron task added: %s (%s)", task.task_id, task.schedule)
        return task

    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self.tasks:
                return False
            del self.tasks[task_id]
            self._save()
        return True

    def enable_task(self, task_id: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return False
            task.enabled = True
            self._save()
        return True

    def disable_task(self, task_id: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return False
            task.enabled = False
            self._save()
        return True

    def list_tasks(self) -> list[CronTask]:
        return list(self.tasks.values())

    def run_task_now(self, task_id: str) -> CronRunResult:
        """Exécute une tâche immédiatement (sans attendre le schedule)."""
        with self._lock:
            task = self.tasks.get(task_id)
        if task is None:
            return CronRunResult(
                task_id=task_id, status="error", error="task not found"
            )
        return self._execute(task)

    def start(self) -> None:
        """Démarre le scheduler en arrière-plan."""
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            log.warning("Cron scheduler already running")
            return
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="rune-cron"
        )
        self._scheduler_thread.start()
        log.info("Cron scheduler started")

    def stop(self, timeout: float = 5.0) -> None:
        """Arrête le scheduler."""
        self._stop_event.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=timeout)
            self._scheduler_thread = None
        log.info("Cron scheduler stopped")

    def status(self) -> dict:
        return {
            "running": self._scheduler_thread is not None and self._scheduler_thread.is_alive(),
            "task_count": len(self.tasks),
            "enabled_count": sum(1 for t in self.tasks.values() if t.enabled),
            "tasks": [t.as_dict() for t in self.tasks.values()],
        }

    # ── Internes ──────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Boucle principale — vérifie les schedules toutes les 30s."""
        while not self._stop_event.is_set():
            try:
                now = time.time()
                with self._lock:
                    tasks_to_run = [
                        t for t in self.tasks.values()
                        if t.enabled and self._should_run(t, now)
                    ]
                for task in tasks_to_run:
                    self._execute(task)
            except Exception:
                log.exception("Cron loop error")
            # Attend 30s ou stop
            self._stop_event.wait(timeout=30.0)

    def _should_run(self, task: CronTask, now: float) -> bool:
        """Vérifie si une tâche doit tourner maintenant."""
        if task.schedule.startswith("every:"):
            try:
                # Format: "every:Ns" (seconds), "every:Nm" (minutes),
                # "every:Nh" (hours), ou "every:N" (seconds par défaut)
                raw = task.schedule.split(":", 1)[1].strip()
                if raw.endswith("m"):
                    interval_sec = float(raw[:-1]) * 60
                elif raw.endswith("h"):
                    interval_sec = float(raw[:-1]) * 3600
                elif raw.endswith("s"):
                    interval_sec = float(raw[:-1])
                else:
                    interval_sec = float(raw)
            except (ValueError, IndexError):
                return False
            return (now - task.last_run_ts) >= interval_sec
        # Cron unix — implémentation simplifiée
        # TODO: brancher APScheduler proper pour parsing cron complet
        # Pour l'instant, on supporte juste "*/N * * * *" (toutes les N minutes)
        try:
            parts = task.schedule.split()
            if len(parts) != 5:
                return False
            minute, hour, dom, month, dow = parts
            if minute.startswith("*/"):
                interval_min = int(minute[2:])
                minutes_since = (now - task.last_run_ts) / 60
                return minutes_since >= interval_min
            # Pour les autres patterns, fallback 1h
            return (now - task.last_run_ts) >= 3600
        except Exception:
            return False

    def _execute(self, task: CronTask) -> CronRunResult:
        """Exécute une tâche."""
        start = time.time()
        log.info("Executing cron task: %s", task.task_id)
        result = CronRunResult(task_id=task.task_id)

        try:
            if self._subagent_runner is not None:
                output = self._subagent_runner(task.action, "")
            else:
                # Runner local simple (mock)
                output = self._default_runner(task.action)

            result.status = output.get("status", "ok")
            result.result = output.get("result", "")
            result.error = output.get("error")

            # Consolidation post-run (innovation Lythea)
            if result.status == "ok" and self._consolidation_trigger is not None:
                try:
                    consolidation = self._consolidation_trigger()
                    result.consolidated = consolidation.get("status") == "ok"
                except Exception:
                    log.exception("Consolidation trigger failed")

        except Exception as exc:
            log.exception("Cron task failed")
            result.status = "error"
            result.error = str(exc)

        result.elapsed_sec = time.time() - start

        # Update task stats
        with self._lock:
            task.last_run_ts = time.time()
            task.run_count += 1
            if result.status == "ok":
                task.success_count += 1
            else:
                task.failure_count += 1
            task.last_result = result.as_dict()
            self._save()

        log.info(
            "Cron task %s done in %.2fs (status=%s)",
            task.task_id, result.elapsed_sec, result.status,
        )
        return result

    def _default_runner(self, action: str) -> dict:
        """Runner local simple — utilisé si subagent_runner est None."""
        try:
            from ..perf.backend import get_backend
            from ..perf.backend import GenerationConfig
            backend = get_backend({"backend": "mock"})
            messages = [
                {"role": "system", "content": (
                    "Tu exécutes une tâche automatisée en arrière-plan. "
                    "Sois concis et factuel."
                )},
                {"role": "user", "content": action},
            ]
            result = backend.generate(messages, GenerationConfig(max_new_tokens=256))
            return {
                "status": "ok" if result.finish_reason != "error" else "error",
                "result": result.text,
                "error": result.meta.get("error"),
            }
        except Exception as exc:
            return {"status": "error", "result": "", "error": str(exc)}

    def _save(self) -> None:
        data = {
            "version": 1,
            "tasks": [t.as_dict() for t in self.tasks.values()],
        }
        tmp = self._index_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._index_path)

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with self._index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("tasks", []):
                try:
                    task = CronTask(**{
                        k: v for k, v in entry.items()
                        if k in CronTask.__dataclass_fields__  # type: ignore[attr-defined]
                    })
                    self.tasks[task.task_id] = task
                except Exception as exc:
                    log.warning("Failed to load cron task: %s", exc)
        except Exception:
            log.exception("Failed to load cron tasks")
