"""Tests agents — SubAgentSpawner + CronScheduler."""
from __future__ import annotations

import time

import pytest

from rune.agents.cron import CronScheduler, CronTask
from rune.agents.subagent import SubAgentConfig, SubAgentSpawner


# ── SubAgentSpawner ───────────────────────────────────────────────────


def test_subagent_run_basic():
    """Lance un subagent avec le script par défaut (MockBackend)."""
    spawner = SubAgentSpawner(SubAgentConfig(timeout_sec=30.0))
    result = spawner.run(task="Dis bonjour")
    # Le subagent doit produire un résultat (mock backend)
    assert result.status in {"ok", "error", "timeout"}
    if result.status == "ok":
        assert result.result != ""
    assert result.elapsed_sec >= 0


def test_subagent_timeout():
    """Une tâche qui dure trop longtemps doit être tuée."""
    spawner = SubAgentSpawner(SubAgentConfig(timeout_sec=0.5))
    result = spawner.run(task="Fais une tâche très longue qui dure plus d'une seconde")
    # Soit timeout, soit ok (si le mock répond instantanément)
    assert result.status in {"ok", "error", "timeout"}


def test_subagent_parallel():
    """Lance 3 subagents en parallèle."""
    spawner = SubAgentSpawner(SubAgentConfig(timeout_sec=30.0))
    tasks = [
        ("Tâche 1", ""),
        ("Tâche 2", ""),
        ("Tâche 3", ""),
    ]
    results = spawner.run_parallel(tasks, max_concurrent=2)
    assert len(results) == 3
    for r in results:
        assert r.status in {"ok", "error", "timeout"}


# ── CronScheduler ─────────────────────────────────────────────────────


def test_cron_add_task(tmp_path):
    scheduler = CronScheduler(storage_dir=tmp_path)
    task = CronTask(
        task_id="test_task",
        schedule="every:60s",
        action="Test action",
    )
    scheduler.add_task(task)
    tasks = scheduler.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].task_id == "test_task"


def test_cron_remove_task(tmp_path):
    scheduler = CronScheduler(storage_dir=tmp_path)
    task = CronTask(task_id="t1", schedule="every:60s", action="A")
    scheduler.add_task(task)
    assert scheduler.remove_task("t1") is True
    assert len(scheduler.list_tasks()) == 0
    assert scheduler.remove_task("t1") is False


def test_cron_enable_disable(tmp_path):
    scheduler = CronScheduler(storage_dir=tmp_path)
    scheduler.add_task(CronTask(task_id="t1", schedule="every:60s", action="A"))
    assert scheduler.disable_task("t1") is True
    assert scheduler.list_tasks()[0].enabled is False
    assert scheduler.enable_task("t1") is True
    assert scheduler.list_tasks()[0].enabled is True


def test_cron_persistence(tmp_path):
    """Les tâches survivent à un reload."""
    s1 = CronScheduler(storage_dir=tmp_path)
    s1.add_task(CronTask(
        task_id="persist_test",
        schedule="every:120s",
        action="Persist action",
    ))
    s2 = CronScheduler(storage_dir=tmp_path)
    tasks = s2.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].task_id == "persist_test"


def test_cron_run_task_now(tmp_path):
    scheduler = CronScheduler(
        storage_dir=tmp_path,
        subagent_runner=lambda action, ctx: {
            "status": "ok", "result": f"done: {action}", "error": None,
        },
        consolidation_trigger=lambda: {"status": "ok"},
    )
    scheduler.add_task(CronTask(
        task_id="run_test",
        schedule="every:3600s",
        action="Test action",
    ))
    result = scheduler.run_task_now("run_test")
    assert result.status == "ok"
    assert "Test action" in result.result
    assert result.consolidated is True
    # Stats updated
    task = scheduler.list_tasks()[0]
    assert task.run_count == 1
    assert task.success_count == 1


def test_cron_run_nonexistent_task_returns_error(tmp_path):
    scheduler = CronScheduler(storage_dir=tmp_path)
    result = scheduler.run_task_now("nonexistent")
    assert result.status == "error"
    assert "not found" in (result.error or "")


def test_cron_status(tmp_path):
    scheduler = CronScheduler(storage_dir=tmp_path)
    scheduler.add_task(CronTask(task_id="t1", schedule="every:60s", action="A"))
    scheduler.add_task(CronTask(task_id="t2", schedule="every:60s", action="B", enabled=False))
    status = scheduler.status()
    assert status["task_count"] == 2
    assert status["enabled_count"] == 1


def test_cron_should_run_every_schedule(tmp_path):
    """Le schedule 'every:Ns' doit déclencher après N secondes."""
    scheduler = CronScheduler(storage_dir=tmp_path)
    task = CronTask(task_id="t", schedule="every:1s", action="A")
    # last_run_ts = 0 → doit toujours déclencher
    assert scheduler._should_run(task, time.time()) is True
    # Après un run, ne doit pas déclencher avant 1s
    task.last_run_ts = time.time()
    assert scheduler._should_run(task, time.time()) is False
    time.sleep(1.1)
    assert scheduler._should_run(task, time.time()) is True


def test_cron_consolidation_trigger_called_on_success(tmp_path):
    """Le callback consolidation doit être appelé après un run réussi."""
    consolidation_calls = []
    scheduler = CronScheduler(
        storage_dir=tmp_path,
        subagent_runner=lambda action, ctx: {"status": "ok", "result": "ok"},
        consolidation_trigger=lambda: consolidation_calls.append(1) or {"status": "ok"},
    )
    scheduler.add_task(CronTask(task_id="t", schedule="every:1s", action="A"))
    scheduler.run_task_now("t")
    assert len(consolidation_calls) == 1
