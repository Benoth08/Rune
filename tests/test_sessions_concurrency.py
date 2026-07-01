"""Concurrency tests for SessionManager — no message must be lost."""
from __future__ import annotations

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from rune.sessions import Message, SessionManager


# ── Per-session lock identity ─────────────────────────────────────────

def test_lock_for_returns_same_lock_for_same_id():
    """Two calls for the same session id return the same lock instance."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        l1 = sm._lock_for("sess_abc")
        l2 = sm._lock_for("sess_abc")
        assert l1 is l2


def test_lock_for_returns_different_locks_for_different_ids():
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        l1 = sm._lock_for("sess_a")
        l2 = sm._lock_for("sess_b")
        assert l1 is not l2


def test_lock_for_thread_safe_under_race():
    """100 threads racing to get the lock for the same id must agree."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        results: list[threading.RLock] = []
        barrier = threading.Barrier(100)

        def worker():
            barrier.wait()  # release everyone simultaneously
            results.append(sm._lock_for("sess_race"))

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()

        first = results[0]
        assert all(r is first for r in results)


# ── No message loss under concurrent add_message ──────────────────────

def test_concurrent_add_message_loses_nothing():
    """100 concurrent add_message on the same session — all must persist."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        session = sm.create()
        sid = session.session_id

        N = 100
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [
                pool.submit(
                    sm.add_message,
                    sid,
                    Message(role="user", content=f"msg_{i:03d}"),
                )
                for i in range(N)
            ]
            for f in as_completed(futures):
                f.result()  # raise if any thread failed

        loaded = sm.get(sid, limit=N + 10)
        assert loaded is not None
        assert len(loaded.messages) == N

        # Each message content must appear exactly once.
        contents = sorted(m.content for m in loaded.messages)
        expected = sorted(f"msg_{i:03d}" for i in range(N))
        assert contents == expected


def test_concurrent_add_messages_across_different_sessions():
    """Sessions don't block each other — independent locks."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        ids = [sm.create().session_id for _ in range(5)]

        # 50 messages × 5 sessions = 250 writes in parallel
        def add(sid_idx):
            sid = ids[sid_idx % 5]
            sm.add_message(sid, Message(role="user", content=f"x_{sid_idx}"))

        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = [pool.submit(add, i) for i in range(250)]
            for f in as_completed(futures):
                f.result()

        # Each session got exactly 50 messages
        for sid in ids:
            loaded = sm.get(sid, limit=200)
            assert len(loaded.messages) == 50


def test_concurrent_update_and_add_message():
    """Mixed update + add_message on same session must converge correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        sid = sm.create().session_id

        def adder(i):
            sm.add_message(sid, Message(role="user", content=f"m{i}"))

        def updater(i):
            sm.update(sid, title=f"t{i}", pinned=(i % 2 == 0))

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = []
            for i in range(50):
                futures.append(pool.submit(adder, i))
                futures.append(pool.submit(updater, i))
            for f in as_completed(futures):
                f.result()

        loaded = sm.get(sid, limit=200)
        assert len(loaded.messages) == 50
        # Title is whichever updater finished last; not deterministic
        # but it must be a valid t<n>.
        assert loaded.title.startswith("t")


# ── Atomic write resilience ───────────────────────────────────────────

def test_save_json_atomic_creates_no_tmp_residue():
    """After a successful save, no .tmp file remains."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        sid = sm.create().session_id
        sm.add_message(sid, Message(role="user", content="hello"))

        # Look for any leftover *.tmp
        tmp_files = list(Path(tmp).glob("**/*.tmp"))
        assert tmp_files == []


def test_save_json_atomic_overwrites_existing():
    """Updating a session must replace the file content, not append."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        sid = sm.create().session_id

        sm.add_message(sid, Message(role="user", content="A"))
        size_after_one = (Path(tmp) / f"{sid}.json").stat().st_size

        sm.add_message(sid, Message(role="user", content="B"))
        size_after_two = (Path(tmp) / f"{sid}.json").stat().st_size

        # Size grew (added a message) but the file wasn't double-written
        assert size_after_two > size_after_one
        # Sanity: still valid JSON
        import json
        json.loads((Path(tmp) / f"{sid}.json").read_text("utf-8"))


# ── Delete during concurrent activity ─────────────────────────────────

def test_delete_after_writes_cleans_up_lock():
    """Deleting a session also removes its entry from the lock dict."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        sid = sm.create().session_id
        sm.add_message(sid, Message(role="user", content="x"))
        # Force lock creation
        _ = sm._lock_for(sid)
        assert sid in sm._session_locks

        sm.delete(sid)
        assert sid not in sm._session_locks
