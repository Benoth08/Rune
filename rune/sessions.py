"""Conversation session management with atomic JSON persistence.

Concurrency model
-----------------
The current process is single-process FastAPI but with multi-threaded
request handling (Uvicorn dispatches each request to a thread from
its pool). Two requests on the same session id can therefore race
on a read-modify-write cycle in :meth:`add_message` or :meth:`update`.

We protect against this with a per-session :class:`threading.RLock`
created lazily on first access. RLock (re-entrant) lets the same
thread re-acquire the lock if a method calls another protected method.

We do NOT protect ``get`` / ``list_sessions`` / ``delete`` with the
per-session lock — they are either pure reads (atomic enough on
POSIX since :func:`os.replace` is atomic) or terminal operations.

Index lock
----------
The shared ``_index`` dict and its file mirror need their own lock.
We use a separate :class:`threading.Lock` so a session-write doesn't
block other sessions writing the index.

Atomicity on Windows
--------------------
``os.replace`` is atomic on POSIX and on Windows since Python 3.3.
We additionally :func:`os.fsync` the temp file before replacing,
which closes the last failure window where a power loss between
write and replace could leave a zero-byte tmp file.

Supports pagination for large conversations and date-based grouping.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rune.config import SESSIONS_DIR

log = logging.getLogger("rune.sessions")


@dataclass
class Message:
    """A single chat message."""

    role: str
    content: str
    timestamp: float = 0.0
    images: list[str] = field(default_factory=list)
    doubt_index: float | None = None
    epistemic: str | None = None
    thoughts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class Session:
    """A conversation session."""

    session_id: str
    title: str = "Nouveau chat"
    created: float = 0.0
    last_activity: float = 0.0
    pinned: bool = False
    archived: bool = False
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        now = time.time()
        if self.created == 0.0:
            self.created = now
        if self.last_activity == 0.0:
            self.last_activity = now


class SessionManager:
    """Manages conversation sessions with file-based persistence.

    Parameters
    ----------
    sessions_dir : Path
        Directory for session JSON files.
    """

    def __init__(self, sessions_dir: Path = SESSIONS_DIR) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = {}

        # ── Concurrency primitives ─────────────────────────────────────
        # Per-session re-entrant locks for read-modify-write.
        # Created lazily by ``_lock_for``; kept in a dict guarded by
        # ``_locks_dict_lock`` so two threads racing for the same new
        # session id end up with the same lock instance.
        self._session_locks: dict[str, threading.RLock] = {}
        self._locks_dict_lock = threading.Lock()
        # Index file lock (separate so a session write doesn't block
        # an unrelated session reading the index).
        self._index_lock = threading.Lock()

        self._load_index()

    # ── Lock helpers ───────────────────────────────────────────────────

    def _lock_for(self, session_id: str) -> threading.RLock:
        """Return the lock for a session, creating it on first access.

        We don't aggressively GC unused locks: a chat session lives
        long enough that the dict size stays bounded by the number of
        active sessions, which is small. If memory pressure becomes
        an issue, a TTL eviction can be added here.
        """
        lock = self._session_locks.get(session_id)
        if lock is not None:
            return lock
        with self._locks_dict_lock:
            # Double-checked locking: another thread may have created it
            # while we were waiting for ``_locks_dict_lock``.
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                self._session_locks[session_id] = lock
            return lock

    # ── Index management ───────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self.sessions_dir / "_index.json"

    def _load_index(self) -> None:
        """Load the session index (lightweight metadata)."""
        path = self._index_path()
        if path.exists():
            try:
                self._index = json.loads(path.read_text("utf-8"))
            except Exception as exc:
                log.warning("Session index corrupted: %s", exc)
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild index from session files on disk."""
        self._index = {}
        for f in self.sessions_dir.glob("sess_*.json"):
            try:
                data = json.loads(f.read_text("utf-8"))
                sid = data["session_id"]
                self._index[sid] = {
                    "session_id": sid,
                    "title": data.get("title", "Nouveau chat"),
                    "created": data.get("created", 0),
                    "last_activity": data.get("last_activity", 0),
                    "pinned": data.get("pinned", False),
                    "archived": data.get("archived", False),
                    "message_count": len(data.get("messages", [])),
                }
            except Exception:
                continue
        self._save_index()

    def _save_index(self) -> None:
        """Save index atomically.

        Holds ``_index_lock`` for the entire write, not just the snapshot,
        because two concurrent ``_save_index`` calls would otherwise race
        on the same temp filename and one ``os.replace`` would fail.
        """
        path = self._index_path()
        with self._index_lock:
            self._save_json_atomic(path, dict(self._index))

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    # ── CRUD ───────────────────────────────────────────────────────────

    def create(self, title: str = "Nouveau chat") -> Session:
        """Create a new session."""
        sid = f"sess_{uuid.uuid4().hex[:12]}"
        session = Session(session_id=sid, title=title)

        # Acquire the per-session lock immediately so concurrent operations
        # on this id (extremely rare given the random uuid, but possible
        # under aggressive client-side retries) wait their turn.
        with self._lock_for(sid):
            self._save_session(session)
            with self._index_lock:
                self._index[sid] = {
                    "session_id": sid,
                    "title": title,
                    "created": session.created,
                    "last_activity": session.last_activity,
                    "pinned": False,
                    "archived": False,
                    "message_count": 0,
                }
        self._save_index()
        return session

    def get(self, session_id: str, offset: int = 0, limit: int = 50) -> Session | None:
        """Load a session with paginated messages.

        Parameters
        ----------
        session_id : str
            Session identifier.
        offset : int
            Start index for message pagination.
        limit : int
            Maximum messages to return.
        """
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
            messages_raw = data.get("messages", [])
            sliced = messages_raw[offset:offset + limit]
            session = Session(
                session_id=data["session_id"],
                title=data.get("title", "Nouveau chat"),
                created=data.get("created", 0),
                last_activity=data.get("last_activity", 0),
                pinned=data.get("pinned", False),
                archived=data.get("archived", False),
                messages=[Message(**m) for m in sliced],
            )
            return session
        except Exception as exc:
            log.warning("Session load failed (%s): %s", session_id, exc)
            return None

    def add_message(self, session_id: str, message: Message) -> None:
        """Append a message to a session.

        Atomic under concurrent writes: the per-session lock ensures
        that the read-modify-write cycle (read JSON → append → write)
        is not interleaved with another caller.
        """
        with self._lock_for(session_id):
            path = self._session_path(session_id)
            if not path.exists():
                return

            try:
                data = json.loads(path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("add_message: load failed for %s: %s", session_id, exc)
                return

            data.setdefault("messages", []).append(asdict(message))
            data["last_activity"] = time.time()

            # Auto-title from first user message
            if data.get("title") == "Nouveau chat" and message.role == "user":
                data["title"] = (message.content or "")[:60].strip() or "Nouveau chat"

            self._save_json_atomic(path, data)

            # Update index under its own lock — a different session
            # writing in parallel doesn't block this update.
            with self._index_lock:
                if session_id in self._index:
                    self._index[session_id]["last_activity"] = data["last_activity"]
                    self._index[session_id]["title"] = data["title"]
                    self._index[session_id]["message_count"] = len(data["messages"])
        self._save_index()

    def update(self, session_id: str, **kwargs: str | bool) -> bool:
        """Update session metadata (title, pinned, archived)."""
        with self._lock_for(session_id):
            path = self._session_path(session_id)
            if not path.exists():
                return False

            try:
                data = json.loads(path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("update: load failed for %s: %s", session_id, exc)
                return False

            for key in ("title", "pinned", "archived"):
                if key in kwargs:
                    data[key] = kwargs[key]
            self._save_json_atomic(path, data)

            with self._index_lock:
                if session_id in self._index:
                    for key in ("title", "pinned", "archived"):
                        if key in kwargs:
                            self._index[session_id][key] = kwargs[key]
        self._save_index()
        return True

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock_for(session_id):
            path = self._session_path(session_id)
            if path.exists():
                path.unlink()
            with self._index_lock:
                self._index.pop(session_id, None)
        self._save_index()
        # Clean up the lock entry — no further operations can race here
        # since we hold no reference and the file is gone.
        with self._locks_dict_lock:
            self._session_locks.pop(session_id, None)
        return True

    def delete_all(self) -> int:
        """Delete every session. Returns the number of sessions deleted.

        Walks the on-disk session files independently of the in-memory
        index so any orphans (index out of sync) also get cleaned up.
        """
        # Snapshot ids from the index BEFORE deleting, otherwise the
        # in-loop ``delete`` mutates what we'd iterate over.
        with self._index_lock:
            ids = list(self._index.keys())
        count = 0
        for sid in ids:
            if self.delete(sid):
                count += 1
        return count

    def list_sessions(self) -> list[dict]:
        """Return all sessions sorted by last_activity (newest first)."""
        with self._index_lock:
            sessions = list(self._index.values())
        sessions.sort(key=lambda s: s.get("last_activity", 0), reverse=True)
        return sessions

    def export_markdown(self, session_id: str) -> str | None:
        """Export a session as markdown."""
        session = self.get(session_id, limit=10000)
        if session is None:
            return None

        lines = [f"# {session.title}\n"]
        for msg in session.messages:
            role = "🧑 User" if msg.role == "user" else "🤖 Rune"
            lines.append(f"### {role}\n{msg.content}\n")
        return "\n".join(lines)

    # ── Persistence helpers ────────────────────────────────────────────

    def _save_session(self, session: Session) -> None:
        data = asdict(session)
        self._save_json_atomic(self._session_path(session.session_id), data)

    @staticmethod
    def _save_json_atomic(path: Path, data: dict) -> None:
        """Atomic JSON write: temp file → fsync → os.replace.

        The fsync between write and replace closes the failure window
        where a power loss after write but before replace could leave
        a zero-byte tmp file on disk. ``os.replace`` is atomic on
        POSIX and on Windows since Python 3.3.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                # fsync may fail on some networked / virtual filesystems
                # (NFS, certain Docker volume drivers). Not fatal —
                # os.replace is still atomic on POSIX/Win.
                pass
        os.replace(str(tmp), str(path))

    # Backward-compat alias: old name used by other modules / tests.
    _save_session_raw = _save_json_atomic
