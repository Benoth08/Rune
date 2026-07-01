"""Tests for the secure git_sync module."""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest

from rune.git_sync import GitSync, _scrub, _temporary_askpass


def test_scrub_removes_token():
    text = "fatal: error pushing https://abcd1234@github.com/me/repo.git"
    cleaned = _scrub(text, "abcd1234")
    assert "abcd1234" not in cleaned
    assert "[REDACTED]" in cleaned


def test_scrub_no_op_when_no_token():
    assert _scrub("hello", None) == "hello"
    assert _scrub("", "tok") == ""


def test_temporary_askpass_creates_executable_helper():
    with _temporary_askpass("MY_TOKEN") as path:
        p = Path(path)
        assert p.exists()
        # Mode 0700: owner rwx, no group/other access
        st = p.stat()
        assert st.st_mode & stat.S_IRUSR
        assert st.st_mode & stat.S_IXUSR
        assert not (st.st_mode & stat.S_IRGRP)
        assert not (st.st_mode & stat.S_IROTH)
        # Helper prints exactly the token, no trailing newline
        content = p.read_text()
        assert "MY_TOKEN" in content
        assert "printf" in content  # not echo (which adds \n)


def test_temporary_askpass_cleans_up():
    with _temporary_askpass("X") as path:
        assert Path(path).exists()
        captured = path
    assert not Path(captured).exists()
    # parent dir also gone
    assert not Path(captured).parent.exists()


def test_configure_rejects_empty_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        gs = GitSync(Path(tmp))
        assert not gs.configure("", "user", "repo")
        assert not gs.configure("tok", "", "repo")
        assert not gs.configure("tok", "user", "")
        assert not gs._configured


def test_configure_keeps_token_in_memory_only(monkeypatch):
    """The remote URL must NOT contain the token (only in memory)."""
    with tempfile.TemporaryDirectory() as tmp:
        gs = GitSync(Path(tmp))

        run_calls: list[tuple] = []
        def fake_run(*args, check=True, env=None):
            run_calls.append(args)
            class R:
                returncode = 0
                stdout = ""
                stderr = ""
            return R()

        monkeypatch.setattr(gs, "_run", fake_run)
        ok = gs.configure("SECRET_TOKEN", "alice", "myrepo")
        assert ok

        # No call should embed the token in argv
        for call in run_calls:
            joined = " ".join(call)
            assert "SECRET_TOKEN" not in joined

        # The remote URL must be clean
        remote_calls = [c for c in run_calls if "remote" in c and "add" in c]
        assert remote_calls
        for c in remote_calls:
            assert "SECRET_TOKEN" not in str(c)
            assert "https://github.com/alice/myrepo.git" in str(c)


def test_push_aborted_without_token(monkeypatch):
    """Even after configure, if the in-memory token is cleared, push aborts."""
    with tempfile.TemporaryDirectory() as tmp:
        gs = GitSync(Path(tmp))
        monkeypatch.setattr(gs, "_run", lambda *a, **kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
        gs.configure("tok", "u", "r")
        gs._token = None  # simulate post-config clearance
        # _push should not raise
        gs._lock.acquire()  # block
        gs._lock.release()
        gs._push()  # silently returns
