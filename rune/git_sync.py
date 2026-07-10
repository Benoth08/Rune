"""Optional GitHub synchronization for memory backup.

Security model
--------------
Previous versions embedded the GitHub token directly in the remote URL
(``https://<token>@github.com/user/repo.git``). This was loggable in
case of error and would leak via ``git remote -v`` to anyone with
filesystem access.

Current implementation:
- The remote URL is plain (no credentials inline).
- The token is passed at push time via the ``GIT_ASKPASS`` environment
  variable, which Git uses to obtain credentials. We point ASKPASS at
  a tiny helper script that prints the token — script is created in a
  private temp directory with mode 0700 and deleted after the push.
- The token never lands in any persistent file or in process argv.
"""
from __future__ import annotations

import logging
import os
import stat
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("rune.git_sync")


@contextmanager
def _temporary_askpass(token: str):
    """Yield the path to a one-shot GIT_ASKPASS helper.

    The helper is a tiny shell script that ``echo``-es the token. We
    create it in a private tempdir with mode 0700, hand the path to
    Git via env, and unconditionally delete it on exit.

    A POSIX-only path is used because Lythéa's deploy targets (RunPod,
    Colab, Kaggle, Linux) are all POSIX. Windows would need a .bat
    shim — out of scope.
    """
    tmpdir = tempfile.mkdtemp(prefix="lythea_git_")
    helper_path = Path(tmpdir) / "askpass.sh"
    try:
        # Write the helper. Use printf to avoid trailing newlines that
        # would mangle the token when Git reads stdout.
        helper_path.write_text(
            f"#!/bin/sh\nprintf '%s' '{token}'\n",
            encoding="utf-8",
        )
        os.chmod(helper_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        # Lock down the directory itself too (no read for group/other).
        os.chmod(tmpdir, stat.S_IRWXU)
        yield str(helper_path)
    finally:
        try:
            helper_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass


class GitSync:
    """Asynchronous git push for memory backup.

    Parameters
    ----------
    repo_dir : Path
        Local directory to sync.
    """

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir
        self._user: str | None = None
        self._repo: str | None = None
        self._token: str | None = None
        self._configured = False
        self._lock = threading.Lock()

    def configure(self, token: str, user: str, repo: str) -> bool:
        """Configure git remote without leaking the token.

        We store the token in memory (this process only) and write the
        remote URL without credentials. At push time, GIT_ASKPASS feeds
        the token to Git over stdin via the temporary helper script.
        """
        if not token or not user or not repo:
            log.warning("Git configure: missing token/user/repo")
            return False

        # Stash credentials in memory only — never on disk.
        self._user = user
        self._repo = repo
        self._token = token
        clean_url = f"https://github.com/{user}/{repo}.git"

        try:
            self._run("git", "init")
            self._run("git", "remote", "remove", "origin", check=False)
            self._run("git", "remote", "add", "origin", clean_url)
            self._run("git", "config", "user.email", "rune@rune.local")
            self._run("git", "config", "user.name", "Rune")
            # Disable any system-wide credential helper that might cache.
            self._run("git", "config", "credential.helper", "")
            self._configured = True
            log.info("Git sync configured for %s/%s (token kept in memory only)", user, repo)
            return True
        except Exception as exc:
            log.warning("Git configure failed: %s", _scrub(str(exc), self._token))
            return False

    def push_async(self) -> None:
        """Push changes in a background thread."""
        if not self._configured:
            return
        thread = threading.Thread(target=self._push, daemon=True)
        thread.start()

    def _push(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._run("git", "add", "-A")
            self._run(
                "git", "commit", "-m", "auto-sync", "--allow-empty",
                check=False,
            )

            if not self._token:
                log.warning("Git push aborted: no token in memory")
                return
            with _temporary_askpass(self._token) as askpass_path:
                env = os.environ.copy()
                env["GIT_ASKPASS"] = askpass_path
                env["GIT_TERMINAL_PROMPT"] = "0"
                # GitHub fine-grained tokens accept any username, but
                # convention is to use "x-access-token" or the token
                # owner's login. The token itself acts as the password.
                env["GIT_USERNAME"] = self._user or "x-access-token"
                self._run_env(
                    env,
                    "git", "push", "-u", "origin", "main", "--force",
                )
            log.debug("Git push completed")
        except Exception as exc:
            log.warning("Git push failed: %s", _scrub(str(exc), self._token))
        finally:
            self._lock.release()

    # ── Internals ──────────────────────────────────────────────────────

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return self._run_env(None, *args, check=check)

    def _run_env(
        self,
        env: dict | None,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                args,
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                timeout=30,
                check=check,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            # Scrub token from any error output before re-raising.
            if self._token and (exc.stdout or exc.stderr):
                exc.stdout = _scrub(exc.stdout or "", self._token)
                exc.stderr = _scrub(exc.stderr or "", self._token)
            raise


def _scrub(text: str, token: str | None) -> str:
    """Replace any occurrence of the token in a string with [REDACTED]."""
    if not token or not text:
        return text
    return text.replace(token, "[REDACTED]")
