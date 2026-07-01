"""Sandboxed execution for the agent — per-mission venv + whitelisted runs.

SECURITY MODEL (read before touching this file)
-----------------------------------------------
* **The orchestrator chooses what runs, never the model.** Only the
  commands built here can execute (``pytest`` / ``python <file in mission>`` /
  ``pip install <validated package>``). The 7B never supplies a command line.
  Everything is ``argv`` lists — ``shell=True`` is never used.
* **cwd is pinned to the mission directory.** Nothing runs outside it.
* **One venv per mission, created ``--system-site-packages``.** It *reads* the
  pod's heavy libs (torch, numpy, transformers — no multi-GB re-download) but
  every ``pip install`` lands in the mission's venv, so the pod environment /
  the running app are never mutated. This is what makes auto-install
  acceptable: the blast radius is the mission's ``.venv``.
* **Dependency installs are reactive and validated.** They are triggered by a
  real ``ModuleNotFoundError`` from a test run, the import name is mapped to a
  PyPI name, the package is checked to *exist on PyPI*, and the number of
  installs per mission is capped. Network is only reachable during the install
  call itself.
* **Isolation is layered / best-effort.** We prefer a network namespace
  (``unshare -n``) for no-network runs when the kernel allows it; on a stock
  container pod this is usually blocked and we fall back to a plain subprocess
  with rlimits + timeout + the cwd jail. The load-bearing control is the
  command whitelist, not the sandbox technology.

This module is pure mechanism (stdlib only). The plan→run→install→fix loop
lives in the orchestrator.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# A conservative PyPI package-name pattern (defends against injection in the
# install argv even though names come from import errors, not the model).
_SAFE_PKG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_MISSING = re.compile(r"No module named ['\"]?([A-Za-z0-9_]+)")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Import name → PyPI package name, for the common cases where they differ.
_MODULE_TO_PYPI = {
    "cv2": "opencv-python", "PIL": "Pillow", "sklearn": "scikit-learn",
    "yaml": "PyYAML", "bs4": "beautifulsoup4", "skimage": "scikit-image",
    "dotenv": "python-dotenv", "dateutil": "python-dateutil",
    "OpenSSL": "pyOpenSSL", "Crypto": "pycryptodome", "serial": "pyserial",
    "fitz": "PyMuPDF", "attr": "attrs", "jose": "python-jose",
    "magic": "python-magic", "win32api": "pywin32", "google": "protobuf",
}

# Never pip-install these — they are stdlib (the model loves to put "re" in a
# requirements.txt). If a test fails importing one of these, it's a code bug,
# not a missing dependency.
_STDLIB = {
    "re", "os", "sys", "json", "math", "random", "time", "datetime", "unittest",
    "typing", "pathlib", "collections", "itertools", "functools", "abc", "io",
    "subprocess", "threading", "asyncio", "dataclasses", "enum", "logging",
    "string", "statistics", "decimal", "fractions", "hashlib", "base64", "uuid",
    "csv", "sqlite3", "argparse", "copy", "heapq", "bisect", "queue", "socket",
    "struct", "array", "textwrap", "unicodedata", "html", "http", "urllib",
    "email", "glob", "shutil", "tempfile", "warnings", "contextlib", "secrets",
    "operator", "weakref", "inspect", "types", "traceback", "pprint", "gc",
}

_DEFAULT_TIMEOUT = 45
_INSTALL_TIMEOUT = 90
_MAX_OUTPUT = 16000


def module_to_pypi(mod: str) -> str:
    return _MODULE_TO_PYPI.get(mod, mod)


def parse_missing_module(text: str) -> str | None:
    """Top-level module name from a ``ModuleNotFoundError`` in test output."""
    m = _MISSING.search(text or "")
    if not m:
        return None
    return m.group(1).split(".")[0]


def pypi_package_exists(name: str, timeout: float = 8.0) -> bool:
    """Best-effort existence check via the PyPI JSON API (HTTP 200 = exists)."""
    if not _SAFE_PKG.match(name or ""):
        return False
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lythea-agent"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return getattr(resp, "status", resp.getcode()) == 200
    except Exception:  # noqa: BLE001
        return False


# ── isolation detection (cached) ──────────────────────────────────────
def _detect_isolation() -> dict:
    netns: list[str] = []
    if shutil.which("unshare"):
        try:
            p = subprocess.run(
                ["unshare", "-rn", "true"],
                capture_output=True, timeout=5,
            )
            if p.returncode == 0:
                netns = ["unshare", "-rn"]
        except Exception:  # noqa: BLE001
            pass
    tools = {t: bool(shutil.which(t)) for t in ("nsjail", "firejail", "unshare")}
    mode = "unshare-netns" if netns else "subprocess"
    return {"netns_prefix": netns, "tools": tools, "mode": mode}


_ISO: dict | None = None


def isolation() -> dict:
    global _ISO
    if _ISO is None:
        _ISO = _detect_isolation()
        log.info("agent sandbox isolation mode: %s (%s)", _ISO["mode"], _ISO["tools"])
    return _ISO


def _set_limits() -> None:  # POSIX preexec_fn
    # Deliberately NOT capping RLIMIT_AS: a --system-site-packages venv may
    # import torch/numpy, which reserve enormous *virtual* address space; an AS
    # cap would false-OOM legitimate imports. Memory is bounded by the wall
    # timeout and the pod's own cgroup limits instead.
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (90, 100))
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource

        cap = 100 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
    except Exception:  # noqa: BLE001
        pass


@dataclass
class SandboxResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    cmd: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.timed_out:
            return f"⏱ timeout après {self.duration:.0f}s"
        text = _ANSI_RE.sub("", (self.stdout or "") + "\n" + (self.stderr or ""))
        m = re.search(r"\b(\d+ failed[^\n]*|\d+ passed[^\n]*|\d+ error[^\n]*)", text)
        if m:
            return m.group(1).strip()
        for line in reversed(text.splitlines()):
            if line.strip():
                return line.strip()[:160]
        return "tests OK" if self.ok else f"code {self.returncode}"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration": round(self.duration, 2),
            "summary": self.summary(),
            "stdout_tail": (self.stdout or "")[-2000:],
            "stderr_tail": (self.stderr or "")[-2000:],
        }


class MissionSandbox:
    """Run code for a single mission inside its own venv + cwd jail."""

    def __init__(
        self,
        mission_dir,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        install_timeout: int = _INSTALL_TIMEOUT,
        max_installs: int = 5,
        allow_install: bool = True,
    ):
        self.dir = Path(mission_dir).resolve()
        self.timeout = timeout
        self.install_timeout = install_timeout
        self.max_installs = max_installs
        self.allow_install = allow_install
        self._venv_py: str | None = None
        self.installs: int = 0
        self._pytest_ok: bool = False

    # ── venv ─────────────────────────────────────────────────────────
    @property
    def venv_dir(self) -> Path:
        return self.dir / ".venv"

    def ensure_venv(self) -> str:
        if self._venv_py and Path(self._venv_py).exists():
            return self._venv_py
        py = self.venv_dir / "bin" / "python"
        if not py.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", "--system-site-packages",
                     str(self.venv_dir)],
                    capture_output=True, timeout=120, check=False,
                )
            except Exception:  # noqa: BLE001
                log.exception("venv creation failed for %s", self.dir)
        # Fall back to the running interpreter if venv creation didn't work.
        self._venv_py = str(py) if py.exists() else sys.executable
        return self._venv_py

    # ── low-level run ──────────────────────────────────────────────────
    def _make_env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PIP_NO_INPUT"] = "1"
        # Put the venv first on PATH so `python`/`pip`/`pytest` resolve to it
        # while `node`/`npm`/`ruff`… resolve from the system.
        venv_bin = str(self.venv_dir / "bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(self.venv_dir)
        return env

    def _exec(self, cmd: list[str], *, net: bool, timeout: int) -> SandboxResult:
        prefix = [] if net else isolation()["netns_prefix"]
        full = [*prefix, *cmd]
        env = self._make_env()
        t0 = time.monotonic()
        timed_out = False
        try:
            p = subprocess.run(
                full, cwd=str(self.dir), capture_output=True, text=True,
                timeout=timeout, env=env,
                preexec_fn=_set_limits if os.name == "posix" else None,
            )
            rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
        except subprocess.TimeoutExpired as e:
            timed_out = True
            rc = -1
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            err = ((e.stderr or "") if isinstance(e.stderr, str) else "") + "\n[timeout]"
        except Exception as exc:  # noqa: BLE001
            log.exception("sandbox run failed: %s", full)
            rc, out, err = -1, "", f"[sandbox error] {exc}"
        dur = time.monotonic() - t0
        out = _ANSI.sub("", out)
        err = _ANSI.sub("", err)
        return SandboxResult(
            ok=(rc == 0 and not timed_out), returncode=rc,
            stdout=out[-_MAX_OUTPUT:], stderr=err[-_MAX_OUTPUT:],
            duration=dur, timed_out=timed_out, cmd=full,
        )

    def _run(self, args: list[str], *, net: bool, timeout: int) -> SandboxResult:
        return self._exec([self.ensure_venv(), *args], net=net, timeout=timeout)

    def run_argv(self, argv: list[str], *, net: bool, timeout: int | None = None) -> SandboxResult:
        """Run an arbitrary argv (executable + args) in the mission cwd/venv.

        The *allowlisting* of which executables may run lives in the bounded
        tool layer (``tools.validate_command``); this method is pure mechanism.
        """
        self.ensure_venv()
        return self._exec(list(argv), net=net, timeout=timeout or self.timeout)

    def serve_and_probe(
        self, argv: list[str], paths: list[str], *,
        port: int = 8000, boot_timeout: int = 12, probe_timeout: int = 5,
    ) -> dict:
        """Boot a server (whitelisted argv) in the background, hit a few HTTP
        paths on localhost, then tear it down. No network namespace here — the
        probe must reach 127.0.0.1."""
        self.ensure_venv()
        env = self._make_env()
        started = False
        probes: list[dict] = []
        proc = None
        try:
            proc = subprocess.Popen(
                list(argv), cwd=str(self.dir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            deadline = time.monotonic() + boot_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break  # process exited/crashed before binding
                with socket.socket() as s:
                    s.settimeout(0.5)
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        started = True
                        break
                time.sleep(0.3)
            if started:
                for path in (paths or ["/"])[:8]:
                    p = path if str(path).startswith("/") else "/" + str(path)
                    url = f"http://127.0.0.1:{port}{p}"
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "lythea-agent"})
                        with urllib.request.urlopen(req, timeout=probe_timeout) as r:  # noqa: S310
                            body = r.read(500).decode("utf-8", "replace")
                            probes.append({
                                "path": p,
                                "status": getattr(r, "status", r.getcode()),
                                "body": body,
                            })
                    except Exception as exc:  # noqa: BLE001
                        probes.append({"path": p, "error": str(exc)})
        finally:
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=3)
                except Exception:  # noqa: BLE001
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:  # noqa: BLE001
                        pass
        log_tail = ""
        if proc is not None and proc.stdout:
            try:
                log_tail = (proc.stdout.read() or "")[-2000:]
            except Exception:  # noqa: BLE001
                pass
        return {"started": started, "probes": probes, "log_tail": log_tail}

    # ── whitelisted high-level actions ─────────────────────────────────
    def _ensure_pytest(self) -> bool:
        """pytest is the runner — guarantee it's in the venv (the pod's base
        env may not ship it). Installed once, into the venv only."""
        if self._pytest_ok:
            return True
        if self._run(["-c", "import pytest"], net=False, timeout=30).ok:
            self._pytest_ok = True
            return True
        log.info("pytest absent du venv — installation dans le venv…")
        res = self._run(
            ["-m", "pip", "install", "--no-input", "pytest"],
            net=True, timeout=self.install_timeout,
        )
        self._pytest_ok = res.ok
        return res.ok

    def run_pytest(self) -> SandboxResult:
        self._ensure_pytest()
        # -rf + --tb=short surface a concise "FAILED test::name - AssertionError"
        # summary so a weak model can target the fix instead of guessing.
        return self._run(
            ["-m", "pytest", "-q", "--color=no", "--tb=short", "-rf"],
            net=False, timeout=self.timeout)

    def run_file(self, rel_path: str) -> SandboxResult:
        # Only allow a path that stays inside the mission dir.
        target = (self.dir / rel_path).resolve()
        try:
            target.relative_to(self.dir)
        except ValueError:
            return SandboxResult(False, -1, "", "[refusé: hors mission]", 0.0)
        return self._run([str(target)], net=False, timeout=self.timeout)

    def pip_install(self, module_or_pkg: str) -> tuple[bool, SandboxResult | None]:
        """Install one validated package. Returns (ok, result|None)."""
        if not self.allow_install or self.installs >= self.max_installs:
            return False, None
        name = module_to_pypi(module_or_pkg)
        if not _SAFE_PKG.match(name) or not pypi_package_exists(name):
            return False, None
        self.installs += 1
        res = self._run(
            ["-m", "pip", "install", "--no-input", "--disable-pip-version-check", name],
            net=True,
            timeout=self.install_timeout,
        )
        return res.ok, res

    # ── discovery ──────────────────────────────────────────────────────
    def discover_tests(self) -> bool:
        if not self.dir.exists():
            return False
        for p in self.dir.rglob("*.py"):
            if ".venv" in p.parts:
                continue
            n = p.name
            if n.startswith("test_") or n.endswith("_test.py"):
                return True
        return False


def make_mission_sandbox(mission_dir, settings=None) -> MissionSandbox:
    """Factory honouring settings (read with getattr so no schema change)."""
    g = lambda k, d: getattr(settings, k, d) if settings is not None else d  # noqa: E731
    return MissionSandbox(
        mission_dir,
        timeout=int(g("agent_exec_timeout_s", _DEFAULT_TIMEOUT)),
        install_timeout=int(g("agent_install_timeout_s", _INSTALL_TIMEOUT)),
        max_installs=int(g("agent_max_installs", 5)),
        allow_install=bool(g("agent_pip_install", True)),
    )
