"""Tests for the agent sandbox — pure mechanism + a real local venv run.

No network is exercised (PyPI checks are only run on invalid names, which
short-circuit before any HTTP call). venv creation is local and offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rune.agentic.sandbox import (
    MissionSandbox,
    SandboxResult,
    isolation,
    module_to_pypi,
    parse_missing_module,
    pypi_package_exists,
)


# ── pure helpers ──────────────────────────────────────────────────────
def test_parse_missing_module_top_level():
    err = "...\nModuleNotFoundError: No module named 'numpy.linalg'\n"
    assert parse_missing_module(err) == "numpy"
    assert parse_missing_module("all good") is None


def test_module_to_pypi_mapping():
    assert module_to_pypi("PIL") == "Pillow"
    assert module_to_pypi("sklearn") == "scikit-learn"
    assert module_to_pypi("requests") == "requests"  # unmapped → identity


def test_pypi_existence_rejects_bad_names_without_network():
    # Invalid names fail the regex and never hit the network.
    assert pypi_package_exists("../evil") is False
    assert pypi_package_exists("a b c") is False
    assert pypi_package_exists("") is False


def test_isolation_reports_a_mode():
    iso = isolation()
    assert iso["mode"] in ("subprocess", "unshare-netns")
    assert "tools" in iso


def test_sandbox_result_summary():
    r = SandboxResult(ok=False, returncode=1, stdout="1 failed, 2 passed in 0.1s",
                      stderr="", duration=0.1)
    assert "failed" in r.summary()
    t = SandboxResult(ok=False, returncode=-1, stdout="", stderr="", duration=45.0,
                      timed_out=True)
    assert "timeout" in t.summary()


# ── discovery + install guards ────────────────────────────────────────
def test_discover_tests():
    with tempfile.TemporaryDirectory() as d:
        sb = MissionSandbox(d)
        assert sb.discover_tests() is False
        (Path(d) / "test_thing.py").write_text("def test_x():\n    assert True\n")
        assert sb.discover_tests() is True


def test_pip_install_guards():
    with tempfile.TemporaryDirectory() as d:
        # Cap at 0 → never installs.
        sb = MissionSandbox(d, max_installs=0)
        assert sb.pip_install("requests") == (False, None)
        # Disabled → never installs.
        sb2 = MissionSandbox(d, allow_install=False)
        assert sb2.pip_install("requests") == (False, None)
        # Invalid name → rejected before any network.
        sb3 = MissionSandbox(d)
        assert sb3.pip_install("../evil")[0] is False


# ── real (local, offline) execution ───────────────────────────────────
def test_run_file_in_venv():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "say.py").write_text("print('hello-from-sandbox')\n")
        sb = MissionSandbox(d, timeout=60)
        res = sb.run_file("say.py")
        assert res.ok, res.stderr
        assert "hello-from-sandbox" in res.stdout


def test_run_pytest_pass_and_fail():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
        sb = MissionSandbox(d, timeout=60)
        res = sb.run_pytest()
        assert res.ok, res.stdout + res.stderr

        (Path(d) / "test_ko.py").write_text("def test_ko():\n    assert False\n")
        res2 = sb.run_pytest()
        assert res2.ok is False
        assert "failed" in res2.summary()


def test_run_file_outside_mission_refused():
    with tempfile.TemporaryDirectory() as d:
        sb = MissionSandbox(d)
        res = sb.run_file("../../etc/passwd")
        assert res.ok is False


# ── run_argv + serve_and_probe (A/B/E), real local ───────────────────
def test_run_argv_python():
    with tempfile.TemporaryDirectory() as d:
        sb = MissionSandbox(d, timeout=60)
        res = sb.run_argv(["python", "-c", "print('hi-argv')"], net=False)
        assert res.ok, res.stderr
        assert "hi-argv" in res.stdout


def test_serve_and_probe_local():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "srv.py").write_text(
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "class H(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200); self.end_headers(); self.wfile.write(b'OK-PROBE')\n"
            "    def log_message(self, *a):\n        pass\n"
            "HTTPServer(('127.0.0.1', 8077), H).serve_forever()\n"
        )
        sb = MissionSandbox(d, timeout=60)
        out = sb.serve_and_probe(["python", "srv.py"], ["/"], port=8077, boot_timeout=10)
        assert out["started"] is True, out
        assert out["probes"] and out["probes"][0].get("status") == 200
