"""V4 — Tests for /api/config/v4/* routes (status + toggle).

These tests use FastAPI's TestClient with a fake hippocampe that
exposes ``v4_status`` + ``v4_set_module`` matching the real contract.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")  # routes.py imports lythea.model → torch

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from rune.server.routes import router  # noqa: E402


# ════════════════════════════════════════════════════════════════════
# Fake Hippocampe with the minimum surface routes need
# ════════════════════════════════════════════════════════════════════


class _FakeHippo:
    def __init__(self):
        self._modules: dict[str, bool] = {
            "cognitive_state": False,
            "inhibition": False,
            "planning": False,
            "predictive_coding": False,
            "timeline": False,
            "metacognition": False,
        }
        self._affect_modulates = False
        self._pc_apply_gating = False

    def v4_status(self) -> dict:
        snap: dict = {}
        for m, on in self._modules.items():
            snap[m] = {"enabled": on}
        snap["affect_modulates_consolidation"] = self._affect_modulates
        snap["predictive_coding"]["apply_gating"] = self._pc_apply_gating
        return snap

    def v4_set_module(self, module: str, enabled: bool) -> dict:
        if module in self._modules:
            self._modules[module] = bool(enabled)
        elif module == "affect_modulates_consolidation":
            self._affect_modulates = bool(enabled)
        elif module == "predictive_coding_apply_gating":
            self._pc_apply_gating = bool(enabled)
        return self.v4_status()


class _FakeApp:
    def __init__(self):
        self.hippocampe = _FakeHippo()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    fake = _FakeApp()
    app.state.lythea = fake
    return TestClient(app)


# ════════════════════════════════════════════════════════════════════
# 1. GET /api/config/v4
# ════════════════════════════════════════════════════════════════════


def test_get_v4_status_returns_all_modules(client):
    res = client.get("/api/config/v4")
    assert res.status_code == 200
    body = res.json()
    for m in (
        "cognitive_state", "inhibition", "planning",
        "predictive_coding", "timeline", "metacognition",
    ):
        assert m in body
        assert body[m]["enabled"] is False  # all OFF on fresh fake


def test_get_v4_status_idempotent(client):
    a = client.get("/api/config/v4").json()
    b = client.get("/api/config/v4").json()
    assert a == b


# ════════════════════════════════════════════════════════════════════
# 2. POST /api/config/v4/toggle
# ════════════════════════════════════════════════════════════════════


def test_toggle_explicit_true(client):
    res = client.post(
        "/api/config/v4/toggle",
        json={"module": "planning", "enabled": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["planning"]["enabled"] is True


def test_toggle_explicit_false(client):
    # First turn it on
    client.post("/api/config/v4/toggle", json={"module": "planning", "enabled": True})
    # Then off
    res = client.post(
        "/api/config/v4/toggle", json={"module": "planning", "enabled": False},
    )
    assert res.json()["planning"]["enabled"] is False


def test_toggle_flip_when_no_enabled_field(client):
    """No `enabled` in body → flip current state."""
    # Currently False
    res = client.post("/api/config/v4/toggle", json={"module": "timeline"})
    assert res.json()["timeline"]["enabled"] is True
    # Currently True → flip back
    res = client.post("/api/config/v4/toggle", json={"module": "timeline"})
    assert res.json()["timeline"]["enabled"] is False


def test_toggle_unknown_module_returns_error(client):
    res = client.post(
        "/api/config/v4/toggle", json={"module": "bogus", "enabled": True},
    )
    body = res.json()
    assert body.get("error") == "unknown_module"
    assert "known" in body


def test_toggle_empty_body_returns_error(client):
    res = client.post("/api/config/v4/toggle", json={})
    assert res.json().get("error") == "unknown_module"


def test_toggle_affect_modulates_consolidation(client):
    res = client.post(
        "/api/config/v4/toggle",
        json={"module": "affect_modulates_consolidation", "enabled": True},
    )
    body = res.json()
    assert body["affect_modulates_consolidation"] is True


def test_toggle_pc_apply_gating(client):
    res = client.post(
        "/api/config/v4/toggle",
        json={"module": "predictive_coding_apply_gating", "enabled": True},
    )
    body = res.json()
    assert body["predictive_coding"]["apply_gating"] is True


def test_toggle_each_module_in_sequence(client):
    """Toggle every cognitive module ON one at a time."""
    modules = [
        "cognitive_state", "inhibition", "planning",
        "predictive_coding", "timeline", "metacognition",
    ]
    for m in modules:
        res = client.post(
            "/api/config/v4/toggle", json={"module": m, "enabled": True},
        )
        assert res.status_code == 200
        assert res.json()[m]["enabled"] is True
    # All should now be on
    final = client.get("/api/config/v4").json()
    for m in modules:
        assert final[m]["enabled"] is True
