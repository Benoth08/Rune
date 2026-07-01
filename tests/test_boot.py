"""Tests for the boot orchestrator."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Skip the integration tests below if torch isn't installed (sandbox CI).
# The unit tests for BootState don't need torch.
torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False

from rune.boot import STAGES, BootRunner, BootState


def test_initial_state():
    """A fresh BootState should report not-ready and step_index=0."""
    s = BootState()
    assert s.ready is False
    assert s.step_index == 0
    assert s.step_total == len(STAGES)
    assert s.progress_pct == 0.0
    assert s.components == {}


def test_to_dict_keys():
    """to_dict must expose all fields the UI relies on."""
    s = BootState()
    d = s.to_dict()
    expected_keys = {
        "ready", "current_step", "step_index", "step_total",
        "progress_pct", "elapsed_s", "details", "components",
        "messages", "stage_labels",
    }
    assert expected_keys.issubset(d.keys())


def test_begin_and_end_stage():
    """begin_stage updates current_step and step_index, end_stage records status."""
    s = BootState()
    s.begin_stage("gliner", details="loading…")
    assert s.current_step == "loading_gliner"
    assert s.step_index == STAGES.index("gliner") + 1
    assert s.details == "loading…"

    s.end_stage("gliner", "ok")
    assert s.components["gliner"] == "ok"


def test_finalize_marks_ready():
    """finalize sets ready=True and progress 100%."""
    s = BootState()
    s.finalize()
    assert s.ready is True
    assert s.progress_pct == 100.0
    assert s.current_step == "done"


def test_failed_stage_recorded_in_messages():
    """A failed stage records a human-readable message."""
    s = BootState()
    s.end_stage("gliner", "failed", "module missing")
    assert "gliner" in s.components
    assert s.components["gliner"] == "failed"
    assert any("gliner" in m for m in s.messages)


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_runner_handles_chromadb_failure_gracefully(monkeypatch):
    """A stage exception must not crash the boot — runner converges to ready=True."""
    state = BootState()

    fake_app = MagicMock()
    fake_app.chroma_collection.count.side_effect = RuntimeError("boom")
    fake_app.retriever._maybe_rebuild_bm25 = MagicMock()
    # Other stages: GLiNER passes
    fake_app.entity_extractor.extract.return_value = []
    fake_app.entity_extractor.encode.return_value = MagicMock()
    fake_app.retriever._get_cross_encoder.return_value = None
    fake_app.hippocampe.captioner.select.return_value = {"status": "loaded", "backend": "blip"}

    # Bypass VRAM detection
    monkeypatch.setattr(
        "lythea.model.vram_free_gb", lambda: 0.0,
    )

    runner = BootRunner(fake_app, state)
    runner._run()  # synchronous version of the boot

    assert state.ready is True
    assert state.components.get("chromadb") == "failed"
    assert any("chromadb" in m for m in state.messages)


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_runner_picks_blip_when_low_vram(monkeypatch):
    """Captioner stage must prefer BLIP when VRAM < 5 GB."""
    state = BootState()

    fake_app = MagicMock()
    fake_app.chroma_collection.count.return_value = 0
    fake_app.entity_extractor.extract.return_value = []
    fake_app.entity_extractor.encode.return_value = MagicMock()
    fake_app.retriever._get_cross_encoder.return_value = None

    selected: list[str] = []
    def fake_select(choice):
        selected.append(choice)
        return {"status": "loaded", "backend": choice}
    fake_app.hippocampe.captioner.select.side_effect = fake_select

    monkeypatch.setattr("lythea.model.vram_free_gb", lambda: 2.0)

    runner = BootRunner(fake_app, state)
    runner._run()

    assert state.ready is True
    assert "blip" in selected
    # qwen2vl should NOT have been tried at low VRAM
    assert "qwen2vl" not in selected


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_runner_tries_qwen2vl_when_high_vram(monkeypatch):
    """Captioner stage must try Qwen2-VL when VRAM ≥ 5 GB."""
    state = BootState()

    fake_app = MagicMock()
    fake_app.chroma_collection.count.return_value = 0
    fake_app.entity_extractor.extract.return_value = []
    fake_app.entity_extractor.encode.return_value = MagicMock()
    fake_app.retriever._get_cross_encoder.return_value = None

    selected: list[str] = []
    def fake_select(choice):
        selected.append(choice)
        return {"status": "loaded", "backend": choice}
    fake_app.hippocampe.captioner.select.side_effect = fake_select

    monkeypatch.setattr("lythea.model.vram_free_gb", lambda: 18.0)

    runner = BootRunner(fake_app, state)
    runner._run()

    assert state.ready is True
    assert "qwen2vl" in selected
