"""Tests for model_loadability_info."""
from __future__ import annotations

from unittest.mock import patch

import pytest

torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="torch not installed")


def _setup_vram(monkeypatch, free_gb: float, total_gb: float = 24.0):
    """Patch VRAM helpers to predictable values."""
    monkeypatch.setattr("lythea.model.vram_free_gb", lambda: free_gb)
    monkeypatch.setattr("lythea.model.vram_total_gb", lambda: total_gb)
    # DEVICE override — must patch the symbol where it's used (model.py)
    monkeypatch.setattr("lythea.model.DEVICE", "cuda")


def test_loadable_when_enough_vram(monkeypatch):
    from rune.model import model_loadability_info
    _setup_vram(monkeypatch, free_gb=20.0)

    info = model_loadability_info("Qwen/Qwen2.5-7B-Instruct")
    assert info["loadable"] is True
    assert info["block_reason"] is None
    assert info["vram_required_gb"] == 15.0
    assert info["vram_available_gb"] == 20.0


def test_blocked_when_not_enough_vram(monkeypatch):
    from rune.model import model_loadability_info
    _setup_vram(monkeypatch, free_gb=10.0)

    info = model_loadability_info("Qwen/Qwen2.5-7B-Instruct")
    assert info["loadable"] is False
    assert info["block_reason"] is not None
    assert "VRAM insuffisante" in info["block_reason"]


def test_blip_swap_unblocks(monkeypatch):
    """A model blocked by Qwen2-VL captioner should be loadable_with_blip=True."""
    from rune.model import model_loadability_info
    # 13 GB free + 4.5 GB freed by switching to BLIP = 17.5 GB → 15 GB needed = OK
    _setup_vram(monkeypatch, free_gb=13.0)

    info = model_loadability_info(
        "Qwen/Qwen2.5-7B-Instruct",
        captioner_backend="qwen2vl",
        captioner_vram_gb=4.5,
    )
    assert info["loadable"] is False
    assert info["loadable_with_blip"] is True
    assert "BLIP" in info["block_reason"]


def test_unload_unblocks(monkeypatch):
    """A model blocked by current LLM should be loadable_after_unload=True."""
    from rune.model import model_loadability_info
    # 1 GB free + 7 GB from current LLM unload = 8 GB → 7 GB needed = OK for the 3B
    _setup_vram(monkeypatch, free_gb=1.0)

    info = model_loadability_info(
        "Qwen/Qwen2.5-3B-Instruct",
        current_loaded_id="some/other-model",
        current_loaded_size_gb=7.0,
    )
    assert info["loadable"] is False
    assert info["loadable_after_unload"] is True


def test_truly_too_big_for_gpu(monkeypatch):
    """A model bigger than the GPU's total VRAM should report total mismatch.

    Note: previously used Qwen2.5-14B (28 GB) which was removed from
    CATALOG in v9. Qwen3-8B (17 GB) suffices to fail a 12 GB GPU check.
    """
    from rune.model import model_loadability_info
    _setup_vram(monkeypatch, free_gb=8.0, total_gb=12.0)

    info = model_loadability_info("Qwen/Qwen3-8B")  # 17 GB > 12 GB total
    assert info["loadable"] is False
    assert info["loadable_with_blip"] is False
    assert info["loadable_after_unload"] is False


def test_cpu_mode_always_loadable(monkeypatch):
    from rune.model import model_loadability_info
    monkeypatch.setattr("lythea.model.DEVICE", "cpu")

    info = model_loadability_info("Qwen/Qwen2.5-7B-Instruct")
    assert info["loadable"] is True
    assert info["block_reason"] is None


def test_unknown_model(monkeypatch):
    from rune.model import model_loadability_info
    _setup_vram(monkeypatch, free_gb=20.0)

    info = model_loadability_info("nonexistent/model-x")
    assert info["loadable"] is False
    assert info["block_reason"] == "Modèle inconnu"
