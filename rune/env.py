"""Bootstrap environment variables before any HuggingFace import.

This module MUST be imported before `transformers`, `torch`, or any HF
library. It redirects all caches to a platform-aware root directory to
avoid filling up the system disk on RunPod / Colab.
"""
from __future__ import annotations

import os
from pathlib import Path

PLATFORMS = ("runpod", "colab", "kaggle", "local")


def detect_platform() -> str:
    """Detect the execution platform from filesystem markers."""
    if Path("/workspace").is_dir():
        return "runpod"
    if Path("/content").is_dir() and "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    return "local"


def bootstrap_env() -> Path:
    """Set environment variables and return the cache root.

    Returns
    -------
    Path
        The root directory for all Lythéa caches and data.
    """
    platform = detect_platform()

    roots = {
        "runpod": Path("/workspace/.lythea"),
        "colab": Path("/content/.lythea"),
        "kaggle": Path("/kaggle/working/.lythea"),
        "local": Path.home() / ".lythea",
    }
    root = roots[platform]
    root.mkdir(parents=True, exist_ok=True)

    defaults = {
        "HF_HOME": str(root / "hf"),
        "HF_HUB_CACHE": str(root / "hf" / "hub"),
        "TORCH_HOME": str(root / "torch"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)

    return root


PLATFORM = detect_platform()
CACHE_ROOT = bootstrap_env()
