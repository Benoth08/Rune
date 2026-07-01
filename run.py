#!/usr/bin/env python3
"""Lythéa V3 — One-click launcher.

Handles dependency installation, re-exec after critical upgrades,
and launches the FastAPI server.

Usage:
    python run.py [--port 7860] [--host 0.0.0.0] [--no-install] [--cache PATH]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import warnings
from pathlib import Path

# ── Suppress noisy warnings ───────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure package is importable
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Bootstrap environment FIRST ────────────────────────────────────────
from rune.env import bootstrap_env, detect_platform  # noqa: E402

cache_root = bootstrap_env()


# ── Dependency management ──────────────────────────────────────────────

DEPS = {
    # V5.6.7 — transformers>=5.5.0 requis pour Gemma 4 (architecture
    # 'gemma4' ajoutée dans la v5.5.0, avril 2026). Sans ça, erreur
    # "The checkpoint you are trying to load has model type 'gemma4'
    # but Transformers does not recognize this architecture".
    # Si transformers 5.x cause des régressions sur d'autres modèles
    # du catalogue, revenir à "4.57" et retirer les Gemma 4 du
    # CONFIG.CATALOG dans config.py.
    "transformers": "5.5.0",
    "accelerate": "0.30",
    "huggingface_hub": "0.24",
    "tokenizers": "0.20",
    "einops": "0",
    "timm": "0",
    "chromadb": "0.5",
    "fastapi": "0.110",
    "uvicorn": "0.27",
    "sse_starlette": "2.0",
    "pydantic": "2.10",
    "pydantic_settings": "2.0",
    "slowapi": "0.1.9",
    "gliner": "0.2",
    "sentence_transformers": "3.0",
    "rank_bm25": "0.2",
    "rapidfuzz": "3.0",
    "ddgs": "0",
    "pillow": "10.0",
    "numpy": "0",
    "pandas": "0",
    "httpx": "0",
    # V6.0.0-rc rev9 — Libs d'extraction documentaire (ingest.py +
    # server/document_ingest.py). Sans ça, l'upload de PDF/DOCX/XLSX
    # crashe avec ImportError obscur. Ajoutées au DEPS pour qu'elles
    # soient installées dès le premier boot.
    "pdfplumber": "0.10",
    "docx": "0",            # nom d'import (paquet pip = python-docx)
    "openpyxl": "3.1",      # extract_xlsx
    "multipart": "0",       # python-multipart, requis par FastAPI UploadFile
    "bs4": "0",             # BeautifulSoup (HTML/XML extraction propre)
    "lxml": "0",            # parser HTML/XML rapide pour bs4
    # V6.0.0-rc rev9 — Libs pour le knowledge graph + communautés.
    # graph_communities.py utilise networkx pour la structure de base
    # et igraph+leidenalg pour le clustering Leiden (meilleur que
    # Louvain sur petits graphes). python-louvain en fallback.
    "networkx": "3.0",
    "igraph": "0",          # nom d'import (paquet pip = python-igraph)
    "leidenalg": "0.10",
    "community": "0",       # nom d'import (paquet pip = python-louvain)
    "packaging": "23.0",    # version checks
}

# Packages that require re-exec if upgraded
REEXEC_PACKAGES = {"pydantic", "pydantic_core", "pydantic_settings", "typing_extensions"}

# NEVER touch these — they're pinned by the CUDA image
FROZEN = {"torch", "torchvision", "torchaudio"}

PIP_TO_IMPORT = {
    "pillow": "PIL",
    "sentence_transformers": "sentence_transformers",
    "ddgs": "ddgs",
    "rank_bm25": "rank_bm25",
    "sse_starlette": "sse_starlette",
    "huggingface_hub": "huggingface_hub",
}

PIP_NAME = {
    "PIL": "pillow",
    "sse_starlette": "sse-starlette",
    "sentence_transformers": "sentence-transformers",
    "ddgs": "ddgs",
    "rank_bm25": "rank-bm25",
    "huggingface_hub": "huggingface_hub",
    "pydantic_settings": "pydantic-settings",
    "uvicorn": "uvicorn[standard]",
    # V6.0.0-rc rev9 — Mappings pour les libs dont le nom d'import
    # diffère du nom pip. Sinon run.py essaierait d'installer "docx"
    # (qui n'existe pas) au lieu de "python-docx".
    "docx": "python-docx",
    "multipart": "python-multipart",
    "bs4": "beautifulsoup4",
    "igraph": "python-igraph",
    "community": "python-louvain",
}


def _get_version(pkg: str) -> str | None:
    """Get installed version, return None if not found."""
    try:
        return importlib.metadata.version(pkg.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        try:
            return importlib.metadata.version(pkg.replace("-", "_"))
        except importlib.metadata.PackageNotFoundError:
            return None


def _version_ok(installed: str, floor: str) -> bool:
    """Check if installed version meets minimum floor."""
    if floor == "0":
        return True
    try:
        from packaging.version import Version
        return Version(installed) >= Version(floor)
    except Exception:
        # Simple fallback
        return installed >= floor


def install_deps(skip: bool = False) -> bool:
    """Install missing or outdated dependencies. Returns True if re-exec needed."""
    if skip:
        return False

    need_reexec = False
    to_install: list[str] = []

    for pkg, floor in DEPS.items():
        if pkg in FROZEN:
            continue

        import_name = PIP_TO_IMPORT.get(pkg, pkg)
        installed = _get_version(pkg)

        if installed is None:
            pip_name = PIP_NAME.get(pkg, pkg)
            to_install.append(f"{pip_name}>={floor}" if floor != "0" else pip_name)
        elif floor != "0" and not _version_ok(installed, floor):
            pip_name = PIP_NAME.get(pkg, pkg)
            to_install.append(f"{pip_name}>={floor}")
            if pkg in REEXEC_PACKAGES:
                need_reexec = True

    if not to_install:
        return False

    print(f"[run.py] Installing: {', '.join(to_install)}")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-deps" if len(to_install) == 1 else "--quiet",
        *to_install,
    ]

    # On some systems pip needs --break-system-packages
    try:
        subprocess.run(
            [*cmd, "--break-system-packages"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(cmd, check=True)

    return need_reexec


def verify_torch() -> None:
    """Verify torch is available (never install/upgrade it)."""
    try:
        import torch
        print(f"[run.py] PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("[run.py] WARNING: PyTorch not found. Install it manually for your CUDA version.")
        print("[run.py] Continuing in CPU mode...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lythéa V3 Launcher")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--share", action="store_true", help="(reserved for future)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.cache:
        os.environ["HF_HOME"] = args.cache

    # Protection (7): re-exec after critical pip installs
    need_reexec = install_deps(skip=args.no_install)
    if need_reexec:
        print("[run.py] Critical dependencies upgraded — re-executing…")
        os.execv(sys.executable, [sys.executable, *sys.argv])

    verify_torch()

    print(f"[run.py] Platform: {detect_platform()}")
    print(f"[run.py] Cache root: {cache_root}")
    print(f"[run.py] Starting Lythéa on http://{args.host}:{args.port}")

    from rune.logging_setup import configure_logging, build_uvicorn_log_config
    configure_logging(args.log_level)

    import uvicorn
    from rune.server.app import create_app

    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        log_config=build_uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
