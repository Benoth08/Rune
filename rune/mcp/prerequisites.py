"""Node.js prerequisites for MCP servers.

Lythéa's MCP servers (filesystem, GitHub, YouTube) are all distributed
as npm packages and run via ``npx``. This module detects whether Node
is installed and gives the user clear install instructions if not.

Design choice : we do NOT auto-install Node.js at boot. Reasons :
  - Auto-install requires either ``sudo`` (apt) or modifies ``~/.bashrc``
    (nvm/fnm). Both are intrusive without user consent.
  - First boot would become long and surprising.
  - Better to fail fast with a clear instruction than silently install.

A separate script ``scripts/install_node.sh`` is provided for users
who want one-line auto-installation via fnm.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger("lythea.mcp.prerequisites")


# Minimal Node version : MCP servers typically need 18+. We check
# but don't enforce strictly — a warning is enough.
MIN_NODE_MAJOR = 18


@dataclass(frozen=True)
class NodeStatus:
    """Result of checking Node.js availability.

    Attributes
    ----------
    available : bool
        True if ``node`` and ``npx`` are both on PATH and runnable.
    node_version : str
        Detected version string (e.g. "v20.10.0"), empty if absent.
    npx_path : str
        Resolved path to npx, empty if absent.
    too_old : bool
        True if Node is present but older than :data:`MIN_NODE_MAJOR`.
    """

    available: bool
    node_version: str
    npx_path: str
    too_old: bool


NODE_INSTALL_INSTRUCTIONS = """
Node.js est requis pour les outils MCP de Lythéa (filesystem, GitHub,
YouTube). Voici comment l'installer en 1 minute :

  ── Linux / macOS (recommandé : fnm, portable, sans sudo) ──
  curl -fsSL https://fnm.vercel.app/install | bash
  source ~/.bashrc   # ou ~/.zshrc selon ton shell
  fnm install 20
  fnm use 20

  ── Linux (alternative : apt) ──
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
  sudo apt-get install -y nodejs

  ── macOS (alternative : Homebrew) ──
  brew install node

  ── Windows ──
  https://nodejs.org/  (télécharger le .msi LTS)

Une fois Node installé, relance Lythéa. Les serveurs MCP démarreront
automatiquement.

Tu peux aussi utiliser le script fourni : bash scripts/install_node.sh
"""


def check_node() -> NodeStatus:
    """Detect Node.js + npx availability.

    Returns
    -------
    NodeStatus
        Diagnostic dataclass. If ``available=False``, the caller should
        log :data:`NODE_INSTALL_INSTRUCTIONS` so the user knows what
        to do.
    """
    node_bin = shutil.which("node")
    npx_bin = shutil.which("npx")

    if not node_bin or not npx_bin:
        return NodeStatus(
            available=False,
            node_version="",
            npx_path="",
            too_old=False,
        )

    # Try to get version
    version_str = ""
    too_old = False
    try:
        proc = subprocess.run(
            [node_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        version_str = proc.stdout.strip()
        # Parse "v20.10.0" → major = 20
        if version_str.startswith("v"):
            try:
                major = int(version_str[1:].split(".")[0])
                too_old = major < MIN_NODE_MAJOR
            except (ValueError, IndexError):
                pass
    except Exception:
        log.warning("Failed to check Node version", exc_info=True)

    return NodeStatus(
        available=True,
        node_version=version_str,
        npx_path=npx_bin,
        too_old=too_old,
    )


def log_status_or_instructions(status: NodeStatus) -> None:
    """Helper that logs either confirmation or install instructions."""
    if status.available:
        if status.too_old:
            log.warning(
                "Node.js détecté (%s) mais ancien. MCP servers nécessitent "
                "Node 18+. Mise à jour recommandée.",
                status.node_version,
            )
        else:
            log.info(
                "Node.js OK (%s) — MCP servers peuvent démarrer.",
                status.node_version,
            )
    else:
        log.warning(
            "Node.js absent — outils MCP désactivés.\n%s",
            NODE_INSTALL_INSTRUCTIONS,
        )
