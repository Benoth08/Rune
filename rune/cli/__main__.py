"""Point d'entrée pour ``python -m rune.cli``.

Permet de lancer la CLI Typer sans installer l'entry point ``rune`` :

    python -m rune.cli chat
    python -m rune.cli serve
    python -m rune.cli status

Évite le RuntimeWarning de ``python -m rune.cli.main`` (qui exécute un
sous-module déjà importé par le package).
"""
from __future__ import annotations

from rune.cli.main import app

if __name__ == "__main__":
    app()
