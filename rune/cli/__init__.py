"""CLI Typer — point d'entrée principal.

Commandes :
    rune chat            # mode interactif console
    rune run "task"      # exécute une mission (one-shot)
    rune serve           # démarre l'API HTTP
    rune skills list     # liste les skills
    rune skills show ID  # affiche un skill
    rune skills archive ID
    rune cron list       # liste les tâches cron
    rune cron add ...    # ajoute une tâche
    rune cron run ID     # exécute une tâche immédiatement
    rune cron start      # démarre le scheduler en arrière-plan
    rune status          # affiche le statut global
    rune consolidate     # force un microsleep
    rune deep-sleep      # force un deep sleep
"""
from __future__ import annotations

from .main import app

__all__ = ["app"]
