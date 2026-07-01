"""Couche agents — subagents isolés + cron cognitif.

Modules
-------
- :class:`SubAgentSpawner` — lance des sous-agents en subprocess isolé
  (sandbox). Inspiré de Rune (délégation spatiale).
- :class:`CronScheduler` — planificateur de tâches de fond qui déclenche
  un microsleep post-run pour consolider les patterns appris.

Inspirations
------------
- **Rune** : subagents isolés en conteneur via RPC, détruits après
  usage. On adapte en subprocess Python (plus léger que Docker).
- **OpenClaw** : cron de fond pour scraping/audit. On ajoute la
  consolidation post-run (unique à Lythea).
"""
from __future__ import annotations

from .subagent import SubAgentSpawner, SubAgentResult, SubAgentConfig
from .cron import CronScheduler, CronTask, CronRunResult

__all__ = [
    "SubAgentSpawner",
    "SubAgentResult",
    "SubAgentConfig",
    "CronScheduler",
    "CronTask",
    "CronRunResult",
]
