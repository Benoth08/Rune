"""V6 agentic layer — bounded plan/act/critique loop over a worker pool.

Sibling of the chat Hippocampe: composes the shared model + memory and
adds a multi-step, interruptible loop. See ``orchestrator.py`` and
``workers.py``.
"""

from __future__ import annotations

from rune.agentic.orchestrator import AgentOrchestrator
from rune.agentic.workers import (
    InProcessWorker,
    OllamaWorker,
    Worker,
    WorkerPool,
)

__all__ = [
    "AgentOrchestrator",
    "WorkerPool",
    "InProcessWorker",
    "OllamaWorker",
    "Worker",
]
