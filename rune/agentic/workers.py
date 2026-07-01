"""Worker pool — the inference backends the agent loop routes over.

V6 Phase 1. A *worker* is anything that turns a prompt into text. We
decouple **roles** (planner/executor/critic) from **workers** so the
agent can run on whatever is available:

* :class:`InProcessWorker` wraps Lythéa's in-process ModelWrapper — the
  steered "Rune voice" core. It is the only worker that can carry the
  ``soft_memory`` prefix / activation-level features, so any step that
  must *be* Rune is routed here (``needs_prefix``).
* :class:`OllamaWorker` calls a local Ollama daemon over HTTP — a cheap,
  parallel, text-only auxiliary worker for mechanical steps. Optional:
  if Ollama is unreachable the worker simply reports itself unavailable
  and the pool falls back to the in-process core.

The pool degrades gracefully: with no Ollama configured, everything runs
on the in-process core (sequential). Adding Ollama later, or cloud
workers, is a config change — the loop above never changes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

# Serialise access to the single in-process model across threads. A
# transformers model does one generate at a time; concurrent calls on the
# same object corrupt state. (Wiring the chat path onto this same lock for
# full chat/agent mutual-exclusion is a follow-up — see docs.)
# Verrou de génération : on réutilise le verrou PARTAGÉ défini dans model.py
# (même objet pour chat + agent). L'agent le prend toujours ; le chat le prend
# si le flag agent_chat_shared_lock_enabled est ON. Pas de cycle d'import :
# model.py n'importe pas ce module.
from rune.genlock import GENERATION_LOCK as _INPROCESS_LOCK


@runtime_checkable
class Worker(Protocol):
    name: str
    needs_prefix: bool

    def available(self) -> bool: ...
    def generate(self, prompt: str, **kwargs) -> str: ...


@dataclass
class InProcessWorker:
    """Wraps the shared, steered ModelWrapper (the Rune core)."""

    model: object  # lythea.model.ModelWrapper
    name: str = "taelys-core"
    needs_prefix: bool = True

    def available(self) -> bool:
        return bool(getattr(self.model, "is_loaded", False))

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.available():
            return ""
        with _INPROCESS_LOCK:
            try:
                return self.model.generate(prompt, **kwargs) or ""
            except Exception:  # noqa: BLE001
                log.exception("InProcessWorker.generate failed")
                return ""


@dataclass
class OllamaWorker:
    """Calls a local Ollama daemon. Text-only, parallel-friendly, optional."""

    model_id: str
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 300.0
    needs_prefix: bool = False
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"ollama:{self.model_id}"

    def available(self) -> bool:
        try:
            import httpx

            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            if r.status_code != 200:
                return False
            tags = {m.get("name", "").split(":")[0] for m in r.json().get("models", [])}
            return self.model_id.split(":")[0] in tags or True
        except Exception:  # noqa: BLE001 — daemon down / not installed
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import httpx

            r = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": kwargs.get("temperature", 0.4)},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            return (r.json().get("message", {}) or {}).get("content", "") or ""
        except Exception:  # noqa: BLE001
            log.exception("OllamaWorker.generate failed (%s)", self.name)
            return ""


@dataclass
class WorkerPool:
    """Registry + routing. Routes a role to a concrete worker.

    Routing policy (Phase 1, simple):
    * a step that must carry the steered voice (``needs_prefix``) → always
      the in-process core;
    * otherwise → the first *available* auxiliary (Ollama) worker, else the
      in-process core as fallback.
    """

    core: InProcessWorker
    auxiliaries: list[Worker] = field(default_factory=list)

    def available_names(self) -> list[str]:
        names = [self.core.name] if self.core.available() else []
        names += [w.name for w in self.auxiliaries if w.available()]
        return names

    def pick(self, *, needs_prefix: bool) -> Worker:
        if needs_prefix:
            return self.core
        for w in self.auxiliaries:
            if w.available():
                return w
        return self.core

    @classmethod
    def from_settings(cls, model, settings=None) -> "WorkerPool":
        """Build the pool from settings; Ollama auxiliaries are opt-in.

        Reads ``agent_ollama_workers`` (a list of model ids) and
        ``agent_ollama_base_url`` from settings if present. Absent → core
        only. Never raises: a bad config yields a core-only pool.
        """
        core = InProcessWorker(model=model)
        aux: list[Worker] = []
        try:
            ids = list(getattr(settings, "agent_ollama_workers", []) or [])
            base = getattr(settings, "agent_ollama_base_url", "") or "http://127.0.0.1:11434"
            for mid in ids:
                aux.append(OllamaWorker(model_id=mid, base_url=base))
        except Exception:  # noqa: BLE001
            log.exception("WorkerPool.from_settings: bad Ollama config, core-only")
        return cls(core=core, auxiliaries=aux)
