"""Runtime subagent — script exécuté dans le subprocess.

Ce module est importé par le script généré par SubAgentSpawner. Il
définit :class:`SubAgentRuntime` qui prend le payload stdin et exécute
la mission.

Intégration Trinity (Option A)
-----------------------------
Si le payload contient ``model_id``, le runtime charge ce modèle via
``HFModelWrapper.load()`` (backend transformers in-process). C'est le
cas quand le parent a Trinity activé — il passe le model_id du Worker
au subprocess.

Si pas de ``model_id`` dans le payload, le runtime utilise MockBackend
(utile pour tests et dev sans GPU).

Pour override manuel : set env ``RUNE_SUBAGENT_MODEL_ID=<model_id>``.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("rune.agents.runtime")


class SubAgentRuntime:
    """Runtime exécuté dans le subprocess.

    Parameters
    ----------
    payload : dict
        Contient au minimum ``task`` et ``context``. Optionnellement
        ``model_id`` pour activer le backend transformers.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.task: str = payload.get("task", "")
        self.context: str = payload.get("context", "")
        # model_id peut venir du payload (Trinity) ou de l'env (override)
        self.model_id: str | None = (
            payload.get("model_id")
            or os.environ.get("RUNE_SUBAGENT_MODEL_ID")
        )

    def run(self) -> dict:
        """Exécute la mission et retourne le dict résultat."""
        try:
            # ── Choix du backend ──────────────────────────────────────
            # 1. Si model_id fourni (Trinity Option A) → transformers backend
            # 2. Sinon → MockBackend (tests/dev sans GPU)
            if self.model_id:
                log.info(
                    "SubAgent loading model %s (Trinity Worker)",
                    self.model_id,
                )
                backend = self._load_transformers_backend(self.model_id)
            else:
                from ..perf.backend import get_backend
                backend = get_backend({"backend": "mock"})

            from ..perf.backend import GenerationConfig

            messages = [
                {"role": "system", "content": (
                    "Tu es un sous-agent Rune. Exécute la mission "
                    "qui t'est confiée de façon autonome et concise."
                )},
                {"role": "user", "content": self.task},
            ]
            if self.context:
                messages.insert(1, {
                    "role": "system",
                    "content": f"[CONTEXTE]\n{self.context}"
                })

            config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.4,  # plus factuel pour subagent
                return_entropies=False,
                return_hidden_states=False,
            )
            result = backend.generate(messages, config)

            return {
                "status": "ok" if result.finish_reason != "error" else "error",
                "result": result.text,
                "artifacts": [],
                "error": result.meta.get("error"),
                "model_id": self.model_id,
            }

        except Exception as exc:
            log.exception("SubAgentRuntime failed")
            return {
                "status": "error",
                "result": "",
                "artifacts": [],
                "error": str(exc),
                "model_id": self.model_id,
            }

    def _load_transformers_backend(self, model_id: str) -> Any:
        """Charge un backend transformers in-process pour le model_id donné.

        Utilise HFModelWrapper de Lythea (gère NF4, hooks, etc.) si
        disponible. Sinon, fallback sur TransformersBackend de rune.perf.

        Le modèle est chargé en NF4 par défaut pour économiser la VRAM
        (les sous-agents n'ont pas besoin de hooks mémoire — ils sont
        éphémères et isolés).
        """
        try:
            # Tente d'abord HFModelWrapper (Lythea) — gère le catalogue,
            # sampling profiles, etc.
            from ..model import HFModelWrapper
            wrapper = HFModelWrapper()
            ok = wrapper.load(model_id)
            if not ok:
                raise RuntimeError(f"HFModelWrapper.load({model_id}) returned False")

            # Adapter HFModelWrapper vers l'interface ModelBackend
            return _HFModelWrapperAdapter(wrapper)
        except ImportError:
            # Fallback sur TransformersBackend (rune.perf)
            log.warning(
                "HFModelWrapper not available — falling back to "
                "TransformersBackend"
            )
            from ..perf.backend import get_backend
            return get_backend({
                "backend": "transformers",
                "model_id": model_id,
                "quant_4bit": True,
            })
        except Exception as exc:
            log.warning(
                "HFModelWrapper load failed (%s) — falling back to "
                "MockBackend. Set RUNE_SUBAGENT_MODEL_ID='' to silence.",
                exc,
            )
            from ..perf.backend import get_backend
            return get_backend({"backend": "mock"})


class _HFModelWrapperAdapter:
    """Adapte HFModelWrapper (Lythea) vers l'interface ModelBackend (rune.perf).

    HFModelWrapper a sa propre API (load/unload/generate avec kwargs
    Lythea). On wrap pour exposer l'interface GenerationConfig/GenerationResult
    que le SubAgentRuntime attend.
    """

    def __init__(self, wrapper: Any) -> None:
        self.wrapper = wrapper

    @property
    def name(self) -> str:
        return "hf_model_wrapper"

    def generate(
        self,
        messages: list[dict[str, str]],
        config: Any,
    ) -> Any:
        """Génère via HFModelWrapper, retourne un GenerationResult-compatible."""
        from ..perf.backend import GenerationResult
        import time as _time

        start = _time.time()
        try:
            # HFModelWrapper.generate attend (messages, **kwargs)
            # Le contrat exact varie selon Lythea — on tente les kwargs courants
            kwargs: dict[str, Any] = {
                "max_new_tokens": getattr(config, "max_new_tokens", 512),
                "temperature": getattr(config, "temperature", 0.4),
                "do_sample": True,
            }
            # Appel — HFModelWrapper peut exiger un stream=True/False
            text = self.wrapper.generate(messages, **kwargs)
            if isinstance(text, dict):
                text = text.get("text", str(text))
            elif not isinstance(text, str):
                text = str(text)

            return GenerationResult(
                text=text,
                raw_text=text,
                finish_reason="stop",
                elapsed_sec=_time.time() - start,
                meta={"backend": "hf_model_wrapper"},
            )
        except Exception as exc:
            log.exception("HFModelWrapper generate failed")
            return GenerationResult(
                finish_reason="error",
                elapsed_sec=_time.time() - start,
                meta={"error": str(exc)},
            )
