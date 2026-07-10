"""Runtime subagent — script exécuté dans le subprocess.

Ce module est importé par le script généré par SubAgentSpawner. Il
définit :class:`SubAgentRuntime` qui prend le payload stdin et exécute
la mission.

Intégrations
------------
1. **Trinity (Option A)** : Si le payload contient ``model_id``, le runtime
   charge ce modèle via ``HFModelWrapper.load()`` (backend transformers
   in-process). C'est le cas quand le parent a Trinity activé — il passe
   le model_id du Worker au subprocess.

2. **HotContext (mémoire partagée, Solution A)** : Si le payload contient
   ``hot_context``, le runtime l'injecte dans le system prompt. Le
   sous-agent a accès en lecture seule aux chunks RAG, skills et
   anti-patterns pertinents. L'écriture reste côté parent.

3. **Blackboard (Solution B)** : Si le payload contient ``blackboard_path``
   et ``blackboard_section``, le runtime ouvre le blackboard partagé,
   lit les sections des autres agents + le contract, et écrit dans sa
   propre section. Le parent re-load le blackboard au retour.

Si pas de ``model_id`` dans le payload, le runtime utilise MockBackend
(utile pour tests et dev sans GPU).
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
        Contient au minimum ``task`` et ``context``. Optionnellement :
        - ``model_id`` pour activer le backend transformers (Trinity)
        - ``hot_context`` pour le contexte mémoire (Solution A)
        - ``blackboard_path`` + ``blackboard_section`` pour le blackboard partagé (Solution B)
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.task: str = payload.get("task", "")
        self.context: str = payload.get("context", "")
        # model_id peut venir du payload (Trinity Option A) ou de l'env (override)
        self.model_id: str | None = (
            payload.get("model_id")
            or os.environ.get("RUNE_SUBAGENT_MODEL_ID")
        )
        # HotContext (Solution A — mémoire partagée lecture seule)
        self.hot_context: dict | None = payload.get("hot_context")
        # Blackboard (Solution B — partage fichier lecture + écriture)
        self.blackboard_path: str | None = payload.get("blackboard_path")
        self.blackboard_section: str | None = payload.get("blackboard_section")

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

            # ── Construit le system prompt avec hot_context + blackboard ──
            system_prompt = self._build_system_prompt()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.task},
            ]
            if self.context:
                messages.insert(1, {
                    "role": "system",
                    "content": f"[CONTEXTE FROID]\n{self.context}"
                })

            config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.4,  # plus factuel pour subagent
                return_entropies=False,
                return_hidden_states=False,
            )
            result = backend.generate(messages, config)

            # ── Écrit dans le blackboard si configuré ─────────────────
            blackboard_written = False
            if self.blackboard_path and self.blackboard_section:
                blackboard_written = self._write_to_blackboard(result.text)

            return {
                "status": "ok" if result.finish_reason != "error" else "error",
                "result": result.text,
                "artifacts": [],
                "error": result.meta.get("error"),
                "model_id": self.model_id,
                "hot_context_used": self.hot_context is not None,
                "blackboard_written": blackboard_written,
            }

        except Exception as exc:
            log.exception("SubAgentRuntime failed")
            return {
                "status": "error",
                "result": "",
                "artifacts": [],
                "error": str(exc),
                "model_id": self.model_id,
                "hot_context_used": self.hot_context is not None,
                "blackboard_written": False,
            }

    def _build_system_prompt(self) -> str:
        """Construit le system prompt avec hot_context + blackboard.

        Sections injectées (si présentes) :
        1. Prompt de base (sous-agent Rune)
        2. [CONTEXTE MÉMOIRE] — chunks RAG, skills, anti-patterns (HotContext)
        3. [BLACKBOARD] — sections des autres agents + contract
        """
        parts = [
            "Tu es un sous-agent Rune. Exécute la mission "
            "qui t'est confiée de façon autonome et concise."
        ]

        # HotContext (Solution A)
        if self.hot_context:
            hot_block = self._render_hot_context(self.hot_context)
            if hot_block:
                parts.append(hot_block)

        # Blackboard (Solution B) — lecture seule ici, l'écriture se fait après
        if self.blackboard_path and self.blackboard_section:
            bb_block = self._read_blackboard_context()
            if bb_block:
                parts.append(bb_block)

        return "\n\n".join(parts)

    def _render_hot_context(self, hot_context: dict) -> str:
        """Render le HotContext en bloc de prompt.

        Reprend le format de HotContext.as_prompt_block() mais en
        travaillant depuis le dict sérialisé (pas l'objet).
        """
        lines: list[str] = ["[CONTEXTE MÉMOIRE]"]

        # RAG chunks
        rag_chunks = hot_context.get("rag_chunks", [])
        if rag_chunks:
            lines.append("[RAG]")
            for chunk in rag_chunks[:5]:
                content = str(chunk.get("content", ""))[:500]
                source = chunk.get("kind", "unknown")
                lines.append(f"  ({source}) {content}")

        # Skills
        skills = hot_context.get("skills", [])
        if skills:
            lines.append("[SKILLS APPLICABLES]")
            for skill in skills[:3]:
                trigger = str(skill.get("trigger", ""))[:100]
                approach = " | ".join(skill.get("approach", []))[:300]
                lines.append(f"  trigger: {trigger}")
                lines.append(f"  approach: {approach}")

        # Anti-patterns
        anti_patterns = hot_context.get("anti_patterns", [])
        if anti_patterns:
            lines.append("[ANTI-PATTERNS À ÉVITER]")
            for ap in anti_patterns[:3]:
                context = str(ap.get("context", ""))[:100]
                correction = str(ap.get("correction", ""))[:200]
                lines.append(f"  ⚠️ {context} → préférer: {correction}")

        # KG entities
        kg_entities = hot_context.get("kg_entities", [])
        if kg_entities:
            lines.append("[ENTITÉS CONNUES]")
            for ent in kg_entities[:5]:
                name = str(ent.get("name", ""))[:50]
                etype = ent.get("type", "")
                summary = str(ent.get("summary", ""))[:200]
                lines.append(f"  {name} ({etype}): {summary}")

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _read_blackboard_context(self) -> str:
        """Lit le blackboard partagé et retourne un bloc de prompt.

        Utilise ``MissionBlackboard.render_for()`` qui fait déjà le boulot :
        retourne le contract + la section du sous-agent + un résumé des
        autres sections (lecture seule).

        Returns
        -------
        str
            Bloc de prompt formaté, ou "" si le blackboard est vide/inaccessible.
        """
        if not self.blackboard_path or not self.blackboard_section:
            return ""

        try:
            from pathlib import Path
            from ..agentic.blackboard import MissionBlackboard

            bb_path = Path(self.blackboard_path)
            if not bb_path.exists():
                return ""

            bb = MissionBlackboard.load(bb_path)
            # render_for(owner, peers=True) retourne le bloc formaté
            # avec la section du owner + résumé des autres
            rendered = bb.render_for(self.blackboard_section, peers=True)
            if not rendered or not rendered.strip():
                return ""
            return f"[BLACKBOARD PARTAGÉ]\n{rendered}"
        except Exception as exc:
            log.warning("Failed to read blackboard: %s", exc)
            return ""

    def _write_to_blackboard(self, result_text: str) -> bool:
        """Écrit le résultat dans sa section du blackboard.

        Écrit :
        - Le résultat comme "win" (succès)
        - Une note avec un résumé de ce qui a été fait
        - Le statut "done"

        Returns
        -------
        bool
            True si l'écriture a réussi, False sinon.
        """
        if not self.blackboard_path or not self.blackboard_section:
            return False

        try:
            from pathlib import Path
            from ..agentic.blackboard import MissionBlackboard

            bb_path = Path(self.blackboard_path)
            bb = MissionBlackboard.load(bb_path) if bb_path.exists() else MissionBlackboard(path=bb_path)

            # Écrit le résultat comme win + note + status
            bb.record_win(
                self.blackboard_section,
                f"Résultat: {result_text[:200]}",
            )
            bb.note(
                self.blackboard_section,
                f"Sous-agent a terminé la mission (model={self.model_id or 'mock'})",
            )
            bb.set_status(self.blackboard_section, "done")

            # Save atomique (MissionBlackboard.save() fait déjà .tmp + replace)
            # On force le path pour s'assurer qu'il save au bon endroit
            bb.path = bb_path
            bb.save()
            log.info(
                "Blackboard updated: section=%s, path=%s",
                self.blackboard_section, bb_path,
            )
            return True
        except Exception as exc:
            log.warning("Failed to write to blackboard: %s", exc)
            return False

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
            # ATTENTION : HFModelWrapper.load() retourne None (pas un bool)
            # et lève une exception en cas d'échec réel. On ne peut donc
            # PAS tester `if not wrapper.load(...)` — ça vaudrait toujours
            # True (not None) et ferait échouer un chargement réussi vers
            # MockBackend. On appelle load(), puis on vérifie l'état réel
            # via is_loaded.
            wrapper.load(model_id)
            if not getattr(wrapper, "is_loaded", False):
                raise RuntimeError(
                    f"HFModelWrapper.load({model_id}): modèle non chargé "
                    f"après load() (is_loaded=False)"
                )

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
            # HFModelWrapper.generate attend un PROMPT (str), pas une liste
            # de messages : il tokenize directement l'entrée. On aplatit
            # donc les messages [{role, content}, …] en une string via le
            # chat template du tokenizer (ajoute le prompt d'assistant).
            prompt = self._messages_to_prompt(messages)
            kwargs: dict[str, Any] = {
                "max_new_tokens": getattr(config, "max_new_tokens", 512),
                "temperature": getattr(config, "temperature", 0.4),
            }
            text = self.wrapper.generate(prompt, **kwargs)
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

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Aplatit une liste de messages en un prompt string.

        Utilise le chat template du tokenizer du wrapper (format natif du
        modèle, ajoute le tour d'assistant). Repli robuste sur une simple
        concaténation ``role: content`` si le template n'est pas
        disponible (tokenizer sans chat_template, ou erreur).
        """
        if isinstance(messages, str):
            return messages
        tok = getattr(self.wrapper, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception as exc:
                log.warning(
                    "apply_chat_template failed (%s) — repli concaténation",
                    exc,
                )
        # Repli : concaténation lisible + amorce d'assistant.
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)
