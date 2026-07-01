"""RuneCortex — wrap Hippocampe Lythea avec les extensions Rune.

Ce module n'a pas vocation à réécrire Hippocampe (qui fait déjà 3500 lignes
de travail riche). Il le **wrap** en ajoutant les briques manquantes :

1. :class:`WorkingMemoryBuffer` (Core 4±1 chunks) injecté avant Phase B
2. :class:`AutoSkillStore` + :class:`SkillExtractor` — extraction post-succès
3. :class:`FailureMemory` + :class:`FailureAnalyzer` — analyse post-échec
4. :class:`TieredRetriever` — wrap le RetrievalPhase existant avec fallback strict
5. :class:`SubAgentSpawner` — délégation spatiale (subprocess isolé)
6. :class:`CronScheduler` — tâches de fond + consolidation post-run

Architecture
------------
RuneCortex délègue TOUT le travail cognitif à Hippocampe (composition).
Il intercepte juste :

- Avant Phase B : injecte WorkingMemoryBuffer + skills + anti-patterns
- Après génération : analyse succès/échec et met à jour AutoSkill/FailureMemory

Ça permet de garder les 42 000 lignes de Lythea intactes tout en ajoutant
les capacités agentiques de Rune.
"""
from __future__ import annotations

import logging
from typing import Any

from ..agents.cron import CronScheduler, CronTask
from ..agents.subagent import SubAgentConfig, SubAgentSpawner
from ..cognition.consolidation_rune import (
    ConsolidationConfig, ConsolidationScheduler,
)
from ..memory.auto_skill import AutoSkillStore, Skill, SkillExtractor
from ..memory.failure_memory import FailureAnalyzer, FailureMemory, FailurePattern
from ..memory.tiered_retriever import RetrievalResult, TieredRetriever
from ..memory.working_memory import WorkingMemoryBuffer, WorkingMemoryChunk

log = logging.getLogger("rune.cortex_ext.integration")


class RuneCortex:
    """Wrap un Hippocampe Lythea avec les extensions Rune.

    Composition plutôt qu'héritage — Hippocampe a une __init__ très
    lourde (SDM, MHN, KG, Chroma, model, etc.) qu'on ne veut pas
    dupliquer. On prend une instance déjà construite et on ajoute
    les briques Rune autour.

    Usage :

        from rune.hippocampe import Hippocampe
        from rune.cortex_ext.integration import RuneCortex

        base = Hippocampe(model=..., sdm=..., mhn=..., chroma=..., git=...)
        rune = RuneCortex(base)
        # rune.process_message(...) délègue à base.process_message
        # mais ajoute auto-skill + failure analysis + working memory
    """

    def __init__(
        self,
        hippocampe: Any,
        skills_dir: str = "data/skills",
        failures_dir: str = "data/failures",
        cron_dir: str = "data/cron",
        working_memory_capacity: int = 5,
        doubt_gate: float = 0.15,
        enable_subagent: bool = True,
        enable_cron: bool = True,
    ) -> None:
        self.hippocampe = hippocampe

        # Working memory (Core 4±1 chunks)
        self.working_memory = WorkingMemoryBuffer(
            capacity=working_memory_capacity
        )

        # Tiered retriever — wrap les backends Lythea existants
        self.tiered_retriever = TieredRetriever(
            working_memory=self.working_memory,
            sdm=getattr(hippocampe, "sdm", None),
            mhn=getattr(hippocampe, "mhn", None),
            kg=getattr(hippocampe, "kg", None),
            chroma=getattr(hippocampe, "retriever", None),
            doubt_gate=doubt_gate,
        )

        # AutoSkill + FailureMemory
        self.skills = AutoSkillStore(storage_dir=skills_dir)
        self.failures = FailureMemory(storage_dir=failures_dir)
        self.skill_extractor = SkillExtractor()
        self.failure_analyzer = FailureAnalyzer()

        # SubAgent spawner (désactivable pour tests)
        self.subagent_spawner: SubAgentSpawner | None = (
            SubAgentSpawner(SubAgentConfig()) if enable_subagent else None
        )

        # ── Trinity (Option A) — passe le model_id du Worker aux sous-agents ──
        # Si Trinity est activé côté parent (rune serve), on récupère le
        # model_id du Worker depuis le TrinityPool attaché à l'app Lythea,
        # et on le passe au SubAgentSpawner. Les sous-agents l'utiliseront
        # via HFModelWrapper (mode transformers in-process).
        #
        # Si Trinity n'est pas activé, _trinity_worker_model_id reste None
        # et les sous-agents tombent sur MockBackend (comportement par défaut).
        self._configure_trinity_worker()

        # Cron scheduler
        self.cron_scheduler: CronScheduler | None = (
            CronScheduler(
                storage_dir=cron_dir,
                subagent_runner=self._run_subagent_for_cron,
                consolidation_trigger=self._trigger_consolidation,
            ) if enable_cron else None
        )

        # Consolidation scheduler (wrap le ConsolidationPhase Lythea)
        self.consolidation_scheduler = ConsolidationScheduler(
            config=ConsolidationConfig(
                on_microsleep=self._run_microsleep,
                on_deep_sleep=self._run_deep_sleep,
            ),
            exchange_counter=lambda: self.hippocampe.exchange_count,
        )

        log.info(
            "RuneCortex initialized — skills=%d, failures=%d",
            self.skills.stats()["total"],
            self.failures.stats()["total"],
        )

    # ── API principale : wrap process_message ──────────────────────────

    def process_message(
        self,
        message: str,
        history: list[dict] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wrap Hippocampe.process_message avec les hooks Rune.

        Avant : injecte WorkingMemoryBuffer + skills + anti-patterns.
        Après : analyse succès/échec et met à jour AutoSkill/FailureMemory.
        """
        history = list(history or [])

        # ── Pre-process : lookup skills + anti-patterns ────────────────
        user_embedding = self._encode_safely(message)

        applicable_skills: list[Skill] = []
        if user_embedding:
            applicable_skills = self.skills.find_by_trigger_embedding(
                user_embedding, threshold=0.75, top_k=2
            )
            for skill in applicable_skills:
                self.working_memory.add(WorkingMemoryChunk(
                    kind="skill",
                    content=skill.to_markdown()[:500],
                    relevance=0.8,
                    metadata={"skill_id": skill.skill_id},
                ))

            anti_patterns = self.failures.find_by_embedding(
                user_embedding, threshold=0.65, top_k=2
            )
            if anti_patterns:
                warning_block = self.failures.as_warning_block(
                    user_embedding, top_k=2
                )
                if warning_block:
                    self.working_memory.add(WorkingMemoryChunk(
                        kind="system_note",
                        content=warning_block,
                        relevance=0.7,
                    ))

        # Ajoute le message utilisateur au Core
        self.working_memory.add(WorkingMemoryChunk(
            kind="user_message",
            content=message,
            relevance=1.0,
        ))

        # ── Délègue à Hippocampe ───────────────────────────────────────
        gen = self.hippocampe.process_message(
            message=message,
            history=history,
            **kwargs,
        )

        final_text = ""
        final_doubt = 0.5
        final_confidence_label = "certaine"
        web_used = False
        kg_hits = 0

        try:
            while True:
                event = next(gen)
                event_type = event.get("type", "")
                if event_type == "done":
                    final_text = event.get("text", "")
                    final_doubt = event.get("doubt_index", 0.5)
                    final_confidence_label = event.get(
                        "confidence_label", "certaine"
                    )
                    web_used = event.get("web_used", False)
                    kg_hits = event.get("kg_facts_count", 0)
                yield event
        except StopIteration as stop:
            if stop.value and isinstance(stop.value, dict):
                final_text = stop.value.get("text", final_text)

        # ── Post-process : AutoSkill / FailureMemory ───────────────────
        self._post_generation_analysis(
            user_message=message,
            assistant_response=final_text,
            doubt_index=final_doubt,
            confidence_label=final_confidence_label,
            web_used=web_used,
            kg_hits=kg_hits,
            user_embedding=user_embedding,
            applied_skill=applicable_skills[0] if applicable_skills else None,
        )

        # ── Consolidation opportuniste ─────────────────────────────────
        self.consolidation_scheduler.maybe_microsleep()

        # Clear le WorkingMemory pour le tour suivant
        self.working_memory.clear()

    # ── API utilitaire ────────────────────────────────────────────────

    def run_subagent(
        self, task: str, context: str = "", timeout: float = 120.0
    ) -> dict:
        """Lance un subagent isolé pour une tâche ponctuelle."""
        if self.subagent_spawner is None:
            return {"status": "error", "error": "subagent disabled"}
        result = self.subagent_spawner.run(task=task, context=context)
        return result.as_dict()

    def add_cron_task(
        self, task_id: str, schedule: str, action: str
    ) -> CronTask:
        """Ajoute une tâche cron."""
        if self.cron_scheduler is None:
            raise RuntimeError("cron scheduler disabled")
        task = CronTask(
            task_id=task_id, schedule=schedule, action=action,
        )
        self.cron_scheduler.add_task(task)
        return task

    def status(self) -> dict[str, Any]:
        """Snapshot complet pour /status."""
        base_status: dict[str, Any] = {}
        if hasattr(self.hippocampe, "v4_status"):
            try:
                base_status = self.hippocampe.v4_status()
            except Exception:
                pass
        return {
            "hippocampe": base_status,
            "exchange_count": getattr(self.hippocampe, "exchange_count", 0),
            "working_memory": self.working_memory.status(),
            "skills": self.skills.stats(),
            "failures": self.failures.stats(),
            "consolidation": self.consolidation_scheduler.status(),
            "cron": (
                self.cron_scheduler.status()
                if self.cron_scheduler else {"running": False, "task_count": 0}
            ),
        }

    # ── Internes ──────────────────────────────────────────────────────

    def _encode_safely(self, text: str) -> list[float]:
        """Calcule l'embedding via le model Lythea, sans crasher."""
        try:
            model = getattr(self.hippocampe, "model", None)
            if model is None or not getattr(model, "is_loaded", False):
                return []
            extractor = getattr(self.hippocampe, "entity_extractor", None)
            if extractor is not None:
                emb = extractor.encode(text)
                if emb is not None:
                    try:
                        return emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    except Exception:
                        return []
            if hasattr(model, "encode"):
                emb = model.encode(text)
                if emb is not None:
                    return list(emb) if not hasattr(emb, "tolist") else emb.tolist()
            return []
        except Exception as exc:
            log.debug("Encode failed: %s", exc)
            return []

    def _post_generation_analysis(
        self,
        user_message: str,
        assistant_response: str,
        doubt_index: float,
        confidence_label: str,
        web_used: bool,
        kg_hits: int,
        user_embedding: list[float],
        applied_skill: Skill | None,
    ) -> None:
        """Analyse post-génération : extrait un skill ou un failure pattern."""
        if not assistant_response or len(assistant_response) < 20:
            return

        success = confidence_label in {"très_certaine", "certaine"} and doubt_index < 0.4

        if success:
            skill = self.skill_extractor.extract(
                user_message=user_message,
                assistant_response=assistant_response,
                verifier_ok=True,
                doubt_index=doubt_index,
                confidence_label=confidence_label,
                trigger_embedding=user_embedding,
                source_episode_id=f"ep_{self.hippocampe.exchange_count}",
            )
            if skill is not None:
                self.skills.add(skill)
                if applied_skill:
                    self.skills.record_success(applied_skill.skill_id)
        else:
            pattern = self.failure_analyzer.analyze(
                context=user_message[:200],
                attempted_action=assistant_response[:200],
                verifier_reasons=[f"Confiance trop faible: {confidence_label}"],
                user_message=user_message,
                assistant_response=assistant_response,
                context_embedding=user_embedding,
            )
            if pattern is not None:
                self.failures.add(pattern)
                if applied_skill:
                    self.skills.record_failure(
                        applied_skill.skill_id,
                        anti_pattern=pattern.correction,
                    )

    def _run_subagent_for_cron(
        self, action: str, context: str
    ) -> dict:
        """Runner pour CronScheduler — utilise le subagent spawner."""
        if self.subagent_spawner is None:
            return {"status": "error", "error": "subagent disabled"}
        result = self.subagent_spawner.run(task=action, context=context)
        return result.as_dict()

    def _trigger_consolidation(self) -> dict:
        """Trigger consolidation post-cron — wrap ConsolidationPhase Lythea."""
        try:
            consolidation_phase = getattr(
                self.hippocampe, "consolidation_phase", None
            )
            if consolidation_phase is None:
                return {"status": "skipped", "reason": "no consolidation phase"}
            if hasattr(consolidation_phase, "_trigger_microsleep"):
                consolidation_phase._trigger_microsleep()
                return {"status": "ok"}
            return {"status": "skipped", "reason": "no _trigger_microsleep method"}
        except Exception as exc:
            log.warning("Consolidation trigger failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _run_microsleep(self) -> dict:
        """Callback pour ConsolidationScheduler.on_microsleep."""
        return self._trigger_consolidation()

    def _run_deep_sleep(self) -> dict:
        """Callback pour ConsolidationScheduler.on_deep_sleep."""
        try:
            if hasattr(self.hippocampe, "deep_sleep"):
                self.hippocampe.deep_sleep()
                return {"status": "ok"}
            return {"status": "skipped", "reason": "no deep_sleep method"}
        except Exception as exc:
            log.warning("Deep sleep failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ── Trinity (Option A) ────────────────────────────────────────────

    def _configure_trinity_worker(self) -> None:
        """Récupère le model_id du Worker Trinity et le passe au SubAgentSpawner.

        Cherche le TrinityPool sur l'app Lythea (attaché par le stage
        ``_stage_trinity`` du boot). Si trouvé et activé, récupère le
        model_id du Worker et l'injecte dans le SubAgentSpawner pour
        que tous les sous-agents l'utilisent.

        Si Trinity n'est pas activé ou pas chargé, ne fait rien — les
        sous-agents resteront en mode MockBackend (ou transformers si
        RUNE_SUBAGENT_MODEL_ID est set dans l'env).
        """
        if self.subagent_spawner is None:
            return

        # Cherche le TrinityPool sur hippocampe ou sur l'app Lythea
        trinity_pool = None
        # Cas 1 : TrinityPool attaché directement à l'hippocampe
        if hasattr(self.hippocampe, "trinity_pool"):
            trinity_pool = self.hippocampe.trinity_pool
        # Cas 2 : TrinityPool sur l'app Lythea (via app.trinity_pool)
        if trinity_pool is None and hasattr(self.hippocampe, "app"):
            trinity_pool = getattr(self.hippocampe.app, "trinity_pool", None)

        if trinity_pool is None:
            log.debug("No TrinityPool found — subagents use MockBackend")
            return

        # Vérifie que Trinity est activé et chargé
        if not getattr(trinity_pool, "config", None) or not trinity_pool.config.enabled:
            log.debug("Trinity disabled — subagents use MockBackend")
            return

        # Récupère le model_id du Worker depuis la config (pas depuis le
        # modèle chargé — le subprocess doit charger son propre modèle)
        worker_model_id = trinity_pool.config.worker.model_id
        if worker_model_id:
            self.subagent_spawner.set_trinity_worker_model_id(worker_model_id)
            log.info(
                "Trinity Worker model_id '%s' configured for subagents",
                worker_model_id,
            )
