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

        # ── Jugement LLM pour l'extraction de skills ──────────────────
        # Sans callback, SkillExtractor tombe sur l'heuristique naïve
        # (premières phrases de la réponse = « approche »), qui produit
        # des skills grossiers. On branche ici le modèle déjà chargé
        # comme juge : il décide si l'échange mérite un skill et, si oui,
        # en extrait un pattern structuré. Le prompt gère lui-même le
        # cas « trop trivial » via {"skip": true}. C'est le second rideau
        # après le filtre heuristique _looks_conversational().
        self.skill_extractor.set_llm_callback(self._llm_skill_judge)

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
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Wrap Hippocampe.process_message avec les hooks Rune.

        Signature compatible avec Lythea : accepte les args positionnels
        supplémentaires (images, cancelled, last_message_ts, ...) et les
        passe tels quels à l'hippocampe sous-jacent.

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

        # ── Délègue à Hippocampe (version originale, pas le wrapper) ───
        # Important : self.hippocampe.process_message est maintenant le
        # wrapper (RuneCortex.process_message). Si on l'appelle, on crée
        # une récursion infinie. On doit appeler l'original stocké dans
        # _original_process (par server/app.py lors du monkey-patch).
        original_process = getattr(self, "_original_process", None)
        if original_process is not None:
            # Appelle l'original avec tous les args positionnels + kwargs
            gen = original_process(message, history, *args, **kwargs)
        else:
            # Fallback : pas de monkey-patch (CLI mode, pas API)
            gen = self.hippocampe.process_message(
                message=message,
                history=history,
                *args,
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
                    # Les champs sont normalement DANS event["data"]
                    # (format réel de hippocampe.process_message). On lit
                    # data en priorité, avec repli sur le niveau racine pour
                    # rester compatible avec d'éventuels producteurs à plat.
                    # Le texte final est data["final_text"] (pas "text").
                    # Sans ça, final_text restait vide et l'AutoSkill
                    # extraction ne se déclenchait jamais (bug historique).
                    d = event.get("data")
                    if not isinstance(d, dict):
                        d = event  # repli : champs à plat sur l'event
                    final_text = (
                        d.get("final_text")
                        or d.get("text")
                        or event.get("final_text")
                        or event.get("text", "")
                    )
                    final_doubt = d.get(
                        "doubt_index", event.get("doubt_index", 0.5)
                    )
                    final_confidence_label = d.get(
                        "confidence_label",
                        event.get("confidence_label", "certaine"),
                    )
                    web_used = d.get("web_used", event.get("web_used", False))
                    kg_hits = d.get(
                        "kg_facts_count", event.get("kg_facts_count", 0)
                    )
                yield event
        except StopIteration as stop:
            if stop.value and isinstance(stop.value, dict):
                sv = stop.value
                sv_data = sv.get("data", sv) if isinstance(sv, dict) else {}
                if isinstance(sv_data, dict):
                    final_text = (
                        sv_data.get("final_text")
                        or sv_data.get("text")
                        or final_text
                    )

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
        self,
        task: str,
        context: str = "",
        timeout: float = 120.0,
        use_hot_context: bool = True,
        blackboard_path: str | None = None,
        blackboard_section: str | None = None,
    ) -> dict:
        """Lance un subagent isolé pour une tâche ponctuelle.

        Parameters
        ----------
        task : str
            Mission du sous-agent.
        context : str
            Contexte froid additionnel.
        timeout : float
            Timeout en secondes.
        use_hot_context : bool
            Si True (défaut), construit automatiquement un HotContext
            à partir des mémoires du parent (RAG chunks + skills +
            anti-patterns) et le passe au sous-agent. Désactive si False
            (tests, mode dégradé).
        blackboard_path : str | None
            Chemin vers blackboard.json partagé. Si fourni, le sous-agent
            lit les sections des autres agents + contract, et écrit dans
            sa propre section.
        blackboard_section : str | None
            Nom de la section du blackboard pour ce sous-agent.
        """
        if self.subagent_spawner is None:
            return {"status": "error", "error": "subagent disabled"}

        # ── Construit le HotContext (Solution A — mémoire partagée) ────
        hot_context_dict: dict | None = None
        if use_hot_context:
            try:
                from ..agents.hot_context import HotContextSerializer
                serializer = HotContextSerializer(
                    tiered_retriever=self.tiered_retriever,
                    skills_store=self.skills,
                    failures_store=self.failures,
                    kg=getattr(self.hippocampe, "kg", None),
                    embed_fn=self._embed_fn,
                )
                hot_context = serializer.build(task)
                hot_context_dict = hot_context.as_dict()
            except Exception as exc:
                log.warning("Failed to build HotContext: %s", exc)
                hot_context_dict = None

        # ── Lance le sous-agent ────────────────────────────────────────
        result = self.subagent_spawner.run(
            task=task,
            context=context,
            hot_context=hot_context_dict,
            blackboard_path=blackboard_path,
            blackboard_section=blackboard_section,
        )
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

    def _llm_skill_judge(self, prompt: str) -> dict | None:
        """Callback LLM pour SkillExtractor : juge + extrait un skill.

        Utilise le modèle déjà chargé (self.hippocampe.model) pour
        décider si un échange mérite un skill et, si oui, produire le
        JSON structuré (trigger / approach / validation / anti_patterns).
        Le prompt (construit par SkillExtractor._extract_via_llm) demande
        explicitement de renvoyer {"skip": true} si l'épisode est trivial.

        Retourne le dict parsé, ou None (skip / erreur / pas de modèle).
        Ne lève jamais : l'extraction de skill est un bonus, elle ne doit
        pas casser le flux de chat.
        """
        model = getattr(self.hippocampe, "model", None)
        if model is None or not getattr(model, "is_loaded", False):
            # Pas de modèle chargé → on signale l'indisponibilité pour que
            # SkillExtractor retombe sur l'heuristique (au lieu de traiter
            # ça comme un « skip » qui empêcherait toute extraction).
            from rune.memory.auto_skill import _LLM_UNAVAILABLE
            return _LLM_UNAVAILABLE
        try:
            raw = model.generate(
                prompt,
                max_new_tokens=300,
                temperature=0.2,  # déterministe : on veut du JSON stable
            )
        except Exception:
            log.debug("LLM skill judge: génération échouée", exc_info=True)
            return None

        # Parse le JSON, tolérant aux fences markdown et au texte autour.
        return self._parse_skill_json(raw)

    @staticmethod
    def _parse_skill_json(raw: str) -> dict | None:
        """Extrait le premier objet JSON d'une réponse LLM.

        Tolère : fences ```json, préambule, texte après. Retourne None
        si rien d'exploitable ou si le JSON signale {"skip": true}.
        """
        import json
        import re

        if not raw or not raw.strip():
            return None
        text = raw.strip()
        # Retire les fences markdown éventuelles.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Isole le premier bloc {...} équilibré si du texte l'entoure.
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        try:
            obj = json.loads(text[start:end])
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        if obj.get("skip"):
            return None
        # Un skill valide doit au minimum avoir une approche non vide.
        if not obj.get("approach"):
            return None
        return obj

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

    @property
    def _embed_fn(self) -> Any:
        """Fonction d'embedding pour le HotContextSerializer.

        Utilise l'entity_extractor (GLiNER) de l'hippocampe si dispo.
        Sinon retourne None (le HotContext sera limité aux skills/failures
        sans similarity search).
        """
        try:
            extractor = getattr(self.hippocampe, "entity_extractor", None)
            if extractor is None:
                return None
            def _embed(text: str):
                emb = extractor.encode(text)
                if emb is None:
                    return None
                # torch.Tensor → list[float]
                if hasattr(emb, "tolist"):
                    return emb.tolist()
                return list(emb)
            return _embed
        except Exception:
            return None
