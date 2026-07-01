"""CognitiveLoop — orchestrateur principal (allégé d'hippocampe.py).

Héritage Lythea
---------------
Lythea v5 a un hippocampe.py de 3500 lignes avec 5 phases (A-E), des
hooks multiples, et la gestion des modules V4 expérimentaux. Ici on
refait un orchestrateur minimal qui préserve l'essentiel :

- Phase A : encode + compute surprise input
- Phase B : retrieve (TieredRetriever) + assemble context
- Phase C : generate + compute doubt output
- Phase D : metacognition + skill extract + failure analyze
- Phase E : consolidate (différé, via ConsolidationScheduler)

On supprime : timeline, planning, inhibition, predictive_coding, deep
reasoning chain. Ces modules restent disponibles dans Lythea v5 si on
veut les réintégrer plus tard. Le cœur cognitif (surprise, retrieval,
generation, metacognition) est préservé.

Ajouts vs Lythea
----------------
- AutoSkill extraction post-succès (avec garde-fous métacognitifs)
- FailureMemory analyse post-échec
- TieredRetriever strict (Core → SDM → MHN → KG → Chroma)
- WorkingMemoryBuffer borné (4±1 chunks)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..memory.auto_skill import AutoSkillStore, Skill, SkillExtractor
from ..memory.failure_memory import FailureAnalyzer, FailureMemory
from ..memory.tiered_retriever import RetrievalResult, TieredRetriever
from ..memory.working_memory import WorkingMemoryBuffer, WorkingMemoryChunk
from ..perf.backend import GenerationConfig, GenerationResult, ModelBackend
from .metacognition import Metacognition, MetacognitiveDecision
from .surprise import SurpriseMeter, SurpriseSignals

log = logging.getLogger("rune.cognition.loop")


# ── System prompt (headless, plus court que Lythea) ───────────────────
# On garde l'essentiel : identité, ton FR sobre, citation des sources.
# Pas de splash UI, pas de badges — c'est un agent CLI/API.

SYSTEM_PROMPT = """Tu es Rune, un assistant IA cognitif local.
Tu réponds en français, de façon concise et structurée.
Quand tu utilises une source web, cite-la par [N] dans ta réponse.
Si tu n'es pas sûr, dis-le explicitement plutôt que d'inventer.
Préserve la voix de l'assistant : sobre, factuelle, sans flatterie.
"""


@dataclass
class TurnResult:
    """Résultat d'un tour cognitif complet."""
    text: str = ""
    raw_text: str = ""
    surprise: SurpriseSignals = field(default_factory=SurpriseSignals)
    metacognition: MetacognitiveDecision | None = None
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    skill_applied: Skill | None = None
    skill_extracted: Skill | None = None
    failure_analyzed: bool = False
    elapsed_sec: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "surprise": self.surprise.as_dict(),
            "metacognition": (
                self.metacognition.as_dict() if self.metacognition else None
            ),
            "retrieval": self.retrieval.as_dict(),
            "skill_applied": (
                {"id": self.skill_applied.skill_id,
                 "trigger": self.skill_applied.trigger}
                if self.skill_applied else None
            ),
            "skill_extracted": (
                {"id": self.skill_extracted.skill_id,
                 "trigger": self.skill_extracted.trigger}
                if self.skill_extracted else None
            ),
            "failure_analyzed": self.failure_analyzed,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "error": self.error,
        }


class CognitiveLoop:
    """Orchestrateur cognitif — remplace hippocampe.py en plus léger.

    Parameters
    ----------
    backend : ModelBackend
        Backend modèle (Mock ou Transformers).
    working_memory : WorkingMemoryBuffer
        Tampon Core du TieredRetriever.
    retriever : TieredRetriever
        Retriever hiérarchisé.
    skills : AutoSkillStore
        Magasin de compétences auto-apprises.
    failures : FailureMemory
        Mémoire des échecs (anti-patterns).
    surprise_meter : SurpriseMeter
        Calcul de surprise composite.
    metacognition : Metacognition
        Auto-monitoring de la confiance.
    skill_extractor : SkillExtractor
        Extracteur de skills post-succès.
    failure_analyzer : FailureAnalyzer
        Analyseur d'échecs.
    system_prompt : str
        Prompt système de base.
    """

    def __init__(
        self,
        backend: ModelBackend,
        working_memory: WorkingMemoryBuffer,
        retriever: TieredRetriever,
        skills: AutoSkillStore,
        failures: FailureMemory,
        surprise_meter: SurpriseMeter,
        metacognition: Metacognition,
        skill_extractor: SkillExtractor | None = None,
        failure_analyzer: FailureAnalyzer | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.backend = backend
        self.working_memory = working_memory
        self.retriever = retriever
        self.skills = skills
        self.failures = failures
        self.surprise_meter = surprise_meter
        self.metacognition = metacognition
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.failure_analyzer = failure_analyzer or FailureAnalyzer()
        self.system_prompt = system_prompt

        # Compteurs de session
        self.exchange_count: int = 0
        self.last_activity: float = time.time()

    # ── API publique ──────────────────────────────────────────────────

    def process(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        config: GenerationConfig | None = None,
    ) -> TurnResult:
        """Traite un message utilisateur — point d'entrée principal."""
        start = time.time()
        result = TurnResult()
        gen_config = config or GenerationConfig()
        hist = list(history or [])

        try:
            # ── Phase A : encode + surprise input ──────────────────────
            user_embedding = self.backend.encode(user_message)
            input_surprise = self.surprise_meter.compute_input_surprise(
                user_entropies=None,  # pas d'entropie sur l'input ici
                mhn_match_score=None,
                sdm_prediction_error=None,
                chroma_similarity=None,
            )

            # Ajoute le message utilisateur au Core
            self.working_memory.add(WorkingMemoryChunk(
                kind="user_message",
                content=user_message,
                relevance=1.0,
                metadata={"turn": self.exchange_count},
            ))

            # ── Phase A.bis : lookup skills + anti-patterns ────────────
            applicable_skills = self.skills.find_by_trigger_embedding(
                user_embedding, threshold=0.75, top_k=2
            )
            if applicable_skills:
                result.skill_applied = applicable_skills[0]
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
                        metadata={"kind": "anti_pattern"},
                    ))

            # ── Phase B : retrieval hiérarchisé ────────────────────────
            # doubt_index initial = 1 - input_surprise.confidence (approx)
            # Si input très surprenant → doute élevé → on descend plus bas
            initial_doubt = input_surprise.global_surprise
            retrieval = self.retriever.retrieve(
                query=user_message,
                query_embedding=user_embedding,
                doubt_index=initial_doubt,
                context={},
            )
            result.retrieval = retrieval
            result.surprise = input_surprise

            # ── Phase C : generation ───────────────────────────────────
            context_block = self.working_memory.as_prompt_block(max_chars=4000)
            messages = self._assemble_messages(
                user_message, hist, context_block
            )

            gen_result = self.backend.generate(messages, gen_config)
            result.raw_text = gen_result.raw_text
            result.text = gen_result.text

            # ── Phase D : surprise output + metacognition ──────────────
            doubt, epistemic = self.surprise_meter.compute_output_doubt(
                out_entropies=gen_result.entropies,
                kg_hits=len([c for c in retrieval.chunks if c.kind == "kg_entity"]),
                web_used=False,  # TODO: brancher depuis le routeur web
                rag_coverage=min(1.0, retrieval.confidence),
            )
            result.surprise.doubt_index = doubt
            result.surprise.epistemic_label = epistemic

            meta_decision = self.metacognition.observe(
                doubt_index=doubt,
                epistemic_label=epistemic,
                web_used=False,
                kg_hits=len([c for c in retrieval.chunks if c.kind == "kg_entity"]),
                rag_coverage=min(1.0, retrieval.confidence),
            )
            result.metacognition = meta_decision

            # Applique hedge si activé
            if self.metacognition.apply_hedge and meta_decision.hedge_prefix:
                result.text = meta_decision.hedge_prefix + result.text

            # ── Phase D.bis : skill extract + failure analyze ──────────
            # Heuristique de succès : confiance ≥ "certaine"
            success = meta_decision.confidence_label in (
                "très_certaine", "certaine"
            )

            if success:
                skill = self.skill_extractor.extract(
                    user_message=user_message,
                    assistant_response=result.text,
                    verifier_ok=True,
                    doubt_index=doubt,
                    confidence_label=meta_decision.confidence_label,
                    trigger_embedding=user_embedding,
                    source_episode_id=f"ep_{self.exchange_count}",
                )
                if skill is not None:
                    self.skills.add(skill)
                    result.skill_extracted = skill
                    # Si un skill était appliqué, on enregistre le succès
                    if result.skill_applied:
                        self.skills.record_success(
                            result.skill_applied.skill_id
                        )
            else:
                # Échec : analyse
                pattern = self.failure_analyzer.analyze(
                    context=user_message[:200],
                    attempted_action=result.text[:200],
                    verifier_reasons=["Confiance trop faible"],
                    user_message=user_message,
                    assistant_response=result.text,
                    context_embedding=user_embedding,
                )
                if pattern is not None:
                    self.failures.add(pattern)
                    result.failure_analyzed = True
                # Si un skill était appliqué, on enregistre l'échec
                if result.skill_applied:
                    self.skills.record_failure(
                        result.skill_applied.skill_id,
                        anti_pattern=pattern.correction if pattern else None,
                    )

            # ── Phase E : post-tour ────────────────────────────────────
            self.exchange_count += 1
            self.last_activity = time.time()
            self.working_memory.clear()

            result.elapsed_sec = time.time() - start
            return result

        except Exception as exc:
            log.exception("CognitiveLoop.process failed")
            result.error = str(exc)
            result.elapsed_sec = time.time() - start
            return result

    def stream(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        config: GenerationConfig | None = None,
    ):
        """Version streaming de process — yield les tokens au fur et à mesure.

        Retourne (via StopIteration value) le TurnResult final.
        """
        gen_config = config or GenerationConfig()
        hist = list(history or [])

        # Phase A-B préparatoires (non-streaming)
        user_embedding = self.backend.encode(user_message)
        input_surprise = self.surprise_meter.compute_input_surprise()

        self.working_memory.add(WorkingMemoryChunk(
            kind="user_message",
            content=user_message,
            relevance=1.0,
        ))

        applicable_skills = self.skills.find_by_trigger_embedding(
            user_embedding, threshold=0.75, top_k=1
        )
        if applicable_skills:
            self.working_memory.add(WorkingMemoryChunk(
                kind="skill",
                content=applicable_skills[0].to_markdown()[:500],
                relevance=0.8,
            ))

        retrieval = self.retriever.retrieve(
            query=user_message,
            query_embedding=user_embedding,
            doubt_index=input_surprise.global_surprise,
        )

        context_block = self.working_memory.as_prompt_block(max_chars=4000)
        messages = self._assemble_messages(user_message, hist, context_block)

        # Stream
        start = time.time()
        gen = self.backend.stream_generate(messages, gen_config)
        try:
            while True:
                tok = next(gen)
                yield tok
        except StopIteration as stop:
            gen_result = stop.value

        # Post-traitement
        result = TurnResult(
            text=gen_result.text,
            raw_text=gen_result.raw_text,
            surprise=input_surprise,
            retrieval=retrieval,
            skill_applied=applicable_skills[0] if applicable_skills else None,
            elapsed_sec=time.time() - start,
        )
        doubt, epistemic = self.surprise_meter.compute_output_doubt(
            out_entropies=gen_result.entropies,
            kg_hits=len([c for c in retrieval.chunks if c.kind == "kg_entity"]),
        )
        result.surprise.doubt_index = doubt
        result.surprise.epistemic_label = epistemic
        result.metacognition = self.metacognition.observe(
            doubt_index=doubt,
            epistemic_label=epistemic,
            kg_hits=len([c for c in retrieval.chunks if c.kind == "kg_entity"]),
            rag_coverage=min(1.0, retrieval.confidence),
        )
        self.exchange_count += 1
        self.last_activity = time.time()
        self.working_memory.clear()
        return result

    # ── API utilitaire ────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset session state (pas la mémoire persistante)."""
        self.working_memory.clear()
        self.exchange_count = 0
        self.last_activity = time.time()

    def status(self) -> dict[str, Any]:
        """Snapshot pour /status endpoint."""
        return {
            "backend": self.backend.name,
            "model_hidden_dim": self.backend.hidden_dim,
            "model_n_layers": self.backend.n_layers,
            "is_thinking_model": self.backend.is_thinking_model,
            "exchange_count": self.exchange_count,
            "working_memory": self.working_memory.status(),
            "skills": self.skills.stats(),
            "failures": self.failures.stats(),
            "metacognition": self.metacognition.to_dict(),
        }

    # ── Internes ──────────────────────────────────────────────────────

    def _assemble_messages(
        self,
        user_message: str,
        history: list[dict[str, str]],
        context_block: str,
    ) -> list[dict[str, str]]:
        """Assemble la liste de messages pour le backend."""
        sys_content = self.system_prompt
        if context_block:
            sys_content += "\n\n[CONTEXTE COURANT]\n" + context_block
        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_content}
        ]
        # History (sans le tour courant)
        for msg in history[-10:]:  # 10 derniers tours max
            if msg.get("role") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages
