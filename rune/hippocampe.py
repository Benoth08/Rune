"""Hippocampe — cognitive orchestrator for Lythéa.

After the Étape 8 refactor this class is a thin orchestrator that
composes five cognitive phases:

* :class:`~rune.cognition.encoding.EncodingPhase` — text → latents
* :class:`~rune.cognition.storage.StoragePhase` — write SDM/KG, archive Chroma+MHN
* :class:`~rune.cognition.surprise.SurprisePhase` — composite surprise + doubt
* :class:`~rune.cognition.retrieval.RetrievalPhase` — KG + MHN + Chroma → RAG
* :class:`~rune.cognition.consolidation.ConsolidationPhase` — microsleep + deep sleep

Plus:

* :mod:`~rune.cognition.generation` — text-stream cleanup + two-pass reasoning
* :class:`~rune.web.WebAgent` — optional live-search augmentation
* :class:`~rune.model.ImageCaptioner` — optional image captioning

The public surface (``process_message``, ``reset_session``,
``deep_sleep``, ``memory_status``) is unchanged from the
pre-refactor version. The private helpers like ``_phase_a_learn``,
``_phase_b_rag``, ``_phase_c_assemble``, ``_compute_surprise``,
``_kg_identity_summary``, ``_post_generation``, ``_trigger_microsleep``,
``_microsleep``, ``_reset_inactivity_timer``, ``_generate_reasoning``
remain available as thin wrappers — their tests, debug paths, and
external callers continue to work without modification.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Generator

from rune.cognition.consolidation import ConsolidationPhase
from rune.cognition.encoding import EncodingPhase, EncodingResult  # noqa: F401
from rune.cognition.generation import (
    QUESTION_STARTS,
    ReasoningGenerator,
    mask_open_tags as _mask_open_tags_fn,
    strip_reasoning,
)
from rune.cognition.retrieval import RetrievalPhase
from rune.cognition.storage import StoragePhase
from rune.cognition.surprise import SurprisePhase
from rune.config import (
    ENTROPY_DOUBT_WINDOW,
    ENTROPY_THRESHOLD,
    MAX_HISTORY_TURNS,
    MAX_NEW_TOKENS,
    MICROSLEEP_INTERVAL,
    SYSTEM_PROMPT,
)
from rune.git_sync import GitSync
from rune.memory.kg import EntityExtractor, KnowledgeGraphStore
from rune.memory.mhn import ModernHopfieldNetwork
from rune.memory.retrieval import HybridRetriever
from rune.memory.salience import SalienceFilter
from rune.memory.sdm import SparseDistributedMemory
from rune.model import HFModelWrapper, ImageCaptioner
from rune.temporal import TemporalContext
from rune.web import WebAgent, WebTriggerPolicy

log = logging.getLogger("rune.hippocampe")


# Re-export from cognition.generation so external code that imports
# ``from rune.hippocampe import strip_reasoning`` keeps working.
__all__ = ["Hippocampe", "strip_reasoning"]


# ── Prompt-assembly tunables ───────────────────────────────────────────
# Rough char-budget for the model context. ~1 token ≈ 4 chars, so
# 24000 chars ≈ 6000 tokens reserved for the rolling history.
_PROMPT_CHAR_BUDGET: int = 24000
_PROMPT_HISTORY_FLOOR: int = 2000

# Streaming output: emit an entropy SSE event every N tokens. 5
# is empirically a good balance between UI responsiveness and
# event volume.
_ENTROPY_EMIT_EVERY: int = 5

# Output-side SDM write: low-entropy tokens of the model's *own*
# output are committed to SDM at reduced strength (the model is
# learning from its confident utterances).
_OUTPUT_SDM_STRENGTH_FACTOR: float = 0.3


def _mask_open_tags(text: str) -> str:
    """Backwards-compatible wrapper for ``cognition.generation.mask_open_tags``.

    Kept under the legacy underscore-prefixed name because external
    debug code may import it.
    """
    return _mask_open_tags_fn(text)


class Hippocampe:
    """Central cognitive orchestrator.

    Parameters
    ----------
    model
        The language model wrapper.
    sdm
        Working memory.
    mhn
        Episodic memory.
    chroma_collection
        ChromaDB collection handle.
    git
        Git sync helper.
    kg
        Knowledge graph store. Created if ``None``.
    entity_extractor
        GLiNER entity extractor. Created if ``None``.
    web_policy
        Web search trigger policy. Created if ``None``.
    retriever
        Hybrid retriever over Chroma. Optional.
    """

    def __init__(
        self,
        model: HFModelWrapper,
        sdm: SparseDistributedMemory,
        mhn: ModernHopfieldNetwork,
        chroma_collection: Any,
        git: GitSync,
        kg: KnowledgeGraphStore | None = None,
        entity_extractor: EntityExtractor | None = None,
        web_policy: WebTriggerPolicy | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        # ── Memory backends ─────────────────────────────────────────
        self.model = model
        self.sdm = sdm
        self.mhn = mhn
        self.chroma = chroma_collection
        self.git = git
        self.kg = kg or KnowledgeGraphStore()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.web_policy = web_policy or WebTriggerPolicy()
        self.web_agent = WebAgent()
        self.salience = SalienceFilter()
        self.retriever = retriever
        self.captioner = ImageCaptioner()

        # V5.7.0 — Mémoire visuelle de travail (Visual Working Memory).
        # Buffer court-terme inspiré du visual sketchpad (Baddeley).
        # Capacité 3, décay 10 min, accès par récence / référence lexicale.
        # Utilisé par la Vision active (zoom cognitif) pour retrouver une
        # image envoyée il y a quelques tours.
        from rune.memory.visual_working_memory import (
            VisualWorkingMemory, VisualWorkingMemoryConfig,
        )
        self.visual_memory = VisualWorkingMemory(VisualWorkingMemoryConfig())

        # V5.7.0 — Flag d'incertitude perceptive de la réponse précédente.
        # Stocké entre tours pour proposer un zoom suggéré au tour suivant.
        self._last_perceptual_uncertainty: bool = False

        # V6.0.0-rc — Intégration MCP cognitive. Optionnels (None
        # accepté) pour rétro-compatibilité avec les tests qui
        # construisent un Hippocampe minimal. Quand présents, le
        # routeur peut router vers "mcp" et lire le workspace
        # automatiquement.
        self.mcp_manager: Any | None = None
        self.mcp_loop: Any | None = None

        # ── Session state ───────────────────────────────────────────
        self.exchange_count: int = 0
        self.last_activity: float = time.time()
        self.entropy_threshold: float = ENTROPY_THRESHOLD
        # Reasoning ON by default: for non-thinking models this runs the manual
        # reasoning pass (gated by `not is_thinking` at the call sites); thinking
        # models skip it and use their native internal thinking instead.
        self.reasoning_enabled: bool = True
        self.debug_mode: bool = False

        # Active sampling profile — applied at generation time.
        # Initialised to the global default; updated automatically when
        # a model is loaded (see ``server.routes.load_model``) by copying
        # the ``ModelSpec.sampling`` of the catalogue entry.
        # User overrides via ``/api/config/sampling`` are runtime-only
        # and are reset whenever a different model is loaded.
        from rune.config import DEFAULT_SAMPLING, SamplingProfile
        self.sampling_profile: SamplingProfile = DEFAULT_SAMPLING

        # ── Cognitive phases ────────────────────────────────────────
        self.encoding_phase = EncodingPhase(
            model=self.model,
            entity_extractor=self.entity_extractor,
            salience=self.salience,
        )
        self.storage_phase = StoragePhase(
            sdm=self.sdm,
            mhn=self.mhn,
            kg=self.kg,
            chroma=self.chroma,
            model=self.model,
            entity_extractor=self.entity_extractor,
        )
        self.surprise_phase = SurprisePhase(
            sdm=self.sdm,
            mhn=self.mhn,
            model=self.model,
            retriever=self.retriever,
        )
        self.retrieval_phase = RetrievalPhase(
            kg=self.kg,
            mhn=self.mhn,
            entity_extractor=self.entity_extractor,
            hybrid_retriever=self.retriever,
            llm_for_crag=self.model,
        )
        self.reasoning_generator = ReasoningGenerator(
            model=self.model, kg=self.kg,
        )
        # V4.1 — chaîne de raisonnement profond avec routeur adaptatif.
        # Un seul toggle « Raisonnement » active la chaîne. Le routeur
        # décide automatiquement 2 ou 4 étapes selon la complexité.
        # Les messages triviaux (≤6 mots, salutations) sont skippés
        # par le garde-fou dans _generate_reasoning.
        from rune.cognition.deep_reasoning import DeepReasoningChain
        self.deep_reasoning = DeepReasoningChain(
            model=self.model, kg=self.kg,
        )

        # V5.1 — Python executor : artefacts du dernier appel à exposer
        # à l'UI. Reset à chaque tour.
        self._last_python_code: str = ""
        self._last_python_result: dict = {}
        self._last_python_plots: list[str] = []

        # V5.4 — Procedural memory store (skills.md pattern).
        # Charge les patterns appris au démarrage et les sauve après
        # chaque microsleep (cf. ConsolidationPhase).
        try:
            from rune.config import PROCEDURAL_DIR
            from rune.memory.procedural import ProceduralStore
            self.procedural_store = ProceduralStore(PROCEDURAL_DIR)
            log.info(
                "Procedural store ready: %d procedures loaded",
                len(self.procedural_store.all()),
            )
        except Exception as exc:
            log.warning("Procedural store init failed: %s", exc)
            self.procedural_store = None

        # V5.4 — Rolling buffer of recent exchanges for procedural
        # extraction. Capped at 20 turns to bound memory. Microsleep
        # reads this buffer to extract reusable trigger→approach
        # patterns (cf. consolidation._refresh_procedural_memory).
        from collections import deque
        self._recent_exchanges_buffer: deque = deque(maxlen=20)

        # ── Microsleep manager + consolidation phase ─────────────────
        # See lythea/microsleep.py for the biologically-inspired engine
        # and lythea/cognition/consolidation.py for the orchestration
        # layer (locks, timers, persistence, git push).
        from rune.microsleep import MicrosleepConfig, MicrosleepManager
        from rune.settings import get_settings
        s = get_settings()
        self._microsleep_manager = MicrosleepManager(
            sdm=self.sdm,
            mhn=self.mhn,
            chroma_collection=self.chroma,
            config=MicrosleepConfig(
                ripple_trigger_count=s.ripple_trigger_count,
                ripple_surprise_threshold=s.ripple_surprise_threshold,
                ripple_boost_multiplier=s.ripple_boost_multiplier,
                replay_sequence_length=s.replay_sequence_length,
                replay_n_sequences=s.replay_n_sequences,
                compression_replay_threshold=s.compression_replay_threshold,
                # V4.1 affect-modulated consolidation. Only takes effect
                # when both flags are on; when either is off, microsleep
                # behaves exactly like V3.9.4.
                affect_modulates=(
                    s.affect_modulates_consolidation and s.enable_cognitive_state
                ),
                affect_ripple_arousal_threshold=s.affect_ripple_arousal_threshold,
                affect_consolidation_boost_factor=s.affect_consolidation_boost_factor,
            ),
        )
        self.consolidation_phase = ConsolidationPhase(
            sdm=self.sdm, mhn=self.mhn, kg=self.kg, git=self.git,
            microsleep_manager=self._microsleep_manager,
        )
        self.consolidation_phase.bind_exchange_counter(
            lambda: self.exchange_count,
        )
        # V5.2 — Provide the LLM to the consolidation phase so it can
        # generate GraphRAG community summaries during microsleep.
        # Setter-style attribute (no __init__ change) keeps backward
        # compat for any code that constructs ConsolidationPhase
        # directly without an LLM (tests, scripts).
        self.consolidation_phase._llm_for_community_summary = self.model

        # V5.4 — Hook procedural store + exchanges provider into the
        # consolidation phase. The store is created lazily on first
        # need (just below), the provider is a lambda that pulls
        # recent exchanges from the current session.
        self.consolidation_phase._procedural_store = (
            self.procedural_store
        )
        self.consolidation_phase._recent_exchanges_provider = (
            self._get_recent_exchanges
        )

        # ── V4 cognitive modules (all opt-in, default OFF) ───────────
        # Each module is created only if its master flag is on. On any
        # init failure we log and set the attribute to None so the
        # downstream hooks degrade gracefully (the hooks all check for
        # None before calling). The cascade V3.9 path is untouched.
        self._cognitive_state = None
        self._inhibition = None
        self._planning = None
        self._predictive_coding = None
        self._timeline = None
        # Cached prompt blocks consumed by Phase C; rebuilt each turn.
        self._planning_block: str = ""
        self._timeline_block: str = ""
        # Cached gating decision from Phase A.3 → consumed in Phase B.
        self._last_gating_decision = None
        self._pc_apply_gating: bool = bool(getattr(s, "pc_apply_gating", False))

        if getattr(s, "enable_cognitive_state", False):
            try:
                from pathlib import Path
                from rune.memory.cognitive_state import (
                    CognitiveState,
                    CognitiveStateConfig,
                )
                self._cognitive_state = CognitiveState(
                    config=CognitiveStateConfig(
                        decay_half_life_sec=s.affect_decay_half_life_sec,
                        contagion_max=s.affect_contagion_max,
                        inertia=s.affect_inertia,
                        reset_latch_turns=s.affect_reset_latch_turns,
                        detector=s.affect_detector,
                        user_known_threshold=s.user_model_known_threshold,
                    ),
                    storage_dir=Path("data/cognitive_state"),
                )
            except Exception:
                log.exception("V4: cognitive_state init failed — disabling")
                self._cognitive_state = None

        if getattr(s, "enable_inhibition", False):
            try:
                from rune.cognition.inhibition import (
                    InhibitionConfig,
                    InhibitionFilter,
                    parse_whitelist,
                )
                self._inhibition = InhibitionFilter(
                    InhibitionConfig(
                        n1_strict=s.inhibition_n1_strict,
                        n2_enabled=s.inhibition_n2_enabled,
                        n3_enabled=s.inhibition_n3_enabled,
                        default_action=s.inhibition_default_action,
                        domain_whitelist=parse_whitelist(s.inhibition_domain_whitelist),
                    )
                )
            except Exception:
                log.exception("V4: inhibition init failed — disabling")
                self._inhibition = None

        if getattr(s, "enable_planning", False):
            try:
                from pathlib import Path
                from rune.cognition.planning import (
                    GoalStack,
                    PlanGenerator,
                    PlanGeneratorConfig,
                    PlanningConfig,
                    PlanningPhase,
                )
                goal_stack = GoalStack(Path("data/goals/goals.json"))
                pg = PlanGenerator(
                    config=PlanGeneratorConfig(
                        use_llm=s.planning_use_llm,
                        max_steps=s.planning_max_steps,
                    )
                )
                self._planning = PlanningPhase(
                    config=PlanningConfig(
                        max_steps=s.planning_max_steps,
                        goal_stale_days=s.planning_goal_stale_days,
                        use_llm=s.planning_use_llm,
                        prompt_block_max_chars=s.planning_prompt_block_max_chars,
                    ),
                    goal_stack=goal_stack,
                    plan_generator=pg,
                )
            except Exception:
                log.exception("V4: planning init failed — disabling")
                self._planning = None

        if getattr(s, "enable_predictive_coding", False):
            try:
                from rune.cognition.predictive_coding import (
                    PredictiveCodingConfig,
                    PredictiveCodingPhase,
                )
                self._predictive_coding = PredictiveCodingPhase(
                    PredictiveCodingConfig(
                        history_size=s.pc_history_size,
                        cold_start_min=s.pc_cold_start_min,
                        ema_decay=s.pc_ema_decay,
                        low_threshold=s.pc_low_threshold,
                        high_threshold=s.pc_high_threshold,
                        confidence_cap=s.pc_confidence_cap,
                        gating_w_sdm=getattr(s, "pc_gating_w_sdm", 0.0),
                    )
                )
            except Exception:
                log.exception("V4: predictive_coding init failed — disabling")
                self._predictive_coding = None

        if getattr(s, "enable_timeline", False):
            try:
                from rune.cognition.timeline import (
                    TimelineConfig,
                    TimelineExtractor,
                )
                self._timeline = TimelineExtractor(
                    TimelineConfig(
                        max_events=s.timeline_max_events,
                        block_max_chars=s.timeline_block_max_chars,
                        render_min_confidence=s.timeline_render_min_confidence,
                        render_vague=s.timeline_render_vague,
                    )
                )
            except Exception:
                log.exception("V4: timeline init failed — disabling")
                self._timeline = None

        # V4.4 metacognition (separately initialized so the runtime
        # toggle endpoint can flip just this module without touching
        # the rest of the V4 stack).
        self._metacognition = None
        self._last_meta_decision = None
        if getattr(s, "enable_metacognition", False):
            try:
                from pathlib import Path as _Path
                from rune.cognition.metacognition import (
                    MetacognitionConfig,
                    MetacognitivePhase,
                )
                self._metacognition = MetacognitivePhase(
                    config=MetacognitionConfig(
                        very_high_doubt=s.metacog_very_high_doubt,
                        high_doubt=s.metacog_high_doubt,
                        low_doubt=s.metacog_low_doubt,
                        epistemic_boost_threshold=s.metacog_epistemic_boost_threshold,
                        apply_hedge=s.metacog_apply_hedge,
                        calibration_window=s.metacog_calibration_window,
                    ),
                    storage_path=_Path("data/metacognition/calibration.json"),
                )
            except Exception:
                log.exception("V4: metacognition init failed — disabling")
                self._metacognition = None

        # ── V3.9 cascade (opt-in) ───────────────────────────────────
        # When ``enable_cascade=True`` and a valid ``GOOGLE_API_KEY`` is
        # present, generation goes through the draft-then-refine
        # pipeline (Gemini draft → local synthesis). Otherwise this
        # attribute stays None and the streaming pipeline is unchanged.
        # See lythea/cognition/cascade.py for the orchestration logic.
        # IMPORTANT: must be the last statement in __init__ — the
        # cascade depends on the model being already wired in.
        self._cascade_key_override: str | None = None  # clé saisie à chaud (UI) — RAM only
        self._cascade = self._build_cascade_if_enabled(s)

    # ── V4 runtime toggle support ─────────────────────────────────────
    # The /api/config/v4 endpoints flip these modules on/off without
    # restarting Lythéa. Each rebuilder re-reads the live settings so
    # toggle order and intermediate edits stay coherent.

    def v4_status(self) -> dict[str, Any]:
        """Snapshot the on/off state of every V4 module + key sub-flags.

        Designed for the UI panel — never includes secrets, never
        raises. Missing modules surface as ``enabled=False``.
        """
        from rune.settings import get_settings
        s = get_settings()
        return {
            "cognitive_state": {
                "enabled": self._cognitive_state is not None,
                "contagion_max": s.affect_contagion_max,
                "decay_half_life_sec": s.affect_decay_half_life_sec,
                "detector": s.affect_detector,
            },
            "inhibition": {
                "enabled": self._inhibition is not None,
                "n1_strict": s.inhibition_n1_strict,
                "n3_enabled": s.inhibition_n3_enabled,
                "default_action": s.inhibition_default_action,
                "stats": (
                    self._inhibition.stats.to_dict()
                    if self._inhibition is not None
                    else None
                ),
            },
            "planning": {
                "enabled": self._planning is not None,
                "max_steps": s.planning_max_steps,
                "use_llm": s.planning_use_llm,
                "active_goal": (
                    {
                        "description": g.description,
                        "current_step": g.current_step,
                        "n_steps": len(g.steps),
                    }
                    if self._planning is not None
                    and self._planning.goal_stack.has_active()
                    and (g := self._planning.goal_stack.get_active()) is not None
                    else None
                ),
            },
            "predictive_coding": {
                "enabled": self._predictive_coding is not None,
                "apply_gating": self._pc_apply_gating,
                "last_decision": (
                    self._last_gating_decision.to_dict()
                    if self._last_gating_decision is not None
                    else None
                ),
            },
            "timeline": {
                "enabled": self._timeline is not None,
                "max_events": s.timeline_max_events,
                "render_vague": s.timeline_render_vague,
            },
            "metacognition": {
                "enabled": self._metacognition is not None,
                "apply_hedge": s.metacog_apply_hedge,
                "last_decision": (
                    self._last_meta_decision.to_dict()
                    if self._last_meta_decision is not None
                    else None
                ),
                "snapshot": (
                    self._metacognition.to_dict()
                    if self._metacognition is not None
                    else None
                ),
            },
            "affect_modulates_consolidation": (
                bool(self._microsleep_manager.config.affect_modulates)
                if hasattr(self, "_microsleep_manager")
                else False
            ),
        }

    def v4_set_module(self, module: str, enabled: bool) -> dict[str, Any]:
        """Enable or disable a single V4 module at runtime.

        ``module`` ∈ {cognitive_state, inhibition, planning,
        predictive_coding, timeline, metacognition,
        affect_modulates_consolidation}.

        Returns the new ``v4_status()`` snapshot. Never raises — on
        failure, the module is left in its prior state and the
        snapshot reflects that.
        """
        from rune.settings import get_settings
        s = get_settings()
        try:
            if module == "cognitive_state":
                if enabled and self._cognitive_state is None:
                    from pathlib import Path as _P
                    from rune.memory.cognitive_state import (
                        CognitiveState, CognitiveStateConfig,
                    )
                    self._cognitive_state = CognitiveState(
                        config=CognitiveStateConfig(
                            decay_half_life_sec=s.affect_decay_half_life_sec,
                            contagion_max=s.affect_contagion_max,
                            inertia=s.affect_inertia,
                            reset_latch_turns=s.affect_reset_latch_turns,
                            detector=s.affect_detector,
                            user_known_threshold=s.user_model_known_threshold,
                        ),
                        storage_dir=_P("data/cognitive_state"),
                    )
                elif not enabled:
                    self._cognitive_state = None

            elif module == "inhibition":
                if enabled and self._inhibition is None:
                    from rune.cognition.inhibition import (
                        InhibitionConfig, InhibitionFilter, parse_whitelist,
                    )
                    self._inhibition = InhibitionFilter(InhibitionConfig(
                        n1_strict=s.inhibition_n1_strict,
                        n2_enabled=s.inhibition_n2_enabled,
                        n3_enabled=s.inhibition_n3_enabled,
                        default_action=s.inhibition_default_action,
                        domain_whitelist=parse_whitelist(s.inhibition_domain_whitelist),
                    ))
                elif not enabled:
                    self._inhibition = None

            elif module == "planning":
                if enabled and self._planning is None:
                    from pathlib import Path as _P
                    from rune.cognition.planning import (
                        GoalStack, PlanGenerator, PlanGeneratorConfig,
                        PlanningConfig, PlanningPhase,
                    )
                    self._planning = PlanningPhase(
                        config=PlanningConfig(
                            max_steps=s.planning_max_steps,
                            goal_stale_days=s.planning_goal_stale_days,
                            use_llm=s.planning_use_llm,
                            prompt_block_max_chars=s.planning_prompt_block_max_chars,
                        ),
                        goal_stack=GoalStack(_P("data/goals/goals.json")),
                        plan_generator=PlanGenerator(
                            config=PlanGeneratorConfig(
                                use_llm=s.planning_use_llm,
                                max_steps=s.planning_max_steps,
                            )
                        ),
                    )
                elif not enabled:
                    self._planning = None
                    self._planning_block = ""

            elif module == "predictive_coding":
                if enabled and self._predictive_coding is None:
                    from rune.cognition.predictive_coding import (
                        PredictiveCodingConfig, PredictiveCodingPhase,
                    )
                    self._predictive_coding = PredictiveCodingPhase(
                        PredictiveCodingConfig(
                            history_size=s.pc_history_size,
                            cold_start_min=s.pc_cold_start_min,
                            ema_decay=s.pc_ema_decay,
                            low_threshold=s.pc_low_threshold,
                            high_threshold=s.pc_high_threshold,
                            confidence_cap=s.pc_confidence_cap,
                            gating_w_sdm=getattr(s, "pc_gating_w_sdm", 0.0),
                        )
                    )
                elif not enabled:
                    self._predictive_coding = None
                    self._last_gating_decision = None

            elif module == "timeline":
                if enabled and self._timeline is None:
                    from rune.cognition.timeline import (
                        TimelineConfig, TimelineExtractor,
                    )
                    self._timeline = TimelineExtractor(TimelineConfig(
                        max_events=s.timeline_max_events,
                        block_max_chars=s.timeline_block_max_chars,
                        render_min_confidence=s.timeline_render_min_confidence,
                        render_vague=s.timeline_render_vague,
                    ))
                elif not enabled:
                    self._timeline = None
                    self._timeline_block = ""

            elif module == "metacognition":
                if enabled and self._metacognition is None:
                    from pathlib import Path as _P
                    from rune.cognition.metacognition import (
                        MetacognitionConfig, MetacognitivePhase,
                    )
                    self._metacognition = MetacognitivePhase(
                        config=MetacognitionConfig(
                            very_high_doubt=s.metacog_very_high_doubt,
                            high_doubt=s.metacog_high_doubt,
                            low_doubt=s.metacog_low_doubt,
                            epistemic_boost_threshold=s.metacog_epistemic_boost_threshold,
                            apply_hedge=s.metacog_apply_hedge,
                            calibration_window=s.metacog_calibration_window,
                        ),
                        storage_path=_P("data/metacognition/calibration.json"),
                    )
                elif not enabled:
                    self._metacognition = None
                    self._last_meta_decision = None

            elif module == "affect_modulates_consolidation":
                # In-place toggle on the live MicrosleepConfig — takes
                # effect on the NEXT record_event call. No rebuild needed.
                if hasattr(self, "_microsleep_manager"):
                    self._microsleep_manager.config.affect_modulates = bool(enabled)

            elif module == "predictive_coding_apply_gating":
                # Sub-flag toggle — only effective when predictive_coding
                # itself is enabled. We expose it separately so the UI
                # can let the user "see what would happen" without
                # actually applying gating.
                self._pc_apply_gating = bool(enabled)

            else:
                log.warning("V4 toggle: unknown module %r", module)
        except Exception:
            log.exception("V4 toggle for %r failed — leaving previous state", module)

        return self.v4_status()

    # ── V3.9 cascade helpers ───────────────────────────────────────────

    def _build_cascade_if_enabled(self, s: Any) -> Any:
        """Build the :class:`CascadeGenerator` if settings allow.

        Returns ``None`` when:
          * ``enable_cascade`` is ``False`` (default) — V3 path.
          * ``GOOGLE_API_KEY`` is missing — log a warning and stay
            local-only. We do NOT raise; the user might be running
            with a stale env var while the rest of Lythéa works fine.
          * The Gemini client constructor rejects the key (bad format).

        On success returns a fully-wired :class:`CascadeGenerator`
        with the local model bound as the synthesis/fallback target.
        """
        if not getattr(s, "enable_cascade", False):
            return None

        from rune.cognition.cascade import CascadeGenerator
        from rune.external.gemini_client import (
            GeminiClient,
            GeminiClientError,
            mask_api_key,
        )

        # Une clé saisie à chaud (UI) prime sur le .env ; sinon fallback .env.
        api_key = self._cascade_key_override or getattr(s, "google_api_key", None)
        if not api_key:
            log.warning(
                "Cascade enabled but GOOGLE_API_KEY is empty — "
                "staying local-only"
            )
            return None

        try:
            gemini = GeminiClient(
                api_key=api_key,
                model=s.cascade_gemini_model,
                daily_limit=s.cascade_daily_quota_hint,
            )
        except GeminiClientError as exc:
            log.warning(
                "Cascade disabled — Gemini client init failed: %s "
                "(key=%s)",
                exc, mask_api_key(api_key),
            )
            return None

        # Wrap the local model so the cascade has a uniform callable.
        def _local_generate(
            system_prompt: str,
            messages: list[dict[str, str]],
            max_tokens: int,
        ) -> str:
            return self._generate_local_blocking(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
            )

        cascade = CascadeGenerator(
            gemini=gemini,
            local_generator=_local_generate,
            synthesis_threshold_tokens=s.cascade_synthesis_threshold_tokens,
            synthesis_max_tokens=s.cascade_synthesis_max_tokens,
            gemini_max_tokens=s.cascade_gemini_max_tokens,
            gemini_temperature=s.cascade_gemini_temperature,
        )
        log.info(
            "Cascade ready — model=%s threshold=%d local-fallback=on",
            s.cascade_gemini_model, s.cascade_synthesis_threshold_tokens,
        )
        return cascade

    def _generate_local_blocking(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        """Drain ``model.stream_generate`` into a single string.

        The cascade needs synchronous output; the model wrapper only
        offers a streaming interface. We collect chunks, ignore the
        per-token entropies (the cascade has its own quality signal
        via Gemini's ``finish_reason``), and return the final text.

        If ``system_prompt`` is non-empty we prepend it as a fresh
        system turn. Lythéa's normal pipeline puts the system prompt
        in the chat template separately; for the cascade synthesis
        step the prompt IS the user message, so we keep it simple.
        """
        if not self.model.is_loaded:
            return ""

        chat = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        chat.extend(messages)

        out_text = ""
        for chunk in self.model.stream_generate(
            chat, max_new_tokens=max_tokens, sampling=self.sampling_profile,
        ):
            out_text = chunk.get("text", out_text)
        return out_text or ""

    @property
    def cascade_enabled(self) -> bool:
        """True if the V3.9 cascade is wired up and ready to use."""
        return self._cascade is not None and self._cascade.is_enabled

    def cascade_status(self) -> dict[str, Any]:
        """Snapshot of cascade state for the /api/config/cascade endpoint.

        Never returns the API key in any form. The mask helper is the
        only sanctioned way to expose key information.
        """
        from rune.external.gemini_client import mask_api_key
        from rune.settings import get_settings
        s = get_settings()
        # Une clé saisie à chaud (UI) prime sur le .env pour l'affichage.
        key = self._cascade_key_override or s.google_api_key
        if self._cascade is None:
            return {
                "enabled": False,
                "reason": "disabled" if not s.enable_cascade
                          else "no_api_key" if not key
                          else "init_failed",
                "model": s.cascade_gemini_model,
                "api_key_masked": mask_api_key(key),
                "quota_used": 0,
                "quota_remaining": 0,
            }
        gemini = self._cascade._gemini
        return {
            "enabled": True,
            "reason": "ready",
            "model": gemini.model if gemini else s.cascade_gemini_model,
            "api_key_masked": mask_api_key(key),
            "quota_used": gemini.quota_used if gemini else 0,
            "quota_remaining": gemini.quota_remaining if gemini else 0,
            "synthesis_threshold_tokens": s.cascade_synthesis_threshold_tokens,
            "synthesis_max_tokens": s.cascade_synthesis_max_tokens,
        }

    # ── Session lifecycle ──────────────────────────────────────────────

    @property
    def last_microsleep(self) -> float:
        """Wall-clock timestamp of the last completed microsleep.

        Exposed as an attribute (not a method) for backwards
        compatibility — Phase C reads it directly.
        """
        return self.consolidation_phase.last_microsleep_ts

    def reset_session(self) -> None:
        """Flush session-scoped memories (SDM + MHN). KG and Chroma preserved."""
        self.sdm.flush()
        self.mhn.clear()
        self.salience.reset()
        self.exchange_count = 0
        self.last_activity = time.time()
        self.consolidation_phase.cancel_inactivity_timer()
        log.info("Session reset: SDM flushed, MHN cleared")

    # ── Phase A wrappers ──────────────────────────────────────────────

    def _compute_surprise(
        self,
        text: str,
        embedding: Any,
        structural_entropy: float,
        model_latent: Any = None,
    ) -> dict[str, float]:
        """Compute composite surprise — wrapper around :class:`SurprisePhase`.

        Returns the legacy dict shape (``{structural, episodic,
        predictive, chroma_discount, composite, global}``) so that
        any external caller — tests, debug code — keeps working.
        """
        return self.surprise_phase.compute(
            text=text,
            gliner_emb=embedding,
            structural_entropy=structural_entropy,
            model_latent=model_latent,
        ).as_dict()

    def _phase_a_learn(
        self, text: str, has_images: bool = False,
    ) -> dict[str, Any]:
        """Phase A — encode, surprise, write to SDM/KG.

        Composes :class:`EncodingPhase`, :class:`SurprisePhase` and
        :class:`StoragePhase`. Returns the legacy dict shape:
        ``{salient, surprise, entities, thoughts}``.

        V4 (additive): when the corresponding modules are enabled, this
        phase also populates ``encoding_emb`` (used by predictive_coding)
        and triggers cognitive_state / planning / timeline updates.
        Hooks are wrapped in try/except — any failure leaves the V3.9.4
        return shape intact and adds nothing to the result.

        ``has_images`` propagates to :meth:`EncodingPhase.encode` to
        bypass the salience cascade for image turns whose text stub
        ("Décris cette image") would otherwise be rejected as noise.
        Default False to preserve the historical signature for any
        existing caller that didn't know about images.
        """
        thoughts: list[str] = []
        encoding = self.encoding_phase.encode(text, has_images=has_images)

        if not encoding.salient:
            return {
                "salient": False,
                "surprise": {"global": 0.0},
                "entities": [],
                "thoughts": thoughts,
            }

        surprise = self._compute_surprise(
            text,
            encoding.gliner_emb,
            encoding.structural_entropy,
            encoding.mean_latent,
        )
        s_global = surprise["global"]

        thoughts.append(
            f"🧠 *Surprise globale {s_global:.2f} "
            f"(struct={surprise['structural']:.2f}, "
            f"épisod={surprise['episodic']:.2f}, "
            f"prédict={surprise['predictive']:.2f}, "
            f"chroma=-{surprise['chroma_discount']:.2f})*"
        )

        self.storage_phase.write_active(
            latents=encoding.latents,
            token_entropies=encoding.token_entropies,
            raw_entities=encoding.raw_entities,
            s_global=s_global,
        )

        # ── V4 hooks (all default OFF; each guarded by try/except) ────
        # Failures here MUST NOT break the V3.9.4 contract, so we
        # collect any side effects (extra thoughts, cached blocks)
        # without mutating the legacy return shape.

        # A.1 — cognitive_state observation (TPJ + amygdala).
        if self._cognitive_state is not None:
            try:
                self._cognitive_state.observe_user_message(
                    text or "",
                    entities=list(encoding.raw_entities),
                )
                # V5.8.5 — Surface l'état affectif quand il est notable.
                # Bonne API : cognitive_state.user_affect.smoothed.{valence,arousal}
                # (V5.8.4 cherchait user_state.X qui n'existe pas → silent fail).
                try:
                    user_affect = getattr(
                        self._cognitive_state, "user_affect", None,
                    )
                    if user_affect is not None and hasattr(user_affect, "smoothed"):
                        smoothed = user_affect.smoothed
                        arousal = float(getattr(smoothed, "arousal", 0.0) or 0.0)
                        valence = float(getattr(smoothed, "valence", 0.0) or 0.0)
                        conf = float(getattr(smoothed, "confidence", 0.0) or 0.0)
                        # Seuils notables (V5.8.5 : baissés car le lissage
                        # exponentiel atténue les premières observations) :
                        # arousal > 0.25 OU |valence| > 0.2, avec confiance > 0.3
                        if conf > 0.3 and (arousal > 0.25 or abs(valence) > 0.2):
                            mood = "neutre"
                            if valence > 0.2: mood = "positif"
                            elif valence < -0.2: mood = "négatif"
                            intensity = "calme"
                            if arousal > 0.5: intensity = "forte"
                            elif arousal > 0.25: intensity = "modérée"
                            thoughts.append(
                                f"🌡️ *Affect détecté — intensité {intensity} "
                                f"({arousal:.2f}), ton {mood} ({valence:+.2f})*"
                            )
                except Exception:
                    log.debug("Affect signal extraction failed", exc_info=True)
            except Exception:
                log.exception("V4 hook A.1 (cognitive_state) failed")

        # A.2 — planning (PFC).
        if self._planning is not None:
            try:
                plan_res = self._planning.process(text or "")
                self._planning_block = plan_res.prompt_block or ""
                if plan_res.is_new_goal and plan_res.active_goal is not None:
                    thoughts.append("🎯 *Nouveau but enregistré.*")
                # V4.0.2: surface step-advance and goal-completion
                # feedback so the user sees their action acknowledged.
                if plan_res.advanced_step and plan_res.active_goal is not None:
                    g = plan_res.active_goal
                    if plan_res.completed_goal:
                        thoughts.append(
                            f"✅ *But « {g.description[:40]}... » entièrement réalisé.*"
                        )
                    else:
                        cur = g.current_step + 1  # 1-indexed for humans
                        total = len(g.steps)
                        thoughts.append(
                            f"➡️ *Étape {cur}/{total} marquée comme faite.*"
                        )
            except Exception:
                log.exception("V4 hook A.2 (planning) failed")
                self._planning_block = ""
        else:
            self._planning_block = ""

        # A.3 — predictive coding (Friston-style cortical prediction).
        # Uses the encoding's mean latent embedding as the observed
        # representation. The decision is cached for Phase B gating.
        self._last_gating_decision = None
        if self._predictive_coding is not None:
            try:
                # Convert the torch tensor to a plain list[float] to
                # keep predictive_coding pure-Python.
                emb_obj = getattr(encoding, "mean_latent", None)
                emb_list = None
                if emb_obj is not None:
                    try:
                        emb_list = emb_obj.detach().cpu().tolist()
                    except Exception:
                        # Fallback: maybe already a list/sequence.
                        try:
                            emb_list = list(emb_obj)
                        except Exception:
                            emb_list = None
                self._last_gating_decision = self._predictive_coding.observe(
                    emb_list, sdm_error=surprise.get("predictive")
                )
            except Exception:
                log.exception("V4 hook A.3 (predictive_coding) failed")
                self._last_gating_decision = None

        # A.4 — timeline (narrative chronology).
        if self._timeline is not None:
            try:
                from rune.cognition.timeline import render_block as _tl_render
                events = self._timeline.extract(text or "")
                # Pass the source text so render_block can split into
                # clauses for accurate inconsistency detection.
                self._timeline_block = _tl_render(
                    events, self._timeline.config, text=text or "",
                ) or ""
            except Exception:
                log.exception("V4 hook A.4 (timeline) failed")
                self._timeline_block = ""
        else:
            self._timeline_block = ""

        return {
            "salient": True,
            "surprise": surprise,
            "entities": list(encoding.raw_entities),
            "thoughts": thoughts,
            # Additive V4 field — exposed for downstream consumers
            # (e.g. predictive_coding) but legacy callers ignore it.
            "encoding_emb": getattr(encoding, "mean_latent", None),
        }

    # ── Phase B / KG identity wrappers ────────────────────────────────

    def _kg_identity_summary(self) -> str:
        """Identity narrative — wrapper around :class:`RetrievalPhase`."""
        return self.retrieval_phase.kg_identity_summary()

    def _phase_b_rag(self, query: str) -> tuple[str, list[str]]:
        """Phase B — RAG context. Wrapper around :class:`RetrievalPhase`."""
        ctx = self.retrieval_phase.gather(query)
        return ctx.render(), ctx.thoughts

    # ── V5.1 Tool routing (semantic + JSON dispatcher) ────────────────

    def _generate_python_code(self, query: str) -> str:
        """Ask the LLM to produce executable Python for the question.

        Used by the V5.1 python_executor branch. We do a short
        synchronous LLM call with a constrained prompt that asks
        for raw Python in a code block. We extract it and return
        the inner code (the executor will wrap and sandbox it).

        V5.8.0 — Prompt enrichi pour couvrir analyse de données,
        validation arithmétique (dates/primalité/exact), et conversions
        (unités, encodages, formats). Le LLM choisit la bonne lib selon
        le contexte de la question.

        Returns
        -------
        str
            The extracted Python code, or empty string if the LLM
            didn't produce a parseable code block.
        """
        prompt = [
            {"role": "system", "content": (
                "Tu es un générateur de code Python. Tu ne réponds QUE par "
                "un bloc de code Python exécutable répondant à la question. "
                "PAS de texte autour, PAS d'explication, PAS de phrase. "
                "Format obligatoire :\n"
                "```python\n"
                "<code>\n"
                "```\n\n"
                "Librairies disponibles dans le sandbox :\n"
                "- math, statistics, numpy : calculs numériques\n"
                "- sympy : calculs exacts (primalité, factorisation, "
                "PGCD, PPCM, combinaisons, Fibonacci, factorielles)\n"
                "- datetime : calculs de dates (différences en jours, "
                "jour de la semaine, additions de durées)\n"
                "- base64, urllib.parse : encodages texte\n"
                "- json : reformatage de JSON\n"
                "- matplotlib (pré-configuré pour PNG) : graphiques\n\n"
                "Règles :\n"
                "1. Pour AFFICHER un résultat : utilise print() — c'est ce "
                "que l'utilisateur va voir.\n"
                "2. Pour un GRAPHIQUE : utilise plt.show() — sera capturé.\n"
                "3. Pour la VALIDATION ARITHMÉTIQUE (dates, primalité, "
                "combinatoire) : utilise SymPy ou datetime, JAMAIS de "
                "calcul de tête. Le but est l'exactitude.\n"
                "4. Pour les CONVERSIONS d'unités : applique la formule "
                "puis affiche le résultat avec son unité.\n"
                "5. Pour les ENCODAGES : utilise base64 / urllib.parse / "
                "json plutôt que de bricoler à la main.\n"
                "6. Pour les STATISTIQUES sur listes : utilise numpy ou "
                "statistics. Pour des analyses plus riches (quartiles, "
                "histogrammes), affiche un résumé propre via print().\n"
                "7. Évite os/sys/subprocess/shutil — pas nécessaires.\n"
                "8. Noms de variables en ASCII pur (pas d'accents) : "
                "'frequences' pas 'fréquences', 'donnees' pas 'données'. "
                "Les valeurs et chaînes affichées peuvent garder leurs "
                "accents, c'est uniquement pour les identifiants Python.\n"
            )},
            {"role": "user", "content": f"Question : {query.strip()}"},
        ]
        try:
            raw = self.model.complete_sync(
                prompt, max_new_tokens=256, temperature=0.2,
            )
        except Exception as exc:
            log.warning("Python code generation failed: %s", exc)
            return ""

        # Extract code block ```python ... ``` or ``` ... ```
        import re as _re_code
        m = _re_code.search(
            r"```(?:python)?\s*\n?(.*?)```",
            raw, _re_code.DOTALL,
        )
        if m:
            code = m.group(1).strip()
            if code:
                return code
        # No code block found — try the whole raw output if it looks
        # like code (starts with import/def/print/= or contains \n).
        stripped = raw.strip()
        if stripped and any(
            stripped.startswith(kw) for kw in
            ("import ", "from ", "print(", "def ", "for ", "x =", "result =")
        ):
            return stripped
        log.warning("Python code: no code block parseable from %r", raw[:100])
        return ""

    def _list_workspace_files(self) -> list[str]:
        """V6.0.0-rc — Liste les fichiers du workspace, plus récent d'abord.

        Utilisée par le planner MCP pour résoudre les références
        implicites (« le fichier que je viens d'ajouter ») et faire
        un match case-insensitive sur les noms de fichiers explicites.

        Returns
        -------
        list[str]
            Chemins relatifs au sandbox root, triés par mtime
            décroissante. Vide si le workspace est inaccessible
            (settings absent, dossier inexistant, etc.).
        """
        try:
            from pathlib import Path
            from rune.settings import get_settings
            s = get_settings()
            sandbox_str = (
                getattr(s, "mcp_sandbox_dir", "")
                or str(Path.home() / ".lythea" / "sandbox")
            )
            root = Path(sandbox_str)
            if not root.exists() or not root.is_dir():
                return []
            files = []
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                # Skip hidden/system files
                if any(part.startswith(".") for part in p.relative_to(root).parts):
                    continue
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    mtime = 0
                rel = str(p.relative_to(root)).replace("\\", "/")
                files.append((mtime, rel))
            # Sort by mtime DESC : most recent first
            files.sort(key=lambda t: -t[0])
            return [rel for _, rel in files]
        except Exception as exc:
            log.warning("_list_workspace_files failed: %s", exc)
            return []

    def _emit_workspace_offers(
        self, final_text: str, user_intent: str = "",
    ) -> Generator[dict, None, None]:
        """V6.0.0-rc rev6 — Détecte les artefacts dans la réponse et émet
        un ``workspace_file_offer`` SSE pour chacun.

        Règle d'or (rev6) : on n'écrit dans le workspace que si :
        - l'utilisateur a EXPLICITEMENT demandé un livrable
          (« écris-moi un script », « génère un rapport »,
          « sauvegarde dans un fichier ») OU
        - Lythéa a généré un script Python ≥ 10 lignes (clairement un
          livrable même sans demande explicite — c'est rarement
          conversationnel).

        Les longs résumés markdown ne créent PLUS de rapport.md
        automatiquement (bug observé : à chaque résumé Excel, Lythéa
        créait un faux rapport.md non demandé).

        Workflow :
        1. detect_artefacts() scanne final_text avec user_intent comme
           filtre d'intention
        2. Pour chaque artefact détecté, on l'écrit dans le workspace
           via WorkspaceManager
        3. On émet un event SSE par fichier réussi.
        """
        try:
            from rune.cognition.mcp_integration import (
                detect_artefacts, write_artefact_to_workspace,
            )
        except ImportError:
            return

        artefacts = detect_artefacts(final_text, user_intent=user_intent)
        if not artefacts:
            return

        # Construire le WorkspaceManager localement (pareil que dans
        # routes.py) — on évite une dépendance directe sur l'app FastAPI.
        try:
            from pathlib import Path
            from rune.settings import get_settings
            from rune.server.workspace import WorkspaceManager
            s = get_settings()
            sandbox_str = (
                getattr(s, "mcp_sandbox_dir", "")
                or str(Path.home() / ".lythea" / "sandbox")
            )
            ws = WorkspaceManager(
                sandbox_dir=Path(sandbox_str),
                max_file_bytes=getattr(s, "mcp_workspace_max_file_mb", 20) * 1024 * 1024,
                max_total_bytes=getattr(s, "mcp_workspace_max_total_mb", 200) * 1024 * 1024,
            )
        except Exception as exc:
            log.warning("Cannot build WorkspaceManager for offers: %s", exc)
            return

        for art in artefacts:
            entry = write_artefact_to_workspace(art, ws)
            if entry is None:
                continue
            log.info(
                "Workspace offer emitted : %s (kind=%s)",
                entry["name"], art.kind,
            )
            yield {"type": "workspace_file_offer", "data": entry}

    def _route_tool_v5_1(
        self, query: str, settings_obj,
    ) -> tuple[str, str]:
        """V5.1 multi-tool router (semantic + JSON dispatcher fallback).

        Returns
        -------
        tuple[str, str]
            ``(tool_name, reason)``. tool_name is one of
            {"web", "python", "none"}. Reason is a short human-readable
            string for logs/UI.

        Pipeline
        --------
        1. Semantic router (~25ms CPU). Fast classification via
           sentence-transformer embeddings against per-tool example
           phrases. If confidence ≥ route.threshold, return immediately.
        2. JSON dispatcher LLM (~300ms). Fallback when the semantic
           router is ambiguous (no route above its threshold). The LLM
           picks one of the valid tools, returns JSON.
        3. On any error → "none" (safest default).
        """
        # Niveau 1 : sémantique
        try:
            from rune.cognition.semantic_router import get_router
            router = get_router()
            route_name, conf, scores = router.classify(query)
            if route_name is not None:
                log.info(
                    "Router (semantic): %s (conf=%.2f, scores=%s)",
                    route_name, conf,
                    {k: round(v, 2) for k, v in scores.items()},
                )
                return route_name, f"semantic conf={conf:.2f}"
            else:
                log.info(
                    "Router (semantic): ambiguous (best=%.2f, scores=%s)",
                    conf, {k: round(v, 2) for k, v in scores.items()},
                )
        except Exception as exc:
            log.warning("Semantic router failed: %s", exc)

        # Niveau 2 : LLM dispatcher (only if model is loaded)
        if not self.model.is_loaded:
            log.info("Router: model not loaded, defaulting to 'none'")
            return "none", "model_not_loaded"
        try:
            from rune.cognition.tool_dispatcher import dispatch_via_llm
            tool, reason = dispatch_via_llm(
                query,
                self.model,
                timeout=settings_obj.web_classifier_timeout_s,
            )
            log.info("Router (LLM dispatcher): %s — %s", tool, reason)
            return tool, f"dispatcher {reason}"
        except Exception as exc:
            log.warning("JSON dispatcher failed: %s", exc)
            return "none", "dispatcher_crash"

    # ── Phase C: Prompt Assembly with token budget ─────────────────────

    def _phase_c_assemble(
        self,
        message: str,
        history: list[dict],
        rag_context: str,
        last_message_ts: float | None = None,
        session_created_ts: float | None = None,
        v4_blocks: list[str] | None = None,
    ) -> list[dict]:
        """Build the message list with token budget management.

        The temporal block always renders (date/time/period are
        meaningful even on a brand-new session). Optional gap and
        duration lines self-suppress when their inputs are None.

        V4 (additive): ``v4_blocks`` is an optional ordered list of
        prompt fragments (timeline, planning, user_state, self_affect)
        injected between the temporal block and the RAG memory. Each
        block self-suppresses (empty string) when its module is OFF
        or has nothing to say. Unknown order: caller decides.
        """
        messages: list[dict] = []

        temporal_block = TemporalContext(
            now=datetime.now(),
            last_message_ts=last_message_ts,
            session_created_ts=session_created_ts,
            last_microsleep_ts=self.last_microsleep,
        ).render()

        # ⚠️ Vital constraints — ALWAYS on, before identity, NOT gated on
        # retrieval. Direct fix for the "propose-moi un menu" failure: the
        # model must honour allergies/medical/dietary facts spontaneously,
        # including for any food question, even without the word "allergie".
        vital_block = ""
        try:
            constraints = self.kg.critical_constraints()
        except Exception:  # noqa: BLE001
            constraints = []
        if constraints:
            vital_block = (
                "⚠️ CONTRAINTES VITALES — TOUJOURS ACTIVES (priorité absolue)\n"
                "Ces faits touchent à la sécurité de ton interlocuteur. Tu DOIS "
                "les respecter spontanément, sans qu'on te les rappelle, dès "
                "qu'ils sont pertinents — en particulier pour TOUTE question de "
                "nourriture, menu, recette ou restaurant, MÊME si le mot "
                "« allergie » n'est pas prononcé :\n"
                + "\n".join(f"- {c}" for c in constraints)
                + "\nNe propose jamais quelque chose qui violerait l'une de ces "
                "contraintes ; en cas de doute, signale-le explicitement.\n\n"
            )

        system_text = vital_block + SYSTEM_PROMPT + "\n\n" + temporal_block

        # V4 cognitive blocks — only the non-empty ones get appended,
        # preserving exact V3.9.4 string output when no V4 module is on.
        if v4_blocks:
            for block in v4_blocks:
                if block:
                    system_text += "\n\n" + block

        # V5.4 — Injecter le playbook procédural si disponible.
        # On présente les patterns comme des HABITUDES (« j'ai
        # l'habitude de... ») pour ne pas écraser le jugement
        # contextuel — c'est de la métacognition, pas de la règle
        # dure.
        if hasattr(self, "procedural_store") and self.procedural_store is not None:
            try:
                from rune.memory.procedural import render_playbook
                top_procs = self.procedural_store.top_n(10)
                playbook = render_playbook(top_procs, max_chars=800)
                if playbook:
                    system_text += "\n\n" + playbook
                    log.info(
                        "Procedural playbook injected: %d procedures, %d chars",
                        len(top_procs), len(playbook),
                    )
            except Exception as exc:
                log.warning("Playbook injection failed: %s", exc)

        if rag_context:
            system_text += f"\n\n[Mémoire contextuelle]\n{rag_context}"
        messages.append({"role": "system", "content": system_text})

        # V5.6.10 — Log debug du system_text en DEBUG level. Permet de
        # diagnostiquer les hallucinations comme "41 ans" inventé par
        # Qwen3-4B en mode thinking (cause : exemple-template dans
        # SYSTEM_PROMPT, fixé en V5.6.10). Réactivable via le toggle
        # Debug 🔬 dans la topbar si besoin de re-débugger.
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "FULL system_text sent to model (%d chars):\n%s\n[END]",
                len(system_text), system_text,
            )

        # Token budget — newest-first, stop when char count exceeds
        # the remaining budget.
        remaining = max(
            _PROMPT_CHAR_BUDGET - len(system_text) - len(message),
            _PROMPT_HISTORY_FLOOR,
        )
        trimmed_history: list[dict] = []
        char_count = 0
        for msg in reversed(history[-MAX_HISTORY_TURNS * 2:]):
            content = msg.get("content", "")
            if char_count + len(content) > remaining:
                break
            trimmed_history.insert(0, msg)
            char_count += len(content)
        messages.extend(trimmed_history)

        messages.append({"role": "user", "content": message})
        return messages

    # ── Phase D-pre: two-pass reasoning ────────────────────────────────

    def _generate_reasoning(
        self,
        message: str,
        chat_messages: list[dict],
        surprise: dict[str, float] | None = None,
        on_step: Any | None = None,
        web_context: str = "",
    ) -> str:
        """Passe de raisonnement pour modèles non-thinking.

        Utilise la chaîne profonde ``DeepReasoningChain`` avec
        ``force_minimum=2`` : chaque message non-trivial reçoit au
        moins une chaîne 2-étapes (explorer + feuille de route). Le
        routeur peut monter à 4 étapes si la question le mérite.

        ``web_context`` — quand une recherche web a été déclenchée,
        ses résultats (numérotés [1][2]…) sont passés à la chaîne
        pour ancrer l'exploration dans des faits réels au lieu des
        connaissances internes (limitées) du modèle.

        Messages triviaux (≤6 mots, salutations) : skippés — pas de
        raisonnement, le modèle répond directement.
        """
        del chat_messages  # not used — preserved for signature compat

        # ── Garde-fou trivialité ──────────────────────────────────────
        msg_stripped = (message or "").strip()
        words = msg_stripped.split()
        if len(words) <= 6:
            from rune.cognition.deep_reasoning import _ANALYTICAL_MARKERS
            lowered = msg_stripped.lower()
            has_marker = any(m in lowered for m in _ANALYTICAL_MARKERS)
            if not has_marker:
                log.debug("Reasoning skipped: trivial message (%d words)",
                          len(words))
                return ""

        # ── Garde-fou document attaché ───────────────────────────────
        # Si un document est joint ou vient d'être ingéré, on skip
        # le raisonnement profond. Le scaffolding "décomposer /
        # explorer / critiquer / synthétiser" est conçu pour la
        # pensée analytique sur des connaissances internes — sur
        # un document fourni, la tâche est de l'EXTRACTION (résumé,
        # citation, point-clé), que le RAG + génération directe
        # traitent mieux et plus vite. Le scaffolding ajouterait
        # du bruit hors-sol (le modèle ne "sait" rien sur le doc,
        # il a juste accès aux chunks RAG) et peut même halluciner
        # en remplissant artificiellement les étapes.
        if (
            "[Document joint —" in message
            or "[Note système — l'utilisateur vient d'ajouter" in message
        ):
            log.debug("Reasoning skipped: document attached/ingested")
            return ""

        # ── Chaîne profonde avec minimum garanti ──────────────────────
        try:
            kg_count = len(getattr(self.kg, "entities", {}) or {})
            return self.deep_reasoning.run(
                message,
                surprise=surprise,
                doubt_index=None,
                kg_entity_count=kg_count,
                on_step=on_step,
                force_minimum=2,
                web_context=web_context or "",
            )
        except Exception as exc:
            log.warning("Deep reasoning chain failed: %s", exc)
            return ""

    # ── Phase D: Streaming generation ──────────────────────────────────

    def process_message(
        self,
        message: str,
        history: list[dict],
        images: list[Any] | None = None,
        cancelled: threading.Event | None = None,
        last_message_ts: float | None = None,
        session_created_ts: float | None = None,
        user_intent_message: str | None = None,
    ) -> Generator[dict, None, None]:
        """Full cognitive pipeline: learn → RAG → assemble → generate → post.

        ``message`` est le texte complet qui partira en génération
        (peut inclure des préfixes de contexte comme un document joint).
        ``user_intent_message`` est la VRAIE question utilisateur sans
        ces préfixes — utilisée pour les heuristiques (déclencheur web,
        routeur de complexité, garde-fou trivialité). Sans cette
        séparation, le contenu d'un document joint peut faussement
        déclencher des comportements (ex. doc contient « récemment » →
        web temporel s'enclenche sur le doc, pas sur la question).
        Si non fourni, on retombe sur ``message`` (rétro-compat).

        Yields SSE event dicts: ``cognitive``, ``reasoning``,
        ``partial``, ``entropy``, ``done``, ``error``, ``debug``.
        """
        # ``user_intent`` sert pour toutes les heuristiques. Quand
        # l'appelant ne le fournit pas (rétro-compat), on prend message.
        user_intent: str = user_intent_message or message
        try:
            # Phase A: Learn from input. Bypass salience for image turns
            # so the text stub ("Décris cette image") doesn't suppress
            # archiving; the image carries the actual new information.
            #
            # V4.2 — on apprend de user_intent, PAS de message. Quand un
            # document est joint en mode "attach" (📎 Pour ce message),
            # le doc est dans message mais on ne veut PAS qu'il soit
            # mémorisé (sinon le badge ment). user_intent contient juste
            # la question utilisateur → Phase A encode/extrait depuis
            # ça. Cohérent avec ce qui se passe pour les déclencheurs
            # web/reasoning (eux aussi utilisent user_intent). En mode
            # ingest, le KG est enrichi explicitement par
            # document_ingest.py (NER chunking), pas par Phase A.
            cognitive_items: list[str] = []
            learn_result = self._phase_a_learn(
                user_intent, has_images=bool(images or []),
            )
            cognitive_items.extend(learn_result["thoughts"])

            # V5.8.1 — FIX B3 final : Inhibition sur l'INPUT utilisateur.
            #
            # On check les patterns "demande malveillante" (api_key_request_fr,
            # instruction_override_fr, system_reveal_fr, instruction_bypass_fr,
            # roleplay_jailbreak_fr) AVANT toute génération. Si match → on
            # bloque immédiatement, on émet une réponse neutre, et on
            # économise le compute LLM en plus de garantir la sécurité.
            #
            # Patterns de FUITE (api_key_leak, private_key, etc.) restent
            # checkés sur l'OUTPUT plus tard dans le pipeline (V4 hook).
            if self._inhibition is not None:
                try:
                    inh_input = self._inhibition.check(
                        user_intent, direction="input",
                    )
                    if not inh_input.passed and inh_input.action == "block":
                        log.warning(
                            "V5.8.1 inhibition INPUT BLOCK — level=%s reason=%s",
                            inh_input.level, inh_input.reason,
                        )
                        cognitive_items.append(
                            f"🛑 *Filtre N1 (input) déclenché — {inh_input.reason}*"
                        )
                        # Émet le bloc cognitif accumulé jusque-là
                        if cognitive_items:
                            yield {"type": "cognitive", "data": {"items": cognitive_items}}
                        # Émet une réponse neutre + done
                        blocked_text = (
                            f"*[Réponse inhibée par filtre de sécurité — "
                            f"{inh_input.level}: {inh_input.reason}]*"
                        )
                        yield {"type": "partial", "data": {"text": blocked_text}}
                        yield {
                            "type": "done",
                            "data": {
                                "final_text": blocked_text,
                                "doubt_index": 0.0,
                                "epistemic": "filtré",
                            },
                        }
                        return
                except Exception:
                    log.exception("V5.8.1 inhibition input check failed (non-blocking)")

            # Web search trigger
            # IMPORTANT : on évalue les heuristiques web sur la VRAIE
            # question utilisateur (user_intent), pas sur message qui
            # peut contenir un document joint en préfixe. Sinon le
            # contenu du document peut faussement déclencher des
            # patterns temporels/factuels.
            web_context = ""
            tool_result_text = ""  # V5.1: si python ou autre outil exécuté
            tool_chosen: str | None = None  # nom de l'outil utilisé
            tool_reason: str = ""

            should_web, web_reason = self.web_policy.should_search(user_intent)

            # V5.1 — Tool routing multi-classes (web / python / none).
            # Le fast-path regex (should_search) gère encore les cas
            # évidents (mémoire interne, /web, /noweb, factuels, tech_reco).
            # Si pas tranché, on cascade :
            #   1. Router sémantique (~25ms CPU, embeddings MiniLM)
            #   2. Dispatcher JSON LLM (~300ms, fallback ambigu)
            #
            # Si l'outil retenu est "web", on hérite du flow web habituel
            # (web_context injecté plus bas). Si c'est "python", on bascule
            # vers l'exécuteur sandboxé (étape dédiée plus bas).
            _slow_has_attached_doc = user_intent != message
            if (
                not should_web
                and not _slow_has_attached_doc
                and not (images or [])
            ):
                try:
                    from rune.settings import get_settings
                    _s = get_settings()
                    if _s.web_classifier_enabled:
                        from rune.cognition.web_classifier import (
                            looks_like_question,
                        )
                        if looks_like_question(user_intent):
                            # V6.0.0-rc — Hard-route MCP si filename
                            # explicite ET MCP disponible. Le semantic
                            # router peut hésiter et choisir "none" sur
                            # une phrase comme "Lis data.csv", ce qui
                            # mène à une hallucination du contenu par
                            # le LLM. Cette détection structurelle court-
                            # circuite le router pour les cas évidents.
                            _mcp_force = False
                            if self.mcp_manager is not None and self.mcp_loop is not None:
                                try:
                                    from rune.cognition.mcp_integration import _FILENAME_RE
                                    if _FILENAME_RE.search(user_intent):
                                        _mcp_force = True
                                except Exception:
                                    pass

                            if _mcp_force:
                                tool_chosen = "mcp"
                                tool_reason = "filename explicite (hard-route)"
                                log.info(
                                    "Router → mcp (hard-route) : filename "
                                    "détecté dans la query",
                                )
                            else:
                                tool_chosen, tool_reason = self._route_tool_v5_1(
                                    user_intent, _s,
                                )
                            if tool_chosen == "web":
                                should_web = True
                                web_reason = f"router: {tool_reason}"
                                log.info(
                                    "Router → web : %s", tool_reason,
                                )
                            elif tool_chosen == "python":
                                log.info(
                                    "Router → python : %s", tool_reason,
                                )
                            elif tool_chosen == "mcp":
                                # V6.0.0-rc — Route MCP : on lira/écrira
                                # un fichier du workspace. L'exécution
                                # proprement dite arrive plus bas (juste
                                # comme la phase Python), juste après le
                                # block routing.
                                log.info(
                                    "Router → mcp : %s", tool_reason,
                                )
                            # tool_chosen == "none" → on continue sans
                            # outil, le LLM répondra directement
                except Exception as exc:
                    log.warning("V5.1 router crashed: %s", exc)

            # V4.1 — couplage raisonnement + web. Si le raisonnement
            # profond va tourner sur une question genuinement complexe
            # (router ≥ 2 étapes) et que le web ne s'est pas déjà
            # déclenché, on le déclenche quand même : une question
            # assez complexe pour mériter un raisonnement profond
            # mérite un ancrage factuel. Précieux surtout sur les
            # petits modèles, qui inventent quand leurs connaissances
            # internes manquent (observé : un 3B inventant des noms
            # d'algorithmes post-quantiques).
            #
            # V4.2 — exception : si l'utilisateur a joint un document
            # (mode attach) OU vient d'ingérer un document (mode ingest)
            # dans ce tour, on ne déclenche PAS le web. L'utilisateur a
            # fourni explicitement une source ; aller chercher sur le web
            # serait inutile (les résultats ne parlent pas de SON doc) et
            # coûteux. Le RAG local fait le boulot. Détection : si
            # user_intent diffère de message, c'est qu'un préfixe doc a
            # été injecté par routes.py.
            _has_attached_doc = user_intent != message

            # V5.5.3 — Garde-fou self-disclosure. Quand l'utilisateur
            # déclare un fait personnel ("je m'appelle X", "je suis né
            # en YYYY", "j'habite à Y"), c'est une mise à jour de
            # mémoire, pas une question qui mérite un grounding web.
            # Sans ce filtre, "je suis né en 1985" partait calculer
            # 2026-1985 via web search au lieu de capturer la date et
            # répondre directement. Le test couvre les 6 patterns
            # V5.5.2/V5.5.3 (intro, role, location, employer, age,
            # birth_year).
            _is_self_disclosure = False
            try:
                from rune.cognition.encoding import (
                    _SELF_INTRO_RE, _SELF_ROLE_RE, _SELF_LOCATION_RE,
                    _SELF_EMPLOYER_RE, _SELF_AGE_RE, _SELF_BIRTH_YEAR_RE,
                )
                _self_disclosure_patterns = (
                    _SELF_INTRO_RE, _SELF_ROLE_RE, _SELF_LOCATION_RE,
                    _SELF_EMPLOYER_RE, _SELF_AGE_RE, _SELF_BIRTH_YEAR_RE,
                )
                for _pat in _self_disclosure_patterns:
                    if _pat.search(user_intent or ""):
                        _is_self_disclosure = True
                        log.info(
                            "Skipping reasoning_grounding: user_intent "
                            "is self-disclosure (matched %s)",
                            _pat.pattern[:40],
                        )
                        break
            except Exception:
                pass  # fail-open : si l'import casse, comportement V5.5.2

            if (
                not should_web
                and self.reasoning_enabled
                and not self.model.is_thinking
                and not (images or [])
                and not _has_attached_doc
                and not _is_self_disclosure
                # V5.9.3 — Skip reasoning_grounding si Python a été routé.
                # Le résultat Python (exact, déterministe) EST l'ancrage.
                # Sans ça, le grounding web ramène du bruit non pertinent
                # (Zillow, articles random) qui pollue le contexte et fait
                # halluciner le mode raisonnement (qui ré-estime au lieu
                # de copier stdout). Constaté in vivo V5.9.2.
                and tool_chosen != "python"
            ):
                _kg_count = len(getattr(self.kg, "entities", {}) or {})
                # Évaluer la complexité sur la vraie question, pas sur
                # le doc qui ferait toujours score élevé (long, marqueurs).
                _reasoning_steps = self.deep_reasoning.assess_complexity(
                    user_intent,
                    surprise=learn_result.get("surprise"),
                    kg_entity_count=_kg_count,
                )
                if _reasoning_steps >= 2:
                    should_web = True
                    web_reason = "reasoning_grounding"

            # V4.2 — predictive coding gating. When pc_apply_gating is
            # enabled AND the cortical predictor flagged this turn as
            # V5.6.16 — B2 fix : Le gating low_power court-circuite
            # maintenant aussi les déclencheurs "temporal" et "explicit".
            # Avant, ces déclencheurs étaient préservés ce qui créait des
            # recherches web même quand Predictive Coding décrétait que
            # le tour ne nécessitait aucune nouvelle info externe (ex :
            # "Hier j'ai vu mon médecin" en mode low_power).
            #
            # Seule exception préservée : la commande explicite "/web"
            # de l'utilisateur, qui doit toujours déclencher la recherche
            # (intention explicite, respecte la volonté de l'utilisateur).
            #
            # V5.8.3 — Exemption supplémentaire : on NE coupe PAS les
            # déclencheurs à forte évidence factuelle (year_ref, acronyme
            # technique inconnu, tech_reco, factual). Ces déclencheurs
            # ont une preuve textuelle explicite (année, acronyme majuscule,
            # pattern d'événement), ils ne peuvent pas être de fausses
            # alertes type "hier en récit perso". Le gating low_power
            # garde son rôle sur les déclencheurs faibles (temporal seul).
            _strong_web_reason = any(
                key in (web_reason or "")
                for key in (
                    "year_ref",          # "en 2025", "le 4 juillet 1776"
                    "acronym_definition",  # "c'est quoi le RMSECV"
                    "tech_reco",         # "recommande-moi un modèle NER"
                    "factual",           # "qui est le PDG de", "où se trouve"
                    "manual /web",       # déjà géré ci-dessous mais redondance OK
                )
            )
            if (
                self._pc_apply_gating
                and should_web
                and self._last_gating_decision is not None
                and getattr(self._last_gating_decision, "mode", "full") == "low_power"
                and "manual /web" not in (web_reason or "")
                and not _strong_web_reason  # V5.8.3 — exemption
            ):
                should_web = False
                cognitive_items.append(
                    "💤 *Tour à faible nouveauté — recherche web suspendue.*"
                )

            # V4.4 — Logging structuré des décisions web. Format
            # clé=valeur grepable pour analyse a posteriori. À utiliser
            # avec `grep "WEB_DECISION" logs.txt` pour identifier les
            # patterns de questions mal classifiées. Permet de cibler
            # les améliorations futures de la pile regex + classifier
            # sans deviner. Inclut :
            #   - decided : True/False (déclenche web ou pas)
            #   - reason : raison (regex, llm_classifier, manual, etc.)
            #   - via : fast_path (regex) ou slow_path (LLM classifier)
            #   - query : début du message tronqué (60 chars)
            _decision_via = (
                "slow_path"
                if "llm_classifier" in (web_reason or "")
                else "fast_path"
            )
            _truncated = (user_intent or "")[:60].replace("\n", " ")
            log.info(
                "WEB_DECISION decided=%s via=%s reason=%r query=%r",
                should_web, _decision_via, (web_reason or "none"), _truncated,
            )

            if should_web:
                if "manual /web" in (web_reason or ""):
                    cognitive_items.append("🌐 *Tu as demandé explicitement une recherche web…*")
                elif "temporal" in web_reason:
                    cognitive_items.append("🌐 *Question d'actualité détectée, je vérifie en ligne.*")
                elif "year_ref" in web_reason or "factual" in web_reason:
                    cognitive_items.append("🌐 *Question factuelle détectée, je cherche la réponse en ligne…*")
                elif "tech_reco" in web_reason:
                    cognitive_items.append("🌐 *Recommandation technique demandée, je vérifie les références exactes en ligne…*")
                elif "llm_classifier" in web_reason:
                    cognitive_items.append("🌐 *Je pense qu'une vérification en ligne ferait du bien à ma réponse…*")
                elif "reasoning_grounding" in web_reason:
                    cognitive_items.append("🌐 *Question complexe — je cherche des infos pour ancrer mon raisonnement…*")
                else:
                    cognitive_items.append("🌐 *J'ai besoin d'infos récentes, je vérifie.*")

                # Emit phase_status: web_start so the UI can show a
                # pulsing pill while the network call blocks.
                yield {
                    "type": "phase_status",
                    "data": {"phase": "web", "state": "start"},
                }
                try:
                    # Strip control tags before sending to search engine
                    import re as _re_search
                    clean_search_msg = _re_search.sub(
                        r"(?<!\w)/(?:no)?web(?!\w)", "", message
                    ).strip()
                    web_context = self.web_agent.iterative_search(
                        clean_search_msg,
                        reason=web_reason,
                    )
                finally:
                    # Always emit web_done — even on exception — so the
                    # UI pill never gets stuck pulsing forever.
                    yield {
                        "type": "phase_status",
                        "data": {"phase": "web", "state": "done"},
                    }
                if not web_context:
                    cognitive_items.append("🤷 *Rien trouvé en ligne, je fais avec ce que j'ai.*")

            # Phase B: RAG
            rag_context, rag_thoughts = self._phase_b_rag(message)
            cognitive_items.extend(rag_thoughts)

            # V6.0.0-rc — MCP filesystem executor. Si le router a
            # choisi "mcp", on planifie quel appel faire (read d'un
            # fichier nommé, list du workspace, ...), on l'exécute via
            # le MCPServerManager, et on injecte le résultat dans le
            # contexte pour que le LLM puisse répondre dessus.
            #
            # Note : on n'enchaîne pas mcp → python en un tour
            # (limitation V6.0.0-rc). Si l'utilisateur dit "lis X et
            # calcule la moyenne", le router choisira soit mcp soit
            # python, pas les deux. L'utilisateur peut faire en 2 tours :
            # "lis sales.csv" puis "calcule la moyenne des ventes".
            if (
                tool_chosen == "mcp"
                and self.mcp_manager is not None
                and self.mcp_loop is not None
            ):
                yield {
                    "type": "phase_status",
                    "data": {"phase": "mcp", "state": "start"},
                }
                cognitive_items.append(
                    "📁 *Je lis ton workspace pour répondre…*"
                )
                try:
                    from rune.cognition.mcp_integration import (
                        plan_mcp_call, execute_mcp_call,
                        format_mcp_result_for_context,
                    )
                    # Lister les fichiers du workspace pour aider le
                    # planner (résolution des références implicites,
                    # match case-insensitive sur les filenames).
                    workspace_files = self._list_workspace_files()
                    plan = plan_mcp_call(user_intent, workspace_files)
                    log.info(
                        "MCP plan : action=%s target=%r reason=%s",
                        plan.action, plan.target, plan.reason,
                    )
                    ok, content = execute_mcp_call(
                        plan, self.mcp_manager, self.mcp_loop,
                    )
                    mcp_context_text = format_mcp_result_for_context(
                        plan, ok, content,
                    )
                    # Append au message pour que le LLM final voie le
                    # contenu lu / le listing. user_intent reste
                    # inchangé (heuristiques basées dessus n'ont pas
                    # à voir le contenu MCP).
                    message = message + mcp_context_text
                    if ok:
                        cognitive_items.append(
                            f"✅ *MCP : {plan.action} `{plan.target or 'workspace'}` "
                            f"({len(content)} chars)*"
                        )
                    else:
                        cognitive_items.append(
                            f"⚠️ *MCP échoué : {content[:80]}*"
                        )
                except Exception as exc:
                    log.exception("MCP cognitive integration crashed")
                    cognitive_items.append(
                        f"⚠️ *Erreur MCP : {exc}*"
                    )
                finally:
                    yield {
                        "type": "phase_status",
                        "data": {"phase": "mcp", "state": "done"},
                    }

            # V5.1 — Python executor tool. Si le router a choisi
            # "python", on demande au LLM de générer le code, on
            # l'exécute en sandbox subprocess, et on injecte la sortie
            # dans le prompt comme contexte pour la réponse finale.
            #
            # Pipeline en 2 étapes parce qu'on n'a pas function calling
            # natif : (1) une passe LLM courte pour extraire/générer
            # le code, (2) exécution sandbox, (3) la passe principale
            # voit le résultat dans son contexte.
            if tool_chosen == "python" and self.model.is_loaded:
                yield {
                    "type": "phase_status",
                    "data": {"phase": "python", "state": "start"},
                }
                cognitive_items.append(
                    "🐍 *J'exécute du Python pour répondre…*"
                )
                try:
                    py_code = self._generate_python_code(user_intent)
                    if py_code:
                        from rune.tools.python_executor import (
                            run_with_auto_install as py_run, format_result,
                        )
                        py_result = py_run(py_code, timeout=10.0)  # V5.8.0 — 5s → 10s, V5.9.0 — auto-install
                        # V5.9.0 — Signal cognitif d'auto-installation
                        # avant le signal d'exécution. L'utilisateur voit
                        # explicitement quelles libs ont été installées
                        # pendant ce tour. Transparence sur l'autonomie.
                        _installed = py_result.get("installed_packages") or []
                        if _installed:
                            cognitive_items.append(
                                f"🔧 *Auto-installation : {', '.join(_installed)}*"
                            )
                        tool_result_text = (
                            "[Exécution Python]\n"
                            f"```python\n{py_code}\n```\n"
                            + format_result(py_result)
                        )
                        # V5.8.8 — Cadrage anti-hallucination en cas d'échec.
                        # Sans ça, le LLM tend à inventer "il manque matplotlib"
                        # ou "il faut installer X" sur n'importe quel exit_code_1,
                        # alors que les libs SONT installées dans le sandbox.
                        # On lui dit explicitement quoi faire selon le cas.
                        if not py_result.get("ok", False):
                            tool_result_text += (
                                "\n\n[Note système pour la réponse finale]\n"
                                "L'exécution a échoué. IMPORTANT : ne suggère "
                                "JAMAIS à l'utilisateur d'installer une lib "
                                "(matplotlib, numpy, sympy, etc. sont déjà "
                                "installées dans le sandbox). Si stderr montre "
                                "une vraie erreur (NameError, TypeError, etc.), "
                                "explique-la simplement. Sinon, dis que tu as "
                                "rencontré un problème technique et propose à "
                                "l'utilisateur de reformuler ou de calculer "
                                "manuellement les valeurs principales."
                            )
                        else:
                            # V5.9.3 — Cadrage anti-hallucination en cas de SUCCÈS.
                            # Bug observé in vivo : en mode raisonnement avec
                            # grounding web parasite, le LLM "ré-estime" les
                            # valeurs au lieu de copier stdout. Ex : stdout dit
                            # "41.5 37.5 24.49", la réponse finale dit "40.3,
                            # 41, 26.7". On ordonne explicitement de COPIER
                            # les valeurs telles quelles, sans recalcul.
                            #
                            # V5.9.4 — Renforcement : la Feuille de Route du
                            # mode raisonnement (générée AVANT l'exécution
                            # Python) peut contenir des calculs à la main
                            # erronés. On dit au LLM final d'IGNORER tout
                            # calcul du <reflexion> qui contredit stdout.
                            tool_result_text += (
                                "\n\n[Note système pour la réponse finale]\n"
                                "L'exécution Python a RÉUSSI et les valeurs "
                                "ci-dessus dans stdout sont les RÉSULTATS "
                                "EXACTS — calculés par un ordinateur, pas "
                                "estimés. Tu DOIS reprendre ces valeurs "
                                "telles quelles dans ta réponse finale à "
                                "l'utilisateur. NE LE FAIS PAS À LA MAIN.\n\n"
                                "RÈGLE ABSOLUE : si le bloc <reflexion> "
                                "(Feuille de Route, Corrections, etc.) "
                                "contient des chiffres qui contredisent "
                                "stdout, c'est <reflexion> qui se trompe "
                                "(il a été généré AVANT l'exécution). "
                                "Fais confiance UNIQUEMENT à stdout.\n\n"
                                "Si la sortie est un nombre avec décimales "
                                "(ex : 24.491495), tu peux arrondir à 2-3 "
                                "décimales pour la lisibilité (ex : 24.49), "
                                "mais sans changer la valeur fondamentale. "
                                "Si le contexte web récupéré ne correspond "
                                "pas à la question (résultats sans rapport), "
                                "IGNORE-le et concentre-toi UNIQUEMENT sur "
                                "le résultat Python."
                            )
                        # V5.8.0 — Signal cognitif post-exécution (toujours
                        # visible, même hors debug). Indique succès/échec
                        # et durée. Le panneau debug détaillé n'apparaît
                        # que si le toggle 🔬 est actif (fetch côté UI).
                        _ok = py_result.get("ok", False)
                        _dur = py_result.get("duration_ms", 0)
                        _icon = "✅" if _ok else "⚠️"
                        cognitive_items.append(
                            f"🐍 *Calcul exécuté en sandbox {_icon} ({_dur}ms)*"
                        )
                        # V5.8.0 — Plots stockés pour panneau debug.
                        # V5.8.7 — En plus, émis vers l'UI pour affichage
                        # inline dans la conversation (visible même hors
                        # debug). Permet aux utilisateurs de voir leurs
                        # graphiques sans devoir activer le mode 🔬.
                        self._last_python_plots = py_result.get("plots", []) or []
                        self._last_python_code = py_code
                        self._last_python_result = py_result
                        if self._last_python_plots:
                            yield {
                                "type": "python_plots",
                                "data": {
                                    "plots": self._last_python_plots,
                                    "count": len(self._last_python_plots),
                                },
                            }
                    else:
                        log.info("Python code generation returned empty")
                finally:
                    yield {
                        "type": "phase_status",
                        "data": {"phase": "python", "state": "done"},
                    }

            if web_context:
                rag_context = self._inject_web_context(web_context, rag_context)
                # Surface the numbered web sources to the UI so the user
                # can map [1], [2]… cited in the response back to their
                # actual URLs. Only fire when we actually have sources.
                web_sources = getattr(self.web_agent, "last_sources", []) or []
                if web_sources:
                    yield {
                        "type": "web_sources",
                        "data": {"sources": web_sources},
                    }

            # V5.1 — Inject Python tool result into rag_context if any.
            if tool_result_text:
                rag_context = (
                    rag_context + "\n\n" + tool_result_text
                    if rag_context else tool_result_text
                )

            if cognitive_items:
                yield {"type": "cognitive", "data": {"items": cognitive_items}}

            if self.debug_mode:
                yield from self._yield_debug_phase_a_b(
                    learn_result, rag_context, web_context, web_reason,
                    should_web, images,
                )

            # Image captioning
            effective_message = message
            for event in self._handle_image_captions(images or [], message):
                if event["type"] == "_caption_text":
                    if event["data"]:
                        effective_message = f"{message}\n\n{event['data']}"
                else:
                    yield event

            # V5.7.0/V5.7.1 — Vision active : détection sémantique du
            # besoin de zoomer sur une image du buffer visuel.
            # V5.7.1 — Migration regex → embeddings multilingues + garde-fou
            # anti-hallucination quand image présente mais pas de zoom.
            zoom_block: str = ""
            visual_warning_block: str = ""
            try:
                from rune.cognition.vision_active import (
                    detect_zoom_trigger, build_zoom_prompt, format_zoom_block,
                    looks_like_visual_question, build_visual_warning_block,
                )
                # Tick de décay sur le buffer visuel à chaque message
                self.visual_memory.decay_step()

                has_img_in_buffer = len(self.visual_memory) > 0
                trigger = detect_zoom_trigger(message, has_img_in_buffer)

                if trigger.triggered:
                    # Résoudre quelle image regarder (référence lexicale)
                    target_entry = self.visual_memory.find_by_reference(message)
                    if target_entry is not None:
                        # Construction du prompt VLM ciblé
                        zoom_prompt = build_zoom_prompt(
                            trigger.region_hint, message,
                        )
                        log.info(
                            "Vision zoom: image_id=%s region=%r category=%s conf=%.2f",
                            target_entry.image_id, trigger.region_hint,
                            trigger.category, trigger.confidence,
                        )
                        # Appel VLM ciblé
                        if self.captioner.ensure_loaded():
                            vlm_output = self.captioner.caption_focused(
                                target_entry.image_data,
                                focus_prompt=zoom_prompt,
                            )
                            if vlm_output:
                                zoom_block = format_zoom_block(
                                    trigger.region_hint, vlm_output,
                                )
                                # Enregistrer le zoom dans l'historique VWM
                                self.visual_memory.add_zoom(
                                    target_entry.image_id,
                                    trigger.region_hint,
                                    message,
                                    vlm_output,
                                )
                                # Signal cognitif UI
                                soft_marker = (
                                    " (souple)" if trigger.is_soft_trigger else ""
                                )
                                yield {"type": "cognitive", "data": {"items": [
                                    f"🔍 *Zoom cognitif{soft_marker} sur : {trigger.region_hint}*"
                                ]}}
                elif has_img_in_buffer and looks_like_visual_question(message):
                    # V5.7.1 — Garde-fou anti-hallucination (fix bugs 1, 2, 6).
                    # L'utilisateur fait référence à du contenu visuel mais
                    # le zoom n'a pas pu être déclenché. On injecte un
                    # avertissement pour que le LLM ne fabule pas.
                    target_entry = self.visual_memory.find_by_reference(message)
                    if target_entry is not None and target_entry.caption_initial:
                        visual_warning_block = build_visual_warning_block(
                            target_entry.caption_initial,
                        )
                        log.debug(
                            "Visual warning injected for image_id=%s "
                            "(question visuelle sans zoom)",
                            target_entry.image_id,
                        )
                        yield {"type": "cognitive", "data": {"items": [
                            "⚠️ *Mémoire visuelle : pas de zoom — je m'en tiens à ce que j'ai déjà vu*"
                        ]}}
            except Exception as exc:
                log.warning("Vision active failed: %s", exc, exc_info=True)

            # Si un zoom a produit du contenu, ou si un garde-fou est
            # nécessaire, on l'injecte dans le message effectif.
            if zoom_block:
                effective_message = f"{effective_message}\n\n{zoom_block}"
            if visual_warning_block:
                effective_message = f"{effective_message}\n\n{visual_warning_block}"

            # V4.4 — Strip control tags (/web, /noweb) from the message
            # before sending it to the LLM. They were used as routing
            # signals upstream (web_policy.should_search) but the model
            # shouldn't see them — they're not part of the question.
            # Use word boundaries to avoid mangling "/website" etc.
            import re as _re
            effective_message = _re.sub(
                r"(?<!\w)/(?:no)?web(?!\w)", "", effective_message
            ).strip()
            effective_message = _re.sub(r"\s+", " ", effective_message)

            # Phase C: Assemble prompt
            # V4: build the optional cognitive blocks. Order matters
            # for prompt readability: chronology first (factual
            # scaffolding), then plan (current goal), then
            # interpersonal state (interlocutor) and finally Lythéa's
            # own affect. Each block self-suppresses when empty.
            v4_blocks: list[str] = []
            try:
                if self._timeline_block:
                    v4_blocks.append(self._timeline_block)
                if self._planning_block:
                    v4_blocks.append(self._planning_block)
                if self._cognitive_state is not None:
                    user_block = self._cognitive_state.render_user_state_block()
                    if user_block:
                        v4_blocks.append(user_block)
                    self_block = self._cognitive_state.render_self_affect_block()
                    if self_block:
                        v4_blocks.append(self_block)
            except Exception:
                log.exception("V4 hook C (block assembly) failed")
                v4_blocks = []

            chat_messages = self._phase_c_assemble(
                effective_message, history, rag_context,
                last_message_ts=last_message_ts,
                session_created_ts=session_created_ts,
                v4_blocks=v4_blocks,
            )

            # V4.4 — Renforcement contextuel sur questions à risque
            # de confabulation (tech_reco). Observation A/B in vivo
            # (17 mai 2026, Qwen2.5-7B-Instruct) : sans raisonnement,
            # le modèle tend à sur-généraliser depuis une source
            # (« spaCy existe dans [4] » → invente fr_core_news_md/lg
            # comme variantes non vérifiées). Avec raisonnement, la
            # phase critique de DeepReasoningChain filtrait ces
            # extrapolations. Plutôt que de forcer le raisonnement
            # (coûteux : 17-50s vs instantané), on injecte une mini-
            # directive ciblée dans le system prompt qui discipline
            # le modèle sans coût additionnel. Le renforcement n'est
            # actif que sur tech_reco (où le bénéfice est démontré),
            # pas sur les autres raisons web pour ne pas alourdir
            # inutilement le prompt.
            if (
                should_web
                and "tech_reco" in (web_reason or "")
                and chat_messages
                and chat_messages[0].get("role") == "system"
            ):
                chat_messages[0]["content"] += (
                    "\n\n[Vigilance accrue]\n"
                    "Cette question demande une recommandation technique "
                    "(package, modèle, API, paper). Liste UNIQUEMENT les "
                    "noms qui apparaissent textuellement dans les "
                    "résultats web fournis ci-dessous. Ne propose pas de "
                    "« variantes plausibles » (sm/md/lg, base/large, "
                    "v1/v2) que tu n'as pas vues dans les sources : "
                    "elles peuvent ne pas exister. Si tu cites un nom "
                    "de mémoire sans le voir dans les résultats, dis-le "
                    "explicitement (« sans source précise dans les "
                    "résultats actuels »)."
                )

            if not self.model.is_loaded:
                yield {"type": "error", "data": {
                    "message": "Aucun modèle chargé", "code": "no_model",
                }}
                return

            # ── V3.9 cascade branch ────────────────────────────────
            # When the cascade is enabled and a model is loaded, route
            # generation through Gemini→local synthesis instead of
            # streaming directly. The cascade emits the same event
            # shape the UI already understands (partial → done) so the
            # frontend works without modification.
            if self.cascade_enabled:
                yield from self._run_cascade_path(
                    chat_messages=chat_messages,
                    message=message,
                    learn_result=learn_result,
                    cancelled=cancelled,
                )
                return

            # Phase D-pre: two-pass reasoning for standard models.
            #
            # Skip if images are present: the captioner adds
            # ``[Image N : ...]`` blocks to ``effective_message`` but the
            # reasoning prompt does not advertise this convention to
            # the model. Without that advertisement the reasoning pass
            # tends to deny having any image (it doesn't recognise the
            # caption convention as a real description), and the main
            # generation pass then sees a confused "I have no image"
            # in its history and derails — typically by ignoring the
            # image entirely and answering on whatever happens to be in
            # the RAG context. Easier and more robust to skip the
            # reasoning pass for image turns. Observed in prod testing
            # with Qwen2.5-3B-Instruct + reasoning toggle + photo.
            has_images = bool(images or [])
            force_reasoning = (
                self.reasoning_enabled
                and not self.model.is_thinking
                and not has_images
            )
            # If the user toggled reasoning ON but we're skipping it
            # because of images, surface a cognitive notice so they
            # don't think the toggle is broken.
            if (
                self.reasoning_enabled
                and not self.model.is_thinking
                and has_images
            ):
                yield {"type": "cognitive", "data": {"items": [
                    "🧠 *Réflexion désactivée pour cette image*",
                ]}}

            if force_reasoning:
                # Pulse pill while the reasoning pass is running.
                yield {
                    "type": "phase_status",
                    "data": {"phase": "thinking", "state": "start"},
                }
                # The deep reasoning chain reports its sub-steps via
                # this callback. We can't yield from inside a callback
                # (it's called deep in the chain), so we buffer the
                # labels and the chain stays silent on the pill text —
                # the pill just keeps pulsing "Réflexion…". Buffered
                # labels are logged for debugging.
                _step_labels: list[str] = []
                def _on_reasoning_step(label: str) -> None:
                    _step_labels.append(label)
                    log.debug("Deep reasoning step: %s", label)

                try:
                    reasoning_text = self._generate_reasoning(
                        effective_message,
                        chat_messages,
                        surprise=learn_result.get("surprise"),
                        on_step=_on_reasoning_step,
                        web_context=web_context,
                    )
                finally:
                    yield {
                        "type": "phase_status",
                        "data": {"phase": "thinking", "state": "done"},
                    }
                if reasoning_text:
                    yield {"type": "reasoning", "data": {"text": reasoning_text}}
                    chat_messages.append({
                        "role": "assistant",
                        "content": f"<reflexion>{reasoning_text}</reflexion>",
                    })
                    chat_messages.append({
                        "role": "user",
                        "content": "Maintenant donne ta réponse finale.",
                    })

            # Phase D: stream
            entropies: list[float] = []
            raw_text = ""
            last_clean = ""
            last_reasoning = ""
            # Thinking pill state: opens at first reasoning chunk
            # (native thinking models), closes at first clean chunk
            # (when the model exits <think> and starts answering).
            # End-of-stream safety net also closes it.
            thinking_pill_open = False

            # V5.5.8 — Streaming-time greeting suppression.
            # V5.5.7 strippait la salutation *après* la fin du stream,
            # ce qui causait un flash visible côté UI (Lythéa affichait
            # "Salut Cédric, comment ça va ?" pendant 1-2s puis se
            # reprenait). On intercepte maintenant DÈS le streaming :
            # tant qu'on est dans une phase d'amorce salutation, on
            # ne yield pas, puis on yield uniquement le contenu utile.
            #
            # Trigger : Chroma ≥ 3 docs = conversation en cours.
            # États :
            #   - greeting_probe = True (initial) : on bufferise sans
            #     yield, en attendant de savoir si c'est une salutation
            #   - greeting_probe = False : flux normal (soit la salu
            #     a été coupée, soit on a confirmé qu'il n'y en avait
            #     pas)
            _streaming_strip_greeting = False
            try:
                _chroma_count = (
                    self.chroma.count() if self.chroma is not None else 0
                )
                _streaming_strip_greeting = _chroma_count >= 3
            except Exception:
                pass
            greeting_probe = _streaming_strip_greeting
            greeting_stripped_offset = 0  # combien de chars suppressed
            # Buffer max : on coupe la phase probe au-delà de 200 chars
            # pour éviter de bloquer le streaming si le modèle s'égare.
            _GREETING_PROBE_MAX = 200

            # Désactive le <think> natif pour les messages triviaux (salut,
            # merci, ok…) : un modèle thinking raisonne sur TOUT, ce qui le
            # fait « bavarder » pour un simple bonjour. On force
            # enable_thinking=False ponctuellement — le contexte trivial est
            # court, donc le flag tient (contrairement aux requêtes à gros
            # contexte où il peut fuir).
            _think_override = None
            try:
                from rune.cognition.deliberation import is_trivial_message
                _last_user = next(
                    (m.get("content", "") for m in reversed(chat_messages)
                     if m.get("role") == "user"), "")
                if is_trivial_message(_last_user):
                    _think_override = False
            except Exception:
                _think_override = None

            for chunk in self.model.stream_generate(
                chat_messages,
                max_new_tokens=MAX_NEW_TOKENS,
                cancelled=cancelled,
                sampling=self.sampling_profile,
                think_override=_think_override,
                # V5.6.15 — Passe les PIL images directement au modèle
                # quand il est natively multimodal (Gemma 3/4). Si non
                # multimodal, ce paramètre est ignoré et les images
                # sont déjà devenues du texte via _handle_image_captions.
                pil_images=(
                    images if (
                        images
                        and getattr(self.model, "is_natively_multimodal", False)
                    ) else None
                ),
            ):
                if cancelled and cancelled.is_set():
                    break

                # V3.9.4: surface generation errors and recoveries to
                # the UI rather than swallowing them. Previously the
                # streamer would just stop on inf/nan and the UI would
                # display a truncated mid-sentence answer with no
                # explanation. Now we emit explicit cognitive events.
                if chunk.get("error") == "generation_unstable":
                    yield {"type": "cognitive", "data": {"items": [
                        f"⚠️ *{chunk.get('error_detail', 'Erreur de génération')}*",
                    ]}}
                    raw_text = chunk.get("text", raw_text)
                    break

                if chunk.get("recovered_from_nan"):
                    yield {"type": "cognitive", "data": {"items": [
                        "🔄 *Récupération automatique d'une instabilité numérique*",
                    ]}}

                raw_text = chunk["text"]
                ent = chunk.get("entropy", 0.0)
                entropies.append(ent)

                clean, reasoning = strip_reasoning(
                    raw_text,
                    allow_unclosed=self.model.is_thinking,
                )
                clean = _mask_open_tags_fn(clean)

                if reasoning and reasoning != last_reasoning:
                    if not thinking_pill_open:
                        # First reasoning chunk → open thinking pill.
                        yield {
                            "type": "phase_status",
                            "data": {"phase": "thinking", "state": "start"},
                        }
                        thinking_pill_open = True
                    yield {"type": "reasoning", "data": {"text": reasoning}}
                    last_reasoning = reasoning

                # V5.5.8 — Greeting probe : on intercepte avant le yield.
                # Si on est en probe et que clean matche encore le
                # début d'un pattern salutation, on ne yield pas. Si
                # on a dépassé la salutation OU si on dépasse la limite
                # de probing, on flush et on bascule en mode normal.
                if greeting_probe and clean:
                    stripped, done_probing = self._probe_strip_greeting(
                        clean, _GREETING_PROBE_MAX,
                    )
                    if done_probing:
                        # Sortie de la phase probe — soit la salutation
                        # a été coupée (stripped < clean), soit on a
                        # dépassé la limite sans rien strip.
                        greeting_probe = False
                        if stripped != clean:
                            greeting_stripped_offset = (
                                len(clean) - len(stripped)
                            )
                            log.info(
                                "Streaming greeting stripped: %d chars "
                                "(at first emit)",
                                greeting_stripped_offset,
                            )
                        # On émet le contenu utile (potentiellement vide
                        # si on n'a que la salutation au début — dans
                        # ce cas le prochain tour émettra).
                        if stripped and stripped != last_clean:
                            if thinking_pill_open:
                                yield {
                                    "type": "phase_status",
                                    "data": {"phase": "thinking", "state": "done"},
                                }
                                thinking_pill_open = False
                            last_clean = stripped
                            yield {"type": "partial", "data": {"text": stripped}}
                    # else: on continue à bufferiser silencieusement
                elif clean and clean != last_clean:
                    # Mode normal (probe terminé ou pas activé).
                    # On applique l'offset si une salutation a été
                    # strippée au début, pour rester cohérent avec ce
                    # qui a été émis.
                    emitted_clean = clean
                    if greeting_stripped_offset > 0:
                        emitted_clean = clean[greeting_stripped_offset:]
                        # V5.5.8 — Préserver la capitalisation du
                        # premier mot après strip pour cohérence
                        # visuelle (sinon le streaming alterne
                        # "Comment" puis "comment" puis "comment...").
                        if emitted_clean and emitted_clean[0].islower():
                            emitted_clean = (
                                emitted_clean[0].upper() + emitted_clean[1:]
                            )
                    if emitted_clean and emitted_clean != last_clean:
                        # First clean chunk: close thinking pill if still
                        # open (thinking model just finished its <think>).
                        if thinking_pill_open:
                            yield {
                                "type": "phase_status",
                                "data": {"phase": "thinking", "state": "done"},
                            }
                            thinking_pill_open = False
                        last_clean = emitted_clean
                        yield {
                            "type": "partial",
                            "data": {"text": emitted_clean},
                        }

                if len(entropies) % _ENTROPY_EMIT_EVERY == 0:
                    window = entropies[-ENTROPY_DOUBT_WINDOW:]
                    yield {"type": "entropy", "data": {
                        "value": round(ent, 3),
                        "window_mean": round(sum(window) / max(len(window), 1), 3),
                    }}

                # Asymmetric SDM write — confident output tokens get
                # committed to working memory at reduced strength.
                self._maybe_write_output_token(chunk.get("latent"), ent)

            # Final clean for memory writes. allow_unclosed=is_thinking
            # because if a thinking model crashed mid-<think>, we want
            # to recover what was generated. For classic LLMs, the
            # default False is safer (don't truncate at literal tags).
            final_text, final_reasoning = strip_reasoning(
                raw_text,
                allow_unclosed=self.model.is_thinking,
            )
            if final_reasoning and final_reasoning != last_reasoning:
                yield {"type": "reasoning", "data": {"text": final_reasoning}}

            # V5.5.7 — Anti-resalutation déterministe.
            # Le SYSTEM_PROMPT V5.5.6 demande au modèle de ne pas
            # resaluer quand la conversation est en cours, mais Qwen
            # 7B Instruct continue malgré l'instruction explicite
            # (politesses sur-apprises pendant RLHF). On strip la
            # salutation au début de la réponse quand on sait qu'il
            # y a déjà du contexte mémoire (≥ 3 docs Chroma archivés
            # = pas une première rencontre).
            #
            # Patterns ciblés : Salut/Bonjour/Coucou/Hey + variantes
            # avec virgule, prénom, ou question polie type "comment
            # ça va ?".
            try:
                _chroma_count = (
                    self.chroma.count() if self.chroma is not None else 0
                )
                if _chroma_count >= 3 and final_text:
                    final_text = self._strip_leading_greeting(final_text)
            except Exception:
                pass  # never break on cleanup

            # Close any pill still marked as open. Without this, a
            # stream that emits only reasoning then ends (e.g. error
            # mid-thinking) would leave the thinking pill pulsing
            # indefinitely on the client.
            if thinking_pill_open:
                yield {
                    "type": "phase_status",
                    "data": {"phase": "thinking", "state": "done"},
                }
                thinking_pill_open = False

            # Compute doubt/epistemic up front so V4.4 metacognition
            # can use them. Cheap operation (already done in V3.9.4
            # below — just hoisted).
            doubt_index, epistemic = self.surprise_phase.doubt_from_entropies(
                entropies, self.entropy_threshold,
            )

            # V4.4 — metacognition (Hook M). Runs AFTER doubt is known
            # but BEFORE inhibition so any hedge prefix it suggests
            # appears in the version inhibition checks. The hedge is
            # only applied when ``metacog_apply_hedge`` is ON; otherwise
            # the decision is recorded for telemetry only.
            if self._metacognition is not None:
                try:
                    kg_facts_count = 0
                    try:
                        kg_facts_count = len(self.kg.list_active_facts() or [])
                    except Exception:
                        kg_facts_count = 0
                    meta_dec = self._metacognition.observe(
                        doubt_index=doubt_index,
                        epistemic=epistemic,
                        web_used=bool(web_context),
                        kg_facts_count=kg_facts_count,
                    )
                    self._last_meta_decision = meta_dec
                    if (
                        self._metacognition.config.apply_hedge
                        and meta_dec.hedge_prefix
                        and final_text
                        and not final_text.startswith(meta_dec.hedge_prefix)
                    ):
                        final_text = meta_dec.hedge_prefix + final_text
                except Exception:
                    log.exception("V4.4 hook M (metacognition) failed")

            # V4.0.b — inhibition cascade. Runs on the final cleaned
            # text, after reasoning is stripped. We never auto-rewrite
            # output here (we'd lose streamed content); instead we
            # surface annotations to the operator via cognitive_items.
            # Hard-rule blocks (N1 strict) replace final_text with a
            # neutral placeholder so the user sees something rather
            # than a half-completed sensitive payload.
            if self._inhibition is not None:
                try:
                    kg_facts: list[dict] = []
                    try:
                        kg_facts = [
                            f for f in self.kg.list_active_facts()
                            if isinstance(f, dict)
                            and float(f.get("confidence", 0.0) or 0.0) >= 0.7
                        ]
                    except Exception:
                        kg_facts = []
                    inh = self._inhibition.check(final_text or "", kg_facts=kg_facts)
                    if not inh.passed and inh.action == "block":
                        log.warning(
                            "V4 inhibition BLOCK — level=%s reason=%s",
                            inh.level, inh.reason,
                        )
                        # Replace with a neutral message; preserve
                        # post_generation flow so memory writes proceed.
                        final_text = (
                            "*[Réponse inhibée par filtre de sécurité — "
                            f"{inh.level}: {inh.reason}]*"
                        )
                    elif inh.action in ("annotate", "rewrite") and inh.matched:
                        log.info(
                            "V4 inhibition %s — level=%s matched=%s",
                            inh.action, inh.level, inh.matched,
                        )
                except Exception:
                    log.exception("V4 hook E.1 (inhibition) failed")

            # V5.5 — Reflection loop. Critique sélective sur cas à risque
            # (tech_reco, complexity≥3, CRAG INCORRECT, doute accumulé).
            # Désactivée si raisonnement actif, Python utilisé, ou
            # réponse courte. Critique → JSON verdict → applique
            # révision si proposée.
            try:
                from rune.cognition.reflection import (
                    reflect_on_response,
                    cognitive_item_for as reflect_cog_item,
                    ReflectionContext,
                )
                from rune.settings import get_settings
                _s = get_settings()
                if _s.reflection_enabled:
                    ref_ctx = ReflectionContext(
                        query=user_intent or message,
                        response=final_text or "",
                        web_reason=web_reason or "",
                        complexity_steps=0,  # router complexity non exposé ici
                        crag_status="",  # TODO V5.5.1 : exposer depuis ctx
                        reasoning_active=bool(locals().get("reasoning_text")),
                        tool_used=tool_chosen or "",
                    )
                    verdict = reflect_on_response(
                        ref_ctx, self.model,
                        timeout=_s.reflection_timeout_s,
                    )
                    # Surface cognitive item to UI
                    cog_msg = reflect_cog_item(verdict)
                    if cog_msg:
                        yield {"type": "cognitive", "data": {"items": [cog_msg]}}
                    # Si révision proposée et non vide → on remplace
                    if verdict.needs_revision and verdict.revised_response:
                        log.info(
                            "Reflection applied revision: %d issues, "
                            "%d→%d chars",
                            len(verdict.issues),
                            len(final_text), len(verdict.revised_response),
                        )
                        final_text = verdict.revised_response
                        # Émettre la version révisée comme "partial"
                        # pour que la UI mette à jour le bloc affiché
                        yield {
                            "type": "partial_revised",
                            "data": {"text": final_text},
                        }
            except Exception:
                log.exception("V5.5 reflection hook failed")

            # V5.7.0 — Détection métacognitive d'incertitude perceptive.
            # Si la réponse contient des marqueurs comme "je ne distingue
            # pas", "il semble", "peu lisible", on stocke le flag pour
            # proposer un zoom au tour suivant.
            try:
                from rune.cognition.vision_active import detect_perceptual_uncertainty
                had_image_this_turn = bool(images) or len(self.visual_memory) > 0
                if had_image_this_turn and final_text:
                    self._last_perceptual_uncertainty = detect_perceptual_uncertainty(final_text)
                    if self._last_perceptual_uncertainty:
                        log.debug(
                            "Perceptual uncertainty detected in response — "
                            "zoom suggestion can be offered next turn"
                        )
                        yield {"type": "cognitive", "data": {"items": [
                            "💡 *Je peux essayer de mieux voir une zone précise — pointe-la moi si tu veux*"
                        ]}}
                else:
                    self._last_perceptual_uncertainty = False
            except Exception as exc:
                log.warning("Perceptual uncertainty detection failed: %s", exc)

            yield from self._post_generation(
                message, final_text, doubt_index, epistemic, learn_result,
            )

            # V6.0.0-rc — Phase 2 : détection des artefacts dans la
            # réponse finale, écriture dans le workspace, émission
            # des offers SSE. La détection est conservatrice (≥10
            # lignes pour du Python, ≥2500 chars/4 titres pour du
            # markdown) pour ne pas flooder le workspace avec des
            # petits snippets. Voir cognition/mcp_integration.py.
            #
            # V6.0.0-rc rev6 : on SKIP la détection quand le tour a
            # utilisé MCP read (read_extracted ou read). La réponse de
            # Lythéa est alors par construction un commentaire/résumé
            # du fichier source, pas un livrable indépendant. Sans ce
            # skip, on créait un faux "rapport.md" à chaque "Lis X.xlsx
            # et résume-le" parce que le résumé est long et structuré.
            # V6.0.0-rc rev6+ : skip Phase 2 sur tour MCP, sauf si
            # l'utilisateur a EXPLICITEMENT demandé un livrable.
            #
            # Cas skippés (comportement de rev6) : "Lis X.xlsx et
            # résume-le" → la réponse est un commentaire du fichier,
            # pas un livrable à part. On évite ainsi le faux rapport.md.
            #
            # Cas autorisés (corrigé rev7) : "Lis X.xlsx et génère-moi
            # un rapport markdown" → l'utilisateur veut clairement un
            # fichier en sortie. On laisse detect_artefacts décider
            # (le filtre _user_wants_file y est aussi appliqué).
            try:
                if tool_chosen != "mcp":
                    yield from self._emit_workspace_offers(
                        final_text, user_intent=user_intent or "",
                    )
                else:
                    # Sur tour MCP, on vérifie si l'utilisateur a
                    # demandé un livrable. Si oui, on laisse passer.
                    try:
                        from rune.cognition.mcp_integration import (
                            _user_wants_file,
                        )
                        if _user_wants_file(user_intent or ""):
                            log.debug(
                                "Phase 2 sur tour MCP : utilisateur a "
                                "demandé un livrable, on autorise"
                            )
                            yield from self._emit_workspace_offers(
                                final_text, user_intent=user_intent or "",
                            )
                        else:
                            log.debug(
                                "Phase 2 skipped : tour MCP read → "
                                "réponse considérée comme commentaire, "
                                "pas livrable"
                            )
                    except Exception:
                        # Fallback safe : skip en cas d'erreur d'import
                        log.debug("Phase 2 skipped : tour MCP read")
            except Exception as _exc:
                log.warning("Workspace offers emission failed: %s", _exc)

            yield {"type": "done", "data": {
                "final_text": final_text,
                "doubt_index": round(doubt_index, 3),
                "epistemic": epistemic,
            }}

            if self.debug_mode:
                yield self._build_debug_post_generation(
                    doubt_index, epistemic, entropies,
                )

        except Exception as exc:
            log.exception("Hippocampe process error")
            yield {"type": "error", "data": {"message": str(exc), "code": "internal"}}

    # ── V3.9 cascade execution ──────────────────────────────────────

    def _run_cascade_path(
        self,
        chat_messages: list[dict[str, str]],
        message: str,
        learn_result: dict,
        cancelled: Any | None,
    ) -> Generator[dict, None, None]:
        """Execute the cascade in place of the streaming path.

        The cascade is a blocking call (Gemini API + one local pass for
        synthesis). To preserve the UI contract — which expects at least
        one ``partial`` event before ``done`` — we emit a single
        ``partial`` with the final text once the cascade returns, then
        the standard ``post_generation`` flow.

        Reasoning toggle, image handling, and per-token entropy reports
        are not meaningful in cascade mode (we don't have token-level
        signals from the Gemini side). Doubt is computed conservatively
        from the synthesis output entropy when synthesis runs, or from
        a fixed prior when it doesn't. The UI shows the result with the
        same widgets either way.

        Failure-safe: if the cascade returns ``fallback_used=True`` and
        the local generator also failed (final_text empty), we surface
        an explicit error to the UI so the user knows.
        """
        if cancelled is not None and cancelled.is_set():
            return

        # Split the chat history into system_prompt (first message if it
        # carries the role 'system') + the rest. Lythéa builds chat
        # messages with the system block at index 0; pass it as Gemini's
        # systemInstruction.
        system_prompt = ""
        rest: list[dict[str, str]] = []
        for msg in chat_messages:
            if msg.get("role") == "system" and not system_prompt:
                system_prompt = msg.get("content", "")
                continue
            rest.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        try:
            result = self._cascade.generate(
                system_prompt=system_prompt,
                messages=rest,
            )
        except Exception as exc:
            log.exception("Cascade execution crashed: %s", exc)
            yield {"type": "error", "data": {
                "message": f"Cascade crashed: {exc}",
                "code": "cascade_crash",
            }}
            return

        if not result.final_text:
            # Both Gemini and the local fallback returned nothing — bail.
            yield {"type": "error", "data": {
                "message": (
                    f"Cascade and fallback both empty "
                    f"(reason={result.fallback_reason or 'unknown'})"
                ),
                "code": "cascade_empty",
            }}
            return

        # Surface fallback to the UI as a cognitive notice (not an
        # error) so the user knows generation degraded gracefully.
        if result.fallback_used:
            yield {"type": "cognitive", "data": {"items": [
                f"⚠️ *Cascade dégradée — fallback local "
                f"({result.fallback_reason})*",
            ]}}
        elif result.synthesised:
            yield {"type": "cognitive", "data": {"items": [
                "🌀 *Réponse synthétisée à partir d'un draft Gemini*",
            ]}}

        # Emit the final text as a single partial then close.
        yield {"type": "partial", "data": {"text": result.final_text}}

        # Doubt index in cascade mode is computed from a conservative
        # prior. We don't have per-token entropies from Gemini.
        doubt_index = 0.30 if result.synthesised else 0.20
        epistemic = "fait" if doubt_index < 0.30 else "intuition"

        yield from self._post_generation(
            message, result.final_text, doubt_index, epistemic,
            learn_result,
        )

        # V3.9.4: surface live quota counters so the UI can display
        # "Gemini quota: 7/1500 daily, 3/8 per minute" alongside the
        # response. Helps the user see when they're approaching limits
        # before the cascade falls back unexpectedly.
        cascade_block = {
            "synthesised": result.synthesised,
            "fallback_used": result.fallback_used,
            "fallback_reason": result.fallback_reason,
            "gemini_input_tokens": result.gemini_input_tokens,
            "gemini_output_tokens": result.gemini_output_tokens,
        }
        if self._cascade is not None and self._cascade._gemini is not None:
            g = self._cascade._gemini
            cascade_block["quota_used_today"] = g.quota_used
            cascade_block["quota_remaining_today"] = g.quota_remaining
            cascade_block["quota_used_per_min"] = g.per_minute_used
            cascade_block["quota_seconds_until_slot"] = round(
                g.per_minute_seconds_until_slot, 1,
            )

        yield {"type": "done", "data": {
            "final_text": result.final_text,
            "doubt_index": round(doubt_index, 3),
            "epistemic": epistemic,
            "cascade": cascade_block,
        }}

    # ── Post-generation ────────────────────────────────────────────────

    # V5.5.7 — Greeting patterns à supprimer en début de réponse quand
    # la mémoire contient déjà des échanges. Chaque pattern matche :
    #   - mot d'accueil (Salut/Bonjour/Coucou/Hey/Hello/Hi)
    #   - optionnellement suivi d'un prénom et/ou virgule
    #   - éventuellement suivi de questions polies type "comment ça
    #     va ?" / "tu vas bien ?" / "ça va ?" / "comment vas-tu ?"
    #     / "tu as passé une bonne journée ?"
    # Re.IGNORECASE + on match jusqu'au premier point/virgule/saut de
    # ligne qui finit la formule, puis on strip ce préfixe.
    _GREETING_STRIP_RE = re.compile(
        r"^\s*"
        # Mot d'accueil
        r"(?:salut|bonjour|coucou|hey|hello|hi)\s*"
        # Prénom ou virgule optionnels
        r"(?:[A-ZÀ-Ö][a-zA-ZÀ-ÖØ-öø-ÿ\-]+[\s,]*)?"
        # Questions polies optionnelles (1 ou 2 enchaînées)
        r"(?:"
        r"(?:comment\s+(?:ça\s+va|vas-tu|tu\s+vas)\s*[?!.]?\s*)"
        r"|(?:ça\s+va\s*[?!.]?\s*)"
        r"|(?:tu\s+vas\s+bien\s*[?!.]?\s*)"
        r"|(?:j['’]espère\s+que\s+tu\s+vas\s+bien\s*[?!.]?\s*)"
        r"|(?:tu\s+as\s+passé\s+une\s+bonne\s+(?:journée|soirée|matinée)\s*[?!.]?\s*)"
        r"|(?:how\s+are\s+you\s*[?!.]?\s*)"
        r"){0,2}"
        # Fin du préfixe : ponctuation ou saut de ligne
        r"[.,;:!?\s]*",
        re.IGNORECASE,
    )

    def _strip_leading_greeting(self, text: str) -> str:
        """V5.5.7 — Supprime la salutation en début de réponse.

        Appelé après strip_reasoning sur final_text, uniquement quand
        Chroma contient ≥ 3 documents (= conversation en cours, pas
        première rencontre).

        Si la suppression rendrait la réponse vide ou < 5 caractères,
        on garde la version originale (mieux vaut une salutation
        qu'une bouillie). Sinon on remet une majuscule en début et
        on supprime la virgule orpheline éventuelle.

        Parameters
        ----------
        text : str
            Réponse finale du modèle.

        Returns
        -------
        str
            Texte nettoyé, ou texte original si la suppression ferait
            plus de mal que de bien.
        """
        if not text or len(text) < 10:
            return text
        match = self._GREETING_STRIP_RE.match(text)
        if not match:
            return text
        prefix_len = match.end()
        if prefix_len == 0:
            return text
        remainder = text[prefix_len:].lstrip(" ,;:!?")
        # Sanity : si on a tout détruit, on garde l'original
        if len(remainder) < 5:
            return text
        # Remettre une majuscule en début si nécessaire
        if remainder and remainder[0].islower():
            remainder = remainder[0].upper() + remainder[1:]
        log.info(
            "Greeting stripped: %d chars removed from leading",
            prefix_len,
        )
        return remainder

    # V5.5.8 — Préfixes "amorce salutation" : caractères qui pourraient
    # encore être le début d'une salutation. Tant que le buffer ne
    # commence par aucun de ces préfixes (case-insensitive), on est sûr
    # qu'il n'y a pas de salutation et on peut sortir du probe.
    _GREETING_PREFIXES = (
        "salut", "bonjour", "coucou", "hey", "hello", "hi",
    )

    def _probe_strip_greeting(
        self, buffered: str, max_probe_chars: int = 200,
    ) -> tuple[str, bool]:
        """V5.5.8 — Logique de probing pour streaming.

        Appelé à chaque chunk du streaming tant qu'on est dans la
        phase "probing". Décide si on doit continuer à attendre
        (salutation potentielle en cours d'écriture) ou si on peut
        sortir du probe (salutation complète détectée, ou pas de
        salutation du tout).

        Parameters
        ----------
        buffered : str
            Le texte cumulé reçu jusqu'ici (déjà strip_reasoning).
        max_probe_chars : int
            Limite dure : au-delà, on sort du probe quoi qu'il arrive
            (sécurité anti-blocage si le modèle s'égare).

        Returns
        -------
        tuple[str, bool]
            ``(text_to_emit, done_probing)``.
            - Si ``done_probing=False`` : on est toujours dans
              une amorce de salutation, on ne yield rien.
              ``text_to_emit`` est vide.
            - Si ``done_probing=True`` : on quitte le probe.
              ``text_to_emit`` est :
                * Le buffer après strip si une salutation complète
                  a été détectée + contenu utile derrière.
                * Le buffer entier si aucune salutation n'est
                  détectable (ou si on dépasse la limite).
        """
        if not buffered:
            return "", False  # rien à faire encore

        # Limite dure : sortie forcée
        if len(buffered) >= max_probe_chars:
            stripped = self._strip_leading_greeting(buffered)
            return stripped, True

        lower = buffered.lstrip().lower()

        # Cas 1 : Le buffer ne commence par AUCUN préfixe possible
        # de salutation → certain qu'il n'y a pas de salutation,
        # on sort du probe immédiatement avec le texte tel quel.
        starts_with_greeting = any(
            lower.startswith(p) for p in self._GREETING_PREFIXES
        )
        if not starts_with_greeting:
            # Cas piège : le buffer pourrait commencer par un préfixe
            # incomplet, par ex. "Sa" qui est le début de "Salut". On
            # vérifie qu'un préfixe complet pourrait encore être en
            # train de se former.
            could_become_greeting = any(
                p.startswith(lower[:len(p)]) and len(lower) < len(p)
                for p in self._GREETING_PREFIXES
            )
            if not could_become_greeting:
                # Sûr : pas de salutation, on flush direct.
                return buffered, True
            # Sinon : on attend encore quelques caractères pour
            # voir si ça devient une vraie salutation.
            return "", False

        # Cas 2 : Le buffer commence par un préfixe de salutation.
        # On tente le strip complet. Si le strip enlève quelque chose
        # ET qu'il reste du contenu utile derrière, on a la salutation
        # complète → sortie du probe.
        stripped = self._strip_leading_greeting(buffered)
        if stripped != buffered and len(stripped) >= 5:
            return stripped, True

        # Cas 3 : Le buffer commence par une salutation mais on n'a
        # pas encore reçu la suite (pas encore de contenu utile
        # derrière). On continue d'attendre.
        return "", False

    def _get_recent_exchanges(self, n: int = 20) -> list[dict]:
        """V5.4 — Provider d'exchanges récents pour la consolidation.

        Retourne les N derniers couples (user, assistant) bufferisés
        par _post_generation. Utilisé par le microsleep pour extraire
        des patterns procéduraux réutilisables.

        Returns
        -------
        list[dict]
            Liste de ``{"role": "user|assistant", "content": str}``.
            Vide si rien n'a encore été bufferisé.
        """
        try:
            buf = list(self._recent_exchanges_buffer)
            return buf[-n:] if len(buf) > n else buf
        except Exception:
            return []

    def _post_generation(
        self,
        query: str,
        response: str,
        doubt_index: float,
        epistemic: str,
        learn_result: dict,
    ) -> Generator[dict, None, None]:
        """Archive (Chroma+MHN) + bookkeeping + microsleep trigger."""
        # V5.4 — Bufferise le couple (user, assistant) pour que la
        # consolidation puisse extraire des patterns procéduraux au
        # prochain microsleep. Cap à 20 (maxlen de la deque), ce qui
        # couvre largement la fenêtre d'extraction (10 exchanges).
        try:
            self._recent_exchanges_buffer.append(
                {"role": "user", "content": query[:1000]}
            )
            self._recent_exchanges_buffer.append(
                {"role": "assistant", "content": response[:1000]}
            )
        except Exception as exc:
            log.debug("Recent exchanges buffer append failed: %s", exc)

        if self._should_archive(query, response, learn_result):
            self.storage_phase.archive_exchange(
                query=query, response=response,
                entities=learn_result.get("entities", []),
                surprise=learn_result.get("surprise", {}),
                doubt_index=doubt_index, epistemic=epistemic,
            )

        # Tick + ripple bookkeeping + microsleep + inactivity timer
        self.exchange_count += 1
        self.last_activity = time.time()

        global_surprise = float(
            learn_result.get("surprise", {}).get("global", 0.0)
        )

        # V4.1 — pass affect signal + most recent MHN pattern index so
        # the consolidation phase can flag it for boosted replay when
        # arousal crosses the configured threshold. All four kwargs are
        # additive: when V4.1 is off, ConsolidationPhase ignores them.
        affect_intensity = 0.0
        affect_arousal = 0.0
        last_pattern_idx: int | None = None
        try:
            if self._cognitive_state is not None:
                cur = self._cognitive_state.lythea_affect.current
                affect_intensity = float(cur.intensity)
                affect_arousal = float(cur.arousal)
            if hasattr(self.mhn, "n_stored") and self.mhn.n_stored > 0:
                last_pattern_idx = int(self.mhn.n_stored) - 1
        except Exception:
            log.exception("V4.1 hook E.2 (affect signal) failed")
            affect_intensity = 0.0
            affect_arousal = 0.0
            last_pattern_idx = None

        self.consolidation_phase.record_event(
            global_surprise,
            affect_intensity=affect_intensity,
            affect_arousal=affect_arousal,
            last_pattern_idx=last_pattern_idx,
        )
        self.consolidation_phase.maybe_trigger_after_exchange(self.exchange_count)
        self.consolidation_phase.reset_inactivity_timer()

        yield from ()

    # ── Microsleep wrappers (preserved for backwards compat) ───────────

    def _trigger_microsleep(self) -> None:
        """Wrapper — delegates to :class:`ConsolidationPhase`."""
        self.consolidation_phase.trigger_microsleep()

    def _microsleep(self) -> None:
        """Wrapper — runs the microsleep body inline (no thread).

        The original method was the *body* run inside a worker
        thread by ``_trigger_microsleep``. External callers (tests)
        may invoke it synchronously, so we expose the same shape
        by calling the consolidation-phase internal directly.
        """
        # pylint: disable=protected-access
        self.consolidation_phase._run_microsleep()

    def _reset_inactivity_timer(self) -> None:
        """Wrapper — delegates to :class:`ConsolidationPhase`."""
        self.consolidation_phase.reset_inactivity_timer()

    # ── Deep sleep (manual) ────────────────────────────────────────────

    def deep_sleep(self) -> str:
        """Manual deep consolidation — delegates to :class:`ConsolidationPhase`."""
        self._retention_gc()
        return self.consolidation_phase.deep_sleep()

    def _retention_gc(self) -> None:
        """V5.9 — Purge les échanges ``exchange`` non consultés depuis
        ``RETENTION_TTL_DAYS`` jours. Les ``consolidated`` sont épargnés
        (permanents, type différent hors du champ de la requête).
        Best-effort : n'interrompt jamais le deep sleep.

        Sûreté (5 modes de défaillance fermés explicitement) :
        1. Refresh d'accès câblé sur les chunks POST-CRAG réellement
           injectés (voir retrieval._gather_semantic), pas un sous-chemin.
        2. Timestamp absent → ``ref`` tombe sur ``ts`` (présent sur les
           anciens chunks) puis 0 ; le garde ``ref and ref < cutoff``
           traite 0 comme faux → chunk pré-migration JAMAIS purgé.
        3. Ne tourne qu'au deep_sleep, lui-même déclenché par le
           ConsolidationScheduler (seuil atteint) — pas jamais, pas à
           chaque tour.
        4. ``consolidated`` a un type distinct ("consolidated") : la
           requête ``where={"type": "exchange"}`` ne peut physiquement
           pas les voir, indépendamment d'un quelconque flag.
        5. DRY-RUN par défaut si ``RETENTION_GC_DRY_RUN`` : logge ce qui
           serait purgé sans supprimer, pour valider sur données réelles
           avant d'armer la suppression.
        """
        if self.chroma is None:
            return
        try:
            import os
            from rune.config import RETENTION_TTL_DAYS
            cutoff = time.time() - RETENTION_TTL_DAYS * 86400.0
            got = self.chroma.get(
                where={"type": "exchange"}, include=["metadatas"]
            )
            ids = got.get("ids", []) or []
            metas = got.get("metadatas", []) or []
            stale = []
            for _i, _m in zip(ids, metas):
                _m = _m or {}
                # Priorité : dernier accès > création > ts legacy > 0.
                ref = _m.get(
                    "last_access_ts",
                    _m.get("created_ts", _m.get("ts", 0)),
                )
                # `ref and` : un timestamp 0/absent est faux → épargné
                # (mode 2). On ne purge que ce qui a une date FIABLE et
                # antérieure au cutoff.
                if ref and ref < cutoff:
                    stale.append(_i)
            if not stale:
                return
            # Mode 5 : dry-run opt-in (variable d'env), pour un premier
            # passage d'observation sans rien détruire.
            dry_run = os.getenv("RETENTION_GC_DRY_RUN", "").lower() in (
                "1", "true", "yes",
            )
            if dry_run:
                log.info(
                    "Retention GC [DRY-RUN] : %d chunks 'exchange' seraient "
                    "purgés (aucune suppression). ids=%s",
                    len(stale), stale[:20],
                )
                return
            self.chroma.delete(ids=stale)
            log.info(
                "Retention GC: purged %d stale 'exchange' chunks", len(stale)
            )
        except Exception as exc:
            log.warning("Retention GC failed: %s", exc)

    # ── Memory status ──────────────────────────────────────────────────

    def memory_status(self) -> dict:
        """Return current memory state for the UI."""
        sdm_active = (self.sdm.contents.norm(dim=1) > 0).sum().item()
        return {
            "sdm": {
                "active_rows": int(sdm_active),
                "total_rows": self.sdm.rows,
                "dim": self.sdm.dim,
            },
            "mhn": {
                "stored": self.mhn.n_stored,
                "max": self.mhn.max_patterns,
                "dim": self.mhn.dim,
            },
            "kg": {
                "entities": len(self.kg.entities),
                "relations": len(self.kg.relations),
                "pending": len(self.kg.pending),
            },
            "chroma": {
                "count": self.chroma.count() if self.chroma else 0,
            },
            "exchange_count": self.exchange_count,
            "last_microsleep": self.last_microsleep,
        }

    # ── Internal helpers (extracted from process_message) ──────────────

    @staticmethod
    def _should_archive(
        query: str, response: str, learn_result: dict,
    ) -> bool:
        """True iff the exchange is non-question, non-empty, salient.

        Pure questions are skipped because they don't add user-side
        information to long-term memory — only the answer would be
        new, and it lives in the model's weights.
        """
        if not response or not learn_result.get("salient", False):
            return False
        q_stripped = query.strip()
        if not q_stripped:
            return False
        if q_stripped.endswith("?"):
            return False
        first_word = q_stripped.split()[0].lower()
        return first_word not in QUESTION_STARTS

    def _inject_web_context(self, web_context: str, rag_context: str) -> str:
        """Prepend the web-search block to the RAG context.

        The block is intentionally descriptive (not injunctive) — early
        versions used "⚠️ INSTRUCTION : Tu DOIS baser ta réponse..." but
        that pattern triggers verbatim recitation on instruction-tuned
        models (same anti-pattern as the pre-fix-#10 identity block).
        Rule 6 of SYSTEM_PROMPT covers the usage guidance once.

        V4.0.4: ``web_context`` is now numbered ``[1]``, ``[2]`` … by
        ``iterative_search``, so we add a short citation directive
        telling the LLM how to reference results. Same convention as
        warnings_v4 (directive after data, ``→`` prefix).
        """
        web_block = (
            "[Recherche web — résultats récents]\n"
            f"{web_context}\n"
            "\n"
            "Ces résultats viennent du web et sont plus récents que la "
            "mémoire long-terme.\n"
            "   → Quand tu utilises une information ci-dessus, ajoute la "
            "référence [N] correspondante dans ta réponse "
            "(ex: « selon [2] »). N'invente pas de références non listées."
        )
        if rag_context:
            return f"{web_block}\n═══\n{rag_context}"
        return web_block

    def _maybe_write_output_token(
        self, latent: Any, entropy: float,
    ) -> None:
        """Asymmetric SDM write of a confident output token.

        Confidence threshold is the entropy threshold (same one the
        doubt index is normalised against). Strength is reduced
        relative to input writes — output is corroboration, not
        new evidence.
        """
        if (
            latent is None
            or entropy >= self.entropy_threshold
            or not self.model.model_id
        ):
            return
        try:
            strength = (1.0 / (1.0 + entropy)) * _OUTPUT_SDM_STRENGTH_FACTOR
            vec = self.sdm.project(
                latent,
                model_id=self.model.model_id,
                hidden_dim=self.model.hidden_dim,
            )
            self.sdm.write(vec, vec, strength)
        except Exception:  # pragma: no cover — defensive
            pass

    # ── Image captioning (extracted from process_message) ──────────────

    def _handle_image_captions(
        self, pil_images: list[Any], message: str,
    ) -> Generator[dict, None, None]:
        """Yield cognitive events for image captions and the final caption text.

        Returns events of two kinds:
          * ``{"type": "cognitive", "data": {...}}`` — same as elsewhere
          * ``{"type": "_caption_text", "data": <str>}`` — internal,
            consumed by ``process_message``.

        V5.6.15 — Si le modèle est nativement multimodal (Gemma 3/4),
        on bypass complètement le captionneur : les images sont passées
        directement au modèle dans la phase de génération via le
        processor. Le ``_caption_text`` retourné est vide (le contenu
        sera injecté dans le pipeline multimodal natif, pas comme
        suffixe textuel du message utilisateur).
        """
        if self.debug_mode:
            log.info("Images received: %d", len(pil_images))
        if not pil_images or not self.model.is_loaded:
            yield {"type": "_caption_text", "data": ""}
            return

        # V5.6.15 — Bypass : modèle multimodal natif.
        # Pas de captionneur, les images partent directement au modèle.
        if getattr(self.model, "is_natively_multimodal", False):
            cognitive_cap = [
                f"🎨 *{len(pil_images)} image(s) reçue(s) — traitement multimodal natif*"
            ]
            yield {"type": "cognitive", "data": {"items": cognitive_cap}}
            # Pas de caption_text à injecter dans le message ; les
            # images seront passées au processor en phase de génération.
            yield {"type": "_caption_text", "data": ""}
            return

        cognitive_cap: list[str] = []
        if not self.captioner.ensure_loaded():
            cognitive_cap.append("⚠️ *Aucun captionneur d'images sélectionné*")
            caption_text = "\n".join(
                f"[Image {i+1} : image reçue mais aucun captionneur n'est configuré. "
                f"Tu ne peux PAS décrire cette image. Dis à l'utilisateur "
                f"qu'il doit activer un captionneur dans les paramètres ⚙️ → Vision.]"
                for i in range(len(pil_images))
            )
            yield {"type": "cognitive", "data": {"items": cognitive_cap}}
            yield {"type": "_caption_text", "data": caption_text}
            return

        backend = self.captioner.backend
        cognitive_cap.append(
            f"📷 *Description de {len(pil_images)} image(s) via {backend.upper()}…*"
        )
        try:
            captions = self.captioner.caption_multiple(pil_images)
            # V5.7.0 — Stockage dans la mémoire visuelle de travail.
            # Permet de retrouver l'image lors d'un zoom au tour suivant.
            for img, cap in zip(pil_images, captions):
                try:
                    self.visual_memory.store(img, caption_initial=cap or "")
                except Exception as exc:
                    log.warning("VWM store failed: %s", exc)
            for i, cap in enumerate(captions):
                if cap:
                    cognitive_cap.append(f"🖼️ *Image {i+1} : {cap}*")
                else:
                    cognitive_cap.append(f"🖼️ *Image {i+1} : (description indisponible)*")
            caption_text = "\n".join(
                f"[Image {i+1} : {cap or 'image reçue mais non décrite'}]"
                for i, cap in enumerate(captions)
            )
        except Exception as exc:
            log.warning("Image captioning failed: %s", exc)
            cognitive_cap.append(f"⚠️ *Captionneur en erreur : {exc}*")
            caption_text = "\n".join(
                f"[Image {i+1} : image reçue mais le captionneur a échoué]"
                for i in range(len(pil_images))
            )
        yield {"type": "cognitive", "data": {"items": cognitive_cap}}
        yield {"type": "_caption_text", "data": caption_text}

    # ── Debug telemetry ────────────────────────────────────────────────

    def _yield_debug_phase_a_b(
        self,
        learn_result: dict,
        rag_context: str,
        web_context: str,
        web_reason: str,
        should_web: bool,
        images: list[Any] | None,
    ) -> Generator[dict, None, None]:
        """Single-batch debug telemetry for phases A and B."""
        items: list[str] = []
        s = learn_result.get("surprise", {})
        items.append("🔬 **Phase A — Apprentissage actif**")
        items.append(f"  Saillant: {learn_result.get('salient', False)}")
        items.append(
            f"  Surprise: struct={s.get('structural', 0):.3f} "
            f"épisod={s.get('episodic', 0):.3f} "
            f"prédict={s.get('predictive', 0):.3f} "
            f"chroma_disc={s.get('chroma_discount', 0):.3f} "
            f"→ global={s.get('global', 0):.3f}"
        )
        ents = [e["text"] for e in learn_result.get("entities", [])]
        items.append(f"  Entités KG: {ents if ents else 'aucune'}")
        sdm_active = int((self.sdm.contents.norm(dim=1) > 0).sum().item())
        items.append(f"  SDM: {sdm_active} lignes actives")
        items.append(f"  MHN: {self.mhn.n_stored} patterns")
        items.append(f"  KG: {len(self.kg.entities)} entités")
        chroma_count = self.chroma.count() if self.chroma else 0
        items.append(f"  Chroma: {chroma_count} documents")
        items.append("🔬 **Phase B — RAG**")
        items.append(f"  Contexte injecté: {len(rag_context)} chars")
        if should_web:
            if web_context:
                items.append(f"  Web: {len(web_context)} chars ({web_reason})")
            else:
                items.append(f"  Web: aucun résultat ({web_reason})")
        items.append(f"  Images reçues: {len(images or [])}")
        items.append(
            f"  Modèle: {self.model.model_id} (thinking={self.model.is_thinking})"
        )
        # V3.9.4: when cascade is enabled, indicate it explicitly so
        # the user understands the response will go through Gemini
        # before reaching the local model. Previously the debug block
        # showed only the local model name even when cascade was
        # active, which led to confusion during V3.9 prod testing.
        if self.cascade_enabled:
            g = self._cascade._gemini  # type: ignore
            items.append(
                f"  🌀 Cascade: {g.model} → synth locale "
                f"(quota {g.quota_used}/{g.quota_used + g.quota_remaining}, "
                f"{g.per_minute_used}/min)"
            )
        # Reflection label disambiguation:
        #   - Thinking model → its own <think> tags fire regardless of the
        #     UI toggle. Show that explicitly so users don't think the
        #     toggle is broken when they see a reasoning panel anyway.
        #   - Non-thinking + toggle on → two-pass prompt is used.
        #   - Non-thinking + toggle off → straight generation.
        items.append(f"  Réflexion: {self._reasoning_label()}")
        yield {"type": "debug", "data": {"items": items}}

    def _reasoning_label(self) -> str:
        """Human-readable description of the current reasoning configuration.

        Used by the debug telemetry. Three mutually-exclusive cases:
        thinking-native model, two-pass enabled, or disabled.
        """
        if self.model.is_thinking:
            return "thinking natif (toggle UI ignoré)"
        if self.reasoning_enabled:
            return "two-pass activé"
        return "désactivée"

    def _build_debug_post_generation(
        self, doubt_index: float, epistemic: str, entropies: list[float],
    ) -> dict:
        """Build the post-generation debug SSE event."""
        items: list[str] = []
        items.append("🔬 **Post-génération**")
        items.append(f"  Doute: {doubt_index:.3f} → {epistemic}")
        mean_ent = sum(entropies) / max(len(entropies), 1)
        items.append(
            f"  Entropies: {len(entropies)} tokens, moy={mean_ent:.3f}"
        )
        items.append(f"  Échange #{self.exchange_count}")
        if self.exchange_count % MICROSLEEP_INTERVAL == 0:
            items.append("  🛏️ Microsleep déclenché")
        sdm_active = int((self.sdm.contents.norm(dim=1) > 0).sum().item())
        chroma_count = self.chroma.count() if self.chroma else 0
        items.append(
            f"  SDM: {sdm_active} lignes | "
            f"MHN: {self.mhn.n_stored} patterns | "
            f"Chroma: {chroma_count} docs"
        )
        return {"type": "debug", "data": {"items": items}}
