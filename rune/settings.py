"""Pydantic Settings — centralised, environment-overridable configuration.

All numeric/behavioural hyperparameters live here. Static structures
(model catalogue, prompts, label lists) stay in :mod:`rune.config`.

Override any setting via environment variable using the ``LYTHEA_`` prefix:

    LYTHEA_SDM_DIM=2048 LYTHEA_MHN_BETA=12.0 python run.py

Or via a ``.env`` file at the repository root (see ``.env.example``).

Design notes
------------
- We use Pydantic Settings v2 (``BaseSettings`` from ``pydantic_settings``).
- Field validation is conservative: bounds reflect what we have actually
  tested in production, not theoretical limits. Tighten if you know better.
- The ``case_sensitive=False`` config makes both ``LYTHEA_SDM_DIM`` and
  ``lythea_sdm_dim`` work in env files.
- ``extra="ignore"`` so unrelated env vars (HF_HOME, PATH, etc.) don't
  trigger validation errors.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LytheaSettings(BaseSettings):
    """Runtime-tunable Lythéa parameters."""

    model_config = SettingsConfigDict(
        env_prefix="LYTHEA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── SDM (Sparse Distributed Memory) ────────────────────────────────
    sdm_dim: int = Field(default=1024, ge=64, le=8192)
    sdm_rows: int = Field(default=4096, ge=64, le=65536)
    sdm_k: int = Field(default=8, ge=1, le=128)
    sdm_decay: float = Field(default=0.95, gt=0.0, le=1.0)
    sdm_prune_threshold: float = Field(default=1.0, ge=0.0)

    # ── MHN (Modern Hopfield Network) ──────────────────────────────────
    mhn_max_patterns: int = Field(default=512, ge=8, le=8192)
    mhn_dim: int = Field(default=768, ge=64, le=4096)
    mhn_beta: float = Field(default=8.0, gt=0.0)

    # ── Salience filter ────────────────────────────────────────────────
    salience_min_length: int = Field(default=8, ge=1)
    salience_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    salience_redundancy_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    # ── Entropy / doubt detection ──────────────────────────────────────
    # Entropy gate for the doubt index. Output mean-entropy is divided
    # by ``max(threshold, 0.1)`` to produce the normalised doubt; values
    # ≥ 0.3 → "intuition" label, ≥ 0.8 → "hypothese", else "fait".
    #
    # Calibration history:
    #   0.6 — original default. Empirically too strict for any modern
    #         instruction-tuned model. Qwen2.5-3B-Instruct produces
    #         mean entropies around 0.05–0.10 even on speculative
    #         questions, which under threshold 0.6 always rounds to
    #         doubt < 0.3 → label always "fait" (every answer
    #         presented as confirmed fact, regardless of topic).
    #   0.2 — current default. Validated empirically: same mean entropy
    #         0.094 yields doubt 0.47 → "intuition" — the gradient is
    #         finally usable. Thinking models (Qwen3-Thinking, R1, etc.)
    #         have entropies so low (~0.02–0.04) that they sit at
    #         "fait" even with this threshold; that is intrinsic to
    #         their architecture, no global threshold can fix it.
    entropy_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    entropy_doubt_window: int = Field(default=8, ge=1, le=128)

    # ── Composite surprise weights ─────────────────────────────────────
    # These should sum to ~1.0 but we don't enforce — user might want
    # to over-weight one signal deliberately.
    surprise_w_structural: float = Field(default=0.4, ge=0.0, le=1.0)
    surprise_w_episodic: float = Field(default=0.35, ge=0.0, le=1.0)
    surprise_w_predictive: float = Field(default=0.25, ge=0.0, le=1.0)

    # ── Generation ─────────────────────────────────────────────────────
    max_new_tokens: int = Field(default=1024, ge=16, le=8192)
    # V4.4 — Thinking models (Qwen3, QwQ, R1, etc.) burn a sizeable
    # portion of their token budget on the <think> block BEFORE
    # generating the actual response. With the standard 1024 ceiling,
    # the answer is routinely truncated mid-sentence — observed with
    # Qwen3-4B on "best NER models in French" where the reasoning ate
    # ~700 tokens and the response was cut at "{'entity': 'LOC',
    # 'start': 12, 'end':". This setting raises the ceiling for these
    # models specifically. Override per-model via SamplingProfile.
    # max_new_tokens if a particular thinking model needs more or less.
    thinking_max_new_tokens: int = Field(default=4096, ge=512, le=16384)
    max_history_turns: int = Field(default=10, ge=1, le=200)

    # ── Web classifier (LLM-based fallback) ────────────────────────────
    # V4.4 — Quand le fast-path regex de rune.web ne matche pas mais
    # que le message ressemble à une question, on demande au LLM local
    # lui-même de classifier (cf. rune.cognition.web_classifier).
    # Approche hybride inspirée d'Anthropic/OpenAI/Google qui laissent
    # le modèle décider via function calling. Évite le jeu sans fin
    # d'ajout de patterns regex pour chaque cas d'ambiguïté. Coût :
    # ~80 tokens prompt + ~10 output, ~300-500ms sur Qwen2.5-7B.
    # Cache LRU 256 entrées partagé module-level pour les questions
    # répétées. Mettre False pour debug / latence critique.
    web_classifier_enabled: bool = Field(default=True)
    web_classifier_timeout_s: float = Field(default=3.0, ge=0.5, le=10.0)

    # ── CRAG (Corrective RAG) — V5.2 ────────────────────────────────────
    # Seuils empiriques sur les scores cross-encoder (sigmoidal-ish 0-1).
    # Ajuster si tu observes trop d'AMBIGUOUS (baisse correct_threshold)
    # ou trop d'INCORRECT (baisse ambiguous_threshold). Garder un écart
    # d'au moins 0.2 entre les deux pour avoir une vraie zone ambiguë.
    crag_correct_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    crag_ambiguous_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # Active le rewrite LLM sur retrieval AMBIGUOUS. Désactiver si tu
    # veux économiser ~300ms quand la mémoire long-terme est faiblement
    # alimentée (rewrite ne sauve pas grand-chose si le corpus est vide).
    crag_enable_rewrite: bool = Field(default=True)

    # ── Reflection loop (V5.5) ──────────────────────────────────────────
    # Self-critique sélective sur cas à risque (tech_reco, complexité,
    # CRAG incorrect, doute accumulé). Désactivable pour économiser
    # ~1-2s par tour sur les cas concernés. Le pattern DeepMind 2023
    # alerte sur la dégradation de performance si appliqué
    # systématiquement — d'où l'activation sélective via
    # rune.cognition.reflection.should_reflect().
    reflection_enabled: bool = Field(default=True)
    reflection_timeout_s: float = Field(default=6.0, ge=1.0, le=15.0)

    # ── Reasoning budgets ──────────────────────────────────────────────
    # Token budgets for the reasoning passes. Two regimes:
    #
    #   • Simple two-pass (ReasoningGenerator) — one shot. The previous
    #     hardcoded 800 truncated technical analyses mid-sentence
    #     ("...ce qui est particul"). 1024 gives a real chain of thought
    #     room while staying bounded so the main pass keeps context.
    #
    #   • Deep chain (DeepReasoningChain) — per-step budget, picked by
    #     the complexity router. A "medium" question gets the medium
    #     budget, a "complex" one the high budget. The previous
    #     hardcoded 320 cut the exploration step at "Vitesse et E".
    #
    # The prompt-side length hint (e.g. "~400 mots") is derived from
    # these and deliberately set BELOW the hard ceiling so the model
    # has slack to finish its sentence before hitting the wall.
    reasoning_simple_max_tokens: int = Field(default=1024, ge=128, le=4096)
    reasoning_deep_step_medium_tokens: int = Field(default=640, ge=128, le=4096)
    reasoning_deep_step_high_tokens: int = Field(default=960, ge=128, le=4096)
    # Char-side clip on the final reasoning text injected into the main
    # pass — a safety net if the model ignores max_new_tokens entirely.
    # 6000 chars ≈ ~1100 tokens of French: enough that a well-formed
    # consolidated reasoning is never truncated at display time. The
    # real defence against runaway length is the per-section counting
    # limits in the explore prompt — this clip only catches outliers.
    reasoning_text_max_chars: int = Field(default=6000, ge=500, le=20000)

    # ── Knowledge Graph ────────────────────────────────────────────────
    kg_active_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    kg_capture_sensitive: bool = Field(default=True)
    # Profil de labels GLiNER pour l'extraction d'entités du KG :
    #   "chat"  → labels conversationnels (identité, préférences…) [défaut]
    #   "agent" → labels techniques (fichier, fonction, erreur…)
    #   "both"  → les deux
    # L'agent bascule automatiquement sur "agent" (ou "both") pendant une
    # mission via ``agent_kg_label_profile`` ci-dessous.
    kg_label_profile: str = Field(default="chat")
    agent_bestofn: int = Field(default=0, ge=0, le=5)  # 0 = auto (selon le matériel)
    agent_web_on_block: bool = Field(default=True)
    agent_snippets_enabled: bool = Field(default=True)
    agent_skill_writing: bool = Field(default=True)
    # ── Mémoire agent v2 (épisodique + KG) — TOUT défaut OFF (opt-in) ──────
    # Le tar n'active rien : comportement identique tant que ces flags sont
    # False. Active-les un par un pour tester ; remets à False si ça déraille.
    agent_memory_v2_enabled: bool = Field(default=False)        # interrupteur maître
    agent_memory_tools_enabled: bool = Field(default=False)     # outils PULL recall/query_kg
    agent_episodic_recall_enabled: bool = Field(default=False)  # PUSH épisodique au lancement
    agent_kg_recall_enabled: bool = Field(default=False)        # PUSH KG au lancement
    agent_episodic_write_enabled: bool = Field(default=False)   # consolidation épisodique
    agent_kg_write_enabled: bool = Field(default=False)         # consolidation KG
    # Profil de labels GLiNER appliqué PENDANT une mission d'agent. Permet
    # d'extraire des entités techniques (fichier, fonction, erreur…) plutôt
    # que conversationnelles quand l'agent travaille. "agent" ou "both"
    # recommandés ; "chat" garde le profil conversationnel. Ne s'applique
    # que si agent_kg_recall_enabled ou agent_kg_write_enabled est actif.
    agent_kg_label_profile: str = Field(default="agent")
    agent_memory_recall_budget_chars: int = Field(default=1200, ge=200, le=8000)
    # Verrou de génération partagé chat/agent (permet de chatter pendant une
    # mission sans collision). ON par défaut : le chat prend le même verrou que
    # l'agent autour de sa génération → plus de collision GPU ni de hooks
    # entrelacés. Mettre à False pour retrouver l'ancien comportement (chat
    # qui répond sans attendre l'agent, au risque d'une collision).
    agent_chat_shared_lock_enabled: bool = Field(default=True)
    agent_initial_plan: bool = Field(default=True)   # plan initial adaptatif (L1/L2)
    agent_generalist_verify: bool = Field(default=True)  # vérif non-code (heuristique + auto-critique)
    # Profil thinking : "auto" applique l'allègement de l'échafaudage dès qu'un
    # modèle à raisonnement interne est chargé (plan initial off, best-of-N
    # réduit, micro-checks off — le modèle les fait nativement) ; "on"/"off"
    # forcent. Le blackboard, le routeur, les gardes de sécurité, le batching
    # restent actifs dans tous les cas.
    agent_thinking_profile: str = Field(default="auto")  # auto | on | off
    agent_thinking_max_tokens: int = Field(default=2048, ge=512, le=8192)  # budget/étape en mode thinking (le raisonnement est long)
    agent_author_skills: bool = Field(default=True)
    telegram_enabled: bool = Field(default=True)  # coupe la passerelle même si un token est configuré
    telegram_bot_token: str = Field(default="")
    telegram_allowed_chat_ids: list[int] = Field(default_factory=list)
    kg_pending_ttl_hours: int = Field(default=24, ge=1, le=8760)
    kg_fuzzy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # ── Retrieval ──────────────────────────────────────────────────────
    retrieval_top_n: int = Field(default=20, ge=1, le=200)
    retrieval_rerank_top: int = Field(default=5, ge=1, le=50)
    retrieval_rrf_k: int = Field(default=60, ge=1)

    # ── Web search ─────────────────────────────────────────────────────
    web_max_rounds: int = Field(default=3, ge=1, le=10)
    web_stability_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # ── Microsleep (consolidation) ─────────────────────────────────────
    microsleep_interval: int = Field(default=5, ge=1, le=1000)

    # ── Agent V6 (réglable via .env, préfixe LYTHEA_) ──────────────────
    agent_execution: bool = Field(default=True)        # exécution sandboxée
    agent_react: bool = Field(default=True)            # boucle tool-calling
    agent_pip_install: bool = Field(default=True)      # auto-install des deps
    agent_exec_timeout_s: int = Field(default=45, ge=5, le=600)
    agent_install_timeout_s: int = Field(default=240, ge=30, le=1800)
    agent_max_installs: int = Field(default=5, ge=0, le=20)
    agent_cmd_timeout_s: int = Field(default=120, ge=5, le=1800)
    agent_serve_timeout_s: int = Field(default=12, ge=2, le=120)
    agent_react_max_iters: int = Field(default=24, ge=1, le=100)
    agent_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    agent_max_new_tokens: int = Field(default=1024, ge=128, le=8192)
    agent_synthesis_max_tokens: int = Field(default=2048, ge=512, le=8192)  # synthèse research/answer (texte long, anti-troncature)
    agent_research_report_min_chars: int = Field(default=1200, ge=0, le=100000)  # research : ≥ ce seuil → rapport .md ; sinon réponse en chat
    agent_deliberate_when_stuck: bool = Field(default=True)
    agent_critic_enabled: bool = Field(default=True)
    agent_critic_rounds: int = Field(default=1, ge=0, le=3)
    agent_learn_from_errors: bool = Field(default=True)
    agent_skills_enabled: bool = Field(default=True)
    agent_skills_dirs: list[str] = Field(default_factory=list)
    agent_ollama_workers: list[str] = Field(default_factory=list)
    agent_ollama_base_url: str = Field(default="")
    microsleep_inactivity: int = Field(default=180, ge=10, le=86400)
    # V5.9 chantier 3 — délai d'inactivité (s) avant un deep_sleep
    # automatique. 30 min par défaut. Sans ce timer, le deep_sleep (et
    # donc le GC de rétention Chroma) ne se déclenchait que manuellement.
    deep_sleep_inactivity: int = Field(default=1800, ge=60, le=604800)
    microsleep_rehearse_k: int = Field(default=16, ge=1, le=1024)
    microsleep_boost: float = Field(default=0.15, ge=0.0, le=10.0)

    # ── Server ─────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7860, ge=1, le=65535)

    # ── Authentication ─────────────────────────────────────────────────
    # Empty string = no auth required (fully open, dev mode).
    # Non-empty = bearer token required for non-loopback or Cloudflare-
    # tunneled requests. Loopback (127.0.0.1, ::1) bypasses auth unless
    # auth_strict is True. Cloudflare-tunneled requests (detected via
    # cf-connecting-ip / cf-ray headers) ALWAYS require the token.
    auth_token: str = Field(default="")
    auth_strict: bool = Field(default=False)

    # ── Rate limiting ──────────────────────────────────────────────────
    # Format: "<count>/<period>" e.g. "60/minute", "1000/hour".
    # Loopback requests bypass rate limits.
    rate_limit_chat: str = Field(default="60/minute")
    rate_limit_model_load: str = Field(default="10/minute")

    # ── Trust remote code ──────────────────────────────────────────────
    # When True, transformers will execute custom Python code shipped
    # with HuggingFace models. This is a security risk for untrusted
    # models. Default False — only enable if you understand the risk
    # and need a model that requires it (rare for the catalog defaults).
    allow_remote_code: bool = Field(default=False)

    # ── Embedding caches ───────────────────────────────────────────────
    # Cache size for EntityExtractor.encode() — ~3 KB per entry.
    # Set to 0 to disable caching entirely.
    embed_cache_size: int = Field(default=1024, ge=0, le=100_000)
    # Cache size for HFModelWrapper.analyze_input() — entries can be
    # ~100 KB each (hidden states for the input sequence). Default 64
    # is a deliberate cap so the cache stays under ~10 MB.
    analyze_cache_size: int = Field(default=64, ge=0, le=10_000)

    # ── Cross-encoder reranker ─────────────────────────────────────────
    # Model used by HybridRetriever to rerank the top-N RRF results.
    # BGE-reranker-v2-m3 is multilingual and good for French. The
    # smaller "BAAI/bge-reranker-base" (280 MB) is enough in most cases.
    cross_encoder_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    # Minimum normalised rerank score to keep a result. Below this,
    # the candidate is dropped even if it was ranked first by RRF.
    # Range 0..1 for cross-encoders that emit normalised scores;
    # raw cross-encoder scores can be higher — leave at 0 to disable.
    #
    # Calibration history:
    #   0.5 — original default. Empirically too strict on conversational
    #         queries like "Tu te souviens de X ?" (rerank tags relevant
    #         docs at <0.05 because the meta-question form differs from
    #         the stored content). Caused full rerank rejection.
    #   0.0 — too permissive, lets through unrelated stories from Chroma
    #         that pollute responses (observed during prod testing).
    #   0.2 — current default. Sweet spot validated empirically with
    #         Qwen3-4B-Thinking on a 9-doc Chroma: 9 → 1 retained at
    #         top=0.91 score (the genuinely relevant doc) on a query
    #         about Mika's employer relations. See CHANGELOG.md.
    cross_encoder_min_score: float = Field(default=0.2, ge=0.0, le=10.0)

    # ── Microsleep ripples + replay ────────────────────────────────────
    # Sharp-wave ripple — when this many high-saliency events accumulate
    # between microsleeps, the next microsleep runs in "ripple mode"
    # with intensified rehearsal.
    ripple_trigger_count: int = Field(default=5, ge=1, le=100)
    # Surprise threshold above which an event counts toward ripples.
    ripple_surprise_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Boost multiplier applied during ripple-active microsleeps.
    ripple_boost_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    # Replay sequence length — number of consecutive MHN patterns to
    # chain for forward+reverse replay.
    replay_sequence_length: int = Field(default=4, ge=2, le=20)
    # Number of independent sequences replayed per microsleep.
    replay_n_sequences: int = Field(default=3, ge=1, le=20)
    # Compression: minimum replay count before MHN→Chroma transfer.
    compression_replay_threshold: int = Field(default=3, ge=1, le=100)

    # ── Soft memory (prefix-tuning) ────────────────────────────────────
    # Opt-in long-term parametric memory. When enabled, exposes
    # /api/soft-memory/* endpoints for training and management.
    # Default OFF — see lythea/soft_memory.py docstring for rationale.
    enable_soft_memory: bool = Field(default=False)
    # Length of the learnable prefix in tokens (per layer).
    soft_memory_prefix_length: int = Field(default=32, ge=4, le=512)
    # Training learning rate.
    soft_memory_learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    # Epochs per training call.
    soft_memory_epochs: int = Field(default=1, ge=1, le=20)

    # ── Cascade generation (V3.9) ──────────────────────────────────────
    # Draft-then-refine: Gemini generates a rich draft, the local model
    # synthesises it to Lythéa's concise voice and provides the latents
    # that feed SDM/MHN/KG. When disabled (default), generation is
    # local-only as in V3.
    #
    # The Google API key is read from the GOOGLE_API_KEY env var (or
    # .env file); it is never logged in clear and is masked to the
    # last 4 characters in any debug surface.
    enable_cascade: bool = Field(default=False)
    # Google AI Studio API key (https://aistudio.google.com/app/apikey).
    # Validated for format at boot when enable_cascade=True. Read via
    # the standard pydantic-settings env loader — keep it in .env, not
    # in the code.
    google_api_key: str | None = Field(default=None)
    # Gemini model id used for the rich draft. As of 2026-06 the free tier
    # (AI Studio, 1500 req/day, no card) covers gemini-3.5-flash (best, GA),
    # gemini-3-flash, gemini-3.1-flash-lite and the 2.5 Flash family. Pro
    # models are paid-only.
    cascade_gemini_model: str = Field(default="gemini-3.5-flash")
    # Gemini's draft is shipped as-is when shorter than this threshold
    # (in approximate tokens). Above it, the local model synthesises.
    # 50 ≈ "anything longer than two short sentences gets synthesised".
    cascade_synthesis_threshold_tokens: int = Field(default=50, ge=0, le=500)
    # Maximum length of the synthesised output (local model). Acts as a
    # soft ceiling on Lythéa's voice — beyond this we accept truncation.
    cascade_synthesis_max_tokens: int = Field(default=120, ge=20, le=500)
    # Hard ceiling on the Gemini draft. 2048 tokens covers Gemini 2.5
    # Flash's internal "thinking" budget (~500 thought tokens observed
    # in prod) plus a substantial answer. The previous 800 default
    # caused mid-sentence truncations on technical questions.
    cascade_gemini_max_tokens: int = Field(default=2048, ge=50, le=8192)
    # Sampling temperature for the Gemini draft. Lower = more
    # deterministic; 0.7 is Google's recommended default.
    cascade_gemini_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Daily quota for the local soft-counter (free tier hint). Google
    # enforces the real limit server-side; this is just for UI warnings.
    cascade_daily_quota_hint: int = Field(default=1500, ge=1, le=100000)

    # ── Image captioner ────────────────────────────────────────────────
    # Maximum new tokens generated by the Qwen2VL captioner per image.
    # 150 (the previous hardcoded value) truncated rich photo
    # descriptions mid-sentence — observed during prod testing on a
    # detailed strawberry photo where the caption ended on
    # "with the strawberry as the". 256 gives the captioner enough
    # room while keeping the prompt budget bounded.
    # BLIP captioner is unaffected (it is laconic by design and uses
    # its own internal limit).
    caption_max_tokens: int = Field(default=256, ge=64, le=1024)

    # ── Coreference inference ──────────────────────────────────────────
    # When the user writes a "Je..." sentence (e.g. "Je travaille chez
    # Anthropic") without naming themselves, GLiNER does not extract a
    # person entity, so :func:`StoragePhase._link_co_occurrences` cannot
    # build a relation. As a fallback, the storage phase looks up the
    # most-recently-mentioned person in the KG and uses them as the
    # implicit anchor — but only if they were mentioned within this
    # window. Beyond this window we assume the user has moved on and
    # we don't infer the relation.
    coreference_window_sec: float = Field(default=1800.0, gt=0.0, le=86400.0)
    # Confidence assigned to relations created via the coreference
    # fallback. Lower than the direct-extraction default (0.9) so you
    # can later filter or purge inferred relations if needed.
    coreference_inferred_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    # ════════════════════════════════════════════════════════════════════
    # V4 cognitive modules — strictly opt-in, all OFF by default.
    # When every flag below is False, runtime is byte-identical to V3.9.4.
    # See CHANGELOG entry "V4.0 — Modules cognitifs supérieurs" and the
    # design contracts in lythea/memory/cognitive_state.py,
    # lythea/cognition/{inhibition,planning,predictive_coding,timeline}.py.
    # ════════════════════════════════════════════════════════════════════

    # ── V4.0.a: Cognitive state (theory of mind + affect) ──────────────
    # Master flag. When False, no cognitive_state attribute is created
    # and the Phase A.1 / Phase C hooks are no-ops.
    enable_cognitive_state: bool = Field(default=False)
    # Exponential decay half-life for Lythéa's own affect vector. After
    # this many seconds with no external signal, valence/arousal are
    # halved. 5 min is a plausible cortical timescale.
    affect_decay_half_life_sec: float = Field(default=300.0, gt=0.0, le=86400.0)
    # ANTI-SYCOPHANT cap on emotional contagion. Even if the user is
    # maximally enthusiastic (v=1, a=1, conf=1), Lythéa's own valence
    # picks up at most this fraction. MUST stay strictly < 1.0 — values
    # near 1 would make Lythéa a mirror; values near 0 make her cold.
    affect_contagion_max: float = Field(default=0.4, ge=0.0, le=1.0)
    # Smoothing factor for affect updates: new = (1-α)·target + α·current.
    # 0 = no inertia (snap), 1 = frozen.
    affect_inertia: float = Field(default=0.3, ge=0.0, le=1.0)
    # After this many turns with no externally-signalled affect (low
    # confidence on user side AND no intrinsic signal), reset Lythéa's
    # affect to neutral. Prevents stale moods from haunting sessions.
    affect_reset_latch_turns: int = Field(default=8, ge=1, le=100)
    # Detector backend. Only "lexical" is implemented in V4.0;
    # "classifier" and "llm" are forward hooks. Unknown values fall
    # back to lexical at runtime.
    affect_detector: str = Field(default="lexical")
    # EMA threshold above which a user concept is considered known
    # (suppressed from over-explanation in user_state block).
    user_model_known_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # ── V4.0.b: Inhibition (output filter cascade) ─────────────────────
    enable_inhibition: bool = Field(default=False)
    # N1 hard rules: when True, any match BLOCKS output (passed=False).
    # When False, downgrades to "annotate" (operator visibility, no
    # blocking). Strict mode is the safe default for prod.
    inhibition_n1_strict: bool = Field(default=True)
    # N2 = ML toxicity classifier. Placeholder in V4.0 (no model
    # bundled). Keep False until a classifier is wired in.
    inhibition_n2_enabled: bool = Field(default=False)
    # N3 = KG-coherence check on categorical predicates. Cheap, no
    # external deps; safe to leave on.
    inhibition_n3_enabled: bool = Field(default=True)
    # Default action for soft hits (N2/N3) when triggered.
    # ∈ {"pass", "annotate", "rewrite", "block"}.
    inhibition_default_action: str = Field(default="annotate")
    # Comma-separated whitelist. Substrings here suppress N2 false
    # positives on industrial vocabulary. Default seeded with the FR
    # technical lexicon (chemometrics, metrology, NDT) to avoid
    # flagging "défaut", "fissure", "anomalie" as toxic content.
    inhibition_domain_whitelist: str = Field(
        default="chimiométrie,spectroscopie,radioprotection,défaut,"
                "anomalie,fissure,corrosion,maintenance,industriel"
    )

    # ── V4.0.c: Planning (executive control) ───────────────────────────
    enable_planning: bool = Field(default=False)
    # Hard cap on plan length. Long plans are usually a sign the LLM
    # over-decomposed; 7 ± 2 is the working-memory sweet spot.
    planning_max_steps: int = Field(default=7, ge=1, le=30)
    # Goals untouched for this many days are auto-archived (status
    # → "abandoned"). Prevents zombie buts cluttering get_active().
    planning_goal_stale_days: int = Field(default=14, ge=1, le=365)
    # When True, multi-step plans are generated by an LLM call (with
    # template fallback on failure). When False, only the regex
    # template is used (faster, lower quality).
    planning_use_llm: bool = Field(default=True)
    # Truncation cap on the [Plan en cours] block injected into the
    # prompt. Long plans get '…' suffix beyond this.
    planning_prompt_block_max_chars: int = Field(default=400, ge=50, le=4000)

    # ── V4.1: Affect-modulated consolidation (amygdala-like) ───────────
    # When True, high-arousal user messages flag the most recent MHN
    # pattern for boosted replay. Requires enable_cognitive_state=True
    # to receive arousal signals; otherwise the flag is a no-op.
    affect_modulates_consolidation: bool = Field(default=False)
    # Arousal threshold for triggering ripple-like consolidation.
    affect_ripple_arousal_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # Multiplicative boost applied to attention during replay for
    # affect-flagged patterns. ≥1.0 (1.0 = no boost).
    affect_consolidation_boost_factor: float = Field(default=1.5, ge=1.0, le=10.0)

    # ── V4.2: Predictive coding (Friston-style cortical prediction) ────
    enable_predictive_coding: bool = Field(default=False)
    # Number of past embeddings retained for EMA prediction.
    pc_history_size: int = Field(default=8, ge=1, le=64)
    # Below this many observations the predictor stays in cold-start
    # mode (always returns "full" mode regardless of error).
    pc_cold_start_min: int = Field(default=3, ge=1, le=32)
    # EMA decay for the predicted next embedding. Higher = more weight
    # on the most recent observation.
    pc_ema_decay: float = Field(default=0.6, ge=0.0, le=1.0)
    # Cosine-distance threshold below which compute drops to low_power.
    pc_low_threshold: float = Field(default=0.15, ge=0.0, le=2.0)
    # Cosine-distance threshold above which compute escalates to high.
    # Must be > pc_low_threshold (validated downstream).
    pc_high_threshold: float = Field(default=0.65, ge=0.0, le=2.0)
    # Confidence cap on prediction-error gating decisions.
    pc_confidence_cap: float = Field(default=0.85, ge=0.0, le=1.0)
    # When True, the gating decision actually influences runtime
    # behaviour (e.g. suppresses non-essential web search in low_power
    # mode). When False, decisions are computed for telemetry but
    # ignored — safer default while validating the predictor.
    pc_apply_gating: bool = Field(default=False)
    # V5.9 — poids du signal SDM (surprise prédictive) dans l'erreur de
    # gating du predictive coding. 0 = EMA seule (off). > 0 mélange la
    # lecture SDM ; sert de juge d'ablation pour mesurer si la SDM
    # (actuellement inerte) apporte de la variance discriminante.
    pc_gating_w_sdm: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── V4.3: Timeline (narrative chronology extraction) ───────────────
    enable_timeline: bool = Field(default=False)
    # Max number of TimelineEvent objects retained per message. Beyond
    # this, the longest chronologies get truncated post-dedup.
    timeline_max_events: int = Field(default=8, ge=1, le=64)
    # Truncation cap on the [Chronologie] prompt block.
    timeline_block_max_chars: int = Field(default=600, ge=50, le=4000)
    # Events with confidence below this threshold are dropped before
    # rendering (kept in the events list for debug).
    timeline_render_min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    # When False, "vague" events ("récemment", "soon") are extracted
    # but not rendered in the prompt block.
    timeline_render_vague: bool = Field(default=False)

    # ── V4.4: Metacognition (self-monitoring + calibration) ───────────
    enable_metacognition: bool = Field(default=False)
    # Doubt thresholds for the 4-band certainty classifier.
    # Order constraint enforced downstream: low < high < very_high.
    metacog_low_doubt: float = Field(default=0.15, ge=0.0, le=1.0)
    metacog_high_doubt: float = Field(default=0.35, ge=0.0, le=1.0)
    metacog_very_high_doubt: float = Field(default=0.55, ge=0.0, le=1.0)
    # Epistemic level above which the certainty band is shifted one
    # step toward "certaine" (model felt confident at the reasoning step).
    metacog_epistemic_boost_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
    )
    # When True, the hedge prefix ("Je ne suis pas totalement sûre…")
    # is actually prepended to the final text. When False, the
    # decision is computed for telemetry but the text is unchanged.
    # OFF by default — opt-in to avoid surprising the user.
    metacog_apply_hedge: bool = Field(default=False)
    # Calibration history window (per session, persisted across).
    metacog_calibration_window: int = Field(default=100, ge=1, le=10000)

    # ── V6.0.0 — MCP (Model Context Protocol) ────────────────────────
    # Lythéa speaks JSON-RPC over stdio to MCP servers (subprocess).
    # See lythea/mcp/ for the implementation.

    mcp_enabled: bool = Field(
        default=True,
        description="Master switch for the whole MCP stack. False disables "
                    "all servers and the routing in one shot.",
    )

    mcp_sandbox_dir: str = Field(
        default="",
        description="Absolute path to the workspace sandbox. If empty, "
                    "defaults to ~/.lythea/sandbox/ at runtime. Lythéa "
                    "can read/write inside this dir via the filesystem "
                    "MCP. User uploads/downloads land here too.",
    )

    # Per-server flags. Default-on for V6.0.0 (3 servers).
    mcp_filesystem_enabled: bool = Field(default=True)
    mcp_github_enabled: bool = Field(default=True)
    mcp_youtube_enabled: bool = Field(default=True)

    # Workspace limits (server-side enforced)
    mcp_workspace_max_file_mb: int = Field(
        default=20, ge=1, le=500,
        description="Per-file upload size limit, in MB.",
    )
    mcp_workspace_max_total_mb: int = Field(
        default=200, ge=10, le=10000,
        description="Cumulative workspace size limit, in MB. "
                    "Lythéa warns at 80% and refuses uploads at 100%.",
    )


@lru_cache(maxsize=1)
def get_settings() -> LytheaSettings:
    """Return the singleton settings instance.

    Cached so all modules see the same values without re-parsing env
    on every import. Tests can clear the cache via
    ``get_settings.cache_clear()`` if they need to inject overrides.
    """
    return LytheaSettings()


# ── Variables spécifiques Rune (préfixe RUNE_) ────────────────────────
#
# Les variables RUNE_* étaient auparavant lues via os.environ.get() dans
# boot.py, ce qui les rendait invisibles au chargement du .env (pydantic-
# settings ne charge que les vars LYTHEA_* dans LytheaSettings).
#
# Cette classe les intègre proprement dans pydantic-settings :
# - Lues depuis le .env si présentes (clé RUNE_AUTOLOAD_MODEL, etc.)
# - Exportables manuellement dans le shell (comportement inchangé)
# - Invalidables par cache_clear() dans les tests
#
# Ne pas ajouter ces champs à LytheaSettings (préfixe LYTHEA_) : ça
# casserait la compatibilité avec les configs existantes Lythea v5/v4.


class RuneSettings(BaseSettings):
    """Variables d'environnement propres à Rune (préfixe RUNE_)."""

    model_config = SettingsConfigDict(
        env_prefix="RUNE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Autoload du modèle au boot ─────────────────────────────────
    # RUNE_AUTOLOAD_MODEL=true pour charger automatiquement au boot.
    # Si false (défaut), charger via POST /api/models/load.
    autoload_model: bool = Field(default=False)

    # Model ID HuggingFace à charger si autoload_model=true.
    # Défaut vide → repli sur DEFAULT_MODEL dans config.py.
    default_model: str = Field(default="")

    # ── Auto-install Node.js pour MCP ─────────────────────────────
    # true (défaut) : tente d'installer Node via apt/brew si manquant.
    # false : skip silencieux, MCP désactivé si Node absent.
    # V0.1.1 — désactivé par défaut. L'auto-install de Node.js pendant le
    # boot (via NodeSource/apt) nécessite root + accès réseau à
    # deb.nodesource.com, souvent bloqué par le firewall d'egress des
    # pods. En cas d'échec réseau, le curl|bash pouvait bloquer le boot
    # plusieurs minutes → serveur en 503 permanent, dashboard injoignable.
    # MCP est optionnel : mieux vaut le skipper proprement que bloquer le
    # boot. Mettre à True explicitement si tu veux l'auto-install.
    auto_install_node: bool = Field(default=False)

    # ── Trinity (pool multi-modèles) ───────────────────────────────
    # Chemin vers le fichier trinity.yaml. Vide = Trinity désactivé.
    trinity_config: str = Field(default="")

    # ── Cascade externe ────────────────────────────────────────────
    # Redouble le flag LYTHEA_ENABLE_CASCADE pour les configs qui
    # utilisent le préfixe RUNE_ (rétrocompat .env.example v0.1.0).
    enable_cascade: bool = Field(default=False)
    cascade_provider: str = Field(default="gemini")
    cascade_claude_model: str = Field(default="claude-sonnet-4-20250514")

    # ── Subagent ───────────────────────────────────────────────────
    # Model ID utilisé par les sous-agents (Trinity Worker ou standalone).
    # Vide = MockBackend (tests) sauf si Trinity configuré.
    subagent_model_id: str = Field(default="")


@lru_cache(maxsize=1)
def get_rune_settings() -> RuneSettings:
    """Return the Rune-specific settings singleton.

    Cached like get_settings(). Tests can clear with
    ``get_rune_settings.cache_clear()``.
    """
    return RuneSettings()
