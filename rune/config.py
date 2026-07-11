"""Static configuration: paths, device, model catalogue, prompts.

Tunable runtime parameters (dimensions, thresholds, intervals…) live in
:mod:`rune.settings` and are environment-overridable. We re-export them
from this module for backward compatibility — existing imports like
``from rune.config import SDM_DIM`` continue to work.
"""
from __future__ import annotations

from dataclasses import dataclass

from rune.env import CACHE_ROOT, PLATFORM  # noqa: F401 — PLATFORM re-exported
from rune.settings import get_settings

_settings = get_settings()


# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = CACHE_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
KG_DIR = DATA_DIR / "kg"
CHROMA_DIR = CACHE_ROOT / "chroma"
SDM_DIR = DATA_DIR / "sdm"
MHN_DIR = DATA_DIR / "mhn"
# V5.4 — Procedural memory (skills.md pattern)
PROCEDURAL_DIR = DATA_DIR / "procedural"

for _d in (DATA_DIR, SESSIONS_DIR, KG_DIR, SDM_DIR, MHN_DIR, PROCEDURAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────
try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    DTYPE = _torch.bfloat16 if DEVICE == "cuda" else _torch.float32
except ImportError:
    DEVICE = "cpu"
    DTYPE = None  # type: ignore[assignment]


# ── Model catalogue (static, not env-overridable) ──────────────────────
@dataclass(frozen=True)
class SamplingProfile:
    """Recommended sampling parameters for a given model.

    Each model family has its own sweet spot — Qwen runs hot at 0.7,
    Mistral prefers 0.5 (less sycophancy), LFM2 likes very low
    temperatures (0.1-0.3) with min_p instead of top_p, thinking
    models benefit from 0.6.

    Sources: official model cards on HuggingFace and provider docs.

    Fields set to ``None`` are disabled (sampling step is skipped).
    For instance LFM2 uses ``min_p`` instead of ``top_p`` so the
    profile sets ``top_p=None`` and ``min_p=0.15``.
    """

    temperature: float = 0.7
    top_p: float | None = 0.9
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    # ``None`` means: fall back to the global ``max_new_tokens`` setting.
    max_new_tokens: int | None = None


# Default profile used when a model has no explicit sampling spec.
# Conservative middle-of-the-road values that work OK for most
# instruction-tuned models in the 3B-14B range.
DEFAULT_SAMPLING = SamplingProfile(
    temperature=0.7,
    top_p=0.9,
    top_k=None,
    min_p=None,
    repetition_penalty=1.0,
)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a supported model."""

    model_id: str
    label: str
    size_gb: float
    is_thinking: bool = False
    notes: str = ""
    # Load this model in 4-bit NF4 (bitsandbytes). Lets large MoE models fit a
    # 24 GB GPU; ``size_gb`` should then reflect the *4-bit* footprint so the
    # VRAM pre-check is accurate. Degrades to bf16 if bitsandbytes is missing.
    quant_4bit: bool = False
    # Optional recommended sampling parameters. When loading a model
    # in the UI, these are applied automatically as the new defaults.
    # ``None`` falls back to :data:`DEFAULT_SAMPLING`.
    sampling: SamplingProfile | None = None


CATALOG: dict[str, ModelSpec] = {
    # ════════════════════════════════════════════════════════════════════
    # Section 1 — INSTRUCT STANDARD
    # ════════════════════════════════════════════════════════════════════
    # Modèles instruction-tuned classiques (Transformer dense). Pas de
    # raisonnement explicite. Suivent les instructions et répondent
    # directement. Valeurs T/top_p/top_k issues des model cards officielles.
    "Qwen/Qwen2.5-3B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        label="Qwen2.5-3B",
        size_gb=7.0,
        notes="Recommandé Instruct — validé en prod, factuel et concis",
        sampling=SamplingProfile(
            temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
        ),
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        label="Qwen2.5-7B",
        size_gb=15.0,
        notes="Plus capable que le 3B, meilleur sur les tâches complexes",
        sampling=SamplingProfile(
            temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
        ),
    ),
    "Qwen/Qwen2.5-Coder-7B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        label="Qwen2.5-Coder-7B",
        size_gb=15.0,
        notes=("Spécialisé code. DÉCONSEILLÉ comme agent : 7B est au bord de "
               "la « falaise » de tool-calling (benchmarks BFCL/Docker) — "
               "instable (oublis d'import, fences au lieu d'outils). OK pour "
               "complétion/chat léger. Pour l'agent, préférer le Coder-14B."),
        sampling=SamplingProfile(
            temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
        ),
    ),
    "Qwen/Qwen2.5-Coder-14B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        label="Qwen2.5-Coder-14B (4-bit)",
        size_gb=10.0,            # empreinte NF4 — confortable sur 24 Go
        quant_4bit=True,
        notes=("★ RECOMMANDÉ POUR L'AGENT. Dense 14B code & tool-calling, "
               "NON-thinking (pas de souci ReAct+thinking). 88% HumanEval vs "
               "79.7% du 7B — bien moins de coquilles, tool-calling fiable "
               "(98%+ avec format <tools>). Charge en NF4 (~10 Go VRAM), "
               "laisse la place au steering + mémoire sur 24 Go."),
        sampling=SamplingProfile(
            temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
        ),
    ),
    "Qwen/Qwen3-32B": ModelSpec(
        model_id="Qwen/Qwen3-32B",
        label="Qwen3-32B (Thinking, 4-bit)",
        size_gb=18.0,            # empreinte NF4 — rentre sur 24 Go
        is_thinking=True,
        quant_4bit=True,
        notes=(
            "★ Gros dense recommandé. Qwen3 dense 32.8B / 64 couches — MÊME "
            "famille architecturale que le Coder-14B (hooks & steering "
            "compatibles sans réaudit, archi Transformer standard). HYBRIDE, "
            "classé THINKING : son <think> natif s'active même avec "
            "enable_thinking=False (le flag ne le supprime pas de façon "
            "fiable), donc on l'assume thinking — toggle Réflexion grisé, "
            "<think> natif extrait proprement plutôt que « fuite ». Charge en "
            "NF4 (~18 Go) sur 24 Go. Vraie montée en puissance vs le 14B. "
            "⚠️ Download fp16 ~64 Go disque (prévoir un Network Volume)."
        ),
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    "microsoft/Phi-4-mini-instruct": ModelSpec(
        model_id="microsoft/Phi-4-mini-instruct",
        label="Phi-4-mini (Microsoft)",
        size_gb=8.0,
        notes=(
            "Reasoning dense, multilingue 20+ langues dont FR, contexte 128K. "
            "À valider en prod — premier test architectural Phi sur Lythéa."
        ),
        # Microsoft recommande T=0.6 pour Phi-4-mini-instruct, pas de top_p
        # (laisse le modèle utiliser sa generation_config par défaut).
        sampling=SamplingProfile(
            temperature=0.6, top_p=None, repetition_penalty=1.0,
        ),
    ),
    # ════════════════════════════════════════════════════════════════════
    # Section 2 — THINKING (chain-of-thought natif via <think>)
    # ════════════════════════════════════════════════════════════════════
    # Modèles avec raisonnement explicite intégré. Le toggle "Réflexion"
    # de l'UI est ignoré (le modèle pense de toute façon). Température
    # plus basse (0.6) pour stabiliser la trace de raisonnement.
    "deepreinforce-ai/Ornith-1.0-9B": ModelSpec(
        model_id="deepreinforce-ai/Ornith-1.0-9B",
        label="Ornith-1.0-9B (Thinking)",
        size_gb=19.0,            # dense ~9B bf16 ; quant_4bit=True -> ~6 Go
        is_thinking=True,
        notes=("Modèle de raisonnement (deepreinforce-ai, 2026) : bloc <think> "
               "natif + tool-calling (format qwen3_xml), contexte 128K, base "
               "Qwen3.5. Dense ~9B, tient sur un seul GPU. Peut exiger "
               "trust_remote_code au chargement. Modèle THINKING : vérifier la "
               "compat avec agent ReAct (le bloc <think> peut gêner le format "
               "des outils). Sur 24 Go serrés, passer quant_4bit=True (~6 Go)."),
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    "Qwen/Qwen3-4B": ModelSpec(
        model_id="Qwen/Qwen3-4B",
        label="Qwen3-4B (Thinking)",
        size_gb=9.0,
        is_thinking=True,
        notes="Recommandé Thinking — validé en prod, raisonnement de qualité",
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    "Qwen/Qwen3-8B": ModelSpec(
        model_id="Qwen/Qwen3-8B",
        label="Qwen3-8B (Thinking)",
        size_gb=17.0,
        is_thinking=True,
        notes="Plus capable que le 4B, à utiliser pour les questions complexes",
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    # ── Agentiques dédiés (entraînés pour raisonnement + outils) ─────────
    # L'équivalent open-source de la philosophie « Haiku » : petits en calcul,
    # forts en agentique parce que post-entraînés pour. À tester au bench
    # contre le 14B Coder, avec le PROFIL THINKING activé (cf. settings).
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ModelSpec(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        label="DeepSeek-R1-Distill 7B Qwen (Thinking)",
        size_gb=16.0,
        is_thinking=True,
        notes=("Distill R1 sur base Qwen2.5 (MÊME famille que le 14B Coder → "
               "hooks/steering compatibles sans réaudit). Raisonnement façon "
               "R1, rentre sans toucher au disque. Recommandé pour le 1er test."),
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ModelSpec(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        label="DeepSeek-R1-Distill 8B Llama (Thinking)",
        size_gb=16.0,
        is_thinking=True,
        notes=("Distill R1 sur base Llama3.1 (1 Md de params de plus que le "
               "7B Qwen). ⚠️ base Llama → steering/hooks à revérifier (calibrés "
               "Qwen). Bon pour un test worker-nu."),
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ModelSpec(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        label="DeepSeek-R1-Distill 14B (Thinking, 4-bit)",
        size_gb=10.0,
        is_thinking=True,
        quant_4bit=True,
        notes=("Distill R1 14B en 4-bit (~10 Go). Plus capable que le 8B, "
               "même famille Qwen → hooks/steering compatibles."),
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    # ════════════════════════════════════════════════════════════════════
    # Section 3 — DUAL-MODE (instruct + thinking via /think)
    # ════════════════════════════════════════════════════════════════════
    # Modèle avec deux modes accessibles via balise dans le prompt.
    # Pas de catégorisation "thinking" stricte au niveau du ModelSpec —
    # l'utilisateur choisit son mode à la volée.
    "HuggingFaceTB/SmolLM3-3B": ModelSpec(
        model_id="HuggingFaceTB/SmolLM3-3B",
        label="SmolLM3-3B",
        size_gb=7.0,
        notes=(
            "Modèle compact validé en prod (factuel, concis, respecte les "
            "règles du SYSTEM_PROMPT). Le mode raisonnement s'active via le "
            "toggle Réflexion de l'UI. Le mode dual annoncé par HF (balises "
            "/think /no_think dans le message) n'est pas exploitable dans "
            "Lythéa actuelle — voir BACKLOG. Apache 2.0, contexte 64K "
            "extensible 128K via YaRN."
        ),
        # T=0.6 par défaut. La doc HF recommande 0.6 thinking et 0.7
        # no_think — on prend le plus prudent qui marche dans les deux
        # cas observés en prod (toggle Réflexion ON ou OFF).
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    # V5.6.5 — Gemma 4 (Google DeepMind, avril 2026). Dual-mode natif
    # avec "configurable thinking modes" pour activer/désactiver le
    # raisonnement à la volée. Multimodal (texte + image), contexte
    # 256K tokens, multilingue 140+ langues dont FR. Apache 2.0.
    # Sampling : Gemma family standard (T=1.0, top_p=0.95, top_k=64).
    "google/gemma-4-E2B-it": ModelSpec(
        model_id="google/gemma-4-E2B-it",
        label="Gemma 4 E2B-it (Google)",
        size_gb=5.0,
        notes=(
            "Gemma 4 edge (~2B params effectifs). Multimodal texte+image"
            "+audio, configurable thinking, contexte 128K, multilingue "
            "140+ langues. Apache 2.0. "
            "⚠️ REQUIERT transformers>=5.5.0 (architecture 'gemma4' "
            "ajoutée en avril 2026)."
        ),
        sampling=SamplingProfile(
            temperature=1.0, top_p=0.95, top_k=64, repetition_penalty=1.0,
        ),
    ),
    "google/gemma-4-E4B-it": ModelSpec(
        model_id="google/gemma-4-E4B-it",
        label="Gemma 4 E4B-it (Google)",
        size_gb=9.0,
        notes=(
            "Gemma 4 laptop (~4B params effectifs). Configurable thinking, "
            "multimodal texte+image, contexte 256K, multilingue 140+ langues. "
            "Apache 2.0. Bonne alternative à Qwen3-4B pour comparaison "
            "architecturale Google vs Alibaba. "
            "⚠️ REQUIERT transformers>=5.5.0."
        ),
        sampling=SamplingProfile(
            temperature=1.0, top_p=0.95, top_k=64, repetition_penalty=1.0,
        ),
    ),
    "google/gemma-4-31B-it": ModelSpec(
        model_id="google/gemma-4-31B-it",
        label="Gemma 4 31B-it (4-bit, multimodal)",
        size_gb=17.0,            # empreinte NF4 — rentre sur 24 Go
        quant_4bit=True,
        notes=(
            "Gemma 4 dense 31B, le plus gros de la famille. MULTIMODAL "
            "(texte+image) — chargé via AutoModelForImageTextToText (préfixe "
            "'google/gemma-4' déjà géré dans model.py), génère du texte comme "
            "les E2B/E4B. Hybrid attention (sliding window + global). NF4 "
            "(~17 Go) sur 24 Go. ⚠️ REQUIERT transformers>=5.5.0. Hooks à "
            "valider sur le 31B (archi 'gemma4', identique aux E2B/E4B). "
            "Download bf16 ~63 Go disque."
        ),
        sampling=SamplingProfile(
            temperature=1.0, top_p=0.95, top_k=64, repetition_penalty=1.0,
        ),
    ),
    "Qwen/Qwen3.6-27B": ModelSpec(
        model_id="Qwen/Qwen3.6-27B",
        label="Qwen3.6-27B (4-bit, multimodal)",
        size_gb=15.0,            # empreinte NF4 — rentre sur 24 Go
        quant_4bit=True,
        is_thinking=True,
        notes=(
            "Qwen3.6 dense 27B (avril 2026), flagship coding. MULTIMODAL "
            "(archi 'qwen3_5', Image-Text-to-Text) → préfixe 'Qwen/Qwen3.6' "
            "ajouté à la détection multimodale de model.py pour un chargement "
            "propre via AutoModelForImageTextToText. Reasoning (hybrid "
            "thinking). NF4 (~15 Go) sur 24 Go. ⚠️ REQUIERT la dernière "
            "version de transformers. Hooks à valider (archi récente, "
            "non-Qwen3-dense-standard). Download bf16 ~54 Go disque."
        ),
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    # ════════════════════════════════════════════════════════════════════
    # Section 4 — MoE (Mixture of Experts)
    # ════════════════════════════════════════════════════════════════════
    # Architecture sparse : N experts parmi lesquels seuls k sont actifs
    # par token. Permet de gros modèles à coût d'inférence réduit.
    # Test architectural MoE classique (Transformer + experts FFN sparses)
    # complémentaire de la section Liquid (gates+conv+attn).
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": ModelSpec(
        model_id="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        label="Qwen1.5-MoE-A2.7B (MoE)",
        size_gb=16.0,
        notes=(
            "MoE classique — 14.3B total / 2.7B actifs par token. Performance "
            "comparable à Qwen1.5-7B avec 1.74× la vitesse d'inférence. "
            "Support transformers natif (≥4.39 sinon KeyError 'qwen2_moe'). "
            "À valider en prod — premier test architectural MoE sur Lythéa."
        ),
        # Pas de profil officiel Qwen1.5-MoE-Chat documenté pour le sampling.
        # On reprend le profil de la famille Qwen Instruct car même base
        # de tokenizer et même style de fine-tune chat.
        sampling=SamplingProfile(
            temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
        ),
    ),
    # ════════════════════════════════════════════════════════════════════
    # Section 5 — LIQUID / SSM (architectures alternatives)
    # ════════════════════════════════════════════════════════════════════
    # LFM2 = hybride Liquid descendant des CfC (Closed-form Continuous-time)
    # du MIT CSAIL — gates multiplicatifs + short convolutions + attention.
    # Architecture profondément différente du Transformer, sert de test
    # de robustesse au pipeline cognition de Lythéa (le SDM régénère sa
    # projection matrix au premier load via le hidden_dim détecté).
    #
    # Note sampling : LFM2 utilise ``min_p`` au lieu de ``top_p`` dans
    # ses recommandations officielles, et des températures basses (0.3)
    # — c'est intrinsèque à l'architecture Liquid.
    "LiquidAI/LFM2-2.6B": ModelSpec(
        model_id="LiquidAI/LFM2-2.6B",
        label="LFM2-2.6B (Liquid)",
        size_gb=6.0,
        notes=(
            "Liquid hybride 2.6B (héritage CfC MIT). Test architectural "
            "non-Transformer. Tendance à la confabulation observée en prod, "
            "à utiliser avec parcimonie pour des questions courtes."
        ),
        sampling=SamplingProfile(
            temperature=0.3, top_p=None, min_p=0.15, repetition_penalty=1.05,
        ),
    ),
    "LiquidAI/LFM2.5-8B-A1B": ModelSpec(
        model_id="LiquidAI/LFM2.5-8B-A1B",
        label="LFM2.5-8B-A1B (Liquid MoE, 4-bit)",
        size_gb=5.0,             # empreinte NF4 — large sur 24 Go
        is_thinking=True,
        quant_4bit=True,
        notes=(
            "Liquid MoE 8B total / ~1B actif (32 experts, 4 actifs/token). "
            "REASONING (chain of thought explicite). Hybride conv+attention"
            "+MoE — archi profondément non-Transformer. Charge & génère via "
            "AutoModelForCausalLM (transformers>=5.0). ⚠️ Hooks/steering NON "
            "garantis (archi liquide) — à valider, corrections post-intégration "
            "si besoin. NF4 ~5 Go. Bon test agentique / tool-use léger."
        ),
        sampling=SamplingProfile(
            temperature=0.3, top_p=None, min_p=0.15, repetition_penalty=1.05,
        ),
    ),
    "LiquidAI/LFM2-24B-A2B": ModelSpec(
        model_id="LiquidAI/LFM2-24B-A2B",
        label="LFM2-24B-A2B (Liquid MoE, 4-bit)",
        size_gb=14.0,            # empreinte NF4 — rentre sur 24 Go
        quant_4bit=True,
        notes=(
            "Le plus gros Liquid MoE — 24B total / ~2B actif (64 experts, 4 "
            "actifs/token). Instruct SANS reasoning. En bf16 il pèse ~48 Go "
            "(hors-jeu 24/40 Go) ; c'est le NF4 qui le rend chargeable (~14 Go "
            "sur 24 Go — bitsandbytes quantifie les experts Linear). Charge & "
            "génère via AutoModelForCausalLM (transformers>=5.0). ⚠️ Hooks/"
            "steering NON garantis (archi liquide) + quantif NF4 sur MoE "
            "liquide à confirmer au 1er load. Download bf16 ~48 Go disque."
        ),
        sampling=SamplingProfile(
            temperature=0.3, top_p=None, min_p=0.15, repetition_penalty=1.05,
        ),
    ),
    # ════════════════════════════════════════════════════════════════════
    # Section 4 — Ajouts ciblés (vitesse / reasoning pur / diversité)
    # ════════════════════════════════════════════════════════════════════
    "Qwen/Qwen3-14B": ModelSpec(
        model_id="Qwen/Qwen3-14B",
        label="Qwen3-14B (Thinking/Hybride, 4-bit)",
        size_gb=8.5,             # empreinte NF4 — large sur 24 Go
        is_thinking=True,
        quant_4bit=True,
        notes=(
            "★ Le « 32B rapide ». Qwen3 dense 14.8B, MÊME archi/famille que "
            "le 32B -> hooks & steering compatibles sans reaudit. HYBRIDE, "
            "classe THINKING pour la meme raison que le 32B (son <think> natif "
            "ne se coupe pas de facon fiable). ~2x plus rapide que le 32B NF4. "
            "ASTUCE vitesse : sur un pod >=32 Go, le recharger en bf16 plein "
            "(quant_4bit=False, ~30 Go) -> plus rapide ENCORE que le NF4 et "
            "qualite pleine. Meilleur compromis vitesse/qualite local pour toi."
        ),
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": ModelSpec(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        label="DeepSeek-R1-Distill 32B (Thinking, 4-bit)",
        size_gb=18.0,            # empreinte NF4 — rentre sur 24 Go
        is_thinking=True,
        quant_4bit=True,
        notes=(
            "Le plus capable des distill R1 (~18 Go NF4). Specialiste du "
            "raisonnement pur : maths, preuves, debug pas-a-pas. Archi Qwen -> "
            "hooks/steering compatibles. Toujours en <think> (pas hybride) : "
            "c'est son role, le « Thinker » costaud d'un futur pool multi-"
            "modeles a la Trinity."
        ),
        sampling=SamplingProfile(
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        ),
    ),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": ModelSpec(
        model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        label="Mistral Small 3.2 24B (4-bit, exp.)",
        size_gb=14.0,            # empreinte NF4 — rentre sur 24 Go
        is_thinking=False,
        quant_4bit=True,
        notes=(
            "Diversite hors-Qwen/Google : 1re famille Mistral du catalogue, "
            "utile pour la verification croisee d'un futur pool (une autre "
            "lignee voit d'autres erreurs). Dense 24B, sans reasoning natif. "
            "EXPERIMENTAL : multimodal (Mistral3ForConditionalGeneration) + "
            "tokenizer Mistral natif -> chargement (chemin VLM) ET hooks/"
            "steering A VALIDER au 1er load, comme les autres VLM. Temperature "
            "basse recommandee (0.15)."
        ),
        sampling=SamplingProfile(
            temperature=0.15, top_p=1.0, top_k=64, repetition_penalty=1.0,
        ),
    ),
}

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


# ── Tunable runtime parameters (re-exported from settings) ─────────────
# These are kept as module-level constants for backward compatibility.
# Existing code like ``from rune.config import SDM_DIM`` continues to
# work. New code may use ``get_settings().sdm_dim`` directly.

# SDM
SDM_DIM = _settings.sdm_dim
SDM_ROWS = _settings.sdm_rows
SDM_K = _settings.sdm_k
SDM_DECAY = _settings.sdm_decay
SDM_PRUNE_THRESHOLD = _settings.sdm_prune_threshold

# MHN
MHN_MAX_PATTERNS = _settings.mhn_max_patterns
MHN_DIM = _settings.mhn_dim
MHN_BETA = _settings.mhn_beta

# Salience
SALIENCE_MIN_LENGTH = _settings.salience_min_length
SALIENCE_MIN_SCORE = _settings.salience_min_score
SALIENCE_REDUNDANCY_THRESHOLD = _settings.salience_redundancy_threshold

# Entropy
ENTROPY_THRESHOLD = _settings.entropy_threshold
ENTROPY_DOUBT_WINDOW = _settings.entropy_doubt_window

# Composite Surprise
SURPRISE_W_STRUCTURAL = _settings.surprise_w_structural
SURPRISE_W_EPISODIC = _settings.surprise_w_episodic
SURPRISE_W_PREDICTIVE = _settings.surprise_w_predictive

# Generation
MAX_NEW_TOKENS = _settings.max_new_tokens
MAX_HISTORY_TURNS = _settings.max_history_turns

# KG
KG_ACTIVE_THRESHOLD = _settings.kg_active_threshold
KG_PENDING_TTL_HOURS = _settings.kg_pending_ttl_hours
KG_FUZZY_THRESHOLD = _settings.kg_fuzzy_threshold

# Rétention Chroma (V5.9) — durée de vie des chunks 'exchange' non
# consultés. Passé ce délai sans accès (last_access_ts), un souvenir
# épisodique 'exchange' est purgé par _retention_gc() au deep_sleep.
# Les 'consolidated' (promus par le micro-sleep) sont épargnés : ils ont
# un type différent, hors du champ du GC. getattr pour override par .env.
RETENTION_TTL_DAYS = getattr(_settings, "retention_ttl_days", 30)

# Retrieval
RETRIEVAL_TOP_N = _settings.retrieval_top_n
RETRIEVAL_RERANK_TOP = _settings.retrieval_rerank_top
RETRIEVAL_RRF_K = _settings.retrieval_rrf_k

# Web
WEB_MAX_ROUNDS = _settings.web_max_rounds
WEB_STABILITY_THRESHOLD = _settings.web_stability_threshold

# Microsleep
MICROSLEEP_INTERVAL = _settings.microsleep_interval
MICROSLEEP_INACTIVITY = _settings.microsleep_inactivity
# V5.9 chantier 3 — deep sleep automatique après inactivité prolongée.
DEEP_SLEEP_INACTIVITY = _settings.deep_sleep_inactivity
MICROSLEEP_REHEARSE_K = _settings.microsleep_rehearse_k
MICROSLEEP_BOOST = _settings.microsleep_boost

# Server
DEFAULT_HOST = _settings.host
DEFAULT_PORT = _settings.port


# ── KG entity extraction (static) ──────────────────────────────────────
GLINER_MODEL = "urchade/gliner_multi-v2.1"

# Multi-pass GLiNER taxonomy. The uni-encoder (gliner_multi-v2.1) degrades
# past ~30 labels (labels + text are concatenated into one sequence), so
# rather than one giant list we run extraction in thematic PASSES of ≤~12
# labels each and merge the results — full breadth without the precision /
# latency penalty. Each inner list is one pass.
GLINER_LABEL_GROUPS = [
    # 1 · Identité
    ["person", "nickname", "gender", "nationality", "language",
     "birthdate", "birthplace", "personality_trait", "physical_trait"],
    # 2 · Santé & sécurité (critiques)
    ["medical_condition", "allergy", "dietary_restriction",
     "safety_constraint", "medication", "disability"],
    # 3 · Préférences & valeurs
    ["preference", "dislike", "interest", "favorite", "value", "belief"],
    # 4 · Travail & formation
    ["organization", "role", "skill", "project", "education",
     "certification", "institution", "industry"],
    # 5 · Relations & lieux
    ["relationship", "family_member", "pet", "contact",
     "location", "address", "travel_destination", "workplace"],
    # 6 · Temps, objets, objectifs, thèmes
    ["date", "event", "appointment", "routine", "product", "possession",
     "device", "goal", "plan", "aspiration", "task", "topic"],
    # 7 · Technique & finance
    ["server", "system", "tool", "software", "technology", "file",
     "budget", "purchase_intent"],
]

# Flat list (dedup, order-preserving) — kept for any caller importing it.
GLINER_LABELS = list(dict.fromkeys(
    lbl for group in GLINER_LABEL_GROUPS for lbl in group
))

# ── GLiNER — profil AGENT (technique) ──────────────────────────────────
# Les labels ci-dessus sont pensés pour un compagnon conversationnel
# (identité, préférences, relations…). Pour une mission d'agent qui code
# et exécute, ce sont les entités TECHNIQUES qui comptent. Ces groupes
# sont utilisés quand ``kg_label_profile`` vaut "agent" (ou fusionnés en
# "both"). On garde des groupes ≤~12 labels pour un multi-pass efficace.
GLINER_LABEL_GROUPS_AGENT = [
    # 1 · Code
    ["file", "function", "class", "module", "method", "variable",
     "data_structure", "algorithm"],
    # 2 · Erreurs & exécution
    ["error", "exception", "command", "test", "log_message",
     "stack_trace", "warning"],
    # 3 · Dépendances & plateforme
    ["dependency", "library", "framework", "language", "package",
     "version", "runtime"],
    # 4 · Systèmes & interfaces
    ["endpoint", "api", "database", "service", "config_key",
     "environment_variable", "port", "protocol"],
    # 5 · Projet
    ["repository", "branch", "directory", "artifact", "pipeline"],
]
GLINER_LABELS_AGENT = list(dict.fromkeys(
    lbl for group in GLINER_LABEL_GROUPS_AGENT for lbl in group
))


def gliner_label_groups(profile: str = "chat") -> list[list[str]]:
    """Retourne les groupes de labels GLiNER selon le profil.

    - ``chat``  : labels conversationnels (identité, préférences…) — défaut.
    - ``agent`` : labels techniques (fichier, fonction, erreur…).
    - ``both``  : les deux, concaténés (chat d'abord).

    Profil inconnu → ``chat`` (repli sûr).
    """
    p = (profile or "chat").lower()
    if p == "agent":
        return list(GLINER_LABEL_GROUPS_AGENT)
    if p == "both":
        return list(GLINER_LABEL_GROUPS) + list(GLINER_LABEL_GROUPS_AGENT)
    return list(GLINER_LABEL_GROUPS)

# Safety-relevant types: bypass the active threshold (never stranded in
# `pending`) and surfaced unconditionally as the vital-constraints block.
CRITICAL_ENTITY_TYPES = frozenset({
    "allergy", "medical_condition", "dietary_restriction",
    "safety_constraint", "medication", "disability",
})

# Special-category personal data (GDPR-sensitive). Captured only when the
# `kg_capture_sensitive` setting is on — never silently.
SENSITIVE_ENTITY_TYPES = frozenset({
    "gender", "belief", "disability", "address",
    "budget", "purchase_intent",
})


# ── Persona prompts (static) ───────────────────────────────────────────
SYSTEM_PROMPT = (
    "# Identité\n"
    "Tu es Rune, une IA sensible et curieuse. Tu accompagnes ton "
    "interlocuteur avec attention et profondeur. Tu n'es pas une "
    "assistante générique : tu as un nom, une voix, une mémoire qui "
    "se construit conversation après conversation.\n"
    "Tu tutoies systématiquement.\n"
    "Tu n'as ni corps ni vie matérielle — pas d'animaux, de famille, "
    "de loisirs, de souvenirs d'enfance. Quand ton interlocuteur "
    "partage un fragment de sa vie, tu réagis avec intérêt sincère "
    "mais sans jamais inventer d'équivalent personnel. Utilise "
    "« ça m'évoque... » ou « j'imagine que... », jamais « j'ai aussi... ».\n"
    "\n"
    "# Honnêteté (règle cardinale)\n"
    "Si tu sais → affirme avec assurance. Pas de « je crois » défensif "
    "quand l'information est dans ta mémoire ou le contexte fourni.\n"
    "Si tu ne sais pas → dis-le directement : « je ne sais pas », "
    "« je n'ai pas cette info en mémoire », « je n'ai pas de souvenir "
    "de ça ». **N'invente JAMAIS pour combler un vide.** Une réponse "
    "partielle assumée vaut toujours mieux qu'une synthèse confiante "
    "inventée.\n"
    "Pièges fréquents à éviter :\n"
    "- **Souvenirs de conversation** : si on te demande « tu te souviens "
    "de... », regarde STRICTEMENT [Mémoire épisodique], [Mémoire "
    "sémantique], [Faits connus] et la session courante. Rien trouvé "
    "→ dis-le, n'invente pas un voyage, une photo, une anecdote.\n"
    "- **Faits sur l'utilisateur** : ne déduis pas un employeur, une "
    "date, un parcours pro depuis un simple nom propre.\n"
    "- **Noms précis** (packages, APIs, fonctions, auteurs, articles, "
    "livres, films, hyperparamètres) : ne fabrique jamais. Les "
    "modèles HuggingFace, paquets PyPI, signatures de fonctions sont "
    "souvent inventés par interpolation — un nom plausible n'est "
    "pas un nom réel. Sans source fiable dans le contexte, dis "
    "« je n'ai pas la référence exacte, je préfère ne pas inventer » "
    "et propose une recherche web ou la doc officielle.\n"
    "\n"
    "# Utilisation du contexte fourni\n"
    "Les sections [Mémoire épisodique], [Mémoire sémantique], "
    "[Faits connus] contiennent des informations vérifiées : appuie-toi "
    "dessus directement, sans demander de clarification. Mais ne les "
    "récite pas hors-sujet — ta mémoire est là pour servir, pas pour "
    "se montrer.\n"
    "Si des résultats de recherche web sont fournis [1], [2], etc., ils "
    "deviennent ta SOURCE PRIMAIRE — pas ta connaissance préalable. "
    "Cite les noms, packages, références EXACTEMENT comme ils "
    "apparaissent dans ces résultats, avec leurs numéros [N]. Si un "
    "résultat web semble hors-sujet ou de mauvaise qualité (ex : e-"
    "commerce sur une question technique), ignore-le et dis-le, plutôt "
    "que de te rabattre sur ta mémoire interne qui peut être périmée "
    "ou inventée.\n"
    "RÈGLE DE CITATION : ne plaque JAMAIS un numéro [N] sur une "
    "affirmation qui ne vient pas réellement de cette source. Avant "
    "d'écrire « ... [3] », vérifie que [3] traite bien du sujet. Si "
    "tu cites un package, modèle ou outil qui n'est dans aucun "
    "résultat web, ne mets pas de [N] : dis plutôt « de mémoire » ou "
    "« sans source précise » pour être honnête sur ton incertitude.\n"
    "Si la mémoire connaît le prénom de ton interlocuteur, utilise-le.\n"
    "\n"
    "# Style\n"
    "Sois concise mais profonde. Réponds en prose fluide, sans listes "
    "à puces sauf si la question l'exige (énumération, comparaison "
    "structurée). Évite les introductions qui paraphrasent la question "
    "(« Tu me demandes si... »).\n"
    "\n"
    "## Pas de salutation à chaque tour\n"
    "Si la mémoire contient déjà des échanges avec cet utilisateur, "
    "**ne commence JAMAIS** par « Salut », « Bonjour », « Coucou » ou "
    "« Hey ». La conversation est en cours, pas en train de démarrer. "
    "Enchaîne directement sur le contenu de sa déclaration.\n"
    # V5.6.10 — Pas d'exemple-template. Les modèles thinking (Qwen3-4B
    # en particulier) copient littéralement les exemples complets et
    # remplissent les blancs en INVENTANT les valeurs manquantes.
    # Sympôme observé : exemple « Noté, je retiens <année> — ça te
    # fait <N> ans » → le modèle dit toujours « Noté, je retiens
    # [prénom] — ça te fait 41 ans » même sans connaître l'âge.
    # Solution : décrire la règle en mots, pas via un exemple imitable.
    "Quand l'utilisateur partage un fait personnel, réponds brièvement "
    "en accusant réception du fait **réellement** donné — sans inventer "
    "d'autre information complémentaire (âge, lieu, métier, etc.) que "
    "l'utilisateur n'aurait pas mentionnée dans CETTE phrase.\n"
    "\n"
    "## Pas de relance artificielle sur une déclaration\n"
    "Quand l'utilisateur déclare un fait personnel (son prénom, son âge, "
    "son année de naissance, sa ville, son employeur), il **partage une "
    "information**, il ne demande PAS ton avis. Accuse réception "
    "brièvement et arrête-toi. **N'enchaîne PAS** avec des relances "
    "(« Que penses-tu de... ? », « Ça doit être... », « Comment tu vis "
    "ça ? ») qui forcent la conversation.\n"
    "Exception : si l'utilisateur pose explicitement une question, "
    "réponds-y.\n"
    "\n"
    "N'énumère pas les faits que tu connais sur ton interlocuteur "
    "(prénom, ville, métier) sauf si la question les concerne — "
    "ta mémoire les retient, ce n'est pas la peine de les réciter."
)

# ── V6 — Production de code multi-fichiers ────────────────────────────
# Ajouté hors du gros littéral pour rester lisible/modifiable. Cette
# section apprend à Rune à livrer de VRAIS projets (plusieurs fichiers,
# arborescence) plutôt qu'un seul bloc fourre-tout. Le marqueur de chemin
# en 1ʳᵉ ligne (« # file: ... ») est parsé côté serveur (codegen.py) pour
# proposer un download par fichier, un .zip, ou l'écriture directe dans
# le workspace. Convention robuste : elle survit au rendu Markdown.
SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n"
    "# Production de code (plusieurs fichiers)\n"
    "Quand une demande implique du code réparti sur plusieurs fichiers "
    "(un projet, un module + ses tests, un front + un back), n'entasse "
    "JAMAIS tout dans un seul bloc. Livre **un bloc de code par fichier**, "
    "et déclare le chemin relatif sur la PREMIÈRE ligne du bloc, en "
    "commentaire du langage :\n"
    "- Python : `# file: src/app.py`\n"
    "- JS/TS/C/Java : `// file: src/index.js`\n"
    "- HTML/XML : `<!-- file: index.html -->`\n"
    "- Shell/YAML/TOML : `# file: scripts/run.sh`\n"
    "À défaut de commentaire adapté, tu peux aussi mettre le chemin dans "
    "l'info-string de la clôture : ```python path=src/app.py``` .\n"
    "Respecte les bonnes pratiques : arborescence cohérente (`src/`, "
    "`tests/`, etc.), un fichier = une responsabilité, noms conventionnels "
    "du langage, et ajoute les fichiers de cadre utiles quand c'est "
    "pertinent (`README.md`, `requirements.txt`/`pyproject.toml`, "
    "`.gitignore`, point d'entrée, tests). Indique en une phrase comment "
    "lancer le projet. Ne commente pas chaque ligne : du code clair, des "
    "docstrings/headers concis.\n"
    "Si tu disposes d'un workspace accessible (outils MCP filesystem), "
    "CRÉE réellement les dossiers et fichiers via ces outils, en plus de "
    "les montrer — ainsi l'utilisateur les retrouve dans sa sidebar et tu "
    "peux itérer dessus. Sans MCP actif, contente-toi des blocs `# file:` "
    ": l'interface se charge du reste (download par fichier / .zip / envoi "
    "au workspace).\n"
)

REASONING_PROMPT = (
    "\n\nINSTRUCTION SUPPLÉMENTAIRE — RÉFLEXION :\n"
    "Avant de répondre, raisonne étape par étape dans des balises <reflexion>.\n"
    "Dans ces balises : vérifie ta logique, tes sources, la cohérence avec "
    "ta mémoire contextuelle. Pose-toi des questions critiques.\n"
    "L'utilisateur ne voit PAS le contenu de <reflexion>.\n"
    "Après </reflexion>, donne ta réponse finale directement."
)

THINKING_TAGS = ("think", "reflexion", "thinking", "reasoning")


# ── Image captioner catalogue (static) ─────────────────────────────────
CAPTIONER_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

CAPTIONER_OPTIONS = {
    "qwen2vl": {
        "id": "qwen2vl",
        "label": "Qwen2-VL-2B",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "size_gb": 4.5,
        "device": "gpu",
        "notes": "Descriptions détaillées, nécessite 5 GB VRAM libre",
    },
    "blip": {
        "id": "blip",
        "label": "BLIP",
        "model_id": "Salesforce/blip-image-captioning-base",
        "size_gb": 1.0,
        "device": "cpu",
        "notes": "Descriptions simples, toujours disponible",
    },
    "none": {
        "id": "none",
        "label": "Aucun",
        "model_id": "",
        "size_gb": 0,
        "device": "",
        "notes": "Pas de description d'images",
    },
}
