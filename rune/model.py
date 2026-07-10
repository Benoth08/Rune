"""Model wrapper with VRAM protections and entropy capture.

Supports LLM and thinking models from HuggingFace. Images are handled
by a separate Florence-2 captioner on CPU — no VLM code path needed.
"""
from __future__ import annotations

import gc
import logging
import os
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from rune.config import CATALOG, DEVICE, DTYPE, ModelSpec

log = logging.getLogger("rune.model")


# ── V5.6.8 — Helper config multimodal compatibility ────────────────────
#
# Les modèles vision-language (Gemma 4, PaliGemma, Llava, Qwen2-VL…)
# utilisent une config wrapper qui contient des sub-configs (text_config,
# vision_config). Plusieurs attributs (vocab_size, hidden_size,
# max_position_embeddings) ne sont plus directement sur la config root
# mais sur text_config. Sans gérer ça, on a :
#
#   AttributeError: 'Gemma4Config' object has no attribute 'vocab_size'
#
# Ces helpers recherchent l'attribut dans plusieurs emplacements connus,
# avec des fallbacks gracieux.

def _config_attr(config: Any, attr: str, default: Any = None) -> Any:
    """Cherche un attribut sur config, puis sur config.text_config,
    puis via config.get_text_config(). Retourne ``default`` si introuvable.

    Couvre les configs flat (Qwen, Phi, Llama) et nested (Gemma 4,
    PaliGemma, Llava, Qwen2-VL).

    V5.6.11 — élargit la recherche à toutes les sub-configs connues
    (language_model_config, llm_config, decoder_config) car Gemma 4
    et les modèles audio (Gemma-3n) peuvent imbriquer la sub-config
    sous différents noms.
    """
    # 1. Attribut direct (modèles classiques)
    val = getattr(config, attr, None)
    if val is not None:
        return val

    # 2-4. Sub-configs nommées : text_config (Gemma 4, PaliGemma, Llava),
    #      language_model_config (Idefics), llm_config (variantes),
    #      decoder_config (parfois)
    for sub_name in ("text_config", "language_model_config", "llm_config",
                     "decoder_config"):
        sub_cfg = getattr(config, sub_name, None)
        if sub_cfg is not None:
            val = getattr(sub_cfg, attr, None)
            if val is not None:
                log.debug("%s lu depuis config.%s", attr, sub_name)
                return val

    # 5. Méthode officielle get_text_config (transformers ≥5.x)
    if hasattr(config, "get_text_config"):
        try:
            text_cfg = config.get_text_config()
            if text_cfg is not None:
                val = getattr(text_cfg, attr, None)
                if val is not None:
                    log.debug("%s lu depuis config.get_text_config()", attr)
                    return val
        except Exception:
            pass

    # 6. Méthode get_language_model_config (Idefics-style)
    if hasattr(config, "get_language_model_config"):
        try:
            lm_cfg = config.get_language_model_config()
            if lm_cfg is not None:
                val = getattr(lm_cfg, attr, None)
                if val is not None:
                    return val
        except Exception:
            pass

    return default


def _get_vocab_size(config: Any, tokenizer: Any = None) -> int:
    """Retourne la taille du vocabulaire ; fallback sur tokenizer."""
    val = _config_attr(config, "vocab_size")
    if val:
        return int(val)
    if tokenizer is not None:
        size = len(tokenizer)
        log.warning(
            "vocab_size introuvable dans config.%s, fallback tokenizer=%d",
            type(config).__name__, size,
        )
        return size
    log.error(
        "Impossible de déterminer vocab_size pour config %s — "
        "valeur par défaut 50000 utilisée",
        type(config).__name__,
    )
    return 50000


# ── Abstract interface ─────────────────────────────────────────────────

@dataclass
class GenerationOutput:
    """Output for a single generation step."""

    token_id: int
    token_str: str
    entropy: float
    latent_state: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # TODO: CfC states (gate_mean, dt, top_down_error)


class ModelWrapper(ABC):
    """Abstract interface for language model wrappers."""

    @abstractmethod
    def load(self, model_id: str, **kwargs: Any) -> None: ...
    @abstractmethod
    def unload(self) -> None: ...
    @abstractmethod
    def encode(self, text: str) -> torch.Tensor: ...
    @abstractmethod
    def decode(self, ids: torch.Tensor) -> str: ...
    @abstractmethod
    def embed(self, text: str) -> torch.Tensor: ...
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str: ...
    @abstractmethod
    def generate_step(self, input_ids: torch.Tensor, **kwargs: Any) -> tuple[GenerationOutput, Any]: ...
    @abstractmethod
    def analyze_input(self, text: str) -> dict[str, Any]: ...
    @abstractmethod
    def stream_generate(self, messages: list[dict], max_new_tokens: int, cancelled: threading.Event | None) -> Generator[dict, None, None]: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...
    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...


# ── Download progress ──────────────────────────────────────────────────

@dataclass
class DownloadProgress:
    """Tracks model download state for UI streaming."""

    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bps: float = 0.0
    files_done: int = 0
    files_total: int = 0
    current_file: str = ""
    error: str | None = None
    finished: bool = False

    @property
    def pct(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return min(self.downloaded_bytes / self.total_bytes * 100, 100.0)


# ── Entropy capture via LogitsProcessor ────────────────────────────────

class _EntropyProcessor:
    """LogitsProcessor that captures per-token entropy without modifying logits."""

    def __init__(self, entropy_q: queue.Queue, vocab_size: int) -> None:
        self.q = entropy_q
        self.log_vocab = np.log(max(vocab_size, 2))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        try:
            with torch.no_grad():
                probs = torch.softmax(scores, dim=-1)
                ent = -(probs * (probs + 1e-10).log()).sum(-1).mean().item()
                normalized = ent / self.log_vocab
            self.q.put_nowait(normalized)
        except Exception:
            pass
        return scores


# ── LM Head entropy hook (universal backup) ───────────────────────────

class _LMHeadEntropyHook:
    """Hook on lm_head to capture per-token entropy."""

    def __init__(self, vocab_size: int) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=4096)
        self.log_vocab = np.log(max(vocab_size, 2))
        self.handle: Any = None

    def hook_fn(self, module: Any, inp: Any, out: Any) -> None:
        try:
            with torch.no_grad():
                tensor = out[0] if isinstance(out, tuple) else out
                if tensor.dim() >= 3:
                    logits = tensor[:, -1, :]
                elif tensor.dim() == 2:
                    logits = tensor
                else:
                    return
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * (probs + 1e-10).log()).sum(-1).mean().item()
                normalized = ent / self.log_vocab
            self.q.put_nowait(normalized)
        except Exception:
            pass

    def register(self, model: Any) -> bool:
        """Find and hook the lm_head."""
        lm_head = None
        for path in ("lm_head", "language_model.lm_head", "model.lm_head"):
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, torch.nn.Module):
                    lm_head = obj
                    log.info("LM head found via %s", path)
                    break
            except AttributeError:
                continue

        if lm_head is None:
            log.warning("LM head not found — entropy via hook disabled")
            return False
        self.handle = lm_head.register_forward_hook(self.hook_fn)
        return True

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ── Latent capture via forward hook ────────────────────────────────────

class _LatentHook:
    """Captures the last-token hidden state from a decoder layer."""

    def __init__(self, latent_q: queue.Queue) -> None:
        self.q = latent_q
        self.handle: Any = None

    def hook_fn(self, module: Any, inp: Any, out: Any) -> None:
        try:
            tensor = out[0] if isinstance(out, tuple) else out
            if tensor is not None and tensor.dim() >= 2:
                self.q.put_nowait(tensor[:, -1, :].detach().cpu())
        except Exception:
            pass

    def register(self, layer: torch.nn.Module) -> None:
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ── VRAM utilities ─────────────────────────────────────────────────────

def vram_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def vram_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    _, total = torch.cuda.mem_get_info()
    return total / 1e9


def precheck_vram(model_id: str) -> tuple[bool, str]:
    """Check if VRAM is sufficient before downloading."""
    if DEVICE == "cpu":
        return True, "CPU mode"
    spec = CATALOG.get(model_id)
    if spec is None:
        return True, "Model not in catalogue"
    free = vram_free_gb()
    total = vram_total_gb()
    needed = spec.size_gb
    if free >= needed + 1.0:
        return True, f"{free:.1f} GB free"
    if total >= needed + 1.0:
        return True, f"Tight ({free:.1f} free, {total:.0f} total)"
    return False, f"Need {needed} GB, GPU has {total:.0f} GB total"


def model_loadability_info(
    model_id: str,
    captioner_backend: str = "",
    captioner_vram_gb: float = 0.0,
    current_loaded_id: str | None = None,
    current_loaded_size_gb: float = 0.0,
) -> dict:
    """Compute detailed loadability info for a model card in the UI.

    Returns a dict with:
    - ``loadable``: bool — can be loaded right now without freeing anything
    - ``loadable_with_blip``: bool — would be loadable if user switched
      a GPU captioner (Qwen2-VL) to BLIP (CPU)
    - ``loadable_after_unload``: bool — would be loadable after unloading
      the currently-loaded LLM
    - ``vram_required_gb``, ``vram_available_gb``: numeric for tooltips
    - ``block_reason``: human-readable French explanation, or None

    The grayed-out logic on the frontend reads ``loadable``. The other
    fields drive smart suggestions ("switch to BLIP", "auto-unload").

    Parameters
    ----------
    model_id : str
        Target model id from the catalog.
    captioner_backend : str
        Currently active captioner backend ("qwen2vl", "blip", "" for none).
    captioner_vram_gb : float
        VRAM consumed by the current captioner (0 if CPU or none).
    current_loaded_id : str, optional
        Currently loaded LLM id, if any.
    current_loaded_size_gb : float
        Size of currently loaded LLM (already counted in ``vram_free_gb``,
        used to compute the ``loadable_after_unload`` projection).
    """
    spec = CATALOG.get(model_id)
    if spec is None:
        return {
            "loadable": False,
            "loadable_with_blip": False,
            "loadable_after_unload": False,
            "vram_required_gb": 0.0,
            "vram_available_gb": 0.0,
            "block_reason": "Modèle inconnu",
        }

    if DEVICE == "cpu":
        return {
            "loadable": True,
            "loadable_with_blip": True,
            "loadable_after_unload": True,
            "vram_required_gb": spec.size_gb,
            "vram_available_gb": 0.0,
            "block_reason": None,
        }

    free = vram_free_gb()
    total = vram_total_gb()
    needed = spec.size_gb + 1.0  # 1 GB safety margin

    loadable = free >= needed
    # If a GPU captioner is active, switching to BLIP frees its VRAM
    blip_extra = captioner_vram_gb if captioner_backend == "qwen2vl" else 0.0
    loadable_with_blip = (free + blip_extra) >= needed
    # If we unload the current LLM first, we recover its memory
    unload_extra = (
        current_loaded_size_gb
        if current_loaded_id and current_loaded_id != model_id
        else 0.0
    )
    loadable_after_unload = (free + unload_extra) >= needed

    block_reason = None
    if not loadable:
        deficit = needed - free
        block_reason = (
            f"VRAM insuffisante : besoin de {needed:.1f} GB, "
            f"{free:.1f} GB libre (manque {deficit:.1f} GB)."
        )
        if not loadable_with_blip and not loadable_after_unload:
            if total < needed:
                block_reason += f" Le GPU n'a que {total:.0f} GB au total."
        elif loadable_with_blip and captioner_backend == "qwen2vl":
            block_reason += (
                f" Bascule sur BLIP (CPU) dans Paramètres → Vision pour "
                f"libérer {blip_extra:.1f} GB."
            )
        elif loadable_after_unload and current_loaded_id:
            block_reason += (
                f" Le déchargement de {current_loaded_id} libérera "
                f"{unload_extra:.1f} GB."
            )

    return {
        "loadable": loadable,
        "loadable_with_blip": loadable_with_blip,
        "loadable_after_unload": loadable_after_unload,
        "vram_required_gb": round(spec.size_gb, 1),
        "vram_available_gb": round(free, 1),
        "block_reason": block_reason,
    }


# ── Attention fallback chain ───────────────────────────────────────────

_ATTN_KEYWORDS = ("flash_attn", "flashattention", "sdpa", "attn_implementation")


def _resolve_attn() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _attn_chain(preferred: str) -> list[str]:
    chain = [preferred]
    if preferred != "sdpa":
        chain.append("sdpa")
    if preferred != "eager":
        chain.append("eager")
    return chain


def _offload_kwargs(model_id: str) -> dict[str, Any]:
    """Auto CPU offload if VRAM is tight."""
    spec = CATALOG.get(model_id)
    if spec is None or DEVICE == "cpu":
        return {}
    free = vram_free_gb()
    needed = spec.size_gb
    if needed > free - 1.0:
        cap = int(max(free - 1.5, 2)) if free > 2 else 2
        log.info("VRAM tight (%.1f GB free, need %.1f) — enabling CPU offload", free, needed)
        return {"max_memory": {0: f"{cap}GiB", "cpu": "48GiB"}}
    return {}


# ── HuggingFace LLM wrapper ───────────────────────────────────────────

class HFModelWrapper(ModelWrapper):
    """HuggingFace LLM wrapper with entropy hooks."""

    def __init__(self, progress_callback: Any | None = None) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self._model_id: str | None = None
        self._spec: ModelSpec | None = None
        self._hidden_dim: int = 0
        self._is_thinking: bool = False
        # V5.6.15 — Flag multimodal natif. True si le modèle peut
        # ingérer directement des images via son processor (pas besoin
        # de captionneur externe). Mis à jour au chargement par
        # _detect_native_multimodal() en s'appuyant sur _MULTIMODAL_PREFIXES.
        self._is_natively_multimodal: bool = False
        # Processor associé (AutoProcessor) pour les modèles multimodaux.
        # None pour les modèles texte-only.
        self._processor: Any = None
        self._context_length: int = 0
        self._latent_hook: _LatentHook | None = None
        self._progress_cb = progress_callback

        # analyze_input cache. Entries are big (hidden states tensor),
        # so the default size is small. Cleared on every load/unload
        # because results depend on the loaded model's weights.
        from rune.cache import BoundedCache
        from rune.settings import get_settings
        self._analyze_cache: BoundedCache = BoundedCache(
            max_size=get_settings().analyze_cache_size,
            name="model.analyze_input",
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def is_thinking(self) -> bool:
        return self._is_thinking

    @property
    def is_natively_multimodal(self) -> bool:
        """V5.6.15 — True si le modèle ingère les images directement.

        Pour ces modèles (Gemma 3/4 actuellement), Lythéa passe les
        images au modèle via son processor (apply_chat_template avec
        content=[{type:"image", image: PIL}, {type:"text", text: ...}])
        au lieu de les faire passer par le captionneur Qwen2-VL.

        Conséquences :
        - Plus rapide (pas de double inférence VLM puis LLM)
        - Plus précis (le modèle "voit" l'image, pas un texte)
        - Économise la VRAM du captionneur (~3-4 GB pour Qwen2-VL-2B)
        """
        return self._is_natively_multimodal

    @property
    def processor(self) -> Any:
        """V5.6.15 — Processor AutoProcessor pour les modèles multimodaux.

        None pour les modèles texte-only. Utilisé pour combiner texte +
        images dans le prompt et pour appliquer le chat template adapté.
        """
        return self._processor

    @property
    def context_length(self) -> int:
        """Taille max du contexte du modèle en tokens.

        Détectée depuis ``config.max_position_embeddings`` au chargement.
        Utilisée pour calculer dynamiquement la limite de taille des
        documents uploadables (un Qwen 7B avec 32K tokens accepte des
        docs plus petits qu'un Qwen 32B avec 128K tokens). ``0`` si
        aucun modèle chargé.
        """
        return self._context_length

    # ── Load ───────────────────────────────────────────────────────────

    def load(self, model_id: str, progress_q: queue.Queue | None = None, **kwargs: Any) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # V5.6.9 — Auto class additionnelle pour les modèles multimodaux
        # (Gemma 4, Llava, PaliGemma…) qui ne sont pas dans le mapping
        # CausalLM et plantent avec "Unrecognized configuration class".
        try:
            from transformers import AutoModelForImageTextToText
            _HAS_IMAGE_TEXT_AUTO = True
        except ImportError:
            AutoModelForImageTextToText = None
            _HAS_IMAGE_TEXT_AUTO = False

        def _report(pct: float, status: str = "loading"):
            if progress_q is not None:
                progress_q.put({"pct": round(pct, 1), "status": status, "model_id": model_id})

        if self.is_loaded:
            self.unload()

        self._spec = CATALOG.get(model_id)
        self._is_thinking = self._detect_thinking(model_id)

        # ── Phase 1 (0-55%): Download / verify cache ──────────────────
        _report(2, "downloading")
        try:
            from huggingface_hub import snapshot_download, scan_cache_dir
    
            # Check if model is already fully cached
            expected_bytes = (self._spec.size_gb if self._spec else 7.0) * 1024**3
            already_cached = False
            try:
                cache_info = scan_cache_dir()
                for repo in cache_info.repos:
                    if repo.repo_id == model_id:
                        if repo.size_on_disk > expected_bytes * 0.8:
                            already_cached = True
                        break
            except Exception:
                pass

            if already_cached:
                _report(55, "loading_weights")
                log.info("Model %s already cached, skipping download", model_id)
            else:
                # Download in a sub-thread, monitor cache dir growth
                dl_done = threading.Event()
                dl_error: list[str] = []

                def _download():
                    try:
                        snapshot_download(
                            model_id,
                            ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot",
                                             "*.md", "*.txt", "LICENSE*"],
                        )
                    except Exception as exc:
                        dl_error.append(str(exc))
                    finally:
                        dl_done.set()

                dl_thread = threading.Thread(target=_download, daemon=True)
                dl_thread.start()

                while not dl_done.is_set():
                    # Check cache growth
                    try:
                        cache_info = scan_cache_dir()
                        for repo in cache_info.repos:
                            if repo.repo_id == model_id:
                                ratio = min(repo.size_on_disk / max(expected_bytes, 1), 1.0)
                                _report(2 + ratio * 53, "downloading")  # 2% → 55%
                                break
                    except Exception:
                        pass
                    dl_done.wait(1.0)

                if dl_error:
                    log.warning("snapshot_download failed: %s", dl_error[0])

        except Exception as exc:
            log.warning("snapshot_download pre-fetch skipped: %s", exc)

        _report(55, "loading_weights")

        # ── Phase 2 (55-92%): Load weights into GPU ───────────────────
        attn_preferred = _resolve_attn()
        chain = _attn_chain(attn_preferred)
        offload = _offload_kwargs(model_id)

        # Security: trust_remote_code lets HF model repos execute arbitrary
        # Python on import. Default False; opt-in via LYTHEA_ALLOW_REMOTE_CODE.
        from rune.settings import get_settings
        trust_remote = get_settings().allow_remote_code
        if trust_remote:
            log.warning(
                "trust_remote_code=True (LYTHEA_ALLOW_REMOTE_CODE) — "
                "model %s may execute arbitrary code on load",
                model_id,
            )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": DTYPE,
            "device_map": "auto",
            "trust_remote_code": trust_remote,
            "ignore_mismatched_sizes": True,
            **offload,
            **kwargs,
        }

        # 4-bit NF4 (bitsandbytes) for large MoE models, so a 30B fits a 24 GB
        # GPU (~18 GB). Graceful: if bitsandbytes is missing, fall back to bf16
        # (+ CPU offload) with a clear warning rather than crashing.
        # Modèle DÉJÀ quantifié (AWQ/GPTQ) : on NE touche à rien. transformers
        # lit la quantization_config embarquée dans le dépôt et charge les
        # poids 4-bit directement. Pas de bitsandbytes, donc pas le bug v5
        # (#43032). On pose juste un device_map mono-GPU et on retire le
        # torch_dtype (porté par la config du modèle).
        if self._spec is not None and getattr(self._spec, "quant_4bit", False):
            try:
                import bitsandbytes as _bnb  # noqa: F401
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=DTYPE,
                )
                # CRITICAL: a top-level torch_dtype passed ALONGSIDE a
                # BitsAndBytesConfig makes recent transformers materialise the
                # weights in bf16 BEFORE quantising → a 30B tries to allocate
                # ~22 GB and OOMs on a 24 GB card. The compute dtype is already
                # carried by bnb_4bit_compute_dtype, so the top-level one must
                # be removed for the 4-bit path to actually shrink the model.
                load_kwargs.pop("torch_dtype", None)
                # Reduce allocator fragmentation (the OOM hint itself suggests
                # this) — helps the quantised load fit the remaining headroom.
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                                      "expandable_segments:True")
                # v5 BUG WORKAROUND (transformers core_model_loading): with
                # device_map={"":0}, the new loader materialises every tensor
                # on GPU at FULL precision BEFORE quantising → a 30B tries to
                # alloc ~22 GB and OOMs on 24 GB. Using device_map="auto" WITH
                # an explicit per-device max_memory forces accelerate to place
                # (and quantise) shard by shard, so the GPU never holds the
                # full-precision model at once. A small CPU budget catches any
                # overflow tail instead of OOMing outright.
                try:
                    _ngpu = torch.cuda.device_count()
                    _free_gb = (torch.cuda.mem_get_info(0)[0] / 1e9
                                if torch.cuda.is_available() else 0)
                except Exception:  # noqa: BLE001
                    _ngpu, _free_gb = 1, 0
                if _ngpu > 1:
                    load_kwargs["device_map"] = "balanced"
                    load_kwargs.pop("max_memory", None)
                else:
                    # Garde ~1.5 GB de marge GPU pour le KV cache + GLiNER ;
                    # le CPU n'accueille que le débordement éventuel (bnb ne
                    # peut pas exécuter sur CPU, mais accélère le placement).
                    _gpu_budget = max(8, int(_free_gb) - 2) if _free_gb else 20
                    load_kwargs["device_map"] = "auto"
                    # CPU budget large : avec low_cpu_mem_usage=False, la
                    # construction pleine précision transite par la RAM avant
                    # quantification → on laisse de la marge (le pod RunPod a
                    # généralement ≥100 Go de RAM).
                    load_kwargs["max_memory"] = {0: f"{_gpu_budget}GiB",
                                                 "cpu": "80GiB"}
                # CLÉ du contournement du bug v5 (#43032 / diffusers #12799) :
                # low_cpu_mem_usage=FALSE. Contre-intuitif, mais c'est le fix
                # documenté. En v5, avec bitsandbytes + device_map, le chemin
                # "low cpu mem" garde les tenseurs sur META device ; l'état de
                # quantification (absmax, code) reste alors sur meta et ne peut
                # PAS être déplacé vers le GPU → soit ça échoue, soit le loader
                # retombe sur une matérialisation pleine précision (l'OOM à
                # 22 GB). En forçant low_cpu_mem_usage=False, les tenseurs sont
                # matérialisés PENDANT le chargement, la quantification
                # s'applique réellement, et seuls ~18 GB de poids 4-bit
                # atterrissent sur le GPU.
                load_kwargs["low_cpu_mem_usage"] = False
                log.info("Loading %s in 4-bit NF4 (low_cpu_mem_usage=False, "
                         "sharded auto-map, gpu_budget=%sGiB) — v5 OOM workaround",
                         model_id, locals().get("_gpu_budget", "?"))
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "4-bit demandé pour %s mais bitsandbytes indisponible (%s) "
                    "— repli bf16. Installe bitsandbytes pour réduire la VRAM.",
                    model_id, exc,
                )

        # Monitor VRAM growth during from_pretrained
        vram_monitor_stop = threading.Event()
        expected_gb = self._spec.size_gb if self._spec else 7.0

        def _vram_monitor():
            """Poll VRAM usage and estimate loading progress."""
            baseline = 0.0
            if DEVICE != "cpu" and torch.cuda.is_available():
                baseline = torch.cuda.memory_allocated(0) / (1024 ** 3)
            while not vram_monitor_stop.is_set():
                if DEVICE != "cpu" and torch.cuda.is_available():
                    current = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    delta = current - baseline
                    ratio = min(delta / max(expected_gb, 0.5), 1.0)
                    pct = 55 + ratio * 37  # 55% → 92%
                    _report(pct, "loading_weights")
                vram_monitor_stop.wait(0.4)

        monitor_thread = threading.Thread(target=_vram_monitor, daemon=True)
        monitor_thread.start()

        # V5.6.11 — Détection a priori des familles multimodales connues.
        # Pour ces modèles, on saute AutoModelForCausalLM (qui peut
        # charger silencieusement mais produire un wrapper config sans
        # vocab_size top-level) et on utilise directement
        # AutoModelForImageTextToText. Évite l'erreur cryptique
        # "'Gemma4Config' object has no attribute 'vocab_size'".
        _MULTIMODAL_PREFIXES = (
            "google/gemma-3",
            "google/gemma-4",
            "google/paligemma",
            "llava-hf/llava",
            "liuhaotian/llava",
            "Qwen/Qwen2-VL",
            "Qwen/Qwen2.5-VL",
            "Qwen/Qwen3.6",  # Qwen3.6 (27B, 35B-A3B) — archi qwen3_5, Image-Text-to-Text
            "microsoft/Phi-3-vision",
            "microsoft/Phi-3.5-vision",
            "HuggingFaceM4/Idefics",
            "meta-llama/Llama-3.2-11B-Vision",
            "meta-llama/Llama-3.2-90B-Vision",
        )
        force_multimodal = (
            _HAS_IMAGE_TEXT_AUTO and any(
                model_id.startswith(p) for p in _MULTIMODAL_PREFIXES
            )
        )
        if force_multimodal:
            log.info(
                "Modèle %s détecté multimodal a priori → utilisation directe "
                "de AutoModelForImageTextToText",
                model_id,
            )

        try:
            for impl in chain:
                try:
                    log.info("Loading %s with attn=%s", model_id, impl)
                    if force_multimodal:
                        # Chargement direct multimodal — bypasse CausalLM
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            model_id, attn_implementation=impl, **load_kwargs,
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_id, attn_implementation=impl, **load_kwargs,
                        )
                    break
                except (ImportError, ValueError, RuntimeError, KeyError, AttributeError) as exc:
                    msg = str(exc).lower()
                    if any(k in msg for k in _ATTN_KEYWORDS):
                        log.warning("Attn %s failed, trying next: %s", impl, exc)
                        continue
                    # V5.6.11 — Si CausalLM échoue avec un signal
                    # multimodal (mots-clés OU AttributeError sur
                    # vocab_size/text_config indiquant config wrapper),
                    # bascule sur ImageTextToText. AttributeError ajouté
                    # à except pour capturer le cas Gemma 4 quand la
                    # détection a priori a raté.
                    needs_multimodal = (
                        "unrecognized" in msg
                        or "does not support" in msg
                        or "image-text-to-text" in msg
                        or "imagetext" in msg
                        or "vocab_size" in msg
                        or "text_config" in msg
                        or "no attribute" in msg
                    )
                    if needs_multimodal and _HAS_IMAGE_TEXT_AUTO and not force_multimodal:
                        log.info(
                            "CausalLM échec multimodal sur %s, retry avec "
                            "ImageTextToText (cause: %s)",
                            model_id, exc,
                        )
                        try:
                            self.model = AutoModelForImageTextToText.from_pretrained(
                                model_id, attn_implementation=impl, **load_kwargs,
                            )
                            log.info("Loaded as ImageTextToText: %s", model_id)
                            break
                        except Exception as exc2:
                            log.warning("ImageTextToText also failed: %s", exc2)
                            raise exc2
                    raise
        finally:
            vram_monitor_stop.set()
            monitor_thread.join(timeout=2)

        if self.model is None:
            raise RuntimeError(f"All attention backends failed for {model_id}")

        # ── Phase 3 (92-100%): Tokenizer + hooks ─────────────────────
        _report(93, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)

        # V5.6.15 — Chargement du processor pour les modèles
        # nativement multimodaux. Le processor gère :
        #  - le pré-traitement des images (resize, normalize, tile)
        #  - le chat template adapté au format multimodal
        #    (messages avec content=[{type:"image"}, {type:"text"}])
        # On le charge uniquement pour les modèles détectés comme
        # multimodaux a priori (Gemma 3/4 pour l'instant). Pour les
        # autres, on garde tokenizer-only et le pipeline image passe
        # par le captionneur Qwen2-VL externe (comportement V5.5.x).
        if force_multimodal:
            try:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote,
                    padding_side="left",  # recommandé par doc Gemma 3
                )
                self._is_natively_multimodal = True
                log.info(
                    "Processor multimodal chargé pour %s (flag natively_multimodal=True)",
                    model_id,
                )
            except Exception as exc:
                log.warning(
                    "Échec chargement AutoProcessor pour %s — le modèle "
                    "est marqué multimodal mais sans processor, l'inférence "
                    "image directe sera désactivée. Cause: %s",
                    model_id, exc,
                )
                self._processor = None
                self._is_natively_multimodal = False
        else:
            self._processor = None
            self._is_natively_multimodal = False

        config = self.model.config
        # V5.6.9 — Patch config multimodale (Gemma 4, PaliGemma, Llava…)
        # qui n'exposent pas vocab_size/hidden_size/etc au top niveau.
        # Transformers en interne (hooks, generate, samplers) accède
        # parfois directement à config.vocab_size — sans ce patch ça
        # plante avec "AttributeError: 'Gemma4Config' object has no
        # attribute 'vocab_size'". On synchronise les attrs critiques
        # depuis text_config si absents au top niveau.
        _CRITICAL_CONFIG_ATTRS = (
            "vocab_size", "hidden_size", "num_attention_heads",
            "num_hidden_layers", "intermediate_size", "max_position_embeddings",
            "pad_token_id", "bos_token_id", "eos_token_id",
            "rope_theta", "rms_norm_eps", "tie_word_embeddings",
        )
        for attr in _CRITICAL_CONFIG_ATTRS:
            if not hasattr(config, attr) or getattr(config, attr, None) is None:
                val = _config_attr(config, attr)
                if val is not None:
                    try:
                        setattr(config, attr, val)
                    except Exception as exc:
                        log.debug("Config patch skipped %s: %s", attr, exc)

        # V5.6.8 — passe par _config_attr pour gérer les configs
        # multimodales nested (Gemma 4, PaliGemma, Llava, Qwen2-VL).
        self._hidden_dim = (
            _config_attr(config, "hidden_size")
            or _config_attr(config, "d_model")
            or 768
        )

        # Détection du contexte max. Plusieurs noms possibles selon
        # l'archi : max_position_embeddings (Qwen/Llama/Mistral/Gemma),
        # n_positions (GPT-2 style), max_sequence_length (autres).
        # Fallback : 4096 (très conservateur, plus petit que tout
        # modèle moderne).
        self._context_length = (
            _config_attr(config, "max_position_embeddings")
            or _config_attr(config, "n_positions")
            or _config_attr(config, "max_sequence_length")
            or 4096
        )

        _report(97, "hooks")
        self._register_latent_hook()
        self._model_id = model_id
        self.model.eval()
        _report(100, "ready")
        log.info("Model %s loaded (hidden_dim=%d, context=%d tokens, thinking=%s)",
                 model_id, self._hidden_dim, self._context_length, self._is_thinking)

    def _detect_thinking(self, model_id: str) -> bool:
        spec = CATALOG.get(model_id)
        if spec is not None:
            return spec.is_thinking
        thinking_markers = ("QwQ", "R1", "o1", "thinking", "Qwen3")
        return any(m in model_id for m in thinking_markers)

    def _apply_chat_template_safe(self, tokenizer, messages,
                                  think: bool | None = None, **kwargs):
        """Apply chat template, driving ``enable_thinking`` from the spec.

        Hybrid models (Qwen3 and derivatives) accept an
        ``enable_thinking`` boolean. When ``True`` the template stays
        "open" and the model emits real ``<think>...</think>`` reasoning;
        when ``False`` it injects an empty ``<think></think>`` and
        suppresses reasoning.

        CRUCIAL: for Qwen3 the template DEFAULT is
        ``enable_thinking=True``. So to run a hybrid model in
        non-thinking mode we must pass ``enable_thinking=False``
        EXPLICITLY — merely omitting the kwarg leaves the model free to
        reason (the bug that made Qwen3-32B emit ``<think>`` even with
        ``is_thinking=False``). We therefore always pass the flag,
        mirroring the spec:

          - ``is_thinking=True``  → ``True``  → ``<think>`` appears and is
            extracted by :func:`rune.cognition.generation.strip_reasoning`;
          - ``is_thinking=False`` → ``False`` → reasoning suppressed.

        Templates that don't know the kwarg (Qwen2.5, Mistral, Phi…)
        raise ``TypeError`` → we fall back to the plain call. Those
        models can't think anyway, so the behaviour is unchanged.
        """
        # ``think`` overrides the spec's static value when not None — used to
        # force non-thinking on trivial messages (a bare greeting shouldn't
        # trigger a full <think> on a thinking model).
        _think = self._is_thinking if think is None else bool(think)
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=_think, **kwargs,
            )
        except TypeError:
            # Template doesn't accept enable_thinking → plain call.
            return tokenizer.apply_chat_template(messages, **kwargs)

    def _register_latent_hook(self) -> None:
        """Register forward hook on the last decoder layer.

        ⚠️ MoE-safe : sur un gros MoE (Qwen3-30B-A3B = 192 couches × 128
        experts), le fallback ``named_modules()`` parcourt des dizaines de
        milliers de modules et peut FIGER le chargement plusieurs minutes.
        On privilégie donc les chemins directs (``model.layers`` etc.) et on
        ne tombe sur le balayage complet qu'en tout dernier recours, borné.
        """
        if self._latent_hook is not None:
            self._latent_hook.remove()

        layers = None
        search_paths = [
            "model.layers", "model.model.layers", "model.model.model.layers",
            "language_model.model.layers", "transformer.h", "gpt_neox.layers",
        ]
        for attr in search_paths:
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, '__len__') and len(obj) > 10:
                    layers = obj
                    log.info("Decoder layers found via %s (%d layers)", attr, len(obj))
                    break
            except (AttributeError, TypeError):
                continue

        if layers is None:
            # Fallback borné : on s'arrête au PREMIER ModuleList plausible et on
            # ignore les sous-modules d'experts (mlp.experts) pour ne pas
            # parcourir tout le graphe MoE (qui fige sur un 30B).
            for name, module in self.model.named_modules():
                if "experts" in name or "expert" in name:
                    continue
                if isinstance(module, torch.nn.ModuleList) and len(module) > 10:
                    layers = module
                    log.info("Decoder layers found dynamically via '%s' (%d layers)", name, len(module))
                    break

        if layers is not None:
            self._latent_hook = _LatentHook(queue.Queue(maxsize=2048))
            self._latent_hook.register(layers[-1])
            log.info("Latent hook registered on last decoder layer")
        else:
            log.warning("No decoder layers found — SDM latent writes disabled")

    # ── Unload ─────────────────────────────────────────────────────────

    def unload(self) -> None:
        if self._latent_hook is not None:
            self._latent_hook.remove()
            self._latent_hook = None

        if self.model is not None:
            try:
                self.model.to("meta", non_blocking=True)
            except Exception:
                pass

        self.model = None
        self.tokenizer = None
        self._model_id = None
        self._spec = None
        self._hidden_dim = 0
        self._is_thinking = False
        # V5.6.15 — Reset des flags multimodaux au déchargement.
        self._is_natively_multimodal = False
        self._processor = None

        # Cache invalidation: cached hidden states reflected the previous
        # model's weights and would be wrong if reused with a different LLM.
        self._analyze_cache.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        log.info("Model unloaded, VRAM freed")

    # ── Encode / Decode / Embed ────────────────────────────────────────

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt").squeeze(0)

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def embed(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model(**ids, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            mask = ids["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.squeeze(0).cpu()

    # ── Analyze input ──────────────────────────────────────────────────

    def analyze_input(self, text: str) -> dict[str, Any]:
        """Compute per-token entropy and hidden states for input text.

        Cached: repeated analysis of the same input within a session
        (rare but possible) returns the memoised dict. The cache is
        keyed on the *text* and implicitly on the loaded model, since
        we clear it on load/unload.
        """
        cached = self._analyze_cache.get(text)
        if cached is not None:
            return cached
        result = self._analyze_input_uncached(text)
        if result is not None:
            self._analyze_cache.put(text, result)
        return result

    def _analyze_input_uncached(self, text: str) -> dict[str, Any]:
        """Actual analysis logic without cache."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(
                **inputs,
                labels=inputs.get("input_ids"),
                output_hidden_states=True,
            )
            logits = out.logits
            hidden = out.hidden_states[-1]
            native_loss = out.loss.item() if out.loss is not None else None

        vocab_size = logits.shape[-1]
        log_vocab = np.log(max(vocab_size, 2))

        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * (probs + 1e-10).log()).sum(-1)
        normalized = (ent / log_vocab).squeeze(0).cpu().tolist()

        mean_ent = native_loss / log_vocab if native_loss is not None else float(np.mean(normalized))

        # ``hidden`` is detached implicitly by torch.no_grad above; we
        # also call .cpu() so the cache doesn't pin GPU memory.
        return {
            "token_entropies": normalized,
            "latent_states": hidden.squeeze(0).cpu(),
            "mean_entropy": min(mean_ent, 1.0),
            "vocab_size": vocab_size,
        }

    def cache_stats(self) -> dict:
        """Return analyze_input cache statistics."""
        return self._analyze_cache.stats()

    # ── Simple generate ────────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs: Any) -> str:
        pil_images = kwargs.get("pil_images")
        if pil_images and getattr(self, "_is_natively_multimodal", False) \
                and getattr(self, "_processor", None) is not None:
            return self._generate_multimodal(prompt, pil_images, **kwargs)
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            do_sample=True,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
        )
        # Anti-répétition optionnel, réservé aux générations de TEXTE pur (ex.
        # synthèse de l'agent) : on casse à la SOURCE la dérive en boucle des
        # modèles thinking au lieu de la rattraper après coup. Transmis à
        # transformers UNIQUEMENT si fourni → zéro impact sur les autres appels,
        # notamment la génération de code où répéter `return`, une indentation
        # ou un mot-clé est parfaitement légitime.
        if kwargs.get("repetition_penalty") is not None:
            gen_kwargs["repetition_penalty"] = float(kwargs["repetition_penalty"])
        if kwargs.get("no_repeat_ngram_size") is not None:
            gen_kwargs["no_repeat_ngram_size"] = int(kwargs["no_repeat_ngram_size"])
        # Optional early stop on literal strings (e.g. the agent stops right
        # after its first complete tool_call instead of rambling to EOS or the
        # token cap — a direct per-step latency win). Uses a StoppingCriteria
        # on the decoded tail, so it works regardless of tokenization.
        stops = kwargs.get("stop_strings") or []
        cancel_event = kwargs.get("cancel_event")
        criteria = []
        if stops:
            from transformers import StoppingCriteria, StoppingCriteriaList

            tok, n_in = self.tokenizer, ids.input_ids.shape[1]
            max_len = max(len(s) for s in stops)

            class _StopOnStrings(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):  # noqa: ANN001
                    tail_ids = input_ids[0][n_in:][-(max_len // 2 + 16):]
                    tail = tok.decode(tail_ids, skip_special_tokens=True)
                    return any(s in tail for s in stops)

            criteria.append(_StopOnStrings())
        # Réactivité du STOP : un StoppingCriteria qui consulte un threading.Event
        # à CHAQUE token. Quand l'utilisateur clique « stop », l'event est levé et
        # la génération s'arrête au token suivant (< 1 s) — même mécanisme que le
        # chat. C'est ce qui rend le bouton stop réactif PENDANT une génération
        # (et pas seulement entre les étapes).
        if cancel_event is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _StopOnCancel(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):  # noqa: ANN001
                    return bool(cancel_event.is_set())

            criteria.append(_StopOnCancel())
        if criteria:
            from transformers import StoppingCriteriaList
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)
        with torch.no_grad():
            out = self.model.generate(**ids, **gen_kwargs)
        text = self.tokenizer.decode(out[0][ids.input_ids.shape[1]:],
                                     skip_special_tokens=True)
        # The stop string itself stays in the output (callers parse it).
        return text

    def _generate_multimodal(self, prompt: str, pil_images: list,
                             **kwargs: Any) -> str:
        """Génération avec images natives (cerveau multimodal type Gemma 4).

        Construit un message multimodal (images + texte) via le processor et
        décode la réponse. Utilisé uniquement quand le modèle est nativement
        multimodal ET que des images sont fournies."""
        proc = self._processor
        content = [{"type": "image", "image": im} for im in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        try:
            inputs = proc.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
        except Exception:  # noqa: BLE001 — processor sans chat template image
            log.warning("multimodal chat template failed, text-only fallback",
                        exc_info=True)
            return self.generate(prompt, **{k: v for k, v in kwargs.items()
                                            if k != "pil_images"})
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                do_sample=True,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
            )
        n_in = inputs["input_ids"].shape[1]
        return proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate several prompts in ONE GPU pass.

        Decoding is memory-bandwidth-bound, so a batch of 2-4 costs barely
        more than a single generation — this is what makes best-of-N (and
        parallel subagent steps) cheap. Sampling is independent per row, so
        K copies of the same prompt yield K diverse candidates. Left padding
        (decoder-only requirement) is restored afterwards."""
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], **kwargs)]
        tok = self.tokenizer
        old_side = tok.padding_side
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        try:
            enc = tok(list(prompts), return_tensors="pt",
                      padding=True).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=kwargs.get("max_new_tokens", 512),
                    do_sample=True,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    pad_token_id=tok.pad_token_id,
                )
        finally:
            tok.padding_side = old_side
        n_in = enc.input_ids.shape[1]
        return [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]

    def complete_sync(
        self,
        messages: list[dict],
        max_new_tokens: int = 64,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> str:
        """Synchronous chat completion for short utility tasks.

        Used by the V4.4 web classifier (decide if a question needs
        web search). Designed for short prompts and short outputs —
        the timeout safeguards against runaway generations on large
        unexpected inputs.

        Parameters
        ----------
        messages
            List of ``{"role": "system"|"user"|"assistant", "content": ...}``
            dicts. Passed through the model's chat template via the
            safe helper (handles ``enable_thinking`` for thinking
            models). For the classifier we always want a direct
            answer, so even on thinking models we keep the template
            in its default open mode — the model is free to think
            briefly but the short max_new_tokens bounds the cost.
        max_new_tokens
            Hard token budget. Default 64 — enough for a one-line
            classification response.
        temperature
            Sampling temperature. Default 0.3 (low) so the classifier
            is consistent across identical questions.
        timeout
            Soft hint, not enforced inside ``generate()`` (would
            require thread orchestration). Currently logged only.
            The hard limit comes from ``max_new_tokens`` + the
            model's natural speed.

        Returns
        -------
        str
            The generated text (without the prompt). Empty string on
            tokenizer failures (caller should treat empty as failure).
        """
        if not self.is_loaded:
            return ""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return ""
        try:
            text_prompt = self._apply_chat_template_safe(
                self.tokenizer, messages,
                tokenize=False, add_generation_prompt=True,
            )
        except Exception as exc:
            log.warning("complete_sync apply_chat_template failed: %s", exc)
            return ""
        try:
            ids = self.tokenizer(text_prompt, return_tensors="pt", truncation=True)
            ids = {k: v.to(self.model.device) for k, v in ids.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            input_len = ids["input_ids"].shape[1]
            new_tokens = out[0][input_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as exc:
            log.warning("complete_sync generation failed: %s", exc)
            return ""

    # ── Single step (for future CfC) ──────────────────────────────────

    def generate_step(
        self, input_ids: torch.Tensor, past_key_values: Any = None, **kwargs: Any,
        # TODO: CfC states (gate_mean, dt, top_down_error)
    ) -> tuple[GenerationOutput, Any]:
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids.to(self.model.device),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * (probs + 1e-10).log()).sum(-1).item()
        vocab_size = logits.shape[-1]
        normalized_ent = ent / np.log(max(vocab_size, 2))
        token_id = torch.multinomial(probs, 1).item()
        token_str = self.tokenizer.decode([token_id])
        latent = out.hidden_states[-1][:, -1, :].detach().cpu() if out.hidden_states else None

        return GenerationOutput(
            token_id=token_id, token_str=token_str, entropy=normalized_ent,
            latent_state=latent, logits=logits.cpu(),
        ), out.past_key_values

    # ── Streaming generation with entropy hooks ────────────────────────

    def stream_generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        cancelled: threading.Event | None = None,
        sampling: Any = None,
        pil_images: list | None = None,
        think_override: bool | None = None,
    ) -> Generator[dict, None, None]:
        """Stream tokens with real-time entropy capture.

        Parameters
        ----------
        messages
            Chat history in OpenAI-style format.
        max_new_tokens
            Hard cap on output length (used as fallback when ``sampling``
            does not specify its own ``max_new_tokens``).
        cancelled
            Optional event to interrupt streaming early.
        sampling
            Optional :class:`rune.config.SamplingProfile` describing
            the model's recommended sampling parameters. If ``None``,
            falls back to legacy hardcoded defaults (T=0.7, top_p=0.9).
        pil_images
            V5.6.15 — Optional list of PIL Images to attach to the last
            user message when the model is nativally multimodal (Gemma
            3/4). Ignored if the model has no processor or if the list
            is empty. Each image is added to the last user turn's
            ``content`` array following the Gemma 3 chat format::

                content = [
                    {"type": "image", "image": pil_image_1},
                    {"type": "image", "image": pil_image_2},
                    {"type": "text",  "text":  "user message"},
                ]

        Yields
        ------
        dict
            Keys: text, token, entropy, latent, step
        """
        from transformers import TextIteratorStreamer

        # V5.6.15 — Détection du mode multimodal natif.
        # Si le modèle est multimodal natif ET qu'on a des images,
        # on construit les inputs via le processor au lieu du tokenizer.
        use_multimodal = (
            pil_images
            and self._is_natively_multimodal
            and self._processor is not None
        )

        if use_multimodal:
            # Réécrit le dernier message user pour intégrer les images
            # dans son content. Format Gemma 3/4.
            mm_messages: list[dict] = []
            for i, msg in enumerate(messages):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Sur le DERNIER message user, on attache les images.
                # Les anciens user messages restent text-only (les
                # images étaient consommées dans leur tour respectif
                # côté serveur, pas dans l'historique).
                is_last_user = (
                    role == "user"
                    and i == len(messages) - 1
                    and pil_images
                )
                if is_last_user:
                    content_blocks: list[dict] = [
                        {"type": "image", "image": img}
                        for img in pil_images
                    ]
                    content_blocks.append(
                        {"type": "text", "text": str(content)}
                    )
                    mm_messages.append({"role": role, "content": content_blocks})
                else:
                    # Pour les autres tours, on garde la string (format
                    # historique). Le processor de Gemma 3 accepte les
                    # deux formats : string OU liste de blocs.
                    if isinstance(content, str):
                        mm_messages.append({
                            "role": role,
                            "content": [{"type": "text", "text": content}],
                        })
                    else:
                        mm_messages.append({"role": role, "content": content})

            try:
                inputs = self._processor.apply_chat_template(
                    mm_messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                log.info(
                    "Multimodal native generation: %d images, %d messages",
                    len(pil_images), len(messages),
                )
            except Exception as exc:
                log.warning(
                    "Échec apply_chat_template multimodal (%s), "
                    "fallback texte-only",
                    exc,
                )
                use_multimodal = False

        if not use_multimodal:
            # Prepare inputs — chemin texte-only classique (V5.5.x).
            if hasattr(self.tokenizer, "apply_chat_template"):
                text_prompt = self._apply_chat_template_safe(
                    self.tokenizer, messages, think=think_override,
                    tokenize=False, add_generation_prompt=True,
                )
            else:
                text_prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                )
            inputs = self.tokenizer(text_prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Entropy capture
        # V5.6.8 — Helper pour gérer les configs multimodales (Gemma 4,
        # PaliGemma, Llava…) qui n'exposent pas vocab_size au top niveau
        # mais dans config.text_config.vocab_size. Fallback gracieux sur
        # tokenizer si la config est exotique.
        vocab_size = _get_vocab_size(self.model.config, self.tokenizer)
        entropy_q: queue.Queue = queue.Queue(maxsize=4096)
        processor = _EntropyProcessor(entropy_q, vocab_size)

        lm_hook = _LMHeadEntropyHook(vocab_size)
        lm_hook_active = lm_hook.register(self.model)

        latent_q = self._latent_hook.q if self._latent_hook else queue.Queue()

        # Streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Build sampling kwargs from the profile, falling back to the
        # historical defaults when no profile is provided. Profile fields
        # set to ``None`` are omitted entirely so the model uses its
        # generation_config defaults rather than disabled sampling.
        # V4.4 — thinking models (Qwen3, QwQ, R1) get a higher default
        # token ceiling because their <think> block consumes a large
        # chunk of the budget before the answer starts. The profile
        # can still override via its own ``max_new_tokens`` if a
        # specific thinking model needs different.
        if self._is_thinking:
            from rune.settings import get_settings
            thinking_default = get_settings().thinking_max_new_tokens
        else:
            thinking_default = None  # use the standard cascade

        if sampling is not None:
            if sampling.max_new_tokens is not None:
                effective_max = sampling.max_new_tokens
            elif thinking_default is not None:
                effective_max = thinking_default
            else:
                effective_max = max_new_tokens
            sample_kwargs: dict[str, Any] = {
                "do_sample": True,
                "temperature": sampling.temperature,
                "repetition_penalty": sampling.repetition_penalty,
            }
            if sampling.top_p is not None:
                sample_kwargs["top_p"] = sampling.top_p
            if sampling.top_k is not None:
                sample_kwargs["top_k"] = sampling.top_k
            if sampling.min_p is not None:
                sample_kwargs["min_p"] = sampling.min_p
        else:
            effective_max = thinking_default if thinking_default is not None else max_new_tokens
            sample_kwargs = {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": effective_max,
            "logits_processor": [processor],
            **sample_kwargs,
        }

        # V3.9.4: capture thread errors so we can yield a structured
        # error to the consumer instead of silently producing a
        # truncated stream. The legacy implementation logged the error
        # but the caller had no way to know it happened — leading to
        # mid-sentence cuts in the UI when ``probability tensor
        # contains inf/nan`` was raised on long contexts.
        thread_error: dict[str, Any] = {}
        thread = threading.Thread(
            target=self._generate_thread,
            args=(gen_kwargs, thread_error),
            daemon=True,
        )
        thread.start()

        cumulated = ""
        step = 0
        try:
            for chunk in streamer:
                if cancelled and cancelled.is_set():
                    break
                if not chunk:
                    continue

                cumulated += chunk
                step += 1

                ent = 0.0
                if lm_hook_active:
                    try:
                        ent = lm_hook.q.get_nowait()
                    except queue.Empty:
                        pass
                if ent == 0.0:
                    try:
                        ent = entropy_q.get_nowait()
                    except queue.Empty:
                        pass

                latent = None
                try:
                    latent = latent_q.get_nowait()
                except queue.Empty:
                    pass

                yield {"text": cumulated, "token": chunk, "entropy": ent, "latent": latent, "step": step}
        finally:
            lm_hook.remove()
            thread.join(timeout=30)

        # V3.9.4: post-stream error inspection.
        # If the generation thread raised an ``inf/nan`` exception, the
        # streamer simply ended — without notifying the consumer. We
        # check the captured error and: (1) attempt a single retry
        # with a slightly higher temperature, which usually escapes
        # the degenerate distribution; (2) if retry also fails, yield
        # a structured error event so the UI can surface it instead
        # of pretending the truncated text is the real answer.
        err_msg = thread_error.get("error", "")
        if err_msg and "probability tensor" in err_msg.lower():
            log.warning(
                "Generation hit inf/nan after %d tokens — retrying with bumped T",
                step,
            )
            retry_text = self._retry_after_nan(
                inputs=inputs,
                effective_max=max(64, effective_max - step),
                sample_kwargs=sample_kwargs,
                cumulated=cumulated,
            )
            if retry_text:
                # Emit the completed text as a final chunk so the UI
                # gets the full answer.
                yield {
                    "text": cumulated + retry_text,
                    "token": retry_text,
                    "entropy": 0.0,
                    "latent": None,
                    "step": step + 1,
                    "recovered_from_nan": True,
                }
            else:
                # Retry also failed — surface a clear error.
                yield {
                    "text": cumulated,
                    "token": "",
                    "entropy": 0.0,
                    "latent": None,
                    "step": step,
                    "error": "generation_unstable",
                    "error_detail": (
                        "Le modèle a produit une distribution dégénérée "
                        "(inf/nan). Essaie une question plus courte ou "
                        "redémarre la session."
                    ),
                }

    def _retry_after_nan(
        self,
        inputs: dict,
        effective_max: int,
        sample_kwargs: dict,
        cumulated: str,
    ) -> str:
        """Retry generation once with a stable sampling profile.

        Bumps temperature slightly and forces ``do_sample=True`` to
        escape the degenerate distribution that triggered the inf/nan.
        Returns the additional text generated, or ``""`` if the retry
        also failed.

        We use a non-streaming ``generate`` here for simplicity — the
        retry is a recovery path, not the main code path.
        """
        try:
            retry_kwargs = dict(sample_kwargs)
            base_t = float(retry_kwargs.get("temperature", 0.7) or 0.7)
            retry_kwargs["temperature"] = min(1.2, base_t + 0.15)
            retry_kwargs["do_sample"] = True
            retry_kwargs.pop("top_k", None)  # let it sample more freely
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=effective_max,
                    **retry_kwargs,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text or ""
        except Exception as exc:
            log.error("inf/nan retry also failed: %s", exc)
            return ""

    def _generate_thread(self, kwargs: dict, error_out: dict) -> None:
        """Run ``model.generate`` in a thread, capturing any exception.

        ``error_out`` is a shared dict the caller created; on success
        it stays empty, on failure it gets ``{"error": str(exc)}``. The
        consumer of :meth:`stream_generate` inspects this after the
        streamer drains and can react (retry, yield error event, etc.).
        """
        try:
            with torch.no_grad():
                self.model.generate(**kwargs)
        except Exception as exc:
            log.error("Generation thread error: %s", exc)
            error_out["error"] = str(exc)

    # ── Image preprocessing (static, used by captioner) ────────────────

    @staticmethod
    def preprocess_image(img: Image.Image, max_side: int = 1568) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img


# ── Image Captioner (selectable: Qwen2-VL / BLIP / none) ─────────────

class ImageCaptioner:
    """Image captioner with manual selection.

    Supports: qwen2vl (GPU), blip (CPU), none (disabled).
    Default: auto (tries qwen2vl GPU, falls back to blip).
    """

    VRAM_NEEDED_GB = 5.0

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._backend = ""  # "qwen2vl", "blip", or ""
        self._selected = "auto"  # user selection

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def selected(self) -> str:
        return self._selected

    def select(self, choice: str) -> dict:
        """Switch captioner. Returns status dict."""
        if choice == self._selected and self._backend:
            return {"backend": self._backend, "status": "already_loaded"}

        # Unload current
        self._unload()
        self._selected = choice

        if choice == "none":
            self._backend = ""
            return {"backend": "", "status": "disabled"}

        if choice == "qwen2vl":
            if self._try_qwen2vl_gpu():
                return {"backend": "qwen2vl", "status": "loaded"}
            return {"backend": "", "status": "failed"}

        if choice == "blip":
            if self._try_blip():
                return {"backend": "blip", "status": "loaded"}
            return {"backend": "", "status": "failed"}

        if choice == "auto":
            free = vram_free_gb()
            if free >= self.VRAM_NEEDED_GB:
                import shutil
                disk = shutil.disk_usage("/")
                if disk.free / 1e9 >= 6.0 and self._try_qwen2vl_gpu():
                    return {"backend": "qwen2vl", "status": "loaded"}
            if self._try_blip():
                return {"backend": "blip", "status": "loaded"}
            return {"backend": "", "status": "failed"}

        return {"backend": "", "status": "unknown_choice"}

    def ensure_loaded(self) -> bool:
        """Lazy-load on first use if not manually selected."""
        if self._backend:
            return True
        if self._selected == "none":
            return False
        result = self.select(self._selected)
        return result["status"] == "loaded"

    def _unload(self) -> None:
        """Fully unload captioner model from memory."""
        if self._model is not None:
            try:
                self._model.to("meta")
            except Exception:
                pass
            del self._model
        if self._processor is not None:
            del self._processor
        self._model = None
        self._processor = None
        self._backend = ""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Captioner unloaded")

    def _try_qwen2vl_gpu(self) -> bool:
        try:
            import shutil
            disk = shutil.disk_usage("/")
            if disk.free / 1e9 < 6.0:
                log.info("Qwen2-VL skipped: only %.1f GB disk free", disk.free / 1e9)
                return False

            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from rune.config import CAPTIONER_OPTIONS
            from rune.settings import get_settings

            trust_remote = get_settings().allow_remote_code
            model_id = CAPTIONER_OPTIONS["qwen2vl"]["model_id"]
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
            self._model.eval()
            self._backend = "qwen2vl"
            log.info("Qwen2-VL captioner loaded (GPU, %.1f GB VRAM free)", vram_free_gb())
            return True
        except Exception as exc:
            log.warning("Qwen2-VL GPU failed: %s", exc)
            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                cache_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct"
                if cache_path.exists():
                    import shutil as sh
                    sh.rmtree(cache_path, ignore_errors=True)
            except Exception:
                pass
            return False

    def _try_blip(self) -> bool:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            blip_name = "Salesforce/blip-image-captioning-base"
            self._processor = BlipProcessor.from_pretrained(blip_name)
            self._model = BlipForConditionalGeneration.from_pretrained(
                blip_name, torch_dtype=torch.float32, use_safetensors=True,
            )
            self._model.eval()
            self._backend = "blip"
            log.info("BLIP captioner loaded (CPU)")
            return True
        except Exception as exc:
            log.warning("BLIP failed: %s", exc)
            self._model = None
            self._processor = None
            return False

    def caption(self, img: Image.Image) -> str:
        if not self.ensure_loaded():
            return ""
        try:
            img = img.convert("RGB")
            if self._backend == "qwen2vl":
                return self._caption_qwen2vl(img)
            elif self._backend == "blip":
                return self._caption_blip(img)
        except Exception as exc:
            log.warning("Image captioning failed: %s", exc)
        return ""

    def _caption_qwen2vl(self, img: Image.Image) -> str:
        w, h = img.size
        if max(w, h) > 512:
            ratio = 512 / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image in detail."},
        ]}]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text_prompt], images=[img], return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        # Read max_new_tokens from settings — the previous hardcoded 150
        # truncated rich photos mid-sentence. Defaults to 256.
        from rune.settings import get_settings
        max_new_tokens = get_settings().caption_max_tokens
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self._processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)[0].strip()

    def _caption_blip(self, img: Image.Image) -> str:
        inputs = self._processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=120)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()

    def caption_multiple(self, images: list[Image.Image]) -> list[str]:
        return [self.caption(img) for img in images]

    # ── V5.7.0 — Vision active (zoom cognitif) ──────────────────────────
    #
    # caption_focused() prend un prompt custom au lieu du prompt par défaut
    # "Describe this image in detail.". Permet de cibler une zone précise
    # ou de poser une question spécifique sur l'image.
    #
    # Implémentation simple : on n'utilise PAS de crop pixel — les VLM
    # récents (Qwen2VL, Florence-2) font de l'attention spatiale via le
    # prompt. Si plus tard ça ne suffit pas, on ajoutera un GroundingDINO
    # pour crop réel.

    def caption_focused(
        self, img: Image.Image, focus_prompt: str, max_tokens: int | None = None,
    ) -> str:
        """Caption ciblé avec un prompt custom.

        Args:
            img: l'image à analyser
            focus_prompt: prompt VLM custom (remplace "Describe this image
                in detail."). Doit être en anglais pour les modèles type
                Qwen2VL qui sont meilleurs en anglais pour l'instruction.
            max_tokens: budget tokens optionnel. Si None, utilise la
                config caption_max_tokens par défaut.

        Returns:
            Texte de la réponse du VLM, ou chaîne vide si échec.
        """
        if not self.ensure_loaded():
            return ""
        try:
            img = img.convert("RGB")
            if self._backend == "qwen2vl":
                return self._caption_qwen2vl_custom(img, focus_prompt, max_tokens)
            elif self._backend == "blip":
                # BLIP ne supporte pas vraiment les prompts custom (il fait
                # du captioning fixe). On fait un best effort en passant
                # le prompt comme contexte conditional.
                return self._caption_blip_custom(img, focus_prompt)
        except Exception as exc:
            log.warning("Focused captioning failed: %s", exc)
        return ""

    def _caption_qwen2vl_custom(
        self, img: Image.Image, focus_prompt: str, max_tokens: int | None,
    ) -> str:
        """Variante Qwen2VL avec prompt custom (vision active)."""
        w, h = img.size
        if max(w, h) > 512:
            ratio = 512 / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": focus_prompt},
        ]}]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text_prompt], images=[img], return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        if max_tokens is None:
            from rune.settings import get_settings
            max_tokens = get_settings().caption_max_tokens

        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self._processor.batch_decode(
            output[:, prompt_len:], skip_special_tokens=True,
        )[0].strip()

    def _caption_blip_custom(self, img: Image.Image, focus_prompt: str) -> str:
        """BLIP ne supporte pas vraiment prompt-conditioning robuste.

        On utilise quand même le focus_prompt comme texte d'amorce
        conditionnel — pas idéal mais mieux que rien. Si on veut une vraie
        vision active, c'est Qwen2VL qu'il faut utiliser comme backend.
        """
        try:
            inputs = self._processor(
                images=img, text=focus_prompt[:120], return_tensors="pt",
            )
        except Exception:
            inputs = self._processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=120)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()
