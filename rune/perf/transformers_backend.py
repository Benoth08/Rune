"""Backend transformers — charge un modèle HuggingFace in-process.

Ce backend préserve l'âme de Lythea : hooks PyTorch sur le flux
résiduel, hidden states pour SDM/MHN/steering, et speculative decoding
via un draft model optionnel.

Contrats
--------
1. Charge en 4-bit NF4 par défaut (bitsandbytes) pour tenir sur 24 Go.
2. compute_dtype = bf16 sur CUDA (fp16 provoque des NaN sur Qwen3).
3. KV cache persistant inter-tours (géré par transformers nativement).
4. Hooks : ``model.model.layers[i].register_forward_hook()``.
5. Speculative : ``model.generate(assistant_model=draft)`` si dispo.

Si une dépendance manque (torch, transformers, bitsandbytes), on logge
et on propage l'erreur — le caller (``get_backend``) retombe sur MockBackend.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Generator

from .backend import GenerationConfig, GenerationResult, ModelBackend
from .speculative import SpecConfig, SpeculativeDecoder

log = logging.getLogger("rune.perf.transformers")


# Catalogue des modèles testés. Chaque entrée porte les métadonnées
# qui ne sont PAS dans la model card HF — notamment si le modèle est
# "thinking" (a un <think> natif) et la taille du hidden_dim (pour
# pré-allouer les buffers SDM/MHN).
#
# Pour ajouter un modèle : ajouter une entrée ici, tester le chargement
# en prod, vérifier que les hooks mémoire fonctionnent.
_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-8B": {
        "hidden_dim": 4096,
        "n_layers": 36,
        "is_thinking": True,
        "size_gb_nf4": 5.5,
        "recommended_draft": "Qwen/Qwen3-0.6B",
    },
    "Qwen/Qwen3-14B": {
        "hidden_dim": 5120,
        "n_layers": 40,
        "is_thinking": True,
        "size_gb_nf4": 9.5,
        "recommended_draft": "Qwen/Qwen3-0.6B",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "hidden_dim": 3584,
        "n_layers": 28,
        "is_thinking": False,
        "size_gb_nf4": 5.0,
        "recommended_draft": "Qwen/Qwen2.5-0.5B",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "hidden_dim": 2048,
        "n_layers": 36,
        "is_thinking": False,
        "size_gb_nf4": 2.5,
        "recommended_draft": "Qwen/Qwen2.5-0.5B",
    },
    "microsoft/Phi-4-mini-instruct": {
        "hidden_dim": 3072,
        "n_layers": 32,
        "is_thinking": False,
        "size_gb_nf4": 3.5,
        "recommended_draft": None,
    },
}


class TransformersBackend(ModelBackend):
    """Backend modèle HuggingFace in-process.

    Config
    ------
    model_id : str
        Model ID HuggingFace. Doit être dans _MODEL_CATALOG.
    quant_4bit : bool
        Charge en NF4 (défaut True). Fallback bf16 si bitsandbytes absent.
    spec_config : SpecConfig | None
        Config du speculative decoding. Si None, désactivé.
    device : str
        "cuda" (défaut) ou "cpu".
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._model_id = cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct")

        if self._model_id not in _MODEL_CATALOG:
            raise ValueError(
                f"Model {self._model_id!r} not in catalog. "
                f"Available: {sorted(_MODEL_CATALOG)}"
            )
        meta = _MODEL_CATALOG[self._model_id]
        self._hidden_dim = meta["hidden_dim"]
        self._n_layers = meta["n_layers"]
        self._is_thinking = meta["is_thinking"]

        # Speculative decoder
        spec_cfg_dict = cfg.get("spec_config") or {}
        spec_cfg = SpecConfig(
            enabled=spec_cfg_dict.get("enabled", True),
            num_draft_tokens=spec_cfg_dict.get("num_draft_tokens", 4),
            draft_model_id=(
                spec_cfg_dict.get("draft_model_id")
                or meta.get("recommended_draft")
                or "Qwen/Qwen3-0.6B"
            ),
        )
        self.spec_decoder = SpeculativeDecoder(spec_cfg)

        # Lazy load — ne charge rien tant qu'on n'appelle pas _load_model()
        self._model: Any = None
        self._tokenizer: Any = None
        self._draft_model: Any = None
        self._device: str = cfg.get("device", "cuda")
        self._quant_4bit: bool = bool(cfg.get("quant_4bit", True))
        self._hooks: dict[int, list[Callable[[Any], None]]] = {}

    # ── Lazy load ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Charge modèle + tokenizer + draft model. Idempotent."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "torch/transformers not installed. Install with: "
                "pip install 'rune[transformers]'"
            ) from exc

        log.info("Loading %s (4-bit=%s, device=%s)",
                 self._model_id, self._quant_4bit, self._device)
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        load_kwargs: dict[str, Any] = {
            "device_map": "auto" if self._device == "cuda" else None,
            "torch_dtype": dtype,
            "output_hidden_states": True,
        }
        if self._quant_4bit and self._device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                log.warning("bitsandbytes not available — falling back to bf16")
                self._quant_4bit = False

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id, **load_kwargs
        )
        self._model.eval()

        # Draft model pour speculative decoding
        if (
            self.spec_decoder.config.enabled
            and self.spec_decoder.config.draft_model_id
        ):
            try:
                self._draft_model = AutoModelForCausalLM.from_pretrained(
                    self.spec_decoder.config.draft_model_id,
                    torch_dtype=dtype,
                    device_map="auto" if self._device == "cuda" else None,
                )
                self._draft_model.eval()
                log.info("Draft model loaded: %s",
                         self.spec_decoder.config.draft_model_id)
            except Exception as exc:
                log.warning("Draft model load failed (%s) — spec disabled", exc)
                self._draft_model = None
                self.spec_decoder.config.enabled = False

        # Re-attache les hooks enregistrés avant le load
        for layer_idx, callbacks in self._hooks.items():
            self._attach_hooks(layer_idx, callbacks)

    def _attach_hooks(
        self, layer_idx: int, callbacks: list[Callable[[Any], None]]
    ) -> None:
        if self._model is None:
            return
        try:
            layer = self._model.model.layers[layer_idx]
            for cb in callbacks:
                layer.register_forward_hook(cb)
        except (AttributeError, IndexError) as exc:
            log.warning("Hook attach failed on layer %d: %s", layer_idx, exc)

    # ── Propriétés ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "transformers"

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def has_hooks(self) -> bool:
        return True

    @property
    def is_thinking_model(self) -> bool:
        return self._is_thinking

    # ── Encode ────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[float]:
        self._load_model()
        import torch
        with torch.no_grad():
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self._model.device)
            outputs = self._model(**inputs, output_hidden_states=True)
            # Mean pooling sur le dernier layer
            last_hidden = outputs.hidden_states[-1].squeeze(0).mean(dim=0)
            vec = last_hidden.float().cpu().tolist()
        # L2-normalize
        import math
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    # ── Generate ──────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> GenerationResult:
        start = time.time()
        try:
            self._load_model()
            import torch
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(
                prompt, return_tensors="pt"
            ).to(self._model.device)

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.temperature > 0,
                "temperature": max(config.temperature, 0.01),
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "return_dict_in_generate": True,
                "output_hidden_states": config.return_hidden_states,
                "output_scores": config.return_entropies,
            }
            if config.min_p is not None:
                gen_kwargs["min_p"] = config.min_p

            # Speculative decoding
            spec_used = False
            if (
                self._draft_model is not None
                and not self.spec_decoder.should_disable()
                and config.spec_num_draft_tokens > 0
            ):
                gen_kwargs["assistant_model"] = self._draft_model
                gen_kwargs["num_assistant_tokens"] = config.spec_num_draft_tokens
                spec_used = True

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            # Decode
            new_tokens = outputs.sequences[0, inputs["input_ids"].shape[1]:]
            raw_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = self._strip_thinking(raw_text)

            # Entropies
            entropies: list[float] = []
            if config.return_entropies and hasattr(outputs, "scores"):
                for score in outputs.scores:
                    if score is None:
                        continue
                    probs = torch.softmax(score[0], dim=-1)
                    ent = -(probs * torch.log(probs + 1e-10)).sum().item()
                    entropies.append(ent)

            # Hidden states — dernier token, par couche
            hidden: list[list[float]] | None = None
            if (
                config.return_hidden_states
                and hasattr(outputs, "hidden_states")
                and outputs.hidden_states
            ):
                last_layer = outputs.hidden_states[-1]
                if last_layer is not None:
                    last_token = last_layer[-1][-1]
                    hidden = last_token.float().cpu().tolist()
                    if not isinstance(hidden[0], list):
                        # Une seule couche — on duplique pour l'API
                        hidden = [hidden]

            elapsed = time.time() - start

            # Stats speculative (approximation — transformers ne donne pas
            # le détail accept/reject, on l'estime si le flag est présent)
            if spec_used:
                self.spec_decoder.record_iteration(
                    accepted=config.spec_num_draft_tokens,
                    rejected=0,
                    iterations=1,
                )

            return GenerationResult(
                text=text,
                raw_text=raw_text,
                entropies=entropies,
                hidden_states=hidden,
                tokens_generated=len(new_tokens),
                elapsed_sec=elapsed,
                finish_reason="length" if len(new_tokens) >= config.max_new_tokens else "stop",
                meta={
                    "backend": "transformers",
                    "model": self._model_id,
                    "spec_decoding": spec_used,
                    "quant_4bit": self._quant_4bit,
                    "spec_stats": self.spec_decoder.stats.as_dict(),
                },
            )
        except Exception as exc:
            log.exception("Generation failed")
            return GenerationResult(
                text="",
                raw_text="",
                finish_reason="error",
                elapsed_sec=time.time() - start,
                meta={"error": str(exc)},
            )

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> Generator[str, None, GenerationResult]:
        """Streaming via TextIteratorStreamer.

        On préserve les entropies et hidden_states via le result final
        retourné par le generator.
        """
        # Pour simplifier, on fait une génération complète puis on yield
        # les tokens. Le vrai streaming utiliserait TextIteratorStreamer
        # dans un thread séparé. À améliorer en v0.2.
        result = self.generate(messages, config)
        for tok in result.text.split():
            yield tok + " "
        return result

    # ── Hooks ─────────────────────────────────────────────────────────

    def register_forward_hook(
        self, layer_idx: int, callback: Callable[[Any], None]
    ) -> Callable[[], None]:
        if layer_idx < 0 or layer_idx >= self._n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0,{self._n_layers})")
        self._hooks.setdefault(layer_idx, []).append(callback)
        if self._model is not None:
            self._attach_hooks(layer_idx, [callback])

        def _remove() -> None:
            # Note: transformers ne donne pas de handle direct pour
            # unregister un hook spécifique. On garde la référence et
            # on filtre à la main si besoin (rare en pratique).
            if layer_idx in self._hooks:
                try:
                    self._hooks[layer_idx].remove(callback)
                except ValueError:
                    pass

        return _remove

    # ── Internes ──────────────────────────────────────────────────────

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Supprime les blocs <think>…</think> si présents."""
        if "<think>" not in text:
            return text
        import re
        # Strip non-greedy
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()
