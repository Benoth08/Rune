"""Couche performance — backend modèle abstrait + speculative decoding.

Rune garde l'accès aux poids et aux latents (contrairement à
Rune qui utilise des API distantes). On supporte deux backends :

- :class:`MockBackend` — pour tests, démos et dev sans GPU. Répond avec
  des templates simples mais expose la même interface que le backend réel.
- :class:`TransformersBackend` — charge un modèle HuggingFace in-process
  avec bitsandbytes 4-bit, KV cache persistant, hooks forward, et
  speculative decoding via un draft model.

Le speculative decoding (Leviathan et al. 2023) fait proposer 4 tokens
par un petit draft model (Qwen3-0.6B, ~1.2 Go), puis le modèle principal
les vérifie en une seule passe forward. Gain typique 2-3× sur la latence
de génération sans perte de qualité — tout en préservant les hooks
mémoire sur le modèle principal.

Tous les backends implémentent :class:`ModelBackend` — interface minimale
que le cycle cognitif attend. Aucun code en dehors de ce module ne sait
quel backend tourne réellement.
"""
from __future__ import annotations

from .backend import (
    GenerationConfig,
    GenerationResult,
    ModelBackend,
    get_backend,
)
from .mock import MockBackend
from .transformers_backend import TransformersBackend
from .speculative import SpeculativeDecoder

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "ModelBackend",
    "MockBackend",
    "TransformersBackend",
    "SpeculativeDecoder",
    "get_backend",
]
