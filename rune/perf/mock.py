"""MockBackend — backend de dev/test sans GPU.

Répond avec des templates simples mais expose toute l'interface
:class:`ModelBackend`. Les hooks sont des no-ops qui n'enregistrent
rien, mais ils ne plantent pas. Les embeddings sont hash-déterministes
pour que les tests soient reproductibles.

Usage typique :

    backend = MockBackend({"responses": ["Bonjour !"]})
    result = backend.generate(messages=[{"role": "user", "content": "salut"}])

Le backend peut aussi être configuré pour simuler un modèle "thinking"
(avec <think>…</think> dans la réponse) afin de tester le stripping.
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from typing import Any, Callable, Generator

from .backend import GenerationConfig, GenerationResult, ModelBackend

log = logging.getLogger("rune.perf.mock")


# Réponses par défaut — utilisées si la config n'en fournit pas.
_DEFAULT_RESPONSES = [
    "C'est une question intéressante. Voici ce que je peux dire : "
    "le sujet mérite qu'on l'examine sous plusieurs angles, en tenant "
    "compte du contexte et des contraintes que vous évoquez.",
    "Bien reçu. Je vais répondre en m'appuyant sur ce que je sais. "
    "N'hésitez pas à préciser si vous souhaitez aller plus loin.",
    "D'accord. Voici mon analyse : il y a plusieurs dimensions à "
    "considérer, et je vais les présenter de façon structurée.",
]


class MockBackend(ModelBackend):
    """Backend modèle simulé pour dev/test.

    Config
    ------
    responses : list[str]
        Liste de réponses à servir en cycle. Si la liste est épuisée,
        on boucle. Si vide, utilise ``_DEFAULT_RESPONSES``.
    hidden_dim : int
        Dimension des vecteurs simulés (défaut 64 — assez pour les
        tests SDM/MHN sans exploser la mémoire).
    n_layers : int
        Nombre de couches simulées (défaut 4).
    is_thinking : bool
        Si True, les réponses incluent un bloc <think> simulé.
    seed : int
        Graine pour la génération pseudo-aléatoire (embeddings,
        entropies). Défaut 42 — reproductible.
    latency_ms : float
        Latence simulée par token (pour tester le streaming).
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._responses: list[str] = list(
            cfg.get("responses") or _DEFAULT_RESPONSES
        )
        self._hidden_dim: int = int(cfg.get("hidden_dim", 64))
        self._n_layers: int = int(cfg.get("n_layers", 4))
        self._is_thinking: bool = bool(cfg.get("is_thinking", False))
        self._latency_ms: float = float(cfg.get("latency_ms", 1.0))
        self._rng = random.Random(int(cfg.get("seed", 42)))
        self._call_idx: int = 0
        self._hooks: dict[int, list[Callable[[Any], None]]] = {}

    # ── Propriétés d'interface ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "mock"

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def has_hooks(self) -> bool:
        return True  # on émule, sans crasher

    @property
    def is_thinking_model(self) -> bool:
        return self._is_thinking

    # ── Encode ────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[float]:
        """Hash-déterministe → vecteur L2-normalisé.

        On utilise SHA256 du texte pour avoir un vecteur stable. Deux
        textes identiques donnent le même embedding, deux textes
        différents donnent des vecteurs quasi-orthogonaux (diffusion
        du hash).
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # On génère hidden_dim bytes en répétant le hash si besoin.
        raw = bytearray()
        counter = 0
        while len(raw) < self._hidden_dim * 4:
            ext = hashlib.sha256(h + counter.to_bytes(4, "big")).digest()
            raw.extend(ext)
            counter += 1
        # Conversion en floats signés normalisés.
        vec = []
        for i in range(self._hidden_dim):
            chunk = raw[i * 4 : i * 4 + 4]
            val = int.from_bytes(chunk, "big", signed=True) / (2**31)
            vec.append(val)
        # L2-normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    # ── Generate ──────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> GenerationResult:
        start = time.time()
        text = self._next_response(messages)
        raw = text
        if self._is_thinking and "<think>" not in text:
            raw = f"<think>Je réfléchis à la requête.</think>{text}"

        # Simule des entropies décroissantes (le modèle "se rassure").
        n_tokens = max(8, len(text.split()))
        entropies = [
            max(0.05, self._rng.gauss(0.4, 0.2))
            for _ in range(n_tokens)
        ]
        # Hidden state du dernier token (couche par couche).
        last_msg = messages[-1]["content"] if messages else ""
        emb = self.encode(last_msg)
        hidden = [emb[:] for _ in range(self._n_layers)]

        # Simule un peu de latence.
        time.sleep(min(0.05, n_tokens * self._latency_ms / 1000))

        return GenerationResult(
            text=text,
            raw_text=raw,
            entropies=entropies,
            hidden_states=hidden,
            tokens_generated=n_tokens,
            elapsed_sec=time.time() - start,
            finish_reason="stop",
            meta={"backend": "mock", "spec_decoding": False},
        )

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> Generator[str, None, GenerationResult]:
        """Yield les tokens un par un, puis retourne le result final."""
        start = time.time()
        text = self._next_response(messages)
        raw = text
        if self._is_thinking and "<think>" not in text:
            raw = f"<think>Je réfléchis.</think>{text}"

        tokens = text.split()
        entropies: list[float] = []
        for tok in tokens:
            yield tok + " "
            entropies.append(max(0.05, self._rng.gauss(0.4, 0.2)))
            time.sleep(self._latency_ms / 1000)

        last_msg = messages[-1]["content"] if messages else ""
        emb = self.encode(last_msg)
        hidden = [emb[:] for _ in range(self._n_layers)]

        return GenerationResult(
            text=text,
            raw_text=raw,
            entropies=entropies,
            hidden_states=hidden,
            tokens_generated=len(tokens),
            elapsed_sec=time.time() - start,
            finish_reason="stop",
            meta={"backend": "mock", "streamed": True},
        )

    # ── Hooks ─────────────────────────────────────────────────────────

    def register_forward_hook(
        self, layer_idx: int, callback: Callable[[Any], None]
    ) -> Callable[[], None]:
        self._hooks.setdefault(layer_idx, []).append(callback)

        def _remove() -> None:
            if layer_idx in self._hooks:
                try:
                    self._hooks[layer_idx].remove(callback)
                except ValueError:
                    pass

        return _remove

    # ── Internes ──────────────────────────────────────────────────────

    def _next_response(self, messages: list[dict[str, str]]) -> str:
        """Cycle dans self._responses, avec petit contexte utilisateur."""
        if not self._responses:
            return _DEFAULT_RESPONSES[0]
        resp = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1
        # Si l'utilisateur dit "merci" ou "ok", on court-circuite.
        last = (messages[-1]["content"] if messages else "").strip().lower()
        if last in {"merci", "ok", "d'accord", "super", "parfait"}:
            return "Avec plaisir !"
        if last.startswith(("bonjour", "salut", "hello", "coucou")):
            return "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
        return resp
