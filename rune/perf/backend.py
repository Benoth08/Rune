"""Backend modèle abstrait — interface minimale pour le cycle cognitif.

Pourquoi une abstraction
-------------------------
Rune doit pouvoir tourner :
- Sans GPU (tests, dev, démo) — :class:`MockBackend`
- Avec un vrai modèle sur GPU — :class:`TransformersBackend`
- (futur) Avec vLLM en mode rapide pur — backend dédié

Toute la cognition est codée contre :class:`ModelBackend`. Le backend
réel est choisi au boot selon la config et la dispo GPU.

Hooks mémoire
-------------
Le contrat clé : tout backend réel doit exposer ``register_forward_hook``
et ``output_hidden_states=True`` pour que la mémoire SDM/MHN et le
steering CAA fonctionnent. Le MockBackend simule ces hooks avec des
vecteurs aléatoires reproductibles.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

# Import différé pour éviter circularité — MockBackend importe backend.py
# pour la classe ModelBackend. On fait l'import ici en lazy dans get_backend.
log = logging.getLogger("rune.perf")


@dataclass(frozen=True)
class GenerationConfig:
    """Paramètres de génération — profil de sampling par modèle.

    Inspiré de ``SamplingProfile`` de Lythea (cf. lythea/config.py).
    Chaque entrée du catalogue porte son profil recommandé, qui est
    appliqué automatiquement au chargement du modèle.
    """
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float | None = 0.9
    top_k: int | None = 20
    min_p: float | None = None
    repetition_penalty: float = 1.05
    # Spécifique au speculative decoding — 0 = désactivé.
    spec_num_draft_tokens: int = 4
    # Active le retour des entropies par token (pour la métacognition).
    return_entropies: bool = True
    # Active le retour des hidden states (pour SDM/MHN/steering).
    return_hidden_states: bool = True


@dataclass
class GenerationResult:
    """Résultat d'une génération — texte + signaux cognitifs.

    Attributes
    ----------
    text : str
        Le texte généré, sans tokens de raisonnement.
    raw_text : str
        Texte complet incluant ``<think>`` si le modèle en produit.
    entropies : list[float]
        Entropie par token généré (en nats). Vide si return_entropies=False.
        Alimente la surprise structurelle et la métacognition.
    hidden_states : list[list[float]] | None
        Hidden state du dernier token par couche (ou None). Alimente
        SDM, MHN et le steering CAA. Taille = (n_layers, hidden_dim).
    tokens_generated : int
        Nombre de tokens générés.
    elapsed_sec : float
        Temps wall-clock de la génération (mesuré par le caller).
    finish_reason : str
        "stop" | "length" | "error".
    meta : dict
        Backend-specific (draft hits, cache hit, etc.).
    """
    text: str = ""
    raw_text: str = ""
    entropies: list[float] = field(default_factory=list)
    hidden_states: list[list[float]] | None = None
    tokens_generated: int = 0
    elapsed_sec: float = 0.0
    finish_reason: str = "stop"
    meta: dict = field(default_factory=dict)


class ModelBackend(ABC):
    """Interface qu'implémentent tous les backends modèle.

    Contrats
    --------
    1. ``generate()`` ne lève jamais — toute erreur est renvoyée dans
       ``GenerationResult(finish_reason="error")``.
    2. ``register_forward_hook()`` est disponible sur les backends réels
       (Mock l'émule avec un no-op). Le cycle cognitif vérifie
       ``has_hooks`` avant d'attacher quoi que ce soit.
    3. ``hidden_dim`` et ``n_layers`` sont stables pendant toute la vie
       du backend.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def has_hooks(self) -> bool:
        """True si register_forward_hook fonctionne réellement."""

    @property
    @abstractmethod
    def is_thinking_model(self) -> bool:
        """True si le modèle a un <think> natif (Qwen3, DeepSeek-R1…)."""

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Retourne l'embedding moyen du texte (pour SDM/MHN/predictive_coding).

        Contrat : vecteur de taille ``hidden_dim``, L2-normalisé.
        """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Génère une réponse à partir d'une liste de messages chat.

        ``messages`` suit le format OpenAI : [{"role": "system", "content": "..."}, ...].
        """

    @abstractmethod
    def register_forward_hook(
        self, layer_idx: int, callback: Callable[[Any], None]
    ) -> Callable[[], None]:
        """Attache un hook sur une couche — retourne une fonction de removal.

        Le callback reçoit le tenseur d'activations de la couche. Le caller
        ne doit pas modifier le tenseur in-place (sauf pour le steering,
        qui le fait délibérément).
        """

    @abstractmethod
    def stream_generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> Generator[str, None, GenerationResult]:
        """Génère en streaming — yield les tokens au fur et à mesure.

        Retourne (via le return du generator) le résultat final avec
        entropies et hidden_states. Permet à l'UI/CLI d'afficher le
        streaming tout en récupérant les signaux cognitifs à la fin.
        """


def get_backend(config: dict | None = None) -> ModelBackend:
    """Factory — choisit le backend selon la config et l'environnement.

    Logique
    -------
    1. Si ``HERMES_LYTHEA_BACKEND=mock`` (ou pas de GPU) → MockBackend
    2. Si ``HERMES_LYTHEA_BACKEND=transformers`` → TransformersBackend
    3. Sinon : auto-détecte CUDA, fallback mock si indispo

    Cette fonction ne lève jamais — en cas d'erreur de chargement, on
    logge et on retombe sur MockBackend pour que le système reste
    utilisable (mode dégradé explicite).
    """
    # Imports en lazy pour éviter la circularité (mock.py et
    # transformers_backend.py importent ce module pour ModelBackend).
    from .mock import MockBackend
    from .transformers_backend import TransformersBackend

    cfg = config or {}
    backend_kind = (
        cfg.get("backend")
        or os.environ.get("HERMES_LYTHEA_BACKEND", "auto")
        or "auto"
    ).lower()

    if backend_kind == "mock":
        log.info("Using MockBackend (forced by config)")
        return MockBackend(cfg)

    if backend_kind == "transformers":
        try:
            return TransformersBackend(cfg)
        except Exception as exc:
            log.warning(
                "TransformersBackend failed to load (%s) — falling back to "
                "MockBackend. Set HERMES_LYTHEA_BACKEND=mock to silence.",
                exc,
            )
            return MockBackend(cfg)

    # Auto-detection
    try:
        import torch  # noqa: F401
        if not torch.cuda.is_available():
            log.info("No CUDA detected — using MockBackend (dev mode)")
            return MockBackend(cfg)
    except ImportError:
        log.info("torch not installed — using MockBackend (dev mode)")
        return MockBackend(cfg)

    try:
        return TransformersBackend(cfg)
    except Exception as exc:
        log.warning(
            "TransformersBackend failed (%s) — falling back to MockBackend",
            exc,
        )
        return MockBackend(cfg)
