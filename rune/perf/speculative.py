"""Speculative decoding — draft model + verify.

Implémentation du speculative decoding (Leviathan et al. 2023) :

1. Un petit *draft model* (Qwen3-0.6B, ~1.2 Go) propose ``k`` tokens
   de façon autonome (rapide, ~80 tok/s).
2. Le modèle *principal* (Qwen3-14B) vérifie ces ``k`` tokens en une
   seule passe forward parallèle.
3. On accepte le plus long préfixe où draft et main sont d'accord, on
   régénère le token divergent avec le modèle principal.

Gain typique : 2-3× sur la latence de génération, sans perte de qualité
(le modèle principal reste l'oracle). Toutes les contributions au
texte final viennent du modèle principal → on préserve sa voix, ses
latents, et les hooks mémoire.

Contrat clé pour Rune : on n'appelle jamais le draft model
sans le main model. Le draft ne fait que proposer, jamais décider.
C'est ce qui permet de garder SDM/MHN/steering branchés sur le main.

Références
----------
- Leviathan et al. "Fast Inference from Transformers via Speculative
  Decoding" (ICML 2023)
- Cai et al. "Medusa: Simple LLM Inference Acceleration Framework"
  (ICML 2024)
- HuggingFace docs : https://huggingface.co/docs/transformers/en/llm_tutorial_optimization
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("rune.perf.spec")


@dataclass
class SpecConfig:
    """Config du speculative decoder.

    Attributes
    ----------
    enabled : bool
        Master switch. Si False, le decoder se comporte comme un pass-through.
    num_draft_tokens : int
        Nombre de tokens proposés par le draft à chaque itération (k).
        Valeurs typiques : 2-5. Au-delà, le taux d'acceptation chute.
    max_iterations : int
        Nombre max d'itérations speculative-verify avant d'abandonner
        et de générer un token directement (sécurité anti-boucle).
    draft_model_id : str
        Model ID HuggingFace du draft. Doit partager le tokenizer du
        modèle principal (sinon on retombe sur le main seul).
    acceptance_threshold : float
        Si le taux d'acceptation sur les 100 derniers tokens tombe sous
        ce seuil, on désactive auto le speculative (auto-tuning).
    """
    enabled: bool = True
    num_draft_tokens: int = 4
    max_iterations: int = 16
    draft_model_id: str = "Qwen/Qwen3-0.6B"
    acceptance_threshold: float = 0.3


@dataclass
class SpecStats:
    """Statistiques d'exécution du speculative decoder."""
    total_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    iterations: int = 0
    auto_disabled: bool = False

    @property
    def acceptance_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens

    def as_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "iterations": self.iterations,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "auto_disabled": self.auto_disabled,
        }


class SpeculativeDecoder:
    """Orchestre le speculative decoding entre un draft et un main model.

    Note : cette classe ne fait pas elle-même la génération. Elle
    encapsule la logique d'orchestration (compter les acceptations,
    décider quand désactiver, exposer les stats). Les appels PyTorch
    concrets sont dans :mod:`rune.perf.transformers_backend`,
    qui sait comment brancher ``model.generate(assistant_model=draft)``.

    Usage :

        decoder = SpeculativeDecoder(SpecConfig())
        # Le backend appelle decoder.record_iteration(accepted, rejected)
        # après chaque speculative step. Au bout d'un moment, si le taux
        # d'acceptation est trop bas, ``decoder.should_disable()`` renvoie
        # True et le backend repasse en mode normal.
    """

    def __init__(self, config: SpecConfig) -> None:
        self.config = config
        self.stats = SpecStats()
        self._recent_acceptance: list[bool] = []  # sliding window 100

    def record_iteration(
        self, accepted: int, rejected: int, iterations: int = 1
    ) -> None:
        """Enregistre une itération speculative-verify.

        ``accepted`` et ``rejected`` sont les comptes de tokens acceptés
        vs rejetés lors de cette itération.
        """
        if not self.config.enabled or self.stats.auto_disabled:
            return
        self.stats.total_tokens += accepted + rejected
        self.stats.accepted_tokens += accepted
        self.stats.rejected_tokens += rejected
        self.stats.iterations += iterations
        # Mise à jour sliding window
        for _ in range(accepted):
            self._recent_acceptance.append(True)
        for _ in range(rejected):
            self._recent_acceptance.append(False)
        if len(self._recent_acceptance) > 100:
            self._recent_acceptance = self._recent_acceptance[-100:]

        # Auto-disable check
        if (
            len(self._recent_acceptance) >= 50
            and self.recent_acceptance_rate() < self.config.acceptance_threshold
        ):
            log.info(
                "Speculative decoding auto-disabled (acceptance %.1f%% < %.1f%%)",
                self.recent_acceptance_rate() * 100,
                self.config.acceptance_threshold * 100,
            )
            self.stats.auto_disabled = True

    def recent_acceptance_rate(self) -> float:
        if not self._recent_acceptance:
            return 0.0
        return sum(1 for x in self._recent_acceptance if x) / len(
            self._recent_acceptance
        )

    def should_disable(self) -> bool:
        """True si le speculative doit être désactivé (auto-tuning)."""
        return self.stats.auto_disabled

    def reset(self) -> None:
        """Remet à zéro les stats (utile entre deux sessions)."""
        self.stats = SpecStats()
        self._recent_acceptance.clear()

    def as_dict(self) -> dict:
        return {
            "config": {
                "enabled": self.config.enabled,
                "num_draft_tokens": self.config.num_draft_tokens,
                "draft_model_id": self.config.draft_model_id,
            },
            "stats": self.stats.as_dict(),
            "recent_acceptance_rate": round(self.recent_acceptance_rate(), 3),
        }
