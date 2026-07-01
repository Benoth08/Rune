"""SurpriseMeter — surprise composite biomimétique (simplifiée).

Héritage Lythea
---------------
Lythea v5 a une SurprisePhase complète (335 lignes) avec 4 signaux
(structural, episodic, predictive, chroma discount). Ici on fournit
une version simplifiée mais compatible — les mêmes champs sont exposés
pour que les hooks existants (storage, consolidation) continuent à
marcher.

Pour la version complète, voir ``lythea/cognition/surprise.py``.
On peut brancher cette dernière à la place via ``set_implementation``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger("rune.cognition.surprise")


@dataclass
class SurpriseSignals:
    """Signaux de surprise composite — sortie de SurpriseMeter.

    Tous les champs sont dans [0, 1]. global est la combinaison
    pondérée qui pilote le storage (force d'écriture SDM) et la
    consolidation (ripple trigger).
    """
    structural: float = 0.0   # entropie moyenne des tokens utilisateur
    episodic: float = 0.0     # distance à l'épisode MHN le plus proche
    predictive: float = 0.0   # écart entre prédiction SDM et observation
    chroma_discount: float = 0.0  # similarité Chroma (réduit global)
    global_surprise: float = 0.0
    doubt_index: float = 0.0  # doute sur la RÉPONSE (post-generation)
    epistemic_label: str = "fait"  # fait | intuition | hypothese

    def as_dict(self) -> dict:
        return {
            "structural": round(self.structural, 3),
            "episodic": round(self.episodic, 3),
            "predictive": round(self.predictive, 3),
            "chroma_discount": round(self.chroma_discount, 3),
            "global_surprise": round(self.global_surprise, 3),
            "doubt_index": round(self.doubt_index, 3),
            "epistemic_label": self.epistemic_label,
        }


# Poids des composantes (cf. lythea.config — SURPRISE_W_*)
DEFAULT_WEIGHTS = {
    "structural": 0.4,
    "episodic": 0.3,
    "predictive": 0.3,
}

# Seuils épistémiques (cf. lythea/cognition/surprise.py)
DOUBT_FACT_MAX = 0.3
DOUBT_INTUITION_MAX = 0.8


class SurpriseMeter:
    """Calcul de surprise composite.

    Parameters
    ----------
    weights : dict[str, float] | None
        Poids des 3 composantes additives. Default = DEFAULT_WEIGHTS.
    entropy_threshold : float
        Seuil de normalisation pour structural (cf. Lythea).
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        entropy_threshold: float = 0.2,
    ) -> None:
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.entropy_threshold = max(0.01, float(entropy_threshold))

    def compute_input_surprise(
        self,
        user_entropies: list[float] | None = None,
        mhn_match_score: float | None = None,
        sdm_prediction_error: float | None = None,
        chroma_similarity: float | None = None,
    ) -> SurpriseSignals:
        """Calcule la surprise sur le message utilisateur (pre-generation).

        Toutes les entrées sont optionnelles — si une valeur est None,
        la composante correspondante est neutralisée (valeur neutre).
        """
        structural = self._structural_surprise(user_entropies)
        episodic = self._episodic_surprise(mhn_match_score)
        predictive = self._predictive_surprise(sdm_prediction_error)
        discount = self._chroma_discount(chroma_similarity)

        composite = (
            self.weights["structural"] * structural
            + self.weights["episodic"] * episodic
            + self.weights["predictive"] * predictive
        )
        # Le discount Chroma réduit multiplicativement (mais jamais à 0)
        global_s = composite * (1.0 - 0.95 * discount)
        global_s = max(0.0, min(1.0, global_s))

        return SurpriseSignals(
            structural=structural,
            episodic=episodic,
            predictive=predictive,
            chroma_discount=discount,
            global_surprise=global_s,
        )

    def compute_output_doubt(
        self,
        out_entropies: list[float] | None,
        kg_hits: int = 0,
        web_used: bool = False,
        rag_coverage: float = 0.0,
    ) -> tuple[float, str]:
        """Calcule le doute sur la réponse générée (post-generation).

        Étend le calcul Lythea avec des signaux supplémentaires (BACKLOG
        V4.4 — calibration multi-signaux) :

        - doubt_token = mean(out_entropies) / threshold (Lythea original)
        - kg_orphan = +0.3 si 0 hits KG et question factuelle
        - rag_coverage = -0.2 si coverage élevé (RAG riche = moins de doute)
        - web_used = -0.1 si web utilisé (sources fraîches)

        Retourne (doubt_index, epistemic_label).
        """
        if not out_entropies:
            token_doubt = 0.5
        else:
            mean_ent = sum(out_entropies) / len(out_entropies)
            token_doubt = mean_ent / max(self.entropy_threshold, 0.1)
            token_doubt = max(0.0, min(1.5, token_doubt))

        # Ajustements multi-signaux
        kg_penalty = 0.3 if kg_hits == 0 else 0.0
        rag_bonus = -0.2 * max(0.0, min(1.0, rag_coverage))
        web_bonus = -0.1 if web_used else 0.0

        doubt = max(0.0, min(1.0, token_doubt + kg_penalty + rag_bonus + web_bonus))

        if doubt < DOUBT_FACT_MAX:
            label = "fait"
        elif doubt < DOUBT_INTUITION_MAX:
            label = "intuition"
        else:
            label = "hypothese"

        return doubt, label

    # ── Internes ──────────────────────────────────────────────────────

    def _structural_surprise(self, entropies: list[float] | None) -> float:
        if not entropies:
            return 0.5  # neutre
        mean_e = sum(entropies) / len(entropies)
        # Normalize par threshold (cf. Lythea)
        normalized = mean_e / max(self.entropy_threshold, 0.1)
        return max(0.0, min(1.0, normalized))

    def _episodic_surprise(self, match_score: float | None) -> float:
        """match_score = similarité avec l'épisode MHN le plus proche.

        Haute similarité → faible surprise (déjà vu).
        Basse similarité → forte surprise (nouveau).
        """
        if match_score is None:
            return 0.5
        return max(0.0, min(1.0, 1.0 - match_score))

    def _predictive_surprise(self, prediction_error: float | None) -> float:
        """prediction_error = distance cosine entre prédit et observé."""
        if prediction_error is None:
            return 0.5
        return max(0.0, min(1.0, prediction_error))

    def _chroma_discount(self, similarity: float | None) -> float:
        """similarity = similarité Chroma la plus haute.

        Haute similarité → discount élevé (on connaît déjà).
        """
        if similarity is None:
            return 0.0
        return max(0.0, min(0.95, similarity))
