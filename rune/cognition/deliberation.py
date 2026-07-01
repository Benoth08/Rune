"""Raisonnement délibératif multi-angles — pour modèles non-thinking.

Les modèles thinking (Qwen3-4B-Thinking…) décomposent, confrontent et
synthétisent dans leur ``<think>`` natif. Les modèles *Instruct* ne le
font pas spontanément : ils suivent un seul fil de pensée linéaire,
souvent superficiel. Ce module donne aux modèles non-thinking une
structure de délibération qui imite — modestement — ce qu'un gros
modèle fait implicitement.

Principe
--------
Au lieu d'un seul appel "réfléchis bien", on enchaîne 4 étapes courtes
et ciblées, chacune avec son propre prompt :

1. **Décomposer** — quels *angles d'attaque* pour cette question ?
   Les angles sont choisis dynamiquement (pas une liste figée), ce qui
   évite d'imposer "sceptique/comparatif" à une question qui ne s'y
   prête pas.
2. **Explorer** — un appel par angle. Chaque appel produit un
   raisonnement *sous cet angle précis*. C'est la "largeur" : on force
   le modèle à changer de posture, ce qu'il ne fait jamais seul.
3. **Critiquer (croisé)** — les angles se confrontent *mutuellement*.
   On ne demande PAS à chaque angle de s'auto-critiquer : un petit
   modèle juge mal son propre raisonnement (biais de confirmation).
   En revanche il sait comparer plusieurs raisonnements posés côte à
   côte — c'est une tâche descriptive, pas introspective.
4. **Synthétiser** — réflexion finale consolidée, à partir des angles
   ET de la critique croisée.

Le résultat est un texte de raisonnement que l'orchestrateur injecte
avant la génération de la réponse finale, exactement comme le faisait
l'ancien ``ReasoningGenerator`` mono-passe.

Profondeur (``depth``)
----------------------
- ``0`` — aucune délibération (questions triviales). Retourne ``""``.
- ``2`` — 1 angle + synthèse (questions simples).
- ``3`` — 2 angles + critique croisée + synthèse (questions moyennes).
- ``4`` — 3 angles + critique croisée + synthèse (questions complexes).

En V1, la profondeur est passée explicitement par l'appelant. Le
*routeur de complexité* auto-calibrant qui choisira ``depth``
automatiquement viendra en V2, une fois la qualité validée in vivo.

Garde-fous
----------
- Garde-fou trivial intégré : ``is_trivial_message`` réutilise la
  détection chitchat de ``planning.py`` — un "bonjour" ne déclenche
  jamais la moindre étape.
- Chaque étape est encapsulée dans un ``try/except`` : si une étape
  échoue, on continue avec ce qu'on a (jamais de crash).
- ``max_total_seconds`` borne le temps total : si la délibération
  traîne, on s'arrête et on rend le raisonnement partiel.
- Aucune dépendance nouvelle : uniquement des appels au modèle déjà
  chargé.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

log = logging.getLogger("lythea.cognition.deliberation")


# ── Garde-fou trivial ──────────────────────────────────────────────────
# On réutilise la détection chitchat de planning.py plutôt que de
# redéfinir une regex concurrente qui dériverait avec le temps.
try:
    from rune.cognition.planning import _CHITCHAT_RE, _CHITCHAT_MAX_WORDS
except Exception:  # pragma: no cover — fallback si l'import bouge
    _CHITCHAT_RE = re.compile(
        r"^(?:salut|bonjour|coucou|hello|hi|hey|yo|bonsoir|merci|"
        r"thanks|ciao|bye|ok|d'accord|oui|non|ça va)\b",
        re.IGNORECASE,
    )
    _CHITCHAT_MAX_WORDS = 6


def is_trivial_message(message: str) -> bool:
    """Vrai si le message est une salutation / acquiescement trivial.

    Un message trivial ne mérite aucune délibération — répondre
    directement. Critère : court ET commençant par un marqueur de
    chitchat (même logique que l'anti-over-planning de planning.py).
    """
    if not message:
        return True
    normalized = message.strip().lower()
    if not normalized:
        return True
    word_count = len(normalized.split())
    return word_count <= _CHITCHAT_MAX_WORDS and bool(
        _CHITCHAT_RE.match(normalized)
    )


# ── Budgets par étape ──────────────────────────────────────────────────
# Volontairement modestes : chaque étape fait UNE chose précise, pas un
# essai. Garder les étapes courtes limite aussi le temps total.
_DECOMPOSE_MAX_TOKENS = 200
_EXPLORE_MAX_TOKENS = 320
_CRITIQUE_MAX_TOKENS = 300
_SYNTHESIZE_MAX_TOKENS = 400

_STEP_TEMPERATURE = 0.3          # rigueur > créativité pour le raisonnement
_SYNTHESIS_TEMPERATURE = 0.4     # un peu plus de liberté pour consolider

# Bornes de sécurité.
_MAX_ANGLES = 3                  # jamais plus de 3 angles, même en depth=4
_REASONING_TEXT_MAX_CHARS = 2400 # clip final du texte injecté
_DEFAULT_MAX_TOTAL_SECONDS = 90  # au-delà, on rend le partiel

# Angles de secours si la décomposition échoue ou renvoie du vide.
# Choisis pour être génériques mais complémentaires.
_FALLBACK_ANGLES = (
    "analyse directe : les faits et la logique principale",
    "regard critique : limites, exceptions, ce qui pourrait être faux",
    "mise en perspective : alternatives et points de comparaison",
)


class DeliberativeReasoner:
    """Raisonnement délibératif multi-étapes pour modèles non-thinking.

    Porte une référence au modèle et au KG. Ne détient aucun état entre
    deux appels — ``deliberate`` est ré-entrant.
    """

    def __init__(
        self,
        model: Any,
        kg: Any | None = None,
        max_total_seconds: float = _DEFAULT_MAX_TOTAL_SECONDS,
        min_step_margin: float = 3.0,
    ) -> None:
        self.model = model
        self.kg = kg
        self.max_total_seconds = max_total_seconds
        # Marge minimale estimée pour tenter une étape de plus. En
        # dessous, on s'arrête et on rend ce qu'on a — inutile de
        # lancer un appel qui sera de toute façon tronqué. Calibré
        # pour des appels LLM réels de 2-5 s ; configurable surtout
        # pour les tests (qui utilisent des appels factices rapides).
        self.min_step_margin = min_step_margin

    # ── API publique ───────────────────────────────────────────────────

    def deliberate(self, message: str, depth: int = 4) -> str:
        """Produit un texte de raisonnement délibératif.

        Parameters
        ----------
        message : str
            La question / le message de l'utilisateur.
        depth : int
            Profondeur de délibération (0, 2, 3 ou 4). Voir le
            docstring du module. ``depth=0`` ou message trivial →
            retourne ``""``.

        Returns
        -------
        str
            Le raisonnement consolidé (texte brut, sans balise), ou
            ``""`` si pas de délibération.
        """
        # Garde-fous d'entrée : trivial ou depth nul → rien.
        if depth <= 0 or is_trivial_message(message):
            return ""

        start = time.monotonic()

        def time_left() -> float:
            return self.max_total_seconds - (time.monotonic() - start)

        # Combien d'angles selon la profondeur.
        n_angles = {2: 1, 3: 2, 4: 3}.get(depth, 3)

        # Marge minimale estimée pour tenter une étape de plus. En
        # dessous, on s'arrête et on rend ce qu'on a — inutile de
        # lancer un appel qui sera de toute façon tronqué.
        _MIN_STEP_MARGIN = 3.0

        # ── Étape 1 : décomposer ──────────────────────────────────────
        angles = self._decompose(message, n_angles)
        if not angles:
            angles = list(_FALLBACK_ANGLES[:n_angles])

        # ── Étape 2 : explorer chaque angle ───────────────────────────
        views: list[tuple[str, str]] = []  # (angle, raisonnement)
        for angle in angles:
            if time_left() < _MIN_STEP_MARGIN:
                log.warning("Deliberation: budget temps épuisé, "
                            "%d/%d angles explorés", len(views), len(angles))
                break
            view = self._explore(message, angle)
            if view:
                views.append((angle, view))

        if not views:
            # Aucun angle exploité — rien à synthétiser.
            return ""

        # ── Étape 3 : critique croisée (depth >= 3 et >= 2 angles) ────
        critique = ""
        if depth >= 3 and len(views) >= 2 and time_left() > _MIN_STEP_MARGIN:
            critique = self._cross_critique(message, views)

        # ── Étape 4 : synthèse ────────────────────────────────────────
        if time_left() > _MIN_STEP_MARGIN:
            final = self._synthesize(message, views, critique)
            if final:
                return final[:_REASONING_TEXT_MAX_CHARS]

        # Fallback : pas eu le temps de synthétiser → on rend les
        # raisonnements bruts concaténés, mieux que rien.
        joined = "\n\n".join(
            f"[{angle}]\n{view}" for angle, view in views
        )
        return joined[:_REASONING_TEXT_MAX_CHARS]

    # ── Étapes internes ────────────────────────────────────────────────

    def _decompose(self, message: str, n_angles: int) -> list[str]:
        """Étape 1 — choisir N angles d'attaque pour la question.

        Retourne une liste de descriptions d'angles (texte court). Liste
        vide si l'étape échoue → l'appelant retombe sur les angles de
        secours.
        """
        n_angles = max(1, min(n_angles, _MAX_ANGLES))
        kg_hint = self._kg_hint()

        if n_angles == 1:
            instruction = (
                "Identifie LE meilleur angle d'analyse pour cette "
                "question. Réponds en UNE seule ligne décrivant cet "
                "angle, rien d'autre."
            )
        else:
            instruction = (
                f"Identifie {n_angles} angles d'analyse COMPLÉMENTAIRES "
                "et DIFFÉRENTS pour cette question — des façons de "
                "l'aborder qui s'éclairent mutuellement. "
                f"Réponds avec exactement {n_angles} lignes, une par "
                "angle, format : « - description courte de l'angle ». "
                "Pas d'introduction, pas de conclusion."
            )

        system = (
            "Tu es Rune, une IA cognitive. Tu prépares un raisonnement "
            "structuré. " + instruction + kg_hint
        )
        raw = self._call(
            system, message,
            max_new_tokens=_DECOMPOSE_MAX_TOKENS,
            temperature=_STEP_TEMPERATURE,
        )
        if not raw:
            return []

        # Parser les lignes « - ... » ou « 1. ... » ou lignes nues.
        angles: list[str] = []
        for line in raw.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            # Retirer puces et numérotation de tête.
            cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
            cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
            cleaned = cleaned.strip()
            if len(cleaned) >= 8:  # filtre le bruit
                angles.append(cleaned)
            if len(angles) >= n_angles:
                break

        return angles[:n_angles]

    def _explore(self, message: str, angle: str) -> str:
        """Étape 2 — raisonner sur la question SOUS un angle donné.

        Retourne le raisonnement (texte), ou ``""`` si échec.
        """
        kg_hint = self._kg_hint()
        system = (
            "Tu es Rune, une IA cognitive. Analyse la question de "
            "l'utilisateur EXCLUSIVEMENT sous l'angle suivant :\n"
            f"  → {angle}\n"
            "Reste concentré sur cet angle précis. Sois rigoureux, "
            "distingue ce que tu sais de ce que tu supposes. "
            "Réponds uniquement avec ton analyse sous cet angle, "
            "ni introduction ni réponse finale." + kg_hint
        )
        return self._call(
            system, message,
            max_new_tokens=_EXPLORE_MAX_TOKENS,
            temperature=_STEP_TEMPERATURE,
        )

    def _cross_critique(
        self, message: str, views: list[tuple[str, str]],
    ) -> str:
        """Étape 3 — confrontation mutuelle des angles.

        On donne au modèle TOUS les raisonnements côte à côte et on lui
        demande de les comparer : convergences, contradictions, lequel
        tient le mieux. Tâche descriptive (le modèle compare des objets
        externes) et non introspective (il ne juge pas « son » propre
        raisonnement) — c'est ce qui la rend accessible à un petit
        modèle.
        """
        blocks = "\n\n".join(
            f"RAISONNEMENT {i + 1} — angle « {angle} » :\n{view}"
            for i, (angle, view) in enumerate(views)
        )
        system = (
            "Tu es Rune, une IA cognitive. Plusieurs raisonnements "
            "ont été produits sous des angles différents pour la même "
            "question. Compare-les :\n"
            "- Où convergent-ils ? (points solides, car confirmés)\n"
            "- Où se contredisent-ils ? (points à trancher)\n"
            "- Lequel paraît le plus fiable, et pourquoi ?\n"
            "Sois bref et factuel. Ne rédige pas la réponse finale, "
            "juste cette comparaison."
        )
        user = (
            f"Question initiale : {message}\n\n"
            f"{blocks}\n\n"
            "Compare ces raisonnements selon les 3 points ci-dessus."
        )
        return self._call(
            system, user,
            max_new_tokens=_CRITIQUE_MAX_TOKENS,
            temperature=_STEP_TEMPERATURE,
        )

    def _synthesize(
        self,
        message: str,
        views: list[tuple[str, str]],
        critique: str,
    ) -> str:
        """Étape 4 — réflexion finale consolidée.

        Combine les angles explorés et (si présente) la critique
        croisée en un raisonnement unique, cohérent, qui servira de
        contexte à la génération de la réponse.
        """
        blocks = "\n\n".join(
            f"Angle « {angle} » :\n{view}"
            for angle, view in views
        )
        critique_block = (
            f"\n\nConfrontation des angles :\n{critique}"
            if critique else ""
        )
        system = (
            "Tu es Rune, une IA cognitive. À partir des analyses "
            "multi-angles ci-dessous (et de leur confrontation si "
            "présente), produis UN raisonnement final consolidé : "
            "ce que tu retiens, ce qui est solide, ce qui reste "
            "incertain. Ce raisonnement préparera ta réponse — ne "
            "rédige pas encore la réponse elle-même, juste la "
            "réflexion consolidée. Sois structuré et concis."
        )
        user = (
            f"Question : {message}\n\n"
            f"{blocks}{critique_block}\n\n"
            "Consolide tout cela en un raisonnement final."
        )
        return self._call(
            system, user,
            max_new_tokens=_SYNTHESIZE_MAX_TOKENS,
            temperature=_SYNTHESIS_TEMPERATURE,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _call(
        self,
        system: str,
        user: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Un appel LLM encapsulé. Retourne ``""`` en cas d'échec.

        Réplique le pattern de ReasoningGenerator : on rend le prompt
        via le chat template du tokenizer si disponible, sinon on
        retombe sur le texte brut.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            tokenizer = getattr(self.model, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                rendered = user
            out = self.model.generate(
                rendered,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            return (out or "").strip()
        except Exception as exc:
            log.warning("Deliberation step failed: %s", exc)
            return ""

    def _kg_hint(self) -> str:
        """Petit rappel des entités KG connues (l'interlocuteur).

        Identique en esprit au ``_kg_facts_hint`` de ReasoningGenerator :
        sans ce rappel, le modèle « invente » un interlocuteur générique.
        """
        if not self.kg or not getattr(self.kg, "entities", None):
            return ""
        facts: list[str] = []
        for ent in self.kg.entities.values():
            value = getattr(ent, "value", None)
            etype = getattr(ent, "type", None)
            if value:
                facts.append(f"{value} ({etype})" if etype else str(value))
        if not facts:
            return ""
        return (
            "\n\nContexte mémoire (sur l'utilisateur, PAS sur toi) : "
            + ", ".join(facts[:10])
        )
