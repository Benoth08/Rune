"""Raisonnement profond multi-étapes — ``DeepReasoningChain``.

Pourquoi
--------
Un modèle *non-thinking* (Qwen-Instruct, Llama-Instruct…) à qui on
demande « réfléchis étape par étape » fait un effort unique et
souvent superficiel : il lit, il analyse vaguement, il s'arrête.
Les gros modèles de raisonnement, eux, *décomposent* — ils explorent
plusieurs angles, produisent un brouillon, l'attaquent, le corrigent.

``DeepReasoningChain`` reproduit ce comportement par **scaffolding** :
au lieu d'un seul appel « réfléchis bien », on enchaîne plusieurs
appels courts et ciblés, chacun étant une tâche simple que le petit
modèle réussit bien.

Version actuelle — 2 étapes
---------------------------
1. **Explorer** — « Qu'est-ce que tu sais avec certitude ? Qu'est-ce
   qui est incertain ? Quels angles considérer ? » → matière première.
2. **Critiquer** — « Voici ton raisonnement préliminaire. Qu'est-ce
   qui pourrait être faux ou incomplet ? Corrige. » → raisonnement
   consolidé, qui part ensuite vers la génération de la réponse.

Le module est conçu pour grandir : les étapes *décomposer* et
*synthétiser* viendront s'insérer autour de ces deux-là quand la
version 2-étapes aura fait ses preuves in vivo.

Routeur de complexité
---------------------
Lancer 2 appels LLM sur « quelle heure est-il à Tokyo » serait du
gaspillage. ``assess_complexity`` lit les signaux déjà calculés par
le pipeline (surprise, doute, entités, longueur de question) et
renvoie le nombre d'étapes à exécuter : ``0`` (rien — on retombe sur
le reasoning simple existant) ou ``2`` (la chaîne complète).

Périmètre
---------
Pensé pour les modèles **non-thinking** uniquement. Les modèles
thinking (Qwen3-*-Thinking…) produisent déjà ce raisonnement dans
leur ``<think>`` natif — leur superposer une chaîne externe ferait
doublon. L'orchestrateur ne doit donc appeler ``DeepReasoningChain``
que lorsque ``model.is_thinking`` est faux.

Garde-fous
----------
- Activé par le toggle unique ``🧠 Raisonnement`` dans l'UI
  (``reasoning_enabled`` dans l'orchestrateur).
- Chaque étape a un budget de tokens modeste et une température basse.
- Toute étape qui échoue → fallback neutre, jamais d'exception
  propagée : la chaîne dégrade gracieusement vers ce qu'elle a.
- Un budget temps global coupe la chaîne si elle traîne.
- Aucune dépendance nouvelle : uniquement des appels au modèle déjà
  chargé.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

log = logging.getLogger("rune.cognition.deep_reasoning")


# ── Budgets par étape ──────────────────────────────────────────────────
# Les budgets de tokens vivent dans settings.py
# (``reasoning_deep_step_medium_tokens``, ``..._high_tokens``,
# ``reasoning_text_max_chars``) — réglables sans toucher au code, et
# choisis dynamiquement par le routeur de complexité : une question
# "moyenne" reçoit le budget moyen, une question "complexe" le budget
# haut. Une étape tronquée à mi-phrase est inutilisable et pollue
# l'étape suivante — d'où des budgets généreux.
#
# La température reste une constante ici : c'est un paramètre de
# comportement (rigueur du raisonnement), pas un budget.
_STEP_TEMPERATURE: float = 0.3

# Budget temps global de la chaîne. Au-delà, on s'arrête et on
# renvoie ce qu'on a déjà — évite qu'un modèle lent bloque le tour.
_CHAIN_MAX_SECONDS: float = 45.0

# Longueur max du contexte web injecté dans les prompts de
# décomposition / exploration. Le contexte web est déjà numéroté
# [1][2]… par le WebAgent ; on le borne pour qu'il ne noie pas
# l'instruction de raisonnement elle-même.
_WEB_CONTEXT_MAX_CHARS: int = 1800

# ── Seuils du routeur de complexité ────────────────────────────────────
# Une question "mérite" la chaîne profonde si elle est non triviale.
# On combine plusieurs signaux faibles plutôt qu'un seul fort.
_COMPLEXITY_MIN_WORDS: int = 12        # question courte → probablement simple
_COMPLEXITY_SURPRISE_THRESHOLD: float = 0.55
_COMPLEXITY_DOUBT_THRESHOLD: float = 0.40
_COMPLEXITY_MIN_ENTITIES: int = 2      # plusieurs entités → croisement probable

# Marqueurs lexicaux d'une question analytique (FR + EN). Leur
# présence pousse vers la chaîne profonde même sur une question
# relativement courte.
_ANALYTICAL_MARKERS: frozenset[str] = frozenset({
    "compare", "comparer", "comparaison", "différence", "différences",
    "pourquoi", "analyser", "analyse", "expliquer", "explique",
    "évalue", "évaluer", "critique", "avantages", "inconvénients",
    "conséquences", "implications", "relation", "lien", "croiser",
    "synthèse", "synthétiser", "raisonnement", "déduis", "déduire",
    "why", "analyze", "analyse", "evaluate", "implications",
    "trade-off", "tradeoff", "versus",
})


class DeepReasoningChain:
    """Chaîne de raisonnement multi-étapes pour modèles non-thinking.

    Parameters
    ----------
    model
        ``HFModelWrapper``. Doit exposer ``generate`` et un
        ``tokenizer`` avec ``apply_chat_template`` optionnel — même
        contrat que :class:`ReasoningGenerator`.
    kg
        ``KnowledgeGraphStore`` ou ``None``. Lu pour un court rappel
        d'entités, qui ancre le raisonnement dans le bon interlocuteur.
    """

    def __init__(self, model: Any, kg: Any | None) -> None:
        self.model = model
        self.kg = kg

    # ── Routeur de complexité ─────────────────────────────────────────
    def assess_complexity(
        self,
        message: str,
        surprise: dict[str, float] | None = None,
        doubt_index: float | None = None,
        kg_entity_count: int = 0,
    ) -> int:
        """Décide combien d'étapes de raisonnement exécuter.

        Combine des signaux déjà calculés par le pipeline. Aucun
        appel LLM ici — c'est une heuristique pure et rapide.

        Returns
        -------
        int
            ``0``  → question simple, ne pas lancer la chaîne (le
                     reasoning simple existant suffit).
            ``2``  → complexité moyenne, chaîne courte
                     (explorer + critiquer/feuille de route).
            ``4``  → complexité élevée, chaîne complète
                     (décomposer + explorer + critiquer + synthétiser).
        """
        msg = (message or "").strip()
        if not msg:
            return 0

        words = msg.split()
        score = 0

        # Signal 1 — longueur. Une question longue est rarement triviale.
        if len(words) >= _COMPLEXITY_MIN_WORDS:
            score += 1

        # Signal 2 — marqueurs analytiques explicites.
        lowered = msg.lower()
        if any(marker in lowered for marker in _ANALYTICAL_MARKERS):
            score += 2  # signal fort : intention analytique claire

        # Signal 3 — surprise globale élevée (le pipeline a vu de la
        # nouveauté → le modèle est en terrain peu familier).
        if surprise is not None:
            if surprise.get("global", 0.0) >= _COMPLEXITY_SURPRISE_THRESHOLD:
                score += 1

        # Signal 4 — doute élevé sur un tour précédent : le modèle a
        # déjà montré qu'il pataugeait sur ce type de sujet.
        if doubt_index is not None and doubt_index >= _COMPLEXITY_DOUBT_THRESHOLD:
            score += 1

        # Signal 5 — plusieurs entités en jeu → probable croisement
        # d'informations, ce que la chaîne profonde gère bien.
        if kg_entity_count >= _COMPLEXITY_MIN_ENTITIES:
            score += 1

        if score < 2:
            return 0

        # Score atteint → chaîne déclenchée. Le niveau (medium/high)
        # détermine 2 ou 4 étapes.
        level = self._complexity_level(message, surprise, kg_entity_count)
        if level == "high":
            return 4
        return 2

    # ── Appel LLM bas niveau ──────────────────────────────────────────
    def _call(self, system: str, user: str, max_tokens: int) -> str:
        """Un appel LLM court avec un couple (system, user).

        ``max_tokens`` varie selon l'étape — l'exploration a besoin de
        plus de place que la critique. Renvoie ``""`` en cas d'échec :
        l'appelant décide quoi faire d'une étape vide (en général,
        continuer avec ce qu'on a).
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
                # Fallback dégradé : pas de chat template → on aplatit.
                rendered = f"{system}\n\n{user}"
            out = self.model.generate(
                rendered,
                max_new_tokens=max_tokens,
                temperature=_STEP_TEMPERATURE,
            )
            return (out or "").strip()
        except Exception as exc:
            log.warning("Deep reasoning step failed: %s", exc)
            return ""

    # ── Rappel d'entités KG ───────────────────────────────────────────
    def _kg_hint(self) -> str:
        """Court rappel des entités connues, pour ancrer le raisonnement.

        Identique en esprit au ``_kg_facts_hint`` de ReasoningGenerator :
        sans ça, le modèle « imagine » un interlocuteur générique.
        """
        if not self.kg or not getattr(self.kg, "entities", None):
            return ""
        facts: list[str] = []
        for ent in self.kg.entities.values():
            facts.append(f"{ent.value} ({ent.type})")
            if len(facts) >= 10:
                break
        if not facts:
            return ""
        return (
            "\n\nContexte mémoire (informations sur l'utilisateur, "
            "PAS sur toi) : " + ", ".join(facts)
        )

    # ── Helper : bloc de contexte web ─────────────────────────────────
    def _web_hint(self, web_context: str) -> str:
        """Formate le contexte web pour injection dans un prompt.

        Le contexte web arrive déjà numéroté [1][2]… du WebAgent.
        On le présente comme une source de faits sur laquelle le
        modèle DOIT s'appuyer, plutôt que d'inventer. C'est ce qui
        empêche un petit modèle de halluciner des noms (observé :
        un 3B inventant « Systèmes de Galois » comme algorithme
        post-quantique faute de connaissances réelles).
        """
        web_context = (web_context or "").strip()
        if not web_context:
            return ""
        return (
            "\n\nInformations trouvées sur le web (utilise ces faits, "
            "ne les invente pas, et signale si une info manque) :\n"
            + web_context
        )

    # ── Étape 0 (4-step) : Décomposer ───────────────────────────────────
    def _decompose(self, message: str, max_tokens: int,
                   web_context: str = "") -> str:
        """Casser la question en sous-problèmes.

        Uniquement en mode 4-étapes. Produit une liste de 2-4
        sous-questions que l'étape d'exploration traitera. C'est une
        tâche très courte — le modèle n'a qu'à restructurer la
        question, pas à y répondre.

        ``web_context`` — quand présent, aide à découper la question
        selon les dimensions réellement documentées plutôt que selon
        des intuitions du modèle.
        """
        word_hint = int(max_tokens * 0.5 * 0.55)  # très court
        system = (
            "Tu es Rune, une IA cognitive. L'utilisateur pose une "
            "question complexe. Ton seul travail : la décomposer en "
            "sous-problèmes.\n"
            "Produis une liste de 2 à 4 sous-questions, numérotées, "
            "qui couvrent les dimensions essentielles du problème. "
            "Chaque sous-question doit être concrète et indépendante. "
            f"Vise environ {word_hint} mots — juste la liste, pas "
            "d'analyse."
            + self._kg_hint()
            + self._web_hint(web_context)
        )
        return self._call(system, message, max_tokens)

    # ── Étape 1 : Explorer ────────────────────────────────────────────
    def _explore(self, message: str, max_tokens: int,
                 sub_questions: str | None = None,
                 web_context: str = "") -> str:
        """Cartographier ce qu'on sait et ignore.

        En mode 2-étapes : reçoit juste la question de l'utilisateur.
        En mode 4-étapes : reçoit aussi les sous-questions produites
        par ``_decompose`` — l'exploration est alors guidée par ces
        axes au lieu de partir dans tous les sens.

        ``web_context`` — quand une recherche web a été déclenchée,
        ses résultats ancrent l'exploration dans des faits réels. Le
        modèle puise dans ces faits au lieu de ses connaissances
        internes (potentiellement limitées ou erronées).
        """
        word_hint = int(max_tokens * 0.8 * 0.55)
        web_block = self._web_hint(web_context)

        if sub_questions:
            # Mode 4-étapes : exploration guidée par les sous-problèmes.
            system = (
                "Tu es Rune, une IA cognitive. L'utilisateur pose une "
                "question complexe. Tu l'as déjà décomposée en "
                "sous-problèmes. Maintenant, pour CHAQUE sous-problème :\n"
                "- Ce que tu sais avec certitude (1-2 points).\n"
                "- Ce qui est incertain.\n"
                "Reste structuré et dense. "
                f"Vise environ {word_hint} mots au total. "
                "C'est une analyse préparatoire, pas la réponse finale. "
                "Termine tes phrases."
                + self._kg_hint()
                + web_block
            )
            user = (
                f"Question : {message}\n\n"
                f"Sous-problèmes identifiés :\n{sub_questions}"
            )
            return self._call(system, user, max_tokens)
        else:
            # Mode 2-étapes : exploration libre mais cadrée.
            system = (
                "Tu es Rune, une IA cognitive. L'utilisateur pose une "
                "question. Ne réponds PAS encore. Fais d'abord le tour du "
                "problème, de façon DENSE et CADRÉE :\n"
                "- Ce que tu sais avec certitude : 2 à 4 points clés.\n"
                "- Ce qui est incertain ou que tu ignores : 1 à 3 points.\n"
                "- Les angles à considérer : 4 à 5 MAXIMUM, les plus "
                "importants seulement.\n"
                "Ne dépasse pas ces limites — choisir l'essentiel fait "
                "partie du travail. Pas de listes à rallonge, pas de "
                "remplissage : un angle bien posé vaut mieux que cinq "
                "superficiels. "
                f"Vise environ {word_hint} mots. "
                "C'est une analyse préparatoire, pas la réponse finale. "
                "Termine tes phrases — ne laisse pas une idée en suspens."
                + self._kg_hint()
                + web_block
            )
            return self._call(system, message, max_tokens)

    # ── Étape 2 (2-step) / Étape 2 (4-step) : Critiquer ──────────────
    def _critique(self, message: str, exploration: str,
                  max_tokens: int, pure: bool = False) -> str:
        """Critiquer l'exploration.

        En mode 2-étapes (``pure=False``) : critique + feuille de route
        condensée, puisque c'est la dernière étape.

        En mode 4-étapes (``pure=True``) : critique PURE — repérer les
        faiblesses, point. La feuille de route est le travail de
        ``_synthesize``. Séparer les deux donne une critique plus
        honnête (le modèle n'essaie pas de « sauver » ses erreurs
        pour rédiger une belle feuille de route en même temps).
        """
        if pure:
            word_hint = int(max_tokens * 0.6 * 0.55)
            system = (
                "Tu es Rune, une IA cognitive. Voici une analyse "
                "préliminaire que tu as produite. Ton SEUL travail :\n"
                "- Repère ce qui est faux, incomplet, imprécis ou mal "
                "posé.\n"
                "- Pour chaque faille, explique brièvement POURQUOI "
                "c'est problématique.\n"
                "Ne corrige PAS encore — juste l'inventaire des "
                "faiblesses. Pas de feuille de route, pas de "
                "réécriture. "
                f"Vise environ {word_hint} mots. Termine tes phrases."
            )
        else:
            # Mode 2-étapes : critique + feuille de route
            word_hint = int(max_tokens * 0.6 * 0.55)
            system = (
                "Tu es Rune, une IA cognitive. Voici une analyse "
                "préliminaire que tu as produite pour la question de "
                "l'utilisateur. Ton travail maintenant :\n"
                "1. Repère ce qui est faux, incomplet ou mal posé dans "
                "cette analyse — corrige mentalement.\n"
                "2. Produis une FEUILLE DE ROUTE condensée pour ta "
                "future réponse. PAS un brouillon rédigé — une feuille "
                "de route. Format attendu :\n"
                "   - Structure : les axes de la réponse (2-4 max).\n"
                "   - Points-clés : faits/arguments indispensables, "
                "1 ligne chacun.\n"
                "   - Corrections : ce que l'exploration avait de faux, "
                "corrigé en 1-2 lignes.\n"
                "   - Conclusion visée : 1 phrase synthétique.\n"
                f"Vise environ {word_hint} mots — prise de notes, pas "
                "la réponse elle-même. Sois DENSE. Termine tes phrases."
            )

        user = (
            f"Question de l'utilisateur :\n{message}\n\n"
            f"Analyse préliminaire à critiquer :\n{exploration}"
        )
        return self._call(system, user, max_tokens)

    # ── Étape 3 (4-step) : Synthétiser ────────────────────────────────
    def _synthesize(self, message: str, exploration: str,
                    critique: str, max_tokens: int) -> str:
        """Feuille de route finale à partir de l'exploration + la critique.

        Uniquement en mode 4-étapes. Reçoit à la fois l'exploration
        (la matière) et la critique (les failles identifiées), et
        produit la feuille de route finale consolidée.

        Avantage de cette séparation vs le mode 2-étapes : la critique
        a pu pointer des erreurs sans se soucier de « sauver la mise »,
        et la synthèse peut maintenant construire sur du propre.
        """
        word_hint = int(max_tokens * 0.6 * 0.55)
        system = (
            "Tu es Rune, une IA cognitive. Tu as exploré une "
            "question puis critiqué ton exploration. Maintenant, "
            "construis la FEUILLE DE ROUTE FINALE pour ta réponse.\n"
            "Tu as en entrée :\n"
            "- L'exploration (ce que tu sais et ignores).\n"
            "- La critique (les failles identifiées).\n"
            "Produis une feuille de route qui intègre les corrections "
            "de la critique. Format :\n"
            "   - Structure : 2-4 axes de la réponse.\n"
            "   - Points-clés : faits/arguments indispensables, "
            "1 ligne chacun.\n"
            "   - Corrections intégrées : ce qui a été corrigé depuis "
            "l'exploration.\n"
            "   - Conclusion visée : 1 phrase synthétique.\n"
            f"Vise environ {word_hint} mots — dense, pas de rédaction. "
            "Termine tes phrases."
        )
        user = (
            f"Question : {message}\n\n"
            f"Exploration :\n{exploration}\n\n"
            f"Critique :\n{critique}"
        )
        return self._call(system, user, max_tokens)

    # ── Orchestration ─────────────────────────────────────────────────
    def run(
        self,
        message: str,
        surprise: dict[str, float] | None = None,
        doubt_index: float | None = None,
        kg_entity_count: int = 0,
        on_step: Callable[[str], None] | None = None,
        force_minimum: int = 0,
        web_context: str = "",
    ) -> str:
        """Exécute la chaîne de raisonnement adaptée à la complexité.

        Parameters
        ----------
        force_minimum
            Nombre minimum d'étapes à exécuter, même si le routeur
            aurait décidé moins. Typiquement ``2`` quand l'utilisateur
            a explicitement activé le raisonnement — on garantit au
            moins une chaîne 2-étapes pour tout message non-trivial.
            ``0`` laisse le routeur décider librement.
        web_context
            Résultats d'une recherche web (numérotés [1][2]…), quand
            elle a été déclenchée. Injectés dans les étapes de
            décomposition et d'exploration pour ancrer le raisonnement
            dans des faits réels — particulièrement utile sur les
            petits modèles, dont les connaissances internes sont
            limitées. ``""`` = pas d'ancrage web.

        Le routeur décide du nombre d'étapes :

        - **0** → question simple, retour vide (→ fallback reasoning
          simple).
        - **2** → chaîne courte : explorer → critique/feuille de route.
          Budget : ``reasoning_deep_step_medium_tokens`` par étape.
        - **4** → chaîne complète : décomposer → explorer → critiquer
          (pure) → synthétiser. Budget : ``..._high_tokens`` pour
          l'exploration (c'est le cœur), ``..._medium_tokens`` pour
          les 3 autres étapes (tâches ciblées et courtes).

        Parameters
        ----------
        message
            La question de l'utilisateur.
        surprise, doubt_index, kg_entity_count
            Signaux du pipeline pour le routeur de complexité.
        on_step
            Callback appelé avec un libellé court à chaque étape
            (``"Décomposition…"``, ``"Exploration…"``, etc.).

        Returns
        -------
        str
            Le raisonnement consolidé (feuille de route), ou ``""``
            si la chaîne n'a pas été déclenchée ou a entièrement
            échoué. Retour vide → l'orchestrateur retombe sur le
            ``ReasoningGenerator`` simple.
        """
        steps = self.assess_complexity(
            message, surprise, doubt_index, kg_entity_count,
        )
        # force_minimum relève le plancher : si l'utilisateur a
        # explicitement activé le raisonnement, on garantit au moins
        # 2 étapes même si le routeur aurait dit 0.
        steps = max(steps, force_minimum)
        if steps == 0:
            return ""

        from rune.settings import get_settings
        s = get_settings()
        max_chars = s.reasoning_text_max_chars
        budget_medium = s.reasoning_deep_step_medium_tokens
        budget_high = s.reasoning_deep_step_high_tokens

        # Borner le contexte web : on ne veut pas qu'il écrase le
        # prompt d'exploration. Déjà numéroté [1][2]… par le WebAgent.
        web_ctx = (web_context or "").strip()[:_WEB_CONTEXT_MAX_CHARS]

        t0 = time.time()

        if steps == 4:
            return self._run_4_steps(
                message, budget_medium, budget_high, max_chars,
                t0, on_step, web_ctx,
            )
        else:
            return self._run_2_steps(
                message, budget_medium, max_chars, t0, on_step, web_ctx,
            )

    # ── Chaîne 2-étapes ───────────────────────────────────────────────
    def _run_2_steps(
        self,
        message: str,
        step_budget: int,
        max_chars: int,
        t0: float,
        on_step: Callable[[str], None] | None,
        web_ctx: str = "",
    ) -> str:
        """Explorer → Critique/Feuille de route."""
        # Étape 1 : Explorer
        if on_step:
            on_step("Exploration…")
        exploration = self._explore(message, step_budget, web_context=web_ctx)
        if not exploration:
            log.info("Deep reasoning 2-step: exploration vide, abandon")
            return ""
        if (time.time() - t0) > _CHAIN_MAX_SECONDS:
            log.info("Deep reasoning 2-step: timeout après exploration")
            return exploration[:max_chars]

        # Étape 2 : Critique + feuille de route
        if on_step:
            on_step("Feuille de route…")
        result = self._critique(message, exploration, step_budget, pure=False)
        if not result:
            log.info("Deep reasoning 2-step: critique vide, repli sur exploration")
            return exploration[:max_chars]

        elapsed = time.time() - t0
        log.info(
            "Deep reasoning 2-step terminée en %.1fs "
            "(budget=%d tok, web=%s, explo=%d chars, résultat=%d chars)",
            elapsed, step_budget, "oui" if web_ctx else "non",
            len(exploration), len(result),
        )
        return result[:max_chars]

    # ── Chaîne 4-étapes ───────────────────────────────────────────────
    def _run_4_steps(
        self,
        message: str,
        budget_medium: int,
        budget_high: int,
        max_chars: int,
        t0: float,
        on_step: Callable[[str], None] | None,
        web_ctx: str = "",
    ) -> str:
        """Décomposer → Explorer → Critiquer (pure) → Synthétiser."""

        def _timeout() -> bool:
            return (time.time() - t0) > _CHAIN_MAX_SECONDS

        # ── Étape 1 : Décomposer ──────────────────────────────────────
        if on_step:
            on_step("Décomposition…")
        sub_questions = self._decompose(message, budget_medium, web_context=web_ctx)
        if not sub_questions:
            log.info("Deep reasoning 4-step: décomposition vide, "
                     "repli sur chaîne 2-étapes")
            # Dégradation gracieuse : si on ne peut pas décomposer,
            # on retombe sur la chaîne 2-étapes.
            return self._run_2_steps(
                message, budget_medium, max_chars, t0, on_step, web_ctx,
            )
        if _timeout():
            log.info("Deep reasoning 4-step: timeout après décomposition")
            return sub_questions[:max_chars]

        # ── Étape 2 : Explorer (guidé par les sous-problèmes) ─────────
        if on_step:
            on_step("Exploration…")
        exploration = self._explore(
            message, budget_high, sub_questions=sub_questions,
            web_context=web_ctx,
        )
        if not exploration:
            log.info("Deep reasoning 4-step: exploration vide")
            return sub_questions[:max_chars]
        if _timeout():
            log.info("Deep reasoning 4-step: timeout après exploration")
            return exploration[:max_chars]

        # ── Étape 3 : Critiquer (pure — juste les failles) ────────────
        if on_step:
            on_step("Critique…")
        critique = self._critique(
            message, exploration, budget_medium, pure=True,
        )
        if not critique:
            # Pas de critique → on fait la synthèse sans.
            log.info("Deep reasoning 4-step: critique vide, "
                     "synthèse sans critique")
            critique = "(pas de faille majeure identifiée)"
        if _timeout():
            log.info("Deep reasoning 4-step: timeout après critique")
            # On a exploration + critique, on peut quand même synthétiser
            # → on ne coupe pas, on laisse la synthèse se faire.

        # ── Étape 4 : Synthétiser (feuille de route finale) ───────────
        if on_step:
            on_step("Synthèse…")
        result = self._synthesize(
            message, exploration, critique, budget_medium,
        )
        if not result:
            log.info("Deep reasoning 4-step: synthèse vide, "
                     "repli sur exploration")
            return exploration[:max_chars]

        elapsed = time.time() - t0
        log.info(
            "Deep reasoning 4-step terminée en %.1fs "
            "(décomp=%d, explo=%d, critique=%d, synthèse=%d chars)",
            elapsed, len(sub_questions), len(exploration),
            len(critique), len(result),
        )
        return result[:max_chars]

    # ── Niveau de complexité pour le budget ───────────────────────────
    def _complexity_level(
        self,
        message: str,
        surprise: dict[str, float] | None,
        kg_entity_count: int,
    ) -> str:
        """Distingue ``"medium"`` et ``"high"`` pour calibrer le budget.

        ``assess_complexity`` décide *s'il faut* lancer la chaîne ;
        cette méthode décide *combien de place* lui donner. Une
        question est "high" si elle cumule plusieurs signaux forts —
        marqueur analytique ET (surprise élevée OU plusieurs entités
        OU question longue). Sinon "medium".
        """
        msg = (message or "").strip()
        lowered = msg.lower()
        words = msg.split()

        has_marker = any(m in lowered for m in _ANALYTICAL_MARKERS)
        high_surprise = bool(
            surprise
            and surprise.get("global", 0.0) >= _COMPLEXITY_SURPRISE_THRESHOLD
        )
        many_entities = kg_entity_count >= _COMPLEXITY_MIN_ENTITIES
        long_question = len(words) >= _COMPLEXITY_MIN_WORDS

        strong_signals = sum(
            [high_surprise, many_entities, long_question]
        )
        # "high" : un marqueur analytique explicite ET au moins un
        # autre signal fort. C'est le profil d'une vraie question
        # d'analyse qui mérite le grand budget.
        if has_marker and strong_signals >= 1:
            return "high"
        return "medium"
