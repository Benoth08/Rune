"""Encoding phase — text → numerical representations.

Biological inspiration
----------------------
In the cortical-hippocampal pipeline, raw sensory input is first
transduced through entorhinal cortex into a high-dimensional code
before reaching CA3/CA1. The entorhinal stage is *non-runenic*:
it does not store anything, it only *prepares* signals (place/grid
encoding, sparsification, novelty gating) for downstream modules.

This module mirrors that role. It does not read or write any
memory. It produces an :class:`EncodingResult` that downstream
phases (Storage, Surprise, Retrieval) consume.

Three encodings are produced in one pass:

1. **LLM latent encoding** via ``model.analyze_input`` — per-token
   hidden states + per-token entropies + a mean-pooled latent.
   This is the substrate for SDM writes (Storage) and for the
   predictive surprise signal (Surprise).
2. **GLiNER embedding** of the whole utterance — used as the key
   for MHN episodic recall and storage.
3. **Entity extraction** — GLiNER NER pass, with a stop-list filter
   to drop pronouns, articles, and generic nouns that would
   otherwise pollute the knowledge graph.

A salience cascade (N1 → N2 → N3) gates the whole thing: if the
input is judged non-salient (e.g. "ok", "bonjour"), we short-circuit
and return an empty result, mirroring the original ``_phase_a_learn``
behaviour exactly.

Design notes
------------
- All model calls are wrapped: ``analyze_input`` may fail (OOM,
  truncation, unknown tokenizer), GLiNER may not be installed at all.
  In every failure mode we degrade gracefully and log at WARNING.
- The entity noise list is intentionally identical to the one in the
  original ``_phase_a_learn`` (l. 365–368). Do not extend it here
  without updating tests — the noise list is part of the contract.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch

log = logging.getLogger("rune.cognition.encoding")


# Generic-noise stop list for KG entity filtering. Identical to the
# original list in hippocampe._phase_a_learn — kept here as the
# canonical source. Lower-cased, whitespace-stripped lookup.
#
# V5.5.3 — Ajout de "je suis", "i am", "i'm" et variantes : GLiNER
# extrait parfois ces fragments comme PERSON quand ils sont en début
# de phrase ("Je suis né en 1985" → "Je suis" classé person avec
# score > 0.3). Le filtre étend la noise list pour neutraliser ces
# faux positifs à la source.
ENTITY_NOISE: frozenset[str] = frozenset({
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils",
    "elles", "le", "la", "les", "un", "une", "des", "ce",
    "cette", "projet", "travail", "chose", "truc", "ça",
    "i", "me", "my", "we", "you", "he", "she", "it", "they",
    # V5.5.3 — faux positifs GLiNER fréquents (verbes/pronoms +
    # auxiliaire). Toujours en lowercase pour la comparaison.
    "je suis", "tu es", "il est", "elle est", "on est",
    "nous sommes", "vous êtes", "ils sont", "elles sont",
    "j'ai", "tu as", "il a", "elle a", "on a",
    "i am", "i'm", "you are", "you're", "he is", "she is",
    "it is", "we are", "they are",
    "moi", "toi", "lui", "soi", "eux",
    # V5.6.1 — salutations extraites comme entités par GLiNER
    # ("Bonjour" remonté comme person/location parfois).
    "bonjour", "bonsoir", "salut", "coucou", "hello", "hi",
    "hey", "good morning", "good evening", "good afternoon",
    "bonne journée", "bonne soirée",
    # V5.8.4 — impératifs verbe-pronom captés comme person/object
    # par GLiNER ("Aide-moi à organiser..." → person:Aide-moi).
    # Le tiret + majuscule initiale ressemble à un nom propre composé
    # type "Jean-Pierre" pour le NER. Liste des verbes impératifs FR
    # courants + leur variante avec pronom enclitique.
    "aide-moi", "aide moi", "aidez-moi", "aidez moi",
    "dis-moi", "dis moi", "dites-moi", "dites moi",
    "donne-moi", "donne moi", "donnez-moi", "donnez moi",
    "montre-moi", "montre moi", "montrez-moi", "montrez moi",
    "explique-moi", "explique moi", "expliquez-moi", "expliquez moi",
    "raconte-moi", "raconte moi", "racontez-moi", "racontez moi",
    "fais-moi", "fais moi", "faites-moi", "faites moi",
    "écoute-moi", "écoute moi", "écoutez-moi", "écoutez moi",
    "regarde-moi", "regarde moi", "regardez-moi", "regardez moi",
    "envoie-moi", "envoie moi", "envoyez-moi", "envoyez moi",
    "lis-moi", "lis moi", "lisez-moi", "lisez moi",
    "rappelle-moi", "rappelle moi", "rappelez-moi", "rappelez moi",
    "parle-moi", "parle moi", "parlez-moi", "parlez moi",
    # V5.8.5 — jurons et interjections extraits à tort par GLiNER
    # (souvent en majuscule en début de phrase → person/object).
    # Liste conservatrice : on garde les valeurs émotionnelles
    # dans le lexique affectif (Cognitive state) mais on les sort
    # du KG.
    "putain", "merde", "fait chier", "fais chier", "fuck",
    "shit", "damn", "wtf", "bordel", "purée", "zut",
    "mince", "flûte", "mon dieu", "oh mon dieu", "oh là là",
    "oh la la", "oh", "ah", "eh", "bah", "hein", "ouf",
    "argh", "ugh", "beurk", "yeah", "oh yeah", "ouais",
    # V5.8.5 — adjectifs émotionnels qui remontent en entity_label
    # (parfois person, parfois object). Ils appartiennent au signal
    # affectif, pas au KG comme entités stables.
    "furax", "vénère", "saoulé", "saoule", "soulé", "chiant",
    "stressé", "stresse", "angoisse", "marre",
    "super content", "content", "happy", "sad", "angry",
})


# ── V5.5.2 — Self-disclosure fallback patterns ──────────────────────────
#
# GLiNER + ses filtres ratent systématiquement les déclarations en
# minuscule, surtout en français : "je m'appelle cédric" rate alors
# que "je m'appelle Cédric" passe. C'est un problème générique qui ne
# touche pas QUE les prénoms — métier, lieu, employeur, âge sont aussi
# perdus si l'utilisateur tape vite et sans majuscules.
#
# On ajoute donc 5 patterns regex qui capturent les déclarations
# explicites de **faits personnels** (self-disclosure). Chaque pattern :
#   • Est case-insensitive
#   • Capture une valeur normalisée (Title Case pour person/loc/org,
#     lower pour role, int pour age)
#   • Est testé contre des cas négatifs (tierce personne, déplacement
#     vs résidence, futur conditionnel)
#   • Est tagué avec la confidence 0.95 (pattern explicite = haute
#     fiabilité)
#
# Philosophie : la regex est conservatrice. Mieux vaut louper une
# formulation exotique que créer du faux positif qui pollue le KG.
# Les utilisateurs qui veulent garantir l'extraction d'un fait
# peuvent toujours le formuler explicitement.


# 1️⃣ Prénom — "je m'appelle X", "moi c'est X", "my name is X", etc.
_SELF_INTRO_RE = re.compile(
    r"(?:"
    r"je\s+m['’]?appelle"
    r"|moi\s*[,]?\s*c['’]?est"
    r"|mon\s+nom\s+(?:est|c['’]?est)"
    r"|mon\s+prénom\s+(?:est|c['’]?est)"
    r"|appelle[-\s]moi"
    r"|my\s+name\s+is"
    r"|call\s+me"
    r"|i['’]?m"
    r"|i\s+am"
    r")"
    r"\s+"
    r"([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]{1,29})",
    re.IGNORECASE,
)


# 2️⃣ Métier / Rôle — "je suis chimiométricien", "je travaille comme X"
#
# Subtilité : "je suis X" est ambigu en français — peut être une
# profession ("je suis ingénieur") ou un état ("je suis fatigué").
# On résout en exigeant que la valeur capturée soit un substantif
# probable (≥ 5 chars, pas dans un blocklist d'adjectifs courants).
# Les adjectifs émotionnels/états passagers sont filtrés post-capture.
_SELF_ROLE_RE = re.compile(
    r"(?:"
    r"je\s+suis\s+(?:un\s+|une\s+)?"
    r"|je\s+travaille\s+(?:comme|en\s+tant\s+que)\s+"
    r"|mon\s+métier\s+(?:est|c['’]?est)\s+(?:un\s+|une\s+)?"
    r"|mon\s+job\s+(?:est|c['’]?est)\s+(?:un\s+|une\s+)?"
    r"|je\s+bosse\s+comme\s+"
    r"|i['’]?m\s+(?:a|an)\s+"
    r"|i\s+work\s+as\s+(?:a|an)\s+"
    r"|my\s+job\s+is\s+"
    r")"
    # V5.5.2 : exclut explicitement les amorces qui signalent un lieu
    # (à, au, en, dans, chez) ou un état temporel (en train de) — pas
    # un rôle. Le négatif lookahead évite "je suis à Aix" → 'à Aix'.
    r"(?!(?:à|au|aux|en|dans|chez|sur)\b|en\s+train\s+|né|née|d['’])"
    r"([a-zA-ZÀ-ÖØ-öø-ÿ][a-zA-ZÀ-ÖØ-öø-ÿ &\-]{4,49}?)"
    r"(?=[\s.,;!?]*$|[.,;!?]|\s+(?:depuis|since|à|in|chez|at|et|and)\b)",
    re.IGNORECASE,
)

# Adjectifs / états passagers à NE PAS capturer comme rôle.
# "je suis fatigué", "je suis content" ne sont pas des métiers.
_ROLE_BLOCKLIST = frozenset({
    "fatigué", "fatiguée", "content", "contente", "triste", "désolé",
    "désolée", "stressé", "stressée", "épuisé", "épuisée", "heureux",
    "heureuse", "malade", "occupé", "occupée", "libre", "prêt", "prête",
    "perdu", "perdue", "déçu", "déçue", "énervé", "énervée", "ravi",
    "ravie", "curieux", "curieuse", "inquiet", "inquiète", "sûr", "sûre",
    "certain", "certaine", "français", "française", "anglais", "anglaise",
    "marié", "mariée", "célibataire", "divorcé", "divorcée",
    "jeune", "vieux", "vieille", "grand", "grande", "petit", "petite",
    "tired", "happy", "sad", "sorry", "busy", "free", "ready",
    "married", "single", "young", "old",
    # Faux positifs liés au verbe être suivi d'un substantif générique
    "homme", "femme", "personne", "humain", "humaine", "type", "gars",
    "man", "woman", "person", "guy",
})


# 3️⃣ Lieu de résidence — "j'habite à X", "je vis à X", "I live in X"
#
# IMPORTANT : on ne capture PAS "je vais à X" (déplacement futur,
# pas résidence) ni "je viens de X" (origine, pas actuel).
_SELF_LOCATION_RE = re.compile(
    r"(?:"
    r"j['’]?habite\s+(?:à|en|au|aux)\s+"
    r"|je\s+vis\s+(?:à|en|au|aux)\s+"
    r"|je\s+suis\s+(?:basé|basée|installé|installée)\s+(?:à|en|au|aux)\s+"
    r"|je\s+réside\s+(?:à|en|au|aux)\s+"
    r"|i\s+live\s+in\s+"
    r"|i['’]?m\s+based\s+in\s+"
    r"|my\s+(?:home|city)\s+is\s+"
    r")"
    r"([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\s\-]{1,39}?)"
    r"(?=[\s.,;!?]|$|\s+(?:depuis|since|et|and))",
    re.IGNORECASE,
)


# 4️⃣ Employeur — "je travaille chez X", "I work at X"
#
# Distinction critique : "chez" + lieu d'habitation possible (chez moi,
# chez mes parents) → on filtre post-capture via _EMPLOYER_BLOCKLIST.
_SELF_EMPLOYER_RE = re.compile(
    r"(?:"
    r"je\s+travaille\s+(?:chez|pour|à|au|aux)\s+"
    r"|je\s+bosse\s+(?:chez|pour|à|au|aux)\s+"
    r"|je\s+suis\s+(?:employé|employée|salarié|salariée)\s+(?:chez|à|au|aux)\s+"
    # V5.5.2 : ajout "je suis chez X" / "je suis à X" pour employeur
    # — pattern fréquent qui manquait ("je suis chez Thalès").
    r"|je\s+suis\s+chez\s+"
    r"|je\s+suis\s+à\s+(?=[A-ZÀ-Ö])"  # exige proper noun derrière (lieu/employer)
    r"|mon\s+employeur\s+(?:est|c['’]?est)\s+"
    r"|ma\s+(?:boîte|société|entreprise)\s+(?:est|c['’]?est|s['’]?appelle)\s+"
    r"|i\s+work\s+(?:at|for)\s+"
    r"|my\s+(?:company|employer)\s+is\s+"
    r")"
    # V5.5.2 : capture les noms multi-mots ("TOPNIR Systems", "BNP Paribas"),
    # bornée par lookahead ponctuation/connecteur.
    # V5.5.9 : ajout de nombreux connecteurs/circonstanciels qui
    # peuvent suivre le nom de l'employeur ("chez moi en télétravail",
    # "chez mes parents cet été", "chez Framatome pour 2 ans", etc.).
    r"([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ &\-]{1,49}?)"
    r"(?=[\s.,;!?]*$|[.,;!?]"
    r"|\s+(?:depuis|since|à|in|et|and|comme|as|en|au|aux|dans|"
    r"sur|pour|sous|cet|cette|ce|ces|aujourd|hier|demain|"
    r"actuellement|maintenant|currently)\b)",
    re.IGNORECASE,
)

_EMPLOYER_BLOCKLIST = frozenset({
    "moi", "lui", "elle", "eux", "soi", "nous",
    "mes parents", "mes amis", "ma famille", "ma mère", "mon père",
    "me", "myself", "my parents", "my family", "my mom", "my dad",
})


# 5️⃣ Âge — "j'ai N ans", "I'm N years old"
#
# Limite : 1-120 pour éviter de capturer "j'ai 1000 fichiers".
# Capture le NOMBRE, pas le mot environnant.
_SELF_AGE_RE = re.compile(
    r"(?:"
    r"j['’]?ai\s+"
    r"|i['’]?m\s+"
    r"|i\s+am\s+"
    r")"
    r"(\d{1,3})"
    r"\s+(?:ans?|years?\s+old)",
    re.IGNORECASE,
)


# 6️⃣ Année de naissance — "je suis né en 1985", "I was born in 1985"
#
# V5.5.3 — Capture explicite des années 4 chiffres. Bornes :
# 1900-2025 (sanity check post-capture). Stocké avec label "date"
# et préfixe "birth_year:" pour disambiguation côté KG.
_SELF_BIRTH_YEAR_RE = re.compile(
    r"(?:"
    r"je\s+suis\s+née?\s+en\s+"
    r"|né(?:e)?\s+en\s+"
    r"|i\s+was\s+born\s+in\s+"
    r"|born\s+in\s+"
    r")"
    r"(\d{4})\b",
    re.IGNORECASE,
)


# Helper : valeurs trop génériques pour être un fait utilisateur, peu
# importe le pattern qui les a capturées.
_GENERIC_VALUES = frozenset({
    "là", "ici", "ailleurs", "quelqu'un", "personne", "rien",
    "here", "there", "someone", "anyone", "nobody",
})


@dataclass
class EncodingResult:
    """Output of the Encoding phase, consumed by downstream phases.

    Attributes
    ----------
    salient
        Whether the salience cascade let this input through. When
        ``False``, all other fields are at their defaults and the
        downstream phases must be skipped.
    structural_entropy
        Mean per-token entropy from the LLM (clipped to [0, 1]).
        Defaults to 0.5 if the model is not loaded.
    latents
        Per-token hidden states ``(n_tokens, hidden_dim)``. ``None``
        if ``analyze_input`` was unavailable or failed.
    token_entropies
        Per-token entropies aligned with ``latents``. ``None`` in
        the same conditions as ``latents``.
    mean_latent
        Mean-pooled latent ``(hidden_dim,)``. Used as the SDM
        projection input for the predictive surprise signal so that
        Surprise sees the *same* projection as Storage.
    gliner_emb
        GLiNER sentence embedding. Used as MHN key for both episodic
        surprise (Surprise) and episodic recall (Retrieval).
    raw_entities
        NER hits, post-noise-filter. Each entry is the raw GLiNER
        dict ``{text, label, score, ...}``. Storage promotes them
        to the KG; nothing here writes anything.
    """

    salient: bool = False
    structural_entropy: float = 0.5
    latents: torch.Tensor | None = None
    token_entropies: list[float] | None = None
    mean_latent: torch.Tensor | None = None
    gliner_emb: torch.Tensor | None = None
    raw_entities: list[dict[str, Any]] = field(default_factory=list)


class EncodingPhase:
    """Encode text into latents, entropies, embeddings, and entities.

    Parameters
    ----------
    model
        :class:`HFModelWrapper`. Must expose ``is_loaded``,
        ``analyze_input(text)``, ``hidden_dim``.
    entity_extractor
        :class:`EntityExtractor` (GLiNER wrapper). Must expose
        ``encode(text)`` and ``extract(text)``. May be ``None`` —
        in that case no entities are produced and no GLiNER
        embedding is computed (downstream phases must tolerate this).
    salience
        :class:`SalienceFilter`. Must expose ``evaluate(text, emb)``
        returning an object with a ``passed: bool`` attribute.
    """

    def __init__(
        self,
        model: Any,
        entity_extractor: Any | None,
        salience: Any,
    ) -> None:
        self.model = model
        self.entity_extractor = entity_extractor
        self.salience = salience

    # ── Public API ─────────────────────────────────────────────────────

    def encode(self, text: str, has_images: bool = False) -> EncodingResult:
        """Run the full encoding pipeline on a user utterance.

        Order matters: GLiNER embedding is needed *before* the
        salience cascade (N3 uses it as the novelty key). If salience
        rejects, we stop right there — unless ``has_images`` is True,
        in which case we bypass salience and treat the turn as
        always-salient.

        Parameters
        ----------
        text
            User input, raw.
        has_images
            ``True`` if the user attached one or more images to this
            turn. The text alone may be a stub like "Décris cette
            image" that the salience cascade would reject as noise,
            but the image carries new information that is worth
            archiving. We override the salience verdict in that case
            so the exchange is properly committed to long-term memory.

        Returns
        -------
        EncodingResult
            All-empty if ``salient=False``. Otherwise populated as
            far as available models permit.
        """
        # GLiNER embedding goes first — used by salience N3.
        gliner_emb = self._encode_gliner(text)

        # Salience cascade — quick noise gate. Bypassed for image turns
        # because the textual stub doesn't reflect the actual content.
        if not has_images:
            sal_result = self.salience.evaluate(text, gliner_emb)
            if not sal_result.passed:
                return EncodingResult(salient=False)

        # LLM latent encoding (entropies + hidden states).
        structural_entropy, latents, token_entropies, mean_latent = (
            self._encode_latents(text)
        )

        # Entity extraction (post-filter for noise + tiny strings).
        raw_entities = self._extract_entities(text)

        return EncodingResult(
            salient=True,
            structural_entropy=structural_entropy,
            latents=latents,
            token_entropies=token_entropies,
            mean_latent=mean_latent,
            gliner_emb=gliner_emb,
            raw_entities=raw_entities,
        )

    # ── Internal — three encoding sub-steps ────────────────────────────

    def _encode_gliner(self, text: str) -> torch.Tensor | None:
        """GLiNER sentence embedding, or ``None`` if unavailable."""
        if self.entity_extractor is None:
            return None
        try:
            return self.entity_extractor.encode(text)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("GLiNER encode failed: %s", exc)
            return None

    def _encode_latents(
        self, text: str,
    ) -> tuple[float, torch.Tensor | None, list[float] | None, torch.Tensor | None]:
        """LLM latent + entropy encoding via ``model.analyze_input``.

        Returns
        -------
        tuple
            ``(structural_entropy, latents, token_entropies, mean_latent)``.
            On any failure or when the model is not loaded, returns
            ``(0.5, None, None, None)`` — the same fallback as the
            original ``_phase_a_learn``.
        """
        if not getattr(self.model, "is_loaded", False):
            return 0.5, None, None, None
        try:
            analysis = self.model.analyze_input(text)
            structural_entropy = float(analysis["mean_entropy"])
            latents = analysis["latent_states"]
            token_entropies = analysis["token_entropies"]
            mean_latent = latents.mean(dim=0)
            log.info(
                "analyze_input OK: %d tokens, mean_ent=%.3f, latent_dim=%d",
                len(token_entropies), structural_entropy, latents.shape[-1],
            )
            return structural_entropy, latents, token_entropies, mean_latent
        except Exception as exc:
            log.warning("analyze_input FAILED: %s", exc)
            return 0.5, None, None, None

    def _extract_entities(self, text: str) -> list[dict[str, Any]]:
        """GLiNER NER pass + noise + length filter + V5.5.2 self-disclosure fallbacks.

        Filtering rules (preserved verbatim from the original code):

        * lower-cased, stripped text in :data:`ENTITY_NOISE` → drop
        * stripped length < 2 → drop

        V5.5.2 — Self-disclosure fallback patterns
        --------------------------------------------
        GLiNER (en seuil 0.3, modèle multi-v2.1) extrait mal les faits
        personnels en minuscule, surtout en français. On ajoute 5
        fallbacks regex AVANT le filtrage final :

        1. **Prénom** ("je m'appelle X", FR/EN) → label `person`
        2. **Métier** ("je suis chimiométricien") → label `role`
        3. **Lieu** ("j'habite à Aix") → label `location`
        4. **Employeur** ("je travaille chez Framatome") → label `organization`
        5. **Âge** ("j'ai 36 ans") → label `age`

        Garde-fous post-capture :
        - Blocklist d'adjectifs/états passagers (rôle)
        - Blocklist de pronoms (employeur "chez moi")
        - Blocklist de valeurs génériques (lieu "là", "ici")
        - Âge dans [1, 120]
        - Normalisation Title Case pour person/loc/org
        - Dédup case-insensitive contre les résultats GLiNER existants
        """
        if self.entity_extractor is None:
            return []
        try:
            raw = self.entity_extractor.extract(text)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("GLiNER extract failed: %s", exc)
            raw = []

        # V5.5.2 — Apply 5 self-disclosure fallbacks (extended V5.5.3)
        text_safe = text or ""
        self._apply_self_intro(text_safe, raw)
        self._apply_self_role(text_safe, raw)
        self._apply_self_location(text_safe, raw)
        self._apply_self_employer(text_safe, raw)
        self._apply_self_age(text_safe, raw)
        # V5.5.3 — Année de naissance ("je suis né en 1985").
        self._apply_self_birth_year(text_safe, raw)

        kept: list[dict[str, Any]] = []
        for ent in raw:
            # V5.5.3 — Normalisation robuste avant filtrage noise :
            # lowercase, strip, normalisation des apostrophes courbes
            # (’ → '), et compression des espaces multiples. Sans ça,
            # "J’ai" en apostrophe typographique passait le filtre.
            text_norm = (
                ent["text"]
                .lower()
                .strip()
                .replace("\u2019", "'")  # apostrophe typographique
                .replace("\u02bc", "'")  # modifier letter apostrophe
            )
            # Compress multiple spaces (GLiNER renvoie parfois "je  suis")
            text_norm = " ".join(text_norm.split())
            if text_norm in ENTITY_NOISE:
                continue
            if len(ent["text"].strip()) < 2:
                continue
            kept.append(ent)
        return kept

    # ── V5.5.2 — Per-category self-disclosure appliers ──────────────────

    @staticmethod
    def _already_in(raw: list[dict[str, Any]], value: str, label: str) -> bool:
        """True si (value, label) est déjà dans raw (case-insensitive)."""
        v_low = value.lower()
        for e in raw:
            if (e.get("text", "").lower() == v_low
                    and e.get("label", "") == label):
                return True
        return False

    @staticmethod
    def _upgrade_or_insert(
        raw: list[dict[str, Any]], value: str, label: str, score: float,
    ) -> str:
        """V5.6.3 — Si une entrée case-insensitive existe déjà pour
        (value, label), on **remplace** son texte par la valeur normalisée
        passée en argument (plutôt que de skip ou de doubler). Cela
        permet aux fallbacks self-disclosure d'imposer leur normalisation
        Title Case sur les extractions GLiNER en minuscule.

        Retourne :
        - "upgraded" : une entrée existait, on a normalisé son texte
        - "inserted" : pas d'entrée, on a ajouté
        - "skipped"  : déjà identique (rien à faire)
        """
        v_low = value.lower()
        for e in raw:
            if (e.get("text", "").lower() == v_low
                    and e.get("label", "") == label):
                if e.get("text") == value:
                    return "skipped"
                e["text"] = value
                # Garder le meilleur score
                if score > e.get("score", 0.0):
                    e["score"] = score
                return "upgraded"
        raw.append({"text": value, "label": label, "score": score})
        return "inserted"

    @staticmethod
    def _normalize_proper(value: str) -> str:
        """Title-case avec préservation des tirets et accents.

        "cédric" → "Cédric"
        "jean-pierre" → "Jean-Pierre"
        "aix-en-provence" → "Aix-En-Provence"
        """
        return "-".join(p.capitalize() for p in value.split("-"))

    def _apply_self_intro(self, text: str, raw: list[dict[str, Any]]) -> None:
        """Pattern 1/5 — prénom."""
        for raw_name in _SELF_INTRO_RE.findall(text):
            cleaned = raw_name.strip().rstrip(".,;!?").strip()
            if not cleaned or len(cleaned) < 2:
                continue
            if cleaned.lower() in _GENERIC_VALUES:
                continue
            normalized = self._normalize_proper(cleaned)
            # V5.6.3 — upgrade : si GLiNER avait déjà extrait le prénom
            # en minuscule, on remplace par notre version Title Case.
            action = self._upgrade_or_insert(raw, normalized, "person", 0.95)
            if action == "inserted":
                log.info("Self-intro fallback: %r → person (new)", normalized)
            elif action == "upgraded":
                log.info("Self-intro fallback: %r → person (upgraded case)", normalized)

    def _apply_self_role(self, text: str, raw: list[dict[str, Any]]) -> None:
        """Pattern 2/5 — métier / rôle."""
        for raw_role in _SELF_ROLE_RE.findall(text):
            cleaned = raw_role.strip().rstrip(".,;!?").strip()
            if not cleaned or len(cleaned) < 5:
                continue
            cleaned_low = cleaned.lower()
            if cleaned_low in _ROLE_BLOCKLIST:
                continue
            if cleaned_low in _GENERIC_VALUES:
                continue
            # Rôle en lower (les métiers sont des noms communs)
            if self._already_in(raw, cleaned_low, "role"):
                continue
            raw.append({
                "text": cleaned_low, "label": "role", "score": 0.95,
            })
            log.info("Self-role fallback: %r → role", cleaned_low)

    def _apply_self_location(self, text: str, raw: list[dict[str, Any]]) -> None:
        """Pattern 3/5 — lieu de résidence."""
        for raw_loc in _SELF_LOCATION_RE.findall(text):
            cleaned = raw_loc.strip().rstrip(".,;!?").strip()
            if not cleaned or len(cleaned) < 2:
                continue
            if cleaned.lower() in _GENERIC_VALUES:
                continue
            normalized = self._normalize_proper(cleaned)
            action = self._upgrade_or_insert(raw, normalized, "location", 0.95)
            if action == "inserted":
                log.info("Self-location fallback: %r → location (new)", normalized)
            elif action == "upgraded":
                log.info("Self-location fallback: %r → location (upgraded case)", normalized)

    def _apply_self_employer(self, text: str, raw: list[dict[str, Any]]) -> None:
        """Pattern 4/5 — employeur."""
        for raw_emp in _SELF_EMPLOYER_RE.findall(text):
            cleaned = raw_emp.strip().rstrip(".,;!?").strip()
            if not cleaned or len(cleaned) < 2:
                continue
            cleaned_low = cleaned.lower()
            if cleaned_low in _EMPLOYER_BLOCKLIST:
                continue
            # V5.5.9 — Défense en profondeur. Si la capture commence par
            # un préfixe de la blocklist (ex : "moi en télétravail" qui
            # commence par "moi"), on rejette. Ça gère les cas où le
            # regex a englobé un complément circonstanciel inattendu
            # ("chez mes parents cet été"). On vérifie sur des frontières
            # de mots pour éviter les faux positifs ("Moihamed" ne doit
            # PAS être bloqué par "moi").
            if any(
                cleaned_low == bad
                or cleaned_low.startswith(bad + " ")
                or cleaned_low.startswith(bad + "\u00a0")  # nbsp
                for bad in _EMPLOYER_BLOCKLIST
            ):
                continue
            if cleaned_low in _GENERIC_VALUES:
                continue
            normalized = self._normalize_proper(cleaned)
            action = self._upgrade_or_insert(raw, normalized, "organization", 0.95)
            if action == "inserted":
                log.info("Self-employer fallback: %r → organization (new)", normalized)
            elif action == "upgraded":
                log.info("Self-employer fallback: %r → organization (upgraded case)", normalized)

    def _apply_self_age(self, text: str, raw: list[dict[str, Any]]) -> None:
        """Pattern 5/5 — âge en années."""
        for raw_age in _SELF_AGE_RE.findall(text):
            try:
                age = int(raw_age)
            except ValueError:
                continue
            if not (1 <= age <= 120):
                continue
            age_str = str(age)
            # Âge utilise un label custom "age" (hors GLINER_LABELS) —
            # le KG store accepte n'importe quel label arbitraire.
            if self._already_in(raw, age_str, "age"):
                continue
            raw.append({
                "text": age_str, "label": "age", "score": 0.95,
            })
            log.info("Self-age fallback: %d → age", age)

    def _apply_self_birth_year(
        self, text: str, raw: list[dict[str, Any]],
    ) -> None:
        """V5.5.3 — Pattern 6 : année de naissance ("je suis né en 1985").

        Stocké en label "date" (catégorie GLiNER existante) avec
        valeur sous forme "année:YYYY" pour distinguer d'une date
        arbitraire que GLiNER aurait extraite. Permet à Lythéa de
        calculer l'âge dynamiquement (année courante − année de
        naissance) au lieu d'invoquer un calculateur web.

        V5.5.4 — Dédup contre l'extraction brute GLiNER. Quand notre
        pattern enrichi capture "année:1985", on supprime aussi
        l'éventuelle entité "1985" tout court que GLiNER aurait
        ajoutée en parallèle — sinon on a un doublon dans le KG.
        """
        from datetime import datetime
        current_year = datetime.now().year

        for raw_year in _SELF_BIRTH_YEAR_RE.findall(text):
            try:
                year = int(raw_year)
            except ValueError:
                continue
            # Sanity : entre 1900 et l'année courante (pas de futur)
            if not (1900 <= year <= current_year):
                continue
            value = f"année:{year}"
            if self._already_in(raw, value, "date"):
                continue
            raw.append({
                "text": value, "label": "date", "score": 0.95,
            })
            log.info("Self-birth-year fallback: %d → date", year)

            # V5.5.4 — Dédup : retirer "1985" brut si GLiNER l'a
            # aussi extrait. On itère sur une copie pour pouvoir
            # supprimer en place.
            year_str = str(year)
            for ent in list(raw):
                if (
                    ent.get("text", "").strip() == year_str
                    and ent.get("label", "") == "date"
                    and ent.get("text") != value  # ne supprime pas notre nouvelle
                ):
                    raw.remove(ent)
                    log.info(
                        "Self-birth-year dedup: removed raw %s (kept année:%d)",
                        year_str, year,
                    )

            # On dérive aussi l'âge actuel pour qu'il soit directement
            # exploitable au prompt sans calcul ultérieur.
            derived_age = current_year - year
            if 1 <= derived_age <= 120:
                age_str = str(derived_age)
                if not self._already_in(raw, age_str, "age"):
                    raw.append({
                        "text": age_str, "label": "age", "score": 0.90,
                    })
                    log.info(
                        "Self-birth-year derived age: %d → age", derived_age,
                    )
