"""Web search via DuckDuckGo with iterative refinement.

Performs up to 3 rounds of search, stopping when results stabilize
(cosine delta between consecutive round summaries < threshold).
"""
from __future__ import annotations

import logging
import re

from rune.config import WEB_MAX_ROUNDS, WEB_STABILITY_THRESHOLD

log = logging.getLogger("rune.web")

# Temporal keywords that trigger web search (single words)
TEMPORAL_KEYWORDS_SINGLE = {
    "aujourd'hui", "récent", "récemment", "actuel", "actuellement",
    "maintenant", "hier", "today", "latest", "recent", "recently",
    "current", "currently", "now", "yesterday",
}
# Multi-word temporal phrases (checked via substring match)
TEMPORAL_PHRASES = [
    "cette semaine", "ce mois", "cette année", "en cours",
    "dernière nouvelle", "dernières nouvelles",
    "this week", "this month", "this year", "last week", "last month",
]
TEMPORAL_KEYWORDS = TEMPORAL_KEYWORDS_SINGLE  # kept for compat

# V5.6.16 — B1 fix : Détection de récit personnel.
#
# Symptôme corrigé : "Hier j'ai vu mon médecin" déclenchait une recherche
# web sur "hier", ramenait Reverso/dictionnaires et polluait 2K chars de
# contexte. Le mot temporel était DESCRIPTIF (raconter sa journée) et
# non FACTUEL (demander une info externe daté).
#
# Heuristique : si la phrase contient un marker de récit personnel
# (je + verbe passé/futur, possessif "mon/ma/mes"), les mots temporels
# sont probablement descriptifs → on inhibe le déclenchement web.
# Le rappel : "qu'est-ce qui s'est passé hier" reste factuel et passera.
_PERSONAL_NARRATIVE_PATTERNS_FR = [
    r"\bj['e]\s*(?:ai|vais|étais|serai|viens|me|suis)\b",       # j'ai, je vais, j'étais
    r"\bj['e]\s*ai\s+\w+",                                      # j'ai + n'importe quel verbe (trouvé, vu, eu, etc.)
    r"\bje\s+viens\s+de\s+\w+",                                 # je viens de + verbe
    r"\b(?:mon|ma|mes|notre|nos)\s+\w+",                        # mon médecin, ma fille
    r"\bje\s+(?:dois|peux|veux|devrais|vais)\b",               # je dois, je veux, je vais
    r"\bje\s+commence\b",                                        # je commence
    r"\bje\s+(?:me|m['e])\s*\w+",                               # je me + verbe (je me sens, je m'inquiète)
]
_PERSONAL_NARRATIVE_PATTERNS_EN = [
    r"\bi\s+(?:have|am|was|will|did|met|saw|got|found|took)\b",
    r"\bmy\s+\w+",
    r"\bi['']?(?:ve|m|d|ll)\s+(?:had|been|seen|met|found)\b",
]
_PERSONAL_NARRATIVE_RE = re.compile(
    "|".join(_PERSONAL_NARRATIVE_PATTERNS_FR + _PERSONAL_NARRATIVE_PATTERNS_EN),
    re.IGNORECASE,
)


def _is_personal_narrative(text: str) -> bool:
    """V5.6.16 — Détecte si la phrase est un récit personnel.

    Retourne True si on trouve des marqueurs de subjectivité forte
    (pronom je + verbe d'action, possessifs personnels). Dans ce cas
    les mots temporels ("hier", "demain") sont à interpréter comme
    descriptifs et non comme déclencheurs de recherche web.

    Exemples :
    - "Hier j'ai vu mon médecin" → True (récit personnel)
    - "Qu'est-ce qui s'est passé hier ?" → False (question factuelle)
    - "Demain je commence un traitement" → True (récit personnel)
    - "Quelles sont les news d'hier ?" → False (demande factuelle)
    """
    return bool(_PERSONAL_NARRATIVE_RE.search(text))


# Factual / event patterns that suggest web search is needed
_FACTUAL_PATTERNS_FR = [
    r"qui (?:a |est |sont |était |sera )",
    r"(?:quel|quelle|quels|quelles) (?:est|sont|était|étaient|sera) ",
    # V5.9.1 — "résultat" contextualisé pour éviter les faux positifs.
    # Avant : "un résultat propre" déclenchait web alors que c'était
    # juste une demande de stats Python. Maintenant le mot doit être
    # qualifié par un événement compétitif (match/élection/finale/etc.)
    # ou un déterminant qui précise (le/les/du) suivi du contexte sportif.
    r"(?:r[ée]sultats?\s+(?:du|de\s+la|des|d'|de\s+l'?)\s*"
    r"(?:match|finale|élection|coupe|tournoi|championnat|sondage|vote|gp|grand\s+prix))",
    r"(?:score|vainqueur|gagnant|champion|classement|palmarès)\s+(?:du|de|des|d'|en|au)",
    r"(?:a |ont )gagné\s+(?:le|la|les|contre|en)",
    r"(?:combien|quel prix|quel coût)\s+(?:coûte|fait|gagne|a\s+rapporté)",
    r"(?:est-ce que .+ existe|existe-t-il)",
    r"(?:où en est|qu'est-il arrivé|que s'est-il passé)",
    r"(?:dernières? nouvelles?|news|actu)",
    r"(?:a remporté|a obtenu|a décroché)",
]
_FACTUAL_PATTERNS_EN = [
    r"who (?:won|is|are|was|were|will) ",
    r"what (?:is|are|was|were|happened) ",
    # V5.9.1 — même contextualisation pour EN
    r"(?:score|winner|champion|ranking|standings)\s+(?:of|in|at|for)",
    r"results?\s+(?:of|from|for)\s+(?:the\s+)?(?:match|game|election|finals?|cup|tournament)",
    r"(?:how much|what price|how many)",
    r"(?:does .+ exist|is .+ still)",
    r"(?:latest news|what happened)",
]
_FACTUAL_RE = re.compile(
    "|".join(_FACTUAL_PATTERNS_FR + _FACTUAL_PATTERNS_EN),
    re.IGNORECASE,
)

# V5.6.1 — Math calculation pattern : présence d'au moins un opérateur
# arithmétique (× ÷ * / + - ^ %) entre deux nombres. On l'utilise pour
# court-circuiter le pattern "Combien" qui déclencherait à tort une
# recherche web sur un calcul. Python executor sera plus rapide et
# précis pour ces cas.
_MATH_CALC_RE = re.compile(
    r"\d+\s*[×÷*/+\-^%]\s*\d+"
    r"|\d+\s*(?:fois|plus|moins|divisé\s+par|multiplié\s+par)\s+\d+"
    r"|racine\s+(?:carrée|cubique)"
    r"|\^\s*\d+|\*\*\s*\d+",
    re.IGNORECASE,
)

# V5.8.1 — Détecteur d'ACRONYME TECHNIQUE inconnu.
#
# Cas du bug "RMSECV" : Lythéa classait la question en "none" (=je sais
# expliquer), Qwen2.5 n'avait pas une bonne représentation de cet
# acronyme chimiométrique en français, et a halluciné "organisation où
# travaille ton médecin" (par contamination du contexte SDM).
#
# Heuristique : un acronyme technique = 3-12 caractères majuscules
# (lettres + chiffres + tirets) qui n'est PAS dans une whitelist de
# mots courants (USA, OK, FAQ, etc.). Quand un tel acronyme apparaît
# dans une question "c'est quoi X ?" / "what is X ?" / "définis X" /
# "explique X", on force une recherche web pour ancrer la réponse.
#
# Sinon on continue d'avoir des hallucinations sur les concepts
# techniques spécialisés (PLS-DA, RMSECV, SNV, EMSC, GLMM, ANOVA, etc.).
_COMMON_ACRONYMS: frozenset[str] = frozenset({
    # Anglais courant
    "USA", "UK", "EU", "UN", "NATO", "OPEC", "OK", "OKAY",
    "FAQ", "DIY", "ASAP", "LOL", "FYI", "TBA", "TBD", "ETA",
    "AM", "PM", "AD", "BC", "BCE", "CE", "GMT", "UTC", "EST",
    "PST", "CET", "CEST", "PDT", "EDT",
    # Français courant
    "SNCF", "EDF", "RATP", "RTM", "CAF", "URSSAF", "INSEE",
    "CNIL", "ARS", "CPAM", "CPF", "RSI", "TVA", "RIB",
    # Tech grand public
    "URL", "HTTP", "HTTPS", "API", "JSON", "XML", "HTML", "CSS",
    "PDF", "PNG", "JPG", "JPEG", "GIF", "SVG", "MP3", "MP4",
    "CPU", "GPU", "RAM", "ROM", "USB", "HDMI", "WIFI", "LAN",
    "AI", "ML", "DL", "NLP", "GPT", "LLM", "AGI",
    # Sciences générales
    "ADN", "ARN", "DNA", "RNA", "PCR", "IRM", "MRI",
    # Mois/jours qu'on pourrait voir en majuscules
    "JANVIER", "FÉVRIER", "MARS", "AVRIL", "MAI", "JUIN",
})

_TECHNICAL_ACRONYM_RE = re.compile(
    r"\b([A-Z][A-Z0-9\-]{2,11})\b"
)

_DEFINITION_REQUEST_RE = re.compile(
    r"(?i)("
    r"c['e]?\s*est\s+quoi(?:\s+(?:le|la|les|l['e]?))?|"
    r"qu['e]?\s*est[\s\-]?ce\s+que(?:\s+(?:le|la|les|l['e]?))?|"
    r"d[ée]fini[stre]?[\s\-]?moi|qu['e]?\s*est[\s\-]?ce|"
    r"explique[\s\-]?moi(?:\s+(?:le|la|les|l['e]?))?|"
    r"que\s+(?:veut|signifie)\s+dire|que\s+signifie|"
    # V5.8.4 — "comment fonctionne X" / "comment marche X" sont
    # aussi des demandes de définition pour acronymes techniques.
    # Sans ça, "Comment fonctionne PLS-DA" passait en self_answer
    # et le LLM inventait sans recherche web.
    r"comment\s+(?:fonctionne|marche|opère|opere)|"
    r"comment\s+(?:ça|cela|on|tu)\s+(?:utilise|emploie|applique|fonctionne|marche|opère|opere)|"
    r"comment\s+(?:utiliser|appliquer|employer|implémenter)|"
    r"what\s+is(?:\s+(?:the|a|an))?|what\s+does\s+\S+\s+mean|"
    r"how\s+does\s+\S+\s+work|how\s+to\s+(?:use|apply|implement)|"
    r"define|explain(?:\s+(?:to\s+me)?)?|tell\s+me\s+about"
    r")\b"
)


def _looks_like_unknown_acronym(message: str) -> tuple[bool, str]:
    """V5.8.1 — Détecte une demande de définition d'acronyme technique.

    Retourne (True, acronyme) si :
      1. Le message contient un pattern "définis/c'est quoi/what is/etc"
      2. ET le message contient un acronyme 3-12 chars en majuscules
      3. ET cet acronyme n'est PAS dans la whitelist commune

    Cas typiques détectés : RMSECV, PLS-DA, SNV, EMSC, GLMM, ANOVA,
    PCA, ICA, t-SNE, UMAP, RAG, MoE, FFT, DSP, etc.

    Args:
        message: question utilisateur

    Returns:
        (force_web_search, matched_acronym)
    """
    if not message:
        return False, ""
    if not _DEFINITION_REQUEST_RE.search(message):
        return False, ""
    for m in _TECHNICAL_ACRONYM_RE.finditer(message):
        candidate = m.group(1).upper().strip("-")
        if not candidate:
            continue
        if candidate in _COMMON_ACRONYMS:
            continue
        # Filtres complémentaires : éviter les mots tout en majuscules
        # qui ne sont pas des acronymes (genre "OUI" "NON" "STOP")
        if len(candidate) <= 3 and candidate.isalpha():
            # Mots courts purement alphabétiques : on est plus strict
            # (un acronyme tech aura souvent des chiffres ou tirets)
            common_short = {"OUI", "NON", "JE", "TU", "IL", "ON", "VS",
                            "ET", "OU", "SI", "QUI", "QUE", "STOP",
                            "GO", "UP", "IN", "ON", "AT", "TO", "OF",
                            "BY", "AS", "IS", "BE", "DO", "MR", "MS",
                            "DR", "VS"}
            if candidate in common_short:
                continue
        return True, candidate
    return False, ""

# Year reference pattern (2020–2039)
_YEAR_RE = re.compile(r"\b20[2-3]\d\b")

# Keywords that suggest the model can answer alone (no web needed)
_SELF_ANSWER_RE = re.compile(
    r"(?:explique|défini|c'est quoi|qu'est-ce qu'?un|comment fonctionne|explain|define|what is a\b)",
    re.IGNORECASE,
)

# V4.4 — Patterns qui indiquent une question sur la MÉMOIRE INTERNE
# de l'IA, pas sur le monde extérieur. Le mot "récemment" dans
# "nos conversations récemment" ne doit PAS déclencher de recherche
# web. Ces marqueurs court-circuitent toute heuristique web (sauf
# /web explicite ou mode=always). Sans ça, l'IA cherche en ligne
# une réponse sur sa propre histoire, ne trouve rien, et confabule
# à partir des résultats parasites (traductions, blogs, etc.).
_MEMORY_QUESTION_RE = re.compile(
    r"(?:"
    r"tu te souviens|tu te rappelles|"
    r"te souviens-tu|te rappelles-tu|"
    r"tu m'?as (?:dit|expliqué|raconté|parlé|appris|envoyé)|"
    r"on (?:a |s'est |avait )(?:déjà )?(?:parlé|discuté|évoqué|abordé|vu)|"
    r"on en (?:a |avait )(?:déjà )?parlé|"
    r"notre (?:dernière )?(?:conversation|discussion|échange|chat)|"
    r"nos (?:dernières )?(?:conversations|discussions|échanges)|"
    r"dans (?:la|nos|notre|cette) (?:conversation|discussion|session|chat)|"
    r"plus (?:tôt|haut) (?:tu|on|nous)|"
    r"tout à l'heure (?:tu|on|nous)|"
    r"do you remember|you (?:told|said|mentioned)|"
    r"we (?:discussed|talked|spoke|covered)|"
    r"our (?:last |previous )?(?:conversation|chat|discussion)"
    r")",
    re.IGNORECASE,
)

# V4.4 — Patterns techniques : recommandation de package, modèle,
# librairie, API, framework. Sur ces questions les LLM 4B-7B
# confabulent fortement (inventent des noms plausibles type
# "bert-base-french", "xlm-roberta-french", "spacy-fr") car le pattern
# d'un nom HuggingFace ou pypi est très entraîné mais les noms réels
# ne sont pas systématiquement mémorisés. Plutôt que d'espérer que
# le prompt système suffise, on déclenche web par défaut — c'est ce
# que font Cursor, Perplexity, Continue, etc.
#
# V4.4 — Refactor en DEUX PASSES (verbe ET cible) pour éliminer les
# faux positifs comme "recommande-moi un mug" ou "conseille-moi un
# bon vin". Le matching exige désormais :
#   1. Un VERBE de recommandation (recommande/cite/propose/etc.) OU
#      une formulation technique explicite (comment installer X).
#   2. UNE CIBLE TECHNIQUE (modèle, lib, paper, API, framework, etc.).
# Si seul le verbe matche sans cible technique, pas de déclenchement.
# Quelques constructions (cite-moi un paper, comment installer X)
# sont auto-suffisantes car elles encodent déjà la cible.
_TECH_VERB_RE = re.compile(
    r"(?:"
    # Verbes de recommandation génériques (besoin d'une cible technique)
    r"recommande[s-]?[ -]?(?:moi|nous)?|"
    r"conseille[s-]?[ -]?(?:moi|nous)?|"
    r"propose[s-]?[ -]?(?:moi|nous)?|"
    r"suggère[s-]?[ -]?(?:moi|nous)?|"
    r"recommend|suggest|"
    # Verbes d'usage (besoin de cible tech pour matcher)
    r"comment utiliser|how (?:do|to|can) (?:I |you |we )?use"
    r")",
    re.IGNORECASE,
)

_TECH_TARGET_RE = re.compile(
    r"(?:"
    # Cibles techniques explicites
    r"modèle|model|"
    r"librairie|library|package|module|framework|paquet|lib|"
    r"api|sdk|"
    r"algorithme|algorithm|"
    r"outil|tool|"
    r"plugin|extension|"
    r"base de données|database|"
    r"langage de programmation|programming language|"
    # Marqueurs techniques sans ambiguïté
    r"python|javascript|typescript|rust|golang|java[^a-z]|c\+\+|"
    r"huggingface|hugging[ -]?face|pypi|npm|"
    r"transformer|llm|nlp|machine learning|deep learning|"
    r"ner|ocr|cnn|rnn|gan|"
    r"pré-?entraîné|pretrained|"
    # Tâches techniques (questions du genre "modèle pour X")
    r"reconnaissance d'entit|named entity|"
    r"classification|détection|generation|génération|embedding|tokeniz"
    r")",
    re.IGNORECASE,
)

# Patterns auto-suffisants qui n'ont pas besoin de cible additionnelle
# car la cible est déjà encodée dans la formulation.
_TECH_SELF_CONTAINED_RE = re.compile(
    r"(?:"
    # Demandes de citations/références académiques (cible = paper)
    r"cite[s-]?[ -]?(?:moi|nous)? (?:\d+ |des |les |un |une )?(?:papier|paper|article|publication|étude|study|référence)|"
    r"donne[s-]?[ -]?(?:moi|nous)? (?:\d+ |des |les |un |une )?(?:papier|paper|article|publication|étude|référence|bibliographie)|"
    r"(?:papier|paper|article|publication) (?:de référence|fondateur|fondamental|seminal|incontournable)|"
    r"références? (?:bibliographique|académique|scientifique)|"
    # Comment + verbe technique (installer/configurer/importer/etc.
    # indique forcément un contexte code, donc auto-suffisant).
    # Note : "utiliser/use" est délibérément exclu ici car trop large
    # ("comment utiliser ce verre" est non-tech). Il passe par la
    # pass 2 (verbe + cible) via _TECH_VERB_RE si besoin.
    r"comment (?:installer|configurer|importer|appeler|"
    r"intégrer|implémenter|coder|programmer)|"
    r"how (?:do|to|can) (?:I |you |we )?(?:install|configure|import|"
    r"call|integrate|implement|code)|"
    # Quel framework/lib/etc. (cible technique dans la question)
    r"which (?:library|package|model|framework|tool|lib|module|sdk|api)|"
    r"what (?:library|package|model|framework|tool|lib|module|sdk|api)|"
    r"quel(?:le|s|les)? (?:librairie|library|package|module|framework|modèle|outil|api|sdk)"
    r")",
    re.IGNORECASE,
)


def _is_tech_recommendation(message: str) -> tuple[bool, str | None]:
    """Détecte une recommandation technique avec logique 2-passes.

    Returns
    -------
    tuple[bool, str | None]
        (matched, matched_text). matched=True si :
        - un pattern auto-suffisant matche (cite-moi un paper, etc.), OU
        - un verbe de reco ET une cible technique sont présents.
    """
    # Pass 1 : self-contained (formulation qui encode déjà la cible)
    sc_match = _TECH_SELF_CONTAINED_RE.search(message)
    if sc_match:
        return True, sc_match.group().strip()

    # Pass 2 : verbe + cible (les deux nécessaires)
    verb_match = _TECH_VERB_RE.search(message)
    if not verb_match:
        return False, None
    target_match = _TECH_TARGET_RE.search(message)
    if not target_match:
        return False, None  # Verbe seul (genre "recommande-moi un mug") → pas tech
    return True, verb_match.group().strip()


class WebTriggerPolicy:
    """Decides whether a query needs web search.

    Parameters
    ----------
    mode : str
        One of ``"off"``, ``"auto"``, ``"always"``.
    """

    def __init__(self, mode: str = "auto") -> None:
        self.mode = mode

    def should_search(self, message: str) -> tuple[bool, str]:
        """Check if web search should be triggered.

        Returns
        -------
        tuple[bool, str]
            (should_search, reason)
        """
        if self.mode == "off":
            return False, ""

        if self.mode == "always":
            return True, "mode=always"

        # V4.4 — Manual override tags. /web force la recherche, /noweb
        # l'interdit pour ce tour. Utiliser regex avec word boundary
        # pour éviter de matcher "/website" comme "/web".
        # Symétrie : /noweb a priorité sur tout, y compris /web (si
        # l'utilisateur tape les deux par erreur, on respecte le
        # "no" — interprétation conservatrice qui évite l'appel
        # web non voulu).
        if re.search(r"(?<!\w)/noweb(?!\w)", message):
            return False, "manual /noweb tag"
        if re.search(r"(?<!\w)/web(?!\w)", message):
            return True, "manual /web tag"

        # V4.4 — Inhibition mémoire interne. Une question qui porte
        # sur ce que l'IA et l'utilisateur ont déjà discuté ne doit
        # pas déclencher de recherche web même si elle contient des
        # mots comme "récemment" ou des patterns factuels. Sans ça,
        # l'IA cherche en ligne une réponse sur sa propre histoire,
        # ne trouve évidemment rien de pertinent, et confabule à
        # partir des bruits de résultats (traductions, blogs, etc.).
        # Doit passer AVANT les checks temporels/factuels/année pour
        # les neutraliser.
        if _MEMORY_QUESTION_RE.search(message):
            return False, ""

        # V4.4 — Inhibition self-answer aussi devant les checks
        # temporels. Une question comme "explique-moi récemment comment
        # fonctionne X" contient le mot temporel "récemment" mais reste
        # une question conceptuelle pure ("explique-moi", "c'est quoi").
        # Sans cette inhibition, on déclenche du web inutile sur des
        # explications de concepts stables (transformer, backprop, NER).
        # Note : cette inhibition ne s'applique PAS aux questions
        # factuelles (factual/year_ref) qui restent traitées plus bas
        # avec leur propre filtre self-answer si nécessaire.
        #
        # V5.8.3 — Le check d'acronyme inconnu passe AVANT self-answer
        # parce qu'il représente un cas où "c'est quoi X" cible un
        # concept que le LLM ne peut PAS expliquer correctement (acronyme
        # technique spécialisé : RMSECV, PLS-DA, etc.). Sans ça, le LLM
        # invente. On force la recherche web pour ancrer la réponse.
        is_acronym, acronym = _looks_like_unknown_acronym(message)
        if is_acronym and not _MATH_CALC_RE.search(message):
            return True, f"acronym_definition: {acronym}"

        if _SELF_ANSWER_RE.search(message):
            return False, ""

        lower = message.lower()

        # V5.6.16 — B1 fix : si la phrase est un récit personnel,
        # les mots temporels sont descriptifs et non factuels.
        # On les ignore complètement pour éviter les recherches web
        # absurdes ("hier j'ai vu mon médecin" → Reverso/dictionnaires).
        is_personal = _is_personal_narrative(message)
        if is_personal:
            log.debug("Récit personnel détecté → mots temporels ignorés: %r", message[:60])

        # Multi-word temporal phrases (substring match)
        if not is_personal:
            for phrase in TEMPORAL_PHRASES:
                if phrase in lower:
                    return True, f"temporal: {phrase}"

        # Single-word temporal keywords
        if not is_personal:
            words = set(re.findall(r'\w+', lower))
            hits = words & TEMPORAL_KEYWORDS_SINGLE
            if hits:
                return True, f"temporal: {', '.join(hits)}"

        # Year reference (2020–2039) → likely needs current data
        year_match = _YEAR_RE.search(message)
        if year_match:
            return True, f"year_ref: {year_match.group()}"

        # V4.4 — Recommandations techniques (package, modèle, lib, API).
        # Cas où le LLM seul confabule fortement : il connaît le pattern
        # des noms (HuggingFace, pypi) mais invente des références
        # plausibles qui n'existent pas. On délègue au web par défaut,
        # comme Cursor/Perplexity. Placé AVANT le check factuel parce
        # que "Quel est le meilleur framework" matche aussi factual,
        # mais tech_reco est plus spécifique → meilleur libellé UI.
        # Logique 2-passes (verbe + cible) pour éviter les faux
        # positifs type "recommande-moi un mug".
        is_tech, tech_text = _is_tech_recommendation(message)
        if is_tech:
            if _SELF_ANSWER_RE.search(message):
                return False, ""
            return True, f"tech_reco: {tech_text}"

        # V5.8.1 — Le détecteur d'acronyme technique inconnu a été
        # remonté plus haut dans le pipeline (avant _SELF_ANSWER_RE)
        # pour qu'il ne soit pas court-circuité par "c'est quoi".

        # Factual / event question patterns
        fact_match = _FACTUAL_RE.search(message)
        if fact_match:
            # Skip pure definitional questions (model can answer those)
            if _SELF_ANSWER_RE.search(message):
                return False, ""
            # V5.6.1 — Skip si la question est en fait un calcul math
            # ("Combien font 17 × 23 + 89 ÷ 11 ?"). Sinon on déclenche
            # une recherche web alors que le routeur Python fera mieux
            # le job en local et plus vite. On détecte les opérateurs
            # arithmétiques + au moins deux nombres.
            if _MATH_CALC_RE.search(message):
                return False, ""
            return True, f"factual: {fact_match.group().strip()}"

        return False, ""

    @staticmethod
    def build_search_query(message: str, reason: str | None = None) -> str:
        """Extract a cleaner search query from a natural-language question.

        Strips question syntax and keeps meaningful terms. Optionally
        enriches the query based on the reason that triggered the
        search — for technical recommendations (``reason`` starts with
        ``tech_reco``), we append ``"python huggingface"`` to bias
        SearXNG towards code-related results and avoid noise like
        e-commerce or translation sites. Empirically: a query like
        "modèle NER en français" alone returned turmeric exporters
        because "NER" was parsed as a generic word; adding
        "python huggingface" routed it to real model hubs.
        """
        q = message.lower().strip()
        q = re.sub(r"/web\b", "", q)
        # Remove French question prefixes
        q = re.sub(
            r"^(?:qui |que |quel(?:le)?s? |qu'est-ce qu[ei] |est-ce qu[ei] |"
            r"où |comment |combien |pourquoi )",
            "", q,
        )
        # Remove trailing punctuation
        q = re.sub(r"[?!.]+$", "", q)
        q = q.strip()
        # If too short after cleanup, fall back to original
        if len(q) < 5:
            q = re.sub(r"[?!.]+$", "", message).strip()

        # V4.4 — enrichissement contextuel selon le motif de la recherche.
        # Pour les recommandations techniques, on biaise vers les sites
        # de référence (HuggingFace, pypi, github via "python", arxiv
        # pour les papiers académiques) afin d'éviter que SearXNG
        # retourne du e-commerce, de la traduction ou des blogs SEO.
        if reason and reason.startswith("tech_reco"):
            msg_lower = message.lower()
            # Détection : est-ce une demande de papier/citation académique ?
            is_academic = any(kw in msg_lower for kw in (
                "papier", "paper", "article", "publication", "étude",
                "référence", "study", "bibliographie",
            ))
            if is_academic:
                # Pour les citations académiques : arxiv en priorité
                if "arxiv" not in q:
                    q = f"{q} arxiv"
            else:
                # Pour les recommandations de packages/modèles : huggingface
                if "huggingface" not in q and "hugging face" not in q:
                    q = f"{q} python huggingface"

        return q


class WebAgent:
    """Iterative web search agent using pluggable providers.

    Provider selection
    ------------------
    By default uses :func:`rune.web_providers.get_default_provider`
    which returns a composite chain SearXNG → DDG (fallback). Override
    via env vars ``WEB_SEARCH_PROVIDER`` and ``SEARXNG_INSTANCE_URL``,
    or by passing a custom ``provider`` to the constructor.

    Parameters
    ----------
    max_rounds : int
        Maximum search iterations.
    stability_threshold : float
        Cosine delta below which search stops.
    provider : WebSearchProvider | None
        Custom provider (for tests). If None, uses the default chain.
    """

    def __init__(
        self,
        max_rounds: int = WEB_MAX_ROUNDS,
        stability_threshold: float = WEB_STABILITY_THRESHOLD,
        provider=None,
    ) -> None:
        self.max_rounds = max_rounds
        self.stability_threshold = stability_threshold
        self._provider = provider

    def _ensure_provider(self):
        """Lazy-init the default provider on first use."""
        if self._provider is not None:
            return self._provider
        try:
            from rune.web_providers import get_default_provider

            self._provider = get_default_provider()
        except Exception:
            log.exception("Failed to init default web provider")
            self._provider = None
        return self._provider

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Single-round search via the configured provider chain.

        Returns
        -------
        list[dict]
            Each dict: ``{title, body, href}``.
        """
        provider = self._ensure_provider()
        if provider is None:
            return []
        try:
            results = provider.search(query, max_results=max_results)
            return list(results) if results else []
        except Exception as exc:
            log.warning("Web search failed via %s: %s",
                        getattr(provider, "name", "?"), exc)
            return []

    def iterative_search(self, query: str, reason: str | None = None) -> str:
        """Multi-round search with stability check.

        Parameters
        ----------
        query
            Raw user question.
        reason
            Optional trigger reason from :meth:`WebTriggerPolicy.should_search`
            (e.g. ``"tech_reco: Recommande-moi"``). Used to enrich the
            search query so SearXNG returns relevant results — for
            technical recommendations we append ``"python huggingface"``
            to bias towards code-related sources.

        Returns
        -------
        str
            Concatenated search context, numbered ``[1]``, ``[2]``, …
            so the downstream LLM can produce citations like
            "selon [2], …". Each result block contains the title,
            URL, and body snippet, separated visually for clarity.

        Side effect
        -----------
        Sets ``self.last_sources`` to the list of source dicts (one per
        numbered reference), so the caller can expose them in the UI.
        Each dict has ``index`` (1-based), ``title``, ``url``, ``body``.
        Cleared at the start of each call.
        """
        # Reset sources at the start of every search so we never expose
        # stale data from a previous turn.
        self.last_sources: list[dict] = []

        # Optimize query for the search engine (strip question syntax,
        # enrich based on reason).
        clean_query = WebTriggerPolicy.build_search_query(query, reason=reason)
        log.info("Web search: raw=%r → optimized=%r", query[:80], clean_query[:80])

        # Collect raw result dicts across rounds (not just bodies).
        all_results: list[dict] = []
        prev_text = ""

        for round_idx in range(self.max_rounds):
            # Build query — round 0 uses clean_query, later rounds
            # append last bodies to drift the search topic.
            if round_idx == 0:
                search_q = clean_query
            else:
                # Bodies of the last 2 results for drift.
                last_bodies = " ".join(
                    r.get("body", "") for r in all_results[-2:]
                )
                search_q = f"{clean_query} {last_bodies}"

            results = self.search(search_q[:200])

            if not results:
                break

            round_text = " ".join(r.get("body", "") for r in results)
            all_results.extend(results)

            # Stability check (Jaccard word overlap)
            if prev_text and round_idx > 0:
                overlap = self._word_overlap(prev_text, round_text)
                if overlap > (1.0 - self.stability_threshold):
                    log.debug("Web search stabilized at round %d (overlap=%.2f)", round_idx, overlap)
                    break

            prev_text = round_text

        if not all_results:
            return ""

        # Deduplicate by URL (or by body if URL absent).
        seen: set[str] = set()
        unique: list[dict] = []
        for r in all_results:
            key = (r.get("href") or r.get("body", "")[:100] or "").strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(r)

        # Format with numbered references the LLM can cite.
        # Format:
        #   [1] Title of the page
        #       https://example.com/page
        #       Body snippet of the page (truncated)…
        #
        #   [2] ...
        lines: list[str] = []
        total_chars = 0
        max_total = 2000  # budget total chars across all references
        for i, r in enumerate(unique, start=1):
            title = (r.get("title") or "").strip() or "(sans titre)"
            url = (r.get("href") or "").strip()
            body = (r.get("body") or "").strip()
            # Truncate body so each reference stays bounded
            body_limit = max(120, (max_total - total_chars) // max(1, len(unique) - i + 1))
            if len(body) > body_limit:
                body = body[:body_limit].rstrip() + "…"
            block = f"[{i}] {title}"
            if url:
                block += f"\n    {url}"
            if body:
                block += f"\n    {body}"
            lines.append(block)
            # Mirror the same numbered reference into last_sources so
            # the UI can render a clickable bibliography under the
            # response. Keep title/url/body untruncated for the UI —
            # truncation only affects what goes to the LLM.
            self.last_sources.append({
                "index": i,
                "title": title,
                "url": url,
                "body": (r.get("body") or "").strip(),
            })
            total_chars += len(block)
            if total_chars >= max_total:
                break

        return "\n\n".join(lines)

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        """Jaccard similarity of word sets."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)
