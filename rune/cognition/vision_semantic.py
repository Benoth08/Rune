"""Vision Active — détecteur sémantique multilingue.

V5.7.1 — Migration depuis les regex FR/EN vers une détection par
embeddings multilingues. Utilise paraphrase-multilingual-MiniLM-L12-v2
(50+ langues supportées) déjà chargé pour le SemanticRouter.

Principe : au lieu de chercher des mots-clés ("que dit", "zoom sur"),
on compare l'embedding du message utilisateur à des PROTOTYPES
d'intentions précompilés (un embedding moyen par catégorie). Si la
similarité cosine dépasse un seuil, c'est un trigger.

Avantages vs regex :
  - Multilingue out-of-the-box (ES, DE, IT, ZH, JA, RU, etc.)
  - Robuste aux paraphrases et fautes
  - Capture l'intention au-delà des formes de surface
  - Moins de code à maintenir

Coût : ~15-30ms par message (encode du query). Le warmup amortit le
chargement initial des prototypes (~500ms une seule fois).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("lythea.cognition.vision_semantic")


# ──────────────────────────────────────────────────────────────────────
# Prototypes d'intention — chaque catégorie regroupe ~6-10 phrases
# exemples qui DÉCRIVENT l'intention. Les embeddings de ces phrases
# sont moyennés pour former un vecteur prototype par catégorie.
#
# Choix volontaire de mélanger les langues dans les prototypes pour
# que le centroïde soit linguistiquement "neutre" (mais sentence-bert
# multilingue le ferait probablement même avec une seule langue).
# ──────────────────────────────────────────────────────────────────────


# Catégorie 1 : DEMANDER À LIRE / VOIR du contenu sur une image
# Tous les cas où l'utilisateur veut connaître ce qui est écrit ou
# présent visuellement, sans nécessairement spécifier une zone.
#
# V5.7.1 (35 phrases) — Couverture FR/EN/ES/DE/IT/PT/NL + registres
# variés (familier, soutenu, abrégé) + structures syntaxiques diverses.
_PROTOTYPES_READ_VISUAL = [
    # FR — registres variés
    "Que dit ce texte ?",
    "Qu'est-ce qui est écrit ?",
    "Lis-moi ce qui est écrit",
    "Que vois-tu sur l'image ?",
    "Que montre cette photo ?",
    "Tu peux me lire ce qu'il y a ?",
    "Y a quoi marqué sur l'image ?",
    "Décris-moi ce que tu vois",
    "Quels sont les éléments visibles ?",
    "Identifie le contenu de l'image",
    # EN — registres variés
    "What does it say on the image?",
    "Read me the content of this image",
    "What's written on this?",
    "Can you read what's on the sign?",
    "Tell me what you see",
    "What do you see in the picture?",
    "Describe what's in the image",
    "What's shown in this photo?",
    "Can you make out the text?",
    "What does this picture contain?",
    # ES
    "¿Qué dice esta etiqueta?",
    "¿Qué se ve en la imagen?",
    "Léeme lo que está escrito",
    "¿Qué muestra esta foto?",
    # DE
    "Was steht auf dem Schild?",
    "Was siehst du auf dem Bild?",
    "Lies mir vor was geschrieben steht",
    "Was zeigt das Bild?",
    # IT
    "Cosa dice questo?",
    "Cosa vedi nell'immagine?",
    "Leggimi cosa c'è scritto",
    # PT
    "O que diz a imagem?",
    "O que você vê na foto?",
    # NL
    "Wat staat er op de afbeelding?",
    "Wat zie je in de foto?",
]

# Catégorie 2 : DEMANDER À ZOOMER / REGARDER UNE ZONE PRÉCISE
# Cas où l'utilisateur pointe explicitement une zone (haut, bas, etc.)
# ou un élément (étiquette, panneau, bouton).
#
# V5.7.1 (35 phrases) — Plus de combinaisons zone × élément + variations
# d'impératif/interrogatif + plusieurs langues.
_PROTOTYPES_ZOOM_REGION = [
    # FR — combinaisons zone+élément
    "Zoom sur l'étiquette rouge",
    "Regarde plus précisément en haut à droite",
    "Concentre-toi sur le bas de l'image",
    "Peux-tu lire le petit texte dans le coin ?",
    "Quel est le numéro sur l'étiquette en haut ?",
    "Détaille le panneau central",
    "Lis le titre en haut",
    "Que dit le bouton ?",
    "Que dit l'étiquette à gauche ?",
    "Lis-moi le texte du panneau en bas",
    "Regarde le numéro dans le coin droit",
    "Détaille le contenu du panneau de gauche",
    "Zoome sur le code-barres",
    "Que vois-tu en haut à gauche ?",
    "Peux-tu lire ce qui est écrit au centre ?",
    # EN — combinaisons
    "Zoom in on the red label",
    "Look closer at the top right corner",
    "Focus on the bottom of the image",
    "Can you read the small text in the corner?",
    "What's the number on the top label?",
    "Detail the central panel",
    "Read the title at the top",
    "What does the button say?",
    "What does the label on the left say?",
    "Zoom in on the barcode",
    "What's written in the upper right?",
    "Look at the bottom right corner",
    "Read me the text on the central sign",
    # ES
    "Acerca la etiqueta roja",
    "¿Qué dice el cartel del centro?",
    "Lee el texto en la esquina superior",
    # DE
    "Zoom auf das rote Etikett",
    "Was sagt das Schild in der Mitte?",
    # IT
    "Ingrandisci l'etichetta rossa",
    "Cosa dice il cartello al centro?",
    # PT
    "Aproxima a etiqueta vermelha",
    "Lê o texto no canto superior",
]

# Catégorie 3 : REVENIR À UNE IMAGE PRÉCÉDENTE
# Indices qu'on parle d'une image déjà envoyée, pas d'une nouvelle.
#
# V5.7.1 (30 phrases) — Plus de variations conversationnelles +
# multilingue. Inclut les références implicites ("dans la photo
# d'avant", "sur ce que tu as vu").
_PROTOTYPES_RETURN_TO_IMAGE = [
    # FR
    "Reviens à l'image",
    "Regarde encore l'image",
    "À propos de la photo précédente",
    "Concernant la première image",
    "Dans la dernière image",
    "Revenons à la photo",
    "Pour en revenir à l'image",
    "Au sujet de cette photo",
    "Dans la photo que je t'ai montrée",
    "Sur la photo précédente",
    "À propos de l'image",
    "Reviens à ce que tu as vu",
    # EN
    "Come back to the image",
    "Look at the picture again",
    "About the photo you saw",
    "Concerning the first image",
    "In the last picture",
    "Going back to the photo",
    "Regarding the image",
    "Looking at the previous photo",
    "In that picture you saw",
    "Back to the image",
    "About what you saw",
    "Concerning the previous image",
    # ES
    "Volvamos a la imagen",
    "Sobre la foto anterior",
    # DE
    "Zurück zum Bild",
    "Bezüglich des vorherigen Bildes",
    # IT
    "Torniamo all'immagine",
    "Riguardo alla foto precedente",
    # PT
    "Voltando à imagem",
]

# Catégorie 4 (V5.7.1) : CONTRE-MODÈLE — ce qui N'EST PAS une demande visuelle
#
# Pierre angulaire de la détection robuste. Le détecteur ne se contente
# plus de calculer la similarité à un prototype positif et de la
# comparer à un seuil absolu — il fait une décision RELATIVE :
#   trigger ⟺ sim(query, positive) > sim(query, negative) + marge
#
# Ce prototype contient des cas où le verbe "voir/lire/dire" est présent
# mais l'INTENTION est complètement différente :
#   - Demande d'opinion ("qu'en penses-tu ?")
#   - Demande d'explication technique ("explique-moi PLS")
#   - Conversation personnelle ("que fais-tu ce soir ?")
#   - Méta-conversation ("que voulais-tu dire ?")
#   - Demande temporelle ("quoi de neuf ?")
#   - Émotions / état ("comment tu te sens ?")
#   - Préférences ("que préfères-tu ?")
#   - Demande de définition ("c'est quoi X ?")
#
# Le centroïde de ce prototype représente le "non-visuel" sémantique.
_PROTOTYPES_NEGATIVE = [
    # FR — demandes d'opinion / d'explication
    "Qu'en penses-tu ?",
    "Quel est ton avis sur cette question ?",
    "Explique-moi ce concept",
    "Tu peux développer ton raisonnement ?",
    "C'est quoi exactement ?",
    "Que veux-tu dire par là ?",
    "Que voulais-tu dire ?",
    "Comment ça fonctionne ?",
    # FR — conversation personnelle
    "Comment ça va ?",
    "Que fais-tu ce soir ?",
    "Quelle est ta journée ?",
    "Que préfères-tu manger ?",
    "Décris-moi ta journée",
    "Comment te sens-tu ?",
    "Quoi de neuf ?",
    "Tu as bien dormi ?",
    # EN — opinion / explication
    "What do you think about this?",
    "What's your opinion on it?",
    "Explain this concept to me",
    "Can you elaborate?",
    "What do you mean exactly?",
    "What did you mean by that?",
    "How does it work?",
    "What is it about?",
    # EN — conversation
    "How are you doing?",
    "What are you doing tonight?",
    "What's your favorite food?",
    "Describe your day",
    "How do you feel?",
    "What's new?",
    # ES / DE / IT — couvrir multilingue négatif aussi
    "¿Qué piensas?",
    "¿Cómo estás?",
    "Was denkst du?",
    "Wie geht es dir?",
    "Cosa ne pensi?",
    "Come stai?",
]


# ──────────────────────────────────────────────────────────────────────
# Patterns lexicaux pour extraction des références SPATIALES / VISUELLES
#
# On garde des regex pour CETTE étape uniquement (extraction du hint),
# parce que la détection d'intention est faite par embedding, mais
# la composition du prompt VLM bénéficie d'un hint textuel court et
# précis qu'on extrait du message original.
#
# Ces regex sont multilingues-light (FR + EN + ES basiques) et servent
# UNIQUEMENT à extraire un fragment lisible pour le prompt VLM.
# Si rien ne matche, on passe le message complet (nettoyé) au VLM.
# ──────────────────────────────────────────────────────────────────────

_SPATIAL_HINTS = re.compile(
    r"(?i)\b("
    # FR
    r"en\s+haut(?:\s+(?:[àa]\s+)?(?:gauche|droite))?|"
    r"en\s+bas(?:\s+(?:[àa]\s+)?(?:gauche|droite))?|"
    r"[àa]\s+(?:gauche|droite)|au\s+(?:milieu|centre)|"
    r"coin\s+\w+(?:[\s\-]\w+)?|"
    r"c[ôo]t[ée]\s+(?:gauche|droit)|"
    r"partie\s+(?:haute|basse|gauche|droite|centrale)|"
    # EN
    r"top(?:[\s\-](?:left|right))?|bottom(?:[\s\-](?:left|right))?|"
    r"upper[\s\-]?\w+|lower[\s\-]?\w+|"
    r"left(?:[\s\-]side)?|right(?:[\s\-]side)?|"
    r"center|middle|corner|header|footer|"
    # ES
    r"arriba(?:\s+[a-z]+)?|abajo(?:\s+[a-z]+)?|"
    r"izquierda|derecha|centro|esquina"
    r")\b"
)

_VISUAL_OBJECTS = re.compile(
    r"(?i)\b("
    # FR (singulier + pluriel)
    r"[ée]tiquettes?(?:\s+\w+)?|panneaux?(?:\s+\w+)?|"
    r"titres?|noms?|prix|num[ée]ros?|boutons?|champs?|logos?|"
    # "code" uniquement avec qualificatif visuel (barres, QR, couleur)
    r"codes?[\s\-]barres?|code[\s\-]qr|qr[\s\-]?codes?|"
    r"texte\s+(?:rouge|bleu|vert|jaune|noir|blanc|orange|violet|rose|gris)|"
    # EN
    r"labels?(?:\s+\w+)?|signs?(?:post)?|"
    r"titles?|names?|prices?|numbers?|buttons?|fields?|barcodes?|qr[\s\-]?codes?|"
    r"(?:red|blue|green|yellow|black|white|orange|purple|pink|gray|grey)\s+(?:text|label|button|sign)|"
    # ES
    r"etiquetas?|carteles?|t[íi]tulos?|"
    # DE
    r"schild(?:er)?|aufkleber|titel"
    r")\b"
)

# V5.7.1 — Garde-fous anti-faux-positifs pour le mode dégradé
#
# Patterns qui INHIBENT le déclenchement même si visual_refs ou
# has_visual_verb matchent. Couvre :
#   - Affirmations sur un objet visuel ("Je vois un panneau")
#   - Expressions idiomatiques ("tu vois ce que je veux dire")
#   - Demandes d'opinion qui contiennent "voir" ("que penses-tu de
#     ce que tu vois ?")
#
# Ces patterns sont conservateurs : ils visent les cas évidents,
# le détecteur sémantique en prod fera le reste.
_NON_VISUAL_INHIBITORS = re.compile(
    r"(?i)("
    # Expressions idiomatiques "tu vois / je vois (que/ce que)"
    r"tu\s+vois\s+(?:ce\s+que|que)\b|"
    r"vous\s+voyez\s+(?:ce\s+que|que)\b|"
    r"you\s+see\s+what\s+i\s+mean|"
    r"see\s+what\s+i\s+mean|"
    # Affirmations "je vois X" / "I see X" (déclaration, pas question)
    r"^\s*(?:je|j['e])\s*vois\s+(?:un|une|des|le|la|les|mon|ma|mes)\s+\w+|"
    r"^\s*i\s+(?:see|saw)\s+(?:a|an|the|some|my)\s+\w+|"
    # Affirmations sur attribut ("X est rouge", "X is red")
    r"\b\w+\s+(?:est|sont|était|étaient)\s+(?:rouge|bleu|vert|jaune|noir|blanc|orange|violet|rose|gris)|"
    r"\b\w+\s+(?:is|are|was|were)\s+(?:red|blue|green|yellow|black|white|orange|purple|pink|gray|grey)|"
    # Demandes d'opinion explicites (sauf opinion sur l'image elle-même)
    r"que\s+penses[\s\-]?tu(?!\s+de\s+(?:cette\s+)?(?:image|photo|illustration|schéma|graphique))\b|"
    r"qu['e]?\s*en\s+penses[\s\-]?tu|"
    r"quel\s+est\s+ton\s+avis(?!\s+sur\s+(?:cette\s+)?(?:image|photo|illustration|schéma|graphique))|"
    r"ton\s+avis\s+sur(?!\s+(?:cette\s+)?(?:image|photo|illustration|schéma|graphique))|"
    r"what\s+do\s+you\s+think(?!\s+(?:of|about)\s+(?:this\s+|the\s+)?(?:image|picture|photo))|"
    r"what's\s+your\s+opinion(?!\s+(?:of|about)\s+(?:this\s+|the\s+)?(?:image|picture|photo))|"
    # Demande d'explication technique générique
    r"explique[\s\-]?moi|"
    r"explain\s+(?:to\s+me|this|that)|"
    # "Décris-moi ta journée/vie/etc" (déclaratif non-visuel)
    r"d[ée]cri[stre]?[\s\-]?moi\s+(?:ta|ton|tes|votre|vos)\s+(?:journ[ée]e|vie|exp[ée]rience|projet|histoire|enfance|jeunesse|carri[èe]re)"
    r")"
)


def _is_inhibited(message: str) -> bool:
    """V5.7.1 — True si le message matche un pattern d'inhibition.

    Utilisé en mode dégradé pour éviter les faux positifs évidents
    (le détecteur sémantique en prod gère ça via le contre-modèle).
    """
    return bool(_NON_VISUAL_INHIBITORS.search(message))

# Préfixes conversationnels à NETTOYER avant d'envoyer le hint au VLM
# (V5.7.1 fix bug 3 : "Reviens à l'image, que dit ..." → on retire la
# partie "Reviens à l'image" du hint).
_CONVERSATIONAL_PREFIXES = re.compile(
    r"(?i)^\s*(?:"
    r"reviens?[\s\-]?(?:[àa]\s+l['e]?\s*image|to\s+the\s+image)|"
    r"regarde[\s\-]?(?:l['e]?\s*image|the\s+image)|"
    r"come\s+back\s+to\s+the\s+image|"
    r"look\s+at\s+(?:the\s+)?(?:image|picture|photo)|"
    r"au\s+sujet\s+de\s+l['e]?\s*image|"
    r"concernant\s+(?:l['e]?\s*image|la\s+photo)|"
    r"dans\s+(?:l['e]?\s*image|la\s+photo|cette\s+image)|"
    r"sur\s+(?:l['e]?\s*image|la\s+photo|cette\s+image)|"
    r"in\s+the\s+(?:image|picture|photo)|"
    r"on\s+the\s+(?:image|picture|photo)"
    r")[\s,\.:;]*"
)

# Marqueurs d'INCERTITUDE PERCEPTIVE pour la détection post-génération
# (gardé en regex multilingue car c'est appliqué à la sortie du LLM
# qui est plus structurée et utilise des tournures stéréotypées).
_PERCEPTUAL_UNCERTAINTY = re.compile(
    r"(?i)\b(?:"
    # FR
    r"je\s+(?:ne\s+)?(?:distingue|vois|arrive\s+(?:pas\s+)?[àa]\s+(?:lire|voir|distinguer))\s+(?:pas|mal)|"
    r"(?:il\s+)?(?:est\s+)?difficile\s+(?:de\s+|à\s+)?(?:lire|voir|distinguer|d[ée]chiffrer)|"
    r"peu\s+lisible|pas\s+(?:tr[èe]s\s+)?clair|flou(?:e)?|"
    r"il\s+semble(?:\s+que)?|il\s+(?:para[îi]t|appara[îi]t)|"
    r"(?:je\s+)?(?:ne\s+)?(?:suis\s+)?pas\s+s[ûu]r(?:\s+de\s+lire)?|"
    r"(?:image|photo)\s+(?:floue|de\s+mauvaise\s+qualit[ée])|"
    # EN
    r"(?:i\s+)?(?:can't|cannot|couldn't)\s+(?:quite\s+)?(?:tell|see|read|make\s+out)|"
    r"hard\s+to\s+(?:read|see|tell|make\s+out)|difficult\s+to\s+(?:read|see)|"
    r"unclear|blurry|fuzzy|"
    r"it\s+(?:seems|appears|looks)|"
    r"(?:i'm\s+)?not\s+(?:quite\s+)?sure|"
    r"image\s+(?:is\s+)?(?:blurry|unclear)"
    r")\b"
)


# ──────────────────────────────────────────────────────────────────────
# Result data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ZoomTrigger:
    """Résultat de l'analyse sémantique du message utilisateur.

    triggered : True si on doit déclencher un zoom VLM
    region_hint : description courte de la zone à regarder (pour prompt)
    confidence : similarité cosine au meilleur prototype POSITIF (0-1)
    negative_sim : similarité cosine au prototype NÉGATIF (V5.7.1)
    margin : confidence - negative_sim (positif = bon signe)
    category : laquelle des 3 intentions a matché ('read', 'zoom', 'return')
    is_soft_trigger : True si trigger basé uniquement sur une référence
        spatiale/visuelle (pas d'intention explicite de lecture). Force
        un zoom prudent.
    fallback_reason : si triggered=False, raison ('no_image', 'low_sim',
        'no_reference', 'negative_too_close')
    """
    triggered: bool
    region_hint: str = ""
    confidence: float = 0.0
    negative_sim: float = 0.0
    margin: float = 0.0
    category: str = ""
    is_soft_trigger: bool = False
    fallback_reason: str = ""
    matched_spatial: list[str] = field(default_factory=list)
    matched_visual: list[str] = field(default_factory=list)

    def to_debug(self) -> dict:
        return {
            "triggered": self.triggered,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "negative_sim": round(self.negative_sim, 3),
            "margin": round(self.margin, 3),
            "region_hint": self.region_hint[:80],
            "is_soft": self.is_soft_trigger,
            "fallback_reason": self.fallback_reason or None,
            "spatial_refs": self.matched_spatial,
            "visual_refs": self.matched_visual,
        }


# ──────────────────────────────────────────────────────────────────────
# SemanticVisionDetector — le détecteur principal
# ──────────────────────────────────────────────────────────────────────


class SemanticVisionDetector:
    """Détecteur d'intention de zoom basé sur embeddings multilingues.

    Singleton-friendly : le modèle d'embedding est chargé une fois,
    les prototypes sont précomputés au warmup.

    Seuils ajustables via config :
      - threshold_strict : intention claire de lecture/zoom (auto-trigger)
      - threshold_soft : seuil bas, doit être couplé à une référence
        spatiale/visuelle explicite pour déclencher
    """

    _PRIMARY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        threshold_strict: float = 0.50,
        threshold_soft: float = 0.38,
        negative_margin: float = 0.06,
    ):
        """V5.7.1 — Détecteur sémantique avec contre-modèle.

        Args:
            threshold_strict: similarité minimale au prototype positif
                pour un trigger automatique (auto)
            threshold_soft: similarité minimale pour un trigger souple
                (doit être couplé à une référence spatiale/visuelle)
            negative_margin: la similarité positive doit dépasser la
                similarité négative d'au moins cette marge. C'est ce
                qui fait la décision RELATIVE et élimine les faux
                positifs comme "Décris-moi ta journée" qui ressemble
                superficiellement à "Décris l'image" mais s'en éloigne
                sémantiquement.

        Seuils calibrés empiriquement (cf. tests V5.7.1) :
          - strict abaissé 0.55 → 0.50 car le contre-modèle filtre déjà
          - soft abaissé 0.40 → 0.38 idem
          - marge 0.06 mesurée comme la séparation minimale entre les
            vrais positifs et les faux positifs proches.
        """
        self.threshold_strict = threshold_strict
        self.threshold_soft = threshold_soft
        self.negative_margin = negative_margin
        self._model: Any = None
        self._model_name: str | None = None
        self._lock = threading.Lock()
        # Embeddings prototypes : {category_name: numpy array shape (D,)}
        self._prototypes: dict[str, Any] = {}
        self._warmed = False

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load le modèle d'embedding. Idempotent + thread-safe."""
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._PRIMARY_MODEL, device="cpu",
                )
                self._model_name = self._PRIMARY_MODEL
                log.info("Vision semantic detector loaded: %s", self._PRIMARY_MODEL)
                return True
            except Exception as exc:
                log.warning(
                    "Primary embedding %s failed (%s), trying fallback",
                    self._PRIMARY_MODEL, exc,
                )
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._FALLBACK_MODEL, device="cpu",
                )
                self._model_name = self._FALLBACK_MODEL
                log.info("Vision semantic detector loaded (fallback): %s",
                         self._FALLBACK_MODEL)
                return True
            except Exception as exc:
                log.error("Both embedding models failed: %s", exc)
                return False

    def warm_up(self) -> bool:
        """Précompute les prototypes d'intention (positifs + négatif).

        À appeler une fois au boot (sera fait lazily au 1er trigger
        sinon). Coût : ~1-2s pour encoder ~130 exemples.
        """
        if self._warmed:
            return True
        if not self._ensure_model_loaded():
            return False
        with self._lock:
            if self._warmed:
                return True
            try:
                import numpy as np
                # V5.7.1 — 3 catégories positives + 1 négative (contre-modèle).
                # La décision finale est RELATIVE : trigger uniquement si
                # similarité à un prototype positif > similarité au négatif.
                categories = {
                    "read": _PROTOTYPES_READ_VISUAL,
                    "zoom": _PROTOTYPES_ZOOM_REGION,
                    "return": _PROTOTYPES_RETURN_TO_IMAGE,
                    "_negative": _PROTOTYPES_NEGATIVE,
                }
                for name, examples in categories.items():
                    embs = self._model.encode(
                        examples,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    # Prototype = moyenne des exemples (normalisée)
                    proto = embs.mean(axis=0)
                    norm = np.linalg.norm(proto)
                    if norm > 0:
                        proto = proto / norm
                    self._prototypes[name] = proto
                self._warmed = True
                log.info(
                    "Vision semantic detector warmed up: %d positive + 1 negative "
                    "(%d examples total)",
                    len(self._prototypes) - 1,
                    sum(len(e) for e in categories.values()),
                )
                return True
            except Exception as exc:
                log.error("Vision detector warmup failed: %s", exc, exc_info=True)
                return False

    # ── Détection principale ──────────────────────────────────────────

    def detect(self, message: str, has_image_in_vwm: bool) -> ZoomTrigger:
        """Analyse sémantique d'un message pour décider du zoom.

        Stratégie de décision :
          1. Pas d'image en buffer → pas de zoom (peu importe l'intention)
          2. Encode le message, compare aux 3 prototypes (read/zoom/return)
          3. Si meilleure sim > threshold_strict → trigger STRICT (auto)
          4. Sinon, si meilleure sim > threshold_soft ET référence spatiale
             ou visuelle trouvée → trigger SOFT (zoom prudent)
          5. Sinon → pas de zoom

        Args:
            message: message texte utilisateur
            has_image_in_vwm: True si au moins une image dans VWM

        Returns:
            ZoomTrigger avec décision + metadata
        """
        if not message or not message.strip():
            return ZoomTrigger(triggered=False, fallback_reason="empty_message")

        if not has_image_in_vwm:
            return ZoomTrigger(triggered=False, fallback_reason="no_image")

        # V5.7.1 — Garde-fou anti-faux-positifs (mode dégradé ET sémantique)
        # Inhibe les patterns "déclaratifs" ou "idiomatiques" qui contiennent
        # des mots visuels mais ne sont pas des demandes de zoom.
        if _is_inhibited(message):
            log.debug("Vision detect inhibited: %r", message[:60])
            return ZoomTrigger(
                triggered=False,
                fallback_reason="inhibited_pattern",
            )

        # Toujours extraire les références spatiales/visuelles
        spatial_refs = [m.group(1) for m in _SPATIAL_HINTS.finditer(message)]
        visual_refs = [m.group(1) for m in _VISUAL_OBJECTS.finditer(message)]

        # V5.7.1 — Verbes interrogatifs visuels (signal de soft trigger
        # en mode dégradé sans embeddings). Couvre "que vois-tu", "que
        # dit", "lis", "what do you see", "read", etc.
        has_visual_verb = bool(re.search(
            r"(?i)\b(?:"
            r"vois[\s\-]?(?:tu|on|nous)?|voyez[\s\-]?vous|voit[\s\-]?on|"
            r"que\s+dit|qu['e]?\s*est[\s\-]ce\s+qu['e]?\s*il\s+y\s+a|"
            r"lis[\s\-]?(?:moi)?|lit\s+moi|"
            r"d[ée]tail(?:l?es?|le[\s\-]?moi)?|"
            r"zoom(?:e)?|"
            r"regarde(?:[\s\-]plus)?|"
            r"montre[\s\-]?moi|"
            r"décri(?:s|re|t)?[\s\-]?(?:moi)?|"
            r"(?:do\s+you|can\s+you)\s+see|what\s+do\s+you\s+see|"
            r"what\s+does\s+(?:it|\S+\s+\S+|the\s+\w+)\s+say|"
            r"read(?:\s+me|\s+the|\s+out)?|tell\s+me\s+what|"
            r"show\s+me|describe|look\s+(?:at|closer)"
            r")\b",
            message,
        ))

        # Fallback gracieux si le modèle d'embedding n'est pas dispo
        if not self._ensure_model_loaded() or not self._warmed:
            # V5.7.1 — En mode dégradé, on tombe sur une logique
            # lexicale enrichie : trigger si UN des trois signaux est
            # présent (verbe visuel, référence spatiale, référence
            # visuelle). C'est plus permissif que V5.7.0 mais ça réduit
            # les faux négatifs (bug 4 "que vois-tu sur le côté gauche").
            if has_visual_verb or spatial_refs or visual_refs:
                hint = _extract_region_hint(message, spatial_refs + visual_refs)
                return ZoomTrigger(
                    triggered=True,
                    region_hint=hint,
                    confidence=0.0,
                    category="fallback_lexical",
                    is_soft_trigger=True,
                    matched_spatial=spatial_refs,
                    matched_visual=visual_refs,
                )
            return ZoomTrigger(triggered=False, fallback_reason="no_embedding_model")

        try:
            import numpy as np
            # Nettoyage du préfixe conversationnel pour l'encodage
            # (V5.7.1 — l'encodage de "Reviens à l'image, ..." pollue le
            # signal sémantique avec "image", on garde la partie utile).
            cleaned = _CONVERSATIONAL_PREFIXES.sub("", message).strip()
            if not cleaned:
                cleaned = message  # tout était préfixe → on encode brut

            q_emb = self._model.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            sims = {
                name: float(np.dot(q_emb, proto))
                for name, proto in self._prototypes.items()
            }
            # V5.7.1 — Décision relative au contre-modèle.
            # Sépare les prototypes positifs du négatif.
            neg_sim = sims.pop("_negative", 0.0)
            best_cat = max(sims, key=sims.get)
            best_sim = sims[best_cat]

            # Marge : positive doit dépasser le négatif d'au moins X
            margin_ok = (best_sim - neg_sim) >= self.negative_margin

            log.debug(
                "Vision semantic: pos_sims=%s neg=%.3f best=%s @ %.3f "
                "margin_ok=%s cleaned=%r",
                {k: round(v, 3) for k, v in sims.items()},
                neg_sim, best_cat, best_sim, margin_ok, cleaned[:60],
            )

            # Trigger STRICT : haute confiance + marge négative satisfaite
            if best_sim >= self.threshold_strict and margin_ok:
                hint = _extract_region_hint(
                    message, spatial_refs + visual_refs,
                )
                return ZoomTrigger(
                    triggered=True,
                    region_hint=hint,
                    confidence=best_sim,
                    negative_sim=neg_sim,
                    margin=best_sim - neg_sim,
                    category=best_cat,
                    is_soft_trigger=False,
                    matched_spatial=spatial_refs,
                    matched_visual=visual_refs,
                )

            # Trigger SOFT : confiance moyenne + marge négative + référence
            if (
                best_sim >= self.threshold_soft
                and margin_ok
                and (spatial_refs or visual_refs or has_visual_verb)
            ):
                hint = _extract_region_hint(
                    message, spatial_refs + visual_refs,
                )
                return ZoomTrigger(
                    triggered=True,
                    region_hint=hint,
                    confidence=best_sim,
                    negative_sim=neg_sim,
                    margin=best_sim - neg_sim,
                    category=best_cat,
                    is_soft_trigger=True,
                    matched_spatial=spatial_refs,
                    matched_visual=visual_refs,
                )

            # Pas de trigger — raison précise pour debug
            if not margin_ok:
                reason = f"negative_too_close (pos={best_sim:.2f} neg={neg_sim:.2f})"
            elif best_sim < self.threshold_soft:
                reason = "low_sim"
            else:
                reason = "no_reference"

            return ZoomTrigger(
                triggered=False,
                confidence=best_sim,
                negative_sim=neg_sim,
                margin=best_sim - neg_sim,
                category=best_cat,
                fallback_reason=reason,
                matched_spatial=spatial_refs,
                matched_visual=visual_refs,
            )

        except Exception as exc:
            log.warning("Vision semantic detect failed: %s", exc)
            # Fallback ultime : lexical pur
            if spatial_refs or visual_refs:
                hint = _extract_region_hint(message, spatial_refs + visual_refs)
                return ZoomTrigger(
                    triggered=True,
                    region_hint=hint,
                    is_soft_trigger=True,
                    category="fallback_error",
                    matched_spatial=spatial_refs,
                    matched_visual=visual_refs,
                )
            return ZoomTrigger(triggered=False, fallback_reason="detect_error")


# ──────────────────────────────────────────────────────────────────────
# Extraction propre du region_hint (V5.7.1 fix bug 3)
# ──────────────────────────────────────────────────────────────────────


def _extract_region_hint(message: str, refs: list[str]) -> str:
    """Extrait une description courte et propre de la zone à regarder.

    V5.7.1 — Améliorations par rapport à V5.7.0 :
      1. Retire les préfixes conversationnels ("Reviens à l'image, ...")
      2. Si plusieurs refs, prend les 2 plus longues + leur voisinage
         immédiat (3-4 mots autour)
      3. Limite à 60 chars pour éviter de polluer le prompt VLM

    Args:
        message: message utilisateur brut
        refs: liste des références spatiales + visuelles matchées

    Returns:
        Hint nettoyé, court, ciblé sur la zone uniquement
    """
    if not refs:
        # Pas de référence explicite, on prend juste le message nettoyé
        cleaned = _CONVERSATIONAL_PREFIXES.sub("", message).strip()
        return cleaned[:80]

    # Tri par longueur décroissante : les refs longues sont plus informatives
    refs_sorted = sorted(set(refs), key=len, reverse=True)
    top_refs = refs_sorted[:2]

    # On essaie de localiser les refs dans le message et de prendre un
    # petit voisinage pour avoir le contexte (qualificatifs : "rouge",
    # "petit", "en haut", etc.)
    fragments: list[str] = []
    msg_lower = message.lower()
    for ref in top_refs:
        pos = msg_lower.find(ref.lower())
        if pos == -1:
            fragments.append(ref)
            continue
        # Voisinage : ~15 chars avant + ref + 15 chars après
        start = max(0, pos - 15)
        end = min(len(message), pos + len(ref) + 15)
        snippet = message[start:end]
        # Coupe aux limites de mots
        snippet = re.sub(r"^\S*\s+", "", snippet).strip()  # premier mot partiel
        snippet = re.sub(r"\s+\S*$", "", snippet).strip()  # dernier mot partiel
        # Retire préfixes conversationnels et ponctuation
        snippet = _CONVERSATIONAL_PREFIXES.sub("", snippet).strip()
        snippet = snippet.strip(".,!?;:()[]\"'")
        if snippet:
            fragments.append(snippet)

    if not fragments:
        return top_refs[0] if top_refs else ""

    # Combine les fragments (dédup naïve)
    seen: set[str] = set()
    unique_frags: list[str] = []
    for f in fragments:
        f_low = f.lower()
        if f_low not in seen:
            seen.add(f_low)
            unique_frags.append(f)

    hint = " / ".join(unique_frags)
    return hint[:80]


# ──────────────────────────────────────────────────────────────────────
# API publique simplifiée
# ──────────────────────────────────────────────────────────────────────


# Singleton paresseux — instancié à la 1ère utilisation
_GLOBAL_DETECTOR: SemanticVisionDetector | None = None
_GLOBAL_LOCK = threading.Lock()


def get_detector() -> SemanticVisionDetector:
    """Récupère le détecteur global (singleton)."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_DETECTOR is None:
                _GLOBAL_DETECTOR = SemanticVisionDetector()
    return _GLOBAL_DETECTOR


def detect_zoom_trigger(message: str, has_image_context: bool) -> ZoomTrigger:
    """API publique : remplace l'ancien detect_zoom_trigger regex.

    Délègue au SemanticVisionDetector global. Multilingue.
    """
    return get_detector().detect(message, has_image_context)


def detect_perceptual_uncertainty(response_text: str) -> bool:
    """Détecte les marqueurs d'incertitude perceptive dans la réponse.

    Inchangé depuis V5.7.0 (regex multilingue light suffisante car
    on cherche des tournures stéréotypées de la sortie LLM).
    """
    if not response_text:
        return False
    return bool(_PERCEPTUAL_UNCERTAINTY.search(response_text))


def build_zoom_prompt(region_hint: str, user_query: str) -> str:
    """Construit le prompt VLM ciblé. Inchangé depuis V5.7.0."""
    hint = (region_hint or "").strip().strip("\"'")
    query = (user_query or "").strip().strip("\"'")

    if not hint and not query:
        return (
            "Describe this image in detail, focusing on any text, "
            "labels, or readable content."
        )

    parts = ["Look at this image carefully."]
    if hint:
        parts.append(f"Focus your attention specifically on: {hint}.")
    if query:
        parts.append(f'The user is asking: "{query}".')
    parts.extend([
        "Provide a precise, focused answer about this specific region only.",
        "If you can read text, report it verbatim.",
        "If you see specific values, numbers, or labels, report them exactly.",
        "Do not describe the rest of the image unless directly relevant.",
    ])
    return " ".join(parts)


def format_zoom_block(region_hint: str, vlm_output: str) -> str:
    """Formate le bloc Vision zoom pour injection dans system_text."""
    hint = (region_hint or "zone non spécifiée").strip()
    output = (vlm_output or "(aucune observation)").strip()
    return (
        f"[Vision zoom — région: {hint}]\n"
        f"Contenu observé : {output}"
    )


def build_visual_warning_block(image_caption: str) -> str:
    """V5.7.1 — Garde-fou anti-hallucination (fix bugs 1, 2, 6).

    Quand une image est dans le VWM et que l'utilisateur fait référence
    à du contenu visuel SANS qu'un zoom n'ait été déclenché, on injecte
    ce bloc dans le system_text pour rappeler au LLM de ne pas inventer.

    Args:
        image_caption: caption initial de l'image dans VWM

    Returns:
        Bloc texte à injecter dans le system prompt
    """
    cap = (image_caption or "").strip()[:300]
    return (
        "[Mémoire visuelle — avertissement]\n"
        "Une image est présente dans la mémoire visuelle de travail. "
        "L'utilisateur fait référence à son contenu, mais aucun zoom "
        "ciblé n'a été effectué sur cette demande précise.\n"
        f"Caption initial disponible : {cap}\n"
        "⚠️ Ne cite QUE les éléments visibles dans le caption ci-dessus. "
        "Si l'utilisateur demande un détail que tu ne peux pas confirmer "
        "depuis le caption, dis-le explicitement et propose un zoom — "
        "n'invente PAS de texte, de prix, d'horaires ou de détails non "
        "présents dans le caption."
    )


def looks_like_visual_question(message: str) -> bool:
    """V5.7.1 — Détection rapide : le message semble-t-il référencer
    du contenu visuel (sans forcément avoir une zone précise) ?

    Utilisé pour décider d'injecter le garde-fou anti-hallucination
    quand le détecteur sémantique n'a pas triggeré mais qu'il y a
    quand même un signal visuel.

    Heuristique simple basée sur les références visuelles + verbes
    interrogatifs. Pas besoin d'embedding ici, c'est juste un signal
    secondaire low-cost.
    """
    if not message:
        return False
    # Si on trouve une référence à un objet visuel
    if _VISUAL_OBJECTS.search(message):
        return True
    # Si on trouve une référence spatiale + verbe interrogatif/lecture
    has_spatial = bool(_SPATIAL_HINTS.search(message))
    has_question = bool(re.search(
        r"(?i)\b(?:que|qu['e]?|quel(?:le|s|les)?|what|how|which|où|where|"
        r"voir|vois|see|read|lis|describe|décri)\b",
        message,
    ))
    return has_spatial and has_question
