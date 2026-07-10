"""V4.0.a — Cognitive state: theory of mind + own affect.

Inspiration biologique
----------------------
- Théorie de l'esprit : cortex temporo-pariétal (TPJ) + mPFC.
  Permet à Lythéa de modéliser l'état mental de l'interlocuteur
  (humeur perçue, concepts maîtrisés, niveau de confiance dans la
  relation) sans le confondre avec son propre état.
- État affectif propre : cortex orbito-frontal + amygdale modulatrice.
  Lythéa développe sa propre humeur, soumise à 3 forces concurrentes :
    1. Contagion empathique (atténuée — ANTI-SYCOPHANT)
    2. Compassion (déclenchée par valence négative de l'utilisateur)
    3. Signaux intrinsèques (curiosité, frustration interne)

Position dans le cycle cognitif
-------------------------------
Hook A.1 (Phase A, après learn) :
    cognitive_state.observe_user_message(text, entities)
        → met à jour user_affect, user_knowledge, user_trust, lythea_affect

Hook C (Phase C, lors de l'assemblage du prompt) :
    render_user_state_block() → "[État interlocuteur] ..."
    render_self_affect_block() → "[État affectif] ..."

Hook E.2 (Phase E, V4.1) :
    lythea_affect.current.intensity → boost de consolidation

Design contracts
----------------
1. Anti-sycophant : `contagion_max < 1.0` empêche Lythéa de mirroir
   parfaitement l'utilisateur. Test obligatoire.
2. Vocabulaire technique FR (défaut, fissure, anomalie, spectroscopie)
   absent du lexique → ne déclenche pas d'affect négatif. Test obligatoire.
3. Échec d'un sous-système → AffectVector neutre, jamais d'exception
   propagée à Hippocampe.
4. Persistance JSON atomique (.tmp + replace).
5. Pure Python (pas de torch/numpy).

Failure modes
-------------
- Fichier de persistance manquant → load() retourne False, état neutre.
- Fichier corrompu → load() retourne False, état neutre.
- Détecteur inconnu (ex: "bogus") → fallback lexical au runtime.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger("rune.memory.cognitive_state")


# ── Constants ────────────────────────────────────────────────────────

TARGETS = ("self", "user", "topic", "world")


# ════════════════════════════════════════════════════════════════════
# AffectVector — minimal affect representation (Russell's circumplex
# extended with target + confidence + timestamp).
# ════════════════════════════════════════════════════════════════════


@dataclass
class AffectVector:
    """A point in (valence × arousal) affect space.

    Attributes
    ----------
    valence : float ∈ [-1, 1]
        -1 = très déplaisant, +1 = très plaisant.
    arousal : float ∈ [0, 1]
        0 = endormi, 1 = activation maximale.
    target : str ∈ {"self", "user", "topic", "world"}
        Cible vers laquelle l'affect est dirigé. "world" = ambiant.
    confidence : float ∈ [0, 1]
        Confiance du détecteur dans cet affect.
    timestamp : float
        time.time() de création. Auto-set si 0.
    """

    valence: float = 0.0
    arousal: float = 0.0
    target: str = "world"
    confidence: float = 0.5
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        # Defensive clamping — defends against caller bugs.
        try:
            self.valence = max(-1.0, min(1.0, float(self.valence)))
        except (TypeError, ValueError):
            self.valence = 0.0
        try:
            self.arousal = max(0.0, min(1.0, float(self.arousal)))
        except (TypeError, ValueError):
            self.arousal = 0.0
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.0
        if self.target not in TARGETS:
            self.target = "world"
        if not isinstance(self.timestamp, (int, float)) or self.timestamp <= 0:
            self.timestamp = time.time()

    @property
    def intensity(self) -> float:
        """Magnitude amygdaloïde : |v| × a.

        Une valence -0.9 avec arousal 0.1 → 0.09 (tristesse calme,
        peu de boost mémoire). -0.9 × 0.9 → 0.81 (terreur, boost fort).
        """
        return abs(self.valence) * self.arousal

    def label(self) -> str:
        """Label FR sur la grille 3×3 (valence × arousal).

        Bandes valence:
          v < -0.2 → -1, v > 0.2 → +1, sinon 0.
        Bandes arousal:
          a < 0.33 → 0, a < 0.67 → 1, sinon 2.
        """
        v_band = -1 if self.valence < -0.2 else (1 if self.valence > 0.2 else 0)
        a_band = 0 if self.arousal < 0.33 else (1 if self.arousal < 0.67 else 2)
        grid = {
            (-1, 0): "tristesse calme",
            (-1, 1): "préoccupation",
            (-1, 2): "détresse",
            (0, 0): "neutre",
            (0, 1): "attention",
            (0, 2): "vigilance",
            (1, 0): "sérénité",
            (1, 1): "intérêt",
            (1, 2): "enthousiasme",
        }
        return grid[(v_band, a_band)]

    def to_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "target": self.target,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AffectVector":
        if not isinstance(data, dict):
            return cls()
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            target=data.get("target", "world"),
            confidence=data.get("confidence", 0.5),
            timestamp=data.get("timestamp", 0.0),
        )


# ════════════════════════════════════════════════════════════════════
# Lexicon-based affect detector.
#
# Note importante : ce lexique a été curé pour exclure tout terme
# technique industriel (défaut, fissure, anomalie, spectroscopie,
# corrosion). Test sentinelle obligatoire vérifie que ces termes ne
# déclenchent pas d'affect négatif.
# ════════════════════════════════════════════════════════════════════


# Format : "terme": (valence, arousal). Termes en minuscules,
# normalisés NFKC. Multi-mots triés par longueur décroissante pour
# que "j'en ai marre" soit testé avant "marre" (qui n'est PAS dans
# le lexique seul, justement pour éviter les faux positifs).
_AFFECT_LEXICON: dict[str, tuple[float, float]] = {
    # FR positif — multi-mots
    "j'adore": (0.9, 0.7),
    "j'aime": (0.6, 0.4),
    # FR positif — single tokens
    "heureux": (0.8, 0.5),
    "heureuse": (0.8, 0.5),
    "content": (0.7, 0.4),
    "contente": (0.7, 0.4),
    "ravi": (0.8, 0.6),
    "ravie": (0.8, 0.6),
    "génial": (0.9, 0.7),
    "geniale": (0.9, 0.7),
    "géniale": (0.9, 0.7),
    "super": (0.7, 0.6),
    "excellent": (0.8, 0.5),
    "excellente": (0.8, 0.5),
    "parfait": (0.8, 0.4),
    "parfaite": (0.8, 0.4),
    "merci": (0.5, 0.3),
    "bravo": (0.8, 0.6),
    "soulagé": (0.5, 0.2),
    "soulagée": (0.5, 0.2),
    "fier": (0.7, 0.5),
    "fière": (0.7, 0.5),
    # FR négatif — multi-mots
    "j'en ai marre": (-0.7, 0.6),
    "ras le bol": (-0.7, 0.7),
    # FR négatif — single tokens
    "triste": (-0.7, 0.3),
    "déprimé": (-0.8, 0.3),
    "déprimée": (-0.8, 0.3),
    "fatigué": (-0.4, 0.1),
    "fatiguée": (-0.4, 0.1),
    "épuisé": (-0.6, 0.2),
    "épuisée": (-0.6, 0.2),
    "frustré": (-0.6, 0.7),
    "frustrée": (-0.6, 0.7),
    "agacé": (-0.5, 0.6),
    "agacée": (-0.5, 0.6),
    "énervé": (-0.6, 0.8),
    "énervée": (-0.6, 0.8),
    "furieux": (-0.8, 0.9),
    "furieuse": (-0.8, 0.9),
    "colère": (-0.7, 0.8),
    "anxieux": (-0.5, 0.7),
    "anxieuse": (-0.5, 0.7),
    "inquiet": (-0.5, 0.6),
    "inquiète": (-0.5, 0.6),
    "peur": (-0.6, 0.8),
    "panique": (-0.7, 0.95),
    "déçu": (-0.6, 0.4),
    "déçue": (-0.6, 0.4),
    "désolé": (-0.4, 0.3),
    "désolée": (-0.4, 0.3),
    "honte": (-0.7, 0.5),
    "humilié": (-0.8, 0.6),
    "humiliée": (-0.8, 0.6),
    "horrible": (-0.8, 0.7),
    "nul": (-0.6, 0.4),
    "nulle": (-0.6, 0.4),
    "merde": (-0.6, 0.7),
    "putain": (-0.5, 0.7),
    # FR arousal-only
    "wow": (0.5, 0.8),
    "incroyable": (0.6, 0.8),
    "fou": (0.4, 0.8),
    "folle": (0.4, 0.8),
    "urgent": (-0.2, 0.85),
    "vite": (-0.1, 0.7),
    # EN positif
    "happy": (0.8, 0.5),
    "glad": (0.6, 0.3),
    "thanks": (0.5, 0.3),
    "thank": (0.5, 0.3),
    "love": (0.8, 0.5),
    "great": (0.7, 0.5),
    "awesome": (0.9, 0.7),
    "perfect": (0.8, 0.4),
    "excited": (0.7, 0.8),
    # EN négatif
    "sad": (-0.7, 0.3),
    "tired": (-0.4, 0.1),
    "exhausted": (-0.6, 0.2),
    "frustrated": (-0.6, 0.7),
    "angry": (-0.7, 0.8),
    "furious": (-0.9, 0.9),
    "anxious": (-0.5, 0.7),
    "worried": (-0.5, 0.6),
    "scared": (-0.6, 0.8),
    "disappointed": (-0.6, 0.4),
    "awful": (-0.8, 0.6),
    # ──────────────────────────────────────────────────────────────
    # V5.6.16 — Enrichissement lexique affectif (cognition_state optim).
    # Le détecteur lexical était trop limité (n'attrapait pas "furax",
    # "merdique", "ça me gonfle", "aux larmes", "câlin", etc.). Ces
    # ajouts boostent la sensibilité pour éviter que l'arousal reste
    # systématiquement sous le seuil 0.4 du module Affect.
    # ──────────────────────────────────────────────────────────────
    # FR négatif arousal élevé — argot/familier
    "furax": (-0.7, 0.85),
    "vénère": (-0.6, 0.8),
    "venere": (-0.6, 0.8),
    "vénèr": (-0.6, 0.8),
    "chiant": (-0.5, 0.7),
    "chiante": (-0.5, 0.7),
    "chiantes": (-0.5, 0.7),
    "merdique": (-0.6, 0.75),
    "merdiques": (-0.6, 0.75),
    "pourri": (-0.6, 0.6),
    "pourrie": (-0.6, 0.6),
    "atroce": (-0.8, 0.8),
    "catastrophe": (-0.7, 0.8),
    "catastrophique": (-0.7, 0.8),
    "désastre": (-0.7, 0.7),
    "galère": (-0.5, 0.6),
    "galérer": (-0.5, 0.6),
    "saoulé": (-0.5, 0.65),
    "saoulée": (-0.5, 0.65),
    "soûlé": (-0.5, 0.65),
    "marre": (-0.5, 0.6),
    "exaspéré": (-0.6, 0.8),
    "exaspérée": (-0.6, 0.8),
    "exaspérant": (-0.6, 0.75),
    "agacant": (-0.5, 0.6),
    "agaçant": (-0.5, 0.6),
    "écœuré": (-0.6, 0.6),
    "écœurée": (-0.6, 0.6),
    "dégoûté": (-0.6, 0.6),
    "dégoûtée": (-0.6, 0.6),
    "stressé": (-0.5, 0.75),
    "stressée": (-0.5, 0.75),
    "stress": (-0.4, 0.7),
    "angoisse": (-0.6, 0.8),
    "angoissé": (-0.6, 0.8),
    "angoissée": (-0.6, 0.8),
    "désespoir": (-0.8, 0.7),
    "désespéré": (-0.8, 0.7),
    "désespérée": (-0.8, 0.7),
    "perdu": (-0.4, 0.5),
    "perdue": (-0.4, 0.5),
    "épuisant": (-0.5, 0.5),
    "épuisante": (-0.5, 0.5),
    "lourd": (-0.4, 0.4),
    "lourde": (-0.4, 0.4),
    "ça me gonfle": (-0.6, 0.75),
    "ça me saoule": (-0.6, 0.75),
    "j'en peux plus": (-0.7, 0.7),
    "j en peux plus": (-0.7, 0.7),
    "j'ai pas envie": (-0.4, 0.4),
    "j'ai la flemme": (-0.3, 0.3),
    # FR positif intense — manquant
    "câlin": (0.7, 0.4),
    "calin": (0.7, 0.4),
    "câlins": (0.7, 0.4),
    "tendresse": (0.7, 0.5),
    "amour": (0.8, 0.5),
    "amoureux": (0.8, 0.7),
    "amoureuse": (0.8, 0.7),
    "ému": (0.5, 0.6),
    "émue": (0.5, 0.6),
    "émouvant": (0.6, 0.7),
    "émouvante": (0.6, 0.7),
    "touché": (0.4, 0.4),
    "touchée": (0.4, 0.4),
    "joie": (0.8, 0.7),
    "joyeux": (0.7, 0.6),
    "joyeuse": (0.7, 0.6),
    "merveille": (0.8, 0.7),
    "merveilleux": (0.9, 0.7),
    "merveilleuse": (0.9, 0.7),
    "magnifique": (0.8, 0.7),
    "magique": (0.7, 0.7),
    "magiques": (0.7, 0.7),
    "aux larmes": (0.5, 0.85),
    "ému aux larmes": (0.7, 0.9),
    "émue aux larmes": (0.7, 0.9),
    "j'étais aux larmes": (0.5, 0.85),
    "j etais aux larmes": (0.5, 0.85),
    "j'ai pleuré": (-0.1, 0.7),
    "le plus beau": (0.7, 0.7),
    "la plus belle": (0.7, 0.7),
    "le meilleur": (0.7, 0.5),
    "la meilleure": (0.7, 0.5),
    "enchanté": (0.7, 0.5),
    "enchantée": (0.7, 0.5),
    "extraordinaire": (0.8, 0.7),
    "incroyable": (0.6, 0.8),
    # EN — enrichissement
    "annoying": (-0.5, 0.6),
    "stuck": (-0.4, 0.5),
    "stressed": (-0.5, 0.75),
    "stressful": (-0.5, 0.7),
    "anxiety": (-0.6, 0.8),
    "overwhelmed": (-0.6, 0.7),
    "fed up": (-0.6, 0.7),
    "pissed": (-0.7, 0.8),
    "rage": (-0.8, 0.9),
    "hug": (0.7, 0.4),
    "hugs": (0.7, 0.4),
    "tears": (0.0, 0.7),
    "moved": (0.4, 0.5),
    "touching": (0.5, 0.5),
    "beautiful": (0.7, 0.5),
    "amazing": (0.8, 0.7),
    "wonderful": (0.8, 0.6),
    "fantastic": (0.8, 0.7),
}

# Multi-word terms (substring search). Triés par longueur décroissante
# pour matcher la forme la plus spécifique d'abord.
_MULTIWORD_TERMS = sorted(
    [t for t in _AFFECT_LEXICON if " " in t or "'" in t],
    key=len,
    reverse=True,
)
_SINGLE_WORD_TERMS = [t for t in _AFFECT_LEXICON if t not in _MULTIWORD_TERMS]


def _normalize(text: str) -> str:
    """NFKC + lower + collapse whitespace."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text).lower()
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _detect_lexical_affect(text: str) -> AffectVector:
    """Lexicon-based affect detection.

    Pipeline
    --------
    1. Normalize NFKC + lower.
    2. Match multi-word terms first (substring), then single tokens
       with word-boundary regex.
    3. Aggregate: mean valence, mean arousal across matched terms.
    4. Punctuation modifiers:
       - exclamation: arousal *= (1 + 0.15·n_exclamations)
       - CAPS words (≥4 letters): arousal += 0.1·n_caps_words
       - ellipsis "..." or "…" without lexical match: v=-0.2, a=0.1
    5. Confidence: 0 if no match, else min(1, 0.4 + 0.2·n_terms).
    6. Empty text → confidence 0.
    7. CAPS≥1 AND excl≥1 with no lexical match → ambient anger
       (v=-0.3, a=0.5).

    Returns
    -------
    AffectVector
        Detected affect, target left at default "world" (caller may
        rebind to "user" if it knows the message is from the user).
    """
    if not text:
        return AffectVector(confidence=0.0)

    normalized = _normalize(text)

    # Count CAPS words (in original text) before lowercasing destroys it.
    caps_words = re.findall(r"\b[A-ZÀ-ß]{4,}\b", text)
    n_caps = len(caps_words)
    n_excl = text.count("!")
    has_ellipsis = ("..." in text) or ("…" in text)

    # 1. Multi-word matching on normalized.
    matched_terms: list[str] = []
    for term in _MULTIWORD_TERMS:
        if term in normalized:
            matched_terms.append(term)

    # 2. Single-word matching with word boundaries.
    for term in _SINGLE_WORD_TERMS:
        # Build regex that respects French accents — \b in Python re
        # works on \w which now includes accents in Python 3.
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, normalized):
            matched_terms.append(term)

    if not matched_terms:
        # No lexical match. Still consider punctuation-only signals.
        if has_ellipsis:
            return AffectVector(
                valence=-0.2,
                arousal=0.1,
                confidence=0.2,
            )
        if n_caps >= 1 and n_excl >= 1:
            return AffectVector(
                valence=-0.3,
                arousal=0.5,
                confidence=0.3,
            )
        return AffectVector(confidence=0.0)

    # 3. Aggregate matched terms.
    vs = [_AFFECT_LEXICON[t][0] for t in matched_terms]
    a_s = [_AFFECT_LEXICON[t][1] for t in matched_terms]
    v_mean = sum(vs) / len(vs)
    a_mean = sum(a_s) / len(a_s)

    # 4. Apply punctuation modifiers.
    a_mean = a_mean * (1.0 + 0.15 * n_excl)
    a_mean = a_mean + 0.1 * n_caps

    # 5. Confidence ramp with number of matched terms.
    conf = min(1.0, 0.4 + 0.2 * len(matched_terms))

    return AffectVector(
        valence=v_mean,
        arousal=a_mean,
        confidence=conf,
    )


# ════════════════════════════════════════════════════════════════════
# AffectState — Lythéa's own affect with decay + empathic update.
# ════════════════════════════════════════════════════════════════════


@dataclass
class AffectState:
    """Lythéa's running affective state.

    Three simultaneous processes shape it on each empathic update:
    1. CONTAGION — capped (anti-sycophant).
    2. COMPASSION — softens valence toward user when they're sad.
    3. INTRINSIC — internal signals (curiosity, frustration).

    Saturation, not summation, on arousal: max(a_cont, a_comp, a_intr)
    instead of sum, so a calm user + intense intrinsic signal doesn't
    add up to over-arousal.
    """

    current: AffectVector = field(default_factory=AffectVector)
    quiet_turns: int = 0
    decay_half_life_sec: float = 300.0
    contagion_max: float = 0.4  # ANTI-SYCOPHANT — keep < 0.5
    inertia: float = 0.3
    reset_latch_turns: int = 8

    def decay_to(self, now: float) -> None:
        """Exponential decay from current.timestamp to now."""
        if self.current.timestamp <= 0:
            return
        dt = max(0.0, now - self.current.timestamp)
        if dt <= 0 or self.decay_half_life_sec <= 0:
            return
        try:
            factor = 0.5 ** (dt / self.decay_half_life_sec)
        except (OverflowError, ZeroDivisionError):
            factor = 0.0
        self.current = AffectVector(
            valence=self.current.valence * factor,
            arousal=self.current.arousal * factor,
            target=self.current.target,
            confidence=self.current.confidence * factor,
            timestamp=now,
        )

    def empathic_update(
        self,
        user_affect: AffectVector,
        intrinsic: AffectVector | None = None,
    ) -> None:
        """Blend contagion + compassion + intrinsic, then apply inertia.

        Cible composite (avant inertie) :
            v_target = v_contagion + v_compassion + v_intrinsic
            a_target = max(a_contagion, a_compassion, a_intrinsic)

        Inertie :
            v_new = (1-α)·v_target + α·v_current

        Target field priority : compassion > intrinsic > contagion > current.

        Reset latch : si N tours sans signal externe, retour à neutre.
        """
        now = time.time()
        self.decay_to(now)

        # 1. CONTAGION (always present)
        c_strength = min(self.contagion_max, max(0.0, user_affect.confidence))
        v_cont = c_strength * user_affect.valence
        a_cont = c_strength * user_affect.arousal

        # 2. COMPASSION (only if user is sad with reasonable confidence)
        v_comp = 0.0
        a_comp = 0.0
        compassion_active = (
            user_affect.valence < -0.3 and user_affect.confidence > 0.3
        )
        if compassion_active:
            v_comp = -0.25 * abs(user_affect.valence)
            a_comp = 0.4 * user_affect.arousal + 0.2

        # 3. INTRINSIC (full weight)
        v_intr = 0.0
        a_intr = 0.0
        intr_conf = 0.0
        intrinsic_active = False
        if intrinsic is not None and intrinsic.confidence > 0.0:
            v_intr = intrinsic.valence
            a_intr = intrinsic.arousal
            intr_conf = intrinsic.confidence
            intrinsic_active = True

        # Compose target
        v_target = v_cont + v_comp + v_intr
        a_target = max(a_cont, a_comp, a_intr)

        # Clamp pre-inertia
        v_target = max(-1.0, min(1.0, v_target))
        a_target = max(0.0, min(1.0, a_target))

        # Inertia smoothing
        alpha = self.inertia
        v_new = (1 - alpha) * v_target + alpha * self.current.valence
        a_new = (1 - alpha) * a_target + alpha * self.current.arousal

        # Target priority
        if compassion_active:
            target_field = "user"
        elif intrinsic_active:
            target_field = intrinsic.target if intrinsic else "self"
        elif c_strength > 0.05:
            target_field = "user"
        else:
            target_field = self.current.target

        # Confidence: max of the active sources
        conf_new = max(
            c_strength * user_affect.confidence,
            intr_conf * 0.8,
            self.current.confidence * (1 - alpha),
        )
        conf_new = max(0.0, min(1.0, conf_new))

        self.current = AffectVector(
            valence=v_new,
            arousal=a_new,
            target=target_field,
            confidence=conf_new,
            timestamp=now,
        )

        # Reset latch
        externally_signalled = (user_affect.confidence >= 0.3) or (intr_conf > 0.3)
        if externally_signalled:
            self.quiet_turns = 0
        else:
            self.quiet_turns += 1
            if self.quiet_turns >= self.reset_latch_turns:
                self.current = AffectVector(timestamp=now)
                self.quiet_turns = 0

    def to_dict(self) -> dict:
        return {
            "current": self.current.to_dict(),
            "quiet_turns": self.quiet_turns,
            "decay_half_life_sec": self.decay_half_life_sec,
            "contagion_max": self.contagion_max,
            "inertia": self.inertia,
            "reset_latch_turns": self.reset_latch_turns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AffectState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            current=AffectVector.from_dict(data.get("current", {})),
            quiet_turns=int(data.get("quiet_turns", 0)),
            decay_half_life_sec=float(data.get("decay_half_life_sec", 300.0)),
            contagion_max=float(data.get("contagion_max", 0.4)),
            inertia=float(data.get("inertia", 0.3)),
            reset_latch_turns=int(data.get("reset_latch_turns", 8)),
        )


# ════════════════════════════════════════════════════════════════════
# UserKnowledgeState — what concepts the user has demonstrated mastery of.
# ════════════════════════════════════════════════════════════════════


@dataclass
class UserKnowledgeState:
    """Track user's mastery of concepts via EMA on observation strength."""

    mastery: dict[str, float] = field(default_factory=dict)
    known_threshold: float = 0.6

    def observe(self, concept: str, evidence_strength: float = 0.5) -> None:
        """EMA update : mastery = 0.7·prev + 0.3·evidence."""
        if not concept:
            return
        key = concept.strip().lower()
        if not key:
            return
        prev = self.mastery.get(key, 0.0)
        self.mastery[key] = 0.7 * prev + 0.3 * float(evidence_strength)

    def is_known(self, concept: str) -> bool:
        if not concept:
            return False
        return self.mastery.get(concept.strip().lower(), 0.0) >= self.known_threshold

    def known_concepts(self) -> list[str]:
        return sorted(c for c, m in self.mastery.items() if m >= self.known_threshold)

    def to_dict(self) -> dict:
        return {
            "mastery": dict(self.mastery),
            "known_threshold": self.known_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserKnowledgeState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            mastery={
                str(k): float(v)
                for k, v in (data.get("mastery") or {}).items()
            },
            known_threshold=float(data.get("known_threshold", 0.6)),
        )


# ════════════════════════════════════════════════════════════════════
# UserAffectiveState — smoothed view of the user's affect over time.
# ════════════════════════════════════════════════════════════════════


@dataclass
class UserAffectiveState:
    last_detected: AffectVector = field(default_factory=AffectVector)
    smoothed: AffectVector = field(default_factory=AffectVector)
    smoothing: float = 0.4  # alpha for EMA on each axis

    def update(self, detected: AffectVector) -> None:
        """EMA on valence and arousal independently."""
        a = max(0.0, min(1.0, self.smoothing))
        new_v = a * detected.valence + (1 - a) * self.smoothed.valence
        new_a = a * detected.arousal + (1 - a) * self.smoothed.arousal
        new_conf = max(detected.confidence, self.smoothed.confidence * 0.8)
        self.smoothed = AffectVector(
            valence=new_v,
            arousal=new_a,
            target=detected.target if detected.confidence > 0 else self.smoothed.target,
            confidence=new_conf,
            timestamp=time.time(),
        )
        self.last_detected = detected

    def to_dict(self) -> dict:
        return {
            "last_detected": self.last_detected.to_dict(),
            "smoothed": self.smoothed.to_dict(),
            "smoothing": self.smoothing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserAffectiveState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            last_detected=AffectVector.from_dict(data.get("last_detected", {})),
            smoothed=AffectVector.from_dict(data.get("smoothed", {})),
            smoothing=float(data.get("smoothing", 0.4)),
        )


# ════════════════════════════════════════════════════════════════════
# UserTrustState — relational trust score, updates on each exchange.
# ════════════════════════════════════════════════════════════════════


@dataclass
class UserTrustState:
    score: float = 0.3
    exchange_count: int = 0
    gain_per_exchange: float = 0.02
    loss_on_friction: float = 0.05

    def observe_exchange(self, user_valence_toward_self: float = 0.0) -> None:
        """Update trust score based on user's stance toward Lythéa.

        - friction (v < -0.3) → score -= loss_on_friction
        - else → score += gain_per_exchange (slow build-up)
        Bounds [0, 1] enforced.
        """
        self.exchange_count += 1
        try:
            v = float(user_valence_toward_self)
        except (TypeError, ValueError):
            v = 0.0
        if v < -0.3:
            self.score = max(0.0, self.score - self.loss_on_friction)
        else:
            self.score = min(1.0, self.score + self.gain_per_exchange)

    def label(self) -> str:
        if self.score < 0.25:
            return "réservée"
        if self.score < 0.5:
            return "modérée"
        if self.score < 0.75:
            return "établie"
        return "forte"

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "exchange_count": self.exchange_count,
            "gain_per_exchange": self.gain_per_exchange,
            "loss_on_friction": self.loss_on_friction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserTrustState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            score=float(data.get("score", 0.3)),
            exchange_count=int(data.get("exchange_count", 0)),
            gain_per_exchange=float(data.get("gain_per_exchange", 0.02)),
            loss_on_friction=float(data.get("loss_on_friction", 0.05)),
        )


# ════════════════════════════════════════════════════════════════════
# Top-level CognitiveState orchestrator.
# ════════════════════════════════════════════════════════════════════


@dataclass
class CognitiveStateConfig:
    decay_half_life_sec: float = 300.0
    contagion_max: float = 0.4
    inertia: float = 0.3
    reset_latch_turns: int = 8
    detector: str = "lexical"  # ∈ {"lexical","classifier","llm"}
    user_known_threshold: float = 0.6


# Friction markers : explicit user feedback that Lythéa is wrong.
_FRICTION_MARKERS = (
    "tu te trompes",
    "c'est faux",
    "tu as tort",
    "non c'est",
    "non,",
    "tu ne comprends pas",
    "you're wrong",
    "that's wrong",
    "incorrect",
    "you don't understand",
)
_SECOND_PERSON_RE = re.compile(r"\btu\b|\bvous\b|\byou\b", re.IGNORECASE)


class CognitiveState:
    """Top-level cognitive state for a single user × Lythéa session.

    Composes :
        - lythea_affect    : Lythéa's own affect (with decay, contagion)
        - user_affect      : smoothed view of user's affect
        - user_knowledge   : concepts the user has demonstrated mastery of
        - user_trust       : relational trust score

    Persistance
    -----------
    Atomic JSON write to ``<storage_dir>/<session_id>.json`` via
    ``.tmp`` + ``replace()``. Tolerates missing or corrupted files
    (returns False from ``load``, leaves state at defaults).
    """

    def __init__(
        self,
        config: CognitiveStateConfig | None = None,
        storage_dir: Path | None = None,
    ):
        self.config = config or CognitiveStateConfig()
        self.storage_dir = storage_dir
        if storage_dir is not None:
            try:
                storage_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                log.warning("Could not create cognitive_state storage dir", exc_info=True)

        self.lythea_affect = AffectState(
            decay_half_life_sec=self.config.decay_half_life_sec,
            contagion_max=self.config.contagion_max,
            inertia=self.config.inertia,
            reset_latch_turns=self.config.reset_latch_turns,
        )
        self.user_affect = UserAffectiveState()
        self.user_knowledge = UserKnowledgeState(
            known_threshold=self.config.user_known_threshold
        )
        self.user_trust = UserTrustState()

    # ── Detector dispatch ────────────────────────────────────────────

    def _detect(self, text: str) -> AffectVector:
        """Dispatch to the configured detector. Unknown → lexical."""
        try:
            # V4.0 only ships lexical. Forward hooks left for future.
            return _detect_lexical_affect(text)
        except Exception:
            log.warning("Affect detector crashed — returning neutral", exc_info=True)
            return AffectVector(confidence=0.0)

    # ── Main observation entry point ─────────────────────────────────

    def observe_user_message(
        self,
        text: str,
        entities: Iterable[str] | None = None,
        intrinsic_signal: AffectVector | None = None,
    ) -> AffectVector:
        """Update all four sub-states from a single user message.

        Returns the *raw* detected affect (for debug / hippocampe
        downstream consumers). The smoothed view lives in
        ``self.user_affect.smoothed``.

        Pipeline
        --------
        1. Detect user affect (try/except → neutral).
        2. user_affect.update(detected)      — smoothing
        3. lythea_affect.empathic_update(detected, intrinsic_signal)
           Note: uses *raw* detected, not smoothed, so cold starts
           don't get dampened.
        4. For each entity → user_knowledge.observe(ent, 0.4)
        5. Detect user stance toward Lythéa:
           - explicit friction marker → valence_toward_self = -0.5
           - implicit (negative affect + 2nd-person) → use detected.v
           - else → 0.0
        6. user_trust.observe_exchange(valence_toward_self)
        """
        detected = self._detect(text or "")
        # User messages target self/Lythéa by default; we'll reclassify
        # later if needed. For now leave at "world" — affect itself
        # carries no allocentric semantics yet.

        try:
            self.user_affect.update(detected)
        except Exception:
            log.warning("user_affect.update failed", exc_info=True)

        try:
            self.lythea_affect.empathic_update(
                user_affect=detected,
                intrinsic=intrinsic_signal,
            )
        except Exception:
            log.warning("lythea_affect.empathic_update failed", exc_info=True)

        if entities:
            for ent in entities:
                if ent:
                    try:
                        self.user_knowledge.observe(str(ent), 0.4)
                    except Exception:
                        log.warning("user_knowledge.observe failed", exc_info=True)

        # Friction detection
        normalized = _normalize(text or "")
        valence_toward_self = 0.0
        if any(m in normalized for m in _FRICTION_MARKERS):
            valence_toward_self = -0.5
        elif detected.valence < -0.3 and detected.confidence > 0.3:
            if _SECOND_PERSON_RE.search(normalized):
                valence_toward_self = detected.valence

        try:
            self.user_trust.observe_exchange(valence_toward_self)
        except Exception:
            log.warning("user_trust.observe_exchange failed", exc_info=True)

        return detected

    # ── Prompt block rendering ───────────────────────────────────────

    def render_user_state_block(self, max_chars: int = 200) -> str:
        """Render '[État interlocuteur]' block, or '' if nothing useful.

        Format
        ------
        '[État interlocuteur] Familiarité: <label> — Humeur perçue: <label> — Connaît déjà: c1, c2, c3'

        Conditions de rendu :
        - Familiarité : exchange_count >= 3
        - Humeur perçue : smoothed.confidence >= 0.3 AND label != "neutre"
        - Connaît déjà : known_concepts() non-empty (top 5)
        - Si aucun → "" (block omitted entirely)
        """
        try:
            parts: list[str] = []

            if self.user_trust.exchange_count >= 3:
                parts.append(f"Familiarité: {self.user_trust.label()}")

            smoothed = self.user_affect.smoothed
            if smoothed.confidence >= 0.3:
                lbl = smoothed.label()
                if lbl != "neutre":
                    parts.append(f"Humeur perçue: {lbl}")

            known = self.user_knowledge.known_concepts()
            if known:
                parts.append(f"Connaît déjà: {', '.join(known[:5])}")

            if not parts:
                return ""

            block = "[État interlocuteur] " + " — ".join(parts)
            if len(block) > max_chars:
                block = block[: max_chars - 1].rstrip() + "…"
            return block
        except Exception:
            log.warning("render_user_state_block failed", exc_info=True)
            return ""

    def render_self_affect_block(self, max_chars: int = 200) -> str:
        """Render '[État affectif]' block for Lythéa's own state.

        Format
        ------
        '[État affectif] Tu ressens: <label><target_hint>'

        Conditions de rendu :
        - confidence < 0.2 → ""
        - label == "neutre" → ""

        target_hint :
            "user"  → " (à propos de ton interlocuteur)"
            "self"  → " (à propos de toi)"
            "topic" → " (à propos du sujet)"
            "world" → ""
        """
        try:
            cur = self.lythea_affect.current
            if cur.confidence < 0.2:
                return ""
            lbl = cur.label()
            if lbl == "neutre":
                return ""

            hint_map = {
                "user": " (à propos de ton interlocuteur)",
                "self": " (à propos de toi)",
                "topic": " (à propos du sujet)",
                "world": "",
            }
            hint = hint_map.get(cur.target, "")

            block = f"[État affectif] Tu ressens: {lbl}{hint}"
            if len(block) > max_chars:
                block = block[: max_chars - 1].rstrip() + "…"
            return block
        except Exception:
            log.warning("render_self_affect_block failed", exc_info=True)
            return ""

    # ── Persistence ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "lythea_affect": self.lythea_affect.to_dict(),
            "user_affect": self.user_affect.to_dict(),
            "user_knowledge": self.user_knowledge.to_dict(),
            "user_trust": self.user_trust.to_dict(),
        }

    def load_dict(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        try:
            self.lythea_affect = AffectState.from_dict(data.get("lythea_affect", {}))
            # Re-pin config-driven fields so old persisted state respects
            # latest config (e.g. updated half_life).
            self.lythea_affect.decay_half_life_sec = self.config.decay_half_life_sec
            self.lythea_affect.contagion_max = self.config.contagion_max
            self.lythea_affect.inertia = self.config.inertia
            self.lythea_affect.reset_latch_turns = self.config.reset_latch_turns

            self.user_affect = UserAffectiveState.from_dict(data.get("user_affect", {}))
            self.user_knowledge = UserKnowledgeState.from_dict(
                data.get("user_knowledge", {})
            )
            self.user_knowledge.known_threshold = self.config.user_known_threshold
            self.user_trust = UserTrustState.from_dict(data.get("user_trust", {}))
        except Exception:
            log.warning("CognitiveState.load_dict failed — keeping defaults", exc_info=True)

    def save(self, session_id: str) -> bool:
        """Atomic JSON write. Returns True on success, False otherwise."""
        if not session_id or self.storage_dir is None:
            return False
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            target = self.storage_dir / f"{session_id}.json"
            tmp = target.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            tmp.replace(target)
            return True
        except Exception:
            log.warning("CognitiveState.save failed for session %r", session_id, exc_info=True)
            return False

    def load(self, session_id: str) -> bool:
        """Read JSON for session_id. Tolerates missing & corrupted files."""
        if not session_id or self.storage_dir is None:
            return False
        target = self.storage_dir / f"{session_id}.json"
        if not target.exists():
            return False
        try:
            with target.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_dict(data)
            return True
        except Exception:
            log.warning("CognitiveState.load failed for %r — corruption?", session_id, exc_info=True)
            return False

    def reset_session_scoped(self) -> None:
        """Reset affect state, KEEP knowledge + trust.

        Used between distinct sessions where mood should not carry
        over but the user is still the same person.
        """
        self.lythea_affect = AffectState(
            decay_half_life_sec=self.config.decay_half_life_sec,
            contagion_max=self.config.contagion_max,
            inertia=self.config.inertia,
            reset_latch_turns=self.config.reset_latch_turns,
        )
        self.user_affect = UserAffectiveState()
