"""VisualWorkingMemory — buffer visuel court-terme inspiré du visual sketchpad
(Baddeley) / cortex pariétal postérieur.

Maintient un petit nombre d'images "vives" dans la mémoire de travail de
Lythéa, avec décay temporel et salience. Permet :
  - de référencer une image envoyée il y a quelques tours,
  - de zoomer sur une image stockée,
  - de gérer naturellement l'éviction (la moins saillante part en premier).

Capacité par défaut : 3 emplacements (matche la capacité humaine WM).
Décay : exponentiel, half-life = 600 sec (10 min).

V5.7.0 — Periphérique du module captioner. Pas de couplage avec
Cognitive state ou KG pour cette version (réservé pour V5.8.0+).
"""
from __future__ import annotations

import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("lythea.visual_working_memory")


# ──────────────────────────────────────────────────────────────────────
# Config & data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class VisualWorkingMemoryConfig:
    """Paramètres de la mémoire visuelle de travail."""
    capacity: int = 3
    """Nombre d'emplacements (3 = capacité WM humaine pour objets visuels)."""

    salience_half_life_sec: float = 600.0
    """Demi-vie du décay (10 minutes par défaut)."""

    salience_init: float = 1.0
    """Salience d'une image fraîchement stockée."""

    salience_min: float = 0.05
    """Plancher de salience avant éviction prioritaire."""

    reference_boost: float = 0.3
    """Bonus de salience quand l'image est référencée (zoom, mention)."""

    enable_decay: bool = True
    """Si False, les saliences restent figées (utile pour debug)."""


@dataclass
class VisualEntry:
    """Une image présente dans le buffer visuel court-terme."""
    image_id: str
    image_data: Any
    """L'objet PIL.Image ou équivalent."""
    caption_initial: str
    """Caption généré au moment du stockage."""
    timestamp: float
    """Quand l'image a été stockée pour la première fois."""
    last_accessed: float
    """Dernier moment où elle a été lue/référencée."""
    salience: float
    """Score 0-1 décroissant avec le temps, rechargé par accès."""
    zoom_history: list[dict] = field(default_factory=list)
    """Historique des zooms : [{region, query, result, timestamp}]"""
    references_count: int = 0
    """Combien de fois l'image a été récupérée ou citée."""

    def age_seconds(self, now: float | None = None) -> float:
        """Âge depuis le stockage initial."""
        return (now or time.time()) - self.timestamp

    def to_dict(self) -> dict:
        """Snapshot sérialisable pour debug (sans les bytes image)."""
        return {
            "image_id": self.image_id,
            "caption_initial": self.caption_initial[:120],
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "salience": round(self.salience, 3),
            "age_sec": round(self.age_seconds(), 1),
            "references_count": self.references_count,
            "n_zooms": len(self.zoom_history),
            "zoom_regions": [z.get("region", "?") for z in self.zoom_history[-3:]],
        }


# ──────────────────────────────────────────────────────────────────────
# Patterns lexicaux pour la résolution de référence
# ──────────────────────────────────────────────────────────────────────

# Patterns "dernière image", "image précédente", etc.
_RECENCY_PATTERNS = re.compile(
    r"(?i)\b(?:la\s+)?(?:derni[èe]re|pr[ée]c[ée]dente|la\s+pr[ée]c[ée]dente|"
    r"l['e]?\s*image\s+(?:du\s+dessus|d['e]\s*avant|pr[ée]c[ée]dente)|"
    r"(?:the\s+)?(?:last|previous|prior)\s+(?:image|picture|photo))\b"
)

# Pattern "première image" (1ère envoyée encore en buffer)
_FIRST_PATTERNS = re.compile(
    r"(?i)\b(?:la\s+)?(?:premi[èe]re|first)\s+(?:image|photo|picture)\b"
)


# ──────────────────────────────────────────────────────────────────────
# VisualWorkingMemory — l'API principale
# ──────────────────────────────────────────────────────────────────────


class VisualWorkingMemory:
    """Buffer visuel court-terme avec décay et salience.

    Stocke un petit nombre d'images récentes, gère leur décay temporel
    et permet de les retrouver par récence, ID, ou référence lexicale.

    Thread-safety : non garanti (à utiliser depuis le thread du serveur
    qui gère le state de session).
    """

    def __init__(self, config: VisualWorkingMemoryConfig | None = None):
        self.config = config or VisualWorkingMemoryConfig()
        self._entries: list[VisualEntry] = []
        # Stats cumulées pour debug/monitoring.
        self._stats = {
            "total_stored": 0,
            "total_evicted": 0,
            "total_referenced": 0,
            "total_zoomed": 0,
        }

    # ── Décay ─────────────────────────────────────────────────────────

    def _apply_decay(self, entry: VisualEntry, now: float) -> None:
        """Applique le décay exponentiel basé sur last_accessed.

        salience(now) = salience(last_accessed) * exp(-(now - last_accessed) / tau)
        avec tau = half_life / ln(2)

        Met à jour last_accessed à `now` après application pour que les
        appels successifs ne réappliquent pas le même décay (sinon le
        décay s'accumule multiplicativement à chaque tick).
        """
        if not self.config.enable_decay:
            return
        elapsed = now - entry.last_accessed
        if elapsed <= 0:
            return
        tau = self.config.salience_half_life_sec / math.log(2)
        decay_factor = math.exp(-elapsed / tau)
        # On ne descend jamais sous le plancher (sauf éviction).
        entry.salience = max(self.config.salience_min * 0.5, entry.salience * decay_factor)
        # Important : on update last_accessed pour ne pas re-décayer
        # depuis le même point la prochaine fois. Le décay est mémoïsé.
        entry.last_accessed = now

    def decay_step(self) -> None:
        """Tick de décay sur toutes les entrées. À appeler à chaque message.

        Le décay est mémoïsé via last_accessed : appeler plusieurs fois
        sans nouveau temps écoulé ne change rien. C'est `_apply_decay`
        qui mémorise la dernière application.
        """
        if not self._entries:
            return
        now = time.time()
        for entry in self._entries:
            self._apply_decay(entry, now)

    # ── Store / Get / Evict ───────────────────────────────────────────

    def store(self, image_data: Any, caption_initial: str = "") -> str:
        """Stocke une nouvelle image. Retourne son image_id.

        Si la capacité est atteinte, évince l'entrée avec la salience
        la plus faible (pas la plus ancienne — c'est plus juste
        cognitivement, une image souvent référencée reste).
        """
        now = time.time()

        # Décay sur les entrées existantes avant l'éviction (pour que
        # l'éviction soit basée sur la salience décayée actuelle).
        for entry in self._entries:
            self._apply_decay(entry, now)

        # Éviction si nécessaire
        while len(self._entries) >= self.config.capacity:
            # Tri par salience croissante, on enlève la plus faible.
            self._entries.sort(key=lambda e: e.salience)
            evicted = self._entries.pop(0)
            self._stats["total_evicted"] += 1
            log.debug(
                "VWM eviction: image_id=%s salience=%.3f age=%.1fs (%d refs)",
                evicted.image_id, evicted.salience,
                evicted.age_seconds(now), evicted.references_count,
            )

        # Nouvelle entrée
        entry = VisualEntry(
            image_id=f"img_{uuid.uuid4().hex[:8]}",
            image_data=image_data,
            caption_initial=caption_initial,
            timestamp=now,
            last_accessed=now,
            salience=self.config.salience_init,
        )
        self._entries.append(entry)
        self._stats["total_stored"] += 1

        log.info(
            "VWM stored: image_id=%s caption=%r (buffer: %d/%d)",
            entry.image_id,
            (caption_initial or "")[:60],
            len(self._entries), self.config.capacity,
        )
        return entry.image_id

    def get(self, image_id: str) -> VisualEntry | None:
        """Récupère une entrée par son ID. Rafraîchit la salience."""
        for entry in self._entries:
            if entry.image_id == image_id:
                self._refresh(entry)
                return entry
        return None

    def get_most_recent(self) -> VisualEntry | None:
        """L'entrée la plus récemment stockée (par timestamp initial).

        Note : on tri par timestamp (date d'arrivée), pas par
        last_accessed (qui mesure la fraîcheur attentionnelle).
        """
        if not self._entries:
            return None
        most_recent = max(self._entries, key=lambda e: e.timestamp)
        self._refresh(most_recent)
        return most_recent

    def find_by_reference(self, hint: str) -> VisualEntry | None:
        """Résolution lexicale d'une référence à une image.

        Gère :
          - "la dernière image" / "the last image" → get_most_recent()
          - "la première image" → la plus ancienne encore en buffer
          - hint vide ou non-matchant → most recent par défaut (s'il y a
            une image, on suppose qu'on parle d'elle)
        """
        if not self._entries:
            return None

        if _FIRST_PATTERNS.search(hint or ""):
            oldest = min(self._entries, key=lambda e: e.timestamp)
            self._refresh(oldest)
            return oldest

        if _RECENCY_PATTERNS.search(hint or ""):
            return self.get_most_recent()

        # Fallback : la plus récente (interprétation naturelle d'une
        # référence sans préfixe explicite).
        return self.get_most_recent()

    # ── Modifications d'état ──────────────────────────────────────────

    def _refresh(self, entry: VisualEntry) -> None:
        """Marque une entrée comme accédée : salience +boost, last_accessed = now."""
        now = time.time()
        entry.salience = min(1.0, entry.salience + self.config.reference_boost)
        entry.last_accessed = now
        entry.references_count += 1
        self._stats["total_referenced"] += 1

    def add_zoom(
        self, image_id: str, region: str, query: str, result: str,
    ) -> bool:
        """Enregistre un zoom dans l'historique d'une image.

        Rafraîchit aussi la salience (le zoom est une forme d'accès
        attentionnel fort).

        Retourne True si l'entrée a été trouvée et le zoom ajouté.
        """
        entry = self.get(image_id)  # déjà refreshed
        if entry is None:
            return False
        entry.zoom_history.append({
            "region": region,
            "query": query[:120],
            "result": result[:300],
            "timestamp": time.time(),
        })
        # Double boost : un zoom est un accès attentionnel plus fort qu'une
        # simple lecture. On rajoute un petit bonus en plus du refresh standard.
        entry.salience = min(1.0, entry.salience + 0.1)
        self._stats["total_zoomed"] += 1
        log.debug(
            "VWM zoom recorded: image_id=%s region=%r (%d zooms total)",
            image_id, region, len(entry.zoom_history),
        )
        return True

    def clear(self) -> int:
        """Vide complètement le buffer. Retourne le nombre d'entrées évincées."""
        n = len(self._entries)
        self._entries.clear()
        self._stats["total_evicted"] += n
        log.info("VWM cleared (%d entries removed)", n)
        return n

    # ── Introspection / debug ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def get_state(self) -> dict:
        """Snapshot pour debug : config + entries (sans image_data)."""
        # Décay avant snapshot pour avoir les saliences à jour.
        now = time.time()
        for entry in self._entries:
            self._apply_decay(entry, now)
        return {
            "config": {
                "capacity": self.config.capacity,
                "half_life_sec": self.config.salience_half_life_sec,
                "enable_decay": self.config.enable_decay,
            },
            "buffer": [e.to_dict() for e in self._entries],
            "n_active": len(self._entries),
            "stats": dict(self._stats),
        }

    @property
    def entries(self) -> list[VisualEntry]:
        """Liste read-only des entrées (pour itération externe).

        Ne PAS modifier directement — utiliser store/clear/add_zoom.
        """
        return list(self._entries)
