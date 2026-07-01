"""Working memory buffer — tampon de travail 4±1 chunks (Core).

Inspiration neuroscientifique
-----------------------------
Baddeley & Hitch (1974) — modèle de la mémoire de travail avec un
"central executive" et des tampons spécialisés (phonological loop,
visuospatial sketchpad). Cowan (2001) raffine : la capacité réelle
est de **4±1 chunks**, pas 7±2 (Miller 1956 était sur des digits
non chunked).

Application à Rune
---------------------------
Avant chaque génération, on assemble le contexte pertinent dans un
tampon de capacité bornée. Les chunks viennent par ordre de priorité :
1. Le message utilisateur courant (toujours)
2. Les skills applicables (trigger match)
3. Les entités KG mentionnées
4. Les épisodes MHN les plus frais
5. Les chunks Chroma les plus pertinents

Quand le tampon déborde, on évince le chunk le moins utile (combinaison
fraîcheur × pertinence × taille). Ça évite la "surcharge de contexte"
que Rune signale chez OpenClaw (qui remonte tout l'historique vectoriel).

Différence avec SDM/MHN
-----------------------
SDM et MHN sont des mémoires *distribuées* (pas de bornage explicite,
pattern completion via énergie). WorkingMemoryBuffer est un tampon
*explicite* : on sait quels chunks sont dedans, on peut les logguer,
les inspecter, les évincer. C'est la mémoire "au premier plan" —
le SDM reste l'arrière-plan distribué.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("rune.memory.working_memory")


# Capacité par défaut — Cowan 2001. 4±1, on prend 5 (le haut de la
# bande) pour ne pas trop évincer en pratique. Au-delà, on observe
# une dégradation de la cohérence du modèle.
DEFAULT_CAPACITY: int = 5


@dataclass
class WorkingMemoryChunk:
    """Un chunk dans le tampon de travail.

    Attributes
    ----------
    kind : str
        Type de contenu : "user_message" | "skill" | "kg_entity" |
        "mhn_episode" | "chroma_chunk" | "system_note".
    content : str
        Texte du chunk (tel qu'il sera injecté dans le prompt).
    relevance : float
        Score de pertinence [0, 1] par rapport à la requête courante.
        Utilisé pour l'éviction.
    freshness : float
        Score de fraîcheur [0, 1]. 1.0 = juste ajouté. Décroît avec
        le temps (exponential decay, half-life 5 min).
    size : int
        Taille en caractères (pour le budget de tokens).
    metadata : dict
        Source-specific (id KG, id Chroma, etc.).
    added_at : float
        Timestamp d'ajout (epoch seconds).
    """
    kind: str
    content: str
    relevance: float = 0.5
    freshness: float = 1.0
    size: int = 0
    metadata: dict = field(default_factory=dict)
    added_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.size:
            self.size = len(self.content)

    def utility(self) -> float:
        """Score combiné pour décider de l'éviction.

        Plus c'est haut, plus on garde. Combinaison :
        relevance * 0.5 + freshness * 0.3 + (1 / size_normalized) * 0.2
        (les chunks plus petits sont préférés à pertinence égale).
        """
        # size_normalized : 1.0 pour 100 chars, 0.5 pour 1000 chars
        size_score = max(0.1, min(1.0, 100 / max(self.size, 1)))
        return (
            self.relevance * 0.5
            + self.freshness * 0.3
            + size_score * 0.2
        )


class WorkingMemoryBuffer:
    """Tampon de mémoire de travail borné.

    Le tampon est vidé à chaque nouveau tour — c'est le "premier plan"
    de la conscience courante. Les chunks évincés retournent dans les
    mémoires de fond (SDM/MHN/KG/Chroma) si on veut les récupérer plus
    tard (mais c'est le rôle du retrieval, pas du buffer).

    Usage typique :

        buffer = WorkingMemoryBuffer(capacity=5)
        buffer.add(chunk1)
        buffer.add(chunk2)
        prompt_context = buffer.as_prompt_block()
        # ... génération ...
        buffer.clear()  # prêt pour le tour suivant
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        freshness_half_life_sec: float = 300.0,  # 5 min
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.freshness_half_life_sec = float(freshness_half_life_sec)
        self._chunks: list[WorkingMemoryChunk] = []
        self._last_refresh: float = time.time()

    # ── API publique ──────────────────────────────────────────────────

    def add(self, chunk: WorkingMemoryChunk) -> WorkingMemoryChunk | None:
        """Ajoute un chunk. Si débordement, évince le moins utile.

        Retourne le chunk évincé (ou None si pas d'éviction).
        """
        self._refresh_freshness()
        self._chunks.append(chunk)
        evicted: WorkingMemoryChunk | None = None
        if len(self._chunks) > self.capacity:
            # Trouve le moins utile
            min_idx = min(
                range(len(self._chunks)),
                key=lambda i: self._chunks[i].utility(),
            )
            evicted = self._chunks.pop(min_idx)
            log.debug(
                "WorkingMemory evicted %s chunk (utility=%.3f)",
                evicted.kind, evicted.utility(),
            )
        return evicted

    def get(self) -> list[WorkingMemoryChunk]:
        """Retourne les chunks courants, triés par utility décroissante."""
        self._refresh_freshness()
        return sorted(self._chunks, key=lambda c: c.utility(), reverse=True)

    def clear(self) -> None:
        """Vide le tampon (à chaque nouveau tour)."""
        self._chunks.clear()
        self._last_refresh = time.time()

    def as_prompt_block(self, max_chars: int = 4000) -> str:
        """Retourne les chunks formatés en un bloc de prompt.

        Chaque chunk est préfixé par son type entre crochets. L'ordre
        est par utility décroissante. Le total est tronqué à max_chars.
        """
        chunks = self.get()
        lines: list[str] = []
        total = 0
        for chunk in chunks:
            header = f"[{chunk.kind.upper()}]"
            line = f"{header} {chunk.content}"
            if total + len(line) > max_chars:
                # Truncate ce chunk
                remaining = max_chars - total - len(header) - 4
                if remaining > 50:
                    line = f"{header} {chunk.content[:remaining]}…"
                    lines.append(line)
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def status(self) -> dict[str, Any]:
        """Snapshot pour debugging / endpoint /status."""
        self._refresh_freshness()
        return {
            "capacity": self.capacity,
            "used": len(self._chunks),
            "chunks": [
                {
                    "kind": c.kind,
                    "size": c.size,
                    "relevance": round(c.relevance, 3),
                    "freshness": round(c.freshness, 3),
                    "utility": round(c.utility(), 3),
                }
                for c in self._chunks
            ],
        }

    # ── Internes ──────────────────────────────────────────────────────

    def _refresh_freshness(self) -> None:
        """Met à jour le score de fraîcheur de tous les chunks."""
        now = time.time()
        # half-life decay : freshness *= 0.5 par half-life écoulée
        for chunk in self._chunks:
            elapsed = now - chunk.added_at
            half_lives = elapsed / self.freshness_half_life_sec
            chunk.freshness = max(0.0, 0.5 ** half_lives)
        self._last_refresh = now
