"""Suivi de progression GÉNÉRALISTE d'une mission agentique.

Le maquis historique reposait sur ``red_streak`` (run_tests rouges
CONSÉCUTIFS) — un signal purement code-centré, et fragile : la moindre
écriture entre deux tests cassait la consécutivité, donc l'escalade
(web, best-of-N, décomposition) ne se déclenchait jamais au bon moment.

Ce module remplace ça par UN compteur unique et agnostique au type de
tâche : ``stalled`` = nombre de tours SANS PROGRÈS. Le « progrès » est
défini par tâche (le routeur fournit déjà ``kind``) :

- code      : moins de tests échouent qu'avant (ou ils passent au vert) ;
- recherche : de nouvelles sources / observations sont apparues ;
- analyse   : le livrable s'est étoffé (longueur substantielle en hausse) ;
- rédaction : idem (le texte produit grandit).

L'escalade lit ``stalled``, pas une chaîne de rouges consécutifs : un tour
qui n'avance pas incrémente, un tour qui avance remet à zéro — peu importe
les actions intercalées. Module PUR (aucune dépendance) → testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    """État de progression d'une mission, indépendant du type de tâche."""
    kind: str = "code"                 # code | recherche | analyse | redaction
    stalled: int = 0                   # tours consécutifs SANS progrès
    best_failed: int | None = None     # plus petit nb d'échecs vu (code)
    best_len: int = 0                  # plus grande taille de livrable vue
    seen_sources: int = 0              # nb de sources/lectures vues (recherche)
    history_stall: list = field(default_factory=list)  # trace (debug/tests)

    # ── seuils d'escalade, UNE seule échelle pour tous les types ──
    # 2 sans progrès → enrichir le contexte ; 4 → stratégie alternative ;
    # 6 → décomposition ciblée ; 7 → arrêt honnête.
    ENRICH = 2
    ALTERNATE = 4
    DECOMPOSE = 6
    STOP = 7

    def update_code(self, failed: int | None, *, passed: bool) -> bool:
        """Tour de code. Retourne True si PROGRÈS."""
        if passed:
            self._win()
            return True
        if failed is None:
            # pas de verdict exploitable (ex. pas de test) → pas un progrès
            self._stall()
            return False
        if self.best_failed is None or failed < self.best_failed:
            self.best_failed = failed
            self._win()
            return True
        self._stall()
        return False

    def update_text(self, deliverable_len: int) -> bool:
        """Tour analyse/rédaction. Progrès si le livrable s'étoffe nettement."""
        if deliverable_len > self.best_len + 40:
            self.best_len = deliverable_len
            self._win()
            return True
        self._stall()
        return False

    def update_research(self, sources_count: int) -> bool:
        """Tour recherche. Progrès si de nouvelles sources apparaissent."""
        if sources_count > self.seen_sources:
            self.seen_sources = sources_count
            self._win()
            return True
        self._stall()
        return False

    def _win(self) -> None:
        self.stalled = 0
        self.history_stall.append(0)

    def _stall(self) -> None:
        self.stalled += 1
        self.history_stall.append(self.stalled)

    # ── lecture des paliers (l'orchestrateur route l'escalade là-dessus) ──
    @property
    def should_enrich(self) -> bool:
        return self.stalled == self.ENRICH

    @property
    def should_alternate(self) -> bool:
        return self.stalled == self.ALTERNATE

    @property
    def should_decompose(self) -> bool:
        return self.stalled == self.DECOMPOSE

    @property
    def should_stop(self) -> bool:
        return self.stalled >= self.STOP
