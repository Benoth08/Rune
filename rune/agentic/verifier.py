"""Vérificateur de livrable non-code.

Pour le code, l'oracle est binaire : pytest passe ou non. Pour une recherche,
une analyse ou une rédaction, il n'y a pas de test — on combine DEUX signaux
(choix retenu) :

1. heuristique (ici) : longueur suffisante, structure, sources citées si la
   tâche l'exige, pas de formules d'esquive (« je suppose que… ») ;
2. auto-critique du modèle contre les critères du blackboard (branchée dans
   l'orchestrateur, car elle nécessite une génération).

Le livrable est « vert » quand l'heuristique passe ET l'auto-critique conclut
OK. Cette moitié-ci est pure et testable hors GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Esquives qui trahissent un travail non fait (vu dans les traces : le modèle
# « suppose » le livrable produit au lieu de le produire).
_EVASIONS = ("je suppose", "supposons", "on peut supposer", "vous pouvez "
             "supposer", "à compléter", "a completer", "todo", "lorem ipsum",
             "etc etc", "...", "[insérer", "[inserer", "placeholder")
_URL = re.compile(r"https?://|\bwww\.")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reasons": self.reasons}


def check_deliverable(text: str, kind: str, level: int) -> Verdict:
    """Vérifie un livrable texte (rédaction/analyse/recherche/général).

    Seuils proportionnés au niveau de complexité — on n'exige pas un rapport
    de 300 mots pour une réponse simple."""
    reasons: list[str] = []
    body = (text or "").strip()
    words = len(body.split())

    min_words = {0: 12, 1: 80, 2: 180}.get(level, 80)
    if words < min_words:
        reasons.append(
            f"trop court ({words} mots < {min_words} attendus pour ce niveau)")

    low = body.lower()
    hit = [e for e in _EVASIONS if e in low]
    if hit:
        reasons.append(f"formule d'esquive détectée : « {hit[0]} »")

    # La recherche documentée doit s'appuyer sur des sources repérables.
    if kind == "recherche" and not _URL.search(body):
        reasons.append("aucune source/URL citée pour une tâche de recherche")

    # Une analyse/comparaison doit montrer une structure (≥2 points distincts).
    if kind == "analyse" and level >= 1:
        bullets = len(re.findall(r"\n\s*[-*\d]", body))
        if bullets < 2 and "\n\n" not in body:
            reasons.append("analyse peu structurée (attendu : points distincts)")

    return Verdict(ok=not reasons, reasons=reasons)
