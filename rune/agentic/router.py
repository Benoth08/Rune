"""Routeur de tâches — généraliste, heuristique pure (zéro génération).

L'agent n'est pas qu'un codeur : il fait aussi de la recherche, de l'analyse,
de la rédaction, du raisonnement. Ce routeur classe la tâche en tête de boucle
et en dérive TROIS choses qui pilotent tout le reste, sans coût ni appel modèle :

- ``kind``      : code | recherche | analyse | redaction | general
- ``level``     : 0 (trivial) | 1 (moyen) | 2 (complexe)  → profondeur du plan
                  initial et de l'escalade en cas d'échec
- ``deliverable`` : code (fichiers + tests) | fichier (.md auditable)
                  | reponse (synthèse directe, pas de fichier)

Déterministe et entièrement testable hors GPU. Si l'heuristique se trompe,
l'escalade en cours de route rattrape (un niveau monte sur échec) — pas besoin
d'un juge LLM coûteux pour orienter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Indices lexicaux par nature de tâche (français + termes techniques usuels).
_CODE = ("code", "coder", "programme", "fonction", "classe", "script",
         "module", "implémente", "implementer", "api", "bug", "debug",
         "refactor", "test", "tests", "pytest", "algorithme", "compile",
         "python", "regex", "parser", "parse", "endpoint", "sql")
_RECHERCHE = ("cherche", "recherche", "trouve", "documente", "sources",
              "actualité", "actualites", "web", "internet", "qui est",
              "quand", "où", "combien", "dernières", "récent", "recent",
              "veille", "info", "nouvelles")
_ANALYSE = ("analyse", "analyser", "compare", "comparer", "évalue", "evalue",
            "diagnostique", "audit", "examine", "critique", "avantages",
            "inconvénients", "inconvenients", "pour et contre", "synthétise",
            "synthetise", "interprète", "interisse", "tendance")
_REDACTION = ("rédige", "redige", "écris", "ecris", "rapport", "article",
              "résumé", "resume", "synthèse", "synthese", "lettre", "mail",
              "email", "document", "note", "essai", "texte", "présentation",
              "presentation", "plan détaillé")
# Tâche qui appelle clairement un livrable fichier plutôt qu'une réponse courte.
_FILE_HINTS = ("rapport", "document", "article", "fichier", "rédige", "redige",
               "essai", ".md", ".txt", "note de synthèse", "compte rendu",
               "compte-rendu")


def _has(text: str, words) -> int:
    return sum(1 for w in words if w in text)


@dataclass(frozen=True)
class Route:
    kind: str
    level: int
    deliverable: str

    def as_dict(self) -> dict:
        return {"kind": self.kind, "level": self.level,
                "deliverable": self.deliverable}


def _complexity(text: str) -> int:
    """0/1/2 selon longueur + structure (connecteurs, énumérations, multi-
    livrables). Volontairement simple et lisible."""
    n_words = len(text.split())
    connectors = _has(text, (" et ", " puis ", "ensuite", "plusieurs",
                             "compare", "chaque", "ainsi que", "à la fois",
                             "etape", "étape", "d'abord"))
    multi = len(re.findall(r"\d+\)|\b\d+\.\s|\n\s*[-*]", text))  # listes
    score = 0
    if n_words >= 25 or connectors >= 1 or multi >= 1:
        score = 1
    if n_words >= 60 or connectors >= 3 or multi >= 3:
        score = 2
    return score


def route(task: str) -> Route:
    t = " " + (task or "").lower().strip() + " "
    scores = {
        "code": _has(t, _CODE),
        "recherche": _has(t, _RECHERCHE),
        "analyse": _has(t, _ANALYSE),
        "redaction": _has(t, _REDACTION),
    }
    kind = max(scores, key=scores.get) if max(scores.values()) > 0 else "general"
    level = _complexity(t)

    # Forme du livrable : le code a la sienne ; sinon fichier si la tâche
    # réclame un document, réponse directe pour une question/raisonnement court.
    if kind == "code":
        deliverable = "code"
    elif _has(t, _FILE_HINTS) >= 1 or level == 2:
        deliverable = "fichier"
    else:
        deliverable = "reponse"
    return Route(kind=kind, level=level, deliverable=deliverable)
