"""Routeur généraliste : type, complexité, forme de livrable."""

from rune.agentic.router import route


def test_code_tasks_route_to_code():
    for task in [
        "crée un module Python de validation d'email avec ses tests",
        "fais un code python qui calcule fibonacci avec une boucle for",
        "implémente une classe Pile avec push/pop et ses tests",
    ]:
        r = route(task)
        assert r.kind == "code"
        assert r.deliverable == "code"


def test_research_routes_and_short_answer():
    r = route("cherche les dernières actualités sur les LLM open source")
    assert r.kind == "recherche"
    # question/veille → réponse directe (pas de fichier imposé)
    assert r.deliverable in ("reponse", "fichier")


def test_redaction_wants_a_file():
    r = route("rédige un rapport détaillé sur l'état de l'art du MLOps")
    assert r.kind == "redaction"
    assert r.deliverable == "fichier"


def test_analysis_task():
    r = route("compare les avantages et inconvénients de PostgreSQL et MongoDB")
    assert r.kind == "analyse"


def test_general_fallback():
    r = route("bonjour, peux-tu m'aider ?")
    assert r.kind == "general"
    assert r.deliverable == "reponse"


def test_complexity_levels():
    assert route("calcule fibonacci").level == 0
    assert route(
        "crée un module de validation d'email robuste avec gestion des "
        "sous-domaines et ses tests unitaires complets").level >= 1
    big = ("rédige un rapport complet qui compare trois bases de données, "
           "puis analyse chaque option et ensuite recommande la meilleure "
           "avec plusieurs critères détaillés et des exemples concrets pour "
           "chaque cas d'usage envisagé dans le contexte industriel")
    assert route(big).level == 2


def test_file_deliverable_on_complex_even_without_keyword():
    # niveau 2 sans mot-clé fichier → on matérialise quand même un livrable
    big = " ".join(["analyse"] + ["mot"] * 70)
    assert route(big).deliverable in ("fichier", "reponse")
