"""Classification de la NATURE d'un blocage agentique.

L'escalier de déblocage (web → best-of-N → décomposition) suppose
implicitement qu'on est en *erreur* (un bug à corriger). Mais « bloqué »
recouvre des situations très différentes qui appellent des réactions
différentes :

- ``FAILURE``       : le code plante / des tests échouent → il y a une erreur
                      concrète et identifiable. Stratégie : lire l'erreur,
                      corriger, retester (best-of-N, décomposition…).
- ``BLOCKED``       : pas d'erreur technique, mais l'agent n'avance plus
                      (répète des actions sans effet, n'appelle plus d'outil,
                      tourne en rond). Stratégie : réfléchir/redéfinir
                      l'approche — régénérer le même code ne sert à rien.
- ``UNKNOWN``       : il manque une information pour décider (doc, format,
                      symbole introuvable). Stratégie : aller chercher l'info
                      (web, lecture de fichier).
- ``WAITING_INPUT`` : une décision humaine est requise (ambiguïté de la
                      consigne). Stratégie : s'arrêter et demander.

Ce module est PUR (aucune dépendance torch / réseau) → testable hors GPU.
Il ne prend AUCUNE décision : il classe une situation décrite par des
signaux simples, et l'orchestrateur route l'escalade en conséquence.
"""

from __future__ import annotations

FAILURE = "FAILURE"
BLOCKED = "BLOCKED"
UNKNOWN = "UNKNOWN"
WAITING_INPUT = "WAITING_INPUT"

# Indices, dans une sortie d'erreur, d'une information MANQUANTE (→ UNKNOWN)
# plutôt que d'un simple bug logique (→ FAILURE).
_UNKNOWN_HINTS = (
    "modulenotfounderror", "no module named",
    "importerror", "cannot import name",
    "nameerror", "is not defined",
    "no tests ran", "no tests collected", "collected 0 items",
    "error during collection", "fixture", "not found",
)

# Indices d'une consigne ambiguë / d'un besoin de décision humaine.
_WAITING_HINTS = (
    "ambig", "précise", "precise", "clarifi", "quelle option",
    "que préfères", "à confirmer", "manque la spec", "spécification manquante",
)


def classify_blockage(*, has_test_failure: bool, no_call_streak: int,
                       repeat_calls: int, last_error: str = "",
                       asked_user: bool = False) -> str:
    """Classe la nature du blocage à partir de signaux simples.

    - ``has_test_failure`` : un run_tests rouge avec une vraie assertion.
    - ``no_call_streak``   : tours consécutifs SANS appel d'outil.
    - ``repeat_calls``     : répétitions consécutives du même appel.
    - ``last_error``       : dernière sortie d'erreur (stdout/summary).
    - ``asked_user``       : le modèle demande explicitement une décision.

    Ordre de priorité : WAITING_INPUT > UNKNOWN > BLOCKED > FAILURE.
    """
    err = (last_error or "").lower()

    # 1) Décision humaine explicitement demandée / consigne ambiguë.
    if asked_user or any(h in err for h in _WAITING_HINTS):
        return WAITING_INPUT

    # 2) Information manquante (import, symbole, collecte vide…).
    if any(h in err for h in _UNKNOWN_HINTS):
        return UNKNOWN

    # 3) Pas d'erreur franche mais l'agent n'avance plus : il n'appelle plus
    #    d'outil OU répète le même appel → impasse stratégique, pas un bug.
    if no_call_streak >= 2 or repeat_calls >= 2:
        return BLOCKED

    # 4) Sinon : il y a un échec de test concret → erreur à corriger.
    if has_test_failure:
        return FAILURE

    # Par défaut, on traite comme une erreur (le plus fréquent en build).
    return FAILURE


def strategy_for(kind: str) -> str:
    """Conseil de déblocage à injecter, adapté à la nature du blocage."""
    if kind == BLOCKED:
        return ("Tu n'as pas d'erreur technique mais tu n'avances plus. "
                "ARRÊTE de répéter : reformule l'APPROCHE. Quelle est la "
                "prochaine action concrète et DIFFÉRENTE qui te rapproche du "
                "but ? Écris-la, puis exécute-la.")
    if kind == UNKNOWN:
        return ("Il te manque une information (import, symbole, format). "
                "Va la CHERCHER : lis le fichier concerné (read_file) ou "
                "vérifie le symbole exact avant de coder à l'aveugle.")
    if kind == WAITING_INPUT:
        return ("La consigne est ambiguë sur un point qui change le résultat. "
                "Énonce l'hypothèse la plus raisonnable, signale-la clairement, "
                "et CONTINUE avec — ne bloque pas en attendant.")
    # FAILURE
    return ("Tu as une erreur concrète. Lis l'assertion exacte, identifie le "
            "fichier fautif (module ou test), corrige CE point, puis "
            "run_tests.")
