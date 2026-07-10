"""Reflection Loop — V5.5 self-critique sélective sur cas à risque.

Implémente le pattern de réflexion identifié par Zylos Research (2026)
comme *"le mécanisme métacognitif le plus déployé en production"*.
L'agent critique sa propre réponse et la révise avant de la rendre,
**mais uniquement quand un risque concret est détecté** — la critique
systématique dégrade les performances (DeepMind 2023).

Cas où la réflexion s'active
----------------------------
1. **tech_reco** : recommandations techniques (modèle, lib, paper) —
   risque de confabulation prouvé.
2. **Complexité élevée** : router complexity ≥ 3 étapes.
3. **CRAG INCORRECT** : retrieval a échoué, le modèle a répondu sans
   contexte, risque d'invention élevé.
4. **Doute explicite dans la réponse** : la réponse contient des
   marqueurs comme « je ne suis pas sûr », « peut-être », « il me
   semble » accumulés (≥ 3 occurrences).

Cas où elle NE s'active PAS
---------------------------
- Calcul Python (le résultat est déterministe)
- Conversation casual, salutations
- Réponses ultra-courtes (< 50 caractères)
- Mode raisonnement déjà activé (DeepReasoningChain a déjà sa critique)
- Streaming en cours (la réflexion ferait double génération)

Format de la critique
---------------------
Prompt court demandant :
1. Y a-t-il une erreur factuelle vérifiable dans la réponse ?
2. Une affirmation non étayée par les sources/contexte ?
3. Un nom propre / référence à vérifier ?
4. La réponse répond-elle vraiment à la question ?

Réponse attendue en JSON strict :
::

    {
      "needs_revision": bool,
      "issues": ["liste des problèmes concrets"],
      "revised_response": "<réponse révisée si needs_revision=true, sinon vide>"
    }

Sécurité
--------
- Hard timeout 6s sur la critique (vs 3-5s ailleurs)
- Si parsing JSON échoue → on garde la réponse originale (safe)
- Limite : 1 cycle de réflexion max (pas de boucle infinie)
- Pas d'appel récursif sur la réponse révisée
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

log = logging.getLogger("rune.cognition.reflection")


# ── Trigger detection ──────────────────────────────────────────────────


class ReflectionTrigger(str, Enum):
    """Pourquoi la réflexion a été activée (ou skippée)."""
    TECH_RECO = "tech_reco"
    HIGH_COMPLEXITY = "high_complexity"
    CRAG_INCORRECT = "crag_incorrect"
    DOUBT_MARKERS = "doubt_markers"
    SKIP_TOO_SHORT = "skip_too_short"
    SKIP_CASUAL = "skip_casual"
    SKIP_REASONING_ON = "skip_reasoning_on"
    SKIP_PYTHON_RESULT = "skip_python_result"
    NOT_TRIGGERED = "not_triggered"


@dataclass
class ReflectionContext:
    """Contexte fourni pour décider si la réflexion s'active."""
    query: str = ""
    response: str = ""
    web_reason: str = ""  # ex "tech_reco: ..."
    complexity_steps: int = 0  # router complexity if known
    crag_status: str = ""  # "correct" | "ambiguous" | "incorrect" | ""
    reasoning_active: bool = False
    tool_used: str = ""  # "python" | "web" | ""


# Marqueurs de doute dans la réponse (≥ 3 → trigger).
_DOUBT_MARKERS_RE = re.compile(
    r"\b(?:"
    r"je ne suis pas sûre?|"
    r"peut-être|"
    r"il me semble|"
    r"je crois (?:que )?|"
    r"je suppose|"
    r"je pense (?:que )?|"
    r"il est possible|"
    r"sans certitude|"
    r"de mémoire|"
    r"sans source précise"
    r")\b",
    re.IGNORECASE,
)


def should_reflect(ctx: ReflectionContext) -> tuple[bool, ReflectionTrigger]:
    """Décide si la réflexion doit s'activer sur cette réponse.

    Returns
    -------
    tuple[bool, ReflectionTrigger]
        ``(should_run, trigger_reason)``. trigger_reason est utilisé
        pour logging et pour faire varier le prompt de critique selon
        le type de risque.
    """
    response = (ctx.response or "").strip()

    # Skip filters d'abord — économise les appels LLM
    if len(response) < 50:
        return False, ReflectionTrigger.SKIP_TOO_SHORT
    if ctx.reasoning_active:
        # DeepReasoningChain a déjà sa phase critique intégrée
        return False, ReflectionTrigger.SKIP_REASONING_ON
    if ctx.tool_used == "python":
        # Résultat Python = déterministe, pas de réflexion utile
        return False, ReflectionTrigger.SKIP_PYTHON_RESULT

    # Triggers positifs
    if "tech_reco" in (ctx.web_reason or ""):
        return True, ReflectionTrigger.TECH_RECO
    if ctx.complexity_steps >= 3:
        return True, ReflectionTrigger.HIGH_COMPLEXITY
    if (ctx.crag_status or "").lower() == "incorrect":
        return True, ReflectionTrigger.CRAG_INCORRECT
    # Doute accumulé : ≥ 3 marqueurs
    doubt_count = len(_DOUBT_MARKERS_RE.findall(response))
    if doubt_count >= 3:
        return True, ReflectionTrigger.DOUBT_MARKERS

    return False, ReflectionTrigger.NOT_TRIGGERED


# ── Verdict dataclass ─────────────────────────────────────────────────


@dataclass
class ReflectionVerdict:
    """Sortie de la phase de réflexion."""
    needs_revision: bool = False
    issues: list[str] = field(default_factory=list)
    revised_response: str = ""
    trigger: ReflectionTrigger = ReflectionTrigger.NOT_TRIGGERED
    duration_ms: int = 0
    raw_output: str = ""  # pour debug


# ── LLM interface ─────────────────────────────────────────────────────


class LLMCompleter(Protocol):
    is_loaded: bool
    def complete_sync(
        self, messages: list[dict], max_new_tokens: int = 400,
        temperature: float = 0.3, timeout: float | None = None,
    ) -> str: ...


# ── Prompt ────────────────────────────────────────────────────────────


_REFLECTION_SYSTEM_PROMPT = (
    "Tu critiques une réponse que Rune vient de produire. Ton but : "
    "détecter si elle contient une ERREUR CONCRÈTE qui mérite révision.\n"
    "\n"
    "Vérifie spécifiquement :\n"
    "1. Affirmation FACTUELLE non étayée (nom de modèle, lib, paper "
    "inventé sans source)\n"
    "2. Citation [N] qui ne correspond pas à la source\n"
    "3. La réponse répond-elle vraiment à la question posée ?\n"
    "4. Y a-t-il une contradiction interne ?\n"
    "\n"
    "N'invente PAS des problèmes. Si la réponse est correcte, réponds "
    'simplement {"needs_revision": false, "issues": [], "revised_response": ""}.\n'
    "\n"
    "Si tu détectes un VRAI problème concret, propose une réponse "
    "RÉVISÉE qui le corrige — pas une refonte complète, juste les "
    "corrections nécessaires.\n"
    "\n"
    "Format STRICT, JSON sur une ligne, rien d'autre :\n"
    '{"needs_revision": true|false, "issues": ["..."], '
    '"revised_response": "..."}\n'
    "Si needs_revision=false, revised_response doit être vide."
)


def _build_reflection_messages(
    query: str, response: str, trigger: ReflectionTrigger,
) -> list[dict]:
    """Construit le prompt de réflexion adapté au trigger."""
    trigger_hint = {
        ReflectionTrigger.TECH_RECO: (
            "Question technique → vérifie que chaque nom de modèle, "
            "lib ou paper cité existe vraiment et est correctement référencé."
        ),
        ReflectionTrigger.HIGH_COMPLEXITY: (
            "Question complexe → vérifie qu'aucune étape importante "
            "n'a été sautée et que la conclusion suit logiquement."
        ),
        ReflectionTrigger.CRAG_INCORRECT: (
            "La mémoire long-terme n'avait pas de contexte → la réponse "
            "vient du modèle seul, vérifie qu'elle n'invente pas de "
            "détails plausibles mais faux."
        ),
        ReflectionTrigger.DOUBT_MARKERS: (
            "La réponse exprime beaucoup d'incertitude → vérifie si "
            "certaines affirmations doivent être plus clairement "
            "qualifiées de « non vérifié »."
        ),
    }.get(trigger, "Vérifie l'exactitude factuelle.")

    user_msg = (
        f"Question utilisateur :\n{query.strip()[:600]}\n\n"
        f"Réponse de Rune :\n{response.strip()[:1500]}\n\n"
        f"Indication : {trigger_hint}\n\n"
        f"Verdict JSON :"
    )
    return [
        {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ── Parse cascade ─────────────────────────────────────────────────────


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _parse_reflection_response(raw: str) -> ReflectionVerdict | None:
    """Parse la sortie LLM en ReflectionVerdict, ou None si échec."""
    if not raw or not raw.strip():
        return None
    txt = raw.strip()

    # Step 1 : strict JSON
    try:
        obj = json.loads(txt)
        return _verdict_from_dict(obj)
    except json.JSONDecodeError:
        pass

    # Step 2 : substring JSON
    match = _JSON_OBJECT_RE.search(txt)
    if match:
        try:
            obj = json.loads(match.group(0))
            return _verdict_from_dict(obj)
        except json.JSONDecodeError:
            pass

    return None


def _verdict_from_dict(obj) -> ReflectionVerdict | None:
    """Construit un verdict depuis un dict parsé."""
    if not isinstance(obj, dict):
        return None
    needs = bool(obj.get("needs_revision", False))
    issues_raw = obj.get("issues", [])
    if isinstance(issues_raw, list):
        issues = [str(i)[:200] for i in issues_raw if i][:5]
    else:
        issues = []
    revised = str(obj.get("revised_response", "") or "").strip()
    # Cohérence : si needs_revision=true mais pas de revised → on
    # downgrade en pas-de-révision (on garde l'original)
    if needs and not revised:
        needs = False
    # Cap longueur révision (anti-flood)
    revised = revised[:4000]
    return ReflectionVerdict(
        needs_revision=needs,
        issues=issues,
        revised_response=revised if needs else "",
    )


# ── API publique ──────────────────────────────────────────────────────


def reflect_on_response(
    ctx: ReflectionContext,
    llm: LLMCompleter,
    *,
    timeout: float = 6.0,
) -> ReflectionVerdict:
    """Critique optionnelle de la réponse, retourne un verdict.

    Si la réflexion n'est pas activée (cf. should_reflect), retourne
    immédiatement un verdict NOT_TRIGGERED. Sinon appelle le LLM avec
    un prompt adapté au type de risque.

    Toujours retourne un ReflectionVerdict (jamais None) pour que le
    caller n'ait pas à gérer les exceptions.
    """
    should_run, trigger = should_reflect(ctx)
    if not should_run:
        return ReflectionVerdict(trigger=trigger)
    if llm is None or not llm.is_loaded:
        return ReflectionVerdict(trigger=trigger)

    t0 = time.monotonic()
    try:
        raw = llm.complete_sync(
            _build_reflection_messages(ctx.query, ctx.response, trigger),
            max_new_tokens=512,
            temperature=0.2,  # peu de variance pour cohérence
            timeout=timeout,
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.warning("Reflection LLM failed (%dms): %s", elapsed_ms, exc)
        return ReflectionVerdict(trigger=trigger, duration_ms=elapsed_ms)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    parsed = _parse_reflection_response(raw)
    if parsed is None:
        log.warning(
            "Reflection unparseable (%dms): %r", elapsed_ms, raw[:120],
        )
        return ReflectionVerdict(
            trigger=trigger,
            duration_ms=elapsed_ms,
            raw_output=raw[:500],
        )
    parsed.trigger = trigger
    parsed.duration_ms = elapsed_ms
    parsed.raw_output = raw[:500]

    log.info(
        "Reflection: trigger=%s needs_revision=%s issues=%d duration=%dms",
        trigger.value, parsed.needs_revision, len(parsed.issues), elapsed_ms,
    )
    return parsed


# ── UI helpers ────────────────────────────────────────────────────────


def cognitive_item_for(verdict: ReflectionVerdict) -> str | None:
    """Cognitive item à afficher selon le verdict.

    - Pas trigger : None (silencieux)
    - Trigger mais pas de révision : 🪞 *"Je relis ma réponse… c'est bon."*
    - Révision : 🔧 *"J'ai trouvé une imprécision et je l'ai corrigée."*
    """
    if verdict.trigger == ReflectionTrigger.NOT_TRIGGERED:
        return None
    if verdict.trigger in (
        ReflectionTrigger.SKIP_TOO_SHORT,
        ReflectionTrigger.SKIP_CASUAL,
        ReflectionTrigger.SKIP_REASONING_ON,
        ReflectionTrigger.SKIP_PYTHON_RESULT,
    ):
        return None  # skip silencieux

    if verdict.needs_revision:
        n = len(verdict.issues)
        if n == 1:
            return "🔧 *J'ai relu ma réponse et corrigé une imprécision.*"
        return f"🔧 *J'ai relu ma réponse et corrigé {n} points.*"

    # Réflexion sans révision
    return "🪞 *J'ai relu ma réponse, elle me paraît correcte.*"
