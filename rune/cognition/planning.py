"""V4.0.c — Planning: executive control (intent + goal stack + plan generator).

Inspiration biologique
----------------------
Cortex préfrontal latéral (PFC). Maintient des buts à moyen/long
terme, décompose les demandes complexes en étapes, et fournit un
contexte persistant pour les conversations qui s'étendent sur
plusieurs tours.

Position dans le cycle cognitif
-------------------------------
Hook A.2 (Phase A) :
    plan_result = planning.process(user_message)
    → IntentResult, active_goal, prompt_block

Hook C (Phase C) :
    Inject [Plan en cours] block into system_text.

Architecture
------------
3 sous-systèmes coordonnés par PlanningPhase :

1. IntentClassifier — règles ordonnées (premier match gagne).
   Distingue chitchat / one_shot / multi_step / continuation.

2. GoalStack — pile thread-safe avec persistance JSON atomique.
   Garantit qu'un seul goal est "active" à la fois ; les autres
   passent à "blocked" lors de l'activation d'un nouveau.

3. PlanGenerator — décompose une demande en étapes.
   2 modes : LLM (avec parsing JSON tolérant 3-stratégies) +
   fallback template regex.

Design contracts
----------------
1. Anti-over-planning : "salut", "merci", "ok" → chitchat même avec
   un goal actif. Test obligatoire.
2. Try/except wrapper sur PlanningPhase.process → PlanningResult()
   neutre on crash.
3. Persistance atomique pour goals.json.
4. JSON parsing 3-stratégies : fenced ```json``` → bare braces →
   whole string. Première qui donne un dict avec "steps" gagne.
5. Pure Python.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("rune.cognition.planning")


# ── Constants ────────────────────────────────────────────────────────

INTENTS = ("chitchat", "one_shot", "multi_step", "continuation", "step_completion")
GOAL_STATUSES = ("pending", "active", "blocked", "done", "abandoned")


# ════════════════════════════════════════════════════════════════════
# Intent classifier
# ════════════════════════════════════════════════════════════════════

# Chitchat: short greetings + acknowledgements.
_CHITCHAT_RE = re.compile(
    r"^(?:salut|bonjour|coucou|hello|hi|hey|yo|bonsoir|merci|"
    r"thanks|ciao|bye|à plus|à\+|cool|ok|d'accord|ouais|ok\.?|"
    r"oui\.?|non\.?|nope|yep|yes\.?|no\.?|"
    r"ça va|comment (?:ça )?vas?(?:-tu)?|"
    r"how('?s| are) (?:it )?(?:going|you)|what'?s up)\b",
    re.IGNORECASE,
)
_CHITCHAT_MAX_WORDS = 6

# Multi-step indicators in French.
_MULTI_STEP_MARKERS_FR = (
    "étape", "étapes", "d'abord", "ensuite", "puis", "enfin",
    "construire", "développer", "implémenter", "mettre en place",
    "déployer", "refactor", "refactoriser", "migrer", "coder",
    "tout le", "toute la", "complet", "complète", "intégral",
    "from scratch", "de zéro", "plusieurs",
    # V5.6.16 — B4 fix : élargir aux cas vie réelle.
    # Le classifier était trop dev-orienté (build/deploy/refactor).
    # Ajoute organiser/préparer/planifier + verbes d'action vie réelle.
    "organiser", "préparer", "planifier", "gérer", "anticiper",
    "aide-moi à", "aide moi à", "il faut que", "je dois", "je vais",
    "avant la date", "avant le", "d'ici", "dans les",
    "déménager", "déménagement", "voyage", "voyager",
    "événement", "evenement", "fête", "mariage", "anniversaire",
    "checklist", "todo", "to-do", "liste", "planning",
    "trouver", "chercher", "contacter", "résilier", "résiliation",
    "souscrire", "s'inscrire", "ouvrir un compte",
    "rendez-vous", "rdv",
)
# Multi-step indicators in English.
_MULTI_STEP_MARKERS_EN = (
    "step by step", "stage by stage", "first", "then", "finally",
    "build", "develop", "implement", "set up", "deploy",
    "refactor", "migrate", "from scratch", "complete", "full",
    # V5.6.16 — variantes vie réelle EN
    "organize", "organise", "prepare", "plan", "schedule",
    "help me to", "i need to", "i have to", "before the",
    "moving", "move out", "trip", "travel", "event",
    "checklist", "todo", "to-do", "list",
    "find", "search for", "contact", "cancel", "subscribe",
    "appointment", "meeting",
)

# Continuation markers (only fire if a goal is already active).
_CONTINUATION_MARKERS_FR = (
    "continue", "continuons", "on reprend", "reprend", "on en était",
    "tu disais", "tu m'avais dit", "comme prévu", "comme on a dit",
    "la suite", "et après", "et maintenant", "ensuite",
)
_CONTINUATION_MARKERS_EN = (
    "let's continue", "let's resume", "back to", "as we said",
    "where we left off", "next step", "what's next",
)

# Step-completion: user signals they finished a step. Only fires if
# a goal is currently active. Triggers a goal_stack.advance_step() call.
# The /done command is the explicit form; the verbal markers are the
# implicit form. Both are checked.
_STEP_COMPLETION_RE = re.compile(
    r"^/done\b|^/next\b|^/fait\b|^/suivant\b",
    re.IGNORECASE,
)
_STEP_COMPLETION_MARKERS_FR = (
    "j'ai fini", "j ai fini", "c'est fait", "c est fait", "fait !",
    "terminé", "terminée", "done !",
    "étape suivante", "passons à la suite", "passons à la prochaine",
    "ok pour la 1", "ok pour la 2", "ok pour la 3",
    "première étape ok", "deuxième étape ok", "troisième étape ok",
    "step 1 done", "step 2 done", "step 3 done",
    "j'ai terminé", "j ai terminé",
    # V5.6.16 — B5 fix : variantes naturelles vie réelle.
    # Observées en in vivo : "j'ai trouvé un appart", "on passe à la suite",
    # "c'est bon", "ça y est", etc. Le classifier était trop dev-orienté.
    "j'ai trouvé", "j ai trouvé", "j'ai pris", "j ai pris",
    "j'ai obtenu", "j ai obtenu", "j'ai eu", "j ai eu",
    "j'ai fait", "j ai fait", "j'ai réglé", "j ai réglé",
    "on passe", "on enchaîne", "on continue",
    "passe à", "passe au", "passe à la",
    "à la suite", "la suite", "suivant", "suivante", "prochain",
    "c'est bon", "c est bon", "c'est ok", "c est ok",
    "ça y est", "ca y est",
    "go pour", "go suivant", "go next",
    "réglé", "réglée", "réglés", "réglées",
    "bouclé", "bouclée",
)
_STEP_COMPLETION_MARKERS_EN = (
    "i'm done", "im done", "i am done", "finished", "completed",
    "step done", "next step please", "move to next", "i finished",
    "first step done", "second step done", "third step done",
    # V5.6.16 — variantes EN naturelles
    "i found", "i got", "i did", "i've found", "i've got", "i've done",
    "moving on", "let's move on", "what's next", "what next",
    "go next", "next one", "next please",
    "all set", "all good", "good to go",
)


@dataclass
class IntentResult:
    intent: str = "one_shot"
    confidence: float = 0.5
    matched_markers: list[str] = field(default_factory=list)


class IntentClassifier:
    """Rule-based intent classifier (no ML, deterministic).

    Rule order matters — first match wins. Designed to be conservative:
    favors one_shot over multi_step when ambiguous, to avoid
    over-planning trivial requests.
    """

    def classify(self, text: str, has_active_goal: bool = False) -> IntentResult:
        if not text or not text.strip():
            return IntentResult(intent="chitchat", confidence=0.9)

        normalized = text.strip().lower()
        words = normalized.split()
        word_count = len(words)

        # 1. Step completion — fires only when a goal is active. The
        #    explicit /done command is unambiguous (high confidence);
        #    the verbal markers are checked next. Placed before
        #    chitchat so "/done" doesn't get filtered as a short
        #    greeting-like message.
        if has_active_goal:
            if _STEP_COMPLETION_RE.match(normalized):
                return IntentResult(
                    intent="step_completion",
                    confidence=0.95,
                    matched_markers=["/done"],
                )
            sc_markers = []
            for m in _STEP_COMPLETION_MARKERS_FR + _STEP_COMPLETION_MARKERS_EN:
                if m in normalized:
                    sc_markers.append(m)
            # Require: a step-completion marker AND short message
            # (≤15 words). A long message containing "j'ai fini X"
            # is more likely descriptive narration, not an advance signal.
            if sc_markers and word_count <= 15:
                return IntentResult(
                    intent="step_completion",
                    confidence=0.7,
                    matched_markers=sc_markers,
                )

        # 2. Chitchat — short messages matching the canonical opener regex.
        if word_count <= _CHITCHAT_MAX_WORDS and _CHITCHAT_RE.match(normalized):
            return IntentResult(intent="chitchat", confidence=0.85)

        # 3. Continuation — only if a goal is active (else "tu disais"
        #    has no referent and falls through to one_shot/multi_step).
        cont_markers = []
        for m in _CONTINUATION_MARKERS_FR + _CONTINUATION_MARKERS_EN:
            if m in normalized:
                cont_markers.append(m)
        if cont_markers and has_active_goal:
            return IntentResult(
                intent="continuation",
                confidence=0.8,
                matched_markers=cont_markers,
            )

        # 4. Multi-step — count distinct markers.
        all_markers = []
        for m in _MULTI_STEP_MARKERS_FR + _MULTI_STEP_MARKERS_EN:
            if m in normalized:
                all_markers.append(m)

        n_markers = len(all_markers)
        # Heuristic: ≥2 markers, OR ≥1 marker with sentence ≥12 words.
        if n_markers >= 2 or (n_markers >= 1 and word_count >= 12):
            return IntentResult(
                intent="multi_step",
                confidence=0.65,
                matched_markers=all_markers,
            )

        # 5. Default — one_shot.
        return IntentResult(intent="one_shot", confidence=0.5)


# ════════════════════════════════════════════════════════════════════
# Goal + GoalStack
# ════════════════════════════════════════════════════════════════════


@dataclass
class Goal:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    description: str = ""
    status: str = "pending"
    steps: list[str] = field(default_factory=list)
    current_step: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        if not isinstance(data, dict):
            return cls()
        return cls(
            id=str(data.get("id", uuid.uuid4().hex[:12])),
            parent_id=data.get("parent_id"),
            description=str(data.get("description", "")),
            status=str(data.get("status", "pending")),
            steps=[str(s) for s in (data.get("steps") or [])],
            current_step=int(data.get("current_step", 0)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )


class GoalStack:
    """Thread-safe goal stack with atomic JSON persistence.

    Invariants
    ----------
    - At most one goal has status="active" at any time.
    - When add(activate=True) is called, all current actives are
      demoted to "blocked".
    - Atomic write: .tmp file → replace().
    - Tolerates missing or corrupted persistence file (returns to
      empty stack rather than crashing).
    """

    def __init__(self, storage_path: Path | None = None):
        self._goals: list[Goal] = []
        self._lock = threading.Lock()
        self._storage_path = storage_path
        if storage_path is not None:
            try:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                log.warning("Could not create goals storage dir", exc_info=True)
            self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            goals_data = data.get("goals", []) if isinstance(data, dict) else []
            self._goals = [Goal.from_dict(g) for g in goals_data]
        except Exception:
            log.warning("GoalStack._load failed — starting empty", exc_info=True)
            self._goals = []

    def _save_locked(self) -> None:
        """Caller must hold self._lock."""
        if self._storage_path is None:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(
                    {"version": 1, "goals": [g.to_dict() for g in self._goals]},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            tmp.replace(self._storage_path)
        except Exception:
            log.warning("GoalStack._save_locked failed", exc_info=True)

    # ── Public API ───────────────────────────────────────────────────

    def add(
        self,
        description: str,
        steps: list[str] | None = None,
        parent_id: str | None = None,
        activate: bool = True,
    ) -> Goal:
        """Add a new goal. If activate=True, demote any current active to blocked."""
        with self._lock:
            if activate:
                for g in self._goals:
                    if g.status == "active":
                        g.status = "blocked"
                        g.updated_at = time.time()

            goal = Goal(
                parent_id=parent_id,
                description=description,
                status="active" if activate else "pending",
                steps=list(steps or []),
            )
            self._goals.append(goal)
            self._save_locked()
            return goal

    def get_active(self) -> Goal | None:
        with self._lock:
            for g in self._goals:
                if g.status == "active":
                    return g
        return None

    def has_active(self) -> bool:
        return self.get_active() is not None

    def advance_step(self, goal_id: str) -> Goal | None:
        """Advance current_step by 1. Mark done if last step reached."""
        with self._lock:
            for g in self._goals:
                if g.id != goal_id:
                    continue
                if not g.steps:
                    return g
                g.current_step += 1
                g.updated_at = time.time()
                if g.current_step >= len(g.steps):
                    g.status = "done"
                    g.current_step = len(g.steps)
                self._save_locked()
                return g
        return None

    def set_status(self, goal_id: str, status: str) -> Goal | None:
        if status not in GOAL_STATUSES:
            return None
        with self._lock:
            for g in self._goals:
                if g.id == goal_id:
                    g.status = status
                    g.updated_at = time.time()
                    self._save_locked()
                    return g
        return None

    def archive_stale(self, stale_after_sec: float) -> int:
        """Archive goals (active|pending|blocked) untouched for too long."""
        cutoff = time.time() - max(0.0, float(stale_after_sec))
        archivable = {"active", "pending", "blocked"}
        n = 0
        with self._lock:
            for g in self._goals:
                if g.status in archivable and g.updated_at < cutoff:
                    g.status = "abandoned"
                    g.updated_at = time.time()
                    n += 1
            if n > 0:
                self._save_locked()
        return n

    def list_all(self) -> list[Goal]:
        with self._lock:
            return list(self._goals)

    def clear(self) -> None:
        with self._lock:
            self._goals = []
            self._save_locked()


# ════════════════════════════════════════════════════════════════════
# Plan generator
# ════════════════════════════════════════════════════════════════════


# JSON parsing — 3 strategies, tolerant.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BARE_RE = re.compile(r"\{[^{}]*?\"steps\".*?\}", re.DOTALL)


def extract_plan_json(raw: str) -> dict | None:
    """Tolerant JSON extraction.

    Tries 3 strategies in order:
    1. Fenced ```json {...} ``` block.
    2. Bare {... "steps" ...} braces.
    3. Whole string as JSON.

    Returns the first dict that contains a "steps" key, else None.
    """
    if not raw or not isinstance(raw, str):
        return None

    candidates: list[str] = []

    # 1. Fenced block
    m = _JSON_FENCE_RE.search(raw)
    if m:
        candidates.append(m.group(1))

    # 2. Bare braces with "steps"
    for m in _JSON_BARE_RE.finditer(raw):
        candidates.append(m.group(0))

    # 3. Whole string
    candidates.append(raw.strip())

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and "steps" in data:
            return data
    return None


# V5.6.16 — B6 fix : Filtres de qualité pour le fallback PlanGenerator.
#
# Le splitter regex génère des "steps" qui sont en fait des intros
# ("aide-moi à...") ou des méta-commentaires ("il faut plusieurs étapes").
# On filtre ces faux steps pour ne garder que les vraies actions.

# Patterns qui indiquent une INTRO (à filtrer du début)
_INTRO_PATTERNS = re.compile(
    r"(?i)^\s*(?:aide[\s\-]?moi|help\s+me|assist[\s\-]?me|"
    r"je\s+(?:veux|voudrais|aimerais|souhaite|dois)\s+|"
    r"i\s+(?:want|need|would\s+like|have\s+to)\s+|"
    r"peux[\s\-]?tu\s+m['e]?\s*aider|"
    r"can\s+you\s+help)"
)

# Patterns qui indiquent un MÉTA-COMMENTAIRE (à filtrer)
_META_PATTERNS = re.compile(
    r"(?i)\b(?:il\s+(?:faut|faudra|y\s+a)|"
    r"je\s+vais\s+(?:devoir|avoir|prendre)|"
    r"il\s+(?:s'agit|est\s+question)|"
    r"there\s+(?:are|is|will\s+be)|"
    r"i'?ll\s+have\s+to|i'?m\s+going\s+to)\s+"
    r"(?:plusieurs|several|multiple|many|some|a\s+few|du\s+travail|"
    r"étapes?|steps?|phases?|stages?)\b"
)


def _filter_step_quality(step: str) -> bool:
    """V5.6.16 — Retourne False si le step ressemble à une intro ou
    un méta-commentaire au lieu d'une vraie action.
    """
    if not step or len(step.strip()) < 3:
        return False
    # Intros : "aide-moi à...", "je dois..."
    if _INTRO_PATTERNS.search(step):
        return False
    # Méta : "il faut plusieurs étapes", "there are several steps"
    if _META_PATTERNS.search(step):
        return False
    return True


PLAN_GENERATOR_PROMPT = (
    "Tu es un assistant de planification. Analyse la demande et "
    "décompose-la en étapes simples et actionnables.\n\n"
    "Règles:\n"
    "- Réponds UNIQUEMENT avec un objet JSON valide.\n"
    "- Forme: {{\"description\": \"<résumé court>\", \"steps\": [\"étape 1\", ...]}}\n"
    "- Maximum {max_steps} étapes.\n"
    "- Chaque étape <80 caractères.\n"
    "- Si pas multi-étape: {{\"description\": \"\", \"steps\": []}}\n\n"
    "Demande: {request}"
)

# Template fallback split markers.
_TEMPLATE_SPLIT_RE = re.compile(
    r"\s+(?:puis|ensuite|et puis|then|next)\s+|[.;]+\s+",
    re.IGNORECASE,
)


@dataclass
class PlanGeneratorConfig:
    use_llm: bool = True
    max_steps: int = 7


class PlanGenerator:
    """Generate a plan (description + steps) from a user request.

    Strategy: optional LLM call with template fallback.
    """

    def __init__(
        self,
        llm_call: Callable[[str], str] | None = None,
        config: PlanGeneratorConfig | None = None,
    ):
        self.llm_call = llm_call
        self.config = config or PlanGeneratorConfig()

    def generate(self, request: str) -> dict[str, Any]:
        """Return {"description": str, "steps": list[str]}.

        - If use_llm + llm_call available: call LLM, parse JSON, fall
          back to template on any failure.
        - Else: template-only path.
        """
        if not request or not isinstance(request, str):
            return {"description": "", "steps": []}

        # ── LLM path ──────────────────────────────────────────────
        if self.config.use_llm and self.llm_call is not None:
            prompt = PLAN_GENERATOR_PROMPT.format(
                max_steps=self.config.max_steps,
                request=request,
            )
            try:
                raw = self.llm_call(prompt)
            except Exception:
                log.warning("PlanGenerator LLM call raised", exc_info=True)
                raw = ""

            if raw:
                data = extract_plan_json(raw)
                if data is not None:
                    desc = str(data.get("description", ""))[:200]
                    steps_raw = data.get("steps") or []
                    steps = []
                    if isinstance(steps_raw, list):
                        for s in steps_raw[: self.config.max_steps]:
                            s_str = str(s).strip()
                            if s_str:
                                steps.append(s_str[:200])
                    # V5.8.4 — log explicite du résultat LLM pour
                    # diagnostic. Sans ça, on ne voit pas pourquoi le
                    # plan tombe à 0 step en in vivo.
                    log.info(
                        "PlanGenerator LLM: desc=%r, %d steps extracted",
                        desc[:50], len(steps),
                    )
                    return {"description": desc, "steps": steps}
                else:
                    log.warning(
                        "PlanGenerator: extract_plan_json failed on raw=%r",
                        raw[:200],
                    )
            else:
                log.warning("PlanGenerator: LLM returned empty raw")

        # ── Template fallback ─────────────────────────────────────
        # V5.6.16 — B6 fix : on filtre les intros et méta-commentaires
        # qui parasitaient les étapes (ex : "Aide-moi à déménager" en
        # tête, "Il faut plusieurs étapes" en queue).
        parts = _TEMPLATE_SPLIT_RE.split(request)
        all_steps = [p.strip() for p in parts if p and p.strip()]
        steps = [s for s in all_steps if _filter_step_quality(s)]
        if len(steps) >= 2:
            # Cas standard : phrase avec "puis/ensuite/then" déjà découpée.
            return {
                "description": (all_steps[0] if all_steps else request)[:80],
                "steps": steps[: self.config.max_steps],
            }

        # V5.8.4 — Template "vie réelle" : quand la demande mentionne
        # un scénario connu (déménagement, voyage, mariage, etc.) mais
        # sans étapes explicites, on génère un template raisonnable.
        # Mieux qu'un échec silencieux qui retombe en one_shot.
        lower = request.lower()
        if any(kw in lower for kw in (
            "déménagement", "déménager", "moving", "move out",
        )):
            return {
                "description": "Organiser le déménagement",
                "steps": [
                    "Trouver le nouveau logement",
                    "Planifier la date et le transport",
                    "Organiser le tri et le packing",
                    "Prévenir les administrations (changement d'adresse)",
                    "Effectuer les démarches d'arrivée (énergie, internet)",
                ],
            }
        if any(kw in lower for kw in ("voyage", "voyager", "trip", "travel")):
            return {
                "description": "Préparer le voyage",
                "steps": [
                    "Choisir la destination et les dates",
                    "Réserver le transport",
                    "Réserver l'hébergement",
                    "Préparer les documents (passeport, visa, assurance)",
                    "Faire les valises",
                ],
            }
        if any(kw in lower for kw in ("mariage", "wedding")):
            return {
                "description": "Organiser le mariage",
                "steps": [
                    "Fixer la date et le lieu",
                    "Établir la liste des invités",
                    "Choisir le traiteur et le menu",
                    "Organiser la cérémonie",
                    "Préparer les tenues et la décoration",
                ],
            }
        if any(kw in lower for kw in ("événement", "evenement", "event")):
            return {
                "description": "Organiser l'événement",
                "steps": [
                    "Définir l'objectif et le format",
                    "Choisir la date et le lieu",
                    "Inviter les participants",
                    "Préparer la logistique",
                    "Effectuer la communication",
                ],
            }

        return {"description": "", "steps": []}


# ════════════════════════════════════════════════════════════════════
# PlanningPhase — top-level orchestrator
# ════════════════════════════════════════════════════════════════════


@dataclass
class PlanningConfig:
    max_steps: int = 7
    goal_stale_days: int = 14
    use_llm: bool = True
    prompt_block_max_chars: int = 400


@dataclass
class PlanningResult:
    intent: str = "one_shot"
    intent_confidence: float = 0.5
    active_goal: Goal | None = None
    is_new_goal: bool = False
    prompt_block: str = ""
    # V4.0.2: step-completion signals. Both False/None when intent
    # isn't "step_completion".
    advanced_step: bool = False
    """True if a step-completion signal was detected and goal advanced."""
    completed_goal: bool = False
    """True if advancing the step also completed the entire goal."""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "active_goal": self.active_goal.to_dict() if self.active_goal else None,
            "is_new_goal": self.is_new_goal,
            "prompt_block": self.prompt_block,
            "advanced_step": self.advanced_step,
            "completed_goal": self.completed_goal,
        }


class PlanningPhase:
    """Coordinate intent classification + goal stack + plan generation.

    Failure behaviour
    -----------------
    Any internal exception → returns a neutral PlanningResult()
    (intent=one_shot, no goal). Hippocampe's hook is wrapped in
    try/except for an extra layer of safety.
    """

    def __init__(
        self,
        config: PlanningConfig | None = None,
        goal_stack: GoalStack | None = None,
        plan_generator: PlanGenerator | None = None,
    ):
        self.config = config or PlanningConfig()
        self.goal_stack = goal_stack or GoalStack()
        self.plan_generator = plan_generator or PlanGenerator(
            config=PlanGeneratorConfig(
                use_llm=self.config.use_llm,
                max_steps=self.config.max_steps,
            )
        )
        self.classifier = IntentClassifier()

    def process(self, user_message: str) -> PlanningResult:
        try:
            return self._process_inner(user_message)
        except Exception:
            log.warning("PlanningPhase.process crashed", exc_info=True)
            return PlanningResult()

    def _process_inner(self, user_message: str) -> PlanningResult:
        # 1. Archive stale goals (best-effort, never raise).
        try:
            stale_sec = self.config.goal_stale_days * 86400
            self.goal_stack.archive_stale(stale_sec)
        except Exception:
            log.warning("archive_stale failed", exc_info=True)

        # 2. Inspect current active goal.
        active = self.goal_stack.get_active()

        # 3. Classify intent.
        intent_res = self.classifier.classify(
            user_message,
            has_active_goal=(active is not None),
        )

        # 4. Branch on intent.
        if intent_res.intent in ("chitchat", "one_shot"):
            return PlanningResult(
                intent=intent_res.intent,
                intent_confidence=intent_res.confidence,
                active_goal=None,  # don't surface during chitchat/one_shot
                is_new_goal=False,
                prompt_block="",
            )

        # V4.0.2 — Step completion: user signaled they finished a step.
        # Only happens with an active goal (the classifier already
        # filtered on has_active_goal). We advance the goal stack and
        # surface the new (or now-completed) goal in the prompt block
        # so the LLM acknowledges progress in its response.
        if intent_res.intent == "step_completion" and active is not None:
            try:
                advanced = self.goal_stack.advance_step(active.id)
            except Exception:
                log.warning("advance_step failed", exc_info=True)
                advanced = None
            if advanced is None:
                # Could not advance — surface the active goal as-is.
                return PlanningResult(
                    intent="step_completion",
                    intent_confidence=intent_res.confidence,
                    active_goal=active,
                    advanced_step=False,
                    prompt_block=self._render_block(active),
                )
            completed = advanced.status == "done"
            return PlanningResult(
                intent="step_completion",
                intent_confidence=intent_res.confidence,
                active_goal=advanced,
                advanced_step=True,
                completed_goal=completed,
                prompt_block=self._render_block(advanced),
            )

        if intent_res.intent == "continuation" and active is not None:
            return PlanningResult(
                intent="continuation",
                intent_confidence=intent_res.confidence,
                active_goal=active,
                is_new_goal=False,
                prompt_block=self._render_block(active),
            )

        if intent_res.intent == "multi_step":
            try:
                plan = self.plan_generator.generate(user_message)
            except Exception:
                log.warning("plan_generator.generate failed", exc_info=True)
                plan = {"description": "", "steps": []}

            steps = plan.get("steps") or []
            if not steps:
                # Plan generator couldn't decompose — degrade to one_shot.
                return PlanningResult(
                    intent="one_shot",
                    intent_confidence=intent_res.confidence * 0.5,
                )

            new_goal = self.goal_stack.add(
                description=plan.get("description", "") or user_message[:80],
                steps=steps,
                activate=True,
            )
            return PlanningResult(
                intent="multi_step",
                intent_confidence=intent_res.confidence,
                active_goal=new_goal,
                is_new_goal=True,
                prompt_block=self._render_block(new_goal),
            )

        # Unknown intent — fall through to one_shot.
        return PlanningResult(
            intent="one_shot",
            intent_confidence=intent_res.confidence,
        )

    def _render_block(self, goal: Goal) -> str:
        """Render a [Plan en cours] block for prompt injection.

        Format
        ------
        [Plan en cours]
        But: <description>
        ✓ <step done>           (i < current_step)
        → <step current>        (i == current_step)
        · <step pending>        (i > current_step)
        """
        if goal is None:
            return ""
        lines = ["[Plan en cours]", f"But: {goal.description}"]
        for i, step in enumerate(goal.steps):
            if i < goal.current_step:
                lines.append(f"✓ {step}")
            elif i == goal.current_step:
                lines.append(f"→ {step}")
            else:
                lines.append(f"· {step}")
        block = "\n".join(lines)
        cap = self.config.prompt_block_max_chars
        if len(block) > cap:
            block = block[: cap - 1].rstrip() + "…"
        return block
