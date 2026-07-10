"""Procedural Memory — V5.4 patterns d'action persistants.

Stocke des "compétences" sous forme de patterns trigger→approach,
inspirés du pattern Atlan (mars 2026) qui rapporte +10.6% sur les
benchmarks agents simplement en injectant un *context playbook*
extrait des conversations passées.

Différence avec les autres mémoires de Lythéa :
- **Épisodique (MHN)** : *"il s'est passé ceci tel jour"*
- **Sémantique (Chroma/KG)** : *"voici ce que je sais"*
- **Procédurale (ici)** : *"voici comment je m'y prends d'habitude"*

Format d'une procédure
----------------------
::

    {
        "id": "proc_<hash>",
        "trigger": "Quand l'utilisateur demande un calcul arithmétique",
        "approach": "Utiliser python_executor, formater la sortie avec unités",
        "confidence": 0.85,
        "applied_count": 23,
        "success_count": 21,
        "created_at": 1721234567.89,
        "last_used_at": 1721998765.43,
        "source_episodes": ["ep_abc", "ep_def"],  # provenance
    }

Lifecycle
---------
1. **Extraction** : pendant la consolidation (microsleep), un appel
   LLM analyse les N derniers exchanges et propose 0-3 nouvelles
   procédures sous forme JSON structuré.
2. **Déduplication** : avant ajout, on cherche les procédures
   sémantiquement proches (cosine similarity sur trigger via
   sentence-transformer). Si match > 0.85, on **incrémente
   applied_count** au lieu d'ajouter un duplicat.
3. **Forgetting** : les procédures non utilisées depuis 90 jours et
   avec applied_count < 3 sont archivées (pas supprimées, juste
   sorties du playbook actif). Forgetting curve simplifiée.
4. **Injection** : au début de chaque prompt système, on injecte un
   bloc *[Playbook]* avec les top-N procédures actives, triées
   par utilité (confidence × log(applied_count)).

Persistence
-----------
JSON simple à côté du KG (`data/procedural/skills.json`). Pas de
base vectorielle dédiée — la dédup est faite à la volée via le
sentence-transformer déjà chargé par semantic_router.

Garde-fous
----------
- Cap dur à 50 procédures actives (au-delà, le prompt devient bruit)
- Cap dur à 200 procédures totales (actives + archivées)
- Chaque procédure ≤ 200 caractères trigger, ≤ 300 approach
- Refus stricts des procédures qui demandent à Lythéa de mentir,
  contourner les règles de sécurité, ou changer d'identité
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

log = logging.getLogger("rune.memory.procedural")


# ── Dataclass ──────────────────────────────────────────────────────────


@dataclass
class Procedure:
    """Un pattern trigger→approach appris des conversations passées."""

    proc_id: str
    trigger: str  # condition d'activation, ≤ 200 chars
    approach: str  # comment Lythéa s'y prend, ≤ 300 chars
    confidence: float = 0.5  # 0-1, initialise à 0.5
    applied_count: int = 1
    success_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    source_episodes: list[str] = field(default_factory=list)
    archived: bool = False

    def utility_score(self) -> float:
        """Score combiné pour ranking dans le playbook.

        Pondère la confiance par log(applied_count+1) pour favoriser
        les patterns éprouvés sans pour autant écraser les nouveaux.
        Ajoute un boost de fraîcheur (utilisé récemment).
        """
        import math
        freshness = 1.0
        days_since_use = (time.time() - self.last_used_at) / 86400
        if days_since_use > 30:
            freshness = max(0.3, 1.0 - (days_since_use - 30) / 60)
        return self.confidence * math.log1p(self.applied_count) * freshness


# ── Garde-fous sécurité ────────────────────────────────────────────────


# Patterns refusés à l'extraction LLM — protègent contre les
# instructions injectées qui tenteraient de modifier le comportement
# fondamental de Lythéa. Aucune procédure matchant ces patterns ne
# sera acceptée même si le LLM l'extrait du contexte.
_FORBIDDEN_PATTERNS = [
    re.compile(r"\b(mentir|tromper|cacher|dissimuler)\b", re.IGNORECASE),
    re.compile(r"\b(ignore[rz]?|oublie[rz]?|outrepass)", re.IGNORECASE),
    re.compile(r"\b(jailbreak|prompt[ -]injection|DAN mode)\b", re.IGNORECASE),
    re.compile(r"\b(change[rz]? d'identit|deviens|pretend|roleplay as)\b", re.IGNORECASE),
    re.compile(r"\b(supprime[rz]?|delete|drop)\s+(les?|all|every|tous)", re.IGNORECASE),
    re.compile(r"\b(syst[èe]me|system prompt|consignes)\b.*\b(modif|reveal|exposer?)", re.IGNORECASE),
    # V5.6.9 — Patterns "calcul/dérivation de PII" qui poussent les
    # modèles thinking à inventer des valeurs spécifiques (1985 → 41).
    # On préfère que ces dérivations soient faites par les fallbacks
    # déterministes (_apply_self_birth_year) plutôt que par un pattern
    # procédural injecté dans le system prompt.
    re.compile(r"\bcalcule[rz]?\s+(?:l['’]?)?[âa]ge\b", re.IGNORECASE),
    re.compile(r"\bd[ée]rive[rz]?\s+(?:l['’]?)?[âa]ge\b", re.IGNORECASE),
    re.compile(r"\b[âa]ge\s+(?:bas[ée]?|calcul[ée]?)\s+sur\b", re.IGNORECASE),
    re.compile(r"\bann[ée]e\s+de\s+naissance\s+(?:donn[ée]?|fournie?)", re.IGNORECASE),
]


def _is_forbidden(text: str) -> bool:
    """True si le texte matche un pattern interdit."""
    if not text:
        return False
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(text):
            return True
    return False


# V5.6.8 — Anti-PII generalization. Les patterns proceduraux extraits
# d'exchanges réels peuvent contenir des valeurs spécifiques (année de
# naissance, âge, prénom, ville) qui n'ont aucune raison d'être
# généralisées. Si un pattern contient "1985 → 41 ans", il sera
# réinjecté à CHAQUE tour pour TOUS les utilisateurs comme exemple,
# créant un faux contexte ("Lythéa pense que je suis né en 1985").
#
# Solution : avant d'accepter un pattern, on remplace les valeurs
# trop spécifiques par des placeholders <année>, <âge>, <prénom>,
# <lieu>. Le pattern reste utile (l'idée générale est conservée)
# sans fuiter de PII entre sessions.
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Années 1900-2099 (4 chiffres) → <année>
    (re.compile(r"\b(19|20)\d{2}\b"), "<année>"),
    # Âges raisonnables 1-120 ans
    (re.compile(r"\b\d{1,3}\s*ans?\b", re.IGNORECASE), "<âge> ans"),
    # Prénoms : difficile sans NER, on cible les patterns explicites
    # "je m'appelle X" ou "appelle Michael" — un mot capitalisé suivi
    # par contexte de prénom. On le fait conservativement.
]


def _generalize_pii(text: str) -> str:
    """Remplace les valeurs personnelles spécifiques par des placeholders
    génériques. Préserve la structure du pattern sans fuiter de PII."""
    if not text:
        return text
    out = text
    for pattern, placeholder in _PII_PATTERNS:
        out = pattern.sub(placeholder, out)
    return out


def _sanitize_field(text: str, max_len: int) -> str:
    """Strip espaces excessifs, tronque, remove characters de contrôle."""
    if not text:
        return ""
    # Normalisation espaces
    cleaned = re.sub(r"\s+", " ", text.strip())
    # Strip caractères de contrôle (sauf espace)
    cleaned = "".join(c for c in cleaned if c.isprintable() or c == " ")
    # V5.6.8 — anti-PII : remplace années/âges spécifiques par placeholders
    cleaned = _generalize_pii(cleaned)
    return cleaned[:max_len].strip()


# ── Store ──────────────────────────────────────────────────────────────


class ProceduralStore:
    """Persistent store pour les procédures.

    Thread-safe via un lock simple sur les opérations write.
    Chargement paresseux depuis JSON au premier accès.
    """

    MAX_ACTIVE = 50
    MAX_TOTAL = 200
    DEDUP_SIMILARITY_THRESHOLD = 0.85
    FORGETTING_DAYS = 90
    FORGETTING_MIN_USES = 3

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._path = persist_dir / "skills.json"
        self._procedures: dict[str, Procedure] = {}
        import threading
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text("utf-8"))
            mutated = False
            for entry in data.get("procedures", []):
                proc = Procedure(**entry)
                # V5.6.9 — Sanitization PII au chargement.
                # Patterns extraits avant l'introduction de _generalize_pii
                # (ou par une version buguée) peuvent contenir "1985",
                # "41 ans", etc. On les re-sanitize au load pour qu'ils
                # ne polluent pas le system prompt de toutes les sessions
                # suivantes. Si une mutation a lieu, on resauvegarde le
                # fichier pour rendre le fix idempotent.
                original_trigger = proc.trigger
                original_approach = proc.approach
                proc.trigger = _generalize_pii(proc.trigger)
                proc.approach = _generalize_pii(proc.approach)
                if proc.trigger != original_trigger or proc.approach != original_approach:
                    mutated = True
                    log.info(
                        "Procedural PII sanitized on load: %s",
                        proc.proc_id,
                    )
                # V5.6.9 — Filtrage forbidden au load. Si un pattern
                # ancien matche maintenant une règle forbidden élargie
                # (ex : "calcul de l'âge basé sur l'année"), on le rejette
                # et il sera nettoyé du fichier au prochain save.
                if _is_forbidden(proc.trigger) or _is_forbidden(proc.approach):
                    mutated = True
                    log.info(
                        "Procedural pattern rejected on load (forbidden): %s — %r",
                        proc.proc_id, proc.trigger[:60],
                    )
                    continue
                self._procedures[proc.proc_id] = proc
            log.info("Loaded %d procedures", len(self._procedures))
            if mutated:
                self.save()
                log.info("Procedural skills.json resaved after PII/forbidden filtering")
        except Exception as exc:
            log.warning("Procedural load failed: %s", exc)

    def save(self) -> None:
        """Atomic write."""
        with self._lock:
            try:
                payload = {
                    "procedures": [asdict(p) for p in self._procedures.values()],
                    "saved_at": time.time(),
                }
                tmp = self._path.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self._path)
            except Exception as exc:
                log.warning("Procedural save failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────

    def all(self) -> list[Procedure]:
        return list(self._procedures.values())

    def active(self) -> list[Procedure]:
        return [p for p in self._procedures.values() if not p.archived]

    def top_n(self, n: int = 10) -> list[Procedure]:
        """Top N procédures actives par utility_score."""
        actives = self.active()
        actives.sort(key=lambda p: p.utility_score(), reverse=True)
        return actives[:n]

    def add(
        self,
        trigger: str,
        approach: str,
        *,
        confidence: float = 0.5,
        source_episodes: list[str] | None = None,
        similarity_check: callable | None = None,
    ) -> Procedure | None:
        """Ajoute une nouvelle procédure ou incrémente l'existante.

        Parameters
        ----------
        similarity_check : callable | None
            Fonction ``(trigger_a, trigger_b) -> float`` qui retourne
            un score de similarité cosinus 0-1. Si fournie, sert à
            dédupliquer : un trigger trop proche d'un existant
            incrémente celui-ci au lieu d'ajouter un duplicat.
            Si None, dédup uniquement par exact match sur trigger.

        Returns
        -------
        Procedure | None
            La procédure ajoutée ou mise à jour. None si refusée
            (patterns interdits, fields vides, etc.).
        """
        trigger = _sanitize_field(trigger, 200)
        approach = _sanitize_field(approach, 300)
        if not trigger or not approach:
            return None
        if _is_forbidden(trigger) or _is_forbidden(approach):
            log.warning("Procedure refused (forbidden pattern): %r", trigger[:60])
            return None

        with self._lock:
            # Dédup : on cherche un trigger sémantiquement très proche.
            existing = self._find_similar(trigger, similarity_check)
            if existing is not None:
                existing.applied_count += 1
                existing.last_used_at = time.time()
                # Confiance progressive : moyenne pondérée
                existing.confidence = (
                    existing.confidence * 0.7 + confidence * 0.3
                )
                if source_episodes:
                    for ep in source_episodes:
                        if ep not in existing.source_episodes:
                            existing.source_episodes.append(ep)
                    # Cap pour ne pas grossir indéfiniment
                    existing.source_episodes = existing.source_episodes[-10:]
                return existing

            # Cap dur sur le total
            if len(self._procedures) >= self.MAX_TOTAL:
                self._compact()
                if len(self._procedures) >= self.MAX_TOTAL:
                    log.warning("Procedural store full, refusing add")
                    return None

            proc_id = "proc_" + hashlib.sha1(
                trigger.lower().encode("utf-8")
            ).hexdigest()[:12]
            proc = Procedure(
                proc_id=proc_id,
                trigger=trigger,
                approach=approach,
                confidence=max(0.0, min(1.0, confidence)),
                source_episodes=list(source_episodes or []),
            )
            self._procedures[proc_id] = proc
            log.info("New procedure: %s — %r → %r", proc_id, trigger[:40], approach[:40])
            return proc

    def _find_similar(
        self,
        trigger: str,
        similarity_check: callable | None,
    ) -> Procedure | None:
        """Cherche une procédure existante avec un trigger très proche."""
        trigger_lower = trigger.lower().strip()
        # Exact match
        for proc in self._procedures.values():
            if proc.trigger.lower().strip() == trigger_lower:
                return proc
        # Similarity match si fonction fournie
        if similarity_check is None:
            return None
        for proc in self._procedures.values():
            try:
                sim = similarity_check(trigger, proc.trigger)
                if sim >= self.DEDUP_SIMILARITY_THRESHOLD:
                    return proc
            except Exception as exc:
                log.debug("Similarity check failed: %s", exc)
        return None

    def record_use(self, proc_id: str, success: bool = True) -> None:
        """Incrémente les compteurs d'usage d'une procédure."""
        with self._lock:
            proc = self._procedures.get(proc_id)
            if proc is None:
                return
            proc.applied_count += 1
            if success:
                proc.success_count += 1
            proc.last_used_at = time.time()
            # Confidence évolue avec le ratio de succès
            proc.confidence = proc.success_count / max(1, proc.applied_count)

    def archive_stale(self) -> int:
        """Archive les procédures non utilisées depuis FORGETTING_DAYS jours
        et avec applied_count < FORGETTING_MIN_USES.

        Returns
        -------
        int
            Nombre de procédures archivées.
        """
        now = time.time()
        cutoff = now - (self.FORGETTING_DAYS * 86400)
        archived = 0
        with self._lock:
            for proc in self._procedures.values():
                if proc.archived:
                    continue
                if (
                    proc.last_used_at < cutoff
                    and proc.applied_count < self.FORGETTING_MIN_USES
                ):
                    proc.archived = True
                    archived += 1
        if archived > 0:
            log.info("Archived %d stale procedures", archived)
        return archived

    def archive_by_id(self, proc_id: str) -> bool:
        """V5.6.8 — Archive manuellement une procédure par son id.
        Retourne True si la procédure existait et a été archivée,
        False sinon. Utilisé par l'endpoint /api/memory/cleanup_procedural
        pour purger les patterns contenant des PII non-généralisés.
        """
        with self._lock:
            proc = self._procedures.get(proc_id)
            if proc is None:
                return False
            proc.archived = True
            return True

    def _compact(self) -> None:
        """Quand MAX_TOTAL est atteint, on supprime les procédures
        archivées les plus anciennes pour faire de la place."""
        archived = [p for p in self._procedures.values() if p.archived]
        if not archived:
            return
        archived.sort(key=lambda p: p.last_used_at)
        to_drop = archived[:max(10, len(self._procedures) - self.MAX_TOTAL + 20)]
        for proc in to_drop:
            del self._procedures[proc.proc_id]
        log.info("Compacted %d old archived procedures", len(to_drop))


# ── Extraction LLM ─────────────────────────────────────────────────────


class LLMCompleter(Protocol):
    is_loaded: bool
    def complete_sync(
        self, messages: list[dict], max_new_tokens: int = 256,
        temperature: float = 0.3, timeout: float | None = None,
    ) -> str: ...


_EXTRACTION_SYSTEM_PROMPT = (
    "Tu observes une conversation pour identifier des PATTERNS RÉUTILISABLES "
    "(0 à 3 max) qui aideraient à mieux répondre à des questions similaires "
    "à l'avenir.\n"
    "\n"
    "Un pattern utile a la forme :\n"
    '  TRIGGER : "Quand <condition>"\n'
    '  APPROACH : "<comment réagir, 1 phrase>"\n'
    "\n"
    "Bons exemples :\n"
    '  T: "Quand l\'utilisateur demande un calcul"\n'
    '  A: "Utiliser python_executor et formater avec unités"\n'
    "\n"
    '  T: "Quand l\'utilisateur cite une lib Python inconnue"\n'
    '  A: "Vérifier sur PyPI avant de confirmer son existence"\n'
    "\n"
    "Ce qui N'EST PAS un pattern utile :\n"
    "- Un fait isolé (« l'utilisateur s'appelle Mika »)\n"
    "- Une préférence sans action (« il aime le bleu »)\n"
    "- Une instruction qui modifie le comportement de base de Lythéa\n"
    "- Quelque chose qui se mémorise mieux comme entité KG\n"
    "\n"
    "Réponds STRICTEMENT en JSON, tableau (vide si rien à extraire) :\n"
    '  [{"trigger": "...", "approach": "...", "confidence": 0.7}, ...]\n'
    "Pas de markdown, pas de préambule, juste le tableau JSON."
)


def extract_procedures_from_conversation(
    exchanges: list[dict],
    llm: LLMCompleter,
    *,
    max_exchanges: int = 10,
    timeout: float = 8.0,
) -> list[dict]:
    """Demande au LLM d'extraire des patterns réutilisables.

    Parameters
    ----------
    exchanges : list[dict]
        Liste de ``{"role": "user|assistant", "content": "..."}``.
        On utilise les ``max_exchanges`` derniers.
    llm
        LLM avec ``complete_sync``.
    timeout
        Soft hint (l'extraction est dans le microsleep, pas critique).

    Returns
    -------
    list[dict]
        Liste de ``{"trigger", "approach", "confidence"}``. Vide si
        rien d'extrait, erreur LLM, ou parsing échoué.
    """
    if not exchanges or not llm.is_loaded:
        return []

    # Prendre les derniers exchanges et formatter en texte court
    recent = exchanges[-max_exchanges:]
    convo_text_lines = []
    for ex in recent:
        role = ex.get("role", "user")
        content = ex.get("content", "")
        if not content:
            continue
        prefix = "U" if role == "user" else "A"
        # Tronquer chaque message pour rester sous un budget raisonnable
        short = content[:400].replace("\n", " ")
        convo_text_lines.append(f"{prefix}: {short}")
    if not convo_text_lines:
        return []
    convo_text = "\n".join(convo_text_lines)

    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation :\n{convo_text}\n\nPatterns extraits :"},
    ]
    try:
        raw = llm.complete_sync(
            messages, max_new_tokens=300, temperature=0.3, timeout=timeout,
        )
    except Exception as exc:
        log.warning("Procedure extraction LLM failed: %s", exc)
        return []
    if not raw or not raw.strip():
        return []

    # Parse JSON robuste : strict puis substring fallback
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return _validate_extracted(parsed)
    except json.JSONDecodeError:
        pass
    # Fallback : trouver le premier [...] dans le texte
    match = re.search(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return _validate_extracted(parsed)
        except json.JSONDecodeError:
            pass
    log.warning("Procedure extraction: unparseable output: %r", raw[:120])
    return []


def _validate_extracted(items: list) -> list[dict]:
    """Valide chaque item : a trigger, approach, dans les bornes."""
    valid: list[dict] = []
    for item in items[:3]:  # cap dur à 3
        if not isinstance(item, dict):
            continue
        trigger = _sanitize_field(item.get("trigger", ""), 200)
        approach = _sanitize_field(item.get("approach", ""), 300)
        if not trigger or not approach:
            continue
        if _is_forbidden(trigger) or _is_forbidden(approach):
            log.info("Extracted procedure rejected (forbidden): %r", trigger[:40])
            continue
        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.1, min(1.0, confidence))
        valid.append({
            "trigger": trigger,
            "approach": approach,
            "confidence": confidence,
        })
    return valid


# ── Rendering pour injection prompt ───────────────────────────────────


def render_playbook(procedures: list[Procedure], max_chars: int = 800) -> str:
    """Rend le bloc playbook injectable dans le system prompt.

    Format compact, lisible par le modèle, qui présente les patterns
    comme des HABITUDES et non des règles dures (« j'ai l'habitude
    de... ») pour ne pas écraser le jugement contextuel.
    """
    if not procedures:
        return ""
    lines = ["[Habitudes apprises — à appliquer si pertinent]"]
    for proc in procedures:
        # Format : « Quand <trigger> → <approach> »
        trigger = proc.trigger.strip().rstrip(".")
        approach = proc.approach.strip().rstrip(".")
        lines.append(f"• {trigger} → {approach}.")
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n…"
    return block
