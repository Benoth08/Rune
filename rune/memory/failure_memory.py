"""FailureMemory — anti-patterns appris des échecs.

Ce qu'Rune fait mal
---------------------
Rune n'apprend que des succès (SKILL.md après vérification positive).
Quand il échoue, il réessaie sans capitaliser sur l'échec. Un humain,
lui, retient "ne refais pas ça comme ça".

Rune corrige ça avec :class:`FailureMemory` :

1. Quand le verifier dit "échec", on analyse la cause racine
2. On stocke un :class:`FailurePattern` (contexte + action tentée +
   symptôme + cause racine + correction suggérée)
3. Au prochain essai, on consulte d'abord les anti-patterns avant
   les skills positifs — on évite de retomber dans le piège

Couplage avec AutoSkill
-----------------------
- Skill = "comment faire" (positif)
- FailurePattern = "comment ne pas faire" (négatif)
- Au retrieval, on cherche les deux : si un skill matche MAIS qu'un
  failure pattern aussi, on injecte le skill + l'anti-pattern.

Format
------
::

    {
        "failure_id": "fail_<hash>",
        "context": "Recommander un modèle NER français",
        "attempted_action": "Citer spaCy fr_core_news_sm/md/lg sans vérif",
        "symptom": "Variantes md/lg inventées",
        "root_cause": "Généralisation depuis snippet partiel",
        "correction": "Vérifier chaque variante dans la source web",
        "embedding": [...],  # pour retrieval sémantique
        "occurrences": 3,
        "first_seen": 1721234567.89,
        "last_seen": 1721998765.43,
    }

Inspiration neuroscientifique
-----------------------------
Dans le cerveau, les erreurs de prédiction négatives (dopamine drop)
sont aussi importantes que les positives pour l'apprentissage. Le
cortex cingulaire antérieur (ACC) signale les conflits/erreurs et
oriente l'apprentissage vers les patterns à éviter. C'est l'analogue
qu'on modélise ici.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("rune.memory.failure_memory")


MAX_CONTEXT_CHARS = 300
MAX_ACTION_CHARS = 200
MAX_SYMPTOM_CHARS = 200
MAX_ROOT_CAUSE_CHARS = 200
MAX_CORRECTION_CHARS = 300
MAX_FAILURES = 100


@dataclass
class FailurePattern:
    """Un pattern d'échec appris (anti-skill)."""
    failure_id: str
    context: str  # type de tâche où l'échec est survenu
    attempted_action: str  # ce qu'on a tenté et qui a échoué
    symptom: str  # manifestation observable de l'échec
    root_cause: str  # cause racine (hypothèse)
    correction: str  # ce qu'il aurait fallu faire
    embedding: list[float] = field(default_factory=list)
    occurrences: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    related_skill_id: str | None = None  # si lié à un Skill spécifique

    def severity(self) -> float:
        """Score de sévérité pour le ranking lors de l'injection.

        Plus c'est haut, plus l'anti-pattern est important à rappeler.
        Combinaison : occurrences × fraîcheur.
        """
        days_since = (time.time() - self.last_seen) / 86400
        freshness = max(0.2, 1.0 - days_since / 30)
        return math.log1p(self.occurrences) * freshness

    def as_dict(self) -> dict:
        return asdict(self)

    def as_warning_block(self) -> str:
        """Formate l'anti-pattern pour injection dans le prompt.

        Convention : préfixe ⚠️ + directive d'action (cf. Lythea
        BACKLOG V4 — warnings actionnables).
        """
        return (
            f"⚠️ Anti-pattern connu pour ce type de tâche :\n"
            f"   Contexte : {self.context}\n"
            f"   À éviter : {self.attempted_action}\n"
            f"   Symptôme : {self.symptom}\n"
            f"   → Préférer : {self.correction}"
        )


class FailureMemory:
    """Stockage persistant des FailurePatterns.

    Persistence : JSON simple à ``data/failures/failures.json``.
    Pas de vector store dédié — la dédup est faite via similarité
    cosinus sur l'embedding du contexte.
    """

    def __init__(self, storage_dir: Path | str = "data/failures") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.storage_dir / "failures.json"
        self._failures: dict[str, FailurePattern] = {}
        self._load()

    # ── API publique ──────────────────────────────────────────────────

    def add(self, pattern: FailurePattern) -> FailurePattern:
        """Ajoute un pattern. Si similaire existe, incrémente occurrences."""
        # ── Garde-fou sécurité ─────────────────────────────────────────
        # Les FailurePatterns sont RÉINJECTÉS dans le prompt système via
        # as_warning_block() à chaque tour similaire. Sans ce contrôle,
        # un contenu malveillant (venant du message utilisateur repris
        # dans context/attempted_action, ou d'une sortie LLM) devient une
        # injection persistante. Même politique que AutoSkillStore.
        from rune.memory.auto_skill import _content_is_forbidden, sanitize_for_prompt
        full_text = " ".join([
            pattern.context, pattern.attempted_action, pattern.symptom,
            pattern.root_cause, pattern.correction,
        ])
        if _content_is_forbidden(full_text):
            log.warning(
                "Failure pattern %s rejected: forbidden content",
                pattern.failure_id,
            )
            return pattern

        # Neutralise la structure (sauts de ligne / tokens de chat) et
        # borne les champs — défense en profondeur avant persistance.
        pattern.context = sanitize_for_prompt(pattern.context, MAX_CONTEXT_CHARS)
        pattern.attempted_action = sanitize_for_prompt(
            pattern.attempted_action, MAX_ACTION_CHARS
        )
        pattern.symptom = sanitize_for_prompt(pattern.symptom, MAX_SYMPTOM_CHARS)
        pattern.root_cause = sanitize_for_prompt(
            pattern.root_cause, MAX_ROOT_CAUSE_CHARS
        )
        pattern.correction = sanitize_for_prompt(
            pattern.correction, MAX_CORRECTION_CHARS
        )

        # Dédup par similarité
        if pattern.embedding:
            similar = self._find_similar(pattern.embedding, threshold=0.80)
            if similar:
                # _find_similar retourne une liste — on prend le premier
                existing = similar[0]
                existing.occurrences += 1
                existing.last_seen = time.time()
                # Merge correction si différente
                if pattern.correction and pattern.correction != existing.correction:
                    existing.correction = (
                        existing.correction + " | " + pattern.correction
                    )[:MAX_CORRECTION_CHARS]
                self._save()
                log.info("Failure pattern %s updated (occurrences=%d)",
                         existing.failure_id, existing.occurrences)
                return existing

        # Nouveau pattern
        if len(self._failures) >= MAX_FAILURES:
            # Supprime le moins sévère
            least = min(self._failures.values(), key=lambda f: f.severity())
            del self._failures[least.failure_id]
        self._failures[pattern.failure_id] = pattern
        self._save()
        log.info("Failure pattern %s added (context=%r)",
                 pattern.failure_id, pattern.context[:60])
        return pattern

    def find_by_embedding(
        self, embedding: list[float], threshold: float = 0.65, top_k: int = 3
    ) -> list[FailurePattern]:
        """Retourne les anti-patterns les plus pertinents pour ce contexte."""
        return self._find_similar(embedding, threshold=threshold, top_k=top_k)

    def all(self) -> list[FailurePattern]:
        return list(self._failures.values())

    def get(self, failure_id: str) -> FailurePattern | None:
        return self._failures.get(failure_id)

    def stats(self) -> dict:
        return {
            "total": len(self._failures),
            "max": MAX_FAILURES,
            "by_severity": {
                "high": sum(1 for f in self._failures.values() if f.severity() > 1.0),
                "medium": sum(
                    1 for f in self._failures.values() if 0.5 < f.severity() <= 1.0
                ),
                "low": sum(1 for f in self._failures.values() if f.severity() <= 0.5),
            },
        }

    def as_warning_block(
        self, embedding: list[float] | None = None, top_k: int = 2
    ) -> str:
        """Retourne un bloc de warnings pour les anti-patterns pertinents.

        Si ``embedding`` est fourni, on filtre par similarité. Sinon on
        prend les plus sévères récents.
        """
        if not self._failures:
            return ""
        if embedding:
            patterns = self.find_by_embedding(embedding, top_k=top_k)
        else:
            patterns = sorted(
                self._failures.values(),
                key=lambda f: f.severity(),
                reverse=True,
            )[:top_k]
        if not patterns:
            return ""
        blocks = [p.as_warning_block() for p in patterns]
        return "[ANTI-PATTERNS CONNUS]\n" + "\n\n".join(blocks)

    # ── Internes ──────────────────────────────────────────────────────

    def _find_similar(
        self,
        embedding: list[float],
        threshold: float = 0.80,
        top_k: int = 1,
    ) -> list[FailurePattern]:
        if not embedding:
            return []
        results: list[tuple[float, FailurePattern]] = []
        for pattern in self._failures.values():
            if not pattern.embedding:
                continue
            sim = _cosine_similarity(embedding, pattern.embedding)
            if sim >= threshold:
                results.append((sim, pattern))
        results.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in results[:top_k]]

    def _save(self) -> None:
        # _to_native() convertit les scalaires numpy/torch (embedding,
        # occurrences venant de calculs) en types Python natifs, sinon
        # json.dump peut lever « not JSON serializable » et aucun
        # failure pattern n'est persisté.
        data = {
            "version": 1,
            "failures": [_to_native(f.as_dict()) for f in self._failures.values()],
        }
        tmp = self._index_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._index_path)

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        from rune.memory.auto_skill import _content_is_forbidden
        try:
            with self._index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("failures", []):
                try:
                    pattern = FailurePattern(**{
                        k: v for k, v in entry.items()
                        if k in FailurePattern.__dataclass_fields__  # type: ignore[attr-defined]
                    })
                    # Quarantaine au chargement : un failures.json édité à
                    # la main (ou hérité d'une version sans filtre) ne doit
                    # pas réinjecter d'instruction interdite dans le prompt.
                    full = " ".join([
                        pattern.context, pattern.attempted_action,
                        pattern.symptom, pattern.root_cause, pattern.correction,
                    ])
                    if _content_is_forbidden(full):
                        log.warning(
                            "Failure pattern %s skipped at load: forbidden content",
                            pattern.failure_id,
                        )
                        continue
                    self._failures[pattern.failure_id] = pattern
                except Exception as exc:
                    log.warning("Failed to load failure entry: %s", exc)
        except Exception:
            log.exception("Failed to load failures index")


# ── Helper ────────────────────────────────────────────────────────────


def _to_native(obj):
    """Convertit récursivement numpy/torch scalars en types natifs.

    Voir rune.memory.auto_skill._to_native — même rôle : éviter que
    json.dump casse sur np.float32/torch.Tensor à la persistance.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# ── FailureAnalyzer — propose un pattern à partir d'un échec ──────────


class FailureAnalyzer:
    """Analyse un échec et propose un FailurePattern.

    Comme pour SkillExtractor, on a deux modes :
    - heuristique (sans LLM, pour tests)
    - LLM-based (pour prod, via set_llm_callback)
    """

    def __init__(self) -> None:
        self._llm_callback: callable | None = None  # type: ignore[assignment]

    def set_llm_callback(self, callback: callable) -> None:  # type: ignore[assignment]
        self._llm_callback = callback

    def analyze(
        self,
        context: str,
        attempted_action: str,
        verifier_reasons: list[str],
        user_message: str,
        assistant_response: str,
        context_embedding: list[float] | None = None,
    ) -> FailurePattern | None:
        """Analyse un échec et propose un FailurePattern.

        Retourne None si l'échec n'est pas apprenant (trop trivial,
        pas de cause racine identifiable).
        """
        if not verifier_reasons:
            return None

        if self._llm_callback is not None:
            extracted = self._analyze_via_llm(
                context, attempted_action, verifier_reasons,
                user_message, assistant_response,
            )
        else:
            extracted = self._analyze_heuristic(
                context, attempted_action, verifier_reasons,
                assistant_response,
            )

        if not extracted:
            return None

        failure_id = self._make_id(context, attempted_action)
        return FailurePattern(
            failure_id=failure_id,
            context=context[:MAX_CONTEXT_CHARS],
            attempted_action=attempted_action[:MAX_ACTION_CHARS],
            symptom=extracted.get("symptom", " | ".join(verifier_reasons))[:MAX_SYMPTOM_CHARS],
            root_cause=extracted.get("root_cause", "Inconnue")[:MAX_ROOT_CAUSE_CHARS],
            correction=extracted.get("correction", "À déterminer")[:MAX_CORRECTION_CHARS],
            embedding=context_embedding or [],
            occurrences=1,
        )

    def _analyze_heuristic(
        self,
        context: str,
        attempted_action: str,
        verifier_reasons: list[str],
        assistant_response: str,
    ) -> dict | None:
        """Analyse heuristique sans LLM.

        Tente de catégoriser l'échec selon des patterns connus.
        """
        reasons_text = " ".join(verifier_reasons).lower()

        # Catégorisation simple
        if "trop court" in reasons_text:
            return {
                "symptom": "Réponse trop courte",
                "root_cause": "Manque de développement",
                "correction": "Détailler davantage la réponse avec des exemples",
            }
        if "esquive" in reasons_text:
            return {
                "symptom": "Formule d'esquive détectée",
                "root_cause": "Tentative de masquer un manque de connaissances",
                "correction": "Admettre le manque et chercher via web/RAG",
            }
        if "source" in reasons_text or "url" in reasons_text:
            return {
                "symptom": "Aucune source citée",
                "root_cause": "Réponse non sourcée",
                "correction": "Lier chaque fait à une source [N]",
            }
        if "structure" in reasons_text:
            return {
                "symptom": "Réponse peu structurée",
                "root_cause": "Absence de plan ou de points distincts",
                "correction": "Structurer en points numérotés ou bullet list",
            }

        # Cas générique
        return {
            "symptom": verifier_reasons[0][:MAX_SYMPTOM_CHARS],
            "root_cause": "À analyser",
            "correction": "Réessayer avec un contexte enrichi",
        }

    def _analyze_via_llm(
        self,
        context: str,
        attempted_action: str,
        verifier_reasons: list[str],
        user_message: str,
        assistant_response: str,
    ) -> dict | None:
        prompt = f"""Tu es un analyste d'échecs. À partir d'un échec
vérifié, identifie la cause racine et la correction à mémoriser.

Épisode échoué :
- Contexte : {context!r}
- Action tentée : {attempted_action!r}
- Demande utilisateur : {user_message!r}
- Réponse produite : {assistant_response!r}
- Raisons de l'échec (verifier) : {verifier_reasons!r}

Produis UNIQUEMENT un JSON valide avec ces clés :
- "symptom" : manifestation observable (≤200 chars)
- "root_cause" : hypothèse sur la cause profonde (≤200 chars)
- "correction" : ce qu'il aurait fallu faire (≤300 chars)

Si l'échec n'est pas apprenant (trop trivial), retourne {{"skip": true}}.
"""
        try:
            result = self._llm_callback(prompt)  # type: ignore[misc]
            if not result or result.get("skip"):
                return None
            return result
        except Exception:
            log.exception("LLM analysis failed")
            return None

    @staticmethod
    def _make_id(context: str, action: str) -> str:
        h = hashlib.sha256(
            (context + "|" + action).encode("utf-8")
        ).hexdigest()[:12]
        return f"fail_{h}"
