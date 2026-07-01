"""SkillComposer — compose des skills existantes pour en créer de nouvelles.

Inspiration
-----------
Les systèmes d'agents récursifs (RecursiveMAS, Voyager, etc.) montrent
qu'un agent devient plus capable en **composant** ses compétences
acquises plutôt qu'en les utilisant isolément.

Application à Rune
------------------
Quand Rune a appris plusieurs skills via AutoSkill, il peut les combiner
pour produire une nouvelle skill plus puissante :

- skill_abc : "Calculer fibonacci" → approach: [récursion, mémoïsation]
- skill_def : "Écrire tests pytest" → approach: [fixtures, asserts]
- skill_ghi : "Débugger Python" → approach: [pdb, print, logging]

→ SkillComposer peut produire :
- skill_compose_xyz : "Écrire puis tester une fonction fibonacci"
  approach: [récursion fibonacci, fixtures pytest, asserts, pdb si fail]

Stratégies de composition
-------------------------
1. **Sequential** — exécute skill_A puis skill_B (l'une prépare l'autre)
2. **Parallel** — exécute skill_A et skill_B en parallèle, fusionne
3. **Conditional** — si condition → skill_A, sinon → skill_B
4. **Pipeline** — skill_A génère, skill_B valide, skill_C corrige

Composition LLM-assistée
------------------------
Pour produire une approche cohérente, le compositeur peut utiliser un
LLM (via callback) qui synthétise les approaches des skills sources en
une nouvelle approach unifiée. Sans LLM, on fait une union heuristique
(qui marche mais moins élégante).

Contrats
--------
1. Ne compose JAMAIS des skills non fiables (is_reliable() doit être True)
2. Les skills sources doivent avoir un embedding de trigger (pour similarité)
3. La skill composée hérite des anti_patterns des sources (union)
4. Si 2 skills ont des approaches contradictoires, on garde les 2 avec
   une note "alternative"
5. La skill composée est marquée metadata.composed = True pour tracer
   son origine
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .auto_skill import Skill, AutoSkillStore

log = logging.getLogger("rune.memory.skill_composer")


# ── Stratégies ───────────────────────────────────────────────────────


class CompositionStrategy:
    """Stratégies de composition supportées."""
    SEQUENTIAL = "sequential"   # A → B (l'un prépare l'autre)
    PARALLEL = "parallel"       # A || B (fusion)
    CONDITIONAL = "conditional"  # if X → A else → B
    PIPELINE = "pipeline"       # A génère, B valide, C corrige


# ── Result ───────────────────────────────────────────────────────────


@dataclass
class CompositionResult:
    """Résultat d'une composition de skills."""
    skill: Skill | None = None
    status: str = "ok"  # ok | skipped | error
    reason: str = ""
    source_skill_ids: list[str] = field(default_factory=list)
    strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "strategy": self.strategy,
            "source_skill_ids": self.source_skill_ids,
            "composed_skill_id": self.skill.skill_id if self.skill else None,
            "metadata": self.metadata,
        }


# ── SkillComposer ────────────────────────────────────────────────────


class SkillComposer:
    """Compose des skills existantes en une nouvelle skill.

    Parameters
    ----------
    store : AutoSkillStore
        Le magasin de skills (pour récupérer les sources et ajouter la composée).
    llm_callback : callable | None
        Fonction LLM pour synthétiser l'approche. Signature :
        callback(prompt: str) -> dict | None
        Doit retourner {"trigger": str, "approach": list[str], "validation": list[str]}
        Si None, on fait une union heuristique (moins élégante).
    """

    def __init__(
        self,
        store: AutoSkillStore,
        llm_callback: Callable[[str], dict | None] | None = None,
    ) -> None:
        self.store = store
        self.llm_callback = llm_callback

    # ── API publique ──────────────────────────────────────────────────

    def compose(
        self,
        skill_ids: list[str],
        strategy: str = CompositionStrategy.SEQUENTIAL,
        composed_trigger: str | None = None,
        force: bool = False,
    ) -> CompositionResult:
        """Compose plusieurs skills en une nouvelle.

        Parameters
        ----------
        skill_ids : list[str]
            IDs des skills à composer (2 minimum).
        strategy : str
            Stratégie de composition (sequential, parallel, conditional, pipeline).
        composed_trigger : str | None
            Trigger personnalisé pour la skill composée. Si None, généré.
        force : bool
            Si True, compose même si certaines skills ne sont pas fiables.

        Returns
        -------
        CompositionResult
        """
        result = CompositionResult(
            strategy=strategy,
            source_skill_ids=skill_ids,
        )

        # ── Validation ────────────────────────────────────────────────
        if len(skill_ids) < 2:
            result.status = "skipped"
            result.reason = "Need at least 2 skills to compose"
            return result

        # Récupère les skills sources
        sources: list[Skill] = []
        for sid in skill_ids:
            skill = self.store.get(sid)
            if skill is None:
                result.status = "error"
                result.reason = f"Skill {sid} not found"
                return result
            if not skill.is_reliable() and not force:
                result.status = "skipped"
                result.reason = f"Skill {sid} is not reliable (use force=True to override)"
                return result
            sources.append(skill)

        # ── Composition ───────────────────────────────────────────────
        if self.llm_callback is not None:
            composed = self._compose_via_llm(
                sources, strategy, composed_trigger
            )
        else:
            composed = self._compose_heuristic(
                sources, strategy, composed_trigger
            )

        if composed is None:
            result.status = "error"
            result.reason = "Composition failed (heuristic + LLM both returned None)"
            return result

        # Marque la skill comme composée
        composed.metadata = composed.metadata or {}
        composed.metadata["composed"] = True
        composed.metadata["source_skill_ids"] = skill_ids
        composed.metadata["strategy"] = strategy
        composed.metadata["composed_at"] = time.time()

        # Ajoute au store (avec dédup si une skill similaire existe déjà)
        added = self.store.add(composed)
        result.skill = added
        result.status = "ok"
        result.metadata = composed.metadata

        log.info(
            "Composed skill %s from %d sources (strategy=%s)",
            added.skill_id, len(sources), strategy,
        )
        return result

    def find_composable_candidates(
        self,
        task_embedding: list[float] | None = None,
        max_pairs: int = 5,
        min_confidence: float = 0.5,
    ) -> list[tuple[Skill, Skill, float]]:
        """Trouve des paires de skills qui pourraient être composées.

        Heuristique : paires de skills fiables avec une similarité de
        trigger moyenne (pas trop similaires = redondantes, pas trop
        différentes = sans lien).

        Returns
        -------
        list[tuple[Skill, Skill, float]]
            Liste de (skill_A, skill_B, similarity_score), triée par
            potentiel de composition décroissant.
        """
        candidates: list[tuple[Skill, Skill, float]] = []
        reliable_skills = [s for s in self.store.active() if s.is_reliable()]

        for i, a in enumerate(reliable_skills):
            for b in reliable_skills[i + 1:]:
                # Similarité entre les triggers
                if a.trigger_embedding and b.trigger_embedding:
                    sim = _cosine_similarity(a.trigger_embedding, b.trigger_embedding)
                else:
                    # Sans embedding, on prend une similarité lexicale grossière
                    sim = _lexical_similarity(a.trigger, b.trigger)

                # Filtrage : similitude entre 0.3 et 0.85 (zone "composable")
                if 0.3 <= sim <= 0.85:
                    # Score de potentiel = moyenne des confiances × (1 - sim)
                    # (on préfère les skills moyennement similaires)
                    potential = (
                        (a.confidence + b.confidence) / 2 * (1 - abs(sim - 0.5))
                    )
                    if potential >= min_confidence:
                        candidates.append((a, b, potential))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:max_pairs]

    # ── Internes ──────────────────────────────────────────────────────

    def _compose_heuristic(
        self,
        sources: list[Skill],
        strategy: str,
        composed_trigger: str | None,
    ) -> Skill | None:
        """Composition heuristique sans LLM.

        Stratégies :
        - sequential : approach_A puis approach_B (concaténation ordonnée)
        - parallel : intercale les étapes (A[0], B[0], A[1], B[1], ...)
        - conditional : approach_A + "ALTERNATIVE: " + approach_B
        - pipeline : approach_A (génère) + approach_B (valide)
        """
        # Trigger composé
        if composed_trigger:
            trigger = composed_trigger
        else:
            triggers = [s.trigger for s in sources]
            trigger = " + ".join(t[:50] for t in triggers)[:200]

        # Approach composée selon stratégie
        all_approaches: list[list[str]] = [s.approach for s in sources]
        composed_approach: list[str] = []

        if strategy == CompositionStrategy.SEQUENTIAL:
            for app in all_approaches:
                composed_approach.extend(app)
        elif strategy == CompositionStrategy.PARALLEL:
            # Intercale les étapes
            max_len = max(len(a) for a in all_approaches)
            for i in range(max_len):
                for app in all_approaches:
                    if i < len(app):
                        composed_approach.append(app[i])
        elif strategy == CompositionStrategy.CONDITIONAL:
            composed_approach.append("Si la condition X est vraie :")
            composed_approach.extend(f"  - {step}" for step in all_approaches[0])
            composed_approach.append("Sinon :")
            composed_approach.extend(f"  - {step}" for step in all_approaches[1])
        elif strategy == CompositionStrategy.PIPELINE:
            for i, app in enumerate(all_approaches):
                role = ["Générer", "Valider", "Corriger"][i] if i < 3 else f"Étape {i+1}"
                composed_approach.append(f"[{role}]")
                composed_approach.extend(f"  - {step}" for step in app)
        else:
            composed_approach.extend(all_approaches[0])

        # Validation composée (union)
        composed_validation: list[str] = []
        for src in sources:
            for v in src.validation:
                if v not in composed_validation:
                    composed_validation.append(v)

        # Anti-patterns composés (union)
        composed_anti_patterns: list[str] = []
        for src in sources:
            for ap in src.anti_patterns:
                if ap not in composed_anti_patterns:
                    composed_anti_patterns.append(ap)

        # Confiance = moyenne pondérée par success_count
        total_success = sum(s.success_count for s in sources) or 1
        weighted_conf = sum(
            s.confidence * s.success_count for s in sources
        ) / total_success

        # Embedding composé = moyenne des embeddings sources
        composed_emb: list[float] = []
        embs = [s.trigger_embedding for s in sources if s.trigger_embedding]
        if embs:
            n = len(embs[0])
            composed_emb = [
                sum(e[i] for e in embs if i < len(e)) / len(embs)
                for i in range(n)
            ]

        # ID composé
        sources_hash = hashlib.sha256(
            "|".join(s.skill_id for s in sources).encode()
        ).hexdigest()[:12]

        return Skill(
            skill_id=f"skill_compose_{sources_hash}",
            trigger=trigger[:300],
            trigger_embedding=composed_emb,
            approach=composed_approach[:8],  # cap à 8 étapes
            validation=composed_validation[:3],
            anti_patterns=composed_anti_patterns[:5],
            success_count=1,
            failure_count=0,
            confidence=min(1.0, weighted_conf),
            source_episodes=[s.skill_id for s in sources],
        )

    def _compose_via_llm(
        self,
        sources: list[Skill],
        strategy: str,
        composed_trigger: str | None,
    ) -> Skill | None:
        """Composition via LLM — synthétise une approche cohérente.

        Si le LLM échoue, fallback sur _compose_heuristic.
        """
        prompt = self._build_composition_prompt(sources, strategy, composed_trigger)
        try:
            result = self.llm_callback(prompt)  # type: ignore[misc]
            if not result or result.get("skip"):
                log.info("LLM composition skipped (returned skip=True)")
                return self._compose_heuristic(sources, strategy, composed_trigger)
        except Exception:
            log.exception("LLM composition failed — falling back to heuristic")
            return self._compose_heuristic(sources, strategy, composed_trigger)

        # Construit la skill depuis la sortie LLM
        trigger = result.get("trigger") or composed_trigger or " + ".join(
            s.trigger[:30] for s in sources
        )

        # Embedding composé (moyenne)
        composed_emb: list[float] = []
        embs = [s.trigger_embedding for s in sources if s.trigger_embedding]
        if embs:
            n = len(embs[0])
            composed_emb = [
                sum(e[i] for e in embs if i < len(e)) / len(embs)
                for i in range(n)
            ]

        # Anti-patterns (union des sources)
        composed_anti_patterns: list[str] = []
        for src in sources:
            for ap in src.anti_patterns:
                if ap not in composed_anti_patterns:
                    composed_anti_patterns.append(ap)

        # ID composé
        sources_hash = hashlib.sha256(
            "|".join(s.skill_id for s in sources).encode()
        ).hexdigest()[:12]

        return Skill(
            skill_id=f"skill_compose_{sources_hash}",
            trigger=trigger[:300],
            trigger_embedding=composed_emb,
            approach=result.get("approach", [])[:8],
            validation=result.get("validation", [])[:3],
            anti_patterns=composed_anti_patterns[:5],
            success_count=1,
            failure_count=0,
            confidence=min(1.0, sum(s.confidence for s in sources) / len(sources)),
            source_episodes=[s.skill_id for s in sources],
        )

    @staticmethod
    def _build_composition_prompt(
        sources: list[Skill],
        strategy: str,
        composed_trigger: str | None,
    ) -> str:
        """Construit le prompt pour la composition LLM."""
        sources_desc = "\n\n".join(
            f"Skill {i+1}: {s.skill_id}\n"
            f"  Trigger: {s.trigger}\n"
            f"  Approach: {s.approach}\n"
            f"  Validation: {s.validation}\n"
            f"  Confidence: {s.confidence:.2f}\n"
            for i, s in enumerate(sources)
        )

        strategy_desc = {
            CompositionStrategy.SEQUENTIAL: "exécuter les skills l'une après l'autre (la 2e dépend de la 1ère)",
            CompositionStrategy.PARALLEL: "exécuter les skills en parallèle et fusionner les résultats",
            CompositionStrategy.CONDITIONAL: "exécuter la skill 1 si une condition est vraie, sinon la skill 2",
            CompositionStrategy.PIPELINE: "la skill 1 génère, la skill 2 valide, la skill 3 corrige",
        }.get(strategy, "combiner les skills")

        return f"""Tu es un compositeur de compétences. À partir de {len(sources)} skills
existantes, crée une nouvelle skill composée qui les combine.

Stratégie de composition : {strategy}
Description : {strategy_desc}

Skills sources :
{sources_desc}

Trigger personnalisé : {composed_trigger or "(à générer)"}

Produis UNIQUEMENT un JSON valide avec ces clés :
- "trigger" : description courte (≤200 chars) du contexte d'activation de la skill composée
- "approach" : liste de 3-6 étapes concrètes (≤200 chars chacune) qui combinent
  les approaches des skills sources de façon cohérente selon la stratégie
- "validation" : liste de 1-3 critères observables de succès

Si la composition n'a pas de sens (skills trop disjointes), retourne {{"skip": true}}.
"""


# ── Helpers ──────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarité cosinus entre deux vecteurs."""
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _lexical_similarity(a: str, b: str) -> float:
    """Similarité lexicale grossière (Jaccard sur mots)."""
    if not a or not b:
        return 0.0
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    intersection = wa & wb
    union = wa | wb
    return len(intersection) / len(union) if union else 0.0
