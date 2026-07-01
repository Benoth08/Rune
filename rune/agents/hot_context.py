"""HotContext — sérialise le contexte mémoire pertinent pour les sous-agents.

Solution A pour la mémoire partagée entre parent et sous-agents :

Le parent (RuneCortex) récupère les chunks mémoire pertinents pour la
tâche du sous-agent, les sérialise en JSON, et les passe au sous-agent
via le payload stdin. Le sous-agent les injecte dans son system prompt.

C'est une solution LECTURE SEULE — le sous-agent ne peut pas écrire en
mémoire. L'écriture reste côté parent (AutoSkill/FailureMemory tournent
après le retour du sous-agent).

Pourquoi lecture seule ?
- Évite la corruption mémoire (pas de concurrence sur SDM/MHN/KG/Chroma)
- Évite la duplication VRAM (pas besoin de charger SDM/MHN dans le subprocess)
- Évite la latence ChromaDB (~2s de load par subprocess)

Contenu du HotContext
---------------------
1. **rag_chunks** : chunks récupérés via TieredRetriever (Core → SDM → MHN → KG → Chroma)
2. **skills** : skills applicables (AutoSkillStore.find_by_trigger_embedding)
3. **anti_patterns** : anti-patterns pertinents (FailureMemory.find_by_embedding)
4. **kg_entities** : entités KG mentionnées dans la tâche (optionnel)
5. **task_embedding** : embedding de la tâche (pour référence)

Le tout est sérialisé en JSON compact, avec un budget de tokens borné
pour ne pas exploser le contexte du sous-agent.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger("rune.agents.hot_context")


# Budgets de taille pour éviter d'exploser le contexte du sous-agent
MAX_RAG_CHUNKS = 5
MAX_RAG_CHUNK_CHARS = 500
MAX_SKILLS = 3
MAX_SKILL_CHARS = 300
MAX_ANTI_PATTERNS = 3
MAX_ANTI_PATTERN_CHARS = 200
MAX_KG_ENTITIES = 5
MAX_KG_ENTITY_CHARS = 200
MAX_TOTAL_CHARS = 4000  # budget global pour le hot_context sérialisé


@dataclass
class HotContext:
    """Contexte mémoire chaud transmis au sous-agent.

    Tous les champs sont optionnels — si la mémoire est vide (premier tour,
    modèle non chargé, etc.), le HotContext est vide et le sous-agent
    travaille sans contexte.
    """
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    anti_patterns: list[dict[str, Any]] = field(default_factory=list)
    kg_entities: list[dict[str, Any]] = field(default_factory=list)
    task_embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_prompt_block(self, max_chars: int = MAX_TOTAL_CHARS) -> str:
        """Sérialise le HotContext en un bloc de prompt injectable.

        Format :
            [CONTEXTE MÉMOIRE]
            [RAG] chunk1 | chunk2 | ...
            [SKILLS] skill1 trigger + approach
            [ANTI-PATTERNS] ⚠️ à éviter : ...
            [KG ENTITIES] entity1 (type) : summary
        """
        if not self._has_content():
            return ""

        lines: list[str] = ["[CONTEXTE MÉMOIRE]"]
        total = len(lines[0])

        # RAG chunks
        if self.rag_chunks:
            rag_lines = ["[RAG]"]
            for chunk in self.rag_chunks:
                content = str(chunk.get("content", ""))[:MAX_RAG_CHUNK_CHARS]
                source = chunk.get("kind", "unknown")
                line = f"  ({source}) {content}"
                if total + len(line) > max_chars:
                    break
                rag_lines.append(line)
                total += len(line)
            lines.extend(rag_lines)

        # Skills
        if self.skills:
            skill_lines = ["[SKILLS APPLICABLES]"]
            for skill in self.skills:
                trigger = str(skill.get("trigger", ""))[:100]
                approach = " | ".join(skill.get("approach", []))[:MAX_SKILL_CHARS]
                line = f"  trigger: {trigger}"
                line2 = f"  approach: {approach}"
                if total + len(line) + len(line2) > max_chars:
                    break
                skill_lines.append(line)
                skill_lines.append(line2)
                total += len(line) + len(line2)
            lines.extend(skill_lines)

        # Anti-patterns
        if self.anti_patterns:
            ap_lines = ["[ANTI-PATTERNS À ÉVITER]"]
            for ap in self.anti_patterns:
                context = str(ap.get("context", ""))[:100]
                correction = str(ap.get("correction", ""))[:MAX_ANTI_PATTERN_CHARS]
                line = f"  ⚠️ {context} → préférer: {correction}"
                if total + len(line) > max_chars:
                    break
                ap_lines.append(line)
                total += len(line)
            lines.extend(ap_lines)

        # KG entities
        if self.kg_entities:
            kg_lines = ["[ENTITÉS CONNUES]"]
            for ent in self.kg_entities:
                name = str(ent.get("name", ent.get("value", "")))[:50]
                etype = ent.get("type", "")
                summary = str(ent.get("summary", ""))[:MAX_KG_ENTITY_CHARS]
                line = f"  {name} ({etype}): {summary}"
                if total + len(line) > max_chars:
                    break
                kg_lines.append(line)
                total += len(line)
            lines.extend(kg_lines)

        if len(lines) <= 1:
            return ""

        return "\n".join(lines)

    def _has_content(self) -> bool:
        return bool(
            self.rag_chunks
            or self.skills
            or self.anti_patterns
            or self.kg_entities
        )


class HotContextSerializer:
    """Construit un HotContext à partir des mémoires du parent.

    Parameters
    ----------
    tiered_retriever : TieredRetriever | None
        Pour récupérer les chunks RAG pertinents.
    skills_store : AutoSkillStore | None
        Pour récupérer les skills applicables.
    failures_store : FailureMemory | None
        Pour récupérer les anti-patterns pertinents.
    kg : KnowledgeGraphStore | None
        Pour récupérer les entités mentionnées dans la tâche.
    embed_fn : callable | None
        Fonction pour calculer l'embedding de la tâche (pour la similarité).
        Si None, on utilise l'embedding déjà stocké dans les skills/failures
        (moins précis mais fonctionne sans backend modèle).
    """

    def __init__(
        self,
        tiered_retriever: Any = None,
        skills_store: Any = None,
        failures_store: Any = None,
        kg: Any = None,
        embed_fn: Any = None,
    ) -> None:
        self.tiered_retriever = tiered_retriever
        self.skills_store = skills_store
        self.failures_store = failures_store
        self.kg = kg
        self.embed_fn = embed_fn

    def build(
        self,
        task: str,
        max_rag_chunks: int = MAX_RAG_CHUNKS,
        max_skills: int = MAX_SKILLS,
        max_anti_patterns: int = MAX_ANTI_PATTERNS,
    ) -> HotContext:
        """Construit un HotContext pour la tâche donnée.

        Ne lève jamais — si une mémoire est indisponible, on continue
        avec les autres. Le HotContext peut être vide si toutes les
        mémoires sont indisponibles.
        """
        context = HotContext()
        context.metadata["task"] = task[:200]

        # Calcule l'embedding de la tâche (pour similarité skills/failures)
        task_embedding: list[float] = []
        if self.embed_fn is not None:
            try:
                emb = self.embed_fn(task)
                if emb:
                    task_embedding = list(emb) if not hasattr(emb, "tolist") else emb.tolist()
                    context.task_embedding = task_embedding
            except Exception as exc:
                log.debug("embed_fn failed: %s", exc)

        # 1. RAG chunks via TieredRetriever
        if self.tiered_retriever is not None:
            try:
                result = self.tiered_retriever.retrieve(
                    query=task,
                    query_embedding=task_embedding or None,
                    doubt_index=0.5,  # force un peu de retrieval
                )
                for chunk in result.chunks[:max_rag_chunks]:
                    context.rag_chunks.append({
                        "kind": chunk.kind,
                        "content": chunk.content[:MAX_RAG_CHUNK_CHARS],
                        "relevance": round(chunk.relevance, 3),
                        "source": chunk.metadata.get("source", "unknown"),
                    })
            except Exception as exc:
                log.debug("TieredRetriever failed: %s", exc)

        # 2. Skills applicables
        if self.skills_store is not None and task_embedding:
            try:
                skills = self.skills_store.find_by_trigger_embedding(
                    task_embedding, threshold=0.7, top_k=max_skills
                )
                for skill in skills:
                    context.skills.append({
                        "skill_id": skill.skill_id,
                        "trigger": skill.trigger,
                        "approach": skill.approach[:3],  # top 3 étapes
                        "validation": skill.validation[:2],
                        "confidence": round(skill.confidence, 3),
                        "success_count": skill.success_count,
                    })
            except Exception as exc:
                log.debug("Skills retrieval failed: %s", exc)

        # 3. Anti-patterns pertinents
        if self.failures_store is not None and task_embedding:
            try:
                patterns = self.failures_store.find_by_embedding(
                    task_embedding, threshold=0.6, top_k=max_anti_patterns
                )
                for pattern in patterns:
                    context.anti_patterns.append({
                        "failure_id": pattern.failure_id,
                        "context": pattern.context,
                        "attempted_action": pattern.attempted_action,
                        "symptom": pattern.symptom,
                        "correction": pattern.correction,
                        "occurrences": pattern.occurrences,
                    })
            except Exception as exc:
                log.debug("FailureMemory retrieval failed: %s", exc)

        # 4. KG entities (si KG dispo)
        if self.kg is not None:
            try:
                # Lythea KG expose query_by_question
                entities = self.kg.query_by_question(task)
                for ent in (entities or [])[:MAX_KG_ENTITIES]:
                    if isinstance(ent, dict):
                        context.kg_entities.append({
                            "name": ent.get("name", ent.get("value", "")),
                            "type": ent.get("type", ""),
                            "summary": ent.get("summary", ""),
                            "confidence": ent.get("confidence", 0.5),
                        })
            except Exception as exc:
                log.debug("KG query failed: %s", exc)

        log.info(
            "HotContext built: %d chunks, %d skills, %d anti-patterns, %d entities",
            len(context.rag_chunks),
            len(context.skills),
            len(context.anti_patterns),
            len(context.kg_entities),
        )
        return context
