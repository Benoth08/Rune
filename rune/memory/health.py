"""Memory Health — V5.3 dashboard des métriques cognitives.

Calcule un score de santé global de la mémoire de Lythéa à partir
de 5 dimensions inspirées du pattern openclaw-auto-dream (mars 2026) :

- freshness   : taux d'entrées récentes (< 30j) dans Chroma
- coverage    : taux d'entités KG ayant au moins 1 relation
- coherence   : densité moyenne des communautés GraphRAG (V5.2)
- efficiency  : ratio entités actives / pending+actives
- reachability: connectivité du graphe (taille du plus grand
  composant connexe / total entités)

Score global = moyenne pondérée des 5 dimensions × 100.

Pondérations :
- freshness   25% (le plus mesurable)
- coverage    25% (signale la richesse du KG)
- coherence   20% (qualité des communautés)
- efficiency  15% (ménage interne)
- reachability 15% (intégration)

Endpoint exposé : ``GET /api/memory/health``
Cognitive item UI : injecté optionnellement en début de session.

Calcul à la demande (pas de tâche périodique) — le coût est < 50ms
sur un KG de ~500 entités. Pas de cache, on prend toujours la photo
actuelle, ce qui est utile pour observer l'évolution post-microsleep.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

log = logging.getLogger("lythea.memory.health")


# ── Dataclass de sortie ────────────────────────────────────────────────


@dataclass
class HealthSnapshot:
    """Une photo instantanée des métriques mémoire."""

    # 0-100 scores (entiers pour lisibilité UI)
    freshness: int
    coverage: int
    coherence: int
    efficiency: int
    reachability: int
    # Score global pondéré
    health_score: int
    # Stats brutes pour debug
    n_entities: int
    n_relations: int
    n_communities: int
    n_pending: int
    n_chroma: int
    # Horodatage de la mesure
    measured_at: float
    # Message court à afficher à l'utilisateur (selon le score)
    cognitive_hint: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Calcul des dimensions ──────────────────────────────────────────────


def _compute_freshness(chroma_collection, now: float) -> tuple[int, int]:
    """Pourcentage d'entrées Chroma datées de moins de 30 jours.

    Returns
    -------
    tuple[int, int]
        (score_0_100, n_total). Si Chroma est vide ou inaccessible,
        retourne (0, 0) — on évite de scorer du vide à 100%.
    """
    try:
        n_total = chroma_collection.count()
    except Exception as exc:
        log.warning("Chroma count failed: %s", exc)
        return 0, 0

    if n_total == 0:
        return 0, 0

    # Échantillonnage : on lit jusqu'à 200 metadonnées pour estimer
    # la fraîcheur. Plus que ça ralentirait le dashboard pour un
    # gain marginal. La distribution est généralement uniforme,
    # donc l'estimation est fiable.
    sample_n = min(200, n_total)
    try:
        results = chroma_collection.get(limit=sample_n, include=["metadatas"])
        metadatas = results.get("metadatas", []) or []
    except Exception as exc:
        log.warning("Chroma metadata fetch failed: %s", exc)
        return 50, n_total  # ni 0 ni 100 — incertain

    if not metadatas:
        return 50, n_total

    thirty_days_ago = now - (30 * 86400)
    recent_count = 0
    valid_count = 0
    for meta in metadatas:
        ts = (meta or {}).get("ts", 0)
        if ts and ts > 0:
            valid_count += 1
            if ts >= thirty_days_ago:
                recent_count += 1

    if valid_count == 0:
        # Pas de timestamps — on suppose neutre.
        return 50, n_total

    score = int(round((recent_count / valid_count) * 100))
    return score, n_total


def _compute_coverage(kg) -> tuple[int, int, int]:
    """Pourcentage d'entités ayant au moins 1 relation.

    Returns
    -------
    tuple[int, int, int]
        (score_0_100, n_entities, n_relations).
    """
    if not kg or not getattr(kg, "entities", None):
        return 0, 0, 0

    n_entities = len(kg.entities)
    relations = getattr(kg, "relations", {})
    if isinstance(relations, dict):
        relations_iter = relations.values()
    else:
        relations_iter = relations
    n_relations = len(list(relations_iter))

    if n_entities == 0:
        return 0, 0, 0

    # Reset à un nouveau passage pour éviter d'écraser
    if isinstance(relations, dict):
        relations_iter = relations.values()
    else:
        relations_iter = relations

    connected_ids: set[str] = set()
    for rel in relations_iter:
        subj = getattr(rel, "subject_id", None)
        obj = getattr(rel, "object_id", None)
        if subj:
            connected_ids.add(subj)
        if obj:
            connected_ids.add(obj)

    # On ne compte que les entités existantes (un orphelin de relation
    # qui pointe vers une entité supprimée ne devrait pas booster le
    # score).
    valid_connected = sum(
        1 for eid in connected_ids if eid in kg.entities
    )
    score = int(round((valid_connected / n_entities) * 100))
    return score, n_entities, n_relations


def _compute_coherence(kg) -> tuple[int, int]:
    """Densité moyenne des communautés GraphRAG.

    Si pas de communautés détectées, retourne (0, 0) — on ne peut
    pas évaluer la cohérence sans clustering.

    Returns
    -------
    tuple[int, int]
        (score_0_100, n_communities).
    """
    communities = getattr(kg, "communities", None) or []
    if not communities:
        return 0, 0

    densities = [getattr(c, "density", 0.0) for c in communities]
    if not densities:
        return 0, len(communities)

    avg_density = sum(densities) / len(densities)
    # density est dans [0, 1]. On l'amplifie un peu pour la lisibilité
    # UI car en pratique les communautés naturelles plafonnent à
    # ~0.4-0.6. On mappe [0, 0.6] → [0, 100].
    score = int(round(min(avg_density / 0.6, 1.0) * 100))
    return score, len(communities)


def _compute_efficiency(kg) -> tuple[int, int]:
    """Ratio entités actives / (actives + pending).

    Returns
    -------
    tuple[int, int]
        (score_0_100, n_pending).
    """
    if not kg:
        return 0, 0
    n_active = len(getattr(kg, "entities", {}) or {})
    n_pending = len(getattr(kg, "pending", {}) or {})
    total = n_active + n_pending
    if total == 0:
        return 0, 0
    score = int(round((n_active / total) * 100))
    return score, n_pending


def _compute_reachability(kg) -> int:
    """Taille du plus grand composant connexe / total entités.

    Mesure à quel point le graphe est intégré (forte = un seul gros
    composant) vs fragmenté (faible = beaucoup d'îlots isolés).

    Utilise networkx (déjà dépendance V5.2 GraphRAG).
    """
    entities = getattr(kg, "entities", None) or {}
    if len(entities) < 2:
        return 0

    relations = getattr(kg, "relations", None) or {}
    if isinstance(relations, dict):
        relations_iter = list(relations.values())
    else:
        relations_iter = list(relations)
    if not relations_iter:
        return 0

    try:
        import networkx as nx
    except ImportError:
        return 0

    graph = nx.Graph()
    graph.add_nodes_from(entities.keys())
    for rel in relations_iter:
        subj = getattr(rel, "subject_id", None)
        obj = getattr(rel, "object_id", None)
        if subj and obj and subj in entities and obj in entities:
            graph.add_edge(subj, obj)

    if graph.number_of_edges() == 0:
        return 0

    try:
        components = list(nx.connected_components(graph))
        if not components:
            return 0
        largest = max(len(c) for c in components)
        score = int(round((largest / len(entities)) * 100))
        return score
    except Exception as exc:
        log.warning("Reachability calc failed: %s", exc)
        return 0


# ── Hint texte selon le score ──────────────────────────────────────────


def _hint_for_score(score: int, dims: dict) -> str:
    """Court message UI selon le score global et les points faibles."""
    # Pas de mémoire du tout
    if score == 0:
        return "Mémoire encore vierge — patiente, je m'enrichis au fil des conversations."

    # Identification du point le plus faible (>0 pour éviter de pointer
    # une dimension non-applicable, type coherence sans communautés)
    weakest_dim = None
    weakest_val = 101
    for name, val in dims.items():
        if 0 < val < weakest_val:
            weakest_dim = name
            weakest_val = val

    labels = {
        "freshness": "fraîcheur",
        "coverage": "couverture",
        "coherence": "cohérence",
        "efficiency": "efficience",
        "reachability": "connectivité",
    }

    if score >= 80:
        return f"Mémoire en bonne forme ({score}/100)."
    if score >= 60:
        if weakest_dim:
            return (
                f"Mémoire saine ({score}/100), {labels.get(weakest_dim, weakest_dim)} "
                f"un peu basse ({weakest_val})."
            )
        return f"Mémoire saine ({score}/100)."
    if score >= 40:
        if weakest_dim:
            return (
                f"Mémoire en construction ({score}/100), "
                f"{labels.get(weakest_dim, weakest_dim)} à améliorer."
            )
        return f"Mémoire en construction ({score}/100)."
    return (
        f"Mémoire encore jeune ({score}/100) — "
        f"plus de microsleeps et de conversations consolidées m'aideront."
    )


# ── API publique ───────────────────────────────────────────────────────


def compute_health(
    kg: Any,
    chroma_collection: Any | None = None,
) -> HealthSnapshot:
    """Calcule un snapshot complet de la santé mémoire.

    Parameters
    ----------
    kg
        :class:`KnowledgeGraphStore`. Source des entités, relations,
        communautés.
    chroma_collection
        Chroma collection (optionnel). Si fourni, calcule la
        freshness ; sinon la freshness vaut 0.

    Returns
    -------
    HealthSnapshot
        Photo instantanée. Toujours retournée — même si chaque
        composant est en erreur, on remplit avec des 0.
    """
    now = time.time()

    # Freshness
    if chroma_collection is not None:
        freshness, n_chroma = _compute_freshness(chroma_collection, now)
    else:
        freshness, n_chroma = 0, 0

    # Coverage + stats KG
    coverage, n_entities, n_relations = _compute_coverage(kg)

    # Coherence
    coherence, n_communities = _compute_coherence(kg)

    # Efficiency
    efficiency, n_pending = _compute_efficiency(kg)

    # Reachability
    reachability = _compute_reachability(kg)

    # Score global pondéré.
    # Note : on ne sanctionne pas l'absence de communautés (coherence=0)
    # quand le KG est trop petit pour en avoir. On vérifie avant.
    if n_communities == 0 and n_entities < 6:
        # Pas assez d'entités pour avoir des communautés — on neutralise
        # cette dimension du score plutôt que de pénaliser à 0.
        weighted = (
            freshness * 0.25
            + coverage * 0.25
            + efficiency * 0.20
            + reachability * 0.30
        )
        # 100% du poids déjà couvert, pas besoin de renorm
    else:
        weighted = (
            freshness * 0.25
            + coverage * 0.25
            + coherence * 0.20
            + efficiency * 0.15
            + reachability * 0.15
        )

    health_score = int(round(weighted))

    dims = {
        "freshness": freshness,
        "coverage": coverage,
        "coherence": coherence,
        "efficiency": efficiency,
        "reachability": reachability,
    }
    hint = _hint_for_score(health_score, dims)

    snapshot = HealthSnapshot(
        freshness=freshness,
        coverage=coverage,
        coherence=coherence,
        efficiency=efficiency,
        reachability=reachability,
        health_score=health_score,
        n_entities=n_entities,
        n_relations=n_relations,
        n_communities=n_communities,
        n_pending=n_pending,
        n_chroma=n_chroma,
        measured_at=now,
        cognitive_hint=hint,
    )
    log.info(
        "Memory health: score=%d freshness=%d coverage=%d coherence=%d "
        "efficiency=%d reachability=%d (entities=%d relations=%d "
        "communities=%d chroma=%d)",
        health_score, freshness, coverage, coherence, efficiency,
        reachability, n_entities, n_relations, n_communities, n_chroma,
    )
    return snapshot
