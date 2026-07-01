"""GraphRAG communities — V5.2 thematic clustering over the KG.

Adds Microsoft GraphRAG-style community detection on top of the
existing knowledge graph. The idea : entities + relations form a
graph where dense sub-clusters represent coherent topics (work,
family, hobbies, projects…). Detecting these communities and
generating LLM summaries unlocks two retrieval modes :

1. **Macro queries** — "Quels sont les grands thèmes qu'on a abordés ?"
   becomes answerable from community summaries without scanning the
   entire entity list.
2. **Subgraph retrieval** — for a question about entity X, we can
   inject not just X and its direct neighbors, but X's whole
   community summary, giving the LLM thematic context for free.

This is a **simplified Microsoft GraphRAG**. We skip the hierarchical
clustering (multi-level Leiden) and stick to a single-pass partition.
Community summaries are computed once per consolidation cycle (not
per query) so the cost is amortised. Leiden is preferred when available,
with Louvain and connected components as graceful fallbacks.

Reference : "From Local to Global: A Graph RAG Approach to
Query-Focused Summarization" (Edge et al., Microsoft Research, 2024).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("lythea.cognition.graph_communities")


# ── Community dataclass ───────────────────────────────────────────────


@dataclass
class Community:
    """A detected thematic cluster of KG entities.

    Attributes
    ----------
    community_id : str
        Stable identifier (incremented from 0 within a single
        ``detect_communities`` call). Re-clustering generates new
        IDs ; consumers should not persist these long-term across
        cycles unless paired with a versioning scheme.
    entity_ids : list[str]
        Members of the community (KG entity IDs).
    summary : str
        LLM-generated 1-3 sentence description of what binds the
        cluster together. Filled in by :func:`summarise_communities`,
        empty at detection time.
    size : int
        Convenience : len(entity_ids).
    density : float
        Number of internal edges / max possible edges. 1.0 = clique,
        0.0 = isolated nodes. Useful for ranking communities.
    """

    community_id: str
    entity_ids: list[str]
    summary: str = ""
    size: int = 0
    density: float = 0.0
    detected_at: float = field(default_factory=time.time)


# ── LLM interface ─────────────────────────────────────────────────────


class LLMCompleter(Protocol):
    def complete_sync(
        self,
        messages: list[dict],
        max_new_tokens: int = 80,
        timeout: float | None = None,
    ) -> str: ...


# ── Detection ─────────────────────────────────────────────────────────


def _try_louvain(graph) -> dict[Any, int] | None:
    """Try the python-louvain backend ; return None on failure."""
    try:
        import community as community_louvain  # python-louvain
        return community_louvain.best_partition(graph)
    except ImportError:
        return None
    except Exception as exc:
        log.warning("Louvain failed: %s", exc)
        return None


def _try_leiden(graph) -> dict[Any, int] | None:
    """Try the leidenalg + igraph backend ; return None on failure.

    Leiden gives slightly better partitions than Louvain by enforcing
    that all communities are well-connected. But it requires igraph,
    which is a 30+ MB native dep. Only used if already installed.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return None
    try:
        # Convert networkx → igraph
        nodes = list(graph.nodes())
        edges = list(graph.edges())
        g = ig.Graph()
        g.add_vertices(nodes)
        g.add_edges(edges)
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        # Format like python-louvain : dict node → cluster_id
        result: dict[Any, int] = {}
        for cluster_id, members in enumerate(partition):
            for v_idx in members:
                result[nodes[v_idx]] = cluster_id
        return result
    except Exception as exc:
        log.warning("Leiden failed: %s", exc)
        return None


def _try_connected_components(graph) -> dict[Any, int]:
    """Last-resort fallback : pure networkx connected components.

    Gives strict partitions (no fuzzy boundaries) but is always
    available. Useful when louvain/leiden aren't installed — at least
    we get topological islands.
    """
    import networkx as nx
    partition: dict[Any, int] = {}
    for cluster_id, component in enumerate(nx.connected_components(graph)):
        for node in component:
            partition[node] = cluster_id
    return partition


def detect_communities(
    entities: dict[str, Any],
    relations: list[Any],
    min_community_size: int = 2,
    prefer: str = "auto",
) -> list[Community]:
    """Detect thematic communities in the KG.

    Parameters
    ----------
    entities : dict[str, KGEntity-like]
        ``{entity_id: KGEntity}``. We use the keys as graph nodes.
    relations : list[KGRelation-like]
        Edges. We use ``(subject_id, object_id)``.
    min_community_size : int
        Communities smaller than this are filtered out as noise.
        2 is a sensible minimum (1 = isolated node, not a "community").
    prefer : str
        Backend preference : "leiden", "louvain", "auto" (try leiden
        then louvain then components), or "components" (skip clustering
        backends entirely).

    Returns
    -------
    list[Community]
        Sorted by size (desc). Each community has summary="" — call
        :func:`summarise_communities` to fill those.
    """
    if not entities:
        return []
    if not relations:
        # No edges → every entity is its own component, no meaningful
        # communities. Return empty rather than 1-node-per-community noise.
        log.info("Community detection: 0 relations, skipping")
        return []

    try:
        import networkx as nx
    except ImportError:
        log.error("networkx not available — cannot detect communities")
        return []

    # Build the graph. Use the entity IDs as nodes (already strings,
    # stable across sessions). Skip self-loops and missing endpoints.
    graph = nx.Graph()
    graph.add_nodes_from(entities.keys())
    edge_count = 0
    for rel in relations:
        subj = getattr(rel, "subject_id", None)
        obj = getattr(rel, "object_id", None)
        if not subj or not obj or subj == obj:
            continue
        if subj not in entities or obj not in entities:
            continue
        graph.add_edge(subj, obj)
        edge_count += 1

    if edge_count == 0:
        log.info("Community detection: graph has no valid edges")
        return []

    # Select backend
    partition: dict[Any, int] | None = None
    backend_used = "components"
    if prefer in ("auto", "leiden"):
        partition = _try_leiden(graph)
        if partition is not None:
            backend_used = "leiden"
    if partition is None and prefer in ("auto", "louvain"):
        partition = _try_louvain(graph)
        if partition is not None:
            backend_used = "louvain"
    if partition is None:
        partition = _try_connected_components(graph)
        backend_used = "components"

    # Group nodes by cluster id
    clusters: dict[int, list[str]] = {}
    for node, cluster_id in partition.items():
        clusters.setdefault(cluster_id, []).append(node)

    # Build Community objects, filter small ones, compute density.
    communities: list[Community] = []
    for cluster_id, members in clusters.items():
        if len(members) < min_community_size:
            continue
        # Density : edges within community / possible pairs
        member_set = set(members)
        internal_edges = sum(
            1 for u, v in graph.edges()
            if u in member_set and v in member_set
        )
        n = len(members)
        max_edges = n * (n - 1) / 2
        density = internal_edges / max_edges if max_edges > 0 else 0.0
        communities.append(Community(
            community_id=f"c{cluster_id}",
            entity_ids=members,
            size=n,
            density=density,
        ))

    communities.sort(key=lambda c: c.size, reverse=True)
    log.info(
        "Community detection (%s): %d entities, %d edges → %d communities "
        "(sizes: %s)",
        backend_used,
        len(entities), edge_count, len(communities),
        [c.size for c in communities[:5]],
    )
    return communities


# ── Summarisation ─────────────────────────────────────────────────────


_SUMMARY_SYSTEM_PROMPT = (
    "Tu résumes en 1-2 phrases courtes ce qui relie une liste d'entités "
    "extraites du graphe de connaissance de l'utilisateur. Donne le "
    "THÈME commun (ex : « travail à l'ESPC », « projet Lythéa », "
    "« famille »). Pas de préambule, juste le résumé."
)


def _summarise_one(
    community: Community,
    entity_lookup: dict[str, Any],
    llm: LLMCompleter,
    timeout: float = 5.0,
) -> str:
    """Generate one community summary via the LLM."""
    # Build the entity list as readable bullet points. Use value + type
    # so the LLM has both the label and the category.
    lines: list[str] = []
    for eid in community.entity_ids[:20]:  # cap to keep prompt short
        ent = entity_lookup.get(eid)
        if ent is None:
            continue
        val = getattr(ent, "value", "")
        typ = getattr(ent, "type", "")
        if val:
            lines.append(f"- {val} ({typ})" if typ else f"- {val}")
    if not lines:
        return ""
    if len(community.entity_ids) > 20:
        lines.append(f"- … et {len(community.entity_ids) - 20} autres")

    user_msg = (
        f"Entités du cluster #{community.community_id} "
        f"(taille {community.size}, densité {community.density:.2f}) :\n"
        + "\n".join(lines)
    )

    try:
        raw = llm.complete_sync(
            [
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_new_tokens=80,
            temperature=0.4,
            timeout=timeout,
        )
        return (raw or "").strip().splitlines()[0].strip()[:200]
    except Exception as exc:
        log.warning("Community %s summary failed: %s", community.community_id, exc)
        return ""


def summarise_communities(
    communities: list[Community],
    entity_lookup: dict[str, Any],
    llm: LLMCompleter,
    max_communities: int = 10,
    timeout_per_call: float = 5.0,
) -> list[Community]:
    """Generate LLM summaries for the top N communities.

    Mutates each community in-place to set its ``summary`` field,
    and returns the list for chaining. Summarises top-N by size to
    bound LLM cost — tiny clusters often aren't worth a dedicated
    summary anyway.
    """
    for community in communities[:max_communities]:
        if community.summary:
            continue  # already summarised (cache from previous cycle)
        community.summary = _summarise_one(
            community, entity_lookup, llm, timeout=timeout_per_call,
        )
    return communities


# ── Retrieval helpers ─────────────────────────────────────────────────


def find_community_for_entity(
    entity_id: str, communities: list[Community],
) -> Community | None:
    """Return the community containing the given entity, or None."""
    for community in communities:
        if entity_id in community.entity_ids:
            return community
    return None


def render_community_context(
    communities: list[Community],
    entity_lookup: dict[str, Any],
    *,
    focus_entity_ids: set[str] | None = None,
    max_communities: int = 3,
    max_chars: int = 1200,
) -> str:
    """Render a textual block listing relevant communities for prompt injection.

    Parameters
    ----------
    focus_entity_ids : set[str] | None
        If provided, prioritises communities containing these entities.
        Falls back to the top-N largest communities otherwise.
    """
    if not communities:
        return ""

    selected: list[Community]
    if focus_entity_ids:
        # Score communities by overlap with focus
        scored: list[tuple[int, Community]] = []
        for community in communities:
            overlap = sum(
                1 for eid in community.entity_ids if eid in focus_entity_ids
            )
            if overlap > 0:
                scored.append((overlap, community))
        scored.sort(key=lambda x: (x[0], x[1].size), reverse=True)
        selected = [c for _, c in scored[:max_communities]]
        if not selected:
            # No overlap → fall back to largest
            selected = communities[:max_communities]
    else:
        selected = communities[:max_communities]

    lines: list[str] = ["[Thématiques de mémoire]"]
    for community in selected:
        if community.summary:
            lines.append(
                f"• {community.summary} "
                f"(thématique #{community.community_id}, {community.size} éléments)"
            )
        else:
            # Pas de résumé encore : liste les 5 premières entités
            preview = []
            for eid in community.entity_ids[:5]:
                ent = entity_lookup.get(eid)
                if ent and getattr(ent, "value", None):
                    preview.append(ent.value)
            if preview:
                lines.append(
                    f"• Cluster #{community.community_id} : "
                    + ", ".join(preview)
                    + (f" et {community.size - len(preview)} autres"
                       if community.size > len(preview) else "")
                )

    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n…"
    return block
