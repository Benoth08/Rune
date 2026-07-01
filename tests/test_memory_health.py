"""Tests V5.3 — Memory Health Dashboard.

Couvre :
- Calcul des 5 dimensions (freshness, coverage, coherence, efficiency, reachability)
- Score global pondéré
- Hint textuel selon le score
- Cas dégénérés (KG vide, pas de Chroma, etc.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest


# ── Mocks ──────────────────────────────────────────────────────────────


@dataclass
class FakeEnt:
    entity_id: str
    value: str = "x"
    type: str = "concept"


@dataclass
class FakeRel:
    subject_id: str
    object_id: str
    predicate: str = "related_to"


@dataclass
class FakeCommunity:
    community_id: str
    entity_ids: list[str]
    summary: str = ""
    size: int = 0
    density: float = 0.0


class FakeKG:
    def __init__(self, entities=None, relations=None, communities=None, pending=None):
        self.entities = entities or {}
        self.relations = relations or {}
        self.communities = communities or []
        self.pending = pending or {}


class FakeChroma:
    """Mock minimal de chromadb.Collection."""
    def __init__(self, count_value=0, metadatas=None):
        self._count = count_value
        self._metadatas = metadatas or []

    def count(self):
        return self._count

    def get(self, limit=None, include=None):
        metas = self._metadatas[:limit] if limit else self._metadatas
        return {"metadatas": metas}


# ── Tests dimensions individuelles ─────────────────────────────────────


class TestFreshness:

    def test_empty_chroma(self):
        from rune.memory.health import _compute_freshness
        score, n = _compute_freshness(FakeChroma(count_value=0), time.time())
        assert score == 0
        assert n == 0

    def test_all_recent(self):
        from rune.memory.health import _compute_freshness
        now = time.time()
        recent_ts = now - 1000  # quelques minutes
        chroma = FakeChroma(
            count_value=10,
            metadatas=[{"ts": recent_ts}] * 10,
        )
        score, n = _compute_freshness(chroma, now)
        assert score == 100
        assert n == 10

    def test_all_old(self):
        from rune.memory.health import _compute_freshness
        now = time.time()
        old_ts = now - (60 * 86400)  # 60 jours
        chroma = FakeChroma(
            count_value=10,
            metadatas=[{"ts": old_ts}] * 10,
        )
        score, _ = _compute_freshness(chroma, now)
        assert score == 0

    def test_mixed(self):
        from rune.memory.health import _compute_freshness
        now = time.time()
        recent = now - 1000
        old = now - (60 * 86400)
        chroma = FakeChroma(
            count_value=10,
            metadatas=[{"ts": recent}] * 7 + [{"ts": old}] * 3,
        )
        score, _ = _compute_freshness(chroma, now)
        assert score == 70

    def test_no_timestamps(self):
        """Si pas de ts, score neutre 50."""
        from rune.memory.health import _compute_freshness
        chroma = FakeChroma(count_value=5, metadatas=[{}] * 5)
        score, _ = _compute_freshness(chroma, time.time())
        assert score == 50


class TestCoverage:

    def test_empty(self):
        from rune.memory.health import _compute_coverage
        score, n_e, n_r = _compute_coverage(FakeKG())
        assert score == 0 and n_e == 0 and n_r == 0

    def test_full_coverage(self):
        from rune.memory.health import _compute_coverage
        kg = FakeKG(
            entities={"e1": FakeEnt("e1"), "e2": FakeEnt("e2")},
            relations={"r1": FakeRel("e1", "e2")},
        )
        score, n_e, n_r = _compute_coverage(kg)
        # 2 entités, toutes connectées → 100%
        assert score == 100
        assert n_e == 2 and n_r == 1

    def test_partial_coverage(self):
        from rune.memory.health import _compute_coverage
        kg = FakeKG(
            entities={
                "e1": FakeEnt("e1"),
                "e2": FakeEnt("e2"),
                "e3": FakeEnt("e3"),  # orphelin
                "e4": FakeEnt("e4"),  # orphelin
            },
            relations={"r1": FakeRel("e1", "e2")},
        )
        score, _, _ = _compute_coverage(kg)
        # 2 entités connectées / 4 = 50%
        assert score == 50


class TestCoherence:

    def test_no_communities(self):
        from rune.memory.health import _compute_coherence
        kg = FakeKG()
        score, n = _compute_coherence(kg)
        assert score == 0 and n == 0

    def test_high_density(self):
        from rune.memory.health import _compute_coherence
        kg = FakeKG(communities=[
            FakeCommunity("c0", ["e1", "e2", "e3"], density=0.6),
        ])
        score, n = _compute_coherence(kg)
        # density 0.6 → mappé à 100%
        assert score == 100
        assert n == 1

    def test_low_density(self):
        from rune.memory.health import _compute_coherence
        kg = FakeKG(communities=[
            FakeCommunity("c0", ["e1", "e2"], density=0.1),
        ])
        score, _ = _compute_coherence(kg)
        # density 0.1 → 0.1/0.6 ≈ 17
        assert 10 <= score <= 25


class TestEfficiency:

    def test_no_kg(self):
        from rune.memory.health import _compute_efficiency
        score, n_p = _compute_efficiency(None)
        assert score == 0

    def test_no_pending(self):
        from rune.memory.health import _compute_efficiency
        kg = FakeKG(entities={"e1": FakeEnt("e1")})
        score, n_p = _compute_efficiency(kg)
        assert score == 100
        assert n_p == 0

    def test_half_pending(self):
        from rune.memory.health import _compute_efficiency
        kg = FakeKG(
            entities={"e1": FakeEnt("e1"), "e2": FakeEnt("e2")},
            pending={"p1": FakeEnt("p1"), "p2": FakeEnt("p2")},
        )
        score, n_p = _compute_efficiency(kg)
        assert score == 50
        assert n_p == 2


class TestReachability:

    def test_too_few_entities(self):
        from rune.memory.health import _compute_reachability
        kg = FakeKG(entities={"e1": FakeEnt("e1")})
        assert _compute_reachability(kg) == 0

    def test_no_relations(self):
        from rune.memory.health import _compute_reachability
        kg = FakeKG(
            entities={"e1": FakeEnt("e1"), "e2": FakeEnt("e2")},
        )
        assert _compute_reachability(kg) == 0

    def test_fully_connected(self):
        """Tous les nœuds dans 1 composant → 100%."""
        from rune.memory.health import _compute_reachability
        kg = FakeKG(
            entities={f"e{i}": FakeEnt(f"e{i}") for i in range(4)},
            relations={
                "r1": FakeRel("e0", "e1"),
                "r2": FakeRel("e1", "e2"),
                "r3": FakeRel("e2", "e3"),
            },
        )
        assert _compute_reachability(kg) == 100

    def test_two_islands(self):
        """2 composants 2+2 → plus grand = 2/4 = 50%."""
        from rune.memory.health import _compute_reachability
        kg = FakeKG(
            entities={f"e{i}": FakeEnt(f"e{i}") for i in range(4)},
            relations={
                "r1": FakeRel("e0", "e1"),
                "r2": FakeRel("e2", "e3"),
            },
        )
        assert _compute_reachability(kg) == 50


# ── Test global compute_health ─────────────────────────────────────────


class TestComputeHealth:

    def test_empty_state(self):
        """KG vide + Chroma vide → tout à 0 mais pas de crash."""
        from rune.memory.health import compute_health
        kg = FakeKG()
        chroma = FakeChroma(count_value=0)
        snap = compute_health(kg, chroma)
        assert snap.health_score == 0
        assert snap.n_entities == 0
        assert snap.n_relations == 0
        assert "vierge" in snap.cognitive_hint.lower() or snap.health_score == 0

    def test_good_state(self):
        """Bon état général → score élevé."""
        from rune.memory.health import compute_health
        now = time.time()
        kg = FakeKG(
            entities={f"e{i}": FakeEnt(f"e{i}") for i in range(6)},
            relations={
                f"r{i}": FakeRel(f"e{i}", f"e{(i+1) % 6}")
                for i in range(6)  # cycle → tous connectés
            },
            communities=[
                FakeCommunity("c0", ["e0", "e1", "e2"], density=0.5),
            ],
        )
        chroma = FakeChroma(
            count_value=20,
            metadatas=[{"ts": now - 1000}] * 20,  # tous récents
        )
        snap = compute_health(kg, chroma)
        assert snap.health_score >= 70
        assert snap.freshness == 100
        assert snap.coverage == 100
        assert snap.reachability == 100

    def test_no_chroma(self):
        """Pas de Chroma → calcul OK mais freshness=0."""
        from rune.memory.health import compute_health
        kg = FakeKG(
            entities={"e1": FakeEnt("e1"), "e2": FakeEnt("e2")},
            relations={"r1": FakeRel("e1", "e2")},
        )
        snap = compute_health(kg, chroma_collection=None)
        assert snap.freshness == 0
        assert snap.coverage == 100

    def test_small_kg_no_community_penalty(self):
        """KG < 6 entités → on ne pénalise pas l'absence de communautés."""
        from rune.memory.health import compute_health
        kg = FakeKG(
            entities={"e1": FakeEnt("e1"), "e2": FakeEnt("e2")},
            relations={"r1": FakeRel("e1", "e2")},
        )
        chroma = FakeChroma(count_value=5, metadatas=[{"ts": time.time()}] * 5)
        snap = compute_health(kg, chroma)
        # coherence à 0 ne tire pas le score vers le bas (poids redistribué)
        assert snap.health_score > 50


class TestHintForScore:

    def test_high_score(self):
        from rune.memory.health import _hint_for_score
        hint = _hint_for_score(85, {
            "freshness": 90, "coverage": 80, "coherence": 80,
            "efficiency": 90, "reachability": 80,
        })
        assert "bonne" in hint.lower() or "saine" in hint.lower()

    def test_low_score(self):
        from rune.memory.health import _hint_for_score
        hint = _hint_for_score(25, {
            "freshness": 30, "coverage": 20, "coherence": 20,
            "efficiency": 30, "reachability": 20,
        })
        assert "jeune" in hint.lower() or "encore" in hint.lower()

    def test_identifies_weakest(self):
        from rune.memory.health import _hint_for_score
        # coverage faible
        hint = _hint_for_score(55, {
            "freshness": 90, "coverage": 10, "coherence": 70,
            "efficiency": 80, "reachability": 60,
        })
        assert "couverture" in hint.lower()
