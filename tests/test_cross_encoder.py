"""Tests for the cross-encoder configuration in HybridRetriever.

We don't load real cross-encoders here (would require torch + a network
fetch). Instead we mock the CrossEncoder class and verify that:
- the user-configured model is tried first
- fallbacks happen on failure
- the min_score threshold from settings is applied
- the rerank produces a different ordering than the dense-only fallback
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Tests that patch sentence_transformers.CrossEncoder need the module
# to exist so patch() can find an attribute to replace. In the sandbox
# CI it isn't installed; on the dev machine it is. We skip those tests
# here when the module is absent.
_HAS_ST = True
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _HAS_ST = False

from rune.memory.retrieval import HybridRetriever


class _FakeCollection:
    """Minimal Chroma-like collection for retrieval tests."""

    def __init__(self, docs: list[tuple[str, str]]):
        # docs: list of (id, document)
        self._docs = docs

    def count(self) -> int:
        return len(self._docs)

    def get(self, include=None) -> dict:
        return {
            "ids": [d[0] for d in self._docs],
            "documents": [d[1] for d in self._docs],
        }

    def query(self, query_texts, n_results) -> dict:
        # Simulate a "dense" retriever that returns docs in storage order.
        # Our tests will then check that the cross-encoder reorders them.
        ids = [d[0] for d in self._docs[:n_results]]
        documents = [d[1] for d in self._docs[:n_results]]
        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [[{} for _ in ids]],
        }


# ── Fallback chain ────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_ST, reason="sentence_transformers not installed")
def test_user_model_tried_first(monkeypatch):
    """The model name from LYTHEA_CROSS_ENCODER_MODEL is the first tried."""
    attempts: list[str] = []

    class FakeCE:
        def __init__(self, name):
            attempts.append(name)
            if name != "user/preferred":
                raise RuntimeError("not the right one")

    # Settings override
    monkeypatch.setenv("LYTHEA_CROSS_ENCODER_MODEL", "user/preferred")
    from rune.settings import get_settings
    get_settings.cache_clear()

    coll = _FakeCollection([])
    retriever = HybridRetriever(coll, embedder=None, use_cross_encoder=True)

    with patch("sentence_transformers.CrossEncoder", FakeCE):
        ce = retriever._get_cross_encoder()

    assert attempts[0] == "user/preferred"
    assert ce is not None


@pytest.mark.skipif(not _HAS_ST, reason="sentence_transformers not installed")
def test_fallback_chain_on_user_failure(monkeypatch):
    """If the preferred model fails, we try the well-known fallbacks."""
    attempts: list[str] = []

    class FakeCE:
        def __init__(self, name):
            attempts.append(name)
            if name == "BAAI/bge-reranker-base":
                return  # success
            raise RuntimeError("nope")

    monkeypatch.setenv("LYTHEA_CROSS_ENCODER_MODEL", "broken/model")
    from rune.settings import get_settings
    get_settings.cache_clear()

    coll = _FakeCollection([])
    retriever = HybridRetriever(coll, embedder=None, use_cross_encoder=True)

    with patch("sentence_transformers.CrossEncoder", FakeCE):
        ce = retriever._get_cross_encoder()

    assert "broken/model" in attempts
    assert "BAAI/bge-reranker-base" in attempts
    assert ce is not None


@pytest.mark.skipif(not _HAS_ST, reason="sentence_transformers not installed")
def test_no_duplicate_attempts_when_user_model_is_default(monkeypatch):
    """If the user picks the same as the first default, we only try once."""
    attempts: list[str] = []

    class FakeCE:
        def __init__(self, name):
            attempts.append(name)
            return

    monkeypatch.setenv("LYTHEA_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
    from rune.settings import get_settings
    get_settings.cache_clear()

    coll = _FakeCollection([])
    retriever = HybridRetriever(coll, embedder=None, use_cross_encoder=True)

    with patch("sentence_transformers.CrossEncoder", FakeCE):
        retriever._get_cross_encoder()

    # First success means we stopped — only one attempt
    assert attempts == ["BAAI/bge-reranker-v2-m3"]


@pytest.mark.skipif(not _HAS_ST, reason="sentence_transformers not installed")
def test_use_cross_encoder_false_disables_loading(monkeypatch):
    coll = _FakeCollection([])
    retriever = HybridRetriever(coll, embedder=None, use_cross_encoder=False)

    with patch("sentence_transformers.CrossEncoder") as fake:
        ce = retriever._get_cross_encoder()
    assert ce is None
    fake.assert_not_called()


# ── Reranker actually changes ordering ────────────────────────────────

def test_rerank_reorders_candidates_by_score(monkeypatch):
    """Reranker scores override the dense order.

    This test pins the threshold to 0.5 explicitly so it remains
    independent of the default value (which was lowered from 0.5 to
    0.2 during the post-refactor calibration). The point of the test
    is the *reordering* logic, with a side check that values below
    the threshold are dropped — both behaviours are verified at any
    threshold value, but pinning makes the assertions stable.
    """
    monkeypatch.delenv("LYTHEA_CROSS_ENCODER_MODEL", raising=False)
    monkeypatch.setenv("LYTHEA_CROSS_ENCODER_MIN_SCORE", "0.5")
    from rune.settings import get_settings
    get_settings.cache_clear()

    # Storage order: [doc_a, doc_b, doc_c] — dense returns them in that order.
    # Reranker scores will be: doc_c=0.9, doc_a=0.6, doc_b=0.3.
    docs = [
        ("id_a", "Le chat dort sur le tapis"),
        ("id_b", "La pluie tombe sur Paris"),
        ("id_c", "Une banane jaune et mûre"),
    ]
    coll = _FakeCollection(docs)

    fake_ce = MagicMock()
    # predict returns scores aligned with the input pairs ORDER the
    # retriever passes (i.e. the order of `candidates` after RRF).
    # We'll inspect the call to confirm and return our designed scores.
    fake_ce.predict = MagicMock(side_effect=lambda pairs: [
        # Pairs come in candidate order; map by document content
        {"Le chat dort sur le tapis": 0.6,
         "La pluie tombe sur Paris": 0.3,
         "Une banane jaune et mûre": 0.9}[doc]
        for _query, doc in pairs
    ])

    retriever = HybridRetriever(coll, embedder=None, use_cross_encoder=True)
    retriever._cross_encoder = fake_ce
    retriever._cross_encoder_loaded = True

    results = retriever.search("requête de test", n=3, rerank=True)

    # After rerank: doc_c is first (score 0.9), doc_a second (0.6).
    # doc_b (score 0.3) is below the pinned min_score of 0.5 → dropped.
    ids = [r["id"] for r in results]
    assert ids[0] == "id_c"
    assert ids[1] == "id_a"
    assert "id_b" not in ids


def test_rerank_min_score_threshold(monkeypatch):
    """LYTHEA_CROSS_ENCODER_MIN_SCORE filters low-scoring candidates."""
    monkeypatch.setenv("LYTHEA_CROSS_ENCODER_MIN_SCORE", "0.8")
    from rune.settings import get_settings
    get_settings.cache_clear()

    docs = [("a", "alpha"), ("b", "bravo"), ("c", "charlie")]
    coll = _FakeCollection(docs)

    fake_ce = MagicMock()
    fake_ce.predict = MagicMock(return_value=[0.95, 0.75, 0.5])

    retriever = HybridRetriever(coll, embedder=None, use_cross_encoder=True)
    retriever._cross_encoder = fake_ce
    retriever._cross_encoder_loaded = True

    results = retriever.search("q", n=3, rerank=True)
    # With threshold 0.8, only the 0.95 candidate survives
    assert len(results) == 1
    assert results[0]["rerank_score"] == pytest.approx(0.95)


def test_rerank_zero_threshold_keeps_all(monkeypatch):
    monkeypatch.setenv("LYTHEA_CROSS_ENCODER_MIN_SCORE", "0.0")
    from rune.settings import get_settings
    get_settings.cache_clear()

    docs = [("a", "alpha"), ("b", "bravo"), ("c", "charlie")]
    coll = _FakeCollection(docs)

    fake_ce = MagicMock()
    fake_ce.predict = MagicMock(return_value=[0.1, 0.2, 0.05])

    retriever = HybridRetriever(coll, embedder=None, use_cross_encoder=True)
    retriever._cross_encoder = fake_ce
    retriever._cross_encoder_loaded = True

    results = retriever.search("q", n=3, rerank=True)
    assert len(results) == 3


# ── Reset env between tests ───────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_env():
    """Reset settings cache after each test to avoid bleed."""
    yield
    from rune.settings import get_settings
    get_settings.cache_clear()
