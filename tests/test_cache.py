"""Tests for the BoundedCache."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rune.cache import BoundedCache


def test_get_returns_none_on_miss():
    cache = BoundedCache(max_size=10)
    assert cache.get("missing") is None


def test_put_then_get():
    cache = BoundedCache(max_size=10)
    cache.put("hello", "world")
    assert cache.get("hello") == "world"


def test_lru_eviction_when_full():
    cache = BoundedCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.put("d", 4)  # evicts "a" (least recently used)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_get_promotes_to_mru():
    """Reading an entry moves it to most-recently-used position."""
    cache = BoundedCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    # Read "a" → it's now MRU. Adding "d" should evict "b" (now LRU).
    _ = cache.get("a")
    cache.put("d", 4)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_put_existing_refreshes_position():
    """Re-putting an existing key promotes it to MRU."""
    cache = BoundedCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.put("a", 99)  # update existing → "a" now MRU
    cache.put("d", 4)   # evicts "b" (now LRU, not "a")

    assert cache.get("a") == 99
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_disabled_cache_is_no_op():
    """max_size=0 means caching is disabled."""
    cache = BoundedCache(max_size=0)
    cache.put("k", "v")
    assert cache.get("k") is None
    assert len(cache) == 0


def test_negative_max_size_disabled():
    cache = BoundedCache(max_size=-1)
    cache.put("k", "v")
    assert cache.get("k") is None


def test_clear_resets_state():
    cache = BoundedCache(max_size=5)
    cache.put("a", 1)
    cache.put("b", 2)
    _ = cache.get("a")
    cache.clear()
    # Stats must be zero immediately after clear, before any further access.
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["evictions"] == 0
    assert stats["size"] == 0
    # And the data is gone too
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_stats_track_hits_and_misses():
    cache = BoundedCache(max_size=5)
    cache.put("a", 1)
    _ = cache.get("a")  # hit
    _ = cache.get("a")  # hit
    _ = cache.get("missing")  # miss

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)


def test_stats_evictions_counted():
    cache = BoundedCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # evicts a
    cache.put("d", 4)  # evicts b
    stats = cache.stats()
    assert stats["evictions"] == 2


def test_get_or_compute_caches_result():
    cache = BoundedCache(max_size=10)
    call_count = [0]

    def compute():
        call_count[0] += 1
        return "computed"

    out1 = cache.get_or_compute("k", compute)
    out2 = cache.get_or_compute("k", compute)
    out3 = cache.get_or_compute("k", compute)

    assert out1 == out2 == out3 == "computed"
    assert call_count[0] == 1  # only first call runs


def test_get_or_compute_does_not_cache_none():
    """A None return from compute means 'don't cache' (transient failure)."""
    cache = BoundedCache(max_size=10)
    call_count = [0]

    def compute():
        call_count[0] += 1
        return None

    cache.get_or_compute("k", compute)
    cache.get_or_compute("k", compute)
    assert call_count[0] == 2  # both calls executed


def test_key_is_deterministic():
    """Same input string → same key. Different inputs → different keys."""
    assert BoundedCache._key("hello") == BoundedCache._key("hello")
    assert BoundedCache._key("hello") != BoundedCache._key("world")


def test_key_handles_unicode():
    """French accents should hash without errors."""
    k1 = BoundedCache._key("Taëlys")
    k2 = BoundedCache._key("Taelys")
    assert k1 != k2  # they're different strings
    assert isinstance(k1, str)


def test_thread_safety_no_corruption():
    """100 threads racing on get/put must not corrupt internal state."""
    cache = BoundedCache(max_size=50)
    errors: list[Exception] = []

    def worker(i):
        try:
            for j in range(20):
                key = f"k_{(i + j) % 60}"  # creates evictions
                cache.put(key, i * 1000 + j)
                _ = cache.get(key)
        except Exception as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(worker, i) for i in range(100)]
        for f in futures:
            f.result()

    assert errors == []
    # Cache size must respect the bound under all interleavings.
    assert len(cache) <= 50


def test_thread_safety_get_or_compute_serialised():
    """Concurrent get_or_compute on the same key may compute multiple times.

    BoundedCache does NOT guarantee 'compute once' semantics — that
    would require per-key locks. We just check no crash and that the
    final cached value is one of the computed ones.
    """
    cache = BoundedCache(max_size=10)
    counter = [0]
    lock = threading.Lock()

    def compute():
        with lock:
            counter[0] += 1
            return f"v_{counter[0]}"

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = [
            pool.submit(cache.get_or_compute, "race", compute)
            for _ in range(10)
        ]
        values = [r.result() for r in results]

    # All values are valid computed strings
    assert all(v.startswith("v_") for v in values)


def test_len_reflects_real_size():
    cache = BoundedCache(max_size=5)
    assert len(cache) == 0
    cache.put("a", 1)
    cache.put("b", 2)
    assert len(cache) == 2
