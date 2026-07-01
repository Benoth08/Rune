"""Bounded LRU cache for tensor-valued functions.

Why not :func:`functools.lru_cache`?
------------------------------------
- ``lru_cache`` keys on the function arguments. Our values are
  :class:`torch.Tensor` objects of various shapes; the function
  argument is a string but we want to control hashing explicitly so
  cache keys stay short and deterministic.
- ``lru_cache`` is not thread-safe in a meaningful way (it serialises
  but doesn't expose stats or eviction reasons). We want both.
- We sometimes want to ``stats()`` a cache from an admin endpoint
  to tune ``max_size`` empirically.

Design
------
- :class:`BoundedCache` keeps an :class:`OrderedDict` of (key → value)
  pairs. Reads move the entry to the end (most-recently-used). Inserts
  evict the oldest pair when the size exceeds ``max_size``.
- Keys are 16-byte BLAKE2b hashes of the input string. Collisions are
  cryptographically negligible at this size — we use them as
  fixed-length identifiers, not as hash-table buckets.
- Values are stored on CPU. Callers that need them on a different
  device should ``.to(device)`` after retrieval.

Memory footprint
----------------
A 768-dim float32 tensor is ~3 KB. With ``max_size=1000``, the
embedding cache uses ~3 MB. The analyze_input cache stores hidden
states which can be ~100 KB per entry; at ``max_size=50`` that's
~5 MB. Both are negligible compared to the LLM itself.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Callable, Generic, TypeVar

log = logging.getLogger("lythea.cache")

V = TypeVar("V")


class BoundedCache(Generic[V]):
    """Thread-safe LRU cache keyed by hashed strings.

    Parameters
    ----------
    max_size : int
        Maximum number of entries. When exceeded, the least-recently
        used entry is evicted. ``max_size <= 0`` disables caching.
    name : str
        Identifier used in log messages. Defaults to ``"cache"``.

    Notes
    -----
    The cache stores arbitrary values (typed via ``Generic[V]``).
    The value type is the caller's responsibility — for tensors,
    keep them on CPU and clone with ``.detach()`` before caching to
    avoid retaining gradients.
    """

    def __init__(self, max_size: int = 1024, name: str = "cache") -> None:
        self.max_size = max_size
        self.name = name
        self._store: OrderedDict[str, V] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def _key(text: str) -> str:
        """Return a stable 16-byte hex digest for a string.

        We use BLAKE2b with digest_size=16 (128 bits). Faster than
        SHA-256, more than enough for non-cryptographic use, and
        gives short keys that print nicely in debug logs.
        """
        return hashlib.blake2b(
            text.encode("utf-8", errors="replace"),
            digest_size=16,
        ).hexdigest()

    # ── Public API ─────────────────────────────────────────────────────

    def get(self, text: str) -> V | None:
        """Return the cached value, or None on miss."""
        if self.max_size <= 0:
            return None
        key = self._key(text)
        with self._lock:
            value = self._store.get(key)
            if value is None:
                self._misses += 1
                return None
            # Move to end → most-recently-used
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, text: str, value: V) -> None:
        """Insert or refresh a cache entry."""
        if self.max_size <= 0:
            return
        key = self._key(text)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                return
            self._store[key] = value
            # Evict from the front (least-recently-used) until within bounds.
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
                self._evictions += 1

    def get_or_compute(self, text: str, compute: Callable[[], V | None]) -> V | None:
        """Return cached value, or compute, store, and return.

        ``compute`` returning None is treated as "skip cache" — we
        don't store None placeholders, so the next call will re-compute.
        This is the right policy for embedding/analysis pipelines that
        can transiently fail.
        """
        cached = self.get(text)
        if cached is not None:
            return cached
        value = compute()
        if value is not None:
            self.put(text, value)
        return value

    def clear(self) -> None:
        """Drop all entries and reset stats."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict:
        """Return cache statistics for monitoring."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "name": self.name,
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(hit_rate, 3),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
