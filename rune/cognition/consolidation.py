"""Consolidation phase — microsleep, deep sleep, persistence, git sync.

Biological inspiration
----------------------
Outside the active waking phases (encode/store/recall), the brain
periodically *consolidates*: hippocampal traces are replayed,
strengthened or pruned, and selectively transferred to neocortex.
In Lythéa this is split between two cycles:

1. **Microsleep** — short, opportunistic. Triggered every N
   exchanges or after an inactivity timeout. Runs the full
   :class:`MicrosleepManager` cycle (sharp-wave ripples + forward/
   reverse replay + selective compression to Chroma) and persists
   state to disk + git.

2. **Deep sleep** — long, manual. User-triggered (UI button or
   API call). Aggressive SDM prune, persist, flush working memory
   while preserving episodic memory (MHN). Designed for the end
   of a long session.

The :class:`MicrosleepManager` itself lives in
:mod:`rune.microsleep` — it is the cognitive-science engine.
:class:`ConsolidationPhase` here is the orchestration layer:
threading, locks, timers, persistence, and git sync. The split
mirrors how the original Hippocampe code was already structured;
this module just lifts those plumbing concerns out of the
orchestrator class.

Design notes
------------
- All threading is daemon-mode so an in-flight microsleep does
  not prevent process exit.
- Anti-stacking is enforced by both a flag (``_microsleep_pending``)
  and a non-blocking lock (``_microsleep_lock``). Belt + suspenders:
  the flag avoids spinning up a thread at all, the lock guards
  the rare race where two flags get set simultaneously.
- All persistence and git steps are inside the same try-block as
  the consolidation itself. A failure in git push must not
  prevent the microsleep from being marked completed (we still
  release the lock in ``finally``).
- ``last_microsleep_ts`` is exposed via property so the temporal
  block in Phase C can show "X minutes since last consolidation".
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from rune.config import (
    MHN_DIR,
    MICROSLEEP_BOOST,
    MICROSLEEP_INACTIVITY,
    MICROSLEEP_INTERVAL,
    MICROSLEEP_REHEARSE_K,
    SDM_DIR,
)

log = logging.getLogger("rune.cognition.consolidation")


# Aggressive SDM prune threshold for deep sleep. Anything below
# 50% normalised activity is dropped — we keep only the patterns
# that have accumulated meaningful evidence across the session.
_DEEP_SLEEP_PRUNE_THRESHOLD: float = 0.5

# Stable user-facing message returned by :meth:`deep_sleep`. Frontend
# may surface it verbatim in a toast notification.
_DEEP_SLEEP_DONE_MESSAGE: str = (
    "Consolidation profonde terminée. SDM purgée, MHN préservée."
)


class ConsolidationPhase:
    """Manage microsleep and deep-sleep cycles around the orchestrator.

    Parameters
    ----------
    sdm
        :class:`SparseDistributedMemory`. Saved + pruned + flushed
        across cycles.
    mhn
        :class:`ModernHopfieldNetwork`. Saved during microsleep and
        deep sleep but **never flushed** by deep sleep — episodic
        memory persists across sessions until the user resets.
    kg
        :class:`KnowledgeGraphStore`. Saved + has its pending queue
        cleaned up + promoted during microsleep.
    git
        :class:`GitSync`. ``push_async`` called at the end of each
        microsleep and after every deep sleep.
    microsleep_manager
        :class:`MicrosleepManager`. The cognitive-science engine —
        ripples, replay, compression. Owned by the orchestrator
        and injected here, since both this phase and the
        orchestrator's ``record_event`` path need to talk to it.
    """

    def __init__(
        self,
        sdm: Any,
        mhn: Any,
        kg: Any,
        git: Any,
        microsleep_manager: Any,
    ) -> None:
        self.sdm = sdm
        self.mhn = mhn
        self.kg = kg
        self.git = git
        self._microsleep_manager = microsleep_manager

        self._microsleep_lock = threading.Lock()
        self._microsleep_pending: bool = False
        self._inactivity_timer: threading.Timer | None = None
        self._last_microsleep_ts: float = time.time()

        # The orchestrator updates this through its bookkeeping;
        # we read it for log messages but never mutate it.
        self._exchange_count_ref: Any = None

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def last_microsleep_ts(self) -> float:
        """Wall-clock timestamp of the last completed microsleep."""
        return self._last_microsleep_ts

    def bind_exchange_counter(self, getter: Any) -> None:
        """Register a callable returning the current exchange count.

        The orchestrator owns the counter (it ticks it during
        :meth:`Hippocampe._post_generation`). Microsleep logs use
        the value purely for human-readable messages.
        """
        self._exchange_count_ref = getter

    def record_event(
        self,
        surprise_global: float,
        *,
        affect_intensity: float = 0.0,
        affect_arousal: float = 0.0,
        last_pattern_idx: int | None = None,
    ) -> None:
        """Forward a high-saliency event to the ripple tracker.

        Called once per exchange from the orchestrator's post-
        generation step. When enough high-surprise events
        accumulate, the next microsleep will run in *ripple mode*.

        V4.1 additive kwargs (defaults preserve V3.9.4 behaviour):
        ``affect_intensity`` / ``affect_arousal`` / ``last_pattern_idx``
        carry the cognitive_state's signal so the underlying tracker
        can flag the most recent MHN pattern for boosted replay
        when the configured arousal threshold is crossed.
        """
        try:
            self._microsleep_manager.record_event(
                float(surprise_global),
                affect_intensity=affect_intensity,
                affect_arousal=affect_arousal,
                last_pattern_idx=last_pattern_idx,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Ripple record failed: %s", exc)

    def maybe_trigger_after_exchange(self, exchange_count: int) -> None:
        """Trigger a microsleep if ``exchange_count`` hits the interval.

        Mirrors the original ``if exchange_count % MICROSLEEP_INTERVAL == 0``
        check in :meth:`Hippocampe._post_generation`.
        """
        if exchange_count > 0 and exchange_count % MICROSLEEP_INTERVAL == 0:
            self.trigger_microsleep()

    def trigger_microsleep(self) -> None:
        """Launch a microsleep in a background daemon thread.

        Anti-stacking: if a microsleep is already pending or
        running, this is a no-op.
        """
        if self._microsleep_pending:
            return
        self._microsleep_pending = True
        thread = threading.Thread(target=self._run_microsleep, daemon=True)
        thread.start()

    def reset_inactivity_timer(self) -> None:
        """(Re)arm the inactivity timer that triggers microsleep.

        Called after every exchange. If the user goes silent for
        :data:`MICROSLEEP_INACTIVITY` seconds, a microsleep fires
        autonomously — this is the "the system thinks while you
        are away" behaviour.
        """
        if self._inactivity_timer is not None:
            self._inactivity_timer.cancel()
        self._inactivity_timer = threading.Timer(
            MICROSLEEP_INACTIVITY, self.trigger_microsleep,
        )
        self._inactivity_timer.daemon = True
        self._inactivity_timer.start()

    def cancel_inactivity_timer(self) -> None:
        """Cancel any pending inactivity timer.

        Useful at session reset / shutdown so we don't leak
        timer threads.
        """
        if self._inactivity_timer is not None:
            self._inactivity_timer.cancel()
            self._inactivity_timer = None

    def deep_sleep(self) -> str:
        """Aggressive prune + persist. SDM flushed, MHN preserved.

        Returns the user-facing French confirmation string.
        """
        self.sdm.prune(threshold=_DEEP_SLEEP_PRUNE_THRESHOLD)
        self.sdm.save(SDM_DIR / "sdm_state.pt")
        self.mhn.save(MHN_DIR / "mhn_state.pt")
        self.kg.save()
        self.sdm.flush()
        # MHN intentionally NOT flushed — episodic memory persists
        # until the user starts a new session.
        self.git.push_async()
        log.info("Deep sleep completed")
        return _DEEP_SLEEP_DONE_MESSAGE

    # ── Internals — microsleep cycle body ──────────────────────────────

    def _run_microsleep(self) -> None:
        """The actual consolidation work, run in a worker thread.

        Threading contract: the caller has already set
        ``_microsleep_pending = True``; this method must always
        clear it (and release the lock) on exit, even on error.
        """
        if not self._microsleep_lock.acquire(blocking=False):
            self._microsleep_pending = False
            return
        try:
            count_for_log = self._read_exchange_count()
            log.info("🛏️ Microsleep started (exchange #%d)", count_for_log)

            stats = self._microsleep_manager.consolidate(
                rehearse_top_k=MICROSLEEP_REHEARSE_K,
                rehearse_boost=MICROSLEEP_BOOST,
            )
            self._log_consolidation_stats(stats)

            # V5.2 — GraphRAG : detect + summarise communities. Done here
            # (in the microsleep) and not at every retrieval because :
            #   • community detection is O(V+E) for Louvain — fine on
            #     ~100 entities, but adds latency to every chat turn
            #   • summarisation needs N LLM calls, so we amortise the
            #     cost by computing once per consolidation cycle
            # The communities live on self.kg.communities and are read
            # cheaply at retrieval time (see retrieval.py).
            try:
                self._refresh_kg_communities()
            except Exception as exc:
                log.warning("KG community refresh failed: %s", exc)

            # V5.4 — Procedural memory : extract reusable trigger→approach
            # patterns from recent exchanges. Like communities, this is
            # done once per microsleep rather than per turn — the LLM
            # extraction is too costly otherwise. Patterns are deduped
            # against existing ones to avoid the procedural store
            # bloating.
            try:
                self._refresh_procedural_memory()
            except Exception as exc:
                log.warning("Procedural memory refresh failed: %s", exc)

            # Persistence + git push. These run in the same try
            # block as the consolidate call so a failing save does
            # not leave a half-written state without us noticing.
            self._persist_all()
            self.git.push_async()
            self._last_microsleep_ts = time.time()

            sdm_active = int((self.sdm.contents.norm(dim=1) > 0).sum().item())
            log.info(
                "🛏️ Microsleep completed — SDM: %d active, "
                "MHN: %d patterns, KG: %d entities",
                sdm_active, self.mhn.n_stored, len(self.kg.entities),
            )
        except Exception as exc:
            log.warning("Microsleep error: %s", exc)
        finally:
            self._microsleep_pending = False
            self._microsleep_lock.release()

    def _refresh_kg_communities(self) -> None:
        """Detect KG communities and generate summaries.

        Called from the microsleep. Skipped if the KG is too small
        to have meaningful communities (< 6 entities with relations
        is unlikely to cluster usefully). Reuses existing summaries
        if entity membership hasn't changed, to avoid spending tokens
        on identical clusters.
        """
        if len(self.kg.entities) < 6 or not self.kg.relations:
            return
        try:
            from rune.cognition.graph_communities import (
                detect_communities, summarise_communities,
            )
        except ImportError:
            return

        # Detection
        new_communities = detect_communities(
            self.kg.entities,
            list(self.kg.relations.values()),
            min_community_size=2,
        )
        if not new_communities:
            return

        # Reuse summaries from the previous cycle when the community
        # membership is unchanged. We index by frozenset of entity_ids
        # because community IDs can shift across re-clustering runs.
        old_index: dict[frozenset, str] = {}
        for old in self.kg.communities:
            members = frozenset(old.entity_ids)
            if old.summary:
                old_index[members] = old.summary
        reused = 0
        for community in new_communities:
            members = frozenset(community.entity_ids)
            if members in old_index:
                community.summary = old_index[members]
                reused += 1
        if reused > 0:
            log.info("KG communities: reused %d/%d summaries", reused, len(new_communities))

        # Summarise only the unsummarised top-N
        llm = getattr(self, "_llm_for_community_summary", None)
        if llm is not None:
            summarise_communities(
                new_communities,
                self.kg.entities,
                llm,
                max_communities=10,
                timeout_per_call=5.0,
            )
        else:
            log.info("No LLM provided to consolidation for community summaries")

        self.kg.communities = new_communities

    def _refresh_procedural_memory(self) -> None:
        """V5.4 — Extract reusable patterns from recent exchanges.

        Called from microsleep. Uses the configured LLM (set via
        ``self._llm_for_community_summary`` already plumbed for V5.2,
        reused here for procedural extraction). Stores newly found
        patterns in ``self._procedural_store`` if available.

        Strategy :
        1. Pull the last 10 exchanges from the active session (if any).
        2. Ask the LLM to extract 0-3 reusable trigger→approach patterns.
        3. For each, attempt to add to the store (dedup via similarity).
        4. Archive stale procedures (90j unused, <3 uses).
        """
        llm = getattr(self, "_llm_for_community_summary", None)
        store = getattr(self, "_procedural_store", None)
        if llm is None or store is None:
            return

        # Need a session source for recent exchanges. Plugged via
        # bind_session_source by Hippocampe.
        get_recent = getattr(self, "_recent_exchanges_provider", None)
        if get_recent is None:
            return
        try:
            exchanges = get_recent(20)  # take more, the LLM will pick
        except Exception as exc:
            log.warning("Recent exchanges provider failed: %s", exc)
            return
        if not exchanges:
            return

        try:
            from rune.memory.procedural import extract_procedures_from_conversation
            extracted = extract_procedures_from_conversation(
                exchanges, llm, max_exchanges=10, timeout=8.0,
            )
        except Exception as exc:
            log.warning("Procedural extraction failed: %s", exc)
            return

        if not extracted:
            log.info("Procedural extraction: no new patterns this cycle")
        else:
            # Build a similarity checker using sentence-transformer
            # already loaded for the semantic router. Falls back to
            # exact-match dedup if router unavailable.
            sim_check = self._build_similarity_checker()
            added = 0
            for item in extracted:
                proc = store.add(
                    item["trigger"], item["approach"],
                    confidence=item.get("confidence", 0.6),
                    similarity_check=sim_check,
                )
                if proc is not None and proc.applied_count == 1:
                    # applied_count == 1 means freshly added (not dedup)
                    added += 1
            log.info(
                "Procedural extraction: %d new patterns, %d total active",
                added, len(store.active()),
            )

        # Forgetting curve
        archived = store.archive_stale()
        if archived:
            log.info("Procedural memory: archived %d stale procedures", archived)

        # Persist
        store.save()

    def _build_similarity_checker(self):
        """Return a callable (text_a, text_b) → float, using the
        semantic_router's embedding model if available. None → exact
        match only.
        """
        try:
            from rune.cognition.semantic_router import get_router
            router = get_router()
            if not router._warmed:
                router.warm_up()
            if router._model is None:
                return None
            model = router._model

            def cosine_check(a: str, b: str) -> float:
                try:
                    import numpy as np
                    embs = model.encode(
                        [a, b], normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    return float(np.dot(embs[0], embs[1]))
                except Exception:
                    return 0.0

            return cosine_check
        except Exception:
            return None

    def _persist_all(self) -> None:
        """Save SDM, MHN, and run KG cleanup + promotion + save."""
        self.sdm.save(SDM_DIR / "sdm_state.pt")
        self.mhn.save(MHN_DIR / "mhn_state.pt")
        self.kg.cleanup_pending()
        self.kg.promote_pending()
        self.kg.save()

    def _log_consolidation_stats(self, stats: dict) -> None:
        """Pretty-print microsleep stats — ripple-active vs replay-only."""
        if stats.get("ripple_active"):
            log.info(
                "  ⚡ ripple x%d events  •  replay: %d sequences, %d patterns  "
                "•  compressed: %d",
                stats.get("ripple_event_count", 0),
                stats.get("replay_sequences", 0),
                stats.get("replay_patterns", 0),
                stats.get("compressed_to_chroma", 0),
            )
        elif stats.get("replay_sequences", 0) > 0:
            log.info(
                "  replay: %d sequences  •  compressed: %d",
                stats.get("replay_sequences", 0),
                stats.get("compressed_to_chroma", 0),
            )

    def _read_exchange_count(self) -> int:
        """Read the current exchange count via the bound getter."""
        if self._exchange_count_ref is None:
            return 0
        try:
            return int(self._exchange_count_ref())
        except Exception:  # pragma: no cover — defensive
            return 0
