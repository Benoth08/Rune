"""Microsleep consolidation cycle with sharp-wave ripples and replay.

Biological inspiration
----------------------
In hippocampal sleep, sharp-wave ripples (SWRs) are bursts of high-
frequency oscillations that drive the *replay* of recent waking
experiences. Place cells fire in the same order they did during the
preceding navigation episode, but compressed in time (5-20× faster).
Replay also occurs in reverse — the path is "rewound" — which is
believed to support credit assignment and reverse value propagation.

This module implements three enhancements over the baseline Lythéa
microsleep (rehearse → decay → prune):

1. **Sharp-wave ripple detection.** Each high-saliency event during
   waking increments a counter. When the counter exceeds a threshold,
   a ripple is "triggered" at the next microsleep, raising the rehearsal
   intensity for the recent salient memories.

2. **Forward and reverse replay.** Rather than replaying a single
   memory, we replay short *sequences* of consecutive high-attention
   patterns — the most recent K stored MHN patterns. Each sequence
   is replayed in both directions (forward = original temporal order;
   reverse = reversed). The bind operation reinforces the temporal
   association between adjacent memories.

3. **Temporal compression of MHN → Chroma transfer.** Patterns that
   have been replayed many times AND have high attention get
   transferred to long-term Chroma storage with a "consolidated" tag.
   This implements the hippocampus → neocortex memory transfer.

Design notes
------------
- All operations are best-effort: if any step fails (e.g. Chroma down,
  MHN dim mismatch), we log and continue. Microsleep must never raise
  to the caller.
- Replay does NOT rewrite memories — it just reinforces existing
  ones via attention boosts in the MHN. The original timestamps and
  content are preserved.
- Compression to Chroma is gated on multiple criteria (replay count,
  attention, age) to avoid duplicating noisy memories.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

# torch is needed for the actual MHN attention math, but we want
# microsleep.py to be importable without torch installed (for unit
# tests of pure logic like RippleTracker). Real usage paths always
# have torch via Hippocampe's dependencies.
try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

log = logging.getLogger("rune.microsleep")


# ── Configurable thresholds (overridable via settings) ────────────────

@dataclass
class MicrosleepConfig:
    """Parameters for the enhanced microsleep cycle.

    Defaults are tuned for a session of ~50-200 exchanges. If you run
    much longer sessions, raise ``ripple_trigger_count`` to avoid
    triggering ripples on every microsleep.
    """

    # Sharp-wave ripple — number of high-saliency events that build up
    # before a ripple-class microsleep is triggered.
    ripple_trigger_count: int = 5
    # Surprise threshold above which an event counts toward the ripple.
    ripple_surprise_threshold: float = 0.5
    # Boost factor applied during ripple-class rehearsal (vs normal boost).
    ripple_boost_multiplier: float = 2.0

    # Replay — number of recent MHN patterns to chain into a sequence.
    replay_sequence_length: int = 4
    # Number of independent sequences to replay per microsleep.
    replay_n_sequences: int = 3
    # Replay attention boost (additive, applied per pattern in sequence).
    replay_attention_boost: float = 0.05

    # Compression to Chroma — minimum replay count before transfer.
    compression_replay_threshold: int = 3
    # Minimum attention level for a pattern to be eligible.
    compression_attention_threshold: float = 0.4
    # Maximum patterns transferred per microsleep (avoid swamping Chroma).
    compression_max_per_cycle: int = 5

    # ── V4.1: Affect-modulated consolidation (additive, opt-in) ─────
    # When True, the cognitive_state's arousal signal can flag the
    # most recently stored MHN pattern for boosted replay. The flag
    # is consumed (drained) at the end of the next replay cycle so
    # that the boost only applies once per high-arousal event. When
    # False, all V4.1 paths are no-ops — V3.9.4 behaviour is exactly
    # preserved.
    affect_modulates: bool = False
    # V5.6.16 — B7 fix : seuil baissé de 0.6 à 0.4.
    # Avec un détecteur lexical (mots-clés affectifs dans dictionnaire),
    # l'arousal mesuré dépasse rarement 0.5 même sur des messages très
    # chargés ("je suis furax"). À 0.6, le boost ne se déclenchait
    # jamais en pratique. 0.4 = seuil empirique observé en tests.
    # Quand un détecteur ML sera ajouté, on pourra remonter à 0.6.
    affect_ripple_arousal_threshold: float = 0.4
    # Multiplicative boost (≥1.0) applied during replay to attention
    # of affect-flagged patterns. 1.0 = no boost.
    affect_consolidation_boost_factor: float = 1.5


# ── Ripple state tracker ──────────────────────────────────────────────

class RippleTracker:
    """Counts high-saliency events between microsleeps.

    Thread-safe — calls happen from the request-handling thread but
    the read in microsleep happens from the microsleep thread.
    """

    def __init__(self, config: MicrosleepConfig) -> None:
        self.config = config
        self._counter = 0
        self._lock = threading.Lock()
        self._replay_counts: dict[int, int] = {}
        # V4.1: bounded FIFO of pattern indices flagged for boosted
        # consolidation by an external (cognitive_state) arousal signal.
        # Capped at 64 to bound memory under sustained high-arousal
        # bursts; oldest entries fall off the front.
        self._affect_flagged: list[int] = []
        self._affect_flagged_max = 64

    def record_event(
        self,
        surprise: float,
        *,
        affect_intensity: float = 0.0,
        affect_arousal: float = 0.0,
        last_pattern_idx: int | None = None,
    ) -> None:
        """Record a waking event. Called from post-generation.

        V4.1 additive kwargs (all optional, defaults preserve V3.9.4
        behaviour exactly):

        ``affect_intensity`` / ``affect_arousal`` carry the cognitive
        state's signal. When ``affect_modulates`` is enabled in the
        config AND ``affect_arousal`` exceeds the threshold AND a
        valid ``last_pattern_idx`` is provided, the pattern index is
        appended to ``_affect_flagged`` for the next replay cycle.
        """
        if surprise >= self.config.ripple_surprise_threshold:
            with self._lock:
                self._counter += 1

        # V4.1: optional affect flagging — guarded by config flag.
        # If affect_modulates is False, this whole branch is skipped
        # so V3.9.4 timing/state is preserved.
        if not self.config.affect_modulates:
            return
        if last_pattern_idx is None or last_pattern_idx < 0:
            return
        try:
            arousal = float(affect_arousal)
        except (TypeError, ValueError):
            return
        if arousal < self.config.affect_ripple_arousal_threshold:
            return
        with self._lock:
            self._affect_flagged.append(int(last_pattern_idx))
            # FIFO bound — drop oldest entries beyond the cap.
            if len(self._affect_flagged) > self._affect_flagged_max:
                overflow = len(self._affect_flagged) - self._affect_flagged_max
                self._affect_flagged = self._affect_flagged[overflow:]

    def should_ripple(self) -> bool:
        """Check if we have enough accumulated saliency for a ripple."""
        with self._lock:
            return self._counter >= self.config.ripple_trigger_count

    def reset(self) -> int:
        """Reset the counter and return its prior value."""
        with self._lock:
            value = self._counter
            self._counter = 0
            return value

    def increment_replay(self, pattern_idx: int) -> int:
        """Track that an MHN pattern was replayed; return the new count."""
        with self._lock:
            self._replay_counts[pattern_idx] = (
                self._replay_counts.get(pattern_idx, 0) + 1
            )
            return self._replay_counts[pattern_idx]

    def get_replay_count(self, pattern_idx: int) -> int:
        with self._lock:
            return self._replay_counts.get(pattern_idx, 0)

    # ── V4.1: affect flagging API ─────────────────────────────────────

    def is_affect_flagged(self, pattern_idx: int) -> bool:
        """True if pattern_idx is currently flagged for boosted replay."""
        with self._lock:
            return pattern_idx in self._affect_flagged

    def affect_flagged_count(self) -> int:
        with self._lock:
            return len(self._affect_flagged)

    def clear_affect_flags(self) -> list[int]:
        """Drain all current flags and return them.

        Called by the replay engine at the end of a cycle so each
        flag is consumed exactly once.
        """
        with self._lock:
            drained = list(self._affect_flagged)
            self._affect_flagged = []
            return drained


# ── Replay engine ─────────────────────────────────────────────────────

class ReplayEngine:
    """Performs forward and reverse replay of MHN pattern sequences.

    Replay reinforces the temporal association between adjacent
    patterns by boosting the attention of all patterns in a sequence
    proportional to the number of replays they participate in.
    """

    def __init__(self, mhn: Any, config: MicrosleepConfig) -> None:
        self.mhn = mhn
        self.config = config

    def replay(self, tracker: RippleTracker, ripple_active: bool = False) -> dict:
        """Run a replay phase.

        Selects the top-K most recently stored MHN patterns and
        replays them in sequences, both forward and reverse.

        Parameters
        ----------
        tracker : RippleTracker
            Used to record per-pattern replay counts.
        ripple_active : bool
            If True, doubles the boost (sharp-wave ripple intensity).

        Returns
        -------
        dict
            Stats: ``sequences_replayed``, ``patterns_boosted``,
            ``ripple_active``.
        """
        if self.mhn.n_stored < 2:
            return {
                "sequences_replayed": 0,
                "patterns_boosted": 0,
                "ripple_active": ripple_active,
            }

        seq_len = min(self.config.replay_sequence_length, self.mhn.n_stored)
        n_sequences = self.config.replay_n_sequences

        # Select the most recent patterns ranked by their store order.
        # We fall back to all stored patterns if there aren't enough.
        recent_indices = list(range(max(0, self.mhn.n_stored - seq_len * n_sequences),
                                    self.mhn.n_stored))

        if len(recent_indices) < seq_len:
            return {
                "sequences_replayed": 0,
                "patterns_boosted": 0,
                "ripple_active": ripple_active,
            }

        # Build n_sequences contiguous chunks of length seq_len.
        sequences: list[list[int]] = []
        for i in range(0, len(recent_indices) - seq_len + 1, seq_len):
            chunk = recent_indices[i:i + seq_len]
            if len(chunk) == seq_len:
                sequences.append(chunk)

        # Apply attention boost forward + reverse for each sequence.
        boost = self.config.replay_attention_boost
        if ripple_active:
            boost *= self.config.ripple_boost_multiplier

        # V4.1: snapshot the current affect-flag set so we can apply
        # the multiplicative boost during this cycle. We drain the
        # tracker after the cycle so each flag is consumed exactly
        # once.
        affect_flag_set: set[int] = set()
        affect_boost_factor = 1.0
        if self.config.affect_modulates:
            try:
                affect_flag_set = set(tracker._affect_flagged)  # snapshot
                affect_boost_factor = max(
                    1.0, float(self.config.affect_consolidation_boost_factor)
                )
            except Exception:
                affect_flag_set = set()
                affect_boost_factor = 1.0

        boosted = set()
        affect_boosted = set()
        for seq in sequences:
            # Forward
            for idx in seq:
                local_boost = boost
                if affect_boost_factor > 1.0 and idx in affect_flag_set:
                    local_boost = boost * affect_boost_factor
                    affect_boosted.add(idx)
                self._boost_attention(idx, local_boost)
                tracker.increment_replay(idx)
                boosted.add(idx)
            # Reverse — same patterns, opposite order. Half the boost
            # captures the "rewind" reinforcement.
            for idx in reversed(seq):
                local_boost = boost * 0.5
                if affect_boost_factor > 1.0 and idx in affect_flag_set:
                    local_boost = (boost * 0.5) * affect_boost_factor
                self._boost_attention(idx, local_boost)
                tracker.increment_replay(idx)

        # V4.1: consume the flags (drain) so each high-arousal event
        # only boosts one cycle. No-op when affect_modulates=False.
        if self.config.affect_modulates:
            try:
                tracker.clear_affect_flags()
            except Exception:
                pass

        return {
            "sequences_replayed": len(sequences),
            "patterns_boosted": len(boosted),
            "ripple_active": ripple_active,
            "affect_boosted": len(affect_boosted),
        }

    def _boost_attention(self, idx: int, amount: float) -> None:
        """Increase the attention level of MHN pattern at ``idx``.

        Uses the MHN's ``attention`` tensor if present (Lythéa MHN),
        otherwise no-op. Clamped to [0, 1].
        """
        if not hasattr(self.mhn, "attention"):
            return
        if idx < 0 or idx >= self.mhn.attention.shape[0]:
            return
        try:
            new_val = float(self.mhn.attention[idx].item()) + amount
            self.mhn.attention[idx] = max(0.0, min(1.0, new_val))
        except Exception:
            pass


# ── Compression: MHN → Chroma transfer ────────────────────────────────

class MemoryCompressor:
    """Transfer well-replayed, high-attention MHN patterns to Chroma.

    Implements the hippocampus → neocortex consolidation step. Called
    at the end of each microsleep, after replay has updated attention.
    """

    def __init__(
        self,
        mhn: Any,
        chroma_collection: Any,
        config: MicrosleepConfig,
    ) -> None:
        self.mhn = mhn
        self.chroma = chroma_collection
        self.config = config

    def compress(self, tracker: RippleTracker) -> dict:
        """Transfer eligible patterns. Returns count of patterns moved.

        Eligibility:
        - replay_count >= compression_replay_threshold
        - attention >= compression_attention_threshold
        - has a stored text payload (otherwise nothing to archive)
        """
        if not hasattr(self.mhn, "attention") or not hasattr(self.mhn, "metadata"):
            return {"compressed": 0, "skipped_no_text": 0}

        eligible: list[tuple[int, float]] = []
        for idx in range(self.mhn.n_stored):
            replay_count = tracker.get_replay_count(idx)
            if replay_count < self.config.compression_replay_threshold:
                continue
            try:
                attention = float(self.mhn.attention[idx].item())
            except Exception:
                continue
            if attention < self.config.compression_attention_threshold:
                continue
            eligible.append((idx, attention))

        # Take top-N by attention so we don't flood Chroma.
        eligible.sort(key=lambda x: x[1], reverse=True)
        eligible = eligible[: self.config.compression_max_per_cycle]

        compressed = 0
        skipped_no_text = 0
        for idx, attention in eligible:
            text = self._get_pattern_text(idx)
            if not text:
                skipped_no_text += 1
                continue
            doc_id = f"consolidated_{int(time.time() * 1000)}_{idx}"
            try:
                self.chroma.add(
                    documents=[text],
                    ids=[doc_id],
                    metadatas=[{
                        "type": "consolidated",
                        "ts": time.time(),
                        "source": "mhn_replay",
                        "attention": attention,
                        "replay_count": tracker.get_replay_count(idx),
                    }],
                )
                compressed += 1
            except Exception as exc:
                log.warning("Chroma compression add failed for idx=%d: %s",
                            idx, exc)

        return {"compressed": compressed, "skipped_no_text": skipped_no_text}

    def _get_pattern_text(self, idx: int) -> str | None:
        """Best-effort extraction of the textual payload for an MHN pattern."""
        try:
            meta = self.mhn.metadata[idx] if idx < len(self.mhn.metadata) else None
            if isinstance(meta, dict):
                return meta.get("text") or meta.get("doc")
            if isinstance(meta, str):
                return meta
        except Exception:
            pass
        return None


# ── Top-level orchestrator ────────────────────────────────────────────

class MicrosleepManager:
    """Coordinates the enhanced microsleep cycle.

    Composition over inheritance: Hippocampe owns one of these and
    delegates the consolidation work. The manager itself owns the
    ripple tracker and the engine instances.
    """

    def __init__(
        self,
        sdm: Any,
        mhn: Any,
        chroma_collection: Any,
        config: MicrosleepConfig | None = None,
    ) -> None:
        self.sdm = sdm
        self.mhn = mhn
        self.chroma = chroma_collection
        self.config = config or MicrosleepConfig()
        self.tracker = RippleTracker(self.config)
        self.replay_engine = ReplayEngine(mhn, self.config)
        self.compressor = MemoryCompressor(mhn, chroma_collection, self.config)

    def record_event(
        self,
        surprise: float,
        *,
        affect_intensity: float = 0.0,
        affect_arousal: float = 0.0,
        last_pattern_idx: int | None = None,
    ) -> None:
        """Forward a waking event to the ripple tracker.

        V4.1 additive kwargs (defaults preserve V3.9.4 behaviour):
        ``affect_intensity`` / ``affect_arousal`` / ``last_pattern_idx``
        carry the cognitive_state's signal for affect-modulated
        consolidation. See ``RippleTracker.record_event`` for details.
        """
        self.tracker.record_event(
            surprise,
            affect_intensity=affect_intensity,
            affect_arousal=affect_arousal,
            last_pattern_idx=last_pattern_idx,
        )

    def consolidate(
        self,
        rehearse_top_k: int,
        rehearse_boost: float,
    ) -> dict:
        """Run the full enhanced consolidation cycle.

        Order:
            1. Check for ripple → adjust rehearsal boost
            2. SDM rehearse + decay + prune
            3. Replay (forward + reverse on recent sequences)
            4. Compress eligible patterns to Chroma
            5. Reset ripple counter

        Returns a stats dict for logging / monitoring.
        """
        stats: dict = {
            "ripple_active": False,
            "ripple_event_count": 0,
            "rehearsed_top_k": rehearse_top_k,
        }

        # Phase 1: ripple decision
        ripple_active = self.tracker.should_ripple()
        stats["ripple_active"] = ripple_active

        boost = rehearse_boost
        if ripple_active:
            boost = rehearse_boost * self.config.ripple_boost_multiplier
            log.info("⚡ Sharp-wave ripple active — boost x%.1f",
                     self.config.ripple_boost_multiplier)

        # Phase 2: SDM rehearse / decay / prune
        try:
            self.sdm.rehearse(top_k=rehearse_top_k, boost=boost)
            self.sdm.decay()
            stats["sdm_pruned"] = self.sdm.prune()
        except Exception as exc:
            log.warning("SDM consolidation step failed: %s", exc)
            stats["sdm_pruned"] = 0

        # Phase 3: replay
        try:
            replay_stats = self.replay_engine.replay(
                self.tracker, ripple_active=ripple_active,
            )
            stats.update({
                "replay_sequences": replay_stats["sequences_replayed"],
                "replay_patterns": replay_stats["patterns_boosted"],
            })
        except Exception as exc:
            log.warning("Replay phase failed: %s", exc)
            stats["replay_sequences"] = 0
            stats["replay_patterns"] = 0

        # Phase 4: compress to Chroma
        try:
            compress_stats = self.compressor.compress(self.tracker)
            stats.update({
                "compressed_to_chroma": compress_stats["compressed"],
            })
        except Exception as exc:
            log.warning("Compression phase failed: %s", exc)
            stats["compressed_to_chroma"] = 0

        # Phase 5: reset ripple counter
        stats["ripple_event_count"] = self.tracker.reset()

        return stats
