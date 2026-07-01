"""Tests for the temporal awareness module."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rune.temporal import (
    TemporalContext,
    annotate_with_freshness,
    humanise_delta,
    humanise_duration,
)


# ── humanise_delta buckets ────────────────────────────────────────────

def test_humanise_delta_instant():
    assert humanise_delta(0) == "à l'instant"
    assert humanise_delta(15) == "à l'instant"
    assert humanise_delta(29.9) == "à l'instant"


def test_humanise_delta_seconds():
    assert humanise_delta(45).endswith("secondes")
    # Negative input clamped to 0
    assert humanise_delta(-100) == "à l'instant"


def test_humanise_delta_minutes_singular_plural():
    assert humanise_delta(60) == "il y a 1 minute"
    assert humanise_delta(120) == "il y a 2 minutes"
    assert humanise_delta(3540) == "il y a 59 minutes"


def test_humanise_delta_hours():
    assert humanise_delta(3600) == "il y a 1 heure"
    assert humanise_delta(7200) == "il y a 2 heures"
    assert humanise_delta(82800) == "il y a 23 heures"


def test_humanise_delta_days():
    assert humanise_delta(86400) == "il y a 1 jour"
    assert humanise_delta(86400 * 6) == "il y a 6 jours"


def test_humanise_delta_weeks():
    assert humanise_delta(86400 * 7) == "il y a 1 semaine"
    assert humanise_delta(86400 * 21) == "il y a 3 semaines"


def test_humanise_delta_months():
    assert humanise_delta(86400 * 30) == "il y a 1 mois"
    assert humanise_delta(86400 * 90) == "il y a 3 mois"


def test_humanise_delta_years():
    assert humanise_delta(86400 * 365) == "il y a 1 an"
    assert humanise_delta(86400 * 365 * 3) == "il y a 3 ans"


# ── humanise_duration ────────────────────────────────────────────────

def test_humanise_duration_basic():
    assert "moins" in humanise_duration(30).lower()
    assert humanise_duration(60) == "1 minute"
    assert humanise_duration(180) == "3 minutes"
    assert humanise_duration(3600) == "1 heure"


def test_humanise_duration_mixed_hours_minutes():
    # 1 h 30 min
    assert humanise_duration(5400) == "1 h 30"


def test_humanise_duration_days():
    assert humanise_duration(86400) == "1 jour"
    assert humanise_duration(86400 * 5) == "5 jours"


# ── TemporalContext.render ────────────────────────────────────────────

def test_render_minimal_only_now():
    """With only `now`, render still produces date+period+directive."""
    ctx = TemporalContext(now=datetime(2026, 4, 28, 10, 30))
    out = ctx.render()
    assert "[Conscience du temps]" in out
    assert "mardi" in out  # 2026-04-28 was a Tuesday
    assert "avril" in out
    assert "2026" in out
    assert "matinée" in out  # 10:30 → en matinée
    # Should NOT contain optional lines
    assert "dernier message" not in out
    assert "consolidation" not in out


def test_render_with_recent_last_message_skips_line():
    """If the last message was <30s ago, the gap line is omitted (too noisy)."""
    now = datetime(2026, 4, 28, 14, 0, 0)
    last = (now - timedelta(seconds=10)).timestamp()
    ctx = TemporalContext(now=now, last_message_ts=last)
    out = ctx.render()
    assert "dernier message" not in out


def test_render_with_meaningful_gap_includes_line():
    now = datetime(2026, 4, 28, 14, 0, 0)
    last = (now - timedelta(hours=3)).timestamp()
    ctx = TemporalContext(now=now, last_message_ts=last)
    out = ctx.render()
    assert "il y a 3 heures" in out


def test_render_session_age_only_if_meaningful():
    """A session created 30s ago shouldn't trigger the duration line."""
    now = datetime(2026, 4, 28, 14, 0, 0)
    ctx = TemporalContext(
        now=now,
        session_created_ts=(now - timedelta(seconds=20)).timestamp(),
    )
    out = ctx.render()
    assert "commencé" not in out


def test_render_session_age_long_session():
    now = datetime(2026, 4, 28, 14, 0, 0)
    ctx = TemporalContext(
        now=now,
        session_created_ts=(now - timedelta(days=2)).timestamp(),
    )
    out = ctx.render()
    assert "commencé" in out
    assert "2 jours" in out


def test_render_microsleep_freshness():
    now = datetime(2026, 4, 28, 14, 0, 0)
    ctx = TemporalContext(
        now=now,
        last_microsleep_ts=(now - timedelta(minutes=20)).timestamp(),
    )
    out = ctx.render()
    assert "consolidation" in out
    assert "20 minutes" in out


def test_render_periods_of_day():
    """Different hours should map to the right French period."""
    cases = [
        (7, "tôt le matin"),
        (10, "matinée"),
        (13, "déjeuner"),
        (16, "après-midi"),
        (20, "soirée"),
        (23, "tard"),
        (3, "nuit"),
    ]
    for hour, expected in cases:
        ctx = TemporalContext(now=datetime(2026, 4, 28, hour, 0))
        out = ctx.render()
        assert expected in out, f"hour={hour}: '{expected}' not in render"


def test_render_includes_behavioural_directive():
    """The model must be told to USE the time context, not just see it."""
    ctx = TemporalContext(now=datetime.now())
    out = ctx.render()
    # The directive should mention adapting tone/greetings
    assert "23h" in out or "minute" in out  # one of the example anchors


# ── annotate_with_freshness ───────────────────────────────────────────

def test_annotate_skips_zero_ts():
    """Items without a timestamp pass through unchanged."""
    items = [("hello", 0.0), ("world", 0)]
    out = annotate_with_freshness(items, now=datetime(2026, 4, 28))
    assert out == ["hello", "world"]


def test_annotate_skips_recent():
    """Items < 5 min old don't get the noisy 'à l'instant' suffix."""
    now = datetime(2026, 4, 28, 14, 0, 0)
    recent_ts = (now - timedelta(seconds=120)).timestamp()
    out = annotate_with_freshness([("recent fact", recent_ts)], now=now)
    assert out == ["recent fact"]


def test_annotate_with_meaningful_age():
    now = datetime(2026, 4, 28, 14, 0, 0)
    old_ts = (now - timedelta(days=3)).timestamp()
    out = annotate_with_freshness([("old fact", old_ts)], now=now)
    assert out == ["old fact (il y a 3 jours)"]


def test_annotate_default_now_is_current_time():
    """When `now` is None, function uses datetime.now()."""
    import time as _time
    old_ts = _time.time() - 86400 * 2  # 2 days ago
    out = annotate_with_freshness([("x", old_ts)])
    assert "il y a 2 jours" in out[0]
