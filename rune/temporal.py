"""Temporal awareness — gives the LLM a sense of when it is and how time flows.

This module synthesises three layers of temporal context for the prompt:

1. **Absolute now** — date, weekday, hour, period of day.
2. **Conversational time** — gap since last turn, session age, total duration.
3. **Memory freshness** — age of KG facts and Chroma episodes.

The output is meant to be injected into the system prompt right after the
core persona, before the RAG context. This is the same level of importance
as identity: the model needs both "who you talk to" and "when you talk".

Design notes
------------
- We try to set ``fr_FR.UTF-8`` locale once at import time so weekday and
  month names come out in French. If unavailable (Docker minimal images,
  Windows without French pack), we fall back to a manual mapping.
- All formatting helpers are pure functions — easy to unit-test.
- Time gaps are bucketised in a human-readable way ("il y a 3 minutes",
  "il y a 2 jours") rather than emitted as raw seconds, because LLMs reason
  much better about quantised buckets than about exact deltas.
"""
from __future__ import annotations

import locale
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

log = logging.getLogger("rune.temporal")

# ── Locale bootstrap (best-effort) ────────────────────────────────────

_LOCALE_OK = False
for _candidate in ("fr_FR.UTF-8", "fr_FR.utf8", "fr_FR", "French_France.1252"):
    try:
        locale.setlocale(locale.LC_TIME, _candidate)
        _LOCALE_OK = True
        log.debug("Locale set to %s for temporal awareness", _candidate)
        break
    except locale.Error:
        continue

# Manual French mappings — used when the OS locale isn't installed
_WEEKDAYS_FR = (
    "lundi", "mardi", "mercredi", "jeudi",
    "vendredi", "samedi", "dimanche",
)
_MONTHS_FR = (
    "", "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)


def _format_datetime_fr(dt: datetime) -> str:
    """Format a datetime in French regardless of system locale.

    Returns a string like ``"mardi 28 avril 2026 à 14:32"``.
    """
    if _LOCALE_OK:
        # %A and %B will be French
        return dt.strftime("%A %d %B %Y à %H:%M")
    weekday = _WEEKDAYS_FR[dt.weekday()]
    month = _MONTHS_FR[dt.month]
    return f"{weekday} {dt.day} {month} {dt.year} à {dt.hour:02d}:{dt.minute:02d}"


def _period_of_day(dt: datetime) -> str:
    """Return a coarse period of the day in French.

    Buckets aligned with circadian markers used in chronobiology rather
    than purely calendar conventions.
    """
    h = dt.hour
    if 5 <= h < 9:
        return "tôt le matin"
    if 9 <= h < 12:
        return "en matinée"
    if 12 <= h < 14:
        return "à l'heure du déjeuner"
    if 14 <= h < 18:
        return "dans l'après-midi"
    if 18 <= h < 22:
        return "en soirée"
    if 22 <= h < 24:
        return "tard le soir"
    # 0-5
    return "en pleine nuit"


def humanise_delta(seconds: float) -> str:
    """Return a human-friendly French gap string from a positive delta in seconds.

    Examples
    --------
    >>> humanise_delta(15)
    "à l'instant"
    >>> humanise_delta(180)
    "il y a 3 minutes"
    >>> humanise_delta(7200)
    "il y a 2 heures"
    >>> humanise_delta(86400 * 3)
    "il y a 3 jours"
    """
    s = max(seconds, 0)
    if s < 30:
        return "à l'instant"
    if s < 60:
        return f"il y a {int(s)} secondes"
    if s < 3600:
        m = int(s // 60)
        return f"il y a {m} minute" + ("s" if m > 1 else "")
    if s < 86400:
        h = int(s // 3600)
        return f"il y a {h} heure" + ("s" if h > 1 else "")
    if s < 86400 * 7:
        d = int(s // 86400)
        return f"il y a {d} jour" + ("s" if d > 1 else "")
    if s < 86400 * 30:
        w = int(s // (86400 * 7))
        return f"il y a {w} semaine" + ("s" if w > 1 else "")
    if s < 86400 * 365:
        mo = int(s // (86400 * 30))
        return f"il y a {mo} mois"
    y = int(s // (86400 * 365))
    return f"il y a {y} an" + ("s" if y > 1 else "")


def humanise_duration(seconds: float) -> str:
    """Render an elapsed duration as French prose.

    Differs from :func:`humanise_delta` because it talks about a span,
    not a point in the past.
    """
    s = max(seconds, 0)
    if s < 60:
        return "moins d'une minute"
    if s < 3600:
        m = int(s // 60)
        return f"{m} minute" + ("s" if m > 1 else "")
    if s < 86400:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        if m == 0:
            return f"{h} heure" + ("s" if h > 1 else "")
        return f"{h} h {m:02d}"
    d = int(s // 86400)
    return f"{d} jour" + ("s" if d > 1 else "")


# ── Public API ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TemporalContext:
    """A bundle of temporal signals to be rendered into the system prompt."""

    now: datetime
    last_message_ts: float | None = None
    session_created_ts: float | None = None
    last_microsleep_ts: float | None = None

    def render(self) -> str:
        """Render the context as a French paragraph for the LLM.

        Returns
        -------
        str
            Multi-line block ready to be appended to the system prompt.
            Empty string if no temporal info is available (defensive).
        """
        lines: list[str] = ["[Conscience du temps]"]
        lines.append(
            f"Nous sommes le {_format_datetime_fr(self.now)}, "
            f"{_period_of_day(self.now)}."
        )

        if self.last_message_ts:
            gap = self.now.timestamp() - self.last_message_ts
            if gap > 30:  # don't bother for the very first turn or fast back-and-forth
                lines.append(
                    f"Le dernier message échangé était {humanise_delta(gap)}."
                )

        if self.session_created_ts:
            session_age = self.now.timestamp() - self.session_created_ts
            if session_age > 300:  # only mention if non-trivial
                lines.append(
                    f"Cette conversation a commencé {humanise_delta(session_age)} "
                    f"(durée écoulée : {humanise_duration(session_age)})."
                )

        if self.last_microsleep_ts:
            ms_gap = self.now.timestamp() - self.last_microsleep_ts
            if ms_gap > 60:
                lines.append(
                    f"Ta dernière phase de consolidation mémoire date de "
                    f"{humanise_delta(ms_gap)}."
                )

        # Behavioural directive for the LLM. Without it the model may receive
        # the temporal context but not actually use it in its response style.
        lines.append(
            "Prends en compte ce contexte temporel pour adapter naturellement "
            "ton ton et tes salutations (par exemple : ne dis pas 'bonjour' "
            "à 23h, ni 'bon retour' si on vient de se parler il y a une minute)."
        )

        return "\n".join(lines)


# ── Helpers for memory freshness ──────────────────────────────────────

def annotate_with_freshness(
    items: Iterable[tuple[str, float]],
    now: datetime | None = None,
) -> list[str]:
    """Annotate text items with their age in human-readable form.

    Parameters
    ----------
    items : iterable of (text, timestamp)
        The memory items to annotate.
    now : datetime, optional
        Reference point. Defaults to current time.

    Returns
    -------
    list[str]
        Items rewritten as ``"<text> (<freshness>)"``.

    Notes
    -----
    Useful for KG fact summaries and Chroma retrieval results — gives
    the model a sense that "this user told me X last week" rather than
    "X is just a fact in my context".
    """
    ref = (now or datetime.now()).timestamp()
    out: list[str] = []
    for text, ts in items:
        if ts <= 0:
            out.append(text)
            continue
        gap = ref - ts
        # Only annotate if the gap is meaningful (> 5 minutes)
        if gap < 300:
            out.append(text)
        else:
            out.append(f"{text} ({humanise_delta(gap)})")
    return out
