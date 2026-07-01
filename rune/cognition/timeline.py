"""V4.3 — Timeline: narrative chronology extraction.

Inspiration biologique
----------------------
Hippocampe + cortex temporal latéral. Construire une représentation
explicite des événements (passé, présent, futur) cités dans un message
permet de raisonner sur la temporalité narrative — savoir que "j'ai
soutenu ma thèse en 2018" est antérieur à "j'ai rejoint le CEA en 2019".

Position dans le cycle cognitif
-------------------------------
Hook A.4 (Phase A) :
    events = timeline.extract(user_message)
    block = timeline.render_block(events)
    → injecté dans system_text

Architecture
------------
TimelineExtractor parcourt le message avec 5 familles de patterns :
    absolute  — dates ISO, FR DMY, FR named ("12 mai 2026")
    relative  — "hier", "avant-hier", "il y a 3 jours", "dans 2 mois"
    duration  — "pendant 6 mois", "depuis 2018"
    ordinal   — "le premier mai", "the third of June"
    vague     — "récemment", "bientôt", "soon"

Chaque match → TimelineEvent avec confidence, normalized form, raw text.

Design contracts
----------------
1. Extraction tolère les doublons (dédup par (kind, normalized)).
2. Confidence < timeline_render_min_confidence → exclu du rendu mais
   conservé dans events list (debug).
3. Vague events : conservés mais rendus seulement si timeline_render_vague.
4. render_block tronque à block_max_chars avec '…'.
5. Pure Python.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from rune.cognition.warnings_v4 import format_warning

log = logging.getLogger("lythea.cognition.timeline")


EVENT_KINDS = ("absolute", "relative", "duration", "ordinal", "vague")


# ════════════════════════════════════════════════════════════════════
# TimelineEvent dataclass
# ════════════════════════════════════════════════════════════════════


@dataclass
class TimelineEvent:
    raw: str  # original substring as found in the text
    kind: str  # ∈ EVENT_KINDS
    normalized: str = ""  # ISO date / "P3D" / "T-86400" / etc.
    confidence: float = 0.5
    span: tuple[int, int] = (0, 0)  # (start, end) char offsets

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "kind": self.kind,
            "normalized": self.normalized,
            "confidence": self.confidence,
            "span": list(self.span),
        }


# ════════════════════════════════════════════════════════════════════
# Lexicons & patterns
# ════════════════════════════════════════════════════════════════════


# Days of the week to allow disambiguation later (not yet used).
_FR_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
    "decembre": 12,
}

_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


# ── Relative offset patterns (single-word + multi-word) ─────────────

# Map of single tokens → offset in seconds.
_RELATIVE_TOKENS_SEC = {
    "aujourd'hui": 0,
    "today": 0,
    "hier": -86400,
    "yesterday": -86400,
    "avant-hier": -2 * 86400,
    "demain": 86400,
    "tomorrow": 86400,
    "après-demain": 2 * 86400,
    "apres-demain": 2 * 86400,
}

# Numeric relative ("il y a N <unit>", "dans N <unit>", "in N <unit>")
_UNIT_TO_SEC = {
    "seconde": 1, "secondes": 1, "second": 1, "seconds": 1, "sec": 1, "s": 1,
    "minute": 60, "minutes": 60, "min": 60, "m": 60,
    "heure": 3600, "heures": 3600, "hour": 3600, "hours": 3600, "h": 3600,
    "jour": 86400, "jours": 86400, "day": 86400, "days": 86400, "j": 86400,
    "semaine": 7 * 86400, "semaines": 7 * 86400, "week": 7 * 86400, "weeks": 7 * 86400,
    "mois": 30 * 86400, "month": 30 * 86400, "months": 30 * 86400,
    "an": 365 * 86400, "ans": 365 * 86400, "année": 365 * 86400, "années": 365 * 86400,
    "year": 365 * 86400, "years": 365 * 86400,
}

# Pattern: "il y a 3 jours", "il y a deux semaines"
_PAST_NUMERIC_RE = re.compile(
    r"\bil y a\s+(\d+|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+"
    r"([a-zéèêà]+)",
    re.IGNORECASE,
)
# EN: "3 days ago"
_PAST_NUMERIC_EN_RE = re.compile(
    r"\b(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)
# FR future: "dans 2 mois"
_FUTURE_NUMERIC_RE = re.compile(
    r"\bdans\s+(\d+|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+"
    r"([a-zéèêà]+)",
    re.IGNORECASE,
)
# EN future: "in 2 months"
_FUTURE_NUMERIC_EN_RE = re.compile(
    r"\bin\s+(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
# Duration: "pendant 3 mois", "depuis 6 mois", "for 3 months", "since 2018"
_DURATION_RE = re.compile(
    r"\b(?:pendant|depuis|for|since)\s+(\d+|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+"
    r"([a-zéèêà]+)",
    re.IGNORECASE,
)

_FR_NUMBER_WORDS = {
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
}


# ── Absolute date patterns ───────────────────────────────────────────

# ISO YYYY-MM-DD
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
# FR DMY: 12/05/2026 or 12-05-2026
_FR_DMY_RE = re.compile(r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b")
# FR named: "12 mai 2026", "5 octobre 2024"
_FR_NAMED_RE = re.compile(
    r"\b(\d{1,2})\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|"
    r"août|aout|septembre|octobre|novembre|décembre|decembre)"
    r"(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
# EN named: "May 12, 2026" or "May 12 2026"
_EN_NAMED_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+(\d{1,2})(?:,?\s+(\d{4}))?\b",
    re.IGNORECASE,
)

# Year alone (use cautiously — "1800 km" is not 1800 AD).
# Require year-like context: "en 2018", "in 2018", "année 2018".
_YEAR_CTX_RE = re.compile(
    r"\b(?:en|in|année|année\s+de|year)\s+(\d{4})\b",
    re.IGNORECASE,
)


# ── Time-of-day (HH:MM) ──────────────────────────────────────────────

_TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[h:]([0-5]\d)\b", re.IGNORECASE)


# ── Ordinal (le premier, the third of) ───────────────────────────────

_FR_ORDINALS = {
    "premier": 1, "1er": 1, "1ère": 1, "1re": 1,
    "deuxième": 2, "2ème": 2, "2e": 2, "second": 2, "seconde": 2,
    "troisième": 3, "3ème": 3, "3e": 3,
    "quatrième": 4, "4ème": 4, "4e": 4,
    "cinquième": 5, "5ème": 5, "5e": 5,
}
# Pattern: "le premier mai", "le 1er mai 2025"
_FR_ORDINAL_RE = re.compile(
    r"\ble\s+(premier|deuxième|troisième|quatrième|cinquième|"
    r"\d+(?:er|ère|re|ème|e))\s+"
    r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)"
    r"(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_EN_ORDINAL_RE = re.compile(
    r"\bthe\s+(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+(?:of\s+)?"
    r"(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
    r"(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_EN_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
}


# ── Vague markers ────────────────────────────────────────────────────

_VAGUE_RE = re.compile(
    r"\b(récemment|recemment|bientôt|bientot|prochainement|"
    r"il y a longtemps|jadis|plus tard|"
    r"recently|soon|later|long ago|in the past|in the future)\b",
    re.IGNORECASE,
)


# ════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════


def _parse_int_or_word(tok: str) -> int | None:
    if not tok:
        return None
    t = tok.lower().strip()
    if t.isdigit():
        try:
            return int(t)
        except ValueError:
            return None
    return _FR_NUMBER_WORDS.get(t)


def _humanize_relative_seconds(offset_sec: int) -> str:
    """Convert a signed second offset into a French label."""
    if offset_sec == 0:
        return "aujourd'hui"
    abs_sec = abs(offset_sec)
    if offset_sec < 0:
        if abs_sec == 86400:
            return "hier"
        if abs_sec == 2 * 86400:
            return "avant-hier"
        if abs_sec >= 365 * 86400:
            n = abs_sec // (365 * 86400)
            return f"il y a {n} an{'s' if n > 1 else ''}"
        if abs_sec >= 30 * 86400:
            n = abs_sec // (30 * 86400)
            return f"il y a {n} mois"
        if abs_sec >= 7 * 86400:
            n = abs_sec // (7 * 86400)
            return f"il y a {n} semaine{'s' if n > 1 else ''}"
        if abs_sec >= 86400:
            n = abs_sec // 86400
            return f"il y a {n} jour{'s' if n > 1 else ''}"
        if abs_sec >= 3600:
            n = abs_sec // 3600
            return f"il y a {n}h"
        return f"il y a {abs_sec}s"
    else:
        if abs_sec == 86400:
            return "demain"
        if abs_sec == 2 * 86400:
            return "après-demain"
        if abs_sec >= 365 * 86400:
            n = abs_sec // (365 * 86400)
            return f"dans {n} an{'s' if n > 1 else ''}"
        if abs_sec >= 30 * 86400:
            n = abs_sec // (30 * 86400)
            return f"dans {n} mois"
        if abs_sec >= 7 * 86400:
            n = abs_sec // (7 * 86400)
            return f"dans {n} semaine{'s' if n > 1 else ''}"
        if abs_sec >= 86400:
            n = abs_sec // 86400
            return f"dans {n} jour{'s' if n > 1 else ''}"
        if abs_sec >= 3600:
            n = abs_sec // 3600
            return f"dans {n}h"
        return f"dans {abs_sec}s"


def _safe_make_iso(year: int, month: int, day: int) -> str | None:
    """Build YYYY-MM-DD if values are valid, else None."""
    try:
        d = datetime(year=year, month=month, day=day, tzinfo=timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None
    return d.strftime("%Y-%m-%d")


# ════════════════════════════════════════════════════════════════════
# TimelineExtractor
# ════════════════════════════════════════════════════════════════════


@dataclass
class TimelineConfig:
    max_events: int = 8
    block_max_chars: int = 600
    render_min_confidence: float = 0.3
    render_vague: bool = False
    now: datetime | None = None  # injectable for testing


class TimelineExtractor:
    """Extract chronological events from free text."""

    def __init__(self, config: TimelineConfig | None = None):
        self.config = config or TimelineConfig()

    def _now(self) -> datetime:
        return self.config.now or datetime.now(tz=timezone.utc)

    def extract(self, text: str) -> list[TimelineEvent]:
        if not text or not isinstance(text, str):
            return []
        try:
            return self._extract_inner(text)
        except Exception:
            log.warning("TimelineExtractor.extract crashed", exc_info=True)
            return []

    def _extract_inner(self, text: str) -> list[TimelineEvent]:
        events: list[TimelineEvent] = []

        # --- 1. Single-token relatives (hier, demain, …) -----------
        normalized = unicodedata.normalize("NFKC", text)
        normalized_lower = normalized.lower()
        for tok, sec_offset in _RELATIVE_TOKENS_SEC.items():
            # Use word-boundary search where possible.
            pat = r"\b" + re.escape(tok) + r"\b"
            for m in re.finditer(pat, normalized_lower):
                events.append(
                    TimelineEvent(
                        raw=normalized[m.start(): m.end()],
                        kind="relative",
                        normalized=_humanize_relative_seconds(sec_offset),
                        confidence=0.85,
                        span=(m.start(), m.end()),
                    )
                )

        # --- 2. Numeric relatives FR/EN past + future + duration ---
        for m in _PAST_NUMERIC_RE.finditer(text):
            n = _parse_int_or_word(m.group(1))
            unit = m.group(2).lower()
            sec = _UNIT_TO_SEC.get(unit)
            if n is None or sec is None:
                continue
            offset = -n * sec
            events.append(
                TimelineEvent(
                    raw=m.group(0),
                    kind="relative",
                    normalized=_humanize_relative_seconds(offset),
                    confidence=0.8,
                    span=m.span(),
                )
            )
        for m in _PAST_NUMERIC_EN_RE.finditer(text):
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            unit = m.group(2).lower()
            sec = _UNIT_TO_SEC.get(unit) or _UNIT_TO_SEC.get(unit.rstrip("s"))
            if sec is None:
                continue
            offset = -n * sec
            events.append(
                TimelineEvent(
                    raw=m.group(0),
                    kind="relative",
                    normalized=_humanize_relative_seconds(offset),
                    confidence=0.8,
                    span=m.span(),
                )
            )
        for m in _FUTURE_NUMERIC_RE.finditer(text):
            n = _parse_int_or_word(m.group(1))
            unit = m.group(2).lower()
            sec = _UNIT_TO_SEC.get(unit)
            if n is None or sec is None:
                continue
            offset = n * sec
            events.append(
                TimelineEvent(
                    raw=m.group(0),
                    kind="relative",
                    normalized=_humanize_relative_seconds(offset),
                    confidence=0.8,
                    span=m.span(),
                )
            )
        for m in _FUTURE_NUMERIC_EN_RE.finditer(text):
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            unit = m.group(2).lower()
            sec = _UNIT_TO_SEC.get(unit) or _UNIT_TO_SEC.get(unit.rstrip("s"))
            if sec is None:
                continue
            offset = n * sec
            events.append(
                TimelineEvent(
                    raw=m.group(0),
                    kind="relative",
                    normalized=_humanize_relative_seconds(offset),
                    confidence=0.8,
                    span=m.span(),
                )
            )
        for m in _DURATION_RE.finditer(text):
            n = _parse_int_or_word(m.group(1))
            unit = m.group(2).lower()
            sec = _UNIT_TO_SEC.get(unit)
            if n is None or sec is None:
                continue
            events.append(
                TimelineEvent(
                    raw=m.group(0),
                    kind="duration",
                    normalized=f"P{n}{unit[:1].upper()}",
                    confidence=0.7,
                    span=m.span(),
                )
            )

        # --- 3. Absolute dates -----------------------------------
        for m in _ISO_DATE_RE.finditer(text):
            iso = _safe_make_iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if iso:
                events.append(
                    TimelineEvent(
                        raw=m.group(0),
                        kind="absolute",
                        normalized=iso,
                        confidence=0.95,
                        span=m.span(),
                    )
                )
        for m in _FR_DMY_RE.finditer(text):
            try:
                d = int(m.group(1))
                mo = int(m.group(2))
                y = int(m.group(3))
            except ValueError:
                continue
            if y < 100:
                y += 2000
            iso = _safe_make_iso(y, mo, d)
            if iso:
                events.append(
                    TimelineEvent(
                        raw=m.group(0),
                        kind="absolute",
                        normalized=iso,
                        confidence=0.85,
                        span=m.span(),
                    )
                )
        for m in _FR_NAMED_RE.finditer(text):
            d = int(m.group(1))
            mo_name = m.group(2).lower()
            mo = _FR_MONTHS.get(mo_name)
            y_raw = m.group(3)
            y = int(y_raw) if y_raw else self._now().year
            if mo is None:
                continue
            iso = _safe_make_iso(y, mo, d)
            if iso:
                conf = 0.9 if y_raw else 0.6  # missing year → lower confidence
                events.append(
                    TimelineEvent(
                        raw=m.group(0),
                        kind="absolute",
                        normalized=iso,
                        confidence=conf,
                        span=m.span(),
                    )
                )
        for m in _EN_NAMED_RE.finditer(text):
            mo_name = m.group(1).lower()
            d = int(m.group(2))
            mo = _EN_MONTHS.get(mo_name)
            y_raw = m.group(3)
            y = int(y_raw) if y_raw else self._now().year
            if mo is None:
                continue
            iso = _safe_make_iso(y, mo, d)
            if iso:
                conf = 0.9 if y_raw else 0.6
                events.append(
                    TimelineEvent(
                        raw=m.group(0),
                        kind="absolute",
                        normalized=iso,
                        confidence=conf,
                        span=m.span(),
                    )
                )
        for m in _YEAR_CTX_RE.finditer(text):
            try:
                y = int(m.group(1))
            except ValueError:
                continue
            if 1900 <= y <= 2100:
                events.append(
                    TimelineEvent(
                        raw=m.group(0),
                        kind="absolute",
                        normalized=f"{y}",
                        confidence=0.7,
                        span=m.span(),
                    )
                )

        # --- 4. Time of day --------------------------------------
        for m in _TIME_RE.finditer(text):
            events.append(
                TimelineEvent(
                    raw=m.group(0),
                    kind="absolute",
                    normalized=f"T{m.group(1)}:{m.group(2)}",
                    confidence=0.65,
                    span=m.span(),
                )
            )

        # --- 5. Ordinals -----------------------------------------
        for m in _FR_ORDINAL_RE.finditer(text):
            ord_tok = m.group(1).lower()
            day = _FR_ORDINALS.get(ord_tok)
            if day is None:
                # numeric form like "12ème" — extract leading digits
                digits = re.match(r"\d+", ord_tok)
                day = int(digits.group(0)) if digits else None
            mo = _FR_MONTHS.get(m.group(2).lower())
            y_raw = m.group(3)
            y = int(y_raw) if y_raw else self._now().year
            if day is None or mo is None:
                continue
            iso = _safe_make_iso(y, mo, day)
            if iso:
                events.append(
                    TimelineEvent(
                        raw=m.group(0),
                        kind="ordinal",
                        normalized=iso,
                        confidence=0.7,
                        span=m.span(),
                    )
                )
        for m in _EN_ORDINAL_RE.finditer(text):
            ord_tok = m.group(1).lower()
            day = _EN_ORDINALS.get(ord_tok)
            if day is None:
                digits = re.match(r"\d+", ord_tok)
                day = int(digits.group(0)) if digits else None
            mo = _EN_MONTHS.get(m.group(2).lower())
            y_raw = m.group(3)
            y = int(y_raw) if y_raw else self._now().year
            if day is None or mo is None:
                continue
            iso = _safe_make_iso(y, mo, day)
            if iso:
                events.append(
                    TimelineEvent(
                        raw=m.group(0),
                        kind="ordinal",
                        normalized=iso,
                        confidence=0.7,
                        span=m.span(),
                    )
                )

        # --- 6. Vague --------------------------------------------
        for m in _VAGUE_RE.finditer(text):
            events.append(
                TimelineEvent(
                    raw=m.group(0),
                    kind="vague",
                    normalized=m.group(0).lower(),
                    confidence=0.4,
                    span=m.span(),
                )
            )

        # --- Dedup by (kind, normalized) keeping highest confidence ---
        seen: dict[tuple[str, str], TimelineEvent] = {}
        for ev in events:
            key = (ev.kind, ev.normalized)
            if key not in seen or ev.confidence > seen[key].confidence:
                seen[key] = ev

        deduped = sorted(
            seen.values(),
            key=lambda e: (e.span[0], -e.confidence),
        )
        return deduped[: self.config.max_events]


# ════════════════════════════════════════════════════════════════════
# Render block
# ════════════════════════════════════════════════════════════════════


_ICONS = {
    "absolute": "📅",
    "relative": "⏱",
    "duration": "⏳",
    "ordinal": "🔢",
    "vague": "❓",
}


def _humanize_to_offset_sec(label: str) -> int | None:
    """Inverse of ``_humanize_relative_seconds`` for known labels.

    Returns the signed second offset, or None if the label can't be
    parsed back. Used by ``_resolved_date`` to compute absolute dates
    from human-readable relative markers.
    """
    if not label or not isinstance(label, str):
        return None
    s = label.strip().lower()
    if s in ("aujourd'hui", "aujourd hui"):
        return 0
    if s == "hier":
        return -86400
    if s == "avant-hier":
        return -2 * 86400
    if s == "demain":
        return 86400
    if s in ("après-demain", "apres-demain"):
        return 2 * 86400

    # "il y a N jours/semaines/mois/ans" or "dans N jours/..."
    m = re.match(
        r"^(il y a|dans)\s+(\d+)\s+(seconde|minute|heure|jour|semaine|mois|an)s?h?$",
        s,
    )
    if m:
        sign = -1 if m.group(1) == "il y a" else 1
        n = int(m.group(2))
        unit = m.group(3)
        unit_sec = {
            "seconde": 1, "minute": 60, "heure": 3600,
            "jour": 86400, "semaine": 7 * 86400,
            "mois": 30 * 86400, "an": 365 * 86400,
        }.get(unit, 0)
        if unit_sec > 0:
            return sign * n * unit_sec

    # "il y a Nh" / "il y a Ns" — short form
    m = re.match(r"^il y a\s+(\d+)\s*([smh])$", s)
    if m:
        n = int(m.group(1))
        unit_sec = {"s": 1, "m": 60, "h": 3600}.get(m.group(2), 0)
        if unit_sec > 0:
            return -n * unit_sec

    return None


def _parse_iso_date(normalized: str) -> datetime | None:
    """Best-effort ISO-8601 → datetime. Returns None on any failure."""
    if not normalized or not isinstance(normalized, str):
        return None
    # Common shapes the extractor produces:
    #   "2026-05-12"          (absolute, day-precision)
    #   "2026-05-12T14:00"    (absolute with time)
    #   "P3D"                 (duration in ISO 8601, ignored here)
    if normalized.startswith("P"):
        return None
    try:
        # Date-only: pad with T00:00 so fromisoformat eats it.
        if len(normalized) == 10 and normalized[4] == "-" and normalized[7] == "-":
            return datetime.fromisoformat(normalized + "T00:00:00").replace(
                tzinfo=timezone.utc,
            )
        return datetime.fromisoformat(normalized).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _resolved_date(ev: TimelineEvent, now: datetime) -> datetime | None:
    """Resolve an event to an absolute date when possible.

    Absolute events: parse the ISO normalized form.
    Relative events: parse the human-readable label back to an offset.
    Other kinds: None (cannot place on a calendar).
    """
    if ev.kind == "absolute":
        return _parse_iso_date(ev.normalized)
    if ev.kind == "relative":
        offset = _humanize_to_offset_sec(ev.normalized)
        if offset is not None:
            return now + timedelta(seconds=offset)
    return None


def _split_clauses(text: str) -> list[tuple[int, int]]:
    """Split text into clause spans by sentence-end punctuation.

    Returns list of (start_offset, end_offset) tuples covering the
    full text. Used to decide whether two timeline events were
    mentioned within the same narrative clause (and thus could
    contradict each other) vs separate clauses (independent events).
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in ".!?;\n":
            if i > start:
                spans.append((start, i + 1))
            start = i + 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _same_clause(
    span_a: tuple[int, int],
    span_b: tuple[int, int],
    clauses: list[tuple[int, int]],
) -> bool:
    """True if both spans fall inside the same clause."""
    if not clauses:
        return False
    a_clause = None
    b_clause = None
    for idx, (s, e) in enumerate(clauses):
        if s <= span_a[0] < e:
            a_clause = idx
        if s <= span_b[0] < e:
            b_clause = idx
    return a_clause is not None and a_clause == b_clause


def detect_inconsistencies(
    events: Iterable[TimelineEvent],
    now: datetime | None = None,
    text: str | None = None,
    tolerance_days: int = 0,
) -> list[str]:
    """Detect temporal contradictions in events mentioned in the SAME clause.

    The previous version used a character-distance window which led to
    false positives when distinct events happened to sit close together
    in a single sentence-paragraph. Now we use sentence-level splitting
    (``.``, ``!``, ``?``, ``;``, newline) so events in distinct clauses
    are never compared.

    When ``text`` is None we fall back to a simple proximity check
    (within 60 chars) — useful when we only have the events without
    the source text.

    Returns a list of human-readable warning strings (FR). Empty list
    if nothing suspicious. Never raises.

    Heuristics
    ----------
    1. **Past relative + future absolute in the same clause**: classic
       case "hier on a soutenu la réunion du 12 mai 2026" when today is
       earlier than 12 mai 2026.
    2. **Two absolutes in the same clause** more than ``tolerance_days``
       apart: probable typo / semantic conflict.
    3. **Past relative + absolute in same clause that resolve to
       different dates**: e.g. "hier" mentioned alongside a date that
       isn't actually yesterday.
    """
    if events is None:
        return []
    cur = now or datetime.now(tz=timezone.utc)
    warnings: list[str] = []
    try:
        ev_list = [e for e in events if isinstance(e, TimelineEvent)]
        clauses = _split_clauses(text) if text else []

        def _co_located(a: TimelineEvent, b: TimelineEvent) -> bool:
            if clauses:
                return _same_clause(a.span, b.span, clauses)
            # Fallback proximity check when we don't have source text.
            return abs(a.span[0] - b.span[0]) <= 60

        for i, a in enumerate(ev_list):
            a_date = _resolved_date(a, cur)
            if a_date is None:
                continue
            for b in ev_list[i + 1:]:
                if not _co_located(a, b):
                    continue
                b_date = _resolved_date(b, cur)
                if b_date is None:
                    continue

                # Pair as (relative_event, absolute_event) when the
                # types differ, so messages are clearer downstream.
                rel = abs_ev = None
                rel_date = abs_date = None
                if a.kind == "relative" and b.kind == "absolute":
                    rel, abs_ev = a, b
                    rel_date, abs_date = a_date, b_date
                elif b.kind == "relative" and a.kind == "absolute":
                    rel, abs_ev = b, a
                    rel_date, abs_date = b_date, a_date

                if rel is not None and abs_ev is not None:
                    rel_in_past = rel_date < cur
                    abs_in_future = abs_date > cur
                    delta_days = abs((rel_date - abs_date).days)
                    if rel_in_past and abs_in_future and delta_days > tolerance_days:
                        warnings.append(format_warning(
                            issue="Incohérence temporelle",
                            details=(
                                f"« {rel.raw} » (résolu au {rel_date:%d/%m/%Y}) "
                                f"ne correspond pas à « {abs_ev.raw} » "
                                f"({abs_date:%d/%m/%Y}, dans le futur)"
                            ),
                            directive=(
                                f"Date système actuelle : {cur:%d/%m/%Y}. "
                                f"Demande à l'utilisateur laquelle des deux "
                                f"dates correspond réellement à l'événement "
                                f"({rel_date:%d/%m/%Y} ou {abs_date:%d/%m/%Y}). "
                                f"Ne paraphrase pas le message comme s'il "
                                f"était cohérent."
                            ),
                        ))
                    elif delta_days > tolerance_days:
                        warnings.append(format_warning(
                            issue="Discordance",
                            details=(
                                f"« {rel.raw} » ({rel_date:%d/%m/%Y}) et "
                                f"« {abs_ev.raw} » ({abs_date:%d/%m/%Y}) "
                                f"sont à {delta_days} jours d'écart"
                            ),
                            directive=(
                                f"Date système actuelle : {cur:%d/%m/%Y}. "
                                f"Vérifie auprès de l'utilisateur quelle "
                                f"date est exacte avant de poursuivre."
                            ),
                        ))

                # Two absolutes in the same clause and meaningfully apart.
                if a.kind == "absolute" and b.kind == "absolute":
                    delta_days = abs((a_date - b_date).days)
                    if delta_days > tolerance_days:
                        warnings.append(format_warning(
                            issue="Deux dates conflictuelles",
                            details=(
                                f"« {a.raw} » ({a_date:%d/%m/%Y}) et "
                                f"« {b.raw} » ({b_date:%d/%m/%Y}) "
                                f"mentionnées dans la même phrase"
                            ),
                            directive=(
                                "Demande à l'utilisateur de préciser "
                                "laquelle des deux dates est correcte."
                            ),
                        ))
    except Exception:
        log.warning("detect_inconsistencies crashed", exc_info=True)
        return []
    # Dedup while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for w in warnings:
        if w not in seen:
            deduped.append(w)
            seen.add(w)
    return deduped


def render_block(
    events: Iterable[TimelineEvent],
    config: TimelineConfig | None = None,
    now: datetime | None = None,
    text: str | None = None,
) -> str:
    """Render '[Chronologie]' block, or '' if no usable events.

    V4.0.2: appends ``detect_inconsistencies`` warnings at the end of
    the block so the LLM sees explicit temporal contradictions and can
    flag them in the response instead of paraphrasing blindly.

    ``text``: optional source text — when provided, inconsistencies
    are detected per-clause (more accurate). When omitted, falls back
    to a 60-char proximity heuristic.
    """
    cfg = config or TimelineConfig()
    if events is None:
        return ""

    try:
        ev_list = list(events) if not isinstance(events, list) else events

        rendered_lines: list[str] = []
        for ev in ev_list:
            if ev.confidence < cfg.render_min_confidence:
                continue
            if ev.kind == "vague" and not cfg.render_vague:
                continue
            icon = _ICONS.get(ev.kind, "•")
            label = ev.normalized or ev.raw
            rendered_lines.append(f"{icon} {label} ({ev.raw})")

        warnings = detect_inconsistencies(
            ev_list, now=(now or cfg.now), text=text,
        )

        if not rendered_lines and not warnings:
            return ""

        body_lines = list(rendered_lines)
        if warnings:
            if body_lines:
                body_lines.append("")
            body_lines.extend(warnings)

        block = "[Chronologie]\n" + "\n".join(body_lines)
        if len(block) > cfg.block_max_chars:
            block = block[: cfg.block_max_chars - 1].rstrip() + "…"
        return block
    except Exception:
        log.warning("timeline.render_block crashed", exc_info=True)
        return ""
