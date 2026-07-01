"""V4.3 — Tests for timeline module."""

from datetime import datetime, timezone

import pytest

from rune.cognition.timeline import (
    TimelineConfig,
    TimelineEvent,
    TimelineExtractor,
    render_block,
    _humanize_relative_seconds,
    _safe_make_iso,
)


@pytest.fixture
def extractor() -> TimelineExtractor:
    """Fixed reference date for reproducible tests."""
    return TimelineExtractor(
        TimelineConfig(now=datetime(2026, 5, 5, tzinfo=timezone.utc))
    )


# ════════════════════════════════════════════════════════════════════
# 1. Helpers
# ════════════════════════════════════════════════════════════════════


def test_humanize_today():
    assert _humanize_relative_seconds(0) == "aujourd'hui"


def test_humanize_yesterday():
    assert _humanize_relative_seconds(-86400) == "hier"


def test_humanize_tomorrow():
    assert _humanize_relative_seconds(86400) == "demain"


def test_humanize_n_days_ago():
    assert _humanize_relative_seconds(-3 * 86400) == "il y a 3 jours"


def test_humanize_n_months_future():
    assert "mois" in _humanize_relative_seconds(90 * 86400)


def test_safe_make_iso_valid():
    assert _safe_make_iso(2026, 5, 12) == "2026-05-12"


def test_safe_make_iso_invalid_month():
    assert _safe_make_iso(2026, 13, 1) is None


def test_safe_make_iso_invalid_day():
    assert _safe_make_iso(2026, 2, 30) is None


# ════════════════════════════════════════════════════════════════════
# 2. Extraction — relative
# ════════════════════════════════════════════════════════════════════


def test_extract_hier(extractor):
    evs = extractor.extract("Hier j'ai soutenu ma thèse")
    assert any(e.kind == "relative" and e.normalized == "hier" for e in evs)


def test_extract_demain(extractor):
    evs = extractor.extract("Demain on va commencer le projet")
    assert any(e.kind == "relative" and e.normalized == "demain" for e in evs)


def test_extract_avant_hier(extractor):
    evs = extractor.extract("Avant-hier il pleuvait")
    assert any("hier" in e.normalized for e in evs)


def test_extract_yesterday_en(extractor):
    evs = extractor.extract("Yesterday I had a meeting")
    assert any(e.kind == "relative" for e in evs)


def test_extract_il_y_a_3_jours(extractor):
    evs = extractor.extract("Il y a 3 jours j'ai vu un défaut")
    assert any("jour" in e.normalized for e in evs)


def test_extract_il_y_a_words(extractor):
    """Verbal numbers: 'il y a deux semaines'."""
    evs = extractor.extract("Il y a deux semaines on a discuté")
    assert any("semaine" in e.normalized for e in evs)


def test_extract_dans_2_mois(extractor):
    evs = extractor.extract("Dans 2 mois je passe en production")
    assert any("mois" in e.normalized and "dans" in e.normalized for e in evs)


def test_extract_in_2_months_en(extractor):
    evs = extractor.extract("In 2 months we ship")
    assert any(e.kind == "relative" for e in evs)


def test_extract_3_days_ago_en(extractor):
    evs = extractor.extract("3 days ago I noticed a defect")
    assert any(e.kind == "relative" for e in evs)


# ════════════════════════════════════════════════════════════════════
# 3. Extraction — absolute
# ════════════════════════════════════════════════════════════════════


def test_extract_iso_date(extractor):
    evs = extractor.extract("La réunion du 2025-03-14 a été annulée")
    iso_evs = [e for e in evs if e.normalized == "2025-03-14"]
    assert iso_evs and iso_evs[0].kind == "absolute"


def test_extract_fr_dmy(extractor):
    evs = extractor.extract("Le 12/05/2026 nous lançons")
    assert any(e.normalized == "2026-05-12" for e in evs)


def test_extract_fr_named_with_year(extractor):
    evs = extractor.extract("Le 12 mai 2026 j'ai un rendez-vous")
    assert any(e.normalized == "2026-05-12" for e in evs)


def test_extract_fr_named_without_year_uses_now(extractor):
    evs = extractor.extract("Le 12 mai j'ai un rendez-vous")
    # Should default to current year (2026 in fixture)
    assert any(e.normalized == "2026-05-12" for e in evs)


def test_extract_en_named(extractor):
    evs = extractor.extract("Meeting on May 12, 2026")
    assert any(e.normalized == "2026-05-12" for e in evs)


def test_extract_year_in_context(extractor):
    evs = extractor.extract("J'ai commencé en 2018")
    assert any(e.normalized == "2018" for e in evs)


def test_extract_does_not_capture_random_4_digit_numbers(extractor):
    """Bare 1800 (km) should not be captured as a year."""
    evs = extractor.extract("Le tableau contient 1800 lignes")
    # No year context (en/in/année) → no match
    assert not any(e.normalized == "1800" for e in evs)


def test_extract_time_of_day(extractor):
    evs = extractor.extract("Réunion à 14h30")
    assert any("14:30" in e.normalized for e in evs)


# ════════════════════════════════════════════════════════════════════
# 4. Extraction — duration
# ════════════════════════════════════════════════════════════════════


def test_extract_duration_pendant(extractor):
    evs = extractor.extract("J'ai travaillé pendant 6 mois sur ce projet")
    assert any(e.kind == "duration" for e in evs)


def test_extract_duration_depuis(extractor):
    evs = extractor.extract("Depuis 3 ans on essaie de résoudre")
    assert any(e.kind == "duration" for e in evs)


def test_extract_duration_for_en(extractor):
    evs = extractor.extract("For 3 months we worked on this")
    assert any(e.kind == "duration" for e in evs)


# ════════════════════════════════════════════════════════════════════
# 5. Extraction — ordinal
# ════════════════════════════════════════════════════════════════════


def test_extract_ordinal_premier_mai(extractor):
    evs = extractor.extract("Le premier mai je serai à Paris")
    assert any(e.kind == "ordinal" and e.normalized == "2026-05-01" for e in evs)


def test_extract_ordinal_with_year(extractor):
    evs = extractor.extract("Le premier mai 2025 c'était férié")
    assert any(e.kind == "ordinal" and e.normalized == "2025-05-01" for e in evs)


def test_extract_ordinal_en(extractor):
    evs = extractor.extract("On the first of May we celebrate")
    assert any(e.kind == "ordinal" for e in evs)


# ════════════════════════════════════════════════════════════════════
# 6. Extraction — vague
# ════════════════════════════════════════════════════════════════════


def test_extract_vague_recently(extractor):
    evs = extractor.extract("Récemment j'ai changé d'avis")
    assert any(e.kind == "vague" for e in evs)


def test_extract_vague_soon_en(extractor):
    evs = extractor.extract("Soon we will deploy")
    assert any(e.kind == "vague" for e in evs)


# ════════════════════════════════════════════════════════════════════
# 7. Multiple events + dedup
# ════════════════════════════════════════════════════════════════════


def test_extract_multiple(extractor):
    evs = extractor.extract(
        "Hier on a parlé du 12 mai 2026 et dans 2 mois on lance"
    )
    kinds = {e.kind for e in evs}
    assert "relative" in kinds
    assert "absolute" in kinds


def test_extract_max_events_capped(extractor):
    text = " ".join([f"il y a {i} jours" for i in range(1, 20)])
    extractor.config.max_events = 5
    evs = extractor.extract(text)
    assert len(evs) <= 5


def test_extract_dedup_same_kind_normalized(extractor):
    """Two mentions of 'hier' → one TimelineEvent."""
    evs = extractor.extract("Hier hier hier on a parlé")
    rel_hier = [e for e in evs if e.normalized == "hier"]
    assert len(rel_hier) == 1


# ════════════════════════════════════════════════════════════════════
# 8. Empty / defensive
# ════════════════════════════════════════════════════════════════════


def test_extract_empty_text(extractor):
    assert extractor.extract("") == []


def test_extract_none(extractor):
    assert extractor.extract(None) == []  # type: ignore[arg-type]


def test_extract_pure_garbage_returns_empty(extractor):
    evs = extractor.extract("blah blah blah random words")
    assert evs == []


def test_extract_no_crash_on_unicode(extractor):
    evs = extractor.extract("🚀 demain on lance le projet 🎉")
    assert any(e.normalized == "demain" for e in evs)


# ════════════════════════════════════════════════════════════════════
# 9. Render block
# ════════════════════════════════════════════════════════════════════


def test_render_empty_returns_empty_string():
    assert render_block([]) == ""


def test_render_filters_low_confidence():
    cfg = TimelineConfig(render_min_confidence=0.5)
    evs = [TimelineEvent(raw="x", kind="relative", normalized="x", confidence=0.4)]
    assert render_block(evs, cfg) == ""


def test_render_includes_high_confidence():
    cfg = TimelineConfig(render_min_confidence=0.3)
    evs = [TimelineEvent(raw="hier", kind="relative", normalized="hier", confidence=0.85)]
    block = render_block(evs, cfg)
    assert "hier" in block
    assert "[Chronologie]" in block


def test_render_excludes_vague_by_default():
    cfg = TimelineConfig(render_vague=False)
    evs = [TimelineEvent(raw="bientôt", kind="vague", normalized="bientôt", confidence=0.4)]
    assert render_block(evs, cfg) == ""


def test_render_includes_vague_when_enabled():
    cfg = TimelineConfig(render_vague=True, render_min_confidence=0.3)
    evs = [TimelineEvent(raw="bientôt", kind="vague", normalized="bientôt", confidence=0.4)]
    block = render_block(evs, cfg)
    assert "bientôt" in block


def test_render_truncates_long_blocks():
    cfg = TimelineConfig(block_max_chars=80, render_min_confidence=0.3)
    evs = [
        TimelineEvent(
            raw=f"event_{i}",
            kind="absolute",
            normalized=f"event_{i}_normalized_with_very_long_text",
            confidence=0.9,
        )
        for i in range(10)
    ]
    block = render_block(evs, cfg)
    assert len(block) <= 80
    assert block.endswith("…")


def test_render_uses_emojis_per_kind():
    cfg = TimelineConfig(render_min_confidence=0.3, render_vague=True)
    evs = [
        TimelineEvent(raw="2026-05-12", kind="absolute", normalized="2026-05-12", confidence=0.9),
        TimelineEvent(raw="hier", kind="relative", normalized="hier", confidence=0.8),
        TimelineEvent(raw="6 mois", kind="duration", normalized="P6M", confidence=0.7),
        TimelineEvent(raw="premier mai", kind="ordinal", normalized="2026-05-01", confidence=0.7),
        TimelineEvent(raw="bientôt", kind="vague", normalized="bientôt", confidence=0.4),
    ]
    block = render_block(evs, cfg)
    assert "📅" in block
    assert "⏱" in block
    assert "⏳" in block
    assert "🔢" in block
    assert "❓" in block


# ════════════════════════════════════════════════════════════════════
# 10. End-to-end realism
# ════════════════════════════════════════════════════════════════════


def test_realistic_industrial_message(extractor):
    """Mika-style technical message — should not over-trigger."""
    text = (
        "On a observé hier une fissure de fatigue sur la pièce. "
        "L'analyse spectroscopique du 12/03/2025 confirme la corrosion. "
        "Maintenance prévue dans 3 jours."
    )
    evs = extractor.extract(text)
    kinds = {e.kind for e in evs}
    assert "relative" in kinds  # hier
    assert "absolute" in kinds  # 12/03/2025
    # "dans 3 jours" → relative
    assert any("3 jour" in e.normalized for e in evs)


def test_event_to_dict():
    ev = TimelineEvent(raw="hier", kind="relative", normalized="hier", confidence=0.85)
    d = ev.to_dict()
    assert d["kind"] == "relative"
    assert d["normalized"] == "hier"
    assert d["confidence"] == 0.85
