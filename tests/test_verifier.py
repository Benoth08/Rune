"""Vérificateur non-code : longueur, esquives, sources, structure."""

from rune.agentic.verifier import check_deliverable


def test_evasion_is_rejected():
    v = check_deliverable(
        "Pour la synthèse, vous pouvez supposer que le travail a été fait.",
        kind="redaction", level=1)
    assert not v.ok
    assert any("esquive" in r for r in v.reasons)


def test_too_short_is_rejected():
    v = check_deliverable("Trois mots seulement.", kind="redaction", level=2)
    assert not v.ok
    assert any("court" in r for r in v.reasons)


def test_research_needs_sources():
    txt = " ".join(["analyse"] * 100)            # long mais aucune URL
    v = check_deliverable(txt, kind="recherche", level=1)
    assert not v.ok
    assert any("source" in r for r in v.reasons)

    txt2 = txt + " source : https://example.com/rapport"
    v2 = check_deliverable(txt2, kind="recherche", level=1)
    assert v2.ok


def test_good_redaction_passes():
    txt = " ".join(["phrase"] * 200)
    v = check_deliverable(txt, kind="redaction", level=2)
    assert v.ok and v.reasons == []


def test_short_answer_ok_at_level0():
    v = check_deliverable(
        "La capitale de la France est Paris, une ville d'environ deux "
        "millions d'habitants au centre du pays.", kind="general", level=0)
    assert v.ok
