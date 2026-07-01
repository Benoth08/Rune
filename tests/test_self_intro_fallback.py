"""Tests V5.5.1 — Self-introduction fallback (case-insensitive prénom).

Couvre :
- Le regex _SELF_INTRO_RE matche les patterns FR/EN courants
- _extract_entities applique le fallback correctement
- La normalisation Title Case fonctionne (cédric → Cédric)
- Pas de doublon si GLiNER a déjà trouvé
- Robustesse : ponctuation, accents, prénoms composés
"""

from __future__ import annotations

import importlib.util
import pytest

# Tous les tests de ce fichier nécessitent torch (chargé par
# lythea.cognition.encoding). Skip global en sandbox sans torch.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="encoding.py requires torch",
)


# ── Tests directs sur le regex ─────────────────────────────────────────


class TestSelfIntroRegex:

    def test_french_lowercase(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("bonjour je m'appelle cédric")
        assert matches == ["cédric"]

    def test_french_title_case(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("Bonjour je m'appelle Michaël")
        assert matches == ["Michaël"]

    def test_french_uppercase(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("BONJOUR JE M'APPELLE PIERRE")
        # Le regex est case-insensitive mais préserve la casse du nom
        assert matches == ["PIERRE"]

    def test_french_compound_name(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("je m'appelle jean-pierre")
        assert matches == ["jean-pierre"]

    def test_moi_c_est(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("salut moi c'est marie")
        assert matches == ["marie"]

    def test_mon_nom_est(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("mon nom est Dupont")
        assert matches == ["Dupont"]

    def test_mon_prenom_est(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("mon prénom est sophie")
        assert matches == ["sophie"]

    def test_appelle_moi(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("appelle-moi alex")
        assert matches == ["alex"]

    def test_english_my_name_is(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("Hello, my name is John")
        assert matches == ["John"]

    def test_english_call_me(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("Just call me Sarah please")
        assert matches == ["Sarah"]

    def test_english_im(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("Hi I'm Mike")
        assert matches == ["Mike"]

    def test_with_punctuation(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("je m'appelle cédric, ravi.")
        assert matches == ["cédric"]

    def test_apostrophe_typographic(self):
        """Apostrophe courbe ’ acceptée comme la droite."""
        from rune.cognition.encoding import _SELF_INTRO_RE
        matches = _SELF_INTRO_RE.findall("je m’appelle léa")
        assert matches == ["léa"]

    def test_no_intro_pattern(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        # Pas de pattern d'auto-intro → pas de match
        assert _SELF_INTRO_RE.findall("le chat dort") == []
        assert _SELF_INTRO_RE.findall("merci beaucoup") == []
        assert _SELF_INTRO_RE.findall("bonjour comment ça va") == []

    def test_multiple_intros(self):
        """Plusieurs présentations dans le même message → toutes capturées."""
        from rune.cognition.encoding import _SELF_INTRO_RE
        text = "je m'appelle alice et mon ami s'appelle bob"
        matches = _SELF_INTRO_RE.findall(text)
        # On capture au moins alice (et probablement bob via "s'appelle"
        # qui n'est pas exactement notre pattern). Le test est tolérant.
        assert "alice" in matches


# ── Tests sur _extract_entities (intégration) ──────────────────────────


class FakeGLiNER:
    """Mock entity_extractor.

    Par défaut, simule le comportement réel : trouve les noms en
    Title Case, rate les noms en minuscule.
    """

    def __init__(self, return_value: list[dict] | None = None):
        self._fixed_return = return_value

    def extract(self, text: str) -> list[dict]:
        if self._fixed_return is not None:
            return list(self._fixed_return)
        # Mock du vrai comportement GLiNER
        import re
        results = []
        # GLiNER trouve les mots Title Case ≥ 4 lettres
        for m in re.finditer(r"\b[A-Z][a-zàâäéèêëïîôöùûüÿç]{3,}\b", text):
            results.append({
                "text": m.group(0),
                "label": "person",
                "score": 0.7,
            })
        return results


class FakeEncodingPhase:
    """Mini-wrapper qui expose _extract_entities.

    V5.5.x — Utilise __new__ sur la vraie classe EncodingPhase pour
    avoir accès aux méthodes liées (_apply_self_intro, etc.) sans
    déclencher le __init__ lourd.
    """
    def __new__(cls, gliner=None):
        from rune.cognition.encoding import EncodingPhase
        instance = EncodingPhase.__new__(EncodingPhase)
        instance.entity_extractor = gliner
        return instance


class TestExtractEntitiesIntegration:

    def test_lowercase_name_captured(self):
        """Cas du bug : 'je m'appelle cédric' → entité créée."""
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("bonjour je m'appelle cédric")
        names = [e["text"] for e in entities]
        assert "Cédric" in names  # normalisé Title Case
        # Vérifie le label
        cedric = next(e for e in entities if e["text"] == "Cédric")
        assert cedric["label"] == "person"
        assert cedric["score"] >= 0.9  # haute confiance pattern

    def test_uppercase_name_still_works(self):
        """'Cédric' avec majuscule → GLiNER trouve, pas de doublon."""
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("Bonjour je m'appelle Cédric")
        cedric_entries = [e for e in entities if e["text"].lower() == "cédric"]
        # Pas de duplication : 1 seule entrée
        assert len(cedric_entries) == 1

    def test_michael_lowercase(self):
        """Test avec accent + minuscule."""
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("salut je m'appelle michaël")
        names = [e["text"] for e in entities]
        assert "Michaël" in names

    def test_normalization_compound_name(self):
        """jean-pierre → Jean-Pierre."""
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("je m'appelle jean-pierre")
        names = [e["text"] for e in entities]
        assert "Jean-Pierre" in names

    def test_no_self_intro_no_change(self):
        """Sans pattern intro, le résultat est inchangé."""
        gliner_finds_marie = FakeGLiNER([
            {"text": "Marie", "label": "person", "score": 0.8},
        ])
        phase = FakeEncodingPhase(gliner=gliner_finds_marie)
        entities = phase._extract_entities("Marie est venue hier")
        assert len(entities) == 1
        assert entities[0]["text"] == "Marie"

    def test_no_extractor_returns_empty(self):
        phase = FakeEncodingPhase(gliner=None)
        assert phase._extract_entities("je m'appelle X") == []

    def test_gliner_crash_fallback_still_works(self):
        """Si GLiNER plante, le fallback regex tourne quand même."""
        class CrashingGLiNER:
            def extract(self, text):
                raise RuntimeError("GLiNER down")
        phase = FakeEncodingPhase(gliner=CrashingGLiNER())
        entities = phase._extract_entities("je m'appelle cédric")
        # Le fallback regex doit avoir tourné même si GLiNER a crashé
        names = [e["text"] for e in entities]
        assert "Cédric" in names

    def test_noise_filter_still_applies(self):
        """Les entités noise (ex pronoms) sont toujours filtrées."""
        # GLiNER aurait extrait "je" par erreur
        gliner_with_noise = FakeGLiNER([
            {"text": "je", "label": "person", "score": 0.5},
        ])
        phase = FakeEncodingPhase(gliner=gliner_with_noise)
        entities = phase._extract_entities("je m'appelle cédric")
        # "je" filtré par ENTITY_NOISE, "Cédric" ajouté par fallback
        names = [e["text"] for e in entities]
        assert "je" not in names
        assert "Cédric" in names

    def test_short_name_rejected(self):
        """Nom trop court (< 2 chars) → rejeté."""
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        # Le regex demande min 2 caractères en suffixe, et nous rejetons
        # tout < 2 chars
        entities = phase._extract_entities("je m'appelle X")
        names = [e["text"] for e in entities]
        # X seul ne devrait pas passer le filtre length
        assert "X" not in names


# ── Tests bonus : robustesse français ──────────────────────────────────


class TestFrenchEdgeCases:

    def test_no_false_positive_on_question(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        # Question "comment tu t'appelles" — pas une auto-intro
        matches = _SELF_INTRO_RE.findall("comment tu t'appelles ?")
        assert matches == []

    def test_no_false_positive_on_third_person(self):
        from rune.cognition.encoding import _SELF_INTRO_RE
        # "il s'appelle" — n'est pas notre pattern (on cible "je m'appelle")
        matches = _SELF_INTRO_RE.findall("il s'appelle Pierre")
        assert "Pierre" not in matches

    def test_capitalize_preserves_accents(self):
        """capitalize() en Python préserve les accents."""
        assert "cédric".capitalize() == "Cédric"
        assert "élise".capitalize() == "Élise"
