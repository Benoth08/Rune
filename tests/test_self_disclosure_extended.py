"""Tests V5.5.2 — Self-disclosure étendu (5 patterns).

Couvre les 5 fallbacks regex case-insensitive ajoutés pour pallier
les limitations de GLiNER sur les noms communs en minuscules :

1. Prénom (_SELF_INTRO_RE)        — déjà testé dans test_self_intro_fallback.py
2. Rôle / métier (_SELF_ROLE_RE)
3. Lieu de résidence (_SELF_LOCATION_RE)
4. Employeur (_SELF_EMPLOYER_RE)
5. Âge (_SELF_AGE_RE)

Chaque pattern est testé en isolation puis l'intégration dans
_extract_entities est validée avec mocks (faux positifs filtrés par
blocklists).
"""

from __future__ import annotations

import importlib.util
import pytest

# Tous les tests de ce fichier nécessitent torch (chargé par
# rune.cognition.encoding). Skip global en sandbox sans torch.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="encoding.py requires torch",
)


# ── Fixture commune ────────────────────────────────────────────────────


class FakeGLiNER:
    """GLiNER mock : retourne ce qu'on lui dit, ou rien par défaut."""

    def __init__(self, return_value: list[dict] | None = None):
        self._return = return_value or []

    def extract(self, text: str) -> list[dict]:
        return list(self._return)


class FakeEncodingPhase:
    """Wrapper minimal qui expose _extract_entities.

    V5.5.x — On construit une vraie instance EncodingPhase via __new__
    pour pouvoir appeler les méthodes liées (_apply_self_intro, etc.)
    sans déclencher le __init__ lourd (modèles, dépendances).
    """
    def __new__(cls, gliner=None):
        from rune.cognition.encoding import EncodingPhase
        instance = EncodingPhase.__new__(EncodingPhase)
        instance.entity_extractor = gliner
        return instance


# ═══════════════════════════════════════════════════════════════════════
# Pattern 2 — Rôle / Métier (_SELF_ROLE_RE)
# ═══════════════════════════════════════════════════════════════════════


class TestSelfRoleRegex:

    def test_single_word_role(self):
        from rune.cognition.encoding import _SELF_ROLE_RE
        assert _SELF_ROLE_RE.findall("je suis chimiométricien") == ["chimiométricien"]
        assert _SELF_ROLE_RE.findall("je suis développeur") == ["développeur"]

    def test_multi_word_role(self):
        """V5.5.2 : capture les rôles avec espace ("data scientist")."""
        from rune.cognition.encoding import _SELF_ROLE_RE
        assert _SELF_ROLE_RE.findall("je suis data scientist") == ["data scientist"]
        assert _SELF_ROLE_RE.findall("je suis ingénieur R&D") == ["ingénieur R&D"]

    def test_je_travaille_comme(self):
        from rune.cognition.encoding import _SELF_ROLE_RE
        assert _SELF_ROLE_RE.findall("je travaille comme analyste") == ["analyste"]
        assert _SELF_ROLE_RE.findall("je travaille en tant que consultant") == ["consultant"]

    def test_mon_metier(self):
        from rune.cognition.encoding import _SELF_ROLE_RE
        assert _SELF_ROLE_RE.findall("mon métier est ingénieur") == ["ingénieur"]
        assert _SELF_ROLE_RE.findall("mon métier c'est designer") == ["designer"]

    def test_english_role(self):
        from rune.cognition.encoding import _SELF_ROLE_RE
        assert _SELF_ROLE_RE.findall("I'm a developer") == ["developer"]
        assert _SELF_ROLE_RE.findall("I work as a teacher") == ["teacher"]

    def test_no_match_on_location(self):
        """V5.5.2 fix : 'je suis à Aix' ne doit PAS matcher comme rôle."""
        from rune.cognition.encoding import _SELF_ROLE_RE
        assert _SELF_ROLE_RE.findall("je suis à Aix") == []
        assert _SELF_ROLE_RE.findall("je suis en Belgique") == []
        assert _SELF_ROLE_RE.findall("je suis chez Framatome") == []
        # "je suis né" ne doit pas matcher (état civil, pas métier)
        assert _SELF_ROLE_RE.findall("je suis née en 1982") == []


class TestSelfRoleBlocklist:
    """Vérifie que les états passagers sont filtrés."""

    def test_state_filtered(self):
        """'je suis fatigué' est capturé par le regex mais filtré
        par _ROLE_BLOCKLIST dans _apply_self_role."""
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("je suis fatigué")
        roles = [e for e in entities if e.get("label") == "role"]
        assert roles == []  # bloqué

    def test_real_role_passes(self):
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("je suis chimiométricien")
        roles = [e for e in entities if e.get("label") == "role"]
        assert len(roles) == 1
        assert roles[0]["text"] == "chimiométricien"

    def test_emotional_states_all_blocked(self):
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        for state in ["content", "triste", "désolé", "stressé", "ravi"]:
            entities = phase._extract_entities(f"je suis {state}")
            roles = [e for e in entities if e.get("label") == "role"]
            assert roles == [], f"État émotionnel '{state}' devrait être bloqué"


# ═══════════════════════════════════════════════════════════════════════
# Pattern 3 — Lieu (_SELF_LOCATION_RE)
# ═══════════════════════════════════════════════════════════════════════


class TestSelfLocationRegex:

    def test_jhabite_simple(self):
        from rune.cognition.encoding import _SELF_LOCATION_RE
        assert _SELF_LOCATION_RE.findall("j'habite à Paris") == ["Paris"]

    def test_jhabite_lowercase(self):
        """Cas du bug original : ville en minuscule."""
        from rune.cognition.encoding import _SELF_LOCATION_RE
        assert _SELF_LOCATION_RE.findall("j'habite à aix-en-provence") == ["aix-en-provence"]

    def test_je_vis(self):
        from rune.cognition.encoding import _SELF_LOCATION_RE
        assert _SELF_LOCATION_RE.findall("je vis à Marseille") == ["Marseille"]
        assert _SELF_LOCATION_RE.findall("je vis en France") == ["France"]

    def test_je_suis_basee(self):
        from rune.cognition.encoding import _SELF_LOCATION_RE
        assert _SELF_LOCATION_RE.findall("je suis basée à Lyon") == ["Lyon"]
        assert _SELF_LOCATION_RE.findall("je suis installé en Espagne") == ["Espagne"]

    def test_english_location(self):
        from rune.cognition.encoding import _SELF_LOCATION_RE
        assert _SELF_LOCATION_RE.findall("I live in Berlin") == ["Berlin"]
        assert _SELF_LOCATION_RE.findall("I'm based in Tokyo") == ["Tokyo"]

    def test_no_match_on_movement(self):
        """'je vais à Paris demain' = déplacement, pas résidence."""
        from rune.cognition.encoding import _SELF_LOCATION_RE
        assert _SELF_LOCATION_RE.findall("je vais à Paris demain") == []

    def test_no_match_on_origin(self):
        """'je viens de Lyon' = origine, pas résidence actuelle."""
        from rune.cognition.encoding import _SELF_LOCATION_RE
        assert _SELF_LOCATION_RE.findall("je viens de Lyon") == []

    def test_third_person_blocked(self):
        from rune.cognition.encoding import _SELF_LOCATION_RE
        assert _SELF_LOCATION_RE.findall("il habite à Paris") == []


# ═══════════════════════════════════════════════════════════════════════
# Pattern 4 — Employeur (_SELF_EMPLOYER_RE)
# ═══════════════════════════════════════════════════════════════════════


class TestSelfEmployerRegex:

    def test_je_travaille_chez(self):
        from rune.cognition.encoding import _SELF_EMPLOYER_RE
        assert _SELF_EMPLOYER_RE.findall("je travaille chez Framatome") == ["Framatome"]

    def test_lowercase_employer(self):
        from rune.cognition.encoding import _SELF_EMPLOYER_RE
        assert _SELF_EMPLOYER_RE.findall("je travaille chez framatome") == ["framatome"]

    def test_multi_word_employer(self):
        """V5.5.2 : capture les noms d'entreprise multi-mots."""
        from rune.cognition.encoding import _SELF_EMPLOYER_RE
        assert _SELF_EMPLOYER_RE.findall("je bosse chez TOPNIR Systems") == ["TOPNIR Systems"]
        assert _SELF_EMPLOYER_RE.findall("I work at BNP Paribas") == ["BNP Paribas"]

    def test_je_suis_chez(self):
        """V5.5.2 : 'je suis chez X' = employeur."""
        from rune.cognition.encoding import _SELF_EMPLOYER_RE
        assert _SELF_EMPLOYER_RE.findall("je suis chez Thalès") == ["Thalès"]
        assert _SELF_EMPLOYER_RE.findall("je suis chez thalès") == ["thalès"]

    def test_ma_boite(self):
        from rune.cognition.encoding import _SELF_EMPLOYER_RE
        assert _SELF_EMPLOYER_RE.findall("ma boîte s'appelle Acme Corp") == ["Acme Corp"]

    def test_english_employer(self):
        from rune.cognition.encoding import _SELF_EMPLOYER_RE
        assert _SELF_EMPLOYER_RE.findall("I work at Google") == ["Google"]
        assert _SELF_EMPLOYER_RE.findall("my company is Microsoft") == ["Microsoft"]


class TestSelfEmployerBlocklist:
    """Les 'chez moi', 'chez mes parents' sont capturés mais filtrés."""

    def test_chez_moi_blocked(self):
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("je travaille chez moi en télétravail")
        orgs = [e for e in entities if e.get("label") == "organization"]
        # "moi" est dans _EMPLOYER_BLOCKLIST → bloqué
        assert orgs == []

    def test_chez_mes_parents_blocked(self):
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("je bosse chez mes parents cet été")
        orgs = [e for e in entities if e.get("label") == "organization"]
        assert orgs == []

    def test_real_employer_passes(self):
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("je travaille chez Framatome")
        orgs = [e for e in entities if e.get("label") == "organization"]
        assert len(orgs) == 1
        assert orgs[0]["text"] == "Framatome"


# ═══════════════════════════════════════════════════════════════════════
# Pattern 5 — Âge (_SELF_AGE_RE)
# ═══════════════════════════════════════════════════════════════════════


class TestSelfAgeRegex:

    def test_jai_n_ans(self):
        from rune.cognition.encoding import _SELF_AGE_RE
        assert _SELF_AGE_RE.findall("j'ai 36 ans") == ["36"]
        assert _SELF_AGE_RE.findall("J'AI 42 ANS") == ["42"]

    def test_english_age(self):
        from rune.cognition.encoding import _SELF_AGE_RE
        assert _SELF_AGE_RE.findall("I'm 25 years old") == ["25"]
        assert _SELF_AGE_RE.findall("I am 60 year old") == ["60"]

    def test_no_match_without_ans(self):
        from rune.cognition.encoding import _SELF_AGE_RE
        # "j'ai 1000 fichiers" — pas "ans" derrière, ne matche pas
        assert _SELF_AGE_RE.findall("j'ai 1000 fichiers") == []
        assert _SELF_AGE_RE.findall("j'ai 3 enfants") == []


class TestSelfAgeIntegration:
    """L'âge va dans le KG avec label 'age', value numérique string."""

    def test_age_captured(self):
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        entities = phase._extract_entities("j'ai 36 ans")
        ages = [e for e in entities if e.get("label") == "age"]
        assert len(ages) == 1
        assert ages[0]["text"] == "36"

    def test_age_out_of_range_filtered(self):
        """Bornes 1-120, au-delà filtré."""
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        # Test avec un nombre dans un contexte avec "ans" mais > 120
        # Note : le regex {1,3} capture max 999, donc 200 passe le regex
        # mais le filtre 1-120 dans _apply_self_age le rejette.
        entities = phase._extract_entities("j'ai 200 ans bientôt")  # absurde
        ages = [e for e in entities if e.get("label") == "age"]
        assert ages == []


# ═══════════════════════════════════════════════════════════════════════
# Intégration globale : phrase combinant plusieurs patterns
# ═══════════════════════════════════════════════════════════════════════


class TestMultiPatternIntegration:

    def test_intro_complete(self):
        """Une phrase de présentation typique combinant plusieurs patterns."""
        phase = FakeEncodingPhase(gliner=FakeGLiNER())
        text = (
            "bonjour, je m'appelle mika, j'ai 36 ans, "
            "je suis chimiométricien et je travaille chez framatome. "
            "j'habite à aix-en-provence."
        )
        entities = phase._extract_entities(text)
        labels_to_values: dict[str, list[str]] = {}
        for e in entities:
            labels_to_values.setdefault(e["label"], []).append(e["text"])

        # Prénom capturé et normalisé
        assert "Mika" in labels_to_values.get("person", [])
        # Âge capturé
        assert "36" in labels_to_values.get("age", [])
        # Rôle capturé en lowercase
        assert "chimiométricien" in labels_to_values.get("role", [])
        # Employeur capturé et normalisé
        assert "Framatome" in labels_to_values.get("organization", [])
        # Lieu capturé et normalisé
        assert "Aix-En-Provence" in labels_to_values.get("location", []) \
            or "Aix-en-Provence" in labels_to_values.get("location", [])

    def test_no_duplicates_when_gliner_finds_same(self):
        """Si GLiNER trouve déjà, pas de doublon."""
        gliner_with_mika = FakeGLiNER([
            {"text": "Mika", "label": "person", "score": 0.7},
        ])
        phase = FakeEncodingPhase(gliner=gliner_with_mika)
        entities = phase._extract_entities("je m'appelle mika")
        persons = [e for e in entities if e.get("label") == "person"]
        # Une seule occurrence de Mika
        assert len(persons) == 1

    def test_gliner_crash_fallback_still_runs(self):
        """GLiNER crash → les fallbacks regex tournent toujours."""
        class CrashingGLiNER:
            def extract(self, text):
                raise RuntimeError("GLiNER down")
        phase = FakeEncodingPhase(gliner=CrashingGLiNER())
        entities = phase._extract_entities(
            "je m'appelle cédric, j'habite à lyon, j'ai 30 ans"
        )
        labels = {e["label"] for e in entities}
        # Au moins person + location + age extraits malgré GLiNER down
        assert "person" in labels
        assert "location" in labels
        assert "age" in labels
