"""Tests V5.5.3 à V5.5.8 — Anti-pollution mémoire et greeting strip.

Couvre les fixes consécutifs ajoutés en mai 2026 :

V5.5.3 — Pattern année de naissance + ENTITY_NOISE étendu
V5.5.4 — Endpoint /api/memory/cleanup_noise (KG)
V5.5.5 — Endpoint /api/memory/cleanup_chroma (Chroma)
V5.5.6 — SYSTEM_PROMPT renforcé anti-salutation/anti-relance
V5.5.7 — Strip salutation post-streaming (final_text)
V5.5.8 — Strip salutation pendant streaming (probe)

Les tests qui dépendent de torch (extracteur GLiNER) sont marqués
``pytest.mark.skipif`` pour tourner sur RunPod et passer en CI sandbox.
"""

from __future__ import annotations

import importlib.util
import re
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


# ═══════════════════════════════════════════════════════════════════════
# V5.5.3 — Pattern année de naissance (_SELF_BIRTH_YEAR_RE)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_TORCH, reason="encoding.py requires torch")
class TestBirthYearRegex:

    def test_je_suis_ne_en(self):
        from rune.cognition.encoding import _SELF_BIRTH_YEAR_RE
        assert _SELF_BIRTH_YEAR_RE.findall("Je suis né en 1985") == ["1985"]

    def test_feminine(self):
        from rune.cognition.encoding import _SELF_BIRTH_YEAR_RE
        assert _SELF_BIRTH_YEAR_RE.findall("Je suis née en 1990") == ["1990"]

    def test_short_form(self):
        from rune.cognition.encoding import _SELF_BIRTH_YEAR_RE
        assert _SELF_BIRTH_YEAR_RE.findall("née en 1942") == ["1942"]

    def test_english_born_in(self):
        from rune.cognition.encoding import _SELF_BIRTH_YEAR_RE
        assert _SELF_BIRTH_YEAR_RE.findall("I was born in 1985") == ["1985"]
        assert _SELF_BIRTH_YEAR_RE.findall("born in 1942") == ["1942"]

    def test_no_match_on_other_date(self):
        from rune.cognition.encoding import _SELF_BIRTH_YEAR_RE
        # Année dans un autre contexte → pas une date de naissance
        assert _SELF_BIRTH_YEAR_RE.findall(
            "c'est de 1985 que date ce film"
        ) == []
        assert _SELF_BIRTH_YEAR_RE.findall(
            "le projet a commencé en 2020"
        ) == []


@pytest.mark.skipif(not HAS_TORCH, reason="encoding.py requires torch")
class TestBirthYearExtraction:
    """L'extraction produit année:YYYY + age dérivé, et déduplique."""

    @staticmethod
    def _make_phase(gliner):
        """Construit un EncodingPhase minimal sans appeler __init__
        (qui demande modèles + dépendances lourdes).

        On utilise __new__ pour avoir une vraie instance EncodingPhase
        avec toutes ses méthodes liées (_apply_self_intro, etc.) puis
        on injecte manuellement le minimum d'attributs nécessaires.
        """
        from rune.cognition.encoding import EncodingPhase
        phase = EncodingPhase.__new__(EncodingPhase)
        phase.entity_extractor = gliner
        return phase

    def test_creates_year_and_age(self):
        class FakeGLiNER:
            def extract(self, text):
                return []

        phase = self._make_phase(FakeGLiNER())

        from datetime import datetime
        current_year = datetime.now().year
        expected_age = current_year - 1985

        result = phase._extract_entities("Je suis né en 1985")
        years = [e for e in result if e["label"] == "date"]
        ages = [e for e in result if e["label"] == "age"]
        assert any(e["text"] == "année:1985" for e in years)
        assert any(e["text"] == str(expected_age) for e in ages)

    def test_dedup_raw_year(self):
        """V5.5.4 — si GLiNER extrait aussi '1985' brut, on supprime."""
        class FakeGLiNER:
            def extract(self, text):
                return [{"text": "1985", "label": "date", "score": 0.5}]

        phase = self._make_phase(FakeGLiNER())
        result = phase._extract_entities("Je suis né en 1985")
        years = [e["text"] for e in result if e["label"] == "date"]
        # On doit avoir "année:1985" mais PAS "1985" brut
        assert "année:1985" in years
        assert "1985" not in years

    def test_sanity_future_year_rejected(self):
        class FakeGLiNER:
            def extract(self, text):
                return []

        phase = self._make_phase(FakeGLiNER())
        # 2099 = futur, ne doit pas être accepté comme année de naissance
        result = phase._extract_entities("Je suis né en 2099")
        years = [e for e in result if e["label"] == "date"]
        assert years == []


# ═══════════════════════════════════════════════════════════════════════
# V5.5.3 — ENTITY_NOISE étendu + normalisation apostrophes
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_TORCH, reason="encoding.py requires torch")
class TestEntityNoiseExtended:

    def test_je_suis_in_noise(self):
        from rune.cognition.encoding import ENTITY_NOISE
        assert "je suis" in ENTITY_NOISE

    def test_jai_in_noise(self):
        from rune.cognition.encoding import ENTITY_NOISE
        assert "j'ai" in ENTITY_NOISE

    def test_i_am_in_noise(self):
        from rune.cognition.encoding import ENTITY_NOISE
        assert "i am" in ENTITY_NOISE
        assert "i'm" in ENTITY_NOISE

    def test_pronouns_in_noise(self):
        from rune.cognition.encoding import ENTITY_NOISE
        for w in ["moi", "toi", "lui", "soi", "eux"]:
            assert w in ENTITY_NOISE

    def test_legitimate_word_not_in_noise(self):
        from rune.cognition.encoding import ENTITY_NOISE
        for w in ["cédric", "framatome", "aix-en-provence", "chimiométricien"]:
            assert w not in ENTITY_NOISE


@pytest.mark.skipif(not HAS_TORCH, reason="encoding.py requires torch")
class TestApostropheNormalization:
    """V5.5.3 — Les apostrophes typographiques sont normalisées
    avant comparaison avec ENTITY_NOISE."""

    def test_typographic_apostrophe_filtered(self):
        from rune.cognition.encoding import EncodingPhase

        class FakeGLiNER:
            def extract(self, text):
                # GLiNER extrait "J'ai" avec apostrophe courbe
                return [{"text": "J\u2019ai", "label": "person", "score": 0.5}]

        # V5.5.x — Construit un vrai EncodingPhase sans __init__ pour
        # avoir accès aux méthodes liées (_apply_self_intro, etc.).
        # Voir TestBirthYearExtraction._make_phase pour le détail.
        phase = EncodingPhase.__new__(EncodingPhase)
        phase.entity_extractor = FakeGLiNER()

        result = phase._extract_entities("J\u2019ai un projet")
        # "J'ai" normalisé doit être filtré comme noise même avec apostrophe ’
        texts = [e["text"] for e in result]
        assert "J\u2019ai" not in texts


# ═══════════════════════════════════════════════════════════════════════
# V5.5.4 + V5.5.5 — Endpoints de cleanup (signatures, dry-run)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_TORCH, reason="routes.py imports torch via model")
@pytest.mark.skipif(not HAS_FASTAPI, reason="needs fastapi")
class TestCleanupEndpointsRegistered:
    """Vérifie que les endpoints sont bien déclarés dans le router."""

    def test_cleanup_noise_registered(self):
        from rune.server.routes import router
        paths = [r.path for r in router.routes]
        assert "/api/memory/cleanup_noise" in paths

    def test_cleanup_chroma_registered(self):
        from rune.server.routes import router
        paths = [r.path for r in router.routes]
        assert "/api/memory/cleanup_chroma" in paths

    def test_memory_health_registered(self):
        from rune.server.routes import router
        paths = [r.path for r in router.routes]
        assert "/api/memory/health" in paths


# ═══════════════════════════════════════════════════════════════════════
# V5.5.6 — SYSTEM_PROMPT renforcé
# ═══════════════════════════════════════════════════════════════════════


class TestSystemPromptHardened:

    def test_anti_salutation_rule_present(self):
        from rune.config import SYSTEM_PROMPT
        # La règle anti-salutation V5.5.6 doit être présente avec
        # le mot-clé fort "JAMAIS"
        assert "JAMAIS" in SYSTEM_PROMPT
        assert "Salut" in SYSTEM_PROMPT
        assert "Bonjour" in SYSTEM_PROMPT

    def test_anti_relance_rule_present(self):
        from rune.config import SYSTEM_PROMPT
        # La règle anti-relance artificielle V5.5.6 mentionne explicitement
        # les patterns interdits
        assert "relance" in SYSTEM_PROMPT.lower() or "remplissage" in SYSTEM_PROMPT.lower()
        assert "Que penses-tu" in SYSTEM_PROMPT

    def test_concrete_examples_present(self):
        """Le prompt donne des exemples négatifs/positifs concrets."""
        from rune.config import SYSTEM_PROMPT
        # On vérifie qu'il y a au moins 1 exemple négatif et 1 positif
        assert "NE faut PAS faire" in SYSTEM_PROMPT
        assert "il faut faire" in SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════
# V5.5.7 — _strip_leading_greeting (post-streaming)
# ═══════════════════════════════════════════════════════════════════════


# Replicate the regex pour pouvoir tester sans charger Hippocampe entier
# (qui require torch). Si la regex évolue, il faut la maintenir ici en
# parallèle — accepter la duplication pour la testabilité.
_GREETING_STRIP_RE = re.compile(
    r"^\s*"
    r"(?:salut|bonjour|coucou|hey|hello|hi)\s*"
    r"(?:[A-ZÀ-Ö][a-zA-ZÀ-ÖØ-öø-ÿ\-]+[\s,]*)?"
    r"(?:"
    r"(?:comment\s+(?:ça\s+va|vas-tu|tu\s+vas)\s*[?!.]?\s*)"
    r"|(?:ça\s+va\s*[?!.]?\s*)"
    r"|(?:tu\s+vas\s+bien\s*[?!.]?\s*)"
    r"|(?:j['\u2019]espère\s+que\s+tu\s+vas\s+bien\s*[?!.]?\s*)"
    r"|(?:tu\s+as\s+passé\s+une\s+bonne\s+(?:journée|soirée|matinée)\s*[?!.]?\s*)"
    r"|(?:how\s+are\s+you\s*[?!.]?\s*)"
    r"){0,2}"
    r"[.,;:!?\s]*",
    re.IGNORECASE,
)


def _strip_greeting_standalone(text):
    """Implémentation autonome équivalente à
    Hippocampe._strip_leading_greeting."""
    if not text or len(text) < 10:
        return text
    m = _GREETING_STRIP_RE.match(text)
    if not m or m.end() == 0:
        return text
    rem = text[m.end():].lstrip(" ,;:!?")
    if len(rem) < 5:
        return text
    if rem and rem[0].islower():
        rem = rem[0].upper() + rem[1:]
    return rem


class TestStripLeadingGreeting:

    def test_strip_full_pattern(self):
        text = (
            "Salut Cédric, comment ça va ? Tu as passé une bonne "
            "journée ? Tu as 41 ans, c'est vrai !"
        )
        result = _strip_greeting_standalone(text)
        assert not result.lower().startswith("salut")
        assert "41 ans" in result
        # Capitalisation préservée
        assert result[0].isupper()

    def test_strip_bonjour(self):
        result = _strip_greeting_standalone("Bonjour Mika ! Voici ce que je sais.")
        assert "Voici" in result
        assert not result.lower().startswith("bonjour")

    def test_strip_coucou(self):
        result = _strip_greeting_standalone(
            "Coucou, j'espère que tu vas bien. Pour ta question..."
        )
        assert "J'espère" in result or "Pour ta question" in result

    def test_strip_hello(self):
        result = _strip_greeting_standalone("Hello there ! Here is the answer.")
        assert "Here is" in result

    def test_no_strip_when_no_greeting(self):
        text = "Tu as 41 ans selon mes calculs."
        assert _strip_greeting_standalone(text) == text

    def test_no_strip_when_already_direct(self):
        text = "Noté, je retiens 1985."
        assert _strip_greeting_standalone(text) == text

    def test_no_strip_greeting_mid_text(self):
        """'salut' au milieu d'une phrase n'est PAS strippé."""
        text = 'Le mot "salut" signifie un geste amical.'
        assert _strip_greeting_standalone(text) == text

    def test_no_strip_when_too_short(self):
        text = "OK."
        assert _strip_greeting_standalone(text) == text

    def test_no_strip_when_would_destroy(self):
        """Si tout est salutation et rien derrière, on garde l'original."""
        text = "Salut Cédric !"
        result = _strip_greeting_standalone(text)
        # Le reste serait vide → on garde l'original (sécurité)
        assert result == text


# ═══════════════════════════════════════════════════════════════════════
# V5.5.8 — Probe streaming (suppression pendant la génération)
# ═══════════════════════════════════════════════════════════════════════


_GREETING_PREFIXES = ("salut", "bonjour", "coucou", "hey", "hello", "hi")


def _probe_strip_greeting_standalone(buffered, max_probe=200):
    """Equivalent à Hippocampe._probe_strip_greeting (testable sans torch)."""
    if not buffered:
        return "", False
    if len(buffered) >= max_probe:
        return _strip_greeting_standalone(buffered), True
    lower = buffered.lstrip().lower()
    starts_with = any(lower.startswith(p) for p in _GREETING_PREFIXES)
    if not starts_with:
        could_become = any(
            p.startswith(lower[:len(p)]) and len(lower) < len(p)
            for p in _GREETING_PREFIXES
        )
        if not could_become:
            return buffered, True
        return "", False
    stripped = _strip_greeting_standalone(buffered)
    if stripped != buffered and len(stripped) >= 5:
        return stripped, True
    return "", False


class TestProbeStrip:
    """Tests du probe utilisé pendant le streaming."""

    def test_empty_buffer(self):
        out, done = _probe_strip_greeting_standalone("")
        assert (out, done) == ("", False)

    def test_no_greeting_immediate_flush(self):
        """Buffer qui ne peut PAS devenir une salutation → flush direct."""
        out, done = _probe_strip_greeting_standalone("Tu as 41 ans")
        assert done is True
        assert out == "Tu as 41 ans"

    def test_ambiguous_prefix_continues(self):
        """'Sa' = préfixe potentiel de 'salut' → on attend."""
        out, done = _probe_strip_greeting_standalone("Sa")
        assert done is False
        assert out == ""

    def test_disambiguated_not_greeting(self):
        """'Sans doute' → ne peut plus devenir 'Salut' → flush."""
        out, done = _probe_strip_greeting_standalone("Sans doute, tu as...")
        assert done is True
        assert "Sans doute" in out

    def test_greeting_incomplete_continues(self):
        """'Salut' seul → on attend la suite pour confirmer."""
        out, done = _probe_strip_greeting_standalone("Salut Cé")
        assert done is False
        assert out == ""

    def test_greeting_complete_strips_and_flushes(self):
        """Salutation + contenu → strip et flush."""
        out, done = _probe_strip_greeting_standalone(
            "Salut Cédric, comment ça va ? Tu as 41 ans"
        )
        assert done is True
        assert not out.lower().startswith("salut")
        assert "41 ans" in out

    def test_max_probe_forces_exit(self):
        """Buffer > max_probe → flush forcé (anti-blocage)."""
        long_text = "Salut " + "x" * 300  # > 200
        out, done = _probe_strip_greeting_standalone(long_text, max_probe=200)
        assert done is True


class TestStreamingScenarios:
    """Tests d'intégration : simule un streaming chunk par chunk."""

    def _simulate(self, full_text, chunk_size=8):
        """Reproduit la logique du flow streaming et collecte ce qui
        serait affiché à l'utilisateur."""
        greeting_probe = True
        greeting_stripped_offset = 0
        last_emitted = ""
        yielded_texts = []

        for i in range(chunk_size, len(full_text) + chunk_size, chunk_size):
            clean = full_text[:i] if i <= len(full_text) else full_text

            if greeting_probe:
                stripped, done = _probe_strip_greeting_standalone(clean)
                if done:
                    greeting_probe = False
                    if stripped != clean:
                        greeting_stripped_offset = len(clean) - len(stripped)
                    if stripped and stripped != last_emitted:
                        last_emitted = stripped
                        yielded_texts.append(stripped)
            else:
                emitted = (
                    clean[greeting_stripped_offset:]
                    if greeting_stripped_offset > 0
                    else clean
                )
                if emitted and emitted[0:1].islower():
                    emitted = emitted[0].upper() + emitted[1:]
                if emitted and emitted != last_emitted:
                    last_emitted = emitted
                    yielded_texts.append(emitted)

        return yielded_texts, last_emitted

    def test_streaming_strips_greeting(self):
        """Le bug original : "Salut Cédric, comment ça va ?" doit
        disparaître dès le premier yield."""
        yielded, final = self._simulate(
            "Salut Cédric, comment ça va ? Tu as 41 ans, c'est vrai !"
        )
        # Aucun des yields ne doit commencer par "Salut"
        for y in yielded:
            assert not y.lower().startswith("salut"), (
                f"Yield commence par 'Salut' : {y!r}"
            )
        # La version finale doit contenir le contenu utile
        assert "41 ans" in final

    def test_streaming_no_greeting_unchanged(self):
        """Sans salutation, le streaming est identique à l'original."""
        text = "Tu as 41 ans selon mes calculs déterministes."
        yielded, final = self._simulate(text)
        # Le premier yield commence par "Tu as"
        assert yielded[0].startswith("Tu as")
        # Le final est identique au texte (sauf casse premier char)
        assert final == text

    def test_streaming_short_response(self):
        """Une réponse très courte sans salutation passe direct."""
        yielded, final = self._simulate("Noté.", chunk_size=4)
        assert "Noté" in final

    def test_streaming_with_ambiguous_start(self):
        """'Sans' commence par 'Sa' (préfixe potentiel 'salut')
        mais doit être flushé une fois ambiguïté levée."""
        yielded, final = self._simulate(
            "Sans doute, tu as 41 ans cette année.", chunk_size=4
        )
        assert "Sans doute" in final
        # Pas perdu pendant le probing
        assert any("Sans" in y for y in yielded)
