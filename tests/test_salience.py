"""Tests for the 3-cascade salience filter."""
from __future__ import annotations

import torch

from rune.memory.salience import SalienceFilter


def test_n1_rejects_short():
    sf = SalienceFilter(min_length=8)
    result = sf.evaluate("ok")
    assert not result.passed
    assert result.stage == "N1"


def test_n1_rejects_noise():
    sf = SalienceFilter()
    result = sf.evaluate("bonjour")
    assert not result.passed
    assert result.stage == "N1"


def test_n1_rejects_noise_phrase():
    sf = SalienceFilter()
    result = sf.evaluate("oui merci")
    assert not result.passed


def test_n2_scores_proper_nouns():
    sf = SalienceFilter(min_score=0.1)
    result = sf.evaluate("J'ai parlé avec Jean-Pierre Dupont hier à Paris")
    assert result.passed
    assert result.score > 0.3


def test_n2_scores_numbers():
    sf = SalienceFilter(min_score=0.1)
    result = sf.evaluate("Le projet coûte environ 150000 euros pour 2024")
    assert result.passed
    assert result.score > 0.2


def test_n2_rejects_low_score():
    sf = SalienceFilter(min_score=0.8)
    result = sf.evaluate("je ne sais pas trop quoi dire")
    assert not result.passed
    assert result.stage == "N2"


def test_n3_rejects_redundant():
    sf = SalienceFilter(min_score=0.1, redundancy_threshold=0.95)
    emb = torch.randn(64)
    # First time: should pass
    r1 = sf.evaluate("test message long enough to pass", emb)
    assert r1.passed
    # Same embedding again: should be redundant
    r2 = sf.evaluate("another message still long enough", emb)
    assert not r2.passed
    assert r2.stage == "N3"


def test_n3_accepts_different():
    sf = SalienceFilter(min_score=0.1)
    r1 = sf.evaluate("premier message important sur le projet", torch.randn(64))
    assert r1.passed
    r2 = sf.evaluate("deuxième message totalement différent", torch.randn(64))
    assert r2.passed


def test_reset_clears_cache():
    sf = SalienceFilter()
    sf.evaluate("long enough message here", torch.randn(64))
    sf.reset()
    assert len(sf._embed_cache) == 0


# ── V5.6.1 — Self-disclosure bonus ──────────────────────────────────


class TestSelfDisclosureBonus:
    """V5.6.1 — Les phrases self-disclosure doivent passer N2 même en
    minuscule sans noms propres capitalisés. Sinon les fallbacks
    V5.5.1-9 ne tournent jamais et le KG perd les faits."""

    def test_lowercase_intro_passes(self):
        """Le bug original : 'Bonjour, je m'appelle michael' en
        minuscule était rejeté par N2."""
        sf = SalienceFilter()
        score = sf._n2_heuristic("Bonjour, je m'appelle michael")
        assert score >= sf.min_score, (
            f"Score {score:.2f} doit dépasser min_score {sf.min_score}"
        )

    def test_lowercase_habite(self):
        sf = SalienceFilter()
        score = sf._n2_heuristic("j'habite à aix-en-provence")
        assert score >= sf.min_score

    def test_lowercase_age(self):
        sf = SalienceFilter()
        score = sf._n2_heuristic("j'ai 36 ans")
        assert score >= sf.min_score

    def test_lowercase_birth_year(self):
        sf = SalienceFilter()
        score = sf._n2_heuristic("je suis né en 1985")
        assert score >= sf.min_score

    def test_lowercase_employer(self):
        sf = SalienceFilter()
        score = sf._n2_heuristic("je travaille chez framatome")
        assert score >= sf.min_score

    def test_english_intro(self):
        sf = SalienceFilter()
        score = sf._n2_heuristic("my name is john")
        assert score >= sf.min_score

    def test_normal_chat_not_boosted(self):
        """Pas de faux positif : du chat normal sans self-disclosure
        n'est pas artificiellement boosté."""
        sf = SalienceFilter()
        # Texte sans pattern self-disclosure : doit garder son score
        # naturel (peut être bas ou haut selon le contenu, mais le
        # bonus +0.4 ne doit pas s'appliquer).
        score_with = sf._n2_heuristic("Recommande-moi un livre")
        score_without = sf._n2_heuristic("Recommande-moi un livre s'il te plait")
        # Le bonus self-disclosure n'a pas été appliqué (pas de "je m'appelle")
        assert score_with < 0.5
        assert score_without < 0.5

    def test_self_disclosure_evaluate_passes(self):
        """Test end-to-end avec evaluate() complet (et non juste N2)."""
        sf = SalienceFilter()
        # Doit passer : self-disclosure court mais saillant
        result = sf.evaluate("je m'appelle cédric")
        assert result.passed, (
            f"Self-disclosure rejeté: stage={result.stage} reason={result.reason}"
        )

    def test_question_factual_not_self_disclosure(self):
        """Une question factuelle n'est pas self-disclosure et ne doit
        pas avoir le bonus +0.4."""
        sf = SalienceFilter()
        score = sf._n2_heuristic("Quel temps fait-il à Paris ?")
        # Pas de pattern self-disclosure dedans
        # Note : le score peut être correct par d'autres signaux (question
        # mark, "Paris" capitalisé), mais on vérifie juste qu'il n'y a
        # pas de bonus artificiel
        # On vérifie en comparant avec une version sans pattern
        assert sf._SELF_DISCLOSURE_HINT.search("Quel temps fait-il à Paris ?") is None
