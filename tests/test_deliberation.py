"""Raisonnement délibératif multi-angles (modèles non-thinking).

Couvre (1) le DeliberativeReasoner lui-même avec un modèle factice, et
(2) son câblage CONDITIONNEL dans l'orchestrateur : multi-angles pour les
modèles non-thinking, réflexion inline préservée pour les modèles thinking.
"""

from rune.cognition.deliberation import (
    DeliberativeReasoner, is_trivial_message)


class _FakeModel:
    """Modèle factice : compte les appels, renvoie un texte par étape."""
    def __init__(self):
        self.calls = 0
        self.tokenizer = None  # force le repli texte brut dans _call

    def generate(self, prompt, max_new_tokens=256, temperature=0.3):
        self.calls += 1
        # renvoie un contenu plausible non vide pour chaque étape
        return f"raisonnement étape {self.calls} : analyse du problème."


def test_trivial_message_skips_deliberation():
    assert is_trivial_message("bonjour") is True
    assert is_trivial_message("ok merci") is True
    assert is_trivial_message("") is True
    # une vraie question n'est pas triviale
    assert is_trivial_message(
        "pourquoi le calcul de mensualité renvoie une valeur fausse ?") is False


def test_deliberate_returns_reasoning_text():
    m = _FakeModel()
    r = DeliberativeReasoner(model=m, min_step_margin=0.0)
    out = r.deliberate("le test échoue sur le 7e terme de Fibonacci", depth=4)
    assert isinstance(out, str)
    assert out.strip()                 # produit un texte non vide
    assert m.calls >= 2                # plusieurs angles → plusieurs appels


def test_deliberate_depth_zero_is_empty():
    m = _FakeModel()
    r = DeliberativeReasoner(model=m, min_step_margin=0.0)
    assert r.deliberate("question", depth=0) == ""


def test_deliberate_trivial_is_empty():
    m = _FakeModel()
    r = DeliberativeReasoner(model=m, min_step_margin=0.0)
    assert r.deliberate("salut", depth=4) == ""
    assert m.calls == 0                # aucun appel pour un trivial


def test_orchestrator_wires_deliberation_conditionally():
    """Le multi-angles est branché pour non-thinking, l'inline préservé."""
    src = open("rune/agentic/orchestrator.py", encoding="utf-8").read()
    # le module est bien importé/utilisé
    assert "DeliberativeReasoner" in src
    # conditionné au NON-thinking
    assert "if not thinking_mode:" in src
    # la réflexion profonde inline reste présente (thinking + repli)
    assert "Réflexion approfondie" in src
    # le déclencheur quand bloqué est intact
    assert "pending_delib = True" in src
