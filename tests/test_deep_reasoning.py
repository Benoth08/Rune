"""Tests pour la chaîne de raisonnement profond — ``DeepReasoningChain``.

Ces tests utilisent un modèle factice (``_FakeModel``) qui renvoie des
sorties scriptées, donc aucun GPU ni modèle réel requis. On teste :

- le routeur de complexité (``assess_complexity``) sur des cas variés ;
- l'orchestration ``run`` (skip / chaîne complète / dégradations) ;
- la robustesse aux échecs d'étape.
"""
from __future__ import annotations

from rune.cognition.deep_reasoning import DeepReasoningChain


# ── Doublures ──────────────────────────────────────────────────────────

class _FakeTokenizer:
    """Tokenizer minimal avec apply_chat_template."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        # On concatène juste le contenu — le FakeModel se moque du format.
        return "\n".join(m["content"] for m in messages)


class _FakeModel:
    """Modèle factice : renvoie des sorties scriptées, ou lève une erreur.

    ``scripted`` est une liste de réponses consommées dans l'ordre à
    chaque appel de ``generate``. Une valeur ``None`` dans la liste
    déclenche une exception, pour tester la robustesse.

    ``token_budgets`` enregistre les ``max_new_tokens`` reçus à chaque
    appel — permet de vérifier que chaque étape passe bien son budget.
    """

    def __init__(self, scripted: list[str | None]):
        self.tokenizer = _FakeTokenizer()
        self._scripted = list(scripted)
        self.calls: list[str] = []
        self.token_budgets: list[int] = []

    def generate(self, rendered: str, max_new_tokens: int = 0,
                 temperature: float = 0.0) -> str:
        self.calls.append(rendered)
        self.token_budgets.append(max_new_tokens)
        if not self._scripted:
            return ""
        out = self._scripted.pop(0)
        if out is None:
            raise RuntimeError("simulated model failure")
        return out


class _FakeEntity:
    def __init__(self, value: str, etype: str):
        self.value = value
        self.type = etype


class _FakeKG:
    def __init__(self, entities: dict | None = None):
        self.entities = entities or {}


# ── Routeur de complexité ──────────────────────────────────────────────

def test_complexity_trivial_returns_zero():
    chain = DeepReasoningChain(_FakeModel([]), None)
    assert chain.assess_complexity("Salut") == 0
    assert chain.assess_complexity("Quelle heure ?") == 0
    assert chain.assess_complexity("") == 0


def test_complexity_analytical_marker_triggers():
    chain = DeepReasoningChain(_FakeModel([]), None)
    # "pourquoi" est un marqueur analytique fort (+2 → seuil atteint).
    assert chain.assess_complexity("Pourquoi le ciel est bleu ?") == 2
    # "compare" idem.
    assert chain.assess_complexity("Compare X et Y") == 2


def test_complexity_long_question_alone_not_enough():
    chain = DeepReasoningChain(_FakeModel([]), None)
    # Question longue mais sans marqueur, sans signal : 1 point → 0.
    long_plain = " ".join(["mot"] * 15)
    assert chain.assess_complexity(long_plain) == 0


def test_complexity_long_plus_surprise_triggers():
    chain = DeepReasoningChain(_FakeModel([]), None)
    long_plain = " ".join(["mot"] * 15)
    # Longueur (1) + surprise élevée (1) = 2 → déclenche.
    got = chain.assess_complexity(
        long_plain, surprise={"global": 0.7},
    )
    assert got == 2


def test_complexity_entities_contribute():
    chain = DeepReasoningChain(_FakeModel([]), None)
    long_plain = " ".join(["mot"] * 15)
    # Longueur (1) + 2 entités (1) = 2 → déclenche.
    assert chain.assess_complexity(long_plain, kg_entity_count=2) == 2


def test_complexity_doubt_contributes():
    chain = DeepReasoningChain(_FakeModel([]), None)
    long_plain = " ".join(["mot"] * 15)
    # Longueur (1) + doute élevé (1) = 2 → déclenche.
    assert chain.assess_complexity(long_plain, doubt_index=0.6) == 2


# ── Orchestration run() ────────────────────────────────────────────────

def test_run_skips_on_trivial_question():
    """Question simple → run renvoie "" sans appeler le modèle."""
    model = _FakeModel(["ne devrait pas être appelé"])
    chain = DeepReasoningChain(model, None)
    result = chain.run("Salut")
    assert result == ""
    assert model.calls == []  # aucun appel LLM


def test_run_full_chain_two_steps():
    """Question complexe → 2 appels (explorer + critiquer)."""
    model = _FakeModel([
        "EXPLORATION: voici ce que je sais et ignore.",
        "CONSOLIDÉ: raisonnement corrigé et fiable.",
    ])
    chain = DeepReasoningChain(model, None)
    result = chain.run("Pourquoi cette analyse est-elle complexe ?")
    assert result == "CONSOLIDÉ: raisonnement corrigé et fiable."
    assert len(model.calls) == 2  # explorer + critiquer


def test_run_falls_back_to_exploration_if_critique_fails():
    """Si la critique échoue, on renvoie l'exploration plutôt que rien."""
    model = _FakeModel([
        "EXPLORATION valable.",
        None,  # la critique lève une exception
    ])
    chain = DeepReasoningChain(model, None)
    result = chain.run("Pourquoi est-ce compliqué ?")
    assert result == "EXPLORATION valable."
    assert len(model.calls) == 2


def test_run_returns_empty_if_exploration_fails():
    """Si l'exploration échoue, run renvoie "" (→ fallback simple)."""
    model = _FakeModel([None])  # l'exploration lève une exception
    chain = DeepReasoningChain(model, None)
    result = chain.run("Pourquoi est-ce compliqué ?")
    assert result == ""


def test_run_on_step_callback_invoked():
    """Le callback on_step reçoit bien les libellés d'étape."""
    model = _FakeModel(["exploration", "consolidé"])
    chain = DeepReasoningChain(model, None)
    labels: list[str] = []
    chain.run(
        "Pourquoi cette question est-elle complexe ?",
        on_step=labels.append,
    )
    # 2-étapes (question medium) → deux libellés.
    assert len(labels) == 2
    assert "Exploration" in labels[0]
    assert "Feuille de route" in labels[1]


def test_run_truncates_long_output():
    """Le raisonnement renvoyé est borné par reasoning_text_max_chars."""
    from rune.settings import get_settings
    max_chars = get_settings().reasoning_text_max_chars
    huge = "x" * (max_chars + 5000)
    model = _FakeModel([huge, huge])
    chain = DeepReasoningChain(model, None)
    result = chain.run("Pourquoi ce très long raisonnement ?")
    assert len(result) <= max_chars


def test_run_step_token_budgets():
    """Chaque étape passe un budget de tokens suffisant.

    Régression : la v1 utilisait 320 tokens/étape, ce qui coupait
    l'exploration à mi-phrase. Les budgets vivent maintenant dans
    settings.py et sont choisis par le routeur (medium / high). Les
    deux étapes doivent recevoir le même budget par run, et ce budget
    doit largement dépasser l'ancien 320.
    """
    from rune.settings import get_settings
    s = get_settings()
    model = _FakeModel(["exploration", "consolidé"])
    chain = DeepReasoningChain(model, None)
    chain.run("Pourquoi cette question mérite-t-elle une analyse ?")

    # Deux appels, deux budgets enregistrés.
    assert len(model.token_budgets) == 2
    # Les deux étapes d'un même run partagent le budget choisi par
    # le routeur — soit medium, soit high selon la complexité.
    assert model.token_budgets[0] == model.token_budgets[1]
    chosen = model.token_budgets[0]
    assert chosen in (
        s.reasoning_deep_step_medium_tokens,
        s.reasoning_deep_step_high_tokens,
    )
    # Le bug d'origine : budget trop serré (320). On verrouille un
    # plancher bien au-dessus.
    assert chosen >= 500


def test_run_high_complexity_gets_bigger_budget():
    """Une question à signaux forts lance la chaîne 4-étapes avec le bon budget."""
    from rune.settings import get_settings
    s = get_settings()
    # 4 étapes → 4 appels (décomposer + explorer + critiquer + synthétiser)
    model = _FakeModel(["sous-questions", "exploration", "critique", "synthèse"])
    chain = DeepReasoningChain(model, None)
    long_analytical = "compare en détail " + " ".join(["aspect"] * 15)
    chain.run(long_analytical, surprise={"global": 0.7})
    # 4 appels enregistrés
    assert len(model.token_budgets) == 4
    # Étape 0 (décomposer) = budget medium
    assert model.token_budgets[0] == s.reasoning_deep_step_medium_tokens
    # Étape 1 (explorer) = budget HIGH — c'est le cœur du raisonnement
    assert model.token_budgets[1] == s.reasoning_deep_step_high_tokens
    # Étapes 2-3 (critiquer, synthétiser) = budget medium
    assert model.token_budgets[2] == s.reasoning_deep_step_medium_tokens
    assert model.token_budgets[3] == s.reasoning_deep_step_medium_tokens


def test_run_4_step_callback_labels():
    """En mode 4-étapes, on_step reçoit 4 libellés distincts."""
    model = _FakeModel(["sous-questions", "exploration", "critique", "synthèse"])
    chain = DeepReasoningChain(model, None)
    labels: list[str] = []
    long_analytical = "compare en détail " + " ".join(["aspect"] * 15)
    chain.run(long_analytical, surprise={"global": 0.7}, on_step=labels.append)
    assert len(labels) == 4
    assert "Décomposition" in labels[0]
    assert "Exploration" in labels[1]
    assert "Critique" in labels[2]
    assert "Synthèse" in labels[3]


def test_run_4_step_degrades_to_2_if_decompose_fails():
    """Si la décomposition échoue, on retombe sur la chaîne 2-étapes."""
    # 1er appel (décomposer) échoue → None, puis 2 appels pour la chaîne 2-étapes
    model = _FakeModel([None, "exploration", "feuille de route"])
    chain = DeepReasoningChain(model, None)
    long_analytical = "compare en détail " + " ".join(["aspect"] * 15)
    result = chain.run(long_analytical, surprise={"global": 0.7})
    # Le résultat vient de la chaîne 2-étapes de repli
    assert result == "feuille de route"


# ── Intégration KG ─────────────────────────────────────────────────────

def test_kg_hint_included_in_exploration_prompt():
    """Les entités KG sont injectées dans le prompt d'exploration."""
    kg = _FakeKG({
        "e1": _FakeEntity("Mika", "person"),
        "e2": _FakeEntity("SP3H", "organization"),
    })
    model = _FakeModel(["exploration", "consolidé"])
    chain = DeepReasoningChain(model, kg)
    chain.run("Pourquoi ce sujet est-il complexe ?")
    # Le premier appel (exploration) doit contenir le rappel KG.
    first_call = model.calls[0]
    assert "Mika" in first_call
    assert "SP3H" in first_call


def test_kg_hint_empty_when_no_kg():
    """Pas de KG → pas de plantage, prompt sans rappel d'entités."""
    model = _FakeModel(["exploration", "consolidé"])
    chain = DeepReasoningChain(model, None)
    result = chain.run("Pourquoi est-ce une question difficile ?")
    assert result == "consolidé"


# ── Ancrage web (couplage raisonnement + recherche web) ────────────────

def test_web_context_injected_in_exploration():
    """Le contexte web est injecté dans le prompt d'exploration (2-step)."""
    model = _FakeModel(["exploration", "feuille de route"])
    chain = DeepReasoningChain(model, None)
    web = "[1] CRYSTALS-Kyber est le standard NIST pour le chiffrement."
    chain.run(
        "Qu'est-ce que la cryptographie post-quantique et comment ça marche",
        web_context=web, force_minimum=2,
    )
    # Le 1er appel (exploration) doit contenir le contexte web.
    assert "CRYSTALS-Kyber" in model.calls[0]
    assert "ne les invente pas" in model.calls[0]


def test_web_context_in_decompose_and_explore_only():
    """En 4-étapes, le web va dans décompose + explore, pas critique/synthèse."""
    model = _FakeModel([
        "sous-questions", "exploration", "critique", "synthèse",
    ])
    chain = DeepReasoningChain(model, None)
    web = "[1] McEliece repose sur les codes correcteurs d'erreurs."
    long_analytical = "compare en détail " + " ".join(["aspect"] * 15)
    chain.run(long_analytical, surprise={"global": 0.7},
              web_context=web, force_minimum=2)
    assert len(model.calls) == 4
    assert "McEliece" in model.calls[0]   # décomposition
    assert "McEliece" in model.calls[1]   # exploration
    assert "McEliece" not in model.calls[2]  # critique : pas de web
    assert "McEliece" not in model.calls[3]  # synthèse : pas de web


def test_web_context_absent_no_block():
    """Sans web_context, aucun bloc web dans les prompts."""
    model = _FakeModel(["exploration", "feuille de route"])
    chain = DeepReasoningChain(model, None)
    chain.run(
        "Qu'est-ce que la cryptographie post-quantique et comment ça marche",
        force_minimum=2,
    )
    assert "ne les invente pas" not in model.calls[0]


def test_web_context_truncated():
    """Un contexte web énorme est borné avant injection."""
    from rune.cognition.deep_reasoning import _WEB_CONTEXT_MAX_CHARS
    model = _FakeModel(["exploration", "feuille de route"])
    chain = DeepReasoningChain(model, None)
    huge_web = "[1] " + "x" * 8000
    chain.run(
        "Qu'est-ce que la cryptographie post-quantique et comment ça marche",
        web_context=huge_web, force_minimum=2,
    )
    # Le nombre de 'x' dans le prompt ne dépasse pas le plafond.
    assert model.calls[0].count("x") <= _WEB_CONTEXT_MAX_CHARS
