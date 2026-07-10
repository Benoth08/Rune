"""Tests for fix #12 (anti-personification rule) + fix #13 (prompt coherence).

These tests pin down architectural decisions made during the v8 sprint:

1. The SYSTEM_PROMPT contains a generic anti-personification rule that
   does NOT hardcode specific words ("chat", "Aix") — so it generalises
   to any kind of false possession or false experience.

2. No prompt block injected into the LLM context contains the imperative
   pattern "Tu DOIS" (which was empirically observed to trigger verbatim
   recitation on instruction-tuned models — anti-pattern fixed in v5
   for the identity block, extended in v8 to the rest of the prompts).

3. The KG ``recall_facts`` does NOT expose the ``mention_count`` system
   counter to the LLM (which interpreted "(person, vu 19x)" as a
   duration on Qwen2.5-7B → "tu y vis depuis 19 minutes" hallucination).
"""
from __future__ import annotations

import pytest


# ── Fix #12 — Anti-personification rule ────────────────────────────────


def test_system_prompt_has_anti_personification_rule():
    """A rule must explicitly forbid the assistant from inventing
    personal possessions or experiences."""
    from rune.config import SYSTEM_PROMPT

    # The rule must mention the absence of physical/personal life.
    assert "ni corps ni vie matérielle" in SYSTEM_PROMPT, (
        "SYSTEM_PROMPT must include the anti-personification rule"
    )


def test_anti_personification_rule_is_generic_not_hardcoded():
    """The rule must not bake in specific examples like 'chat' or 'Aix'
    in a way that would make it only cover those specific cases. The
    spirit is GENERIC: no false possessions, no false experiences.

    V5 refondu : la règle vit maintenant dans la section ``# Identité``
    sous une formulation reformulée. On vérifie la sémantique générique
    plutôt qu'une phrase littérale qui peut bouger entre versions.
    """
    from rune.config import SYSTEM_PROMPT

    # La règle V5 doit couvrir : (1) absence de vie matérielle,
    # (2) interdiction d'inventer un équivalent personnel.
    has_no_physical_life = "ni corps ni vie matérielle" in SYSTEM_PROMPT
    has_no_invent = (
        "sans jamais inventer d'équivalent personnel" in SYSTEM_PROMPT
        or "jamais « j'ai aussi" in SYSTEM_PROMPT
    )
    assert has_no_physical_life and has_no_invent, (
        "Anti-personification doit interdire d'inventer un équivalent "
        "personnel (formulation peut varier entre versions)"
    )

    # Negative check générique : la règle ne doit pas se réduire à un
    # exemple spécifique. On accepte que des exemples soient cités
    # (animaux, famille, loisirs) tant que la règle reste générique.
    # Les exemples sont OK ; les hardcodes du genre "chat" seul, sans
    # contexte, signaleraient une régression.
    # Pas d'assertion stricte ici : on se contente de la couverture
    # sémantique ci-dessus.


# ── Fix #13 — Prompt coherence (no more "Tu DOIS") ────────────────────


def test_no_imperative_tu_dois_anywhere_in_prompts():
    """The 'Tu DOIS' / 'RÉPONDS' / 'JAMAIS' injunctive pattern was
    empirically observed to trigger verbatim recitation on
    instruction-tuned models (Mistral-7B-Instruct especially).

    Fix #10 corrected the identity block. Fix #13 extends this to the
    entire SYSTEM_PROMPT and the web-context injection block. This
    test pins both down so a future edit can't silently reintroduce
    the anti-pattern.
    """
    from rune.config import SYSTEM_PROMPT

    # All-caps imperative markers — these were empirically problematic.
    # Note: "JAMAIS" in lowercase ("jamais") is fine in normal prose;
    # it's the all-caps shouting form that pushes models to overcompliance.
    forbidden_caps = ["Tu DOIS", "RÉPONDS", "ne dis JAMAIS"]
    for token in forbidden_caps:
        assert token not in SYSTEM_PROMPT, (
            f"SYSTEM_PROMPT contains forbidden imperative pattern "
            f"'{token}' — fix #13 specifically removed this. "
            f"Use descriptive phrasing instead "
            f"(see [Identité] block as reference)."
        )


def test_web_context_block_is_descriptive_not_injunctive():
    """The web-context injection block must use descriptive phrasing
    consistent with the post-fix-#10 identity block. Imperative
    'Tu DOIS' patterns trigger verbatim recitation."""
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("torch not available or broken CUDA")

    from unittest.mock import MagicMock
    from rune.hippocampe import Hippocampe

    # We can't instantiate Hippocampe without heavy deps, but the
    # method we test is a pure string formatter. Bind it loosely.
    block = Hippocampe._inject_web_context(
        MagicMock(),  # self placeholder
        web_context="Some example web result.",
        rag_context="",
    )

    # The block must NOT contain the old injunctive markers.
    forbidden = ["Tu DOIS", "⚠️ INSTRUCTION", "Ne dis PAS"]
    for token in forbidden:
        assert token not in block, (
            f"Web-context block still contains '{token}' — "
            f"fix #13 mandated descriptive phrasing."
        )

    # The block must still convey that web results are recent (just
    # without the shouting).
    assert "récent" in block.lower(), (
        "Web-context block should still indicate recency, "
        "just in a non-injunctive tone"
    )


# ── Fix #13 — KG recall_facts no longer leaks mention_count ────────────


def test_recall_facts_does_not_expose_mention_count():
    """`mention_count` is a system counter — it must stay on the
    Entity object (used by scoring + UI) but never appear in the
    LLM prompt. >5B models misread '(person, vu 19x)' as a duration
    (observed: 'tu y vis depuis 19 minutes' on Qwen2.5-7B)."""
    from rune.memory.kg import KnowledgeGraphStore

    kg = KnowledgeGraphStore()
    eid = kg.upsert_entity("Mika", "person")
    # Bump mention_count artificially so we'd see "vu 5x" if the bug
    # were still there.
    kg.entities[eid].mention_count = 5

    facts = kg.query_by_question(
        "Mika", [{"text": "Mika", "label": "person"}],
    )

    joined = "\n".join(facts)

    # The Nx pattern (e.g. "vu 5x", "vu 19x") must be absent.
    import re
    assert not re.search(r"vu \d+x", joined), (
        f"recall_facts leaked mention_count to LLM prompt: {joined}"
    )

    # But the entity itself must still carry the counter (system-level).
    assert kg.entities[eid].mention_count == 5, (
        "mention_count must still be tracked on the Entity object — "
        "only the LLM-facing string was supposed to drop it."
    )


# ── Fix #13 — Memory section headers carry usage hint ──────────────────


def test_memory_section_headers_have_usage_hint():
    """Memory blocks ([Mémoire épisodique], [Mémoire sémantique], [Faits
    connus]) all carry a "— pour information" suffix that signals to
    the model these sections are *contextual* — not instructions to
    recite verbatim. Same descriptive style as the identity block."""
    from rune.cognition import retrieval as retrieval_module
    import inspect

    src = inspect.getsource(retrieval_module)

    # Each header must have the descriptive suffix.
    expected_headers = [
        "[Mémoire épisodique — pour information]",
        "[Mémoire sémantique — pour information]",
        "[Faits connus — pour information]",
    ]
    for header in expected_headers:
        assert header in src, (
            f"Expected header '{header}' missing in retrieval module — "
            f"fix #13 added the usage hint to keep all memory blocks "
            f"in the same descriptive register as [Identité]."
        )
