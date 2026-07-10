"""Unit tests for :mod:`rune.cognition.generation`.

Covers:
* ``strip_reasoning`` — the reasoning-tag splitter (case-insensitive,
  multiple tags, two-pass artifact prefixes)
* ``mask_open_tags`` — the streaming-time helper for half-rendered tags
* ``QUESTION_STARTS`` — the contract stop list
* :class:`ReasoningGenerator` — two-pass reasoning prompt construction
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# These functions are pure Python — no torch needed.
from rune.cognition.generation import (
    QUESTION_STARTS,
    ReasoningGenerator,
    mask_open_tags,
    strip_reasoning,
)


# ── strip_reasoning ────────────────────────────────────────────────────

def test_strip_reasoning_no_tags_returns_text_unchanged():
    clean, reasoning = strip_reasoning("hello world")
    assert clean == "hello world"
    assert reasoning == ""


def test_strip_reasoning_removes_reflexion_tag():
    clean, reasoning = strip_reasoning(
        "<reflexion>thinking step</reflexion>final answer"
    )
    assert clean == "final answer"
    assert reasoning == "thinking step"


def test_strip_reasoning_case_insensitive():
    """Models that capitalise the tag must still be handled."""
    clean, reasoning = strip_reasoning(
        "<REFLEXION>analysis</REFLEXION>réponse"
    )
    assert clean == "réponse"
    assert reasoning == "analysis"


def test_strip_reasoning_multiple_blocks_concatenated():
    clean, reasoning = strip_reasoning(
        "<reflexion>step 1</reflexion>middle<reflexion>step 2</reflexion>end"
    )
    assert clean == "middleend"
    assert "step 1" in reasoning
    assert "step 2" in reasoning


def test_strip_reasoning_strips_two_pass_prefix():
    clean, _ = strip_reasoning(
        "D'accord, voici ma réponse finale : voici l'info."
    )
    assert clean == "voici l'info."


def test_strip_reasoning_two_pass_prefix_case_insensitive():
    clean, _ = strip_reasoning(
        "VOICI MA RÉPONSE FINALE: réponse réelle"
    )
    assert clean == "réponse réelle"


def test_strip_reasoning_only_strips_prefix_at_start():
    """The artefact strip must NOT remove the prefix from middle of text."""
    text = "Bonjour. Voici ma réponse finale : milieu."
    clean, _ = strip_reasoning(text)
    # The string starts with "Bonjour", not the artefact prefix, so the
    # whole text is preserved.
    assert clean == text.strip()


def test_strip_reasoning_with_thinking_tag():
    """Both <reflexion> and <thinking> are configured stop tags."""
    clean, reasoning = strip_reasoning(
        "<thinking>internal</thinking>spoken"
    )
    assert clean == "spoken"
    assert reasoning == "internal"


def test_strip_reasoning_preserves_inner_whitespace():
    """The inner content of a tag is .strip()'d but the outer text only
    has its outer whitespace trimmed."""
    clean, reasoning = strip_reasoning(
        "<reflexion>  inner  </reflexion>  outer  "
    )
    assert clean == "outer"
    assert reasoning == "inner"


def test_strip_reasoning_multiline_block():
    text = """<reflexion>
line 1
line 2
</reflexion>
final"""
    clean, reasoning = strip_reasoning(text)
    assert clean == "final"
    assert "line 1" in reasoning
    assert "line 2" in reasoning


# ── mask_open_tags ─────────────────────────────────────────────────────

def test_mask_open_tags_no_tag_passthrough():
    text = "no tags here"
    assert mask_open_tags(text) == text


def test_mask_open_tags_closed_tag_passthrough():
    text = "<reflexion>done</reflexion>continuation"
    # No unclosed tag, so passthrough
    assert mask_open_tags(text) == text


def test_mask_open_tags_unclosed_truncates():
    """Half-rendered <reflexion>partial... → strip from opening tag."""
    text = "before<reflexion>partial reasoning still streaming"
    out = mask_open_tags(text)
    assert out == "before"


def test_mask_open_tags_case_insensitive():
    text = "before<REFLEXION>partial"
    out = mask_open_tags(text)
    assert out == "before"


def test_mask_open_tags_multiple_tags_handles_last_unclosed():
    """A closed tag followed by an unclosed one — truncate at the unclosed."""
    text = "<reflexion>step1</reflexion>middle<thinking>still going"
    out = mask_open_tags(text)
    assert out == "<reflexion>step1</reflexion>middle"


def test_mask_open_tags_strips_trailing_whitespace():
    text = "before  <reflexion>partial"
    out = mask_open_tags(text)
    assert out == "before"


# ── QUESTION_STARTS ────────────────────────────────────────────────────

def test_question_starts_contains_french_and_english():
    must = {"quel", "comment", "pourquoi", "what", "how", "why"}
    assert must.issubset(QUESTION_STARTS)


def test_question_starts_is_lowercase():
    """All entries lowercase — caller is expected to .lower() its match key."""
    assert all(s == s.lower() for s in QUESTION_STARTS)


# ── ReasoningGenerator ─────────────────────────────────────────────────

def _kg_with_entities(values: list[tuple[str, str]]):
    """Build a KG mock with ``entities`` dict whose values look like KG entities."""
    kg = MagicMock()
    from types import SimpleNamespace
    kg.entities = {
        f"e{i}": SimpleNamespace(value=v, type=t)
        for i, (v, t) in enumerate(values)
    }
    return kg


def test_reasoning_generator_no_kg_no_context_hint():
    model = MagicMock()
    model.generate.return_value = "raisonnement structuré"
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "<rendered>"

    gen = ReasoningGenerator(model=model, kg=None)
    result = gen.generate("Pourquoi ?")
    assert result == "raisonnement structuré"
    # Verify the KG hint is absent from the system prompt
    call = model.tokenizer.apply_chat_template.call_args
    rendered_prompt = call.args[0] if call.args else call.kwargs.get("conversation")
    sys_msg = rendered_prompt[0]["content"]
    assert "Contexte mémoire" not in sys_msg


def test_reasoning_generator_includes_kg_facts():
    model = MagicMock()
    model.generate.return_value = "ok"
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "<rendered>"
    kg = _kg_with_entities([("Mika", "person"), ("Aix", "location")])

    gen = ReasoningGenerator(model=model, kg=kg)
    gen.generate("?")
    call = model.tokenizer.apply_chat_template.call_args
    rendered_prompt = call.args[0] if call.args else call.kwargs.get("conversation")
    sys_msg = rendered_prompt[0]["content"]
    assert "Contexte mémoire" in sys_msg
    assert "Mika (person)" in sys_msg
    assert "Aix (location)" in sys_msg


def test_reasoning_generator_caps_kg_facts_at_10():
    """Only the first 10 entities are surfaced — keep the prompt bounded."""
    model = MagicMock()
    model.generate.return_value = "ok"
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "<rendered>"
    entities = [(f"v{i}", "thing") for i in range(15)]
    kg = _kg_with_entities(entities)

    gen = ReasoningGenerator(model=model, kg=kg)
    gen.generate("?")
    call = model.tokenizer.apply_chat_template.call_args
    rendered_prompt = call.args[0] if call.args else call.kwargs.get("conversation")
    sys_msg = rendered_prompt[0]["content"]
    listed = sum(1 for i in range(15) if f"v{i} (thing)" in sys_msg)
    assert listed == 10


def test_reasoning_generator_uses_low_temperature():
    """Reasoning should be deterministic-leaning (T=0.3)."""
    model = MagicMock()
    model.generate.return_value = "ok"
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "<rendered>"
    gen = ReasoningGenerator(model=model, kg=None)
    gen.generate("?")
    call = model.generate.call_args
    assert call.kwargs.get("temperature") == 0.3


def test_reasoning_generator_falls_back_when_no_chat_template():
    """No apply_chat_template → use the raw message directly."""
    model = MagicMock()
    model.generate.return_value = "ok"
    # Tokenizer without apply_chat_template
    model.tokenizer = MagicMock(spec=[])
    gen = ReasoningGenerator(model=model, kg=None)
    out = gen.generate("question")
    assert out == "ok"
    # generate was called with the raw message (no rendered prompt)
    args, kwargs = model.generate.call_args
    assert args[0] == "question"


def test_reasoning_generator_returns_empty_on_failure():
    model = MagicMock()
    model.generate.side_effect = RuntimeError("model down")
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "<rendered>"
    gen = ReasoningGenerator(model=model, kg=None)
    assert gen.generate("?") == ""


def test_reasoning_generator_empty_kg_no_facts_hint():
    """KG with no entities → no hint surfaced."""
    model = MagicMock()
    model.generate.return_value = "ok"
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "<rendered>"
    kg = MagicMock()
    kg.entities = {}
    gen = ReasoningGenerator(model=model, kg=kg)
    gen.generate("?")
    call = model.tokenizer.apply_chat_template.call_args
    rendered_prompt = call.args[0] if call.args else call.kwargs.get("conversation")
    sys_msg = rendered_prompt[0]["content"]
    assert "Contexte mémoire" not in sys_msg
