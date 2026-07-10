"""Tests for :mod:`rune.cognition.cascade`.

The cascade is mocked end-to-end — no real Gemini API, no real
local model. We use pytest-style fakes to verify that each branch
of the orchestration logic produces the right :class:`CascadeResult`.

Branches covered
----------------

* Cascade disabled (no Gemini client) → fallback
* Gemini happy path + synthesis above threshold → synthesised result
* Gemini happy path + draft below threshold → no synthesis, ship draft
* Gemini empty response → fallback
* Gemini ``GeminiUnauthorizedError`` → fallback with ``reason="unauthorized"``
* Gemini ``GeminiQuotaExceededError`` → fallback with ``reason="quota"``
* Gemini ``GeminiTransientError`` (via base class) → fallback ``reason="network"``
* Synthesis failure → ship Gemini draft as-is, ``synthesised=False``
* Synthesis returns empty → ship Gemini draft
* Local generator missing entirely → ``fallback_reason="no_local_generator"``
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rune.cognition.cascade import (
    CascadeGenerator,
    CascadeResult,
    _approx_token_count,
)
from rune.external.gemini_client import (
    GeminiClientError,
    GeminiQuotaExceededError,
    GeminiResponse,
    GeminiUnauthorizedError,
)


# ── Fixtures ───────────────────────────────────────────────────────────


def _gemini_resp(text: str, in_tokens: int = 50, out_tokens: int = 30):
    return GeminiResponse(
        text=text,
        finish_reason="STOP",
        usage_input_tokens=in_tokens,
        usage_output_tokens=out_tokens,
        model="gemini-2.5-flash",
        raw={},
    )


def _make_gemini(text: str = "Bonjour Mika, ravi de te voir."):
    cli = MagicMock()
    cli.generate.return_value = _gemini_resp(text)
    return cli


def _make_local(text: str = "Bonjour Mika."):
    """Returns a stub callable matching the LocalGenerator protocol."""
    return MagicMock(return_value=text)


# ── Token approximation ────────────────────────────────────────────────


class TestApproxTokenCount:
    def test_empty(self):
        assert _approx_token_count("") == 0

    def test_short(self):
        # "abcd" = 4 chars → max(1, 1) = 1 token
        assert _approx_token_count("abcd") == 1

    def test_long(self):
        # 200 chars → 50 tokens approx
        assert _approx_token_count("x" * 200) == 50

    def test_min_one(self):
        # Even a single char gives at least 1 token
        assert _approx_token_count("a") == 1


# ── Cascade disabled ───────────────────────────────────────────────────


class TestCascadeDisabled:
    def test_no_gemini_falls_back_to_local(self):
        local = _make_local("local response")
        cascade = CascadeGenerator(gemini=None, local_generator=local)
        assert not cascade.is_enabled

        result = cascade.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result.fallback_used
        assert result.fallback_reason == "cascade_disabled"
        assert result.final_text == "local response"
        local.assert_called_once()

    def test_no_gemini_no_local_returns_empty(self):
        cascade = CascadeGenerator(gemini=None, local_generator=None)

        result = cascade.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result.fallback_used
        assert result.fallback_reason == "no_local_generator"
        assert result.final_text == ""


# ── Happy paths ────────────────────────────────────────────────────────


class TestHappyPath:
    def test_long_draft_gets_synthesised(self):
        # Gemini returns ~80 tokens (320 chars), threshold is 50 → synth.
        long_draft = (
            "C'est très intéressant à entendre, Mika. Aix-en-Provence "
            "est une ville magnifique avec sa lumière particulière, son "
            "histoire artistique et sa proximité avec Cézanne. "
            "J'imagine que tu y trouves une belle inspiration au quotidien."
        )
        gemini = _make_gemini(text=long_draft)
        local = _make_local("Aix est une belle ville. Tu y vis bien ?")
        cascade = CascadeGenerator(
            gemini=gemini,
            local_generator=local,
            synthesis_threshold_tokens=50,
        )

        result = cascade.generate(
            system_prompt="You are Rune.",
            messages=[{"role": "user", "content": "Je vis à Aix"}],
        )

        assert not result.fallback_used
        assert result.synthesised
        assert result.gemini_text == long_draft
        assert result.final_text == "Aix est une belle ville. Tu y vis bien ?"
        assert result.gemini_input_tokens == 50
        assert result.gemini_output_tokens == 30
        # Local was called for synthesis
        local.assert_called_once()

    def test_short_draft_skips_synthesis(self):
        short_draft = "D'accord."
        gemini = _make_gemini(text=short_draft)
        local = _make_local()
        cascade = CascadeGenerator(
            gemini=gemini,
            local_generator=local,
            synthesis_threshold_tokens=50,
        )

        result = cascade.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "ok ?"}],
        )

        assert not result.fallback_used
        assert not result.synthesised
        assert result.final_text == "D'accord."
        # Local NOT called — short answer goes through unchanged
        local.assert_not_called()

    def test_threshold_zero_always_synthesises(self):
        gemini = _make_gemini(text="court.")
        local = _make_local("plus court.")
        cascade = CascadeGenerator(
            gemini=gemini,
            local_generator=local,
            synthesis_threshold_tokens=0,
        )

        result = cascade.generate(
            system_prompt="",
            messages=[{"role": "user", "content": "x"}],
        )

        assert result.synthesised
        assert result.final_text == "plus court."

    def test_no_local_skips_synthesis(self):
        # With no local model, even a long draft ships as-is from Gemini.
        # Must end with valid sentence-final punctuation (V3.9.3 rule),
        # otherwise it would be treated as truncated and fallback.
        long = ("x" * 399) + "."  # ~100 tokens, properly terminated
        gemini = _make_gemini(text=long)
        cascade = CascadeGenerator(gemini=gemini, local_generator=None)

        result = cascade.generate(
            system_prompt="",
            messages=[{"role": "user", "content": "x"}],
        )

        assert not result.synthesised
        assert result.final_text == long


# ── Gemini failures → fallback ─────────────────────────────────────────


class TestGeminiFailures:
    def _setup(self, exc):
        gemini = MagicMock()
        gemini.generate.side_effect = exc
        local = _make_local("local fallback OK")
        return gemini, local

    def test_unauthorized_falls_back(self):
        gemini, local = self._setup(GeminiUnauthorizedError("bad key"))
        cascade = CascadeGenerator(gemini=gemini, local_generator=local)

        result = cascade.generate(
            system_prompt="sys", messages=[{"role": "user", "content": "x"}]
        )

        assert result.fallback_used
        assert result.fallback_reason == "unauthorized"
        assert result.final_text == "local fallback OK"

    def test_quota_falls_back(self):
        gemini, local = self._setup(GeminiQuotaExceededError("daily limit"))
        cascade = CascadeGenerator(gemini=gemini, local_generator=local)

        result = cascade.generate(
            system_prompt="", messages=[{"role": "user", "content": "x"}]
        )

        assert result.fallback_used
        assert result.fallback_reason == "quota"

    def test_generic_client_error_falls_back_as_network(self):
        gemini, local = self._setup(GeminiClientError("HTTP 500"))
        cascade = CascadeGenerator(gemini=gemini, local_generator=local)

        result = cascade.generate(
            system_prompt="", messages=[{"role": "user", "content": "x"}]
        )

        assert result.fallback_used
        assert result.fallback_reason == "network"

    def test_unexpected_exception_falls_back(self):
        gemini, local = self._setup(RuntimeError("something weird"))
        cascade = CascadeGenerator(gemini=gemini, local_generator=local)

        result = cascade.generate(
            system_prompt="", messages=[{"role": "user", "content": "x"}]
        )

        assert result.fallback_used
        assert result.fallback_reason == "malformed"

    def test_empty_gemini_text_falls_back(self):
        gemini = MagicMock()
        gemini.generate.return_value = _gemini_resp(text="   ")
        local = _make_local("local OK")
        cascade = CascadeGenerator(gemini=gemini, local_generator=local)

        result = cascade.generate(
            system_prompt="", messages=[{"role": "user", "content": "x"}]
        )

        assert result.fallback_used
        assert result.fallback_reason == "malformed"


# ── Synthesis failures → ship draft ────────────────────────────────────


class TestSynthesisFailures:
    def test_synthesis_exception_keeps_gemini_draft(self):
        long_draft = (
            "Voilà une réponse très détaillée et un peu trop longue, "
            "qui devrait normalement être synthétisée mais le modèle "
            "local va planter à la synthèse pour ce test. On a besoin "
            "que ce texte fasse au moins 50 tokens approximatifs pour "
            "que la synthèse soit déclenchée par le seuil par défaut."
        )
        gemini = _make_gemini(text=long_draft)
        local = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        cascade = CascadeGenerator(
            gemini=gemini,
            local_generator=local,
            synthesis_threshold_tokens=50,
        )

        result = cascade.generate(
            system_prompt="", messages=[{"role": "user", "content": "x"}]
        )

        assert not result.fallback_used  # Gemini DID succeed
        assert not result.synthesised
        assert result.final_text == long_draft
        # debug payload should record the failure reason
        assert "synthesis_failed" in result.debug

    def test_synthesis_empty_keeps_gemini_draft(self):
        long_draft = (
            "Voilà une longue réponse. " * 10
        )
        gemini = _make_gemini(text=long_draft.strip())
        local = MagicMock(return_value="")  # empty synthesis
        cascade = CascadeGenerator(
            gemini=gemini,
            local_generator=local,
            synthesis_threshold_tokens=10,
        )

        result = cascade.generate(
            system_prompt="", messages=[{"role": "user", "content": "x"}]
        )

        assert not result.fallback_used
        assert not result.synthesised
        assert result.final_text == long_draft.strip()
        assert "synthesis_empty" in result.debug


# ── Fallback edge: local also dies ─────────────────────────────────────


class TestLocalAlsoFailing:
    def test_double_failure_returns_empty_with_reason(self):
        gemini = MagicMock()
        gemini.generate.side_effect = GeminiUnauthorizedError("nope")
        local = MagicMock(side_effect=RuntimeError("local OOM"))
        cascade = CascadeGenerator(gemini=gemini, local_generator=local)

        result = cascade.generate(
            system_prompt="", messages=[{"role": "user", "content": "x"}]
        )

        assert result.fallback_used
        # The reason chains: gemini reason + "+local_error"
        assert "unauthorized" in result.fallback_reason
        assert "local_error" in result.fallback_reason
        assert result.final_text == ""


# ── Configuration ──────────────────────────────────────────────────────


class TestConfig:
    def test_is_enabled_when_both_present(self):
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
        )
        assert cascade.is_enabled

    def test_is_disabled_without_gemini(self):
        cascade = CascadeGenerator(
            gemini=None,
            local_generator=_make_local(),
        )
        assert not cascade.is_enabled

    def test_is_disabled_without_local(self):
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=None,
        )
        assert not cascade.is_enabled

    def test_min_synthesis_max_tokens_clamped(self):
        # Even if the user passes 5, we clamp to 20 (sensible floor).
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
            synthesis_max_tokens=5,
        )
        assert cascade._synthesis_max == 20

    def test_min_gemini_max_tokens_clamped(self):
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
            gemini_max_tokens=10,
        )
        assert cascade._gemini_max == 50


# ── V3.9.2 reinforcement and corrective synthesis ──────────────────────


class TestGeminiReinforcement:
    """Tests for the Gemini-side rule reinforcement preamble.

    Added in V3.9.2 after empirical observation that Gemini 2.5 Flash
    violates the anti-personification rules (10) about half the time
    on Rune's standard 4-message test sequence. The reinforcement
    states the most-violated rules first, framed as non-negotiable.
    """

    def test_reinforcement_preamble_present(self):
        """The preamble must mention the key violations Gemini commits."""
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
        )
        preamble = cascade.GEMINI_REINFORCEMENT_PREAMBLE
        # Anti-personification cues
        assert "je connais" in preamble.lower()
        assert "je visualise" in preamble.lower() or "je sais" in preamble.lower()
        # Anti-possession
        assert "mon chat" in preamble.lower() or "possession" in preamble.lower()
        # Conciseness rule
        assert "concise" in preamble.lower() or "concis" in preamble.lower()

    def test_reinforce_prepends_to_base(self):
        """The base SYSTEM_PROMPT is preserved AFTER the reinforcement."""
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
        )
        base = "Tu es Rune, etc. Règle 1: bla bla. Règle 10: bla."
        result = cascade._reinforce_for_gemini(base)
        # Base is preserved
        assert "Règle 10" in result
        # Preamble comes FIRST
        assert result.startswith(cascade.GEMINI_REINFORCEMENT_PREAMBLE)
        # And is followed by the base
        assert result.endswith(base)

    def test_reinforce_with_empty_base(self):
        """Empty base prompt — return just the reinforcement."""
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
        )
        result = cascade._reinforce_for_gemini("")
        assert result == cascade.GEMINI_REINFORCEMENT_PREAMBLE

    def test_gemini_receives_reinforced_prompt(self):
        """End-to-end: when generate() runs, the gemini.generate call
        gets the reinforced prompt, not the bare base."""
        gemini = _make_gemini(text="ok ok ok")
        cascade = CascadeGenerator(
            gemini=gemini,
            local_generator=_make_local(),
            synthesis_threshold_tokens=10,
        )
        cascade.generate(
            system_prompt="Tu es Rune.",
            messages=[{"role": "user", "content": "x"}],
        )
        # Inspect the actual call made on the Gemini client.
        call_kwargs = gemini.generate.call_args.kwargs
        sent_system = call_kwargs["system_prompt"]
        # Reinforcement preamble is present
        assert "RÈGLES CRITIQUES" in sent_system
        # Original base is also present
        assert "Tu es Rune." in sent_system


class TestCorrectiveSynthesisPrompt:
    """Tests for the V3.9.2 corrective synthesis prompt template.

    The earlier V3.9.0 template ("reformule en gardant l'essentiel")
    preserved Gemini's violations because Qwen interpreted "garder
    l'essentiel" as keeping the violating phrases. The corrective
    template explicitly lists the violations to fix.
    """

    def test_synthesis_template_lists_violations_to_fix(self):
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
        )
        tpl = cascade.SYNTHESIS_PROMPT_TEMPLATE
        # The template must mention the key violations as things to fix.
        assert "je connais" in tpl.lower()
        assert "mon chat" in tpl.lower() or "possessions inventées" in tpl.lower()
        # And it must instruct active rewriting, not paraphrasing.
        assert "réécris" in tpl.lower() or "ne paraphrase pas" in tpl.lower()

    def test_synthesis_template_demands_brevity(self):
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
        )
        tpl = cascade.SYNTHESIS_PROMPT_TEMPLATE
        # Must explicitly cap length.
        assert "1-3 phrases" in tpl or "concis" in tpl.lower()

    def test_synthesis_template_blocks_encyclopedic_drift(self):
        """The Loki/sardines turn 4 of the V3.9.0 prod test produced
        an encyclopedic comment about the name's etymology. The new
        template must explicitly forbid this kind of drift."""
        cascade = CascadeGenerator(
            gemini=_make_gemini(),
            local_generator=_make_local(),
        )
        tpl = cascade.SYNTHESIS_PROMPT_TEMPLATE
        assert (
            "encyclopédique" in tpl.lower()
            or "étymologie" in tpl.lower()
            or "non demandé" in tpl.lower()
            or "non sollicité" in tpl.lower()
            or "demandé" in tpl.lower()
        )

    def test_synthesis_passes_corrective_prompt_to_local(self):
        """The synthesis call to the local model must use the
        corrective template, not the raw Gemini draft."""
        long_draft = (
            "Je connais bien Anthropic, c'est une organisation très "
            "intéressante. Je visualise leur travail comme une révolution "
            "de l'IA. Mon expérience me dit que Rune doit être un "
            "projet fascinant qui regroupe beaucoup de talents."
        )
        gemini = _make_gemini(text=long_draft)
        local = _make_local("Anthropic, ça t'occupe sur Rune ?")
        cascade = CascadeGenerator(
            gemini=gemini,
            local_generator=local,
            synthesis_threshold_tokens=20,
        )

        cascade.generate(
            system_prompt="Tu es Rune.",
            messages=[{"role": "user", "content": "x"}],
        )

        # Inspect the call to the local synthesiser.
        local_call_args = local.call_args
        synth_messages = local_call_args.kwargs.get(
            "messages", local_call_args.args[1] if len(local_call_args.args) > 1 else []
        )
        # The user message contains the corrective template + the draft.
        synth_user = synth_messages[0]["content"]
        # The draft must be embedded.
        assert long_draft in synth_user
        # The corrective instructions must be present.
        assert "RÈGLES STRICTES" in synth_user or "réécris" in synth_user.lower()


# ── V3.9.3 truncation detection ────────────────────────────────────────


class TestTruncationDetection:
    """Tests for the V3.9.3 truncation detection logic.

    Added after empirical observation 2026-05-04: Gemini 2.5 Flash
    silently truncates responses when its internal "thinking" tokens
    consume the budget, leaving the user with mid-sentence cuts.
    """

    def test_clean_stop_is_not_truncated(self):
        text = "PLS-DA est une méthode supervisée. SIMCA est non-supervisée."
        assert not CascadeGenerator._looks_truncated(text, "STOP")

    def test_max_tokens_is_truncated(self):
        text = "PLS-DA cherche à maximiser la séparation."  # well-formed but
        assert CascadeGenerator._looks_truncated(text, "MAX_TOKENS")

    def test_safety_is_truncated(self):
        assert CascadeGenerator._looks_truncated("...", "SAFETY")

    def test_other_is_truncated(self):
        assert CascadeGenerator._looks_truncated("texte", "OTHER")

    def test_no_terminal_punctuation_is_truncated(self):
        # "SIMCA," or "et" — mid-sentence cuts
        assert CascadeGenerator._looks_truncated("...quant à elle, SIMCA", "STOP")
        assert CascadeGenerator._looks_truncated("...maximiser la séparation", "STOP")

    def test_question_mark_is_clean(self):
        assert not CascadeGenerator._looks_truncated("Comment ça va ?", "STOP")

    def test_exclamation_is_clean(self):
        assert not CascadeGenerator._looks_truncated("Bonjour Mika !", "STOP")

    def test_ellipsis_is_clean(self):
        assert not CascadeGenerator._looks_truncated("Ça m'évoque...", "STOP")

    def test_quote_is_clean(self):
        assert not CascadeGenerator._looks_truncated('Il dit "oui."', "STOP")

    def test_empty_text_is_truncated(self):
        assert CascadeGenerator._looks_truncated("", "STOP")
        assert CascadeGenerator._looks_truncated("   ", "STOP")

    def test_trailing_whitespace_ignored(self):
        # Trailing spaces shouldn't count as truncation if the actual
        # last char is fine.
        assert not CascadeGenerator._looks_truncated("Bonjour.   ", "STOP")
        # But trailing space after a comma is still truncation
        assert CascadeGenerator._looks_truncated("SIMCA,    ", "STOP")


class TestTruncatedDraftForcedThroughSynthesis:
    """Tests verifying that truncated drafts are ALWAYS sent through
    synthesis, regardless of length, to avoid shipping mid-sentence
    cuts to users.
    """

    def test_short_truncated_draft_forces_synthesis(self):
        """A 30-token draft that ends with a comma must trigger synthesis,
        even though it's below the 50-token threshold."""
        truncated = "PLS-DA est discriminante. SIMCA, quant à elle,"
        gemini = MagicMock()
        gemini.generate.return_value = GeminiResponse(
            text=truncated,
            finish_reason="MAX_TOKENS",
            usage_input_tokens=20,
            usage_output_tokens=10,
            model="gemini-2.5-flash",
            raw={},
        )
        local = MagicMock(return_value="PLS-DA est discriminante alors que SIMCA modélise chaque classe séparément.")
        cascade = CascadeGenerator(
            gemini=gemini, local_generator=local,
            synthesis_threshold_tokens=50,  # truncated draft is below this
        )

        result = cascade.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "x"}],
        )

        # Synthesis MUST have been called despite low token count
        local.assert_called_once()
        assert result.synthesised is True
        assert "SIMCA," not in result.final_text
        assert result.final_text.endswith(".")

    def test_clean_short_draft_skips_synthesis(self):
        """Symmetric check: a CLEAN short draft (ends with period) skips
        synthesis as before — we don't over-trigger."""
        clean = "D'accord."
        gemini = MagicMock()
        gemini.generate.return_value = GeminiResponse(
            text=clean,
            finish_reason="STOP",
            usage_input_tokens=20,
            usage_output_tokens=2,
            model="gemini-2.5-flash",
            raw={},
        )
        local = MagicMock(return_value="should not be called")
        cascade = CascadeGenerator(
            gemini=gemini, local_generator=local,
            synthesis_threshold_tokens=50,
        )

        result = cascade.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "x"}],
        )

        local.assert_not_called()
        assert result.synthesised is False
        assert result.final_text == clean

    def test_truncated_with_no_local_falls_back(self):
        """If draft is truncated AND no local model is available, we
        cannot complete — must fall back rather than ship the cut."""
        truncated = "PLS-DA est..."  # truncated mid-thought
        gemini = MagicMock()
        gemini.generate.return_value = GeminiResponse(
            text="PLS-DA est",  # no terminal punct — definitely truncated
            finish_reason="MAX_TOKENS",
            usage_input_tokens=20,
            usage_output_tokens=5,
            model="gemini-2.5-flash",
            raw={},
        )
        cascade = CascadeGenerator(
            gemini=gemini, local_generator=None,
            synthesis_threshold_tokens=50,
        )

        result = cascade.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "x"}],
        )

        # Without a local fallback we have no way to complete — must
        # signal fallback even though Gemini "succeeded" technically.
        # The reason chains: cascade detected truncation but local is
        # None, so _local_only returns no_local_generator.
        assert result.fallback_used is True
        assert result.fallback_reason == "no_local_generator"

    def test_finish_reason_logged_in_debug(self):
        """The debug payload should expose finish_reason so we can see
        why a draft was treated as truncated."""
        gemini = MagicMock()
        gemini.generate.return_value = GeminiResponse(
            text="Bonjour Mika.",
            finish_reason="STOP",
            usage_input_tokens=20,
            usage_output_tokens=3,
            model="gemini-2.5-flash",
            raw={},
        )
        cascade = CascadeGenerator(
            gemini=gemini, local_generator=_make_local(),
            synthesis_threshold_tokens=50,
        )
        result = cascade.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "x"}],
        )
        # Short clean draft → skip synth → debug has finish_reason
        assert result.debug.get("finish_reason") == "STOP"
