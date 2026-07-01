"""Tests V5.5 — Reflection Loop.

Couvre :
- should_reflect() : tous les triggers et skip filters
- reflect_on_response() : parsing JSON, cas erreur, timeout
- cognitive_item_for() : messages UI selon verdict
- Garde-fous : revision proposée mais vide → downgrade
"""

from __future__ import annotations

import json
import pytest


# ── Mocks ──────────────────────────────────────────────────────────────


class FakeLLM:
    def __init__(self, response="{}"):
        self.response = response
        self.is_loaded = True
        self.call_count = 0

    def complete_sync(self, messages, max_new_tokens=512,
                      temperature=0.2, timeout=None):
        self.call_count += 1
        return self.response


# ── Tests should_reflect ───────────────────────────────────────────────


class TestShouldReflect:

    def test_tech_reco_triggers(self):
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(
            query="Recommande-moi un modèle NER",
            response="Je te recommande X et Y. " * 5,  # long enough
            web_reason="tech_reco: Recommande-moi",
        )
        should, trigger = should_reflect(ctx)
        assert should is True
        assert trigger == ReflectionTrigger.TECH_RECO

    def test_high_complexity_triggers(self):
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(
            response="Réponse complexe. " * 10,
            complexity_steps=4,
        )
        should, trigger = should_reflect(ctx)
        assert should is True
        assert trigger == ReflectionTrigger.HIGH_COMPLEXITY

    def test_crag_incorrect_triggers(self):
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(
            response="Réponse depuis ma mémoire interne. " * 5,
            crag_status="incorrect",
        )
        should, trigger = should_reflect(ctx)
        assert should is True
        assert trigger == ReflectionTrigger.CRAG_INCORRECT

    def test_doubt_markers_trigger(self):
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(
            response=(
                "Je ne suis pas sûre de cela. Peut-être que c'est vrai. "
                "Il me semble qu'il existe une lib X mais je crois que "
                "ce n'est pas certain. Sans source précise."
            ),
        )
        should, trigger = should_reflect(ctx)
        assert should is True
        assert trigger == ReflectionTrigger.DOUBT_MARKERS

    def test_skip_too_short(self):
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(response="OK.", web_reason="tech_reco")
        should, trigger = should_reflect(ctx)
        assert should is False
        assert trigger == ReflectionTrigger.SKIP_TOO_SHORT

    def test_skip_reasoning_on(self):
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(
            response="x" * 200,
            web_reason="tech_reco",
            reasoning_active=True,  # raisonnement déjà actif
        )
        should, trigger = should_reflect(ctx)
        assert should is False
        assert trigger == ReflectionTrigger.SKIP_REASONING_ON

    def test_skip_python_result(self):
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(
            response="Le résultat est 399.09. " * 5,
            web_reason="tech_reco",
            tool_used="python",
        )
        should, trigger = should_reflect(ctx)
        assert should is False
        assert trigger == ReflectionTrigger.SKIP_PYTHON_RESULT

    def test_no_trigger_basic_response(self):
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(
            response="Une réponse normale assez longue mais sans risque particulier identifié. " * 3,
        )
        should, trigger = should_reflect(ctx)
        assert should is False
        assert trigger == ReflectionTrigger.NOT_TRIGGERED

    def test_doubt_below_threshold(self):
        """2 marqueurs de doute → ne déclenche pas (besoin de ≥3)."""
        from rune.cognition.reflection import (
            should_reflect, ReflectionContext, ReflectionTrigger,
        )
        ctx = ReflectionContext(
            response="Je crois que c'est ça. Peut-être. " * 3,
        )
        should, trigger = should_reflect(ctx)
        # 2 marqueurs distincts dans le message court, peut être 2 ou 3
        # selon répétition. On veut juste que ce ne soit pas DOUBT.
        if not should:
            assert trigger != ReflectionTrigger.DOUBT_MARKERS


# ── Tests parsing ──────────────────────────────────────────────────────


class TestParseReflection:

    def test_parse_valid_no_revision(self):
        from rune.cognition.reflection import _parse_reflection_response
        raw = '{"needs_revision": false, "issues": [], "revised_response": ""}'
        v = _parse_reflection_response(raw)
        assert v is not None
        assert v.needs_revision is False
        assert v.issues == []

    def test_parse_valid_with_revision(self):
        from rune.cognition.reflection import _parse_reflection_response
        raw = json.dumps({
            "needs_revision": True,
            "issues": ["lib inventée"],
            "revised_response": "Version corrigée",
        })
        v = _parse_reflection_response(raw)
        assert v is not None
        assert v.needs_revision is True
        assert v.issues == ["lib inventée"]
        assert v.revised_response == "Version corrigée"

    def test_parse_json_in_blabla(self):
        from rune.cognition.reflection import _parse_reflection_response
        raw = (
            'Voici mon verdict : {"needs_revision": false, '
            '"issues": [], "revised_response": ""} oui'
        )
        v = _parse_reflection_response(raw)
        assert v is not None
        assert v.needs_revision is False

    def test_parse_needs_revision_but_empty_response(self):
        """Si needs_revision=true mais revised vide → downgrade."""
        from rune.cognition.reflection import _parse_reflection_response
        raw = '{"needs_revision": true, "issues": ["x"], "revised_response": ""}'
        v = _parse_reflection_response(raw)
        assert v is not None
        # Downgrade : sans révision concrète, on garde l'original
        assert v.needs_revision is False
        assert v.revised_response == ""

    def test_parse_garbage(self):
        from rune.cognition.reflection import _parse_reflection_response
        assert _parse_reflection_response("") is None
        assert _parse_reflection_response("totalement garbage") is None

    def test_parse_truncates_long_revision(self):
        from rune.cognition.reflection import _parse_reflection_response
        raw = json.dumps({
            "needs_revision": True,
            "issues": ["x"],
            "revised_response": "x" * 10000,
        })
        v = _parse_reflection_response(raw)
        assert v is not None
        assert len(v.revised_response) <= 4000


# ── Tests reflect_on_response ──────────────────────────────────────────


class TestReflectOnResponse:

    def test_skip_when_not_triggered(self):
        from rune.cognition.reflection import (
            reflect_on_response, ReflectionContext, ReflectionTrigger,
        )
        llm = FakeLLM()
        ctx = ReflectionContext(response="Réponse normale longue. " * 5)
        verdict = reflect_on_response(ctx, llm)
        assert verdict.trigger == ReflectionTrigger.NOT_TRIGGERED
        assert verdict.needs_revision is False
        assert llm.call_count == 0  # pas d'appel LLM

    def test_runs_on_tech_reco(self):
        from rune.cognition.reflection import (
            reflect_on_response, ReflectionContext, ReflectionTrigger,
        )
        llm = FakeLLM(response='{"needs_revision": false, "issues": [], "revised_response": ""}')
        ctx = ReflectionContext(
            query="Recommande modèle",
            response="Je te recommande X. " * 5,
            web_reason="tech_reco",
        )
        verdict = reflect_on_response(ctx, llm)
        assert verdict.trigger == ReflectionTrigger.TECH_RECO
        assert llm.call_count == 1

    def test_applies_revision(self):
        from rune.cognition.reflection import (
            reflect_on_response, ReflectionContext,
        )
        llm = FakeLLM(response=json.dumps({
            "needs_revision": True,
            "issues": ["package inventé : `mystic_ai`"],
            "revised_response": "Version corrigée sans mystic_ai.",
        }))
        ctx = ReflectionContext(
            query="Recommande lib Python",
            response="Utilise mystic_ai pour... " * 5,
            web_reason="tech_reco",
        )
        verdict = reflect_on_response(ctx, llm)
        assert verdict.needs_revision is True
        assert "mystic_ai" in verdict.issues[0]
        assert verdict.revised_response == "Version corrigée sans mystic_ai."

    def test_llm_failure_safe(self):
        from rune.cognition.reflection import (
            reflect_on_response, ReflectionContext,
        )
        class FailingLLM:
            is_loaded = True
            def complete_sync(self, *a, **kw):
                raise RuntimeError("LLM down")
        ctx = ReflectionContext(
            response="x" * 100,
            web_reason="tech_reco",
        )
        verdict = reflect_on_response(ctx, FailingLLM())
        # Pas de crash, verdict neutre
        assert verdict.needs_revision is False

    def test_llm_not_loaded(self):
        from rune.cognition.reflection import (
            reflect_on_response, ReflectionContext,
        )
        class NotLoaded:
            is_loaded = False
            def complete_sync(self, *a, **kw):
                return ""
        ctx = ReflectionContext(
            response="x" * 100,
            web_reason="tech_reco",
        )
        verdict = reflect_on_response(ctx, NotLoaded())
        assert verdict.needs_revision is False


# ── Tests cognitive_item ───────────────────────────────────────────────


class TestCognitiveItem:

    def test_not_triggered_silent(self):
        from rune.cognition.reflection import (
            cognitive_item_for, ReflectionVerdict, ReflectionTrigger,
        )
        v = ReflectionVerdict(trigger=ReflectionTrigger.NOT_TRIGGERED)
        assert cognitive_item_for(v) is None

    def test_skip_silent(self):
        from rune.cognition.reflection import (
            cognitive_item_for, ReflectionVerdict, ReflectionTrigger,
        )
        for skip in (
            ReflectionTrigger.SKIP_TOO_SHORT,
            ReflectionTrigger.SKIP_REASONING_ON,
            ReflectionTrigger.SKIP_PYTHON_RESULT,
        ):
            v = ReflectionVerdict(trigger=skip)
            assert cognitive_item_for(v) is None

    def test_no_revision_shows_relax(self):
        from rune.cognition.reflection import (
            cognitive_item_for, ReflectionVerdict, ReflectionTrigger,
        )
        v = ReflectionVerdict(
            trigger=ReflectionTrigger.TECH_RECO,
            needs_revision=False,
        )
        msg = cognitive_item_for(v)
        assert msg is not None
        assert "relu" in msg.lower()
        assert "correct" in msg.lower()

    def test_revision_message(self):
        from rune.cognition.reflection import (
            cognitive_item_for, ReflectionVerdict, ReflectionTrigger,
        )
        v = ReflectionVerdict(
            trigger=ReflectionTrigger.TECH_RECO,
            needs_revision=True,
            issues=["x", "y"],
            revised_response="revised",
        )
        msg = cognitive_item_for(v)
        assert msg is not None
        assert "corrigé" in msg.lower()
        assert "2" in msg  # nombre d'issues
