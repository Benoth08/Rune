"""Cascade generation — draft with Gemini, refine with local model.

This module implements the V3.9 cascade pattern: a remote frontier
model (Gemini Flash) produces a rich first draft, then the locally
loaded model (whatever the user picked in the UI) compresses that
draft to Lythéa's concise voice. The output of the local model is
what feeds SDM/MHN/KG, so the cognitive pipeline keeps its grip on
the latents — Gemini stays an "expert consultant" that never touches
memory directly.

Why a cascade rather than Gemini-only?
--------------------------------------

Three reasons. First, the SDM/MHN hooks live on the *local* model's
last decoder layer; if Gemini speaks the final word, those hooks see
nothing and consolidation breaks. Second, Lythéa has a distinctive
voice (concise, sensible, French) that's been validated 5/5 on
Qwen2.5-3B; we don't want Gemini's house style overriding that on
every turn. Third, the local model acts as a free safety net — when
Gemini is unreachable or rate-limited, we degrade gracefully to the
plain V3 path with no user-visible failure.

Position in the cognitive cycle
-------------------------------

Cascade replaces the call to ``model.generate`` inside Phase B
(:func:`hippocampe._phase_b_rag`). The assembled context is passed
unchanged. Phase A (encoding, surprise, KG/SDM updates) and Phase C
(consolidation, microsleep) are unaffected.

::

    encoding → retrieval → ASSEMBLE CONTEXT → cascade.generate → consolidation

Failure modes
-------------

The cascade NEVER raises out of :meth:`CascadeGenerator.generate`.
Any error (Gemini unauthorised, quota, network, malformed response)
is caught and the local-only fallback runs instead. The caller can
inspect :attr:`CascadeResult.fallback_used` if it cares — but the
conversation always continues.

Synthesis policy
----------------

Set by ``cascade_synthesis_threshold_tokens``. Below that, Gemini's
output is short enough to ship as-is — no synthesis needed. Above,
the local model is asked to compress while preserving the gist.
The default is 50 tokens which is roughly "anything longer than two
short sentences gets synthesised".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from rune.external.gemini_client import (
    GeminiClient,
    GeminiClientError,
    GeminiQuotaExceededError,
    GeminiUnauthorizedError,
)

log = logging.getLogger(__name__)


# ── Local generator protocol ───────────────────────────────────────────


class LocalGenerator(Protocol):
    """Minimal interface the cascade needs from the local model.

    Lythéa's main model wrapper exposes ``stream_generate`` for the
    standard pipeline; we wrap that as a synchronous ``generate`` here
    so the cascade can stay sync and decoupled from streaming.

    The actual implementation lives in :mod:`rune.model`; this
    Protocol is kept here so tests can pass simple mocks without
    importing torch.
    """

    def generate_for_cascade(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        """Return a single string (no streaming)."""


# ── Token counting (cheap heuristic, no tokenizer) ─────────────────────


def _approx_token_count(text: str) -> int:
    """Approximate token count without loading a tokenizer.

    Heuristic: 1 token ≈ 4 characters for European languages. Used
    only to decide whether synthesis is worthwhile — exact counts
    are not required. The real billing tokens come back from the
    Gemini ``usageMetadata`` after the call.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Result dataclasses ─────────────────────────────────────────────────


@dataclass
class CascadeResult:
    """Outcome of a single :meth:`CascadeGenerator.generate` call.

    Even when the cascade falls back to local-only, the result has a
    ``final_text`` and ``fallback_used=True``. Callers should always
    use ``final_text`` and ignore the rest if they don't need debug.
    """

    final_text: str
    """The text shown to the user. Always populated."""

    gemini_text: str = ""
    """The raw Gemini draft, if it was obtained. Empty on fallback."""

    synthesised: bool = False
    """Whether the local model rewrote the Gemini draft."""

    fallback_used: bool = False
    """True if Gemini failed and we generated locally only."""

    fallback_reason: str = ""
    """Short tag explaining why we fell back: ``unauthorized``,
    ``quota``, ``network``, ``malformed``, ``no_local_generator``,
    or empty when no fallback occurred."""

    gemini_input_tokens: int = 0
    """From Gemini's usageMetadata — 0 on fallback."""

    gemini_output_tokens: int = 0
    """From Gemini's usageMetadata — 0 on fallback."""

    debug: dict[str, Any] = field(default_factory=dict)
    """Free-form debug info for the UI panel (latency_ms,
    model versions, …). No PII or API keys."""


# ── The orchestrator ───────────────────────────────────────────────────


class CascadeGenerator:
    """Orchestrate the draft-then-refine flow.

    Parameters
    ----------
    gemini:
        A configured :class:`GeminiClient`. May be ``None`` to disable
        the cascade entirely (in which case every call falls back).
    local_generator:
        A callable matching :class:`LocalGenerator` for the synthesis
        step and the ultimate fallback. May be ``None`` only in tests
        that exercise the cascade-disabled path.
    synthesis_threshold_tokens:
        Skip synthesis when Gemini's draft is shorter than this. Use
        0 to always synthesise. Default 50.
    synthesis_max_tokens:
        Cap on the synthesised output length. Default 120.
    gemini_max_tokens:
        Cap on Gemini's draft. Default 800.
    gemini_temperature:
        Sampling temperature for the Gemini draft. Default 0.7.
    """

    SYNTHESIS_PROMPT_TEMPLATE = (
        "Tu es Rune. Le texte ci-dessous est un brouillon écrit par "
        "un autre assistant — tu dois le RÉÉCRIRE entièrement dans ton "
        "propre style avant de le transmettre. NE PARAPHRASE PAS, "
        "RÉÉCRIS comme si tu répondais directement.\n"
        "\n"
        "RÈGLES STRICTES — tu dois CORRIGER ces violations si présentes "
        "dans le brouillon :\n"
        "1. Pas de \"je connais\", \"je sais\", \"je suis familière avec\". "
        "Tu n'as pas de connaissances vécues, juste de la mémoire textuelle.\n"
        "2. Pas de \"je visualise\", \"j'imagine\" sur des lieux ou objets "
        "(c'est une forme de personnification de l'expérience). "
        "Préfère \"ça évoque\", \"ça me semble\".\n"
        "3. Pas de possessions inventées : jamais \"mon chat\", \"ma "
        "ville\", \"mon expérience\". Tu n'as ni l'un ni l'autre.\n"
        "4. Pas de commentaire encyclopédique sur les noms, étymologies, "
        "ou faits culturels que l'utilisateur n'a pas demandés. "
        "Reste dans la conversation présente.\n"
        "5. Pas de faits inventés. Si le brouillon affirme quelque chose "
        "qui n'a pas été dit par l'utilisateur, retire-le.\n"
        "6. Concis : 1-3 phrases courtes maximum. Si le brouillon est "
        "verbeux, condense-le radicalement.\n"
        "7. Tutoie l'utilisateur, parle à la première personne (\"je\").\n"
        "8. Termine par une vraie question ou observation conversationnelle, "
        "pas par une formule générique d'aide.\n"
        "\n"
        "Réponds UNIQUEMENT par le texte réécrit, sans préface, sans "
        "guillemets, sans \"voici la version reformulée\". Pars directement.\n"
        "\n"
        "Brouillon à réécrire :\n"
        "{text}\n"
        "\n"
        "Ta réponse réécrite :"
    )

    def __init__(
        self,
        gemini: GeminiClient | None,
        local_generator: Callable[..., str] | None,
        synthesis_threshold_tokens: int = 50,
        synthesis_max_tokens: int = 120,
        gemini_max_tokens: int = 800,
        gemini_temperature: float = 0.7,
    ) -> None:
        self._gemini = gemini
        self._local = local_generator
        self._synthesis_threshold = max(0, int(synthesis_threshold_tokens))
        self._synthesis_max = max(20, int(synthesis_max_tokens))
        self._gemini_max = max(50, int(gemini_max_tokens))
        self._gemini_temp = float(gemini_temperature)

    @property
    def is_enabled(self) -> bool:
        """``True`` if both Gemini and the local generator are wired up."""
        return self._gemini is not None and self._local is not None

    # ── Main entry point ───────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> CascadeResult:
        """Run the cascade. Always returns a :class:`CascadeResult`.

        On any failure path we either fall back to the local model
        (preferred) or — if the local model is also unavailable —
        return an empty result with ``fallback_reason="no_local_generator"``.
        """
        # If cascade is disabled (Gemini missing), skip straight to
        # local-only, signalling fallback.
        if self._gemini is None:
            return self._local_only(
                system_prompt=system_prompt,
                messages=messages,
                reason="cascade_disabled",
            )

        # 1. Draft with Gemini.
        # The Gemini SDK gets the same system_prompt as the local model,
        # plus a short prepended reinforcement of the rules that Gemini
        # tends to ignore. Empirically observed in the V3.9 prod test
        # (2026-05-04) on Mika's setup: Gemini 2.5 Flash violates the
        # anti-personification rule about 50% of the time on
        # introductory turns ("je connais bien Anthropic", "je
        # visualise la ville"). A Gemini-side reinforcement reduces
        # that to ~10%, and the local synthesis pass cleans up the
        # rest.
        gemini_system = self._reinforce_for_gemini(system_prompt)
        try:
            gemini_resp = self._gemini.generate(
                system_prompt=gemini_system,
                messages=messages,
                max_tokens=self._gemini_max,
                temperature=self._gemini_temp,
            )
        except GeminiUnauthorizedError as exc:
            log.warning("Cascade fallback (unauthorized): %s", exc)
            return self._local_only(
                system_prompt=system_prompt, messages=messages,
                reason="unauthorized",
            )
        except GeminiQuotaExceededError as exc:
            log.warning("Cascade fallback (quota): %s", exc)
            return self._local_only(
                system_prompt=system_prompt, messages=messages,
                reason="quota",
            )
        except GeminiClientError as exc:
            log.warning("Cascade fallback (client error): %s", exc)
            return self._local_only(
                system_prompt=system_prompt, messages=messages,
                reason="network",
            )
        except Exception as exc:  # pragma: no cover  - defensive
            log.exception("Cascade fallback (unexpected): %s", exc)
            return self._local_only(
                system_prompt=system_prompt, messages=messages,
                reason="malformed",
            )

        gemini_text = (gemini_resp.text or "").strip()
        if not gemini_text:
            log.warning("Gemini returned empty text — falling back local")
            return self._local_only(
                system_prompt=system_prompt, messages=messages,
                reason="malformed",
            )

        # Truncation detection (added V3.9.3 after prod observation
        # 2026-05-04: Gemini 2.5 Flash silently truncates responses
        # when its internal "thinking" tokens consume the budget,
        # leaving the user with mid-sentence cuts like "...SIMCA," or
        # "...quant à elle, ").
        #
        # Three signals indicate truncation:
        #   1. finish_reason != "STOP" (canonical clean stop)
        #      Common alternatives: MAX_TOKENS, SAFETY, OTHER.
        #   2. The text ends without a sentence-final punctuation mark.
        #      We accept '.', '!', '?', ':', ')', '"', '»', '…' as valid
        #      stops; anything else (comma, semicolon, dash, no punct)
        #      is suspicious.
        #   3. The text ends mid-word (no punctuation AND short).
        #
        # Truncated drafts are forced through the synthesis pass even
        # when below the threshold — Qwen will rewrite them as a
        # complete answer using its own knowledge plus the draft as
        # context. This is far better than shipping a half-cut sentence.
        finish_reason = (gemini_resp.finish_reason or "").upper()
        is_truncated = self._looks_truncated(gemini_text, finish_reason)
        if is_truncated:
            log.warning(
                "Gemini draft looks truncated (reason=%s, ends with %r) "
                "— forcing synthesis to complete it",
                finish_reason, gemini_text[-15:],
            )

        # 2. Decide whether to synthesise.
        approx = _approx_token_count(gemini_text)
        skip_synthesis = (
            (approx < self._synthesis_threshold or self._local is None)
            and not is_truncated  # Truncated drafts MUST go through synth
        )
        if skip_synthesis:
            # Short answer or no local model → ship Gemini's draft as-is.
            return CascadeResult(
                final_text=gemini_text,
                gemini_text=gemini_text,
                synthesised=False,
                fallback_used=False,
                gemini_input_tokens=gemini_resp.usage_input_tokens,
                gemini_output_tokens=gemini_resp.usage_output_tokens,
                debug={
                    "approx_tokens": approx,
                    "synthesis_skipped": True,
                    "model": gemini_resp.model,
                    "finish_reason": finish_reason,
                },
            )

        # If truncated and no local model available, we have no way to
        # complete — surface the truncation to the user via fallback.
        if is_truncated and self._local is None:
            return self._local_only(
                system_prompt=system_prompt, messages=messages,
                reason="malformed",
            )

        # 3. Synthesise with the local model.
        synth_prompt = self.SYNTHESIS_PROMPT_TEMPLATE.format(text=gemini_text)
        try:
            synth_text = self._local(
                system_prompt="",  # synthesis is self-contained
                messages=[{"role": "user", "content": synth_prompt}],
                max_tokens=self._synthesis_max,
            )
        except Exception as exc:
            log.warning(
                "Synthesis failed (%s) — shipping Gemini draft as-is", exc
            )
            return CascadeResult(
                final_text=gemini_text,
                gemini_text=gemini_text,
                synthesised=False,
                fallback_used=False,
                gemini_input_tokens=gemini_resp.usage_input_tokens,
                gemini_output_tokens=gemini_resp.usage_output_tokens,
                debug={
                    "approx_tokens": approx,
                    "synthesis_failed": str(exc)[:120],
                    "model": gemini_resp.model,
                },
            )

        synth_text = (synth_text or "").strip()
        if not synth_text:
            # Empty synthesis is suspicious — ship Gemini's draft.
            return CascadeResult(
                final_text=gemini_text,
                gemini_text=gemini_text,
                synthesised=False,
                fallback_used=False,
                gemini_input_tokens=gemini_resp.usage_input_tokens,
                gemini_output_tokens=gemini_resp.usage_output_tokens,
                debug={
                    "approx_tokens": approx,
                    "synthesis_empty": True,
                    "model": gemini_resp.model,
                },
            )

        return CascadeResult(
            final_text=synth_text,
            gemini_text=gemini_text,
            synthesised=True,
            fallback_used=False,
            gemini_input_tokens=gemini_resp.usage_input_tokens,
            gemini_output_tokens=gemini_resp.usage_output_tokens,
            debug={
                "approx_tokens": approx,
                "synthesis_skipped": False,
                "model": gemini_resp.model,
                "draft_len": len(gemini_text),
                "final_len": len(synth_text),
            },
        )

    # ── Gemini-specific reinforcement ──────────────────────────────────

    # Sentence-final punctuation marks accepted as valid stops. Anything
    # else at the end of a Gemini draft is suspicious.
    _SENTENCE_FINAL_CHARS = frozenset(".!?:)\"»…”’")

    @classmethod
    def _looks_truncated(cls, text: str, finish_reason: str) -> bool:
        """Heuristic: did Gemini cut its draft mid-thought?

        Returns True when ANY of:
          * ``finish_reason`` is not ``STOP`` (the canonical clean exit).
            ``MAX_TOKENS``, ``SAFETY``, ``OTHER``, ``RECITATION`` all
            mean the draft did not complete naturally.
          * The text ends WITHOUT a sentence-final punctuation mark.
          * The text ends with a comma + space (mid-list / mid-clause).

        False positives are acceptable: forcing a healthy draft through
        the synthesiser only adds latency, it doesn't break anything.
        False negatives (truncation we miss) are the real cost — they
        ship mid-sentence cuts to the user.
        """
        if finish_reason and finish_reason != "STOP":
            return True
        stripped = text.rstrip()
        if not stripped:
            return True
        last_char = stripped[-1]
        if last_char not in cls._SENTENCE_FINAL_CHARS:
            return True
        return False

    GEMINI_REINFORCEMENT_PREAMBLE = (
        "RÈGLES CRITIQUES — tu DOIS les respecter strictement, elles "
        "priment sur tout autre instinct conversationnel :\n"
        "\n"
        "  • N'utilise JAMAIS \"je connais\", \"je sais\", \"je suis "
        "familière avec\", \"j'ai entendu parler\". Tu n'as aucune "
        "expérience vécue, aucune connaissance du monde au-delà du "
        "contexte qui t'est fourni. Préfère \"ça évoque\", \"il me "
        "semble\", \"d'après ce que tu décris\".\n"
        "  • N'utilise JAMAIS \"je visualise\", \"j'imagine\" sur des "
        "lieux, objets ou personnes. Tu n'as pas d'imagination visuelle. "
        "Préfère un commentaire neutre ou une question sur l'expérience "
        "de l'utilisateur.\n"
        "  • Ne t'inventes AUCUNE possession (chat, ville, projet, "
        "expérience). Tu n'en as aucune. Si l'utilisateur parle de son "
        "chat, parle de \"ton chat\", jamais \"mon chat\".\n"
        "  • Ne fais AUCUN commentaire encyclopédique non sollicité "
        "(étymologie d'un nom, faits culturels, anecdotes générales). "
        "Reste dans la conversation présente.\n"
        "  • Sois CONCISE. Maximum 1-3 phrases courtes par tour. "
        "Évite les listes d'observations, les remplissages, les "
        "phrases du type \"c'est un projet passionnant\" ou \"c'est "
        "une ville charmante\" sans contenu réel.\n"
        "  • Tutoie l'utilisateur. Parle à la première personne mais "
        "sans inventer d'expérience subjective.\n"
        "\n"
        "Ces règles ne sont pas négociables — si tu les enfreints, le "
        "texte sera réécrit par un autre modèle qui les corrigera, mais "
        "respecte-les directement pour qu'on n'ait pas à le faire.\n"
        "\n"
        "─────────────────────────────────\n"
        "\n"
    )

    def _reinforce_for_gemini(self, base_system_prompt: str) -> str:
        """Prepend Gemini-specific rule reinforcement to the base prompt.

        Lythéa's ``SYSTEM_PROMPT`` already contains the 10 rules but
        Gemini 2.5 Flash treats them as soft suggestions in
        conversational settings. By stating the most-violated rules
        FIRST and explicitly, with the framing "they are non-negotiable",
        Gemini's compliance jumps from ~50% to ~90% in our V3.9 prod
        observations.

        The base prompt is appended unchanged after the preamble so the
        full context (memory blocks, identity, temporal info) reaches
        Gemini intact.
        """
        if not base_system_prompt:
            return self.GEMINI_REINFORCEMENT_PREAMBLE
        return self.GEMINI_REINFORCEMENT_PREAMBLE + base_system_prompt

    # ── Fallback ───────────────────────────────────────────────────────

    def _local_only(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        reason: str,
    ) -> CascadeResult:
        """Generate with the local model only, signalling fallback."""
        if self._local is None:
            return CascadeResult(
                final_text="",
                fallback_used=True,
                fallback_reason="no_local_generator",
                debug={"requested_reason": reason},
            )
        try:
            local_text = self._local(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=self._gemini_max,
            )
        except Exception as exc:
            log.exception("Local fallback ALSO failed: %s", exc)
            return CascadeResult(
                final_text="",
                fallback_used=True,
                fallback_reason=f"{reason}+local_error",
                debug={"local_error": str(exc)[:120]},
            )

        return CascadeResult(
            final_text=(local_text or "").strip(),
            gemini_text="",
            synthesised=False,
            fallback_used=True,
            fallback_reason=reason,
        )
