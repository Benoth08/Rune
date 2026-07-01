"""Generation helpers — output cleanup + two-pass reasoning prompt.

Not a phase per se. The orchestrator uses these helpers around the
streaming-generation step (Phase D in the original code), but they
are pure functions / single-responsibility classes that have no
business living inside Hippocampe itself. Putting them here keeps
the orchestrator focused on composition.

What lives here
---------------
* :data:`QUESTION_STARTS` — French + English question-word stop
  list. The orchestrator uses it to decide whether to archive a
  given exchange (pure questions are not archived).
* :func:`strip_reasoning` — split ``<reflexion>...</reflexion>``
  blocks (and the equivalent English tags) out of model output,
  plus drop two-pass artifacts ("D'accord, voici ma réponse :").
* :func:`mask_open_tags` — streaming-time helper: hide the
  half-rendered text starting at an *unclosed* reasoning tag.
* :class:`ReasoningGenerator` — the two-pass-reasoning pre-prompt
  used for non-thinking models. Builds a small KG snippet from
  the entity store and queries the LLM for a structured analysis
  before the main pass.

Design notes
------------
- The text cleanup functions are stateless and live as plain
  functions, not methods. They are imported by the orchestrator
  and could be reused by debug tools.
- :class:`ReasoningGenerator` is a class only because it carries
  the LLM and KG references. It owns no state of its own.
- Tag matching is **case-insensitive** so models that capitalise
  ``<Reflexion>`` are still handled.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from rune.config import THINKING_TAGS

log = logging.getLogger("lythea.cognition.generation")


# Question-starts (FR + EN). Used by the orchestrator's archive
# gate: pure questions are not committed to long-term memory
# because they don't contain new information about the user.
QUESTION_STARTS: frozenset[str] = frozenset({
    "quel", "quelle", "quels", "quelles", "comment", "pourquoi",
    "qui", "quand", "combien",
    "what", "how", "why", "who", "when", "where", "which",
    "is", "are", "do", "does", "can",
})

# Two-pass artifact prefixes. Some non-thinking models, when prompted
# in two passes (reasoning + answer), prepend their final answer with
# a meta line. We strip these at the start of the cleaned text only —
# never mid-stream, to avoid corrupting legitimate occurrences inside
# the response.
_TWO_PASS_PREFIXES: tuple[str, ...] = (
    "D'accord, voici ma réponse finale :",
    "D'accord, voici ma réponse finale:",
    "Voici ma réponse finale :",
    "Voici ma réponse finale:",
    "D'accord, voici ma réponse :",
    "Bien sûr, voici ma réponse finale :",
    "Bien sûr, voici ma réponse :",
)

# Reasoning prompt budget. The hard token ceiling and char clip now
# live in settings.py (``reasoning_simple_max_tokens``,
# ``reasoning_text_max_chars``) so they can be tuned without touching
# code. These module-level names are kept as lazy fallbacks only —
# ReasoningGenerator reads the live settings at call time.
_REASONING_TEMPERATURE: float = 0.3
# How many KG entities to surface in the reasoning prompt's hint.
# More than ~10 starts to crowd out the actual reasoning instruction.
_REASONING_KG_FACT_LIMIT: int = 10


# ── Public functions ──────────────────────────────────────────────────


def strip_reasoning(text: str, allow_unclosed: bool = False) -> tuple[str, str]:
    """Strip reasoning/thinking blocks and two-pass artifacts from text.

    Reasoning is captured in dedicated tags (configured via
    :data:`lythea.config.THINKING_TAGS`, typically ``reflexion`` and
    ``thinking``). The model produces them inline with the response;
    the UI separates them so the user sees only the final answer
    by default.

    Parameters
    ----------
    text
        Raw model output, possibly mid-stream.
    allow_unclosed
        If ``True``, also handle the streaming case for thinking
        models (Qwen3, QwQ, etc.):

        1. **Unclosed tag at tail** — content from ``<think>`` to
           end-of-text is treated as in-progress reasoning.
        2. **Implicit opening** — Qwen3 with ``enable_thinking=True``
           has the ``<think>`` opening consumed by the tokenizer,
           so the stream starts *inside* the think block. We detect
           a ``</think>`` without a matching opening, and treat
           everything before it as reasoning (implicit open).
        3. **Pure reasoning** — if neither open nor close tag has
           been seen yet, but we know this is a thinking model
           (caller passed ``allow_unclosed=True``), treat the entire
           text as in-progress reasoning. Without this, the UI would
           flash the model's internal monologue before the answer.

        For classic LLMs (default ``False``), none of these apply
        and a literal ``<think>`` in the output is left intact — it
        could be the model writing about HTML tags or AI internals.

    Returns
    -------
    tuple[str, str]
        ``(clean_response, reasoning_content)``.
    """
    reasoning_parts: list[str] = []
    clean = text

    # Step 1 — extract CLOSED tags (both modes). Definitive reasoning.
    for tag in THINKING_TAGS:
        closed = re.compile(
            rf"<{tag}>(.*?)</{tag}>",
            re.DOTALL | re.IGNORECASE,
        )
        for match in closed.finditer(clean):
            reasoning_parts.append(match.group(1).strip())
        clean = closed.sub("", clean)

    if allow_unclosed:
        lower = clean.lower()

        # Step 2 — implicit open: a </think> without matching <think>
        # before it. The tokenizer consumed the opening tag (Qwen3
        # with enable_thinking=True). Everything before </think> is
        # reasoning, everything after is the clean response.
        for tag in THINKING_TAGS:
            close_tag = f"</{tag}>"
            open_tag = f"<{tag}>"
            close_idx = lower.find(close_tag)
            if close_idx == -1:
                continue
            # Is there a matching <tag> before this </tag>?
            open_idx = lower.rfind(open_tag, 0, close_idx)
            if open_idx != -1:
                continue  # Already handled by step 1 if closed-pair
            # Implicit open: text[:close_idx] is reasoning.
            implicit_reasoning = clean[:close_idx].strip()
            if implicit_reasoning:
                reasoning_parts.append(implicit_reasoning)
            clean = clean[close_idx + len(close_tag):].lstrip("\n ")
            lower = clean.lower()
            break  # one implicit close per pass

        # Step 3 — unclosed open at tail: <think> without </think>.
        # Standard streaming case.
        for tag in THINKING_TAGS:
            open_tag = f"<{tag}>"
            idx = lower.rfind(open_tag)
            if idx == -1:
                continue
            partial = clean[idx + len(open_tag):].strip()
            if partial:
                reasoning_parts.append(partial)
            clean = clean[:idx]
            lower = clean.lower()

        # NOTE — V4.4 : on avait initialement un "Step 4" qui prenait
        # TOUT le texte comme reasoning si aucun tag (ouvert ou fermé)
        # n'était détecté pendant le streaming. L'idée était que pour
        # un thinking model qui n'a pas encore généré </think>, tout
        # le contenu est du raisonnement implicite. MAIS : si Qwen3
        # juge la question simple et répond directement (sans aucun
        # tag de raisonnement), Step 4 reclassait toute la réponse
        # comme reasoning → un bloc "Réflexion de Rune" apparaissait
        # EN BAS, dupliquant le contenu déjà affiché comme réponse.
        # Step 4 supprimé : sans tag visible, on ne peut pas savoir
        # si on est en raisonnement ou en réponse directe → on traite
        # comme réponse par défaut (cas safe). Le bloc Réflexion
        # n'apparaîtra qu'après détection réelle de </think> (Step 2)
        # ou <think>... (Step 3).

    # Strip two-pass reasoning artifacts at the very start only.
    for prefix in _TWO_PASS_PREFIXES:
        if clean.lower().startswith(prefix.lower()):
            clean = clean[len(prefix):].strip()
            break

    clean = clean.strip()
    reasoning = "\n".join(reasoning_parts).strip()
    return clean, reasoning


def mask_open_tags(text: str) -> str:
    """Remove text from an unclosed reasoning tag to the end.

    Handles the streaming case where ``<reflexion>partial...`` is
    still being generated. Everything from the opening tag onward
    is hidden so the UI does not flash partial reasoning to the
    user. When the tag closes, :func:`strip_reasoning` takes over.
    """
    lower = text.lower()
    for tag in THINKING_TAGS:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        last_open = lower.rfind(open_tag)
        if last_open == -1:
            continue
        last_close = lower.rfind(close_tag)
        if last_close < last_open:
            # Unclosed tag — truncate from opening tag.
            return text[:last_open].strip()
    return text


# ── Reasoning generator (two-pass) ─────────────────────────────────────


class ReasoningGenerator:
    """Two-pass reasoning for non-thinking models.

    Some models (Qwen-Instruct, Llama-Instruct, ...) don't natively
    emit chain-of-thought. To get a structured analysis pass, we
    run a *separate* short LLM call with a dedicated system prompt
    that asks for reasoning *only*, then prepend the result inside
    a ``<reflexion>`` tag for the main pass.

    Parameters
    ----------
    model
        :class:`HFModelWrapper`. Must expose ``generate`` and a
        ``tokenizer`` with optional ``apply_chat_template``.
    kg
        :class:`KnowledgeGraphStore` or ``None``. Read for a
        small entity hint surfaced inside the reasoning system
        prompt — the model needs to know *who* the user is to
        reason coherently.
    """

    def __init__(self, model: Any, kg: Any | None) -> None:
        self.model = model
        self.kg = kg

    def generate(self, message: str) -> str:
        """Return a reasoning chunk, or ``""`` on failure.

        The reasoning is plain text — the orchestrator wraps it in
        a ``<reflexion>`` tag before the main generation pass.

        Token budget and char clip are read live from settings, so
        they can be tuned without a code change. The prompt also
        carries a soft length hint calibrated *below* the hard token
        ceiling, giving the model room to finish its sentence before
        hitting the wall.
        """
        from rune.settings import get_settings
        s = get_settings()
        max_tokens = s.reasoning_simple_max_tokens
        max_chars = s.reasoning_text_max_chars
        # Soft word hint ≈ 80% of the token budget, converted to words
        # (~0.55 word per token in French). Keeps the model from
        # racing to the ceiling.
        word_hint = int(max_tokens * 0.8 * 0.55)

        prompt = self._build_prompt(message, word_hint)
        try:
            tokenizer = getattr(self.model, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True,
                )
            else:
                rendered = message
            reasoning = self.model.generate(
                rendered,
                max_new_tokens=max_tokens,
                temperature=_REASONING_TEMPERATURE,
            )
            return reasoning.strip()[:max_chars]
        except Exception as exc:
            log.warning("Reasoning pass failed: %s", exc)
            return ""

    def _build_prompt(self, message: str, word_hint: int = 0) -> list[dict[str, str]]:
        """Build the (system, user) pair for the reasoning pass.

        The system prompt is bilingual-flavoured French. The KG
        snippet, when present, anchors the reasoning in the actual
        user — without it, the model tends to "imagine" a generic
        interlocutor.

        ``word_hint`` (when > 0) adds a soft length target to the
        system prompt. It is calibrated below the hard token ceiling
        by the caller so the model finishes its sentence rather than
        being cut mid-word.
        """
        kg_context = self._kg_facts_hint()
        length_hint = ""
        if word_hint > 0:
            length_hint = (
                f" Vise une analyse d'environ {word_hint} mots — "
                "assez pour être complet, sans te perdre en longueurs."
            )
        return [
            {
                "role": "system",
                "content": (
                    "Tu es Rune, une IA cognitive. "
                    "L'utilisateur te pose une question. "
                    "Analyse-la étape par étape : vérifie ta logique, "
                    "identifie ce que tu sais et ce que tu ne sais pas, "
                    "et prépare un raisonnement structuré. "
                    "Utilise le contexte mémoire si disponible — "
                    "il décrit TON INTERLOCUTEUR (l'utilisateur), PAS toi. "
                    "Réponds UNIQUEMENT avec ton raisonnement, pas la réponse finale."
                    + length_hint
                    + kg_context
                ),
            },
            {"role": "user", "content": message},
        ]

    def _kg_facts_hint(self) -> str:
        """Render up to N entities as a short hint for the reasoning prompt."""
        if not self.kg or not getattr(self.kg, "entities", None):
            return ""
        facts: list[str] = []
        for ent in self.kg.entities.values():
            facts.append(f"{ent.value} ({ent.type})")
        if not facts:
            return ""
        return (
            "\n\nContexte mémoire (informations sur l'utilisateur) : "
            + ", ".join(facts[:_REASONING_KG_FACT_LIMIT])
        )
