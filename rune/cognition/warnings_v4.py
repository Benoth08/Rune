"""V4.0.2 — Generic warnings format for cognition modules.

Convention V4
-------------
Tout warning produit par un module V4 et injecté dans un bloc du prompt
système doit comporter :

1. Une **icône** (par défaut ⚠️) pour saillance visuelle dans le prompt.
2. Un **libellé court** de l'issue (ex: "Incohérence temporelle").
3. Les **détails factuels** (les valeurs en conflit, la date système, etc.).
4. Une **directive d'action** commençant par "→" qui indique au LLM
   *comment* réagir au warning, pas seulement *qu'il existe*.

Pourquoi
--------
Validation in vivo (V4.0.2, Qwen3-4B Thinking, 10 mai 2026) :
le LLM lisait le warning Timeline dans son `<think>` mais ne le
**caractérisait** pas dans sa réponse finale. Il signalait vaguement
"il y a confusion" sans pointer la contradiction précise.

Adjacent à un warning, une directive concrète ("→ demande à l'utilisateur
quelle des deux dates correspond à l'événement, ne paraphrase pas le
message comme s'il était cohérent") guide le LLM vers la bonne réaction
sans avoir besoin de consigne globale dans le system prompt.

Génericité
----------
Tous les modules V4 (présents et futurs) qui produisent des warnings
prompt-side passent par ``format_warning()`` pour garantir un format
homogène. Quand un nouveau module V4 émet un warning, il fournit
simplement (issue, details, directive) ; le helper s'occupe du
formatage.
"""

from __future__ import annotations


# Default icon for V4 warnings. Stable across modules so the LLM can
# learn to recognise it as a structured-attention marker.
DEFAULT_WARNING_ICON = "⚠️"

# Prefix used for the directive line, immediately after the warning.
# Visually adjacent ⇒ short cognitive distance for the LLM.
DIRECTIVE_PREFIX = "   → "


def format_warning(
    issue: str,
    details: str,
    directive: str,
    icon: str = DEFAULT_WARNING_ICON,
) -> str:
    """Format a V4 module warning with an actionable directive.

    Parameters
    ----------
    issue : str
        Short label naming the issue. Examples:
        "Incohérence temporelle", "Discordance",
        "Confiance non étayée".
    details : str
        Concrete facts that constitute the warning. Should be
        self-contained (no references to "the user said" — the LLM
        already has that context).
    directive : str
        Imperative-mood instruction telling the LLM how to react.
        Should be specific enough that the LLM can produce a
        well-formed response without further interpretation.
        Bad: "Sois prudent."
        Good: "Demande à l'utilisateur quelle des deux dates
              correspond à l'événement et ne paraphrase pas le
              message comme s'il était cohérent."
    icon : str, optional
        Override the default ⚠️ icon. Useful for non-error warnings
        (ex: ℹ️ pour info contextuelle, 🤔 pour suggestion).

    Returns
    -------
    str
        Two-line block, ready to be appended to a [Chronologie] /
        [Métacognition] / [...] prompt block. No trailing newline.

    Examples
    --------
    >>> format_warning(
    ...     "Incohérence temporelle",
    ...     "« hier » (09/05/2026) ne correspond pas à « 12 mai 2026 » (futur)",
    ...     "Demande quelle date est correcte au lieu de paraphraser.",
    ... )
    '⚠️ Incohérence temporelle : « hier » (09/05/2026) ne correspond pas à « 12 mai 2026 » (futur)\\n   → Demande quelle date est correcte au lieu de paraphraser.'
    """
    # Defensive: never crash on bad input — return a degraded but
    # well-formed warning rather than raising.
    issue = (issue or "Avertissement").strip()
    details = (details or "").strip()
    directive = (directive or "").strip()

    head = f"{icon} {issue}"
    if details:
        head = f"{head} : {details}"
    if directive:
        return f"{head}\n{DIRECTIVE_PREFIX}{directive}"
    return head


def is_v4_warning_line(line: str) -> bool:
    """True if a string looks like a V4 warning header.

    Useful for downstream code that wants to detect / count / strip
    warnings from a rendered block (e.g. a UI that renders warnings
    in a separate panel).
    """
    if not line:
        return False
    return line.lstrip().startswith(DEFAULT_WARNING_ICON)
