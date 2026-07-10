"""Tool dispatcher — V5.1 multi-tool slow-path classifier.

When the fast-path semantic router can't decide confidently which
tool to use (ambiguous zone), we fall back to the LLM itself. This
module replaces the V5 binary OUI/NON web classifier with a
multi-class JSON output : the LLM picks one of {web, python, none}.

Why JSON instead of OUI/NON
---------------------------
V5 was hardcoded for one decision (web ?). V5.1 has multiple tools,
so a binary answer doesn't fit. Asking for JSON gives :
    {"tool": "python", "reason": "calcul direct"}
which is parseable and extensible (adding a 4th tool = adding a
new enum value).

Robustness
----------
LLMs in the 4-7B range generate malformed JSON ~5-10% of the time.
We have a parsing cascade :
1. Strict JSON.loads on the full output.
2. Substring extraction of ``{...}`` followed by JSON.loads.
3. Regex extraction of ``"tool": "..."`` and ``"reason": "..."``.
4. Fallback to "none" — safest default (skip the tool, don't fire
   web/python on a malformed verdict).

The prompt explicitly shows the JSON format in the system message
with concrete examples to anchor the format.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import OrderedDict
from typing import Protocol

log = logging.getLogger("rune.cognition.tool_dispatcher")


# ── Valid tool names ────────────────────────────────────────────────────
# Keep aligned with semantic_router.ROUTES + the actual tool registry.
# "none" means : answer directly, no external tool needed.

VALID_TOOLS = ("web", "python", "mcp", "none")


# ── LLM interface ──────────────────────────────────────────────────────


class LLMCompleter(Protocol):
    """Same interface as web_classifier (V5)."""
    def complete_sync(
        self,
        messages: list[dict],
        max_new_tokens: int = 32,
        timeout: float | None = None,
    ) -> str: ...


# ── System prompt ──────────────────────────────────────────────────────


_DISPATCHER_SYSTEM_PROMPT = (
    "Tu es un dispatcher d'outils. Tu décides quel outil utiliser pour "
    "répondre à la question, ou aucun.\n"
    "\n"
    "Outils disponibles :\n"
    "- web : recherche internet en temps réel. Pour les faits actuels "
    "(news, prix, météo, scores, sortie de produit), les recommandations "
    "techniques précises (modèle, lib, paper), les biographies de "
    "personnes peu connues.\n"
    "- python : exécute du code Python. Pour les calculs précis "
    "(arithmétique, statistiques), l'exécution de code utilisateur, "
    "le tracé de graphiques (matplotlib), la manipulation de données.\n"
    "- mcp : lit ou écrit un fichier dans le workspace partagé. "
    "Pour les demandes qui mentionnent un fichier (par nom ou "
    "implicitement : « le fichier que je viens d'ajouter », « mon "
    "CSV », « le rapport »), pour lister/explorer le workspace, "
    "ou pour sauvegarder un livrable dans un fichier.\n"
    "- none : aucun outil. Pour l'explication de concepts stables, les "
    "avis subjectifs, la créativité, la conversation, la mémoire "
    "interne (« tu te souviens... »).\n"
    "\n"
    "Réponds OBLIGATOIREMENT en JSON strict sur une ligne :\n"
    '  {"tool": "web|python|mcp|none", "reason": "3-5 mots"}\n'
    "\n"
    "Exemples :\n"
    'Q: "Quel est le prix actuel du Bitcoin ?"\n'
    '  → {"tool": "web", "reason": "prix volatile temps réel"}\n'
    'Q: "Combien font 17 × 23 + 89 ÷ 11 ?"\n'
    '  → {"tool": "python", "reason": "calcul arithmétique précis"}\n'
    'Q: "Lis sales.csv et donne-moi la moyenne"\n'
    '  → {"tool": "mcp", "reason": "lecture fichier workspace"}\n'
    'Q: "Quels fichiers j\'ai déposés ?"\n'
    '  → {"tool": "mcp", "reason": "listage workspace"}\n'
    'Q: "Explique-moi comment fonctionne un transformer"\n'
    '  → {"tool": "none", "reason": "concept stable explicable"}\n'
    'Q: "Tu te souviens de ce que je t\'ai dit hier ?"\n'
    '  → {"tool": "none", "reason": "mémoire interne"}\n'
    'Q: "Recommande-moi un modèle NER en français"\n'
    '  → {"tool": "web", "reason": "vérifier références exactes"}\n'
)


def _build_dispatcher_messages(query: str) -> list[dict]:
    return [
        {"role": "system", "content": _DISPATCHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Q: \"{query.strip()}\"\n→"},
    ]


# ── Parse cascade ──────────────────────────────────────────────────────


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_TOOL_KEY_RE = re.compile(
    r'"tool"\s*:\s*"(web|python|none)"', re.IGNORECASE,
)
_REASON_KEY_RE = re.compile(
    r'"reason"\s*:\s*"([^"]{0,80})"',
)


def _parse_dispatcher_response(raw: str) -> tuple[str, str] | None:
    """Parse the LLM output, returning (tool, reason) or None.

    Cascade : strict JSON → substring JSON → regex extraction.
    Returns None only if all three fail, leaving the caller to
    apply its own fallback (typically "none").
    """
    if not raw or not raw.strip():
        return None
    txt = raw.strip()

    # Step 1: strict JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            tool = str(obj.get("tool", "")).lower().strip()
            reason = str(obj.get("reason", "")).strip()[:80]
            if tool in VALID_TOOLS:
                return tool, reason or tool
    except json.JSONDecodeError:
        pass

    # Step 2: substring JSON. The model often blathers before the
    # JSON ; we extract the first ``{...}`` block.
    match = _JSON_OBJECT_RE.search(txt)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                tool = str(obj.get("tool", "")).lower().strip()
                reason = str(obj.get("reason", "")).strip()[:80]
                if tool in VALID_TOOLS:
                    return tool, reason or tool
        except json.JSONDecodeError:
            pass

    # Step 3: regex on individual keys. Even more permissive.
    tool_m = _TOOL_KEY_RE.search(txt)
    if tool_m:
        tool = tool_m.group(1).lower()
        reason_m = _REASON_KEY_RE.search(txt)
        reason = reason_m.group(1) if reason_m else tool
        return tool, reason

    return None


# ── Cache LRU (réutilisé pour V5.1) ────────────────────────────────────


def _normalise_for_cache(message: str) -> str:
    """Same as V5 web_classifier."""
    msg = message.strip().lower()
    msg = re.sub(r"\s+", " ", msg)
    msg = re.sub(r"[?!.,;:]+$", "", msg)
    return msg


import threading

class _LRUCache:
    def __init__(self, capacity: int = 256) -> None:
        self._capacity = capacity
        self._data: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[str, str] | None:
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]

    def put(self, key: str, value: tuple[str, str]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0


_decision_cache = _LRUCache(capacity=256)


# ── Public API ─────────────────────────────────────────────────────────


def dispatch_via_llm(
    query: str,
    llm: LLMCompleter,
    timeout: float = 3.0,
    use_cache: bool = True,
) -> tuple[str, str]:
    """Ask the LLM to pick a tool.

    Returns
    -------
    tuple[str, str]
        ``(tool_name, reason)``. tool_name is one of VALID_TOOLS.
        On any failure (LLM error, unparseable output), returns
        ``("none", "dispatcher_failed")`` — safest default that
        skips tool invocation rather than guessing wrong.
    """
    cache_key = _normalise_for_cache(query)
    if use_cache:
        cached = _decision_cache.get(cache_key)
        if cached is not None:
            log.debug("Dispatcher cache hit: %r → %s", cache_key[:40], cached[0])
            return cached

    t0 = time.monotonic()
    try:
        raw = llm.complete_sync(
            _build_dispatcher_messages(query),
            max_new_tokens=32,  # JSON court
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        log.info(
            "Dispatcher LLM: %.0fms, raw=%r",
            elapsed * 1000, (raw or "")[:100],
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.warning("Dispatcher LLM failed (%.0fms): %s", elapsed * 1000, exc)
        return "none", "dispatcher_failed"

    parsed = _parse_dispatcher_response(raw)
    if parsed is None:
        log.warning("Dispatcher output unparseable: %r", (raw or "")[:100])
        return "none", "dispatcher_unparseable"

    tool, reason = parsed
    result = (tool, reason or tool)
    if use_cache:
        _decision_cache.put(cache_key, result)
    return result


def get_cache_stats() -> dict:
    return {
        "size": len(_decision_cache._data),
        "capacity": _decision_cache._capacity,
        "hits": _decision_cache.hits,
        "misses": _decision_cache.misses,
        "hit_rate": (
            _decision_cache.hits / max(1, _decision_cache.hits + _decision_cache.misses)
        ),
    }


def clear_cache() -> None:
    _decision_cache.clear()
