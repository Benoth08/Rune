"""LLM-based web search classifier — Lythéa V4.4.

When the regex fast-path in :mod:`lythea.web` doesn't match (neither
clearly "yes web" nor clearly "no web"), this module asks the local
LLM itself whether the question needs web search. It's the hybrid
approach from the V4.4 session: regex for the obvious cases, model
for the ambiguous ones.

Why this matters
----------------
The regex classifier has known limitations (faux positifs sur
"recommande-moi un mug", faux négatifs sur "tu connais une bonne lib
JS pour les graphes"). Building an ever-growing list of patterns
hits a combinatorial wall. Anthropic/OpenAI/Google all let the LLM
itself decide via function calling. We mimic that pattern locally
with a cheap one-shot classifier.

Design
------
1. The fast-path remains primary — we only invoke this slow-path when
   the regex didn't match AND the message looks like a real question
   (heuristic via :func:`looks_like_question`). A casual "merci !"
   never triggers the classifier.
2. We cache decisions via a simple LRU keyed by normalised message
   (lowercased, ponctuation finale retirée, whitespace dédupliqué).
   Same question twice → 0ms second time.
3. The classifier prompt is short and structured: explicit positive
   cases, explicit negative cases, format strict (OUI/NON + raison).
4. If parsing fails or the model times out → fallback to NO (safe
   default, avoid unnecessary web calls).

Cost
----
Per slow-path call: ~80 tokens prompt + ~10 tokens response.
On Qwen2.5-7B local : ~300-500ms.
Cache hits: ~0ms.

Disable via :data:`lythea.settings.web_classifier_enabled` if needed.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Protocol

log = logging.getLogger("lythea.cognition.web_classifier")


# ── Heuristique "ressemble à une question" ─────────────────────────────
# On n'invoque le slow-path classifier QUE si le message paraît être une
# question — pour éviter d'appeler le LLM sur chaque "merci !" ou
# "ok parfait".

_QUESTION_START_RE = re.compile(
    r"^(?:"
    r"est-ce |quel|quelle|quels|quelles|combien|comment|qui |où |"
    r"pourquoi|peux-tu|sais-tu|connais-tu|"
    r"is |are |what |which |when |where |who |why |how |can you|do you|"
    r"can I|could you|would you|"
    # V5.8.6 — Impératifs d'action qui doivent passer au router
    # (sinon "Calcule moyenne..." ou "Décode ce base64" sont
    # traités comme conversation et le router Python est skippé).
    # Couvrent les verbes typiques de tâche : calcul, conversion,
    # encodage, génération, exécution, tracé.
    # NB : les regex FR avec [ée] couvrent automatiquement les
    # équivalents EN (décode → decode, génère → generate, etc.)
    r"calcule|calcules|calculer|"
    r"convertis|conversion|convertir|"
    r"d[ée]code|d[ée]codes|d[ée]coder|encode|encodes|encoder|"
    r"trace|traces|tracer|plot|"
    r"g[ée]n[èe]re|g[ée]n[èe]res|g[ée]n[èe]rer|"
    r"ex[ée]cute|ex[ée]cutes|ex[ée]cuter|lance|lances|lancer|"
    r"affiche|affiches|afficher|montre[\s\-]me|"
    r"trie|tries|trier|"
    # Verbes EN purs (pas couverts par les regex FR avec [ée])
    r"compute|calculate|convert|generate|decode|run|sort|"
    # V5.8.6 — Questions implicites sans ? terminal
    r"est-il|est-elle|sont-ils|sont-elles|"
    r"vérifie|vérifies|vérifier|teste|testes|tester|"
    # V5.9.1 — Verbes d'usage d'outil/lib (Python sandbox)
    r"utilise|utilises|utiliser|use\s|using\s|"
    r"importe|importes|importer|import\s|"
    r"appelle|appelles|appeler|call\s|"
    # V6.0.0-rc — Verbes pour route MCP (lecture / listing / écriture
    # de fichiers du workspace). Sans ces patterns, "Lis sales.csv" ou
    # "Read data.csv" sont classés comme conversation et le router MCP
    # n'est jamais déclenché → Lythéa hallucine le contenu.
    r"lis|lit|lire|relis|relire|"
    r"ouvre|ouvres|ouvrir|"
    r"liste|listes|lister|list\s|"
    r"sauvegarde|sauvegardes|sauvegarder|save\s|"
    r"écris|écrit|écrire|write\s|writes\s|writing\s|"
    r"read\s|reads\s|reading\s|opens?\s|opening\s|"
    r"montre|montres|montrer|show\s|shows\s|"
    r"analyse|analyses|analyser|examine|examines|examiner"
    r")",
    re.IGNORECASE,
)

_QUESTION_INTENT_RE = re.compile(
    r"(?:"
    r"je cherche|je voudrais savoir|je veux savoir|aide-moi à trouver|"
    r"j'aimerais (?:savoir|connaître|comprendre)|"
    r"dis-moi|donne-moi|trouve-moi|montre-moi|"
    r"i want to know|i'd like to know|help me find|"
    r"tell me|show me|find me"
    r")",
    re.IGNORECASE,
)


def looks_like_question(message: str) -> bool:
    """Heuristique : le message ressemble-t-il à une question/requête ?

    Vrai si :
    - Termine par ``?``, ou
    - Commence par un mot interrogatif (quel, comment, who, what…), ou
    - Contient un marqueur d'intention (je cherche, dis-moi…).

    Faux pour les salutations, remerciements, conversation casual
    (« merci ! », « ok », « bonne nuit »). Évite des appels classifier
    inutiles sur ces messages.
    """
    if not message or not message.strip():
        return False
    msg = message.strip()
    if msg.endswith("?") or msg.endswith("?"):
        return True
    if _QUESTION_START_RE.search(msg):
        return True
    if _QUESTION_INTENT_RE.search(msg):
        return True
    return False


# ── Cache LRU pour les décisions classifier ────────────────────────────


def _normalise_for_cache(message: str) -> str:
    """Clé de cache stable : lowercase, espaces dédupliqués, "?!." de fin."""
    msg = message.strip().lower()
    msg = re.sub(r"\s+", " ", msg)
    msg = re.sub(r"[?!.,;:]+$", "", msg)
    return msg


class _LRUCache:
    """Cache LRU thread-safe simple. Pas d'import externe."""

    def __init__(self, capacity: int = 256) -> None:
        self._capacity = capacity
        self._data: OrderedDict[str, tuple[bool, str]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[bool, str] | None:
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]

    def put(self, key: str, value: tuple[bool, str]) -> None:
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


# Cache module-level. Partagé entre instances Hippocampe (toutes les
# sessions d'un même process). Préserve les décisions à travers les
# tours et les sessions.
_decision_cache = _LRUCache(capacity=256)


# ── Protocole LLM pour le classifier ───────────────────────────────────


class LLMCompleter(Protocol):
    """Interface minimale attendue d'un LLM pour le classifier.

    Doit fournir une méthode de complétion synchrone simple. On utilise
    :meth:`lythea.model.Model.complete_sync` ou équivalent. Le protocole
    permet de mocker en test.
    """

    def complete_sync(
        self,
        messages: list[dict],
        max_new_tokens: int = 24,
        timeout: float | None = None,
    ) -> str: ...


# ── Prompt système du classifier ───────────────────────────────────────


_CLASSIFIER_SYSTEM_PROMPT = (
    "Tu es un classifieur binaire. Tu décides si une question utilisateur "
    "nécessite une recherche web pour bien y répondre.\n"
    "\n"
    "Réponds OUI si la question demande :\n"
    "- Un fait actuel ou récent (météo, prix, news, score, élection, "
    "sortie de produit, disponibilité)\n"
    "- Une recommandation technique précise (nom de modèle, librairie, "
    "package, paper, API, framework)\n"
    "- Un fait que ta connaissance pourrait ignorer ou avoir périmé\n"
    "- Une information qu'on peut vérifier en ligne (statistique, "
    "biographie de personne moins connue, événement)\n"
    "\n"
    "Réponds NON si la question est :\n"
    "- Une explication pédagogique d'un concept stable\n"
    "- Une demande d'avis subjectif, de créativité, d'imagination\n"
    "- Conversationnelle, personnelle, ou émotionnelle\n"
    "- Sur ta mémoire interne (« tu te souviens de... »)\n"
    "- Du calcul, du code à écrire, de la traduction\n"
    "- Une recommandation non technique (mug, vin, film, restaurant)\n"
    "\n"
    "Format de réponse OBLIGATOIRE — un seul mot OUI ou NON, puis "
    "deux-points et 3 à 5 mots de raison. Exemples :\n"
    "  OUI : prix actuel volatile\n"
    "  NON : concept stable explicable\n"
    "  OUI : nom de modèle à vérifier\n"
    "  NON : créativité subjective\n"
)


def _build_classifier_messages(message: str) -> list[dict]:
    """Construit les messages du prompt classifier."""
    return [
        {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question : {message.strip()}"},
    ]


# ── Parse de la sortie classifier ──────────────────────────────────────


_RESPONSE_PARSE_RE = re.compile(
    r"^\s*(OUI|NON|YES|NO)\b\s*[:\-—]?\s*(.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _parse_classifier_response(raw: str) -> tuple[bool, str] | None:
    """Parse la sortie du modèle. Retourne (decision, reason) ou None.

    Le modèle peut produire :
    - "OUI : raison" → (True, "raison")
    - "NON: raison" → (False, "raison")
    - "OUI" tout seul → (True, "")
    - Texte non parseable → None (caller fallback to NON)

    On accepte aussi YES/NO pour robustesse multilingue.
    """
    if not raw or not raw.strip():
        return None
    # Premier mot — souvent le modèle blablate avant, on cherche le
    # premier OUI/NON/YES/NO.
    head = raw.strip().splitlines()[0] if raw.strip() else ""
    match = _RESPONSE_PARSE_RE.match(head)
    if not match:
        # Fallback : chercher OUI/NON n'importe où dans la sortie
        head_lower = head.lower()
        if "oui" in head_lower or "yes" in head_lower:
            return True, head[:60].strip()
        if "non" in head_lower or "no" in head_lower:
            return False, head[:60].strip()
        return None
    verdict = match.group(1).upper()
    reason = (match.group(2) or "").strip()[:60]
    decision = verdict in ("OUI", "YES")
    return decision, reason


# ── API publique ───────────────────────────────────────────────────────


def should_search_via_llm(
    message: str,
    llm: LLMCompleter,
    timeout: float = 3.0,
    use_cache: bool = True,
) -> tuple[bool, str]:
    """Demande au LLM local si la question nécessite une recherche web.

    Parameters
    ----------
    message
        La question utilisateur brute.
    llm
        Objet avec ``complete_sync(messages, max_new_tokens, timeout)``.
    timeout
        Timeout en secondes pour l'appel modèle. Default 3.0.
    use_cache
        Si True (défaut), utilise le cache LRU module-level. Mettre False
        pour les tests.

    Returns
    -------
    tuple[bool, str]
        ``(should_search, reason)``. Reason est un court texte
        explicatif (~3-5 mots). En cas d'échec (parsing, timeout,
        exception) → ``(False, "classifier_failed")`` : fallback
        conservateur, on évite le web par défaut.
    """
    cache_key = _normalise_for_cache(message)

    # Cache hit ?
    if use_cache:
        cached = _decision_cache.get(cache_key)
        if cached is not None:
            log.debug("Classifier cache hit: %r → %s", cache_key[:40], cached[0])
            return cached

    # Heuristique préalable : ressemble pas à une question → NON direct
    # sans appeler le LLM. Évite les appels inutiles sur conversation
    # casual (le caller devrait avoir filtré aussi, mais double check).
    if not looks_like_question(message):
        result = (False, "not_a_question")
        if use_cache:
            _decision_cache.put(cache_key, result)
        return result

    # Appel LLM
    t0 = time.monotonic()
    try:
        messages = _build_classifier_messages(message)
        raw = llm.complete_sync(
            messages,
            max_new_tokens=24,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        log.info(
            "LLM classifier: %.0fms, raw=%r",
            elapsed * 1000,
            (raw or "")[:80],
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.warning(
            "LLM classifier failed (%.0fms): %s",
            elapsed * 1000,
            exc,
        )
        # Pas de cache sur échec — on retentera la prochaine fois
        return False, "classifier_failed"

    parsed = _parse_classifier_response(raw)
    if parsed is None:
        log.warning("LLM classifier output unparseable: %r", (raw or "")[:80])
        return False, "classifier_unparseable"

    decision, reason = parsed
    result = (decision, reason or ("yes" if decision else "no"))

    if use_cache:
        _decision_cache.put(cache_key, result)

    return result


def get_cache_stats() -> dict:
    """Retourne les stats du cache pour monitoring/debug."""
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
    """Vide le cache (utile entre les tests)."""
    _decision_cache.clear()
