"""SearXNG provider — open-source metasearch engine.

SearXNG aggregates Google, Bing, Brave, Wikipedia, GitHub, etc. and
returns deduplicated JSON results. No API key, no rate limit (when
self-hosted), uncensored.

This provider:

1. Tries the user-configured instance URL if any (``SEARXNG_INSTANCE_URL``
   env var or constructor arg).
2. Otherwise rotates through a hardcoded shortlist of known-good
   public instances. The list is regularly updated upstream; see
   https://searx.space for the current health-ranked list.
3. On any failure (HTTP error, timeout, malformed response),
   returns an empty list — the orchestrator falls back to DDG.

Format conversion
-----------------
SearXNG returns ``{"results": [{"title", "content", "url", ...}]}``.
We map ``content`` → ``body`` and ``url`` → ``href`` to match the
legacy DDG output format.
"""

from __future__ import annotations

import json
import logging
import random
import urllib.error
import urllib.parse
import urllib.request

from rune.web_providers.base import SearchResult

log = logging.getLogger("rune.web_providers.searxng")


# Curated public SearXNG instances. Selected for:
# - JSON API enabled (some instances disable it for abuse prevention)
# - Reasonable uptime per searx.space
# - HTTPS only
# These are best-effort defaults — users with reliability needs should
# self-host (https://docs.searxng.org/admin/installation-docker.html)
# and set SEARXNG_INSTANCE_URL.
DEFAULT_PUBLIC_INSTANCES = (
    "https://search.inetol.net",
    "https://baresearch.org",
    "https://searx.be",
    "https://searx.tiekoetter.com",
    "https://opnxng.com",
    "https://priv.au",
)

# HTTP timeout in seconds. SearXNG aggregates upstream engines so we
# need to be patient — Google + Bing + DDG can take 4-8s combined.
DEFAULT_TIMEOUT_SEC = 10.0

# UA — SearXNG instances often filter Python's default UA as bot.
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# V4.4 — Liste de domaines à filtrer côté Lythéa.
# Pourquoi : même avec safesearch=1, certains résultats parasites
# remontent (dictionnaires bilingues pour les questions en français,
# sites NSFW qui ont des URLs corrélées avec des mots-clés
# innocents). Ces patterns sont des sous-chaînes recherchées dans
# l'URL ; ils dégagent les pires faux positifs observés en prod.
# Pour ajuster sans modifier le code : variable d'env
# LYTHEA_WEB_BLOCKLIST="domain1,domain2".
DEFAULT_BLOCKED_PATTERNS = (
    # NSFW / adult content
    "nhentai.net",
    "hentai",
    "porn",
    "xxx",
    "xvideos",
    "redtube",
    # Dictionnaires bilingues : remontent en spam sur les questions
    # françaises ("sont", "quel", etc. matchent des entrées dico)
    "wordhippo.com",
    "frenchdictionary.com",
    "collinsdictionary.com",
    "dictionary.reverso.net",
    # Sites trivia / calendrier — ils répondent à n'importe quelle query
    # parce qu'ils contiennent des listes d'événements pour chaque jour.
    # Observé en prod : "Macaulay Culkin / IMDb" sur une question quantique
    # parce que c'est l'anniversaire du jour.
    "onthisday.com",
    "onthisdaynow.com",
    "timeanddate.com/on-this-day",
    "britannica.com/on-this-day",
    # Sites spam / SEO low-quality
    "answers.com",
    "quora.com/topic",  # quora topic pages, pas les vraies réponses
    # V4.4 — sites administratifs / RH US qui remontent sur "NER" parce
    # que "NER" matche "Series Number" ou des codes administratifs.
    # Observé en prod sur "Recommande-moi un modèle NER" → opm.gov,
    # usgs.gov, federaljobs.net (Management Analysis Series GS-0343).
    "opm.gov",
    "usgs.gov",
    "federaljobs.net",
    "usajobs.gov",
)


# V4.4 — Détection de langue par stopwords + diacritiques. Sans dépendance
# externe (pas besoin d'installer langdetect ni fasttext). Suffisamment
# fiable pour les besoins de Lythéa : on veut juste savoir si on passe
# language=fr ou language=en à SearXNG pour le ranking. Un faux positif
# ne casse rien — juste un léger biais de pertinence — donc une
# heuristique simple suffit largement.
_FR_STOPWORDS = frozenset({
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "et", "ou", "mais", "donc", "car", "ni",
    "qui", "que", "quel", "quelle", "quels", "quelles",
    "comment", "pourquoi", "où", "quand", "combien",
    "est", "sont", "sera", "était", "étaient",
    "ai", "as", "avons", "avez", "ont", "avait",
    "ce", "cette", "ces", "ça", "cela",
    "se", "te", "me", "nous", "vous", "ils", "elles",
    "dans", "pour", "avec", "sur", "sous", "par",
    "depuis", "avant", "après", "pendant",
    "très", "plus", "moins", "aussi", "encore",
    "il", "elle", "on", "je", "tu",
    "mon", "ton", "son", "ma", "ta", "sa",
    "mes", "tes", "ses", "nos", "vos", "leurs",
    "pas", "ne", "non", "oui",
    "fait", "faire", "faut", "peut",
})
_EN_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "but", "so", "if", "then",
    "who", "what", "how", "why", "where", "when", "which", "whose",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "will", "would", "should", "could", "may", "might", "must",
    "this", "that", "these", "those",
    "in", "on", "at", "for", "with", "by", "from", "to", "into",
    "out", "up", "down", "over", "under", "again",
    "do", "does", "did", "doing", "done",
    "i", "you", "he", "she", "we", "they", "it",
    "my", "your", "his", "her", "our", "their", "its",
    "me", "him", "us", "them",
    "not", "no", "yes",
    "than", "as", "such",
})
# Diacritiques typiques du français. Une seule occurrence suffit
# à trancher FR (l'anglais standard n'en utilise pas).
_FR_DIACRITICS = frozenset("éèêëàâçîïôœùûüÿ")


def detect_lang(query: str) -> str:
    """Retourne 'fr', 'en' ou 'auto' selon la langue détectée.

    Heuristique :
    1. Si la query contient un accent → 'fr' (signal fort).
    2. Sinon, compare le nombre de stopwords FR vs EN.
    3. Si peu d'indices (queries courtes type "roland garros 2025"),
       retourne 'auto' et laisse SearXNG décider.

    Exemples :
      "Quelles sont les nouveautés"     → 'fr' (accent + stopwords FR)
      "What is quantum entanglement"    → 'en' (stopwords EN)
      "Qui a gagné Roland Garros 2025"  → 'fr' (qui, a)
      "Roland Garros 2025"              → 'auto' (que des noms propres)
      "best smartphones 2026"           → 'auto' (juste mots-clés)
    """
    if not query or not query.strip():
        return "auto"
    if any(c in _FR_DIACRITICS for c in query):
        return "fr"
    import re
    words = set(re.findall(r"[a-z']+", query.lower()))
    fr_hits = len(words & _FR_STOPWORDS)
    en_hits = len(words & _EN_STOPWORDS)
    if fr_hits >= 1 and fr_hits > en_hits:
        return "fr"
    if en_hits >= 1 and en_hits > fr_hits:
        return "en"
    return "auto"


class SearxngProvider:
    """Pluggable SearXNG search provider.

    Parameters
    ----------
    instance_url : str | None
        If provided, only this instance is tried. If None, rotates
        through ``DEFAULT_PUBLIC_INSTANCES`` until one responds.
    timeout : float
        HTTP timeout per request.
    language : str
        UI language hint (also influences result ranking). Default
        ``"auto"`` — Lythéa détecte la langue de la requête (FR/EN)
        et choisit le bon hint pour SearXNG. Si tu poses une question
        en français, on prioriser les sources françaises ; si tu
        poses en anglais, on prioritise les sources anglaises. Tu
        peux forcer une langue spécifique en passant ``"fr"``,
        ``"en"``, etc.
    safesearch : int
        0 = off, 1 = moderate, 2 = strict. Default **1** (moderate) —
        évite les résultats NSFW qui remontaient par corrélation
        sur des mots-clés innocents (observé en prod : nhentai.net
        sur "roland garros 2025").
    """

    name = "searxng"

    def __init__(
        self,
        instance_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        language: str = "auto",
        safesearch: int = 1,
    ) -> None:
        self.instance_url = (instance_url or "").rstrip("/") or None
        self.timeout = timeout
        self.language = language
        self.safesearch = max(0, min(2, int(safesearch)))
        self._instances_pool: list[str] = list(
            (self.instance_url,) if self.instance_url else DEFAULT_PUBLIC_INSTANCES
        )
        self._known_good: str | None = None  # cached best instance
        self._availability_checked: bool = False
        self._available: bool = False
        # Blocklist : patterns env override → defaults
        import os
        env_blocklist = os.getenv("LYTHEA_WEB_BLOCKLIST", "").strip()
        if env_blocklist:
            self._blocked_patterns = tuple(
                p.strip().lower() for p in env_blocklist.split(",") if p.strip()
            )
        else:
            self._blocked_patterns = DEFAULT_BLOCKED_PATTERNS

    # ── Public API ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Probe one instance. Cached after first call."""
        if self._availability_checked:
            return self._available
        # Treat as available if we have any candidate URL — the actual
        # network probe happens at search time, where failure triggers
        # the fallback. Keeping is_available cheap avoids a startup
        # delay if all public instances are slow.
        self._available = bool(self._instances_pool)
        self._availability_checked = True
        return self._available

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Try instances in order, return first success."""
        if not query or not query.strip():
            return []
        query = query.strip()[:300]  # SearXNG handles long queries poorly

        # Try the last-known-good instance first to minimise warm-up cost
        order = list(self._instances_pool)
        if self._known_good and self._known_good in order:
            order.remove(self._known_good)
            order.insert(0, self._known_good)
        elif len(order) > 1 and not self.instance_url:
            # Light shuffling so we don't hammer one specific public
            # instance from many Lythéa users.
            random.shuffle(order)

        for url in order:
            try:
                results = self._search_one(url, query, max_results)
            except Exception as exc:
                log.debug("SearXNG %s failed: %s", url, exc)
                continue
            if results:
                self._known_good = url
                return results
        log.warning("All SearXNG instances failed for query=%r", query[:60])
        return []

    # ── Internals ───────────────────────────────────────────────────

    def _search_one(
        self,
        instance_url: str,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """One HTTP call to a single SearXNG instance.

        Raises on transport / decode errors so the outer loop can try
        the next instance. Returns empty list on a valid but empty
        response (the orchestrator will then drop to fallback).
        """
        # V4.4 — forcer ``categories=general`` pour ne consulter que
        # les engines généralistes (Brave, DDG, Bing, Mojeek, etc.).
        # Sans ça, SearXNG interroge AUSSI les engines spécialisés
        # (IMDb, "On This Day", Britannica events…) qui répondent
        # à n'importe quelle query et polluent les résultats —
        # exemple observé : Macaulay Culkin remonté en réponse à une
        # question quantique parce que c'est son anniversaire ce
        # jour-là dans l'engine "events".
        # V4.4 — détection auto de la langue de la query si le user a
        # demandé "auto" (default). Sinon on respecte son override.
        # On bypass "auto" (qui était problématique côté SearXNG) en
        # détectant fr/en côté Lythéa avec une heuristique simple.
        if self.language == "auto":
            lang_for_query = detect_lang(query)
        else:
            lang_for_query = self.language
        params = {
            "q": query,
            "format": "json",
            "language": lang_for_query,
            "safesearch": str(self.safesearch),
            "categories": "general",
        }
        full_url = (
            instance_url.rstrip("/")
            + "/search?"
            + urllib.parse.urlencode(params)
        )
        req = urllib.request.Request(
            full_url,
            headers={
                "User-Agent": DEFAULT_UA,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))

        raw_results = payload.get("results") or []
        out: list[SearchResult] = []
        n_blocked = 0
        for r in raw_results:
            title = r.get("title", "")
            body = r.get("content", "") or r.get("snippet", "")
            href = r.get("url", "")
            if not (title or body):
                continue
            # V4.4 — filtre les domaines parasites (NSFW, dictionnaires
            # bilingues, sites spam). Match sur substring de l'URL.
            href_lower = href.lower()
            if any(pat in href_lower for pat in self._blocked_patterns):
                n_blocked += 1
                continue
            out.append({"title": title, "body": body, "href": href})
            if len(out) >= max_results:
                break
        if n_blocked > 0:
            log.debug(
                "SearXNG filtered %d parasitic results (blocklist) for q=%r",
                n_blocked, query[:60],
            )
        return out
