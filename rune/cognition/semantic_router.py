"""Semantic router — V5.1 multi-tool routing via embeddings.

Replaces the V5 binary web/no-web decision with a multi-class route
selector. Each tool (web, python, ...) has a set of example phrases.
At runtime we encode the user's question, compute cosine similarity
against every example, and pick the tool whose best match exceeds a
confidence threshold. Below threshold → return "ambiguous" and let
the slow-path LLM dispatcher decide.

Why this design
---------------
Regex doesn't scale past 2 tools — combinations explode. Function
calling needs LLM passes that cost 300+ ms each turn. Semantic
routing sits in between: 25-50ms CPU, robust to phrasing variations,
scales linearly with the number of tools (just add example phrases).

The chosen model is ``paraphrase-multilingual-MiniLM-L12-v2`` (~120MB)
for FR/EN support. Falls back to ``all-MiniLM-L6-v2`` (already in the
KG path) if the multilingual one fails to load.

The example phrases per tool are deliberately diverse (different
phrasings, registers, lengths) so the embedding space of each tool
is well covered. Add new tools by extending ROUTES below.

Inspired by Aurelio AI's ``semantic-router`` package, but reimplemented
in-process to avoid an extra dependency.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("lythea.cognition.semantic_router")


# ── Route definitions ──────────────────────────────────────────────────
# Each route lists a set of example phrases the router will match
# against. Aim for ~30-50 examples per tool, diverse in phrasing,
# register (formel/casual), length, and language (FR + EN).

@dataclass
class Route:
    """A tool route with its example phrases.

    name : identifier returned by classify() (e.g. "web", "python")
    examples : list of representative questions for this tool
    threshold : minimum cosine similarity (over the best matching
        example) to confidently route to this tool. Default 0.55 ;
        lower = more permissive, higher = more conservative.
        Tuned empirically per route since the example coverage
        varies (web has tons of cases, python is narrower).
    """
    name: str
    examples: list[str]
    threshold: float = 0.55
    # Filled at warm-up by SemanticRouter.
    _embeddings: np.ndarray | None = field(default=None, repr=False)


ROUTES: list[Route] = [
    Route(
        name="web",
        threshold=0.55,
        examples=[
            # ── Actualité, news, événements ────────────────────────────
            "Quelles sont les dernières nouvelles ?",
            "Qu'est-ce qui s'est passé aujourd'hui dans le monde ?",
            "Y a-t-il eu un événement important récemment ?",
            "What's the latest news today?",
            "Any breaking news right now?",
            # ── Prix, marché, finance ──────────────────────────────────
            "Combien coûte un iPhone 17 ?",
            "Quel est le prix actuel du Bitcoin ?",
            "Quel est le cours de l'action Apple ?",
            "How much does a Tesla cost right now?",
            # ── Sport, scores ──────────────────────────────────────────
            "Quel est le score du match PSG-OM ?",
            "Qui a gagné Roland Garros 2025 ?",
            "Résultats du Super Bowl ?",
            "Who won the World Cup final?",
            # ── Météo ──────────────────────────────────────────────────
            "Quel temps fait-il demain à Paris ?",
            "Va-t-il pleuvoir ce week-end ?",
            "What's the weather like tomorrow?",
            # ── Personnes, rôles publics ───────────────────────────────
            "Qui est le président actuel de la République ?",
            "Qui est le PDG de Tesla en ce moment ?",
            "Who is the current CEO of OpenAI?",
            # ── Recommandations techniques (modèles, libs, papers) ─────
            "Recommande-moi un modèle NER en français",
            "Quel package Python pour faire du clustering ?",
            "Conseille-moi une lib pour le scraping",
            "Quel est le meilleur framework deep learning ?",
            "Cite-moi 3 papers sur les Vision Transformers",
            "Donne-moi des articles fondateurs sur les transformers",
            "Suggest a Python library for time series",
            "Which model is best for NER?",
            # ── Vérifications de faits ─────────────────────────────────
            "C'est vrai que l'Everest fait 8848 mètres ?",
            "Quand est sortie la version 3.12 de Python ?",
            "Quelle est la date de sortie de PyTorch 2.6 ?",
            # ── Disponibilité, sortie de produits ──────────────────────
            "Est-ce que GPT-5 est sorti ?",
            "Quand sort le prochain Zelda ?",
            "Is the new MacBook available yet?",
        ],
    ),
    Route(
        name="python",
        threshold=0.58,  # un peu plus strict, beaucoup de fausses pistes possibles
        examples=[
            # ════════════════════════════════════════════════════════════
            # V5.8.0 — Extension multi-domaines (calcul + analyse + valid + conv)
            # ════════════════════════════════════════════════════════════
            # ── Calculs explicites (existant V5.1) ─────────────────────
            "Calcule 17 × 23 + 89 ÷ 11",
            "Combien font 2 puissance 32 ?",
            "Quelle est la racine carrée de 1764 ?",
            "Calcule la moyenne de 12, 18, 23, 41, 7",
            "Évalue cette expression : (3+4)*2^3",
            "Quel est le résultat de 145 modulo 7 ?",
            "Compute the variance of these numbers",
            # ── Exécution / génération de code ─────────────────────────
            "Exécute ce code Python pour moi",
            "Lance ce script et dis-moi ce que ça donne",
            "Run this Python snippet",
            "Test ce bout de code et donne-moi la sortie",
            "Évalue cette fonction Python",
            "Écris un script qui calcule la suite de Fibonacci jusqu'à 100",
            "Code-moi une fonction qui trouve les nombres premiers entre 1 et 50",
            "Génère et exécute un script qui compte les voyelles dans un mot",
            "Fais-moi tourner un calcul de factorielle",
            # ── Plotting / visualisation ───────────────────────────────
            "Trace le graphe de la fonction sin(x) entre 0 et 2*pi",
            "Plot une courbe gaussienne",
            "Génère un histogramme de ces valeurs",
            "Trace un scatter plot avec ces données",
            "Crée une figure matplotlib qui montre x²",
            # ── V5.8.0 DOMAINE 2 : Analyse de données (listes inline) ──
            "Voici une liste : [12.3, 14.5, 11.7, 13.2, 12.9] — donne-moi les stats",
            "Calcule moyenne, médiane et écart-type de ces valeurs : 23, 45, 12, 67, 34",
            "Analyse cette série : 0.85, 0.92, 0.78, 0.81, 0.88, 0.95",
            "Stats descriptives sur ces données",
            "Distribution de ces nombres",
            "Histogramme avec ces valeurs",
            "Quels sont les outliers dans cette série ?",
            "Boxplot de ces mesures",
            "Donne-moi le min, max et quartiles",
            "Compare ces deux séries de valeurs statistiquement",
            "Analyse de variance entre ces groupes",
            "Régression linéaire sur ces points x,y",
            "Corrélation entre ces deux variables",
            "Compute statistics for this dataset",
            "Find outliers in this series",
            # ── V5.8.0 DOMAINE 1 : Validation arithmétique exacte ──────
            "Combien de jours entre le 15 mars 2024 et le 30 juin 2026 ?",
            "Quel jour de la semaine était le 4 juillet 1776 ?",
            "Combien d'heures dans 3.5 années ?",
            "Date dans 90 jours à partir d'aujourd'hui",
            "Quelle date sera dans 6 mois ?",
            "Est-ce que 982451653 est un nombre premier ?",
            "Décompose 1024 en facteurs premiers",
            "Plus grand commun diviseur de 360 et 420",
            "PPCM de 12, 18 et 30",
            "Combien de combinaisons de 5 parmi 49 ?",
            "Quel est le 50e nombre de Fibonacci ?",
            "Factorielle de 20",
            "Pourcentage : 47 sur 156, c'est combien ?",
            "Augmentation en % entre 1250 et 1830",
            "How many days between these two dates?",
            "Is this number prime?",
            # ── V5.8.0 DOMAINE 4 : Conversions utilitaires ─────────────
            "Convertis 145°F en Celsius",
            "75 kg en livres",
            "Convertis 12 miles en kilomètres",
            "Combien de litres dans 3 gallons US ?",
            "Décode ce base64 : SGVsbG8gV29ybGQ=",
            "Encode 'hello world' en base64",
            "URL-encode cette chaîne : hello world & test=1",
            "Convertis 255 en hexadécimal",
            "0xFF en décimal",
            "Convertis 1101101 binaire en décimal",
            "Reformate ce JSON pour qu'il soit lisible",
            "Parse cette date ISO : 2026-05-21T14:30:00Z",
            "Convertis ce timestamp Unix en date lisible : 1716300000",
            "Convertis en JSON compact",
            "Format this CSV row as JSON",
            "Convert this hex color to RGB",
            # ── V5.9.4 DOMAINE : Utilisation explicite d'une lib + listes ──
            # Patterns "Utilise <lib> pour ..." avec listes inline. Sans
            # ces exemples, le router donne un score Python trop bas
            # parce que l'attention sémantique est captée par le nom
            # de lib plutôt que par le verbe d'action.
            "Utilise seaborn pour faire un boxplot de [1, 2, 3], [3, 4, 5]",
            "Utilise pingouin pour calculer un test t entre ces groupes",
            "Utilise matplotlib pour tracer ces données",
            "Utilise numpy pour calculer la moyenne",
            "Utilise pandas pour analyser ce CSV",
            "Utilise scipy pour faire une régression",
            "Use seaborn to plot a violin chart",
            "Use pandas to compute these stats",
            "Use matplotlib to draw this function",
            "Avec numpy, calcule la corrélation entre ces deux séries",
            "Avec pingouin, fais une ANOVA sur ces groupes",
        ],
    ),
    Route(
        # V6.0.0-rc — Route MCP filesystem : pour les demandes qui
        # impliquent un fichier du workspace (lecture, listing, écriture).
        # Threshold 0.60 (assez strict) car les fausses pistes sont
        # possibles (ex : "lis l'article" sans rapport au workspace).
        # En cas d'ambiguïté, fallback sur le LLM dispatcher.
        name="mcp",
        threshold=0.60,
        examples=[
            # ── Lecture explicite d'un fichier nommé ──────────────────
            "Lis le fichier sales.csv",
            "Ouvre rapport.md et résume-le",
            "Que contient data.json ?",
            "Read sales.csv from the workspace",
            "Affiche le contenu de notes.txt",
            "Charge le CSV que je viens d'uploader",
            "Analyse le fichier que j'ai déposé",
            "Donne-moi un aperçu de mon dataset",
            # ── Listage / exploration du workspace ────────────────────
            "Liste les fichiers de mon workspace",
            "Qu'est-ce qu'il y a dans mon workspace ?",
            "Montre-moi tous les fichiers que j'ai stockés",
            "Quels documents ai-je actuellement ?",
            "What files do I have in the workspace?",
            "List my workspace contents",
            # ── Référence implicite à un fichier déposé ───────────────
            "Analyse le rapport que je viens de t'envoyer",
            "Regarde le fichier que j'ai mis dans le workspace",
            "Le PDF que j'ai uploadé, qu'est-ce qu'il dit ?",
            "Prends le CSV de mon workspace et calcule les stats",
            "Le document partagé, peux-tu le résumer ?",
            # ── Écriture / sauvegarde explicite ───────────────────────
            "Sauvegarde le résultat dans un fichier",
            "Écris ce code dans un fichier Python",
            "Crée un fichier markdown avec ce contenu",
            "Génère-moi un script et mets-le dans le workspace",
            "Save the analysis as a CSV file",
            "Écris ces notes dans un fichier .md",
            "Exporte ces données en CSV",
            # ── Suppression / gestion ─────────────────────────────────
            "Supprime le fichier obsolete.txt",
            "Renomme draft.md en final.md",
            "Efface tous les fichiers .tmp du workspace",
            # ── Combinaisons (workspace + analyse) ────────────────────
            "Charge sales.csv et fais une régression linéaire",
            "Prends le fichier que j'ai déposé et trace un histogramme",
            "Lis mon CSV et donne-moi la moyenne des ventes",
            "Open data.csv and compute summary statistics",
        ],
    ),
    Route(
        name="none",
        threshold=0.50,  # plus laxiste, c'est le bucket par défaut
        examples=[
            # ── Conversations, salutations ─────────────────────────────
            "Bonjour Rune",
            "Salut, comment vas-tu ?",
            "Hello",
            "Merci beaucoup",
            "C'est gentil de ta part",
            "Bonne journée",
            # ── Explications de concepts stables ───────────────────────
            "Explique-moi le NER",
            "C'est quoi le machine learning ?",
            "Qu'est-ce qu'un transformer en NLP ?",
            "Comment fonctionne la backpropagation ?",
            "Explique-moi la différence entre supervisé et non supervisé",
            "What is overfitting?",
            # ── Mémoire interne ────────────────────────────────────────
            "Tu te souviens de notre dernière discussion ?",
            "On a déjà parlé de ça non ?",
            "Tu m'as dit quelque chose là-dessus tout à l'heure",
            "Do you remember what we discussed?",
            # ── Avis subjectif, créativité ─────────────────────────────
            "Qu'est-ce que tu penses de la philosophie stoïcienne ?",
            "Donne-moi ton avis sur l'éthique de l'IA",
            "Écris-moi un haïku sur la mer",
            "Raconte-moi une histoire courte",
            "Aide-moi à reformuler ce paragraphe",
            "Traduis cette phrase en anglais",
            # ── Aide rédactionnelle ────────────────────────────────────
            "Corrige cette phrase",
            "Reformule plus poliment",
            "Améliore ce mail",
            # ── Conseils non-techniques (mug, vin, film, etc.) ─────────
            "Recommande-moi un bon vin pour ce soir",
            "Conseille-moi un film à regarder",
            "Suggère-moi un restaurant sympa",
            "Donne-moi une idée de cadeau pour ma mère",
        ],
    ),
]


# ── Le routeur ────────────────────────────────────────────────────────


class SemanticRouter:
    """Cosine-similarity router over pre-computed route examples.

    Lazy-loads the embedding model on first ``classify()`` call.
    Subsequent calls are pure numpy (no model load), so latency is
    bounded by the encode of the query itself : ~10-30 ms CPU on
    short questions.

    Thread-safe : the model load is guarded, and numpy operations
    are inherently per-call (no shared mutable state).
    """

    # Try the multilingual model first, fall back to the EN-only one
    # already used by KG if the multilingual download fails.
    _PRIMARY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, routes: list[Route] | None = None) -> None:
        self._routes = routes if routes is not None else ROUTES
        self._model = None
        self._model_name: str | None = None
        self._lock = threading.Lock()
        self._warmed = False
        # Stats for monitoring
        self.calls = 0
        self.routes_taken: dict[str, int] = {}
        self.ambiguous_count = 0

    # ── Lifecycle ──────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load the embedding model. Idempotent and thread-safe."""
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._PRIMARY_MODEL, device="cpu",
                )
                self._model_name = self._PRIMARY_MODEL
                log.info("Semantic router loaded: %s", self._PRIMARY_MODEL)
                return True
            except Exception as exc:
                log.warning(
                    "Primary embedding model %s failed (%s), trying fallback",
                    self._PRIMARY_MODEL, exc,
                )
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._FALLBACK_MODEL, device="cpu",
                )
                self._model_name = self._FALLBACK_MODEL
                log.info("Semantic router loaded (fallback): %s", self._FALLBACK_MODEL)
                return True
            except Exception as exc:
                log.error("Both embedding models failed: %s", exc)
                self._model = None
                return False

    def warm_up(self) -> bool:
        """Pre-compute embeddings for all route examples.

        Call this once at startup so the first user query doesn't
        eat a ~500ms cold-start hit. Returns True on success.
        """
        if self._warmed:
            return True
        if not self._ensure_model_loaded():
            return False
        with self._lock:
            if self._warmed:
                return True
            try:
                for route in self._routes:
                    embs = self._model.encode(
                        route.examples,
                        convert_to_numpy=True,
                        normalize_embeddings=True,  # cosine = dot product
                        show_progress_bar=False,
                    )
                    route._embeddings = embs
                self._warmed = True
                log.info(
                    "Semantic router warmed up: %d routes, %d examples total",
                    len(self._routes),
                    sum(len(r.examples) for r in self._routes),
                )
                return True
            except Exception as exc:
                log.error("Router warm-up failed: %s", exc)
                return False

    # ── Inference ──────────────────────────────────────────────────────

    def classify(
        self, query: str,
    ) -> tuple[str | None, float, dict[str, float]]:
        """Route a query to the best matching tool.

        Returns
        -------
        tuple[str | None, float, dict[str, float]]
            ``(route_name, confidence, all_scores)``.

            - ``route_name`` : the name of the chosen route (e.g. "web",
              "python", "none") if confidence ≥ route.threshold.
              ``None`` if the query falls in an ambiguous zone — caller
              should fall back to LLM dispatcher.
            - ``confidence`` : best score (cosine similarity, 0-1).
            - ``all_scores`` : full mapping ``{route_name: score}`` for
              transparency, debugging, and downstream tie-breaks.
        """
        self.calls += 1
        if not self._warmed and not self.warm_up():
            log.warning("Router not warmed — returning ambiguous")
            return None, 0.0, {}

        try:
            q_emb = self._model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            log.warning("Router encode failed: %s", exc)
            return None, 0.0, {}

        # Compute max cosine similarity per route.
        scores: dict[str, float] = {}
        for route in self._routes:
            if route._embeddings is None:
                continue
            # All embeddings are normalised, so dot product = cosine.
            sims = route._embeddings @ q_emb
            scores[route.name] = float(np.max(sims))

        if not scores:
            return None, 0.0, {}

        # Winner = highest score above its threshold.
        winner: str | None = None
        winner_score = 0.0
        for route in self._routes:
            s = scores.get(route.name, 0.0)
            if s >= route.threshold and s > winner_score:
                winner = route.name
                winner_score = s

        if winner is None:
            self.ambiguous_count += 1
            return None, max(scores.values()), scores

        self.routes_taken[winner] = self.routes_taken.get(winner, 0) + 1
        return winner, winner_score, scores

    # ── Monitoring ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return router statistics for monitoring."""
        return {
            "model": self._model_name,
            "warmed": self._warmed,
            "calls": self.calls,
            "routes_taken": dict(self.routes_taken),
            "ambiguous": self.ambiguous_count,
            "ambiguous_rate": (
                self.ambiguous_count / max(1, self.calls)
            ),
        }


# Module-level singleton — partagé entre instances Hippocampe pour
# éviter de recharger le modèle d'embeddings à chaque session.
_router_singleton: SemanticRouter | None = None
_singleton_lock = threading.Lock()


def get_router() -> SemanticRouter:
    """Get the global SemanticRouter singleton."""
    global _router_singleton
    if _router_singleton is None:
        with _singleton_lock:
            if _router_singleton is None:
                _router_singleton = SemanticRouter()
    return _router_singleton
