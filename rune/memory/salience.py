"""Three-cascade salience filter for cognitive memory gating.

N1: rule-based noise rejection
N2: heuristic scoring (proper nouns, numbers, entropy)
N3: redundancy check against recent embedding cache
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass

import torch

from rune.config import (
    SALIENCE_MIN_LENGTH,
    SALIENCE_MIN_SCORE,
    SALIENCE_REDUNDANCY_THRESHOLD,
)

log = logging.getLogger("lythea.memory.salience")

NOISE_WORDS = frozenset({
    "ok", "oui", "non", "merci", "salut", "bonjour", "hey", "hi", "hello",
    "yes", "no", "thanks", "bye", "ciao", "lol", "mdr", "haha", "hmm",
    "ah", "oh", "eh", "bah", "bof", "ouais", "nope", "yep", "cool",
    "d'accord", "okay",
})


@dataclass
class SalienceResult:
    """Result of the salience cascade."""

    passed: bool
    score: float
    stage: str
    reason: str = ""


class SalienceFilter:
    """Three-cascade filter determining if a message is worth memorizing.

    Parameters
    ----------
    min_length : int
        Minimum character length (N1 gate).
    min_score : float
        Minimum heuristic score to pass N2.
    redundancy_threshold : float
        Cosine similarity above which a message is redundant (N3).
    cache_size : int
        Number of recent embeddings to keep for N3.
    """

    def __init__(
        self,
        min_length: int = SALIENCE_MIN_LENGTH,
        min_score: float = SALIENCE_MIN_SCORE,
        redundancy_threshold: float = SALIENCE_REDUNDANCY_THRESHOLD,
        cache_size: int = 20,
    ) -> None:
        self.min_length = min_length
        self.min_score = min_score
        self.redundancy_threshold = redundancy_threshold
        self._embed_cache: deque[torch.Tensor] = deque(maxlen=cache_size)

    # ── N1: rule-based rejection ───────────────────────────────────────

    def _n1_rules(self, text: str) -> SalienceResult | None:
        """Reject trivially non-salient messages."""
        stripped = text.strip()
        if len(stripped) < self.min_length:
            return SalienceResult(False, 0.0, "N1", "too short")
        if stripped.lower() in NOISE_WORDS:
            return SalienceResult(False, 0.0, "N1", "noise word")
        words = stripped.split()
        if len(words) <= 2 and all(w.lower() in NOISE_WORDS for w in words):
            return SalienceResult(False, 0.0, "N1", "noise phrase")
        return None

    # ── N2: heuristic scoring ──────────────────────────────────────────

    # V5.6.1 — patterns self-disclosure : phrases qui annoncent un fait
    # personnel saillant ("je m'appelle X", "j'habite à Y", "j'ai N ans",
    # "je suis né en Y", "je travaille chez Z"). Sans ce bonus, ces
    # phrases en minuscule sans noms propres capitalisés tombent sous le
    # seuil N2 et sont rejetées AVANT que les fallbacks V5.5.1-9 puissent
    # tourner — le KG perd alors le fait.
    _SELF_DISCLOSURE_HINT = re.compile(
        r"\b("
        r"je\s+m['’]?appelle"
        r"|moi\s*[,]?\s*c['’]?est"
        r"|mon\s+(?:nom|prénom)\s+(?:est|c['’]?est)"
        r"|appelle[-\s]moi"
        r"|j['’]?habite\s+(?:à|en|au|aux|dans|sur)"
        r"|je\s+vis\s+(?:à|en|au|aux)"
        r"|je\s+suis\s+(?:née?|basée?)\s+(?:à|en|au|aux)"
        r"|je\s+travaille\s+(?:chez|pour|à|au|aux)"
        r"|je\s+bosse\s+(?:chez|pour|à|au|aux)"
        r"|je\s+suis\s+(?:employée?|salariée?)\s+(?:chez|à)"
        r"|j['’]?ai\s+\d+\s+an"
        r"|je\s+suis\s+née?\s+en\s+\d{4}"
        r"|né(?:e)?\s+en\s+\d{4}"
        r"|my\s+name\s+is"
        r"|call\s+me"
        r"|i['’]?m\s+\w"
        r"|i\s+am\s+\w"
        r"|i\s+live\s+in"
        r"|i\s+work\s+(?:at|for)"
        r"|i\s+was\s+born\s+in\s+\d{4}"
        r")",
        re.IGNORECASE,
    )

    def _n2_heuristic(self, text: str) -> float:
        """Score based on linguistic features.

        Returns
        -------
        float
            Score in [0, 1].
        """
        score = 0.0
        words = text.split()
        n = len(words)

        # Proper nouns (capitalized words not at sentence start)
        proper = sum(
            1 for i, w in enumerate(words)
            if i > 0 and w[0].isupper() and len(w) > 1
        )
        score += min(proper * 0.15, 0.45)

        # Numbers, dates, quantities
        nums = len(re.findall(r'\d+[.,]?\d*', text))
        score += min(nums * 0.1, 0.3)

        # Length bonus (longer = more informative, diminishing returns)
        score += min(n / 50, 0.25)

        # Questions carry moderate salience
        if text.strip().endswith("?"):
            score += 0.1

        # Technical markers
        tech_markers = ("::", "->", "=>", "def ", "class ", "import ", "http")
        if any(m in text for m in tech_markers):
            score += 0.15

        # Character diversity (entropy proxy)
        unique_chars = len(set(text.lower()))
        score += min(unique_chars / 60, 0.15)

        # V5.6.1 — Bonus self-disclosure : les phrases qui annoncent un
        # fait personnel doivent toujours passer N2, même en minuscule
        # et sans noms propres. Bonus généreux (+0.4) pour garantir
        # qu'elles dépassent le min_score (typiquement 0.3).
        if self._SELF_DISCLOSURE_HINT.search(text):
            score += 0.4

        return min(score, 1.0)

    # ── N3: redundancy check ───────────────────────────────────────────

    def _n3_redundancy(self, embedding: torch.Tensor | None) -> bool:
        """Check if the message is redundant with recent cache.

        Returns
        -------
        bool
            True if redundant (should skip).
        """
        if embedding is None or len(self._embed_cache) == 0:
            return False

        emb = embedding.view(1, -1)
        for cached in self._embed_cache:
            c = cached.view(1, -1)
            cos = torch.nn.functional.cosine_similarity(emb, c).item()
            if cos > self.redundancy_threshold:
                return True
        return False

    # ── Full cascade ───────────────────────────────────────────────────

    def evaluate(self, text: str, embedding: torch.Tensor | None = None) -> SalienceResult:
        """Run the full salience cascade.

        Parameters
        ----------
        text : str
            Input message.
        embedding : torch.Tensor, optional
            Pre-computed embedding for N3 redundancy check.

        Returns
        -------
        SalienceResult
            Whether the message passes and its score.
        """
        # N1
        n1 = self._n1_rules(text)
        if n1 is not None:
            return n1

        # N2
        score = self._n2_heuristic(text)
        if score < self.min_score:
            return SalienceResult(False, score, "N2", f"score {score:.2f} < {self.min_score}")

        # N3
        if self._n3_redundancy(embedding):
            return SalienceResult(False, score, "N3", "redundant")

        # Passed — update cache
        if embedding is not None:
            self._embed_cache.append(embedding.detach().cpu())

        return SalienceResult(True, score, "passed")

    def reset(self) -> None:
        """Clear the embedding cache (session switch)."""
        self._embed_cache.clear()
