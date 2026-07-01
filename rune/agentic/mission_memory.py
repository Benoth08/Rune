"""Mémoire épisodique de l'agent — un journal des missions passées.

Volontairement ISOLÉ (un fichier, un store JSON) pour rester facile à
désactiver/supprimer. Chaque entrée = un résumé d'une mission terminée. Le
rappel se fait par similarité sémantique (même embedder que le chat, injecté),
avec repli lexical — exactement le schéma déjà éprouvé pour la procédurale.

Rien ici n'est branché tant que les flags `agent_memory_*` sont OFF.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Episode:
    task: str
    summary: str
    source: str = ""              # slug de la mission (pour tracer/purger)
    ts: float = field(default_factory=time.time)


class MissionMemory:
    """Journal append-only des missions, avec rappel sémantique optionnel."""

    def __init__(self, path: str | Path, max_entries: int = 500) -> None:
        self.path = Path(path)
        self.max_entries = max_entries
        self.episodes: list[Episode] = []
        self._load()

    # — écriture —
    def add(self, task: str, summary: str, source: str = "") -> None:
        task = (task or "").strip()
        summary = (summary or "").strip()
        if not task or not summary:
            return
        self.episodes.append(Episode(task=task, summary=summary, source=source))
        # borne dure : on garde les plus récents
        if len(self.episodes) > self.max_entries:
            self.episodes = self.episodes[-self.max_entries:]
        self._save()

    # — lecture —
    def recall(self, task: str, embed_fn=None, k: int = 2,
               min_cos: float = 0.25) -> list[Episode]:
        """Top-k épisodes pertinents. Sémantique si `embed_fn` fourni
        (texte → tenseur), sinon repli lexical. Liste vide si rien de pertinent."""
        if not self.episodes or not (task or "").strip():
            return []
        # exclut une éventuelle reprise exacte de la même tâche
        cands = [e for e in self.episodes if e.task.strip() != task.strip()]
        if not cands:
            return []
        if embed_fn is not None:
            sem = self._recall_semantic(task, cands, embed_fn, k, min_cos)
            if sem is not None:
                return sem
        return self._recall_keyword(task, cands, k)

    def _recall_semantic(self, task, cands, embed_fn, k, min_cos):
        try:
            import torch
        except Exception:  # noqa: BLE001
            return None
        try:
            q = embed_fn(task)
        except Exception:  # noqa: BLE001
            return None
        if q is None:
            return None
        q = q.view(1, -1)
        scored = []
        for e in cands:
            try:
                d = embed_fn(e.task + " " + e.summary)
                if d is None:
                    continue
                cos = torch.nn.functional.cosine_similarity(q, d.view(1, -1)).item()
            except Exception:  # noqa: BLE001
                continue
            if cos >= min_cos:
                scored.append((cos, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _c, e in scored[:k]]

    def _recall_keyword(self, task, cands, k):
        toks = set(re.findall(r"\w{4,}", task.lower()))
        if not toks:
            return []
        scored = []
        for e in cands:
            et = set(re.findall(r"\w{4,}", (e.task + " " + e.summary).lower()))
            ov = len(toks & et)
            if ov:
                scored.append((ov, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _o, e in scored[:k]]

    # — persistance —
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.episodes = [Episode(**d) for d in raw]
        except Exception:  # noqa: BLE001
            self.episodes = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([asdict(e) for e in self.episodes],
                           ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
