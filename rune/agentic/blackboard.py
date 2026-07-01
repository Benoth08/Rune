"""Tableau noir de mission — état partagé, persistant et anti-amnésie.

Une seule brique sert les deux modes :
- mono-agent : une section unique = le carnet de bord de l'agent (ce qui a
  marché, ce qui a échoué ET POURQUOI), relu à chaque étape pour qu'il ne
  rabâche pas des pistes mortes ;
- multi-agent : N sections (une par sous-agent) + une section ``_contract``
  écrite par le lead. Chaque agent ÉCRIT seulement sa section et LIT (seule
  lecture) celles des autres — pas de verrou, pas d'écriture concurrente.

Le détail riche persiste sur disque (``blackboard.json``) pour l'audit ; le
prompt ne reçoit qu'un résumé BORNÉ (le contexte d'un 14B est limité).
Module pur (pas de torch / pas de modèle) → entièrement testable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_MAX_FAILS = 8          # échecs gardés par section (les plus récents)
_MAX_WINS = 6
_MAX_NOTES = 6
_TXT = 240              # cap par entrée (raison d'échec, note…)


def _clip(s: str, n: int = _TXT) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


class _Section:
    """La zone d'UN agent. Seul son propriétaire y écrit."""

    def __init__(self, owner: str, goal: str = "") -> None:
        self.owner = owner
        self.goal = goal
        self.interface = ""        # ce que la section expose aux autres
        self.status = "en_cours"   # en_cours | vert | bloque
        self.wins: list[str] = []
        self.fails: list[dict] = []   # {what, why}
        self.notes: list[str] = []    # pour les autres agents
        self.files: list[str] = []

    def to_dict(self) -> dict:
        return {"owner": self.owner, "goal": self.goal,
                "interface": self.interface, "status": self.status,
                "wins": self.wins, "fails": self.fails,
                "notes": self.notes, "files": self.files}

    @classmethod
    def from_dict(cls, d: dict) -> "_Section":
        s = cls(d.get("owner", "agent"), d.get("goal", ""))
        s.interface = d.get("interface", "")
        s.status = d.get("status", "en_cours")
        s.wins = list(d.get("wins", []))
        s.fails = list(d.get("fails", []))
        s.notes = list(d.get("notes", []))
        s.files = list(d.get("files", []))
        return s


class MissionBlackboard:
    """Tableau noir d'une mission. ``path`` = blackboard.json de la mission."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.contract = ""                       # partagé, écrit par le lead
        self.sections: dict[str, _Section] = {}
        self._ts = time.time()

    # ── ownership ────────────────────────────────────────────────────
    def ensure(self, owner: str, goal: str = "") -> _Section:
        sec = self.sections.get(owner)
        if sec is None:
            sec = _Section(owner, goal)
            self.sections[owner] = sec
        elif goal and not sec.goal:
            sec.goal = goal
        return sec

    # ── écritures (par le propriétaire de la section) ────────────────
    def record_win(self, owner: str, what: str) -> None:
        sec = self.ensure(owner)
        sec.wins.append(_clip(what))
        del sec.wins[:-_MAX_WINS]
        self.save()

    def record_fail(self, owner: str, what: str, why: str = "") -> None:
        """Consigne une tentative qui a ÉCHOUÉ et POURQUOI. Dédupe les
        répétitions exactes (le même échec relogué = bruit)."""
        sec = self.ensure(owner)
        entry = {"what": _clip(what), "why": _clip(why)}
        if sec.fails and sec.fails[-1] == entry:
            return
        sec.fails.append(entry)
        del sec.fails[:-_MAX_FAILS]
        self.save()

    def note(self, owner: str, text: str) -> None:
        sec = self.ensure(owner)
        sec.notes.append(_clip(text))
        del sec.notes[:-_MAX_NOTES]
        self.save()

    def set_status(self, owner: str, status: str) -> None:
        self.ensure(owner).status = status
        self.save()

    def set_interface(self, owner: str, interface: str) -> None:
        self.ensure(owner).interface = _clip(interface, 600)
        self.save()

    def set_files(self, owner: str, files: list[str]) -> None:
        self.ensure(owner).files = list(files)[:20]
        self.save()

    def set_contract(self, text: str) -> None:
        self.contract = _clip(text, 1500)
        self.save()

    # ── rendu pour le prompt (BORNÉ) ─────────────────────────────────
    def render_for(self, owner: str, peers: bool = True) -> str:
        """Bloc à injecter : la section de ``owner`` (détaillée) + un résumé
        lecture-seule des autres (surtout leurs échecs et notes)."""
        own = self.sections.get(owner)
        out: list[str] = ["[Tableau noir de la mission]"]
        if self.contract:
            out.append(f"Contrat partagé : {self.contract}")
        if own is not None:
            out.append(f"— TA section ({owner}) — statut: {own.status}")
            if own.wins:
                out.append("  Acquis (vérifiés) : "
                           + " | ".join(own.wins[-_MAX_WINS:]))
            if own.fails:
                out.append("  DÉJÀ ESSAYÉ ET ÉCHOUÉ (n'y retourne pas) :")
                out += [f"    • {f['what']}"
                        + (f" → {f['why']}" if f.get("why") else "")
                        for f in own.fails[-_MAX_FAILS:]]
        if peers:
            for name, sec in self.sections.items():
                if name == owner:
                    continue
                bits = [f"statut: {sec.status}"]
                if sec.interface:
                    bits.append(f"expose: {sec.interface}")
                if sec.notes:
                    bits.append("notes: " + " | ".join(sec.notes[-3:]))
                if sec.fails:
                    bits.append("a échoué sur: "
                                + " | ".join(f["what"] for f in sec.fails[-3:]))
                out.append(f"— section {name} — " + " ; ".join(bits))
        return "\n".join(out) + "\n"

    # ── persistance ──────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"ts": self._ts, "contract": self.contract,
                "sections": {k: v.to_dict() for k, v in self.sections.items()}}

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                encoding="utf-8")
        except Exception:  # noqa: BLE001 — le blackboard ne doit jamais crasher un run
            pass

    @classmethod
    def load(cls, path: str | Path) -> "MissionBlackboard":
        bb = cls(path)
        p = Path(path)
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                bb.contract = d.get("contract", "")
                bb.sections = {k: _Section.from_dict(v)
                               for k, v in d.get("sections", {}).items()}
            except Exception:  # noqa: BLE001
                pass
        return bb
