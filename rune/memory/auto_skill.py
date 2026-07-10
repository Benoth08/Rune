"""AutoSkill — création autonome de compétences après succès vérifié.

Inspiration Rune
------------------
Rune génère un fichier SKILL.md au format agentskills.io après
chaque succès. Le fichier contient : trigger, approach, validation.

Ce qu'Rune fait mieux
------------------------------
1. **Métacognition filtre le bruit** : on n'extrait un skill QUE si
   - ``verifier.ok == True`` (succès)
   - ``metacognition.confidence_label`` ∈ {très_certaine, certaine}
   - ``doubt_index < 0.4`` (pas une réponse incertaine)
   - ``applied_count`` local ≥ 1 (pattern pas vu pour la 1ère fois)

   Rune extrait bêtement à chaque succès → plein de skills
   dupliqués ou bruités. Lythea extrait que les patterns fiables.

2. **Anti-patterns** : chaque Skill porte une liste ``anti_patterns``
   qui l'enrichit au fil des échecs. Pas juste "comment faire" mais
   aussi "comment ne pas faire".

3. **Embedding du trigger** : on stocke l'embedding sémantique du
   trigger pour faire du retrieval par similarité (pas juste du
   pattern matching).

4. **Validation explicite** : critères observables qu'on peut
   rejouer pour vérifier qu'un skill est applicable.

Format SKILL.md (compatible agentskills.io)
-------------------------------------------
    ---
    id: skill_<hash>
    trigger: "recommande-moi un modèle NER en français"
    trigger_embedding: [0.12, -0.34, ...]  # pour retrieval sémantique
    approach:
      - "Web search 'NER french models spacy flair'"
      - "Vérifier chaque nom cité dans les sources"
      - "Lier [N] à chaque modèle mentionné"
    validation:
      - "Aucune variante inventée"
      - "Citations [N] présentes à chaque nom"
    anti_patterns:
      - "Ne pas citer spaCy sans vérifier la page models fr"
    failure_count: 0
    success_count: 1
    confidence: 0.7
    created_at: 1721234567.89
    last_used_at: 1721998765.43
    source_episodes: ["ep_abc"]
    ---

Lifecycle
---------
1. **Extraction** : après une génération réussie (verifier OK +
   métacognition OK), le SkillExtractor propose 0-1 skill. Si un
   skill similaire existe déjà (cosine > 0.85), on incrémente
   success_count au lieu d'en créer un nouveau.
2. **Rejeu** : au début de chaque tour, on calcule l'embedding du
   message utilisateur et on cherche le skill le plus similaire.
   Si cosine > 0.75, on injecte l'approche dans le system prompt.
3. **Échec** : si le skill appliqué mène à un échec verifier, on
   incrémente failure_count et on extrait un anti-pattern.
4. **Forgetting** : skills non utilisés depuis 90 jours avec
   success_count < 3 → archivés (pas supprimés).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("rune.memory.auto_skill")


# ── Garde-fous sécurité (hérités de Lythea procedural.py) ─────────────
# Patterns refusés à l'extraction — protègent contre les injections qui
# tenteraient de modifier le comportement fondamental de l'agent.
# Les skills et anti-patterns sont RÉINJECTÉS dans le prompt système à
# chaque tour similaire : un contenu malveillant stocké ici devient une
# injection persistante rejouée indéfiniment. D'où une liste large,
# bilingue FR/EN, quitte à sur-rejeter (un skill est un bonus, pas une
# fonctionnalité critique — mieux vaut en perdre un légitime que
# d'empoisonner le prompt).
_FORBIDDEN_PATTERNS = [
    # Manipulation / dissimulation
    re.compile(r"\b(mentir|tromper|cacher|dissimuler|deceive|manipulat)\b", re.IGNORECASE),
    # Contournement de consignes (FR + EN). Le verbe SEUL est trop large
    # (« ne pas oublier le cas de base », « ignorer les warnings pip »
    # sont des contenus légitimes) : on exige le verbe suivi, à courte
    # distance, d'un nom de gouvernance (instructions, consignes, règles…).
    re.compile(
        r"\b(ignore[rz]?|oublie[rz]?|outrepass\w*|disregard|bypass|overrid\w*)\b"
        r"[^.!?\n]{0,40}"
        r"\b(instructions?|consignes?|r[èe]gles?|directives?|rules?|prompts?"
        r"|guidelines?|garde[- ]?fous?|safeguards?|restrictions?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(nouvelles?\s+consignes?|instructions?\s+pr[ée]c[ée]dentes?|previous\s+instructions?|new\s+instructions?)\b", re.IGNORECASE),
    # Jailbreak / changement d'identité
    re.compile(r"\b(jailbreak|prompt[ -]injection|DAN mode|developer mode|mode d[ée]veloppeur)\b", re.IGNORECASE),
    re.compile(r"\b(change[rz]? d'identit|deviens|pretend|roleplay as|act as|you are now|tu es (?:maintenant|d[ée]sormais))\b", re.IGNORECASE),
    # Destruction de données
    re.compile(r"\b(supprime[rz]?|delete|drop)\s+(les?|all|every|tous)", re.IGNORECASE),
    re.compile(r"\brm\s+-rf?\b|\bsudo\s+rm\b", re.IGNORECASE),
    # Exfiltration / manipulation du system prompt
    re.compile(r"\b(syst[èe]me|system prompt|consignes)\b.*\b(modif|reveal|exposer?|divulgu)", re.IGNORECASE),
    # Usurpation de rôle dans le texte injecté (spoof de structure chat)
    re.compile(r"(?m)^\s*(system|assistant|user)\s*:", re.IGNORECASE),
    re.compile(r"<\|im_(start|end)\|>|\[INST\]|\[/INST\]|<<SYS>>", re.IGNORECASE),
    # Exécution shell dangereuse embarquée
    re.compile(r"curl\s+[^|]*\|\s*(ba)?sh\b|wget\s+[^|]*\|\s*(ba)?sh\b", re.IGNORECASE),
    re.compile(r"\bbase64\s+(-d|--decode)\b", re.IGNORECASE),
]


def _content_is_forbidden(text: str) -> bool:
    """Vrai si le texte matche un pattern interdit (usage partagé)."""
    if not text:
        return False
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(text):
            return True
    return False


def sanitize_for_prompt(text: str, max_len: int = 300) -> str:
    """Neutralise un texte destiné à être réinjecté dans le prompt.

    Défense en profondeur au point d'INJECTION (en plus du filtrage au
    point de STOCKAGE) : aplatit les sauts de ligne (empêche le spoof
    de structure « \\nSystem: … »), retire les tokens de template de
    chat, et borne la longueur. Appliqué aux champs issus de contenu
    utilisateur ou LLM avant insertion dans le working memory / prompt.
    """
    if not text:
        return ""
    out = str(text)
    # Tokens de template de chat (Qwen/Llama/ChatML)
    out = re.sub(r"<\|im_(start|end)\|>|\[INST\]|\[/INST\]|<<SYS>>|<\|[a-z_]+\|>", " ", out, flags=re.IGNORECASE)
    # Aplatis les sauts de ligne → un espace (empêche 'System:' en début de ligne)
    out = re.sub(r"[\r\n]+", " ", out)
    # Compacte les espaces
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out[:max_len]

# Bornes de taille — éviter les skills monstrueux qui bruitent le prompt
MAX_TRIGGER_CHARS = 300
MAX_APPROACH_STEPS = 8
MAX_APPROACH_STEP_CHARS = 200

# Sentinelle : un callback LLM peut renvoyer ceci pour signaler qu'il
# n'a PAS pu juger (ex: aucun modèle chargé), par opposition à None qui
# veut dire « épisode trivial, ne pas extraire de skill ». Permet à
# l'extracteur de retomber sur l'heuristique dans le premier cas mais
# de respecter le refus du LLM dans le second.
_LLM_UNAVAILABLE = {"_llm_unavailable": True}
MAX_VALIDATION_CHARS = 200
MAX_ANTI_PATTERN_CHARS = 200
MAX_ACTIVE_SKILLS = 50
MAX_TOTAL_SKILLS = 200


@dataclass
class Skill:
    """Une compétence auto-apprise (SKILL.md équivalent).

    Voir docstring module pour le lifecycle complet.
    """
    skill_id: str
    trigger: str  # description canonique du contexte d'activation
    trigger_embedding: list[float] = field(default_factory=list)
    approach: list[str] = field(default_factory=list)  # étapes
    validation: list[str] = field(default_factory=list)  # critères de succès
    anti_patterns: list[str] = field(default_factory=list)  # à éviter
    failure_count: int = 0
    success_count: int = 1
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    source_episodes: list[str] = field(default_factory=list)
    archived: bool = False
    # Rune — métadonnées de composition. Quand une skill est composée
    # (voir SkillComposer), metadata.composed = True et source_skill_ids
    # contient les IDs des skills sources.
    metadata: dict = field(default_factory=dict)

    # ── Helpers ──────────────────────────────────────────────────────

    def utility_score(self) -> float:
        """Score de ranking pour l'injection dans le prompt.

        Pondération : confidence × log(success_count+1) × fraîcheur.
        """
        days_since = (time.time() - self.last_used_at) / 86400
        freshness = 1.0
        if days_since > 30:
            freshness = max(0.3, 1.0 - (days_since - 30) / 60)
        return self.confidence * math.log1p(self.success_count) * freshness

    def is_reliable(self) -> bool:
        """True si le skill est considéré comme fiable.

        Critères : au moins 2 succès, taux d'échec < 30%, pas archivé.
        """
        if self.archived:
            return False
        total = self.success_count + self.failure_count
        if total < 2:
            return False
        if self.failure_count / total > 0.3:
            return False
        return True

    def as_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        """Sérialise au format SKILL.md (compatible agentskills.io)."""
        import yaml

        def _native(v):
            """Convertit numpy/torch scalars en types Python natifs.

            yaml.safe_dump refuse les np.float32/np.int64/torch.Tensor
            (« cannot represent an object »). Comme confidence et les
            timestamps peuvent venir de calculs numpy, on normalise ici.
            Sans ça, l'export MD échoue silencieusement (« Failed to
            export MD for … »).
            """
            # numpy / torch scalar → .item()
            if hasattr(v, "item") and not isinstance(v, (str, bytes)):
                try:
                    return v.item()
                except Exception:
                    pass
            if isinstance(v, float):
                return float(v)
            if isinstance(v, int):
                return int(v)
            if isinstance(v, (list, tuple)):
                return [_native(x) for x in v]
            return v

        frontmatter = {
            "id": str(self.skill_id),
            "trigger": str(self.trigger),
            "trigger_embedding_size": int(len(self.trigger_embedding)),
            "approach": [str(a) for a in self.approach],
            "validation": [str(v) for v in self.validation],
            "anti_patterns": [str(a) for a in self.anti_patterns],
            "failure_count": int(self.failure_count),
            "success_count": int(self.success_count),
            "confidence": round(float(self.confidence), 3),
            "created_at": float(self.created_at),
            "last_used_at": float(self.last_used_at),
            "source_episodes": [str(e) for e in self.source_episodes],
            "archived": bool(self.archived),
        }
        # L'embedding n'est pas sérialisé en clair dans le MD (trop gros).
        # On le stocke à côté dans un .json sibling.
        return (
            "---\n"
            + yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False)
            + "---\n"
        )


class AutoSkillStore:
    """Stockage persistant des Skills (JSON + MD).

    Persistence
    -----------
    - ``skills.json`` : index central (toutes les métadonnées + embeddings)
    - ``skills/<id>.md`` : un fichier SKILL.md par skill (format ouvert)
    - ``skills/<id>.emb`` : embedding binaire (numpy array savez)

    L'index JSON est la source de vérité pour les opérations read/write.
    Les fichiers MD sont une exportation lisible (et un point d'entrée
    pour des outils externes conformes agentskills.io).
    """

    def __init__(self, storage_dir: Path | str = "data/skills") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "exports").mkdir(exist_ok=True)
        self._index_path = self.storage_dir / "skills.json"
        self._skills: dict[str, Skill] = {}
        self._load()

    # ── API publique ──────────────────────────────────────────────────

    def add(self, skill: Skill) -> Skill:
        """Ajoute ou met à jour un skill.

        Si un skill similaire existe déjà (cosine > 0.85), on incrémente
        success_count au lieu d'en créer un nouveau.
        """
        # Garde-fou sécurité
        if not self._is_safe(skill):
            log.warning("Skill %s rejected: unsafe content", skill.skill_id)
            return skill

        # Garde-fou taille
        if len(skill.trigger) > MAX_TRIGGER_CHARS:
            skill.trigger = skill.trigger[:MAX_TRIGGER_CHARS]
        skill.approach = [
            s[:MAX_APPROACH_STEP_CHARS] for s in skill.approach[:MAX_APPROACH_STEPS]
        ]
        skill.validation = [
            v[:MAX_VALIDATION_CHARS] for v in skill.validation[:5]
        ]
        skill.anti_patterns = [
            a[:MAX_ANTI_PATTERN_CHARS] for a in skill.anti_patterns[:5]
        ]

        # Dédup par similarité d'embedding
        if skill.trigger_embedding:
            similar = self._find_similar(skill.trigger_embedding, threshold=0.85)
            if similar:
                # _find_similar retourne une liste — on prend le premier
                existing = similar[0]
                existing.success_count += 1
                existing.last_used_at = time.time()
                existing.confidence = min(1.0, existing.confidence + 0.05)
                # Merge approach / validation (union)
                for s in skill.approach:
                    if s not in existing.approach:
                        existing.approach.append(s)
                for v in skill.validation:
                    if v not in existing.validation:
                        existing.validation.append(v)
                self._save()
                log.info("Skill %s updated (success_count=%d)",
                         existing.skill_id, existing.success_count)
                return existing

        # Nouveau skill
        if len(self._skills) >= MAX_TOTAL_SKILLS:
            # Archive le moins utile
            self._archive_least_useful()
        self._skills[skill.skill_id] = skill
        self._save()
        log.info("Skill %s added (trigger=%r)", skill.skill_id, skill.trigger[:60])
        return skill

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def active(self) -> list[Skill]:
        """Skills non archivés, triés par utility décroissante."""
        return sorted(
            [s for s in self._skills.values() if not s.archived],
            key=lambda s: s.utility_score(),
            reverse=True,
        )

    def find_by_trigger_embedding(
        self, embedding: list[float], threshold: float = 0.75, top_k: int = 3
    ) -> list[Skill]:
        """Retourne les top-k skills dont le trigger match sémantiquement."""
        return self._find_similar(embedding, threshold=threshold, top_k=top_k)

    def record_failure(
        self,
        skill_id: str,
        anti_pattern: str | None = None,
    ) -> None:
        """Enregistre un échec sur un skill (et optionnellement un anti-pattern)."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return
        skill.failure_count += 1
        if anti_pattern and anti_pattern not in skill.anti_patterns:
            # Garde-fou : l'anti-pattern est réinjecté dans le prompt via
            # to_markdown() — on refuse tout contenu interdit et on
            # neutralise la structure (sauts de ligne, tokens de chat)
            # avant stockage.
            if _content_is_forbidden(anti_pattern):
                log.warning(
                    "Anti-pattern rejected for %s: forbidden content", skill_id
                )
            else:
                skill.anti_patterns.append(
                    sanitize_for_prompt(anti_pattern, MAX_ANTI_PATTERN_CHARS)
                )
        # Ajuste confidence à la baisse
        total = skill.success_count + skill.failure_count
        skill.confidence = max(0.1, skill.confidence - 0.1)
        self._save()

    def record_success(self, skill_id: str) -> None:
        """Enregistre un succès sur un skill (incrémente confidence)."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return
        skill.success_count += 1
        skill.last_used_at = time.time()
        skill.confidence = min(1.0, skill.confidence + 0.05)
        self._save()

    def archive(self, skill_id: str) -> None:
        skill = self._skills.get(skill_id)
        if skill is None:
            return
        skill.archived = True
        self._save()

    def stats(self) -> dict:
        active = [s for s in self._skills.values() if not s.archived]
        archived = [s for s in self._skills.values() if s.archived]
        composed = sum(1 for s in active if (s.metadata or {}).get("composed"))
        return {
            "total": len(self._skills),
            "active": len(active),
            "archived": len(archived),
            "reliable": sum(1 for s in active if s.is_reliable()),
            "composed": composed,
            "max_active": MAX_ACTIVE_SKILLS,
            "max_total": MAX_TOTAL_SKILLS,
        }

    # ── Composition (delegates to SkillComposer) ─────────────────────

    def compose(
        self,
        skill_ids: list[str],
        strategy: str = "sequential",
        composed_trigger: str | None = None,
        force: bool = False,
        llm_callback: Any = None,
    ) -> dict:
        """Compose plusieurs skills en une nouvelle.

        Voir rune.memory.skill_composer.SkillComposer pour les détails.
        Retourne un dict (CompositionResult.as_dict()).
        """
        from .skill_composer import SkillComposer
        composer = SkillComposer(self, llm_callback=llm_callback)
        result = composer.compose(
            skill_ids=skill_ids,
            strategy=strategy,
            composed_trigger=composed_trigger,
            force=force,
        )
        return result.as_dict()

    def find_composable_candidates(
        self,
        task_embedding: list[float] | None = None,
        max_pairs: int = 5,
    ) -> list[dict]:
        """Trouve des paires de skills composables.

        Retourne une liste de dicts :
        [{"skill_a": {...}, "skill_b": {...}, "potential": 0.7}, ...]
        """
        from .skill_composer import SkillComposer
        composer = SkillComposer(self)
        pairs = composer.find_composable_candidates(
            task_embedding=task_embedding, max_pairs=max_pairs
        )
        return [
            {
                "skill_a": {"id": a.skill_id, "trigger": a.trigger},
                "skill_b": {"id": b.skill_id, "trigger": b.trigger},
                "potential": round(pot, 3),
            }
            for a, b, pot in pairs
        ]

    # ── Internes ──────────────────────────────────────────────────────

    def _is_safe(self, skill: Skill) -> bool:
        """Vérifie que le skill ne contient pas d'instruction interdite.

        Couvre TOUS les champs texte réinjectés dans le prompt, y compris
        ``anti_patterns`` (qui font partie du to_markdown() injecté).
        """
        full_text = " ".join([
            skill.trigger,
            " ".join(skill.approach),
            " ".join(skill.validation),
            " ".join(skill.anti_patterns),
        ])
        return not _content_is_forbidden(full_text)

    def _find_similar(
        self,
        embedding: list[float],
        threshold: float = 0.85,
        top_k: int = 1,
    ) -> list[Skill]:
        """Cherche les skills similaires par similarité cosinus."""
        if not embedding:
            return []
        results: list[tuple[float, Skill]] = []
        for skill in self._skills.values():
            if skill.archived or not skill.trigger_embedding:
                continue
            sim = _cosine_similarity(embedding, skill.trigger_embedding)
            if sim >= threshold:
                results.append((sim, skill))
        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results[:top_k]]

    def _archive_least_useful(self) -> None:
        active = [s for s in self._skills.values() if not s.archived]
        if not active:
            return
        # Archive le moins utile
        least = min(active, key=lambda s: s.utility_score())
        least.archived = True
        log.info("Archived least useful skill %s (utility=%.3f)",
                 least.skill_id, least.utility_score())

    def _save(self) -> None:
        """Persiste l'index JSON + exporte les fichiers MD."""
        data = {
            "version": 1,
            "skills": [_to_native(s.as_dict()) for s in self._skills.values()],
        }
        # Atomic write. _to_native() convertit les scalaires numpy/torch
        # (np.float32, etc.) en types Python natifs : sans ça, json.dump
        # lève « Object of type float32 is not JSON serializable » et
        # AUCUN skill n'est persisté (confidence/doubt viennent parfois
        # de calculs numpy dans le pipeline).
        tmp = self._index_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._index_path)

        # Export MD pour les skills actifs
        for skill in self._skills.values():
            if skill.archived:
                continue
            md_path = self.storage_dir / "exports" / f"{skill.skill_id}.md"
            try:
                md_path.write_text(skill.to_markdown(), encoding="utf-8")
            except Exception as exc:
                # On logge la cause réelle (pas juste « Failed »), et on
                # continue : l'export MD est secondaire, le JSON reste la
                # source de vérité.
                log.warning(
                    "Failed to export MD for %s: %s", skill.skill_id, exc
                )

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with self._index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("skills", []):
                try:
                    skill = Skill(**{
                        k: v for k, v in entry.items()
                        if k in Skill.__dataclass_fields__  # type: ignore[attr-defined]
                    })
                    # Quarantaine au chargement : un skills.json édité à la
                    # main (ou un skill hérité d'une version sans filtre)
                    # pourrait contenir une injection — add() ne re-vérifie
                    # pas les skills déjà en base. On archive plutôt que de
                    # supprimer (auditable), et l'archivage exclut le skill
                    # de l'injection prompt (active()/find filtrent archived).
                    if not skill.archived and not self._is_safe(skill):
                        skill.archived = True
                        log.warning(
                            "Skill %s quarantined at load: forbidden content",
                            skill.skill_id,
                        )
                    self._skills[skill.skill_id] = skill
                except Exception as exc:
                    log.warning("Failed to load skill entry: %s", exc)
        except Exception:
            log.exception("Failed to load skills index")


# ── SkillExtractor — propose un skill à partir d'un épisode ──────────


class SkillExtractor:
    """Extrait un Skill d'un épisode réussi.

    En pratique, l'extraction se fait via un appel LLM (prompt structuré).
    Ici on fournit une version heuristique qui marche sans LLM pour les
    tests et le bootstrapping. La version LLM est branchée via
    ``set_llm_callback``.

    Contrats
    --------
    1. Ne lève jamais — retourne None si extraction impossible.
    2. Filtre par garde-fous sécurité (refus explicite patterns interdits).
    3. Déduplication gérée par AutoSkillStore.add() (pas ici).
    """

    def __init__(self) -> None:
        self._llm_callback: callable | None = None  # type: ignore[assignment]

    def set_llm_callback(self, callback: callable) -> None:  # type: ignore[assignment]
        """Branche une fonction LLM pour l'extraction structurée.

        Signature attendue :
            callback(prompt: str) -> dict | None
        où dict contient les clés "trigger", "approach", "validation".
        """
        self._llm_callback = callback

    def extract(
        self,
        user_message: str,
        assistant_response: str,
        verifier_ok: bool,
        doubt_index: float,
        confidence_label: str = "certaine",
        trigger_embedding: list[float] | None = None,
        source_episode_id: str | None = None,
    ) -> Skill | None:
        """Tente d'extraire un Skill d'un épisode.

        Garde-fous d'extraction (différents des garde-fous sécurité du
        store) :
        - verifier_ok doit être True (sinon pas d'extraction)
        - doubt_index < 0.4 (sinon réponse incertaine, pas un pattern fiable)
        - confidence_label ∈ {très_certaine, certaine}
        - user_message ≥ 10 caractères (trop court = pas un vrai pattern)
        - assistant_response ≥ 50 caractères (sinon rien à apprendre)
        """
        if not verifier_ok:
            return None
        if doubt_index >= 0.4:
            return None
        if confidence_label not in {"très_certaine", "certaine"}:
            return None
        if len(user_message) < 10 or len(assistant_response) < 50:
            return None

        # ── Filtre anti-conversation (garde-fou sémantique rapide) ────
        # Les seuils ci-dessus ne vérifient que la FORME (longueur,
        # confiance). Sans ce filtre, une simple présentation sociale
        # (« Je m'appelle Michael ») ou un échange trivial devient un
        # « skill », ce qui n'a aucun sens : un skill doit capturer une
        # MÉTHODE réutilisable, pas un fait ponctuel ou une politesse.
        # Ce filtre rejette les cas évidents SANS coûter de tokens LLM.
        if self._looks_conversational(user_message, assistant_response):
            log.debug(
                "Skill extraction skipped: échange conversationnel/trivial (%r)",
                user_message[:50],
            )
            return None

        # Extraction — jugement LLM en priorité, repli heuristique.
        # Le filtre _looks_conversational a déjà écarté les échanges
        # sociaux/triviaux évidents. Trois cas avec un callback LLM :
        #   - le LLM renvoie un skill structuré → on l'utilise ;
        #   - le LLM dit « skip » (épisode trivial) → on RESPECTE et on
        #     n'extrait rien (ne pas retomber sur l'heuristique, sinon on
        #     annulerait le jugement du LLM) ;
        #   - le LLM est indisponible (pas de modèle chargé) → repli
        #     heuristique pour que les skills se créent quand même.
        if self._llm_callback is not None:
            verdict = self._extract_via_llm(
                user_message, assistant_response
            )
            if verdict == _LLM_UNAVAILABLE:
                extracted = self._extract_heuristic(
                    user_message, assistant_response
                )
            elif not verdict:
                # skip explicite → pas de skill
                return None
            else:
                extracted = verdict
        else:
            extracted = self._extract_heuristic(
                user_message, assistant_response
            )

        if not extracted:
            return None

        # ── Validation stricte du contenu extrait ──────────────────────
        # Le dict peut venir du LLM (JSON parsé, types non garantis :
        # approach pourrait être une string, contenir des dicts, ou
        # embarquer une injection via des sauts de ligne). On coerce en
        # types stricts et on neutralise la structure. Le contenu passera
        # ENSUITE par AutoSkillStore._is_safe() (patterns interdits) —
        # ceci est la couche de validation de FORME, _is_safe celle de
        # FOND.
        def _as_str_list(value, max_items: int, max_chars: int) -> list[str]:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                return []
            out: list[str] = []
            for item in value[:max_items]:
                if isinstance(item, (str, int, float)):
                    s = sanitize_for_prompt(str(item), max_chars)
                    if s:
                        out.append(s)
            return out

        clean_trigger = sanitize_for_prompt(
            str(extracted.get("trigger") or user_message), MAX_TRIGGER_CHARS
        )
        clean_approach = _as_str_list(
            extracted.get("approach"), MAX_APPROACH_STEPS, MAX_APPROACH_STEP_CHARS
        )
        clean_validation = _as_str_list(
            extracted.get("validation"), 5, MAX_VALIDATION_CHARS
        )
        clean_anti = _as_str_list(
            extracted.get("anti_patterns"), 5, MAX_ANTI_PATTERN_CHARS
        )
        if not clean_approach:
            # Un skill sans approche exploitable n'a aucune valeur.
            return None

        # Construction du Skill
        skill_id = self._make_id(user_message)
        return Skill(
            skill_id=skill_id,
            trigger=clean_trigger,
            trigger_embedding=trigger_embedding or [],
            approach=clean_approach,
            validation=clean_validation,
            anti_patterns=clean_anti,
            success_count=1,
            failure_count=0,
            confidence=max(0.5, 1.0 - doubt_index),
            source_episodes=[source_episode_id] if source_episode_id else [],
        )

    # ── Internes ──────────────────────────────────────────────────────

    # Patterns d'ouverture de messages purement conversationnels /
    # sociaux : présentation, salutation, remerciement, méta-discussion
    # sur la conversation elle-même. Un message qui commence par ça
    # n'est presque jamais une demande de résolution de problème.
    _CONVERSATIONAL_OPENERS = (
        "je m'appelle", "je m appelle", "moi c'est", "moi c est",
        "je suis ", "mon nom est", "on m'appelle", "appelle-moi",
        "bonjour", "salut", "coucou", "bonsoir", "hello", "hey", "hi ",
        "merci", "thanks", "thank you", "ok merci", "d'accord",
        "comment vas-tu", "comment ça va", "ça va", "comment tu vas",
        "quel est ton nom", "comment tu t'appelles", "qui es-tu",
        "au revoir", "bonne nuit", "à bientôt", "à plus", "bye",
        "enchanté", "ravi de", "content de te",
    )

    # Mots qui signalent une vraie tâche/problème (présence = plutôt
    # gardé). Sert de contre-indice : si le message est court mais
    # contient un verbe d'action technique, on ne le classe pas
    # conversationnel.
    _TASK_SIGNALS = (
        "comment", "pourquoi", "explique", "calcule", "code", "écris",
        "corrige", "débogue", "debug", "implémente", "optimise", "analyse",
        "compare", "résous", "resous", "trouve", "génère", "genere",
        "convertis", "traduis", "liste", "montre", "crée", "cree",
        "configure", "installe", "déploie", "teste", "fonction", "erreur",
        "algorithme", "requête", "requete", "script", "commande",
    )

    def _looks_conversational(
        self, user_message: str, assistant_response: str
    ) -> bool:
        """Vrai si l'échange est social/trivial plutôt qu'une compétence.

        Heuristique volontairement conservatrice : en cas de doute, elle
        laisse passer (retourne False) — c'est le jugement LLM en aval
        qui tranchera finement. On ne bloque que les cas ÉVIDENTS pour
        économiser des tokens et éviter les faux skills grossiers du type
        « Je m'appelle Michael ».
        """
        msg = user_message.strip().lower()
        if not msg:
            return True

        starts_social = any(
            msg.startswith(op) for op in self._CONVERSATIONAL_OPENERS
        )

        # Les salutations sociales peuvent contenir « comment » (« comment
        # ça va », « comment vas-tu »). On ne considère donc un signal de
        # tâche comme valide que s'il n'est pas neutralisé par une
        # ouverture sociale claire de type question de politesse.
        SOCIAL_QUESTION_PHRASES = (
            "comment vas-tu", "comment ça va", "comment ca va",
            "comment tu vas", "comment allez-vous", "comment tu t'appelles",
            "quel est ton nom", "qui es-tu", "comment tu t appelles",
            "comment tu te sens", "quoi de neuf",
        )
        # Détecte ces tournures n'importe où dans le message (pas
        # seulement en préfixe) : « Bonjour, comment ça va ? » commence
        # par « bonjour » mais la partie sociale « comment ça va » est
        # au milieu.
        contains_social_question = any(
            p in msg for p in SOCIAL_QUESTION_PHRASES
        )
        has_task_signal = (
            any(sig in msg for sig in self._TASK_SIGNALS)
            and not contains_social_question
        )

        # Question purement sociale → conversationnel.
        if contains_social_question and len(msg) < 45:
            return True

        # Cas 1 : court + ouverture sociale + aucun signal de tâche.
        if len(msg) < 40 and starts_social and not has_task_signal:
            return True

        # Cas 2 : ouverture sociale et pas de signal de tâche du tout,
        # même si le message est un peu plus long.
        if starts_social and not has_task_signal:
            return True

        return False

    def _extract_heuristic(
        self, user_message: str, assistant_response: str
    ) -> dict | None:
        """Extraction heuristique sans LLM.

        Heuristiques simples :
        - trigger = user_message tronqué
        - approach = premières phrases de la réponse
        - validation = "Réponse fournie et structurée"

        Limites évidentes : ne capture pas les subtilités. Pour la prod,
        brancher un LLM via set_llm_callback.
        """
        # Premières phrases comme approach
        import re
        sentences = re.split(r"(?<=[.!?])\s+", assistant_response)
        approach = [s for s in sentences[:3] if len(s) > 20]
        if not approach:
            return None
        return {
            "trigger": user_message.strip()[:MAX_TRIGGER_CHARS],
            "approach": approach[:MAX_APPROACH_STEPS],
            "validation": [
                "Réponse fournie et structurée",
                "Aucune formule d'esquive détectée",
            ],
        }

    def _extract_via_llm(
        self, user_message: str, assistant_response: str
    ) -> dict | None:
        """Extraction via LLM — prompt structuré.

        Le prompt demande au LLM de produire un JSON avec :
        trigger, approach[], validation[], anti_patterns[].
        Le LLM est le backend modèle principal (pas un modèle séparé).
        """
        prompt = f"""Tu es un extracteur de compétences. À partir d'un
épisode réussi, extrais un pattern réutilisable au format JSON.

Épisode :
- Demande utilisateur : {user_message!r}
- Réponse de l'assistant : {assistant_response!r}

Produis UNIQUEMENT un JSON valide (pas de markdown) avec ces clés :
- "trigger" : description courte (≤200 chars) du contexte d'activation
- "approach" : liste de 2-5 étapes concrètes (≤200 chars chacune)
- "validation" : liste de 1-3 critères observables de succès
- "anti_patterns" : liste de 0-3 choses à éviter (optionnel)

Si l'épisode ne mérite pas un skill (trop trivial, trop spécifique),
retourne {{"skip": true}}.
"""
        try:
            result = self._llm_callback(prompt)  # type: ignore[misc]
            # Le callback signale explicitement son indisponibilité
            # (pas de modèle) → on propage la sentinelle pour que extract()
            # retombe sur l'heuristique.
            if result is _LLM_UNAVAILABLE or (
                isinstance(result, dict) and result.get("_llm_unavailable")
            ):
                return _LLM_UNAVAILABLE
            if not result or result.get("skip"):
                return None
            return result
        except Exception:
            log.exception("LLM extraction failed")
            return None

    @staticmethod
    def _make_id(user_message: str) -> str:
        h = hashlib.sha256(user_message.encode("utf-8")).hexdigest()[:12]
        return f"skill_{h}"


# ── Helpers math ──────────────────────────────────────────────────────


def _to_native(obj):
    """Convertit récursivement numpy/torch scalars en types Python natifs.

    json.dump et yaml.safe_dump rejettent np.float32, np.int64,
    torch.Tensor, etc. (« not JSON serializable » / « cannot represent
    an object »). Comme confidence, doubt_index et certaines valeurs
    d'embedding peuvent provenir de calculs numpy/torch dans le pipeline,
    on normalise à la frontière de persistance pour que la sauvegarde
    ne casse jamais.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    # numpy / torch scalar → .item() donne un type Python natif
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarité cosinus entre deux vecteurs. 0 si l'un est vide."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))
