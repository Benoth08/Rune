"""Trinity pool — gère le chargement et le routing des 3 modèles.

Le pool est chargé au boot (si Trinity activé). Il expose une API
simple pour choisir quel modèle utiliser selon le contexte :

    pool = TrinityPool(config)
    await pool.load_all()

    # Routing automatique
    model = pool.pick_model(
        complexity_steps=4,
        surprise=0.7,
        doubt_index=0.2,
        phase="reasoning",  # "reasoning" | "execution" | "verification"
    )

    # Ou choix explicite
    model = pool.get_model(TrinityRole.THINKER)

Si VRAM insuffisante pour les 3 modèles, le pool tombe en mode
dégradé : il charge seulement le Worker (le plus essentiel) et
désactive Trinity avec un warning.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

from .config import TrinityConfig, TrinityModelSpec

log = logging.getLogger("rune.trinity.pool")


class TrinityRole(str, enum.Enum):
    """Rôles Trinity."""
    THINKER = "thinker"
    WORKER = "worker"
    CRITIC = "critic"


@dataclass
class TrinityHandoff:
    """Transfert de contexte entre deux modèles Trinity.

    Quand le Thinker passe la main au Worker, on transfère :
    - Le raisonnement produit (texte)
    - Le plan d'action (steps)
    - Le contexte RAG récupéré
    - Les entités KG pertinentes

    Le Worker reçoit ça comme "briefing" et l'utilise pour exécuter.
    """
    from_role: TrinityRole
    to_role: TrinityRole
    reasoning: str = ""
    plan: list[str] = field(default_factory=list)
    context: str = ""
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrinityPool:
    """Pool de 3 modèles spécialisés.

    Attributes
    ----------
    config : TrinityConfig
        Config chargée depuis trinity.yaml.
    models : dict[TrinityRole, Any]
        Modèles chargés (clé = rôle, valeur = HFModelWrapper ou None).
    vram_available_gb : float
        VRAM disponible au moment du load.
    degraded_mode : bool
        True si on n'a pas pu charger tous les modèles (VRAM insuffisante).
    """

    def __init__(self, config: TrinityConfig) -> None:
        self.config = config
        self.models: dict[TrinityRole, Any] = {
            TrinityRole.THINKER: None,
            TrinityRole.WORKER: None,
            TrinityRole.CRITIC: None,
        }
        self.vram_available_gb: float = 0.0
        self.degraded_mode: bool = False
        self._loaded: bool = False

    # ── API publique ──────────────────────────────────────────────────

    def load_all(self, model_loader: Any | None = None) -> dict[str, Any]:
        """Charge les 3 modèles en VRAM.

        Parameters
        ----------
        model_loader : callable | None
            Fonction qui prend un model_id et retourne un HFModelWrapper.
            Si None, utilise rune.model.HFModelWrapper.

        Returns
        -------
        dict
            Rapport de chargement par rôle.
        """
        if not self.config.enabled:
            log.info("Trinity disabled — skipping load_all")
            return {"status": "disabled"}

        if self._loaded:
            log.warning("TrinityPool already loaded — skipping")
            return {"status": "already_loaded"}

        if model_loader is None:
            from rune.model import HFModelWrapper
            model_loader = lambda model_id: HFModelWrapper(model_id=model_id)  # noqa: E731

        # Estime la VRAM disponible
        try:
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                self.vram_available_gb = free / (1024**3)
            else:
                self.vram_available_gb = 0.0
        except Exception:
            self.vram_available_gb = 0.0

        log.info(
            "Trinity loading 3 models — VRAM available: %.1f GB",
            self.vram_available_gb,
        )

        # Charge chaque modèle
        report: dict[str, Any] = {"roles": {}, "degraded": False}
        for role, spec in self._iter_specs():
            try:
                model = model_loader(spec.model_id)
                # Si quant_4bit, configure avant le load
                if spec.quant_4bit and hasattr(model, "config"):
                    # HFModelWrapper gère le 4-bit dans son load()
                    pass
                ok = model.load(spec.model_id) if hasattr(model, "load") else True
                if ok:
                    self.models[role] = model
                    report["roles"][role.value] = {
                        "status": "ok",
                        "model_id": spec.model_id,
                    }
                    log.info("Trinity %s loaded: %s", role.value, spec.model_id)
                else:
                    report["roles"][role.value] = {
                        "status": "failed",
                        "model_id": spec.model_id,
                        "error": "load returned False",
                    }
                    log.warning("Trinity %s failed to load", role.value)
            except Exception as exc:
                report["roles"][role.value] = {
                    "status": "error",
                    "model_id": spec.model_id,
                    "error": str(exc),
                }
                log.exception("Trinity %s load error", role.value)

        # Vérifie qu'au moins le Worker est chargé (le plus essentiel)
        if self.models[TrinityRole.WORKER] is None:
            self.degraded_mode = True
            report["degraded"] = True
            report["status"] = "degraded"
            log.error(
                "Trinity: Worker not loaded — falling back to single model mode. "
                "Trinity is effectively disabled."
            )
        elif (
            self.models[TrinityRole.THINKER] is None
            or self.models[TrinityRole.CRITIC] is None
        ):
            self.degraded_mode = True
            report["degraded"] = True
            report["status"] = "partial"
            log.warning(
                "Trinity: some roles failed to load — running in partial mode. "
                "Worker is OK, but Thinker or Critic is missing."
            )
        else:
            report["status"] = "ok"
            log.info("Trinity fully loaded — 3 models active")

        self._loaded = True
        return report

    def unload_all(self) -> None:
        """Décharge tous les modèles (libère la VRAM)."""
        for role in TrinityRole:
            model = self.models.get(role)
            if model is not None and hasattr(model, "unload"):
                try:
                    model.unload()
                except Exception:
                    log.exception("Failed to unload %s", role.value)
            self.models[role] = None
        self._loaded = False
        log.info("Trinity unloaded — VRAM freed")

    def get_model(self, role: TrinityRole) -> Any | None:
        """Retourne le modèle pour un rôle donné (ou None si non chargé)."""
        return self.models.get(role)

    def pick_model(
        self,
        complexity_steps: int = 0,
        surprise: float = 0.0,
        doubt_index: float = 0.0,
        phase: str = "execution",
    ) -> Any | None:
        """Choisit le modèle à utiliser selon le contexte.

        Règles de routing
        -----------------
        - phase == "reasoning" + (complexity >= threshold ou surprise >= threshold)
            → Thinker
        - phase == "execution"
            → Worker (toujours)
        - phase == "verification"
            → Critic (si critic_always ou doubt_index >= threshold)

        Si le modèle choisi n'est pas chargé (mode dégradé), fallback
        sur le Worker.

        Parameters
        ----------
        complexity_steps : int
            Score de complexité (depuis assess_complexity de Lythea).
        surprise : float
            Score de surprise composite (0-1).
        doubt_index : float
            Index de doute (0-1, depuis métacognition).
        phase : str
            Phase courante : "reasoning" | "execution" | "verification".

        Returns
        -------
        Model | None
            Le modèle choisi, ou None si rien n'est chargé.
        """
        if not self.config.enabled or self.degraded_mode:
            # Mode dégradé — toujours le Worker
            return self.models.get(TrinityRole.WORKER)

        r = self.config.routing
        chosen_role: TrinityRole

        if phase == "reasoning":
            if (
                complexity_steps >= r.thinker_threshold_steps
                or surprise >= r.thinker_threshold_surprise
            ):
                chosen_role = TrinityRole.THINKER
            else:
                chosen_role = TrinityRole.WORKER
        elif phase == "verification":
            if r.critic_always or doubt_index >= r.critic_threshold_doubt:
                chosen_role = TrinityRole.CRITIC
            else:
                chosen_role = TrinityRole.WORKER
        else:  # execution
            chosen_role = TrinityRole.WORKER

        model = self.models.get(chosen_role)
        if model is None:
            # Fallback sur Worker si le rôle choisi n'est pas chargé
            log.debug(
                "Trinity: %s not loaded — falling back to Worker",
                chosen_role.value,
            )
            model = self.models.get(TrinityRole.WORKER)
        return model

    def status(self) -> dict[str, Any]:
        """Snapshot pour /status et /api/trinity/status."""
        return {
            "enabled": self.config.enabled,
            "loaded": self._loaded,
            "degraded_mode": self.degraded_mode,
            "vram_available_gb": round(self.vram_available_gb, 2),
            "roles": {
                role.value: {
                    "model_id": spec.model_id,
                    "loaded": self.models[role] is not None,
                }
                for role, spec in self._iter_specs()
            },
            "routing": {
                "thinker_threshold_steps": self.config.routing.thinker_threshold_steps,
                "thinker_threshold_surprise": self.config.routing.thinker_threshold_surprise,
                "critic_always": self.config.routing.critic_always,
                "critic_threshold_doubt": self.config.routing.critic_threshold_doubt,
            },
        }

    # ── Internes ──────────────────────────────────────────────────────

    def _iter_specs(self) -> list[tuple[TrinityRole, TrinityModelSpec]]:
        """Itère sur (rôle, spec) dans l'ordre de chargement."""
        return [
            (TrinityRole.WORKER, self.config.worker),    # Worker en 1er (essentiel)
            (TrinityRole.THINKER, self.config.thinker),
            (TrinityRole.CRITIC, self.config.critic),
        ]
