"""Boot orchestrator — preload heavy components before opening the UI.

The goal is that when the user opens the URL (Cloudflare or local), all
auxiliary models are already in memory so the first message has zero
warm-up cost. Only the LLM itself is left for the user to choose.

Stages (in order):
1. ChromaDB + retriever
2. GLiNER (entity extractor)
3. SentenceTransformer (encoder)
4. Cross-encoder (BGE-reranker-v2-m3) — used by HybridRetriever
5. Captioner (Qwen2-VL if VRAM ≥ 5 GB else BLIP) — Option B

A live status is exposed via :class:`BootState` and consumed by
``GET /api/boot/status``. While ``ready`` is False, all other ``/api/*``
routes return 503.

Failure policy: a stage failure is logged but does not abort the boot.
We always converge to ``ready=True`` so the UI can open and the user
can react manually (e.g. clear cache, change captioner).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("rune.boot")


# ── Stage names (single source of truth, used by UI too) ──────────────

STAGES = (
    "chromadb",
    "gliner",
    "sentence_transformer",
    "cross_encoder",
    "captioner",
    "mcp",
)
STAGE_LABELS_FR = {
    "chromadb": "ChromaDB + index BM25",
    "gliner": "GLiNER (extraction d'entités)",
    "sentence_transformer": "SentenceTransformer",
    "cross_encoder": "Cross-encoder (reranker)",
    "captioner": "Captioner d'images",
    "mcp": "Outils MCP (filesystem, GitHub, YouTube)",
}


# ── State ──────────────────────────────────────────────────────────────

@dataclass
class BootState:
    """Mutable boot state, polled by the UI splash screen."""

    ready: bool = False
    current_step: str = "init"
    step_index: int = 0
    step_total: int = len(STAGES)
    progress_pct: float = 0.0
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    details: str = ""
    components: dict[str, str | None] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def begin_stage(self, stage: str, details: str = "") -> None:
        """Mark the start of a stage. Called from boot thread."""
        with self._lock:
            try:
                self.step_index = STAGES.index(stage) + 1
            except ValueError:
                pass
            self.current_step = f"loading_{stage}"
            self.details = details
            self.elapsed_s = round(time.time() - self.started_at, 1)
            self.progress_pct = round(
                ((self.step_index - 1) / self.step_total) * 100, 1
            )
            log.info("⏳ Boot stage %d/%d: %s %s",
                     self.step_index, self.step_total,
                     STAGE_LABELS_FR.get(stage, stage),
                     f"({details})" if details else "")

    def end_stage(
        self, stage: str, status: str | None = "ok", message: str = "",
    ) -> None:
        """Mark a stage as finished (ok, failed, or skipped)."""
        with self._lock:
            self.components[stage] = status
            self.elapsed_s = round(time.time() - self.started_at, 1)
            if status == "ok":
                log.info("✅ Boot stage done: %s", stage)
            elif status == "failed":
                log.warning("❌ Boot stage failed: %s — %s", stage, message)
                self.messages.append(f"{stage}: {message}")
            elif status == "skipped":
                log.info("⏭️  Boot stage skipped: %s — %s", stage, message)

    def finalize(self) -> None:
        """Mark boot as complete."""
        with self._lock:
            self.ready = True
            self.current_step = "done"
            self.step_index = self.step_total
            self.progress_pct = 100.0
            self.elapsed_s = round(time.time() - self.started_at, 1)
            log.info("🌟 Boot complete in %.1fs — Lythéa ready", self.elapsed_s)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the /api/boot/status route."""
        with self._lock:
            return {
                "ready": self.ready,
                "current_step": self.current_step,
                "step_index": self.step_index,
                "step_total": self.step_total,
                "progress_pct": self.progress_pct,
                "elapsed_s": round(time.time() - self.started_at, 1),
                "details": self.details,
                "components": dict(self.components),
                "messages": list(self.messages),
                "stage_labels": dict(STAGE_LABELS_FR),
            }


# ── Boot runner ────────────────────────────────────────────────────────

class BootRunner:
    """Runs the preload sequence in a background thread.

    Parameters
    ----------
    lythea_app : Any
        The :class:`LytheaApp` instance whose components must be warmed up.
    state : BootState
        Shared mutable state for status reporting.
    """

    def __init__(self, lythea_app: Any, state: BootState) -> None:
        self.app = lythea_app
        self.state = state
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the boot sequence in a daemon thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="lythea-boot",
        )
        self._thread.start()

    def _run(self) -> None:
        """Sequential preload."""
        try:
            self._stage_chromadb()
            self._stage_gliner()
            self._stage_sentence_transformer()
            self._stage_cross_encoder()
            self._stage_captioner()
            self._stage_mcp()
            # Rune — autoload du modèle au boot si configuré.
            # Désactivé par défaut (RUNE_AUTOLOAD_MODEL=false). Quand activé,
            # le modèle spécifié par RUNE_DEFAULT_MODEL (ou DEFAULT_MODEL
            # Lythea) est chargé en VRAM pendant le boot, avant que le serveur
            # ne déclare ready=true. Sinon, l'utilisateur doit charger le
            # modèle manuellement via POST /api/models/load après le boot.
            self._stage_model_autoload()
            # Rune — Trinity (pool multi-modèles) si activé.
            # Désactivé par défaut. Quand activé via trinity.yaml, charge
            # 3 modèles (Thinker + Worker + Critic) en VRAM. Si Trinity
            # est activé, l'autoload single-model ci-dessus est ignoré
            # (Trinity s'en occupe).
            self._stage_trinity()
        except Exception as exc:
            log.exception("Unexpected boot failure: %s", exc)
        finally:
            self.state.finalize()

    # ── Stages ─────────────────────────────────────────────────────────

    def _stage_chromadb(self) -> None:
        """Verify ChromaDB collection is queryable and warm BM25 index."""
        self.state.begin_stage("chromadb")
        try:
            count = self.app.chroma_collection.count()
            # Warm BM25 index (lazy in HybridRetriever)
            self.app.retriever._maybe_rebuild_bm25()
            self.state.end_stage(
                "chromadb", "ok", f"{count} documents indexed",
            )
        except Exception as exc:
            self.state.end_stage("chromadb", "failed", str(exc))

    def _stage_gliner(self) -> None:
        """Force-load GLiNER on first call by extracting from a tiny prompt."""
        self.state.begin_stage("gliner")
        try:
            self.app.entity_extractor.extract("Préchargement.")
            self.state.end_stage("gliner", "ok")
        except Exception as exc:
            self.state.end_stage("gliner", "failed", str(exc))

    def _stage_sentence_transformer(self) -> None:
        """Force-load the sentence transformer encoder."""
        self.state.begin_stage("sentence_transformer")
        try:
            emb = self.app.entity_extractor.encode("Préchargement.")
            if emb is None:
                self.state.end_stage(
                    "sentence_transformer", "failed", "encode returned None",
                )
                return
            # V5.7.1 — Warmup Vision active semantic detector (multilingue).
            # Précompute les embeddings des prototypes d'intention pour
            # éviter la latence à la 1ère utilisation. Best-effort —
            # si ça échoue, le détecteur tombera sur le fallback lexical.
            try:
                from rune.cognition.vision_semantic import get_detector
                detector = get_detector()
                ok = detector.warm_up()
                if ok:
                    log.info("Vision semantic detector warmed up successfully")
                else:
                    log.warning(
                        "Vision semantic detector warmup failed — "
                        "will use lexical fallback"
                    )
            except Exception as warmup_exc:
                log.warning(
                    "Vision semantic warmup raised: %s — lexical fallback active",
                    warmup_exc,
                )
            self.state.end_stage("sentence_transformer", "ok")
        except Exception as exc:
            self.state.end_stage("sentence_transformer", "failed", str(exc))

    def _stage_cross_encoder(self) -> None:
        """Force-load the cross-encoder reranker.

        The ``HybridRetriever`` lazy-loads the cross-encoder on first use
        (see ``_get_cross_encoder``). To preload, we just call it once.
        """
        self.state.begin_stage("cross_encoder")
        try:
            ce = self.app.retriever._get_cross_encoder()
            if ce is None:
                self.state.end_stage(
                    "cross_encoder", "skipped",
                    "cross-encoder unavailable — cosine fallback in use",
                )
                return
            self.state.end_stage("cross_encoder", "ok")
        except Exception as exc:
            self.state.end_stage("cross_encoder", "failed", str(exc))

    def _stage_captioner(self) -> None:
        """Pick the best captioner per available VRAM (Option B)."""
        self.state.begin_stage("captioner", details="détection VRAM…")
        try:
            from rune.model import vram_free_gb

            free = vram_free_gb()
            captioner = self.app.hippocampe.captioner

            # Option B: Qwen2-VL if VRAM ≥ 5 GB free, else BLIP
            if free >= 5.0:
                self.state.begin_stage(
                    "captioner",
                    details=f"VRAM {free:.1f} GB → tentative Qwen2-VL-2B",
                )
                result = captioner.select("qwen2vl")
                if result.get("status") == "loaded":
                    self.state.end_stage(
                        "captioner", "ok", f"qwen2vl ({free:.1f} GB free)",
                    )
                    return
                # Qwen2-VL failed (download error, OOM during load…) → fallback
                log.warning("Qwen2-VL preload failed, falling back to BLIP")

            self.state.begin_stage(
                "captioner", details=f"VRAM {free:.1f} GB → BLIP (CPU)",
            )
            result = captioner.select("blip")
            if result.get("status") == "loaded":
                self.state.end_stage("captioner", "ok", "BLIP (CPU)")
            else:
                self.state.end_stage(
                    "captioner", "failed", "neither qwen2vl nor blip loaded",
                )
        except Exception as exc:
            self.state.end_stage("captioner", "failed", str(exc))

    def _stage_mcp(self) -> None:
        """V6.0.0 — Start MCP servers (filesystem, GitHub, YouTube).

        Runs the async ``MCPServerManager.start_all()`` in a private
        event loop on this boot thread. The manager lives on for the
        rest of the app lifetime (cleanup on shutdown handled in routes).

        Failure policy : per the boot module's docstring, we never
        abort. If MCP is disabled in settings, Node is absent, or all
        servers fail, we mark the stage as ``skipped`` so the rest of
        Lythéa stays usable.

        Rune — auto-install Node.js si manquant (désactivable via
        RUNE_AUTO_INSTALL_NODE=false). Sur un pod RunPod fraîchement
        créé, Node n'est pas présent par défaut. Plutôt que de juste
        afficher un warning et skipper, on tente l'install via apt
        (Linux) ou Homebrew (macOS). En cas d'échec, on fallback sur
        le comportement d'origine (skip + warning).
        """
        import asyncio
        import shutil
        import subprocess
        from pathlib import Path

        from rune.settings import get_settings

        self.state.begin_stage("mcp", details="initialisation…")

        s = get_settings()
        if not getattr(s, "mcp_enabled", True):
            self.state.end_stage(
                "mcp", "skipped", "désactivé dans settings (mcp_enabled=False)",
            )
            return

        # ── Rune — Auto-install Node.js si manquant ──────────────────
        # RUNE_AUTO_INSTALL_NODE=true par défaut. Mis à false si l'user
        # veut gérer Node lui-même (env contraintes, root interdit, etc.).
        auto_install_node = get_rune_settings().auto_install_node
        if not shutil.which("node") and auto_install_node:
            self.state.begin_stage("mcp", details="installation Node.js…")
            try:
                self._auto_install_node()
                # Re-vérifie
                if shutil.which("node"):
                    node_ver = subprocess.check_output(
                        ["node", "--version"], stderr=subprocess.STDOUT,
                        text=True, timeout=5,
                    ).strip()
                    log.info("Node.js auto-installé: %s", node_ver)
                else:
                    raise RuntimeError("node still not in PATH after install")
            except Exception as exc:
                log.warning(
                    "Auto-install Node.js échoué (%s) — MCP sera skippé. "
                    "Set RUNE_AUTO_INSTALL_NODE=false pour silencer.",
                    exc,
                )
                self.state.end_stage(
                    "mcp", "skipped",
                    f"Node.js absent et auto-install échoué: {exc}",
                )
                return
        elif not shutil.which("node") and not auto_install_node:
            self.state.end_stage(
                "mcp", "skipped",
                "Node.js absent (RUNE_AUTO_INSTALL_NODE=false)",
            )
            return

        # Resolve the sandbox dir. Default: ~/.lythea/sandbox/
        sandbox_dir_str = (
            getattr(s, "mcp_sandbox_dir", "")
            or str(Path.home() / ".lythea" / "sandbox")
        )
        sandbox_dir = Path(sandbox_dir_str).expanduser().resolve()

        try:
            from rune.mcp import MCPServerManager
        except Exception as exc:
            self.state.end_stage(
                "mcp", "failed", f"import MCPServerManager: {exc}",
            )
            return

        try:
            manager = MCPServerManager(sandbox_dir=sandbox_dir)
        except Exception as exc:
            self.state.end_stage(
                "mcp", "failed", f"MCPServerManager init: {exc}",
            )
            return

        # Run start_all() in a temporary loop on this boot thread.
        # We can't use the FastAPI loop here (we're in a daemon thread
        # started before uvicorn boots). The manager keeps no loop-
        # bound state across the boot transition, but its clients DO.
        # See note below : we delay client.start() to the request loop.
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(manager.start_all())
            finally:
                # NB: we don't close the loop here because the manager's
                # subprocess pipes are bound to it. We keep the loop
                # alive and run it in a dedicated background thread for
                # the rest of the app lifetime. See `_run_mcp_loop`.
                pass
        except Exception as exc:
            log.exception("MCP start_all failed: %s", exc)
            self.state.end_stage("mcp", "failed", str(exc))
            return

        # Attach the manager + loop to the Lythéa app for runtime use
        self.app.mcp_manager = manager
        self.app.mcp_loop = loop
        # V6.0.0-rc — Aussi accessibles depuis l'hippocampe pour
        # l'intégration cognitive (router → route "mcp" → appel
        # filesystem.read_file / list_directory).
        if hasattr(self.app, "hippocampe") and self.app.hippocampe is not None:
            self.app.hippocampe.mcp_manager = manager
            self.app.hippocampe.mcp_loop = loop

        # Spin a daemon thread that runs the loop forever — this is
        # how the manager keeps its subprocess pipes drained while the
        # main FastAPI loop deals with HTTP requests.
        import threading as _th
        loop_thread = _th.Thread(
            target=loop.run_forever,
            daemon=True,
            name="mcp-loop",
        )
        loop_thread.start()
        self.app.mcp_loop_thread = loop_thread

        snap = manager.snapshot()
        n_servers = sum(1 for s in snap["servers"].values() if s["alive"])
        n_tools = snap["n_tools_total"]
        if n_servers == 0:
            self.state.end_stage(
                "mcp", "skipped",
                "aucun serveur disponible (Node.js absent ou tous échec)",
            )
        else:
            self.state.end_stage(
                "mcp", "ok",
                f"{n_servers} serveur(s), {n_tools} outil(s)",
            )

    @staticmethod
    def _auto_install_node() -> None:
        """Tente d'installer Node.js automatiquement.

        Linux : apt via NodeSource (Node 20 LTS). Nécessite root ou sudo.
        macOS : Homebrew si dispo.

        Lève RuntimeError en cas d'échec.
        """
        import platform
        import shutil
        import subprocess

        system = platform.system()
        if system == "Linux":
            # Vérifie qu'on a apt
            if not shutil.which("apt-get"):
                raise RuntimeError("apt-get not available — cannot auto-install")
            # Setup NodeSource repo + install
            cmds = [
                ["bash", "-c",
                 "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"],
                ["apt-get", "install", "-y", "nodejs"],
            ]
            for cmd in cmds:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Command failed: {' '.join(cmd[:2])}… "
                        f"(exit {result.returncode}): {result.stderr[:200]}"
                    )
        elif system == "Darwin":
            # macOS — Homebrew
            if not shutil.which("brew"):
                raise RuntimeError("Homebrew not available — cannot auto-install")
            result = subprocess.run(
                ["brew", "install", "node"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"brew install node failed (exit {result.returncode}): "
                    f"{result.stderr[:200]}"
                )
        else:
            raise RuntimeError(
                f"Auto-install not supported on {system} — "
                "install Node.js manually"
            )

    # ── Rune — Stage autoload du modèle ──────────────────────────────

    def _stage_model_autoload(self) -> None:
        """Charge le modèle au boot si RUNE_AUTOLOAD_MODEL=true.

        Config
        ------
        RUNE_AUTOLOAD_MODEL : bool (défaut: false)
            Active l'autoload du modèle au boot.
        RUNE_DEFAULT_MODEL : str (défaut: DEFAULT_MODEL Lythea)
            Model ID HuggingFace à charger.

        Quand activé, le modèle est chargé en VRAM pendant le boot.
        Le stage passe en "ok" si le chargement réussit, "skipped" si
        désactivé, "failed" sinon. Le boot lui-même ne plante jamais
        sur ce stage — l'utilisateur peut toujours charger le modèle
        manuellement via POST /api/models/load après le boot.

        Note : si Trinity est activé (trinity.yaml avec enabled: true),
        ce stage est skippé car Trinity s'occupe de charger les modèles.
        """
        import os
        from rune.config import DEFAULT_MODEL

        self.state.begin_stage("model_autoload", details="vérification config…")

        # Si Trinity est activé, on skip — Trinity gère les modèles
        trinity_enabled = self._is_trinity_enabled()
        if trinity_enabled:
            self.state.end_stage(
                "model_autoload", "skipped",
                "Trinity activé — les modèles sont chargés par le stage Trinity",
            )
            return

        from rune.settings import get_rune_settings
        rs = get_rune_settings()

        if not rs.autoload_model:
            self.state.end_stage(
                "model_autoload", "skipped",
                "RUNE_AUTOLOAD_MODEL=false — chargez via POST /api/models/load",
            )
            return

        # Model ID à charger (RuneSettings > LYTHEA_DEFAULT_MODEL > config)
        model_id = (
            rs.default_model
            or os.environ.get("LYTHEA_DEFAULT_MODEL")
            or DEFAULT_MODEL
        )

        self.state.begin_stage(
            "model_autoload", details=f"chargement {model_id}…",
        )
        log.info("Autoloading model %s…", model_id)

        try:
            # Lythea HFModelWrapper.load() — appel bloquant (1-3 min)
            ok = self.app.model.load(model_id)
            if ok:
                self.state.end_stage(
                    "model_autoload", "ok",
                    f"{model_id} chargé en VRAM",
                )
                log.info("Model %s autoloaded successfully", model_id)
            else:
                self.state.end_stage(
                    "model_autoload", "failed",
                    f"load({model_id}) returned False",
                )
                log.warning("Model autoload failed (load returned False)")
        except Exception as exc:
            log.exception("Model autoload failed: %s", exc)
            self.state.end_stage("model_autoload", "failed", str(exc))

    # ── Rune — Stage Trinity (pool multi-modèles) ────────────────────

    @staticmethod
    def _is_trinity_enabled() -> bool:
        """Vérifie si Trinity est activé (sans charger toute la config)."""
        import os
        from pathlib import Path
        import yaml

        from rune.settings import get_rune_settings
        config_path = get_rune_settings().trinity_config or None
        if not config_path:
            return False
        path = Path(config_path)
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return bool(data.get("enabled", False))
        except Exception:
            return False

    def _stage_trinity(self) -> None:
        """Charge le pool Trinity si activé via trinity.yaml.

        Stage non-fatal : si Trinity échoue (VRAM insuffisante, modèle
        introuvable), le stage passe en "failed" ou "skipped" mais le
        boot continue. Rune fonctionne alors en mode single-model
        standard (Lythea d'origine).

        Config
        ------
        RUNE_TRINITY_CONFIG : str (chemin vers trinity.yaml)
            Si non set ou fichier inexistant → Trinity désactivé.
        """
        import os

        self.state.begin_stage("trinity", details="vérification config…")

        from rune.settings import get_rune_settings
        config_path = get_rune_settings().trinity_config or None
        if not config_path:
            self.state.end_stage(
                "trinity", "skipped",
                "RUNE_TRINITY_CONFIG non configuré — mode single-model",
            )
            return

        from pathlib import Path
        path = Path(config_path)
        if not path.exists():
            self.state.end_stage(
                "trinity", "skipped",
                f"fichier {config_path} introuvable — mode single-model",
            )
            return

        from rune.trinity import TrinityPool, load_trinity_config

        config = load_trinity_config(config_path)
        if not config.enabled:
            self.state.end_stage(
                "trinity", "skipped",
                "trinity.yaml: enabled=false — mode single-model",
            )
            return

        self.state.begin_stage(
            "trinity",
            details=f"chargement 3 modèles (Thinker+Worker+Critic)…",
        )
        log.info("Trinity enabled — loading 3 models from %s", config_path)

        try:
            pool = TrinityPool(config)
            report = pool.load_all()

            # Attache le pool à l'app pour utilisation runtime
            self.app.trinity_pool = pool

            if report.get("status") == "ok":
                self.state.end_stage(
                    "trinity", "ok",
                    f"3 modèles chargés (Thinker+Worker+Critic)",
                )
                log.info("Trinity pool loaded — 3 models active")
            elif report.get("status") == "partial":
                self.state.end_stage(
                    "trinity", "ok",
                    f"mode partiel — Worker OK, Thinker ou Critic manquant",
                )
                log.warning("Trinity partial — some roles missing")
            elif report.get("status") == "degraded":
                self.state.end_stage(
                    "trinity", "failed",
                    "Worker n'a pas pu charger — fallback single-model",
                )
                log.error("Trinity degraded — Worker missing")
            else:
                self.state.end_stage(
                    "trinity", "failed",
                    f"statut inconnu: {report.get('status')}",
                )
        except Exception as exc:
            log.exception("Trinity load failed: %s", exc)
            self.state.end_stage("trinity", "failed", str(exc))
