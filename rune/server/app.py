"""FastAPI application with lifespan management (headless — no static UI)."""
from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import chromadb
from fastapi import FastAPI

from rune.config import CACHE_ROOT, CHROMA_DIR, DATA_DIR, MHN_DIR, SDM_DIR
from rune.env import PLATFORM
from rune.git_sync import GitSync
from rune.hippocampe import Hippocampe
from rune.memory.kg import EntityExtractor, KnowledgeGraphStore
from rune.memory.mhn import ModernHopfieldNetwork
from rune.memory.retrieval import HybridRetriever
from rune.memory.sdm import SparseDistributedMemory
from rune.model import HFModelWrapper
from rune.server.routes import router
from rune.sessions import SessionManager
from rune.web import WebTriggerPolicy

log = logging.getLogger("rune.server.app")

# Rune : pas de dossier static (UI supprimée, mode headless)
STATIC_DIR = Path(__file__).parent / "static"  # peut ne pas exister


class LytheaApp:
    """Top-level application state holding all subsystems."""

    def __init__(self) -> None:
        self.platform = PLATFORM
        self.cache_root = CACHE_ROOT

        # Model
        self.model = HFModelWrapper()

        # Memories
        self.sdm = SparseDistributedMemory()
        self.mhn = ModernHopfieldNetwork()

        # ChromaDB — Protection (8): recovery on corruption
        self.chroma_client, self.chroma_collection = self._init_chroma()

        # KG + GLiNER
        self.entity_extractor = EntityExtractor()
        self.kg = KnowledgeGraphStore()

        # Retrieval — cross-encoder enabled (preloaded at boot, see boot.py)
        embedder_fn = lambda text: self.entity_extractor.encode(text)
        self.retriever = HybridRetriever(
            collection=self.chroma_collection,
            embedder=embedder_fn,
            use_cross_encoder=True,
        )

        # Sessions
        self.sessions = SessionManager()

        # Git
        self.git = GitSync(DATA_DIR)

        # Web policy
        self.web_policy = WebTriggerPolicy()

        # Hippocampe
        self.hippocampe = Hippocampe(
            model=self.model,
            sdm=self.sdm,
            mhn=self.mhn,
            chroma_collection=self.chroma_collection,
            git=self.git,
            kg=self.kg,
            entity_extractor=self.entity_extractor,
            web_policy=self.web_policy,
            retriever=self.retriever,
        )

        # Restore persisted session-scoped state
        sdm_path = SDM_DIR / "sdm_state.pt"
        mhn_path = MHN_DIR / "mhn_state.pt"
        self.sdm.load_state(sdm_path)
        self.mhn.load_state(mhn_path)

        # Rune — Wrap Hippocampe avec RuneCortex pour activer AutoSkill,
        # FailureMemory, TieredRetriever, et tous les hooks Rune.
        # Sans ça, l'API Lythea appelle hippocampe.process_message()
        # directement et les extensions Rune ne tournent jamais.
        self.rune_cortex: Any = None
        try:
            from rune.cortex_ext.integration import RuneCortex
            self.rune_cortex = RuneCortex(self.hippocampe)
            # Monkey-patch : remplace hippocampe.process_message par la
            # version wrappée qui ajoute AutoSkill/FailureMemory hooks.
            # Toutes les routes Lythea qui appellent
            # hippocampe.process_message() passent maintenant par RuneCortex.
            original_process = self.hippocampe.process_message
            self.rune_cortex._original_process = original_process
            self.hippocampe.process_message = self.rune_cortex.process_message
            log.info("RuneCortex wrapped — AutoSkill/FailureMemory active")
        except Exception as exc:
            log.warning("RuneCortex init failed: %s — Rune extensions disabled", exc)

        # Soft memory (opt-in). When disabled, attribute is None and
        # the /api/soft-memory/* endpoints return 503 with a clear
        # message. See lythea/soft_memory.py for the design rationale.
        self.soft_memory: Any | None = None
        try:
            from rune.settings import get_settings
            if get_settings().enable_soft_memory:
                self._init_soft_memory()
        except Exception as exc:
            log.warning("Soft memory initialisation failed: %s", exc)

        log.info("Lythéa initialized (platform=%s, cache=%s)", self.platform, self.cache_root)

    def _init_soft_memory(self) -> None:
        """Create the SoftPrefix and SoftMemoryTrainer.

        The prefix is lazily attached to the LLM the first time the
        user triggers a training round — at construction time we just
        create the parameter container with a placeholder shape that
        will be re-allocated once the LLM is loaded and we know its
        ``num_layers`` and ``hidden_dim``.
        """
        from rune.settings import get_settings
        from rune.soft_memory import SoftMemoryConfig, SoftMemoryTrainer, SoftPrefix
        s = get_settings()
        cfg = SoftMemoryConfig(
            prefix_length=s.soft_memory_prefix_length,
            learning_rate=s.soft_memory_learning_rate,
            epochs=s.soft_memory_epochs,
        )
        # Placeholder: 1 layer × 1 hidden_dim. Real shape is set when
        # the LLM is first loaded; we patch it then.
        self._soft_memory_config = cfg
        prefix = SoftPrefix(
            num_layers=1, hidden_dim=1, config=cfg, device="cpu",
        )
        self.soft_memory = SoftMemoryTrainer(prefix, cfg)
        log.info("Soft memory enabled (prefix_length=%d, lr=%g)",
                 cfg.prefix_length, cfg.learning_rate)

    @staticmethod
    def _init_chroma() -> tuple[Any, Any]:
        """Initialize ChromaDB with corruption recovery.

        V6.0.0-rc rev5 : ajout d'une étape de création explicite du
        tenant 'default_tenant' et database 'default_database'. Sans
        ça, après une purge mémoire complète, ChromaDB peut crasher
        avec "could not connect to tenant default_tenant" parce que
        le fichier sqlite est neuf mais le tenant n'est pas créé
        automatiquement par PersistentClient sur certaines versions.
        """
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        for attempt in range(3):
            try:
                client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )
                # V6.0.0-rc rev5 : s'assure que tenant/database existent.
                # Try/except parce que les versions récentes peuvent ne pas
                # exposer ces APIs au niveau du client, et selon les versions
                # la création peut lever si déjà présents.
                try:
                    from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE
                    admin = getattr(client, "_admin_client", None) or chromadb.AdminClient(
                        chromadb.Settings(anonymized_telemetry=False)
                    )
                    try:
                        admin.get_tenant(DEFAULT_TENANT)
                    except Exception:
                        try:
                            admin.create_tenant(DEFAULT_TENANT)
                            log.info("Chroma: tenant '%s' créé", DEFAULT_TENANT)
                        except Exception as _te:
                            log.debug("Chroma create_tenant skip: %s", _te)
                    try:
                        admin.get_database(DEFAULT_DATABASE, DEFAULT_TENANT)
                    except Exception:
                        try:
                            admin.create_database(DEFAULT_DATABASE, DEFAULT_TENANT)
                            log.info("Chroma: database '%s' créée", DEFAULT_DATABASE)
                        except Exception as _de:
                            log.debug("Chroma create_database skip: %s", _de)
                except Exception as _exc:
                    # Si on n'a pas accès à AdminClient (versions plus anciennes),
                    # on laisse passer — le client a déjà été créé avec succès,
                    # donc le tenant existe probablement par défaut.
                    log.debug("Chroma tenant admin step skipped: %s", _exc)

                coll = client.get_or_create_collection("lythea_memory")
                # Test de santé immédiat
                _ = coll.count()
                return client, coll
            except Exception as exc:
                if attempt < 2:
                    log.warning(
                        "Chroma init failed attempt %d (%s), rebuilding",
                        attempt + 1, exc,
                    )
                    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                    # Vider aussi le cache module pour le prochain essai
                    try:
                        if hasattr(chromadb, "_clear_system_cache"):
                            chromadb._clear_system_cache()
                    except Exception:
                        pass
                else:
                    raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init subsystems, run boot preload, then yield."""
    import os
    from rune.boot import BootState, BootRunner

    # 1. Build the LytheaApp (instantiates components but doesn't load models)
    app.state.lythea = LytheaApp()

    # 2. Set up the boot state. The UI polls /api/boot/status while ready=False.
    app.state.boot = BootState()

    # 3. Run the preload sequence in the background. The HTTP server starts
    # immediately so /api/boot/status is reachable, but other /api/* routes
    # return 503 until ready=True (see middleware in create_app).
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    runner = BootRunner(app.state.lythea, app.state.boot)
    runner.start()

    # Passerelle Telegram (optionnelle) : démarre si un token est configuré.
    # Long-polling sortant — aucun port à ouvrir ; le navigateur devient
    # facultatif (voir aussi le mode standalone `python -m rune.telegram_bot`).
    try:
        from rune.telegram_bot import start_if_configured
        app.state.telegram = start_if_configured(app.state.lythea,
                                                 app.state.boot)
    except Exception:  # noqa: BLE001
        log.warning("telegram gateway failed to start", exc_info=True)
        app.state.telegram = None

    try:
        yield
    finally:
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        lythea = app.state.lythea
        if lythea.model.is_loaded:
            lythea.model.unload()
        log.info("Lythéa shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Middleware order (Starlette executes in REVERSE order of add):
      1. Boot gate   (added first → runs LAST, after auth+limit)
      2. Rate limit  (slowapi, integrated via SlowAPIMiddleware)
      3. Auth        (added last → runs FIRST)

    So the actual request flow is:
        request → AuthMiddleware → SlowAPIMiddleware → BootGate → route
    which is what we want: reject unauthenticated requests before
    spending CPU on rate-limit accounting or boot-state checks.
    """
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.errors import RateLimitExceeded

    from rune.server.auth import AuthMiddleware, auth_banner
    from rune.server.rate_limit import (
        limiter, rate_limit_exceeded_handler,
    )
    from rune.settings import get_settings

    settings = get_settings()

    app = FastAPI(title="Lythéa V4 — Rune", lifespan=lifespan)

    # Print the auth policy banner at startup for the operator.
    log.info(auth_banner(settings.auth_token, settings.auth_strict))

    # ── Boot gate (added FIRST → runs LAST in the chain) ──────────────
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest

    BOOT_ALLOWED_PATHS = {"/api/boot/status", "/api/health"}

    class BootGateMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            path = request.url.path
            if (
                not path.startswith("/api/")
                or path in BOOT_ALLOWED_PATHS
            ):
                return await call_next(request)
            boot = getattr(app.state, "boot", None)
            if boot is None or boot.ready:
                return await call_next(request)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "still_initializing",
                    "boot_status": boot.to_dict(),
                },
            )

    app.add_middleware(BootGateMiddleware)

    # ── Rate limiting (slowapi) ───────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded, rate_limit_exceeded_handler,
    )
    app.add_middleware(SlowAPIMiddleware)

    # ── Auth (added LAST → runs FIRST) ────────────────────────────────
    # "/" est public : c'est le message d'accueil headless. Sans ça, un
    # accès au tunnel Cloudflare exigerait un token et renverrait 401 au
    # lieu du message d'accueil. "/docs" et "/openapi.json" sont aussi
    # publics pour que la doc interactive soit consultable.
    app.add_middleware(
        AuthMiddleware,
        expected_token=settings.auth_token,
        strict=settings.auth_strict,
        public_paths={
            "/",
            "/api/boot/status",
            "/api/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        },
    )

    app.include_router(router)

    # Rune : pas de serving de fichiers statiques ni d'index.html
    # (mode headless — l'UI est supprimée, l'accès se fait via API + CLI)

    return app
