"""FastAPI REST + SSE endpoints for Lythéa."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from rune.config import CATALOG
from rune.model import HFModelWrapper, vram_free_gb, vram_total_gb
from rune.server.rate_limit import make_limit_decorator
from rune.server.schemas import (
    AgentInterjectRequest,
    AgentRunRequest,
    AgentStopRequest,
    ChatRequest,
    CodegenCommitRequest,
    CodegenRequest,
    EntropyConfigRequest,
    GitConfigRequest,
    ModelLoadRequest,
    ReasoningConfigRequest,
    SamplingConfigRequest,
    SessionPatch,
    SteeringApplyRequest,
    SteeringCalibrateRequest,
    SteeringMixRequest,
    WebModeRequest,
)
from rune.sessions import Message

log = logging.getLogger("rune.server.routes")

router = APIRouter()

# FIRE-AND-FORGET : missions agent qui continuent même si le client se
# déconnecte. asyncio peut « garbage-collecter » une tâche sans référence forte
# en plein vol ; on garde donc ici une référence le temps de la mission.
_AGENT_BG_TASKS: set = set()

# Rate limit decorators — built lazily from settings so env overrides apply.
from rune.settings import get_settings as _gs
_settings = _gs()
_limit_chat = make_limit_decorator(_settings.rate_limit_chat)
_limit_model_load = make_limit_decorator(_settings.rate_limit_model_load)


def _lythea(request: Request) -> Any:
    """Get the Lythea app state."""
    return request.app.state.lythea


# ── Racine (mode headless — pas d'UI, juste un message d'accueil) ──────
# Sans cette route, ouvrir l'URL du tunnel affiche {"detail":"Not Found"}
# (404), ce qui est déroutant. On renvoie un petit message JSON qui
# pointe vers les endpoints utiles au lieu d'un 404 sec.

@router.get("/")
async def root() -> dict:
    """Message d'accueil — Rune est headless (pas d'interface web)."""
    return {
        "service": "Rune",
        "status": "running",
        "headless": True,
        "message": (
            "Rune est un agent headless (pas d'interface web). "
            "Utilise l'API ou la CLI."
        ),
        "endpoints": {
            "health": "/api/health",
            "boot_status": "/api/boot/status",
            "docs": "/docs",
            "chat": "POST /api/chat",
            "models": "/api/models",
            "load_model": "POST /api/models/load",
        },
    }


# ── Boot status (always available, even during boot) ──────────────────

@router.get("/api/boot/status")
async def boot_status(request: Request) -> dict:
    """Return current boot/preload state for the splash screen."""
    boot = getattr(request.app.state, "boot", None)
    if boot is None:
        return {"ready": True, "current_step": "done"}
    return boot.to_dict()


# ── Health ─────────────────────────────────────────────────────────────

@router.get("/api/health")
async def health(request: Request) -> dict:
    app = _lythea(request)
    return {
        "status": "ok",
        "platform": app.platform,
        "vram_free_gb": round(vram_free_gb(), 1),
        "vram_total_gb": round(vram_total_gb(), 1),
        "cache_root": str(app.cache_root),
        "model_loaded": app.model.is_loaded,
        "model_id": app.model.model_id,
    }


# ── V5.3 — Memory Health Dashboard ─────────────────────────────────────


@router.get("/api/memory/health")
async def memory_health(request: Request) -> dict:
    """V5.3 — Snapshot des métriques cognitives de la mémoire.

    Retourne 5 dimensions normalisées 0-100 :
    - freshness    : taux d'entrées Chroma récentes
    - coverage     : taux d'entités KG avec relations
    - coherence    : densité des communautés GraphRAG
    - efficiency   : ratio actif/(actif+pending)
    - reachability : connectivité du plus grand composant

    Plus un score global pondéré et un hint textuel court pour l'UI.

    Pas de cache : la valeur reflète l'état du moment, ce qui permet
    d'observer l'évolution post-microsleep (cf. consolidation phase).
    """
    app = _lythea(request)
    try:
        from rune.memory.health import compute_health

        # V5.5.2 fix : LytheaApp expose ``chroma_collection`` (pas ``chroma``).
        # Hippocampe interne utilise ``self.chroma`` mais on n'y a pas
        # accès direct depuis le router. On tombe sur le bon attribut
        # quel que soit le wrapper, pour rester robuste à des renommages
        # futurs.
        chroma_coll = (
            getattr(app, "chroma_collection", None)
            or getattr(app, "chroma", None)
        )
        kg_ref = getattr(app, "kg", None)
        if kg_ref is None:
            # Pas de KG → réponse neutre explicite plutôt qu'une exception
            return {
                "health_score": 0,
                "freshness": 0, "coverage": 0, "coherence": 0,
                "efficiency": 0, "reachability": 0,
                "n_entities": 0, "n_relations": 0, "n_communities": 0,
                "n_pending": 0, "n_chroma": 0,
                "measured_at": 0,
                "cognitive_hint": "KG non initialisé.",
            }

        snapshot = compute_health(kg=kg_ref, chroma_collection=chroma_coll)
        return snapshot.to_dict()
    except Exception as exc:
        # Pas de 500 — on retourne un payload neutre que la UI peut afficher.
        # On log avec stacktrace complète pour faciliter le debug.
        import logging
        logging.getLogger("rune.server.routes").exception(
            "Memory health computation failed",
        )
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=200,
            content={
                "error": str(exc),
                "health_score": 0,
                "freshness": 0, "coverage": 0, "coherence": 0,
                "efficiency": 0, "reachability": 0,
                "n_entities": 0, "n_relations": 0, "n_communities": 0,
                "n_pending": 0, "n_chroma": 0,
                "cognitive_hint": f"Calcul impossible : {type(exc).__name__}",
            },
        )


# ── V5.5.4 — Maintenance : purge des entités polluantes ───────────────


@router.post("/api/memory/cleanup_noise")
async def cleanup_noise_entities(request: Request) -> dict:
    """V5.5.4 — Purge les entités KG matchant ENTITY_NOISE étendu.

    Quand les filtres anti-noise sont étendus (ex : "je suis" ajouté
    après V5.5.3), les entités déjà archivées dans le KG avant le fix
    restent en place. Cet endpoint passe sur toutes les entités KG
    et supprime celles qui correspondent maintenant à un pattern noise.

    Body optionnel : ``{"dry_run": true}`` pour avoir le rapport sans
    supprimer. Par défaut, ``dry_run=false`` (suppression effective).

    Returns
    -------
    dict
        ``{"removed": [...], "kept_count": N, "dry_run": bool}``
        Les éléments removed contiennent ``{entity_id, value, label}``.
    """
    import json
    from rune.cognition.encoding import ENTITY_NOISE

    # Parse body sans dépendre du modèle pydantic (tolérant)
    body_raw = await request.body()
    dry_run = False
    if body_raw:
        try:
            body = json.loads(body_raw.decode("utf-8"))
            dry_run = bool(body.get("dry_run", False))
        except Exception:
            pass

    app = _lythea(request)
    kg = getattr(app, "kg", None)
    if kg is None:
        return {"error": "KG non initialisé", "removed": [], "kept_count": 0}

    # Snapshot des entités à supprimer (on ne mute pas pendant l'itération)
    to_remove: list[dict] = []
    for eid, ent in list(getattr(kg, "entities", {}).items()):
        # Normalisation identique à _extract_entities
        value_norm = (
            getattr(ent, "value", "")
            .lower()
            .strip()
            .replace("\u2019", "'")
            .replace("\u02bc", "'")
        )
        value_norm = " ".join(value_norm.split())
        if value_norm in ENTITY_NOISE:
            to_remove.append({
                "entity_id": eid,
                "value": getattr(ent, "value", ""),
                "label": getattr(ent, "type", ""),
            })

    # Suppression effective si pas dry_run
    if not dry_run:
        for item in to_remove:
            try:
                kg.delete_entity(item["entity_id"])
            except Exception as exc:
                item["error"] = str(exc)
        # Persiste l'état nettoyé sur disque
        try:
            kg.save()
        except Exception:
            pass

    return {
        "removed": to_remove,
        "removed_count": len(to_remove),
        "kept_count": len(getattr(kg, "entities", {})),
        "dry_run": dry_run,
    }


# ── V5.5.5 — Maintenance : purge des documents Chroma polluants ───────


@router.post("/api/memory/cleanup_chroma")
async def cleanup_chroma_docs(request: Request) -> dict:
    """V5.5.5 — Purge les documents Chroma contenant des fragments
    polluants.

    Les anciennes mauvaises extractions (ex : "Je suis" stocké comme
    prénom utilisateur) ont été archivées dans Chroma sous forme
    ``Q: ... R: ... [Atoms: Je suis, ...]``. Même si le KG est nettoyé,
    le RAG remonte ces vieux docs et le LLM les réutilise au prochain
    tour ("Salut Je suis").

    Body JSON :
    - ``dry_run`` (bool, default false) : preview sans suppression
    - ``contains`` (list[str], optionnel) : substrings à matcher dans
      le document (case-insensitive). Default = patterns connus.
    - ``last_n`` (int, optionnel) : si fourni, supprime aussi les N
      derniers documents archivés (utile quand on sait que la
      pollution est récente).

    Returns
    -------
    dict
        ``{"removed_ids": [...], "removed_count": N, "kept_count": M,
        "matched_substrings": {...}, "dry_run": bool}``
    """
    import json

    body_raw = await request.body()
    dry_run = False
    contains_filters: list[str] | None = None
    last_n: int | None = None
    if body_raw:
        try:
            body = json.loads(body_raw.decode("utf-8"))
            dry_run = bool(body.get("dry_run", False))
            contains_filters = body.get("contains")
            last_n = body.get("last_n")
            if last_n is not None:
                last_n = int(last_n)
        except Exception:
            pass

    # Patterns par défaut : "Je suis" et autres fragments noise
    # qui ont pu être archivés comme atomes.
    if contains_filters is None:
        contains_filters = [
            "[Atoms: Je suis",  # Atome au début
            ", Je suis,",        # Atome au milieu
            ", Je suis]",        # Atome à la fin
            "[Atoms: J'ai",
            ", J'ai,",
            ", J'ai]",
            "[Atoms: Emilien",
            ", Emilien,",
            ", Emilien]",
            "Salut Je suis",     # Réponses polluées par le bug
        ]

    app = _lythea(request)
    coll = getattr(app, "chroma_collection", None) or getattr(app, "chroma", None)
    if coll is None:
        return {"error": "Chroma non disponible", "removed_ids": []}

    try:
        total = coll.count()
    except Exception as exc:
        return {"error": f"Chroma count failed: {exc}", "removed_ids": []}

    # On lit tout (limite raisonnable de 5000 pour éviter d'exploser
    # la mémoire si jamais la base est très grosse)
    fetch_limit = min(total, 5000)
    try:
        result = coll.get(limit=fetch_limit, include=["documents", "metadatas"])
    except Exception as exc:
        return {"error": f"Chroma get failed: {exc}", "removed_ids": []}

    ids = result.get("ids", []) or []
    docs = result.get("documents", []) or []

    to_remove_ids: list[str] = []
    matched_substrings: dict[str, int] = {sub: 0 for sub in contains_filters}

    for doc_id, doc_text in zip(ids, docs):
        if not doc_text:
            continue
        matched = False
        for sub in contains_filters:
            if sub.lower() in doc_text.lower():
                matched_substrings[sub] = matched_substrings.get(sub, 0) + 1
                matched = True
        if matched:
            to_remove_ids.append(doc_id)

    # Last N : ajoute les N derniers documents triés par timestamp
    # de leur metadata, si fourni.
    if last_n is not None and last_n > 0:
        metas = result.get("metadatas", []) or []
        indexed = [
            (ids[i], (metas[i] or {}).get("ts", 0))
            for i in range(len(ids))
            if i < len(metas)
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)
        recent_ids = [doc_id for doc_id, _ in indexed[:last_n]]
        for rid in recent_ids:
            if rid not in to_remove_ids:
                to_remove_ids.append(rid)

    # Suppression effective si pas dry_run
    if not dry_run and to_remove_ids:
        try:
            coll.delete(ids=to_remove_ids)
        except Exception as exc:
            return {
                "error": f"Chroma delete failed: {exc}",
                "removed_ids": to_remove_ids,
                "dry_run": dry_run,
            }

    return {
        "removed_ids": to_remove_ids,
        "removed_count": len(to_remove_ids),
        "kept_count": total - (len(to_remove_ids) if not dry_run else 0),
        "matched_substrings": matched_substrings,
        "filters_applied": contains_filters,
        "dry_run": dry_run,
    }


@router.post("/api/memory/cleanup_procedural")
async def cleanup_procedural(request: Request) -> dict:
    """V5.6.8 — Purge la mémoire procédurale qui contiendrait des PII
    spécifiques (années, âges, prénoms d'anciennes sessions).

    Cas d'usage : tu vois Lythéa réutiliser "tu as 41 ans" alors que
    tu n'as rien dit dans la session courante. Cela vient d'un pattern
    procedural qui a stocké la valeur littérale dans son approach.

    Body JSON :
    - ``dry_run`` (bool, default false) : preview sans suppression
    - ``all`` (bool, default false) : si true, supprime TOUS les
      patterns proceduraux (reset complet). Sinon supprime uniquement
      ceux qui contiennent des PII non généralisés (chiffres 4 digits
      ressemblant à des années, "X ans", patterns de prénoms).

    Returns
    -------
    dict avec removed_ids, removed_count, kept_count, dry_run.
    """
    import json
    import re as _re

    body_raw = await request.body()
    dry_run = False
    delete_all = False
    if body_raw:
        try:
            body = json.loads(body_raw.decode("utf-8"))
            dry_run = bool(body.get("dry_run", False))
            delete_all = bool(body.get("all", False))
        except Exception:
            pass

    app = _lythea(request)
    proc_store = getattr(app, "procedural_memory", None)
    if proc_store is None:
        # Le store peut être attaché ailleurs selon la version
        proc_store = getattr(app.hippocampe, "procedural_memory", None)
    if proc_store is None:
        return {"error": "Procedural memory store not available"}

    # Patterns de détection PII non-généralisé
    pii_check = _re.compile(
        r"\b(19|20)\d{2}\b"           # années
        r"|\b\d{1,3}\s*ans?\b"         # âges
        r"|\b(?:michael|michaël|mika|cédric)\b",  # prénoms connus
        _re.IGNORECASE,
    )

    procedures = list(proc_store.list_active())
    total = len(procedures)
    to_remove: list[str] = []
    for proc in procedures:
        if delete_all:
            to_remove.append(proc.id)
            continue
        # Check anti-PII
        if pii_check.search(proc.trigger) or pii_check.search(proc.approach):
            to_remove.append(proc.id)

    # Suppression effective si pas dry_run
    if not dry_run and to_remove:
        for pid in to_remove:
            proc_store.archive(pid)

    return {
        "removed_ids": to_remove,
        "removed_count": len(to_remove),
        "kept_count": total - (len(to_remove) if not dry_run else 0),
        "delete_all": delete_all,
        "dry_run": dry_run,
    }


# ── Models ─────────────────────────────────────────────────────────────

@router.get("/api/models")
@router.get("/api/config/models")  # alias rétro-compat
async def list_models(request: Request) -> list[dict]:
    """List all catalogued models with current loadability info.

    Each entry includes ``loadable``, ``loadable_with_blip``,
    ``loadable_after_unload``, and a ``block_reason`` so the UI can
    grey out non-loadable models and propose smart actions.
    """
    from rune.model import model_loadability_info

    app = _lythea(request)

    # Captioner state — affects VRAM budget
    captioner = app.hippocampe.captioner
    captioner_backend = captioner.backend
    captioner_vram_gb = 0.0
    if captioner_backend == "qwen2vl":
        # See CAPTIONER_OPTIONS in config.py for sizes
        captioner_vram_gb = 4.5

    # Currently loaded LLM
    current_id = app.model.model_id if app.model.is_loaded else None
    current_size = 0.0
    if current_id:
        spec = CATALOG.get(current_id)
        if spec is not None:
            current_size = spec.size_gb

    out: list[dict] = []
    for spec in CATALOG.values():
        info = model_loadability_info(
            spec.model_id,
            captioner_backend=captioner_backend,
            captioner_vram_gb=captioner_vram_gb,
            current_loaded_id=current_id,
            current_loaded_size_gb=current_size,
        )
        out.append({
            "id": spec.model_id,
            "label": spec.label,
            "size_gb": spec.size_gb,
            "is_thinking": spec.is_thinking,
            "notes": spec.notes,
            "loaded": current_id == spec.model_id,
            **info,
        })
    return out


@router.post("/api/models/load")
@_limit_model_load
async def load_model(body: ModelLoadRequest, request: Request) -> StreamingResponse:
    """Load an LLM. Auto-unloads the previously loaded LLM if necessary.

    Strategy when a different LLM is already loaded:
      1. Check if the target is fully cached locally.
      2. If yes → unload the current model first, then load the new one
         (clean swap, never two LLMs in VRAM at once).
      3. If no → keep the current model loaded while we download, then
         do the swap. This way a network failure during download leaves
         the user with a working LLM.
    """
    from rune.model import model_loadability_info

    app = _lythea(request)
    model_id = body.model_id

    if model_id not in CATALOG:
        raise HTTPException(400, f"Model {model_id} not in catalogue")

    # If the target is already loaded, no-op succeeds immediately.
    if app.model.model_id == model_id and app.model.is_loaded:
        async def already_loaded_stream():
            import json
            yield f"data: {json.dumps({'pct': 100, 'finished': True, 'model_id': model_id, 'status': 'already_loaded'})}\n\n"
        return StreamingResponse(already_loaded_stream(), media_type="text/event-stream")

    # Decide whether we need to unload first
    current_id = app.model.model_id if app.model.is_loaded else None
    captioner = app.hippocampe.captioner
    captioner_backend = captioner.backend
    captioner_vram_gb = 4.5 if captioner_backend == "qwen2vl" else 0.0
    current_size = 0.0
    if current_id:
        spec = CATALOG.get(current_id)
        if spec is not None:
            current_size = spec.size_gb

    info = model_loadability_info(
        model_id,
        captioner_backend=captioner_backend,
        captioner_vram_gb=captioner_vram_gb,
        current_loaded_id=current_id,
        current_loaded_size_gb=current_size,
    )

    if not info["loadable"] and not info["loadable_after_unload"]:
        raise HTTPException(400, info["block_reason"] or "VRAM insuffisante")

    needs_unload_first = (
        current_id is not None and not info["loadable"]
        and info["loadable_after_unload"]
    )

    async def stream():
        import queue as queue_mod

        loop = asyncio.get_event_loop()
        done = asyncio.Event()
        error_ref: list[str] = []
        progress_q: queue_mod.Queue = queue_mod.Queue()

        def _swap_and_load():
            try:
                if needs_unload_first and current_id:
                    log.info("Auto-unloading %s before loading %s", current_id, model_id)
                    progress_q.put({
                        "pct": 1, "status": "unloading_previous",
                        "model_id": model_id, "previous": current_id,
                    })
                    app.model.unload()
                app.model.load(model_id, progress_q=progress_q)
            except Exception as exc:
                error_ref.append(str(exc))
            finally:
                loop.call_soon_threadsafe(done.set)

        thread = threading.Thread(target=_swap_and_load, daemon=True)
        thread.start()

        last_pct = 0.0
        last_status = "loading"
        last_extra: dict = {}
        while not done.is_set():
            while True:
                try:
                    data = progress_q.get_nowait()
                    last_pct = data.get("pct", last_pct)
                    last_status = data.get("status", last_status)
                    if "previous" in data:
                        last_extra["previous"] = data["previous"]
                except Exception:
                    break
            payload = {
                "pct": last_pct, "status": last_status, "model_id": model_id,
                **last_extra,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.4)

        if error_ref:
            yield f"data: {json.dumps({'error': error_ref[0], 'finished': True})}\n\n"
        else:
            # Apply the model's recommended sampling profile, if any.
            # Each model family has its own sweet spot (Mistral=0.5,
            # Qwen=0.7, LFM2=0.1-0.3, thinking models=0.6) and the
            # profile is auto-applied so users don't have to manually
            # tune sampling for every model swap. User overrides via
            # /api/config/sampling are runtime-only and are intentionally
            # reset here on model swap — the new model's recommendations
            # take precedence.
            from rune.config import DEFAULT_SAMPLING
            spec = CATALOG.get(model_id)
            new_profile = spec.sampling if (spec and spec.sampling) else DEFAULT_SAMPLING
            app.hippocampe.sampling_profile = new_profile
            log.info(
                "Sampling profile applied for %s: T=%.2f, top_p=%s, top_k=%s, "
                "min_p=%s, rep=%.2f",
                model_id, new_profile.temperature, new_profile.top_p,
                new_profile.top_k, new_profile.min_p,
                new_profile.repetition_penalty,
            )
            # V5.6.15 — Auto-unload du captionneur Qwen2-VL/BLIP quand
            # le modèle nouvellement chargé est multimodal natif
            # (Gemma 3/4). Pas besoin d'un captionneur séparé puisque
            # le modèle principal peut traiter les images directement.
            # Économise la VRAM allouée au captionneur (~3-4 GB).
            try:
                if app.model.is_natively_multimodal:
                    current_backend = app.hippocampe.captioner.backend
                    if current_backend and current_backend != "none":
                        log.info(
                            "Modèle %s nativement multimodal — désactivation "
                            "du captionneur %s (libération VRAM)",
                            model_id, current_backend,
                        )
                        app.hippocampe.captioner.select("none")
            except Exception as exc:
                log.warning("Échec auto-unload captionneur: %s", exc)

            yield f"data: {json.dumps({'pct': 100, 'finished': True, 'model_id': model_id})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/api/models/unload")
async def unload_model(request: Request) -> Response:
    app = _lythea(request)
    app.model.unload()
    return Response(status_code=204)


@router.get("/api/models/current")
async def current_model(request: Request) -> dict | None:
    app = _lythea(request)
    m = app.model
    if not m.is_loaded:
        return {"loaded": False}
    spec = CATALOG.get(m.model_id)
    # Expose the catalogue's recommended sampling separately from the
    # currently active profile, so the UI can offer a "↺ Reset to
    # recommended" action that reverts user overrides without having
    # to reload the whole model.
    recommended = None
    if spec is not None and spec.sampling is not None:
        s = spec.sampling
        recommended = {
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "min_p": s.min_p,
            "repetition_penalty": s.repetition_penalty,
            "max_new_tokens": s.max_new_tokens,
        }
    # Budget de taille de document calculé depuis le contexte du modèle.
    # Formule : context_tokens * CHARS_PER_TOKEN * UPLOAD_BUDGET_RATIO
    # → 25% du contexte total alloué aux uploads. Les 75% restants
    # sont pour le prompt système, l'historique, le RAG, la question
    # et la réponse à générer.
    _CHARS_PER_TOKEN = 4  # approximation pour le français
    _UPLOAD_BUDGET_RATIO = 0.25
    context_tokens = m.context_length or 4096
    max_doc_chars = int(context_tokens * _CHARS_PER_TOKEN * _UPLOAD_BUDGET_RATIO)
    return {
        "loaded": True,
        "id": m.model_id,
        "model_id": m.model_id,  # alias for UI clarity
        "label": (spec.label if spec else m.model_id),
        "hidden_dim": m.hidden_dim,
        "is_thinking": m.is_thinking,
        # V5.6.15 — Flag pour que l'UI sache afficher le banner "vision
        # native activée" dans Settings → Vision et masquer la liste
        # des captionneurs (inutile en mode natif multimodal).
        "is_natively_multimodal": getattr(m, "is_natively_multimodal", False),
        "context_length": context_tokens,
        "max_doc_chars": max_doc_chars,
        "reasoning_enabled": app.hippocampe.reasoning_enabled,
        "recommended_sampling": recommended,
    }


# ── Sessions ───────────────────────────────────────────────────────────

@router.get("/api/sessions")
async def list_sessions(request: Request) -> list[dict]:
    app = _lythea(request)
    return app.sessions.list_sessions()


@router.post("/api/sessions")
async def create_session(request: Request) -> dict:
    app = _lythea(request)
    session = app.sessions.create()
    app.hippocampe.reset_session()
    return {"session_id": session.session_id, "title": session.title}


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request, offset: int = 0, limit: int = 50) -> dict:
    app = _lythea(request)
    session = app.sessions.get(session_id, offset=offset, limit=limit)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session.session_id,
        "title": session.title,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "images": m.images,
                "thoughts": m.thoughts,
                "doubt_index": m.doubt_index,
                "epistemic": m.epistemic,
            }
            for m in session.messages
        ],
        "metadata": {
            "created": session.created,
            "last_activity": session.last_activity,
            "pinned": session.pinned,
        },
    }


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> Response:
    app = _lythea(request)
    app.sessions.delete(session_id)
    return Response(status_code=204)


@router.delete("/api/sessions")
async def delete_all_sessions(request: Request) -> dict:
    """Delete every session. Returns the count for the UI to confirm."""
    app = _lythea(request)
    count = app.sessions.delete_all()
    return {"deleted": count}


@router.patch("/api/sessions/{session_id}")
async def patch_session(session_id: str, body: SessionPatch, request: Request) -> dict:
    app = _lythea(request)
    updates = body.model_dump(exclude_none=True)
    if not app.sessions.update(session_id, **updates):
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@router.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, request: Request) -> Response:
    app = _lythea(request)
    md = app.sessions.export_markdown(session_id)
    if md is None:
        raise HTTPException(404, "Session not found")
    return Response(content=md, media_type="text/markdown")


# ── Chat (SSE via POST) ───────────────────────────────────────────────

@router.post("/api/chat")
@_limit_chat
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    app = _lythea(request)

    if not app.model.is_loaded:
        raise HTTPException(400, "Aucun modèle chargé")

    session = app.sessions.get(body.session_id, limit=10000)
    if session is None:
        raise HTTPException(404, "Session not found")

    # Decode images
    pil_images: list[Image.Image] = []
    for img_input in body.images:
        try:
            raw = base64.b64decode(img_input.data)
            pil_img = Image.open(io.BytesIO(raw))
            pil_img = HFModelWrapper.preprocess_image(pil_img)
            pil_images.append(pil_img)
        except Exception as exc:
            log.warning("Image decode failed: %s", exc)

    # Build history for the model
    history = [{"role": m.role, "content": m.content} for m in session.messages]

    # Capture temporal anchors BEFORE adding the new user message:
    #   - last_message_ts: when the previous turn happened (gap awareness)
    #   - session_created_ts: when this conversation began (duration awareness)
    # These power the [Conscience du temps] block injected into the system
    # prompt by Hippocampe so Rune reacts naturally to time gaps.
    last_message_ts = (
        session.messages[-1].timestamp if session.messages else None
    )
    session_created_ts = session.created or None

    # Save user message
    user_msg = Message(
        role="user",
        content=body.message,
        images=[img.data[:50] + "..." for img in body.images],
    )
    app.sessions.add_message(body.session_id, user_msg)

    cancelled = threading.Event()

    # ── Préparation du contexte documents (V4.2) ───────────────────────
    # Les documents attachés sont déjà extraits côté serveur via
    # /api/upload/document.
    #
    # Mode "attach" : le texte est injecté en contexte du message — la
    # génération le voit directement et peut répondre dessus.
    # Le doc n'est pas persisté.
    #
    # Mode "ingest" : le doc est ingéré dans ChromaDB pour les tours
    # futurs (RAG). MAIS on injecte AUSSI le texte (tronqué) dans le
    # message du tour courant — sinon la synthèse immédiate ne marche
    # pas : la query RAG portant sur la note système, le retrieval
    # remonte des chunks hors sujet et le modèle hallucine. La logique
    # symétrique de attach :
    #   • attach  → utilisé MAINTENANT, oublié après
    #   • ingest  → utilisé MAINTENANT *et* disponible plus tard via RAG
    document_prefix = ""
    for doc in body.documents:
        if doc.mode == "attach" and doc.text:
            document_prefix += (
                f"\n\n[Document joint — {doc.filename}]\n"
                f"{doc.text}\n"
                f"[Fin du document {doc.filename}]\n"
            )
        elif doc.mode == "ingest":
            document_prefix += (
                f"\n\n[Note système — l'utilisateur vient d'ajouter le "
                f"document « {doc.filename} » à ta mémoire long-terme. "
                f"Son contenu complet est désormais disponible via le RAG ; "
                f"un extrait pour ce tour suit.]\n"
            )
            if doc.text:
                # Le client envoie déjà un extrait borné (~16KB) ; on
                # ajoute une marque de troncature au cas où il aurait
                # envoyé tout le doc. La taille seuil correspond au
                # cap envoyé par le client (DOC_ATTACH_MAX_CHARS = 16000).
                is_truncated = len(doc.text) >= 15800
                truncation_note = (
                    "\n[…extrait tronqué — le contenu intégral est dans "
                    "la mémoire long-terme, interrogeable via questions "
                    "ciblées]" if is_truncated else ""
                )
                document_prefix += (
                    f"\n[Contenu du document — {doc.filename}]\n"
                    f"{doc.text}{truncation_note}\n"
                    f"[Fin de l'extrait]\n"
                )

    effective_message = (
        f"{document_prefix}\n\n{body.message}".strip()
        if document_prefix else body.message
    )

    # V6.0.0-rc — Si le message texte est vide mais qu'il y a des
    # fichiers ou images joints, on synthétise un user_intent par
    # défaut. Sinon, les heuristiques basées sur user_intent (router
    # cognitif, trivialité, web temporel) reçoivent une string vide
    # et se comportent bizarrement. Ce message implicite oriente
    # Lythéa vers une réaction "tu m'as donné X, qu'est-ce que je dois
    # en faire ?" plutôt que d'essayer de répondre à du vide.
    implicit_intent = body.message
    if not body.message.strip():
        attached_names = [d.filename for d in body.documents if hasattr(d, "filename")]
        if attached_names:
            files_str = ", ".join(attached_names)
            implicit_intent = (
                f"[message implicite — pas de texte fourni] "
                f"J'ai joint {len(attached_names)} fichier(s) "
                f"({files_str}). Que dois-je en faire ? "
                f"Réagis de façon brève en proposant 2-3 actions "
                f"possibles à partir du contenu."
            )
        elif body.images:
            implicit_intent = (
                f"[message implicite — pas de texte fourni] "
                f"J'ai joint {len(body.images)} image(s). "
                f"Décris brièvement ce que tu vois et propose "
                f"ce qu'on peut en faire."
            )
        # Si message vide ET rien joint, le validator pydantic aura
        # déjà refusé la requête en amont (cf schemas.py).

    async def stream():
        loop = asyncio.get_event_loop()
        result_queue: asyncio.Queue = asyncio.Queue()
        final_text = ""
        thoughts_collected: list[str] = []

        def _generate():
            # Verrou partagé chat/agent : le chat prend le MÊME verrou que
            # l'agent autour de sa génération → plus de collision quand on
            # chatte pendant une mission. ON par défaut (cf. settings).
            import contextlib
            from rune.genlock import GENERATION_LOCK
            from rune.settings import get_settings
            _shared = bool(getattr(get_settings(), "agent_chat_shared_lock_enabled", True))
            _lock_cm = GENERATION_LOCK if _shared else contextlib.nullcontext()
            try:
                with _lock_cm:
                    for event in app.hippocampe.process_message(
                        effective_message, history, pil_images, cancelled,
                        last_message_ts=last_message_ts,
                        session_created_ts=session_created_ts,
                        user_intent_message=implicit_intent,
                    ):
                        loop.call_soon_threadsafe(result_queue.put_nowait, event)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    result_queue.put_nowait,
                    {"type": "error", "data": {"message": str(exc)}},
                )
            finally:
                loop.call_soon_threadsafe(result_queue.put_nowait, None)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        try:
            while True:
                if await request.is_disconnected():
                    cancelled.set()
                    break

                try:
                    event = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if event is None:
                    break

                event_type = event["type"]
                event_data = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event_type}\ndata: {event_data}\n\n"

                if event_type == "cognitive":
                    thoughts_collected.extend(event["data"].get("items", []))
                elif event_type == "done":
                    final_text = event["data"].get("final_text", "")

                    # Save assistant message
                    assistant_msg = Message(
                        role="assistant",
                        content=final_text,
                        thoughts=thoughts_collected,
                        doubt_index=event["data"].get("doubt_index"),
                        epistemic=event["data"].get("epistemic"),
                    )
                    app.sessions.add_message(body.session_id, assistant_msg)
        except Exception:
            cancelled.set()
        finally:
            thread.join(timeout=10)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Memory ─────────────────────────────────────────────────────────────

@router.get("/api/memory/status")
async def memory_status(request: Request) -> dict:
    app = _lythea(request)
    return app.hippocampe.memory_status()


@router.get("/api/agent/status")
async def agent_status(request: Request) -> dict:
    """État live de la mission agentique en cours, pour le dashboard.

    Lit le registre ``_runs`` de l'orchestrateur EXISTANT (sans le
    construire — pas de 503 si aucun modèle chargé) : mission courante,
    fil des dernières actions de l'agent (buffer d'events), et compteurs
    du blackboard. Renvoie ``running: false`` si aucune mission.
    """
    app = _lythea(request)
    ao = getattr(app, "agent_orchestrator", None)
    if ao is None:
        return {"running": False, "current_mission": {}, "recent_events": [],
                "blackboard": {}}

    runs = getattr(ao, "_runs", {}) or {}
    # Mission « courante » = le run le plus récent non terminé, sinon le
    # dernier tout court (pour afficher le résultat juste après la fin).
    active = [(rid, r) for rid, r in runs.items() if not getattr(r, "done", False)]
    picked = None
    if active:
        picked = max(active, key=lambda kv: getattr(kv[1], "started_at", 0.0))
    elif runs:
        picked = max(runs.items(), key=lambda kv: getattr(kv[1], "started_at", 0.0))

    if picked is None:
        return {"running": False, "current_mission": {}, "recent_events": [],
                "blackboard": {}}

    run_id, run = picked
    import time as _t
    mission = {
        "run_id": run_id,
        "task": getattr(run, "task", ""),
        "name": getattr(run, "name", ""),
        "elapsed_sec": round(_t.time() - getattr(run, "started_at", _t.time()), 1),
        "done": bool(getattr(run, "done", False)),
    }
    recent_events = list(getattr(run, "events", []))

    # Blackboard : lu depuis le disque de la mission si présent.
    blackboard: dict = {}
    try:
        slug = getattr(run, "slug", "") or ""
        if slug:
            from rune.agentic.blackboard import MissionBlackboard
            ws = _workspace_manager(request)
            root = getattr(ws, "root", None)
            if root is not None:
                bb_path = root / "missions" / slug / "blackboard.json"
                if bb_path.exists():
                    bb = MissionBlackboard.load(bb_path)
                    blackboard = bb.to_dict()
    except Exception:  # noqa: BLE001 — le status ne casse jamais
        blackboard = {}

    return {
        "running": not mission["done"],
        "current_mission": mission,
        "recent_events": recent_events,
        "blackboard": blackboard,
    }


@router.get("/api/cache/stats")
async def cache_stats(request: Request) -> dict:
    """Expose embedding cache stats — useful for tuning cache sizes.

    Returns hit/miss counters for each cache (entity_extractor.encode,
    model.analyze_input). A persistently low hit rate suggests either
    the cache is too small, or your usage pattern doesn't have enough
    repetition for caching to help.
    """
    app = _lythea(request)
    out: dict = {}
    try:
        out["encode"] = app.entity_extractor.cache_stats()
    except Exception:
        out["encode"] = None
    try:
        out["analyze_input"] = app.model.cache_stats()
    except Exception:
        out["analyze_input"] = None
    return out


@router.post("/api/cache/clear")
async def cache_clear(request: Request) -> dict:
    """Manually flush both embedding caches."""
    app = _lythea(request)
    cleared: list[str] = []
    try:
        app.entity_extractor._encode_cache.clear()
        cleared.append("encode")
    except Exception:
        pass
    try:
        app.model._analyze_cache.clear()
        cleared.append("analyze_input")
    except Exception:
        pass
    return {"cleared": cleared}


# ── Soft memory (opt-in prefix-tuning) ─────────────────────────────────

def _get_soft_memory_or_503(app: Any) -> Any:
    """Return the soft memory trainer or raise 503 if disabled.

    Soft memory is opt-in via LYTHEA_ENABLE_SOFT_MEMORY=1. When
    disabled, the trainer attribute is None and we surface a clear
    503 instead of an obscure attribute error.
    """
    sm = getattr(app, "soft_memory", None)
    if sm is None:
        raise HTTPException(
            503,
            "Soft memory disabled. Set LYTHEA_ENABLE_SOFT_MEMORY=1 to enable.",
        )
    return sm


@router.get("/api/soft-memory/status")
async def soft_memory_status(request: Request) -> dict:
    """Return soft memory state: enabled, prefix shape, checkpoints."""
    app = _lythea(request)
    sm = getattr(app, "soft_memory", None)
    if sm is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "prefix_length": sm.config.prefix_length,
        "checkpoints": sm.list_checkpoints(),
    }


@router.post("/api/soft-memory/train")
async def soft_memory_train(request: Request) -> dict:
    """Run one training round on the soft prefix.

    Pulls examples from Chroma, runs gradient steps, and saves a
    new checkpoint. The base LLM must be loaded.
    """
    app = _lythea(request)
    sm = _get_soft_memory_or_503(app)
    if not app.model.is_loaded:
        raise HTTPException(400, "Charge un LLM avant d'entraîner la soft memory.")

    examples = sm.collect_dataset(app.chroma_collection)
    if not examples:
        raise HTTPException(
            400, "Aucune donnée d'entraînement dans Chroma. "
                 "Discute un peu avec Rune d'abord pour accumuler de la mémoire.",
        )
    stats = sm.train(examples, app.model.model, app.model.tokenizer)
    checkpoint_id = sm.save_checkpoint(label="auto-train")
    return {"stats": stats, "checkpoint": checkpoint_id}


@router.post("/api/soft-memory/rollback")
async def soft_memory_rollback(request: Request) -> dict:
    """Reload a previous checkpoint by id."""
    body = await request.json()
    checkpoint_id = body.get("checkpoint_id", "")
    if not checkpoint_id:
        raise HTTPException(400, "checkpoint_id required")
    app = _lythea(request)
    sm = _get_soft_memory_or_503(app)
    ok = sm.rollback_to(checkpoint_id)
    if not ok:
        raise HTTPException(404, f"Unknown checkpoint: {checkpoint_id}")
    return {"rolled_back_to": checkpoint_id}


@router.post("/api/soft-memory/reset")
async def soft_memory_reset(request: Request) -> dict:
    """Wipe the soft prefix and all its checkpoints."""
    app = _lythea(request)
    sm = _get_soft_memory_or_503(app)
    sm.reset()
    return {"reset": True}


@router.post("/api/memory/deep-sleep")
async def deep_sleep(request: Request) -> dict:
    app = _lythea(request)
    msg = app.hippocampe.deep_sleep()
    return {"message": msg}


@router.get("/api/memory/kg/entities")
async def kg_entities(request: Request) -> list[dict]:
    app = _lythea(request)
    return app.hippocampe.kg.get_all_entities()


@router.delete("/api/memory/kg/entities/{entity_id}")
async def delete_kg_entity(entity_id: str, request: Request) -> dict:
    app = _lythea(request)
    if not app.hippocampe.kg.delete_entity(entity_id):
        raise HTTPException(404, "Entity not found")
    app.hippocampe.kg.save()
    return {"ok": True}


# ── Config ─────────────────────────────────────────────────────────────

@router.post("/api/config/git")
async def config_git(body: GitConfigRequest, request: Request) -> dict:
    app = _lythea(request)
    ok = app.hippocampe.git.configure(body.token, body.user, body.repo)
    if not ok:
        raise HTTPException(500, "Git configuration failed")
    return {"ok": True}


@router.post("/api/config/entropy")
async def config_entropy(body: EntropyConfigRequest, request: Request) -> dict:
    app = _lythea(request)
    app.hippocampe.entropy_threshold = body.threshold
    return {"threshold": body.threshold}


@router.get("/api/config/entropy")
async def get_entropy(request: Request) -> dict:
    """Return the current entropy threshold so the UI can initialise
    its slider with the live value instead of a hardcoded HTML default.

    Without this endpoint, the UI shows the HTML attribute ``value=0.2``
    even if the user has overridden the setting via env, leading to a
    desync between the displayed value and the backend state — and
    worse, clicking "Save" on the unmodified slider would reset the
    backend to the displayed (stale) value.
    """
    app = _lythea(request)
    return {"threshold": app.hippocampe.entropy_threshold}


@router.post("/api/config/web-mode")
async def config_web_mode(body: WebModeRequest, request: Request) -> dict:
    app = _lythea(request)
    app.hippocampe.web_policy.mode = body.mode
    return {"mode": body.mode}


@router.get("/api/config/web-mode")
async def get_web_mode(request: Request) -> dict:
    """Return the current web search mode (off / auto / always) so the
    UI dropdown can initialise on the actual backend state."""
    app = _lythea(request)
    return {"mode": app.hippocampe.web_policy.mode}


def _profile_to_dict(profile: Any) -> dict:
    """Serialise a SamplingProfile to a JSON-safe dict (None preserved)."""
    return {
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "top_k": profile.top_k,
        "min_p": profile.min_p,
        "repetition_penalty": profile.repetition_penalty,
        "max_new_tokens": profile.max_new_tokens,
    }


@router.get("/api/config/sampling")
async def get_sampling(request: Request) -> dict:
    """Return the active sampling profile.

    Used by the UI to populate the generation sliders when the page
    loads or when a model is swapped (the loaded model's recommended
    profile is auto-applied — see ``load_model``).

    Also returns the source model_id so the UI can label the section
    "Profil recommandé pour <model>".
    """
    app = _lythea(request)
    profile = app.hippocampe.sampling_profile
    payload = _profile_to_dict(profile)
    payload["model_id"] = app.model.model_id if app.model.is_loaded else None
    return payload


@router.post("/api/config/sampling")
async def config_sampling(
    body: SamplingConfigRequest, request: Request,
) -> dict:
    """Override one or more sampling parameters at runtime.

    Only the fields explicitly provided in the request body are
    updated — the others keep their current values. This makes the
    UI sliders independent (moving one doesn't reset the others).

    Overrides are runtime-only and are reset whenever a different
    model is loaded (see ``load_model`` — the new model's
    recommended profile takes precedence).
    """
    from dataclasses import replace

    app = _lythea(request)
    current = app.hippocampe.sampling_profile

    # Build kwargs for replace(): only include fields that were sent.
    # ``None`` is a valid value (means "disable that step") so we
    # distinguish absent-from-body vs explicit-None via model_fields_set.
    updates: dict[str, Any] = {}
    for name in (
        "temperature", "top_p", "top_k", "min_p",
        "repetition_penalty", "max_new_tokens",
    ):
        if name in body.model_fields_set:
            updates[name] = getattr(body, name)

    new_profile = replace(current, **updates) if updates else current
    app.hippocampe.sampling_profile = new_profile
    log.info(
        "Sampling override: %s → T=%.2f, top_p=%s, top_k=%s, min_p=%s, "
        "rep=%.2f, max_tokens=%s",
        list(updates.keys()) or "no-op",
        new_profile.temperature, new_profile.top_p, new_profile.top_k,
        new_profile.min_p, new_profile.repetition_penalty,
        new_profile.max_new_tokens,
    )
    return _profile_to_dict(new_profile)


@router.post("/api/config/clear-cache")
async def clear_cache(request: Request) -> dict:
    """V5.6.12 — Delete all HuggingFace caches to free disk space.

    Cette version supprime de façon exhaustive :
    - Tous les `models--*` (poids LLM + captionneur)
    - Tous les `datasets--*` et `spaces--*`
    - Le dossier `snapshots/`, `blobs/` orphelins, `refs/`
    - Les fichiers `.lock` et fichiers temporaires `.incomplete`
    - Le cache `torch/` (compilations, hub Torch)
    - Aussi bien dans CACHE_ROOT que dans le cache HF par défaut
      (`~/.cache/huggingface`)
    """
    import shutil
    from pathlib import Path
    from rune.config import CACHE_ROOT

    app = _lythea(request)

    # Unload everything first (libère VRAM avant de toucher au disque)
    if app.model.is_loaded:
        app.model.unload()
    app.hippocampe.captioner.select("none")

    freed = 0
    deleted_paths: list[str] = []

    # Liste exhaustive des emplacements à nettoyer
    cache_roots = [
        CACHE_ROOT / "hf",
        CACHE_ROOT / "torch",
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
    ]

    for cache_root in cache_roots:
        if not cache_root.exists():
            continue
        try:
            # Calcule la taille avant suppression
            size = sum(
                f.stat().st_size
                for f in cache_root.rglob("*")
                if f.is_file()
            )
            freed += size
            shutil.rmtree(cache_root, ignore_errors=True)
            deleted_paths.append(str(cache_root))
            log.info("Cache supprimé : %s (%.1f GB)", cache_root, size / 1e9)
        except Exception as exc:
            log.warning("Échec suppression %s: %s", cache_root, exc)

    # Recréer le dossier HF vide pour les prochains downloads
    (CACHE_ROOT / "hf" / "hub").mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "torch").mkdir(parents=True, exist_ok=True)

    freed_gb = round(freed / 1e9, 2)
    log.info("Cache cleared: %.2f GB freed across %d locations", freed_gb, len(deleted_paths))
    return {
        "freed_gb": freed_gb,
        "deleted_paths": deleted_paths,
        "message": f"{freed_gb} GB libérés ({len(deleted_paths)} caches supprimés)",
    }


@router.post("/api/memory/wipe_all")
async def wipe_all_memory(request: Request) -> dict:
    """V5.6.12 — Efface TOUTE la mémoire persistante de Lythéa.

    Supprime intégralement :
    - **Chroma DB** : `/workspace/.lythea/chroma/` (sémantique long-terme)
    - **KG** : `entities.json`, `relations.json`, `communities.json` si présent
    - **SDM** : `/workspace/.lythea/data/sdm/` (Sparse Distributed Memory)
    - **MHN** : `/workspace/.lythea/data/mhn/` (Modern Hopfield Network)
    - **Sessions** : tous les `sess_*.json` et `_index.json`
    - **Procedural** : `skills.json` (patterns appris)

    NE supprime PAS :
    - Les modèles téléchargés (utiliser `/api/config/clear-cache` pour ça)
    - Les fichiers de config Lythéa

    Au prochain démarrage, Lythéa recréera des dossiers vides et
    repartira sans aucun souvenir.

    Returns
    -------
    dict avec ``cleared_paths``, ``failed_paths``, ``message``.
    """
    import shutil
    from rune.config import (
        CHROMA_DIR, KG_DIR, SDM_DIR, MHN_DIR,
        SESSIONS_DIR, PROCEDURAL_DIR,
    )

    app = _lythea(request)

    # On vide les structures Python in-memory en premier (sinon le
    # contenu actuel serait re-sauvegardé sur disque par le prochain
    # microsleep ou consolidation, annulant le wipe).
    in_memory_cleared: list[str] = []

    try:
        # KG en mémoire
        if hasattr(app.hippocampe, "kg") and app.hippocampe.kg is not None:
            app.hippocampe.kg.entities.clear()
            if hasattr(app.hippocampe.kg, "relations"):
                app.hippocampe.kg.relations.clear()
            if hasattr(app.hippocampe.kg, "communities"):
                try:
                    app.hippocampe.kg.communities.clear()
                except Exception:
                    pass
            in_memory_cleared.append("kg.entities/relations/communities")
    except Exception as exc:
        log.warning("Échec clear KG in-memory: %s", exc)

    try:
        # SDM en mémoire (tenseur)
        if hasattr(app.hippocampe, "sdm") and app.hippocampe.sdm is not None:
            app.hippocampe.sdm.contents.zero_()
            if hasattr(app.hippocampe.sdm, "addresses"):
                # Garde les adresses random mais reset les contenus
                pass
            in_memory_cleared.append("sdm.contents")
    except Exception as exc:
        log.warning("Échec clear SDM in-memory: %s", exc)

    try:
        # MHN en mémoire
        if hasattr(app.hippocampe, "mhn") and app.hippocampe.mhn is not None:
            mhn = app.hippocampe.mhn
            if hasattr(mhn, "patterns"):
                mhn.patterns = []
            # V5.6.13 — n_stored est typiquement une property sans setter
            # (calculée depuis len(self.patterns)). On essaie le setter,
            # et si échec on tente _n_stored ou _stored qui sont les
            # attributs privés sous-jacents.
            for attr in ("n_stored", "_n_stored", "_stored", "stored_count"):
                try:
                    setattr(mhn, attr, 0)
                    break
                except (AttributeError, TypeError):
                    continue
            in_memory_cleared.append("mhn.patterns")
    except Exception as exc:
        log.warning("Échec clear MHN in-memory: %s", exc)

    try:
        # Procedural en mémoire
        proc_store = getattr(app, "procedural_memory", None) or \
            getattr(app.hippocampe, "procedural_memory", None) or \
            getattr(app.hippocampe, "procedural_store", None)
        if proc_store is not None and hasattr(proc_store, "_procedures"):
            proc_store._procedures.clear()
            in_memory_cleared.append("procedural._procedures")
    except Exception as exc:
        log.warning("Échec clear procedural in-memory: %s", exc)

    try:
        # V5.7.0 — Visual Working Memory : clear buffer images
        vmem = getattr(app.hippocampe, "visual_memory", None)
        if vmem is not None:
            n_cleared = vmem.clear()
            in_memory_cleared.append(f"visual_memory ({n_cleared} images)")
    except Exception as exc:
        log.warning("Échec clear visual_memory in-memory: %s", exc)

    try:
        # Chroma en mémoire : on reset la collection
        chroma = getattr(app.hippocampe, "chroma", None)
        if chroma is not None:
            try:
                # Récupère tous les IDs et les supprime
                all_data = chroma.get()
                if all_data and all_data.get("ids"):
                    chroma.delete(ids=all_data["ids"])
                in_memory_cleared.append(
                    f"chroma.delete({len(all_data.get('ids', []))} docs)"
                )
            except Exception as exc:
                log.warning("Échec reset chroma collection: %s", exc)
    except Exception as exc:
        log.warning("Échec clear chroma in-memory: %s", exc)

    # Maintenant on supprime les fichiers sur disque
    cleared_paths: list[str] = []
    failed_paths: list[dict] = []

    paths_to_clear = [
        ("chroma", CHROMA_DIR),
        ("kg", KG_DIR),
        ("sdm", SDM_DIR),
        ("mhn", MHN_DIR),
        ("sessions", SESSIONS_DIR),
        ("procedural", PROCEDURAL_DIR),
    ]

    for name, path in paths_to_clear:
        if not path.exists():
            continue
        try:
            shutil.rmtree(path, ignore_errors=False)
            cleared_paths.append(name)
            log.info("Mémoire effacée : %s (%s)", name, path)
        except Exception as exc:
            log.warning("Échec wipe %s: %s", path, exc)
            failed_paths.append({"path": str(path), "error": str(exc)})

    # Recréer les dossiers vides (sinon les modules persisters vont
    # crasher au prochain save). On utilise les mêmes paths que au boot.
    for _, path in paths_to_clear:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.warning("Échec recréation %s: %s", path, exc)

    # Réinitialiser le _index.json des sessions
    try:
        index_path = SESSIONS_DIR / "_index.json"
        index_path.write_text("{}", encoding="utf-8")
    except Exception:
        pass

    # V5.6.16 — B0 fix : Reconnexion AGRESSIVE Chroma après wipe.
    #
    # Le bug "'RustBindingsAPI' object has no attribute 'bindings'" venait
    # de ce que chromadb maintient un singleton SystemRunner au niveau du
    # module qui garde un handle Rust vers l'ancien fichier sqlite. Même
    # PersistentClient.reset() + nouveau PersistentClient() ne libère pas
    # ce singleton, donc le nouveau client hérite du binding cassé.
    #
    # Solution V5.6.16 : on appelle System.stop() sur le singleton, on
    # vide les caches internes chromadb, puis on instancie un nouveau
    # client qui crée un fresh System. Comme dernière ligne de défense
    # on tente aussi de réinitialiser le client via chromadb.Client.clear.
    reconnected: list[str] = []
    try:
        from rune.server.app import LytheaApp
        import chromadb

        # 1. Stop l'ancien client + son System singleton
        old_client = getattr(app, "chroma_client", None)
        if old_client is not None:
            # Essaye stop() (méthode officielle de System runner)
            for stop_method in ("stop", "_admin_client_stop", "reset", "close"):
                try:
                    fn = getattr(old_client, stop_method, None)
                    if callable(fn):
                        fn()
                        log.debug("Chroma client.%s() called", stop_method)
                except Exception:
                    continue
            # Si l'objet expose son System interne, on le stoppe aussi
            try:
                sys_obj = getattr(old_client, "_system", None)
                if sys_obj is not None and hasattr(sys_obj, "stop"):
                    sys_obj.stop()
                    log.debug("Chroma internal _system.stop() called")
            except Exception:
                pass

        # 2. Vider le cache interne du module chromadb (singleton SystemRunner)
        try:
            if hasattr(chromadb, "_clear_system_cache"):
                chromadb._clear_system_cache()
            # Variante alternative : api.client_id_cache
            if hasattr(chromadb, "api") and hasattr(chromadb.api, "client_id_cache"):
                try:
                    chromadb.api.client_id_cache.clear()
                except Exception:
                    pass
        except Exception as exc:
            log.debug("chromadb cache clear partial: %s", exc)

        # 3. Force garbage collection des objets Rust orphelins
        import gc
        gc.collect()

        # 4. Recrée client + collection (nouveau System propre)
        new_client, new_coll = LytheaApp._init_chroma()
        app.chroma_client = new_client
        app.chroma_collection = new_coll

        # 5. Propage la nouvelle collection partout
        app.hippocampe.chroma = new_coll
        # V5.8.1 — FIX B0 final : propager aussi au storage_phase qui
        # garde sa propre référence figée. Sinon l'archive Chroma de
        # chaque échange continue d'utiliser l'ancienne collection
        # morte et logger "Chroma archive failed" indéfiniment.
        if hasattr(app.hippocampe, "storage_phase"):
            app.hippocampe.storage_phase.chroma = new_coll
        if hasattr(app, "retriever") and app.retriever is not None:
            if hasattr(app.retriever, "collection"):
                app.retriever.collection = new_coll
            if hasattr(app.retriever, "_collection"):
                app.retriever._collection = new_coll

        # 6. Test de santé immédiat : ping count() sur la nouvelle coll
        try:
            test_count = new_coll.count()
            log.info("Chroma reconnect health check OK (count=%d)", test_count)
            reconnected.append("chroma")
        except Exception as exc:
            log.warning("Chroma reconnect health check FAILED: %s", exc)
            reconnected.append("chroma (unhealthy — restart pod required)")
    except Exception as exc:
        log.warning("Échec reconnexion Chroma après wipe: %s", exc)

    # Reset KG : V5.8.1 — on vide EN PLACE plutôt que de réinstancier.
    #
    # Bug observé en V5.8.0 : la réinstanciation propageait à
    # app.kg et app.hippocampe.kg, mais OUBLIAIT les phases qui
    # avaient reçu le KG au moment de l'init (StoragePhase,
    # RetrievalPhase, ReasoningGenerator, DeepReasoning). Ces phases
    # gardaient une référence figée vers l'ancien KG, donc le storage
    # écrivait dans l'ancien, le retrieval lisait l'ancien, et le
    # compteur (qui pointait vers le nouveau) affichait 0.
    #
    # clear_in_place() vide les structures internes sur place. Toutes
    # les références externes pointent toujours sur la même instance,
    # désormais vide. Pas besoin de propagation.
    try:
        kg = getattr(app, "kg", None) or getattr(app.hippocampe, "kg", None)
        if kg is not None:
            if hasattr(kg, "clear_in_place"):
                cleared_counts = kg.clear_in_place()
                in_memory_cleared.append(
                    f"kg (in-place: {cleared_counts['entities']} ents, "
                    f"{cleared_counts['relations']} rels)"
                )
            else:
                # Fallback rétro-compat : réinstancier + propager partout
                from rune.memory.kg import KnowledgeGraphStore
                new_kg = KnowledgeGraphStore()
                app.kg = new_kg
                app.hippocampe.kg = new_kg
                # Propagation aux phases qui en gardent une référence
                for phase_attr in (
                    "storage_phase", "retrieval_phase",
                    "reasoning_generator", "deep_reasoning",
                ):
                    phase = getattr(app.hippocampe, phase_attr, None)
                    if phase is not None and hasattr(phase, "kg"):
                        phase.kg = new_kg
                in_memory_cleared.append("kg (reinstanciated + propagated)")
            reconnected.append("kg")
    except Exception as exc:
        log.warning("Échec reset KG après wipe: %s", exc, exc_info=True)

    log.info(
        "Wipe memory complete: %d paths cleared, %d failed, %d in-memory structures reset, %d handles reconnected",
        len(cleared_paths), len(failed_paths), len(in_memory_cleared), len(reconnected),
    )
    return {
        "cleared_paths": cleared_paths,
        "failed_paths": failed_paths,
        "in_memory_cleared": in_memory_cleared,
        "reconnected": reconnected,
        "message": (
            f"Mémoire effacée : {len(cleared_paths)} zones disque, "
            f"{len(in_memory_cleared)} structures RAM, "
            f"{len(reconnected)} handles reconnectés"
        ),
    }


@router.post("/api/config/reasoning")
async def config_reasoning(body: ReasoningConfigRequest, request: Request) -> dict:
    app = _lythea(request)
    app.hippocampe.reasoning_enabled = body.enabled
    return {"enabled": app.hippocampe.reasoning_enabled}


@router.get("/api/config/reasoning")
async def get_reasoning(request: Request) -> dict:
    app = _lythea(request)
    is_thinking = app.model.is_thinking if app.model.is_loaded else False
    return {
        "enabled": app.hippocampe.reasoning_enabled,
        "is_thinking_model": is_thinking,
    }


@router.post("/api/config/debug")
async def config_debug(request: Request) -> dict:
    app = _lythea(request)
    body = await request.json()
    app.hippocampe.debug_mode = body.get("enabled", False)
    return {"enabled": app.hippocampe.debug_mode}


@router.get("/api/config/debug")
async def get_debug(request: Request) -> dict:
    app = _lythea(request)
    return {"enabled": app.hippocampe.debug_mode}


# ── Captioner ─────────────────────────────────────────────────────────

@router.get("/api/captioner")
async def list_captioners(request: Request) -> dict:
    from rune.config import CAPTIONER_OPTIONS
    app = _lythea(request)
    captioner = app.hippocampe.captioner
    return {
        "options": list(CAPTIONER_OPTIONS.values()),
        "selected": captioner.selected,
        "backend": captioner.backend,
    }


# ── V5.7.0 — Visual Working Memory (Vision active) ──────────────────

@router.get("/api/cognition/visual_memory")
async def get_visual_memory(request: Request) -> dict:
    """Snapshot de la mémoire visuelle de travail.

    Retourne config + entries (sans les bytes image) + stats cumulées.
    Utile pour le debug et la visualisation UI.
    """
    app = _lythea(request)
    vmem = getattr(app.hippocampe, "visual_memory", None)
    if vmem is None:
        return {"enabled": False, "n_active": 0, "buffer": []}
    return vmem.get_state()


@router.post("/api/cognition/visual_memory/clear")
async def clear_visual_memory(request: Request) -> dict:
    """Vide complètement la mémoire visuelle de travail.

    Utile en debug ou si l'utilisateur veut "oublier" les images
    récemment envoyées sans toucher au reste de la session.
    """
    app = _lythea(request)
    vmem = getattr(app.hippocampe, "visual_memory", None)
    if vmem is None:
        return {"cleared": 0}
    n = vmem.clear()
    return {"cleared": n}


# ── V5.7.2 — Python executor trace (debug only) ──────────────────────

@router.get("/api/cognition/python_last")
async def get_python_last_execution(request: Request) -> dict:
    """Détails de la dernière exécution Python dans le sandbox.

    Retourne le code généré, le résultat complet (stdout/stderr/result/
    duration/plots/error), et le timestamp si dispo.

    Utilisé par le panneau debug 🔬 pour la transparence sur les calculs
    exécutés en sandbox. N'expose que si une exécution a effectivement
    eu lieu — sinon retourne {has_execution: false}.
    """
    app = _lythea(request)
    hippo = app.hippocampe
    code = getattr(hippo, "_last_python_code", "") or ""
    result = getattr(hippo, "_last_python_result", {}) or {}
    if not code:
        return {"has_execution": False}
    # On sérialise proprement (les plots sont en base64, on les garde).
    return {
        "has_execution": True,
        "code": code,
        "result": {
            "ok": result.get("ok", False),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "result": result.get("result", ""),
            "duration_ms": result.get("duration_ms", 0),
            "plots": result.get("plots", []),
            "error": result.get("error"),
        },
    }


# ── V6.0.0 — MCP (Model Context Protocol) endpoints ──────────────────

@router.get("/api/cognition/mcp_status")
async def get_mcp_status(request: Request) -> dict:
    """Diagnostic des serveurs MCP : Node.js, serveurs alive, outils.

    Utilisé par l'UI de debug et pour vérifier en CLI/curl que la stack
    MCP est correctement initialisée. Retourne aussi la liste des outils
    par serveur — pratique pour savoir ce que Lythéa peut faire.
    """
    app = _lythea(request)
    manager = getattr(app, "mcp_manager", None)
    if manager is None:
        return {
            "available": False,
            "reason": "MCPServerManager non initialisé (boot incomplet ou désactivé)",
        }
    snap = manager.snapshot()
    snap["tools"] = [
        {
            "qualified_name": t.qualified_name,
            "server": t.server,
            "name": t.name,
            "description": t.description[:200],
        }
        for t in manager.list_tools()
    ]
    snap["available"] = manager.is_available()
    return snap


@router.post("/api/cognition/mcp_call")
async def call_mcp_tool(request: Request) -> dict:
    """Endpoint de test manuel pour invoquer un outil MCP.

    Body JSON :
      {"server": "filesystem", "tool": "list_directory",
       "arguments": {"path": "/path/here"}}

    Non utilisé par le pipeline cognitif (Lythéa appelle directement
    le manager). Sert au debug / aux tests manuels en curl.
    """
    app = _lythea(request)
    manager = getattr(app, "mcp_manager", None)
    if manager is None:
        return {"ok": False, "error": "MCP manager not initialized"}

    body = await request.json()
    server = body.get("server", "")
    tool = body.get("tool", "")
    arguments = body.get("arguments", {})

    if not server or not tool:
        return {"ok": False, "error": "missing 'server' or 'tool' in body"}

    # The manager's call_tool is async, but it must run on the MCP
    # loop (where the clients live), not the FastAPI loop. We schedule
    # it on the MCP loop and wait for the result.
    import asyncio
    mcp_loop = getattr(app, "mcp_loop", None)
    if mcp_loop is None:
        return {"ok": False, "error": "MCP loop not running"}

    future = asyncio.run_coroutine_threadsafe(
        manager.call_tool(server, tool, arguments, timeout=30.0),
        mcp_loop,
    )
    try:
        # run_coroutine_threadsafe returns a concurrent.futures.Future.
        # asyncio.wrap_future bridges it to an awaitable on the current
        # event loop (FastAPI's loop).
        result = await asyncio.wrap_future(future)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/captioner/select")
async def select_captioner(request: Request) -> dict:
    app = _lythea(request)
    body = await request.json()
    choice = body.get("choice", "auto")
    result = app.hippocampe.captioner.select(choice)
    return result


@router.post("/api/models/load-with-blip")
@_limit_model_load
async def load_with_blip(body: ModelLoadRequest, request: Request) -> StreamingResponse:
    """Switch captioner to BLIP (CPU) then load the requested LLM.

    Convenience endpoint for the "switch captioner and load" button on
    grayed-out model cards. The two operations stream their progress
    through the same SSE channel so the UI shows a continuous progress
    bar from 0% to 100%.
    """
    app = _lythea(request)
    model_id = body.model_id

    if model_id not in CATALOG:
        raise HTTPException(400, f"Model {model_id} not in catalogue")

    async def stream():
        # Phase 1 (0-10%): captioner switch
        yield f"data: {json.dumps({'pct': 2, 'status': 'switching_captioner', 'model_id': model_id})}\n\n"
        try:
            app.hippocampe.captioner.select("blip")
            yield f"data: {json.dumps({'pct': 10, 'status': 'captioner_switched', 'model_id': model_id})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Captioner switch failed: {exc}', 'finished': True})}\n\n"
            return

        # Phase 2 (10-100%): LLM load. We re-use the same logic as load_model
        # but rescale progress from [0,100] to [10,100].
        import queue as queue_mod
        loop = asyncio.get_event_loop()
        done = asyncio.Event()
        error_ref: list[str] = []
        progress_q: queue_mod.Queue = queue_mod.Queue()

        def _do_load():
            try:
                if app.model.is_loaded and app.model.model_id != model_id:
                    app.model.unload()
                app.model.load(model_id, progress_q=progress_q)
            except Exception as exc:
                error_ref.append(str(exc))
            finally:
                loop.call_soon_threadsafe(done.set)

        thread = threading.Thread(target=_do_load, daemon=True)
        thread.start()

        last_pct_inner = 0.0
        last_status = "loading"
        while not done.is_set():
            while True:
                try:
                    data = progress_q.get_nowait()
                    last_pct_inner = data.get("pct", last_pct_inner)
                    last_status = data.get("status", last_status)
                except Exception:
                    break
            scaled = 10 + (last_pct_inner * 0.9)
            yield f"data: {json.dumps({'pct': scaled, 'status': last_status, 'model_id': model_id})}\n\n"
            await asyncio.sleep(0.4)

        if error_ref:
            yield f"data: {json.dumps({'error': error_ref[0], 'finished': True})}\n\n"
        else:
            # Apply the model's sampling profile (mirror of load_model).
            from rune.config import DEFAULT_SAMPLING
            spec = CATALOG.get(model_id)
            new_profile = spec.sampling if (spec and spec.sampling) else DEFAULT_SAMPLING
            app.hippocampe.sampling_profile = new_profile
            log.info(
                "Sampling profile applied for %s (load-with-blip): T=%.2f",
                model_id, new_profile.temperature,
            )
            yield f"data: {json.dumps({'pct': 100, 'finished': True, 'model_id': model_id})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── V3.9 cascade ──────────────────────────────────────────────────────


@router.get("/api/config/cascade")
async def get_cascade_config(request: Request) -> dict:
    """Return the V3.9 cascade state.

    Surfaces:
      * Whether the cascade is enabled and ready
      * The Gemini model id in use
      * A masked view of the API key (last 4 chars, never the full key)
      * The local daily quota counter (used / remaining)
      * The synthesis thresholds

    Returns a structured payload the UI can render directly. The full
    API key is never exposed by this or any other endpoint.
    """
    app = _lythea(request)
    return app.hippocampe.cascade_status()


@router.post("/api/config/cascade/toggle")
async def toggle_cascade(request: Request) -> dict:
    """Toggle the cascade ON/OFF at runtime.

    V3.9.4 addition. Lets the user enable or disable the Gemini
    cascade per-conversation without editing ``.env`` or restarting
    Lythéa. Mirrors the runtime-only override pattern already used
    for ``/api/config/web-mode`` and ``/api/config/entropy``.

    On reboot, the ``LYTHEA_ENABLE_CASCADE`` setting from ``.env``
    takes precedence again, so this is a transient override (not
    persisted to disk).

    Body: ``{"enabled": true}`` or ``{"enabled": false}``. Empty body
    flips the current state.

    Returns the new state in the same format as
    :func:`get_cascade_config`.
    """
    app = _lythea(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    desired = body.get("enabled")

    current = app.hippocampe.cascade_enabled
    if desired is None:
        # No body — flip current state.
        new_state = not current
    else:
        new_state = bool(desired)

    if new_state == current:
        # Nothing to do.
        return app.hippocampe.cascade_status()

    if new_state:
        # Re-enable: rebuild the cascade from current settings.
        from rune.settings import get_settings
        s = get_settings()
        # Force enable_cascade = True on a settings copy so the
        # builder doesn't bail out, but keep .env untouched.
        # We override the in-memory snapshot; .env wins on next boot.
        cascade_settings_override = type(
            "Override", (), {**vars(s), "enable_cascade": True},
        )()
        app.hippocampe._cascade = (
            app.hippocampe._build_cascade_if_enabled(cascade_settings_override)
        )
    else:
        # Disable: drop the cascade so the streaming path takes over.
        app.hippocampe._cascade = None

    return app.hippocampe.cascade_status()


@router.post("/api/config/cascade/key")
async def set_cascade_key(request: Request) -> dict:
    """Set (or clear) the Gemini API key at runtime, from the UI.

    Lets the user paste a Google AI Studio key directly in the app
    instead of editing ``.env``. The key is held in RAM only — never
    written to disk, never logged in clear, masked in any readback —
    and is lost on reboot, at which point ``.env`` takes over again.

    Body: ``{"api_key": "AIzaSy..."}``. An empty/absent key clears the
    runtime override and falls back to ``.env``.

    On a valid key the cascade is rebuilt and enabled immediately.
    Returns the same payload as :func:`get_cascade_config`.
    """
    app = _lythea(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    raw = (body.get("api_key") or "").strip()
    from rune.settings import get_settings

    # Empty → clear the override and rebuild from .env.
    if not raw:
        app.hippocampe._cascade_key_override = None
        app.hippocampe._cascade = (
            app.hippocampe._build_cascade_if_enabled(get_settings())
        )
        return app.hippocampe.cascade_status()

    # Validate format before storing (never log the key itself).
    from rune.external.gemini_client import validate_api_key_format
    if not validate_api_key_format(raw):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Format de clé Google invalide (attendu : AIzaSy + 33 caractères).",
        )

    app.hippocampe._cascade_key_override = raw
    # Rebuild + enable with the new key (RAM override; .env wins on reboot).
    s = get_settings()
    cascade_settings_override = type(
        "Override", (), {**vars(s), "enable_cascade": True},
    )()
    app.hippocampe._cascade = (
        app.hippocampe._build_cascade_if_enabled(cascade_settings_override)
    )
    return app.hippocampe.cascade_status()


# ── V4 cognitive modules: live status + runtime toggle ────────────────


_V4_KNOWN_MODULES = {
    "cognitive_state",
    "inhibition",
    "planning",
    "predictive_coding",
    "timeline",
    "metacognition",
    "affect_modulates_consolidation",
    "predictive_coding_apply_gating",
}


@router.get("/api/config/v4")
async def get_v4_status(request: Request) -> dict:
    """Return the live V4 module status.

    See ``Hippocampe.v4_status`` for the structure. Used by the UI
    panel to render switches + per-module diagnostics. No secrets,
    never raises.
    """
    app = _lythea(request)
    try:
        return app.hippocampe.v4_status()
    except Exception:
        # Defensive — even if introspection fails, surface an empty
        # but well-formed snapshot so the UI doesn't crash.
        return {
            "cognitive_state": {"enabled": False},
            "inhibition": {"enabled": False},
            "planning": {"enabled": False},
            "predictive_coding": {"enabled": False},
            "timeline": {"enabled": False},
            "metacognition": {"enabled": False},
            "affect_modulates_consolidation": False,
        }


@router.post("/api/config/v4/toggle")
async def toggle_v4_module(request: Request) -> dict:
    """Enable / disable a V4 module at runtime.

    Body: ``{"module": "<name>", "enabled": true|false}``.
    Empty ``enabled`` flips the current state.

    Module ∈ cognitive_state, inhibition, planning, predictive_coding,
    timeline, metacognition, affect_modulates_consolidation,
    predictive_coding_apply_gating.

    Like ``/api/config/cascade/toggle``, this is a runtime override —
    on reboot the ``LYTHEA_ENABLE_*`` env vars take precedence.
    """
    app = _lythea(request)
    try:
        body = await request.json()
    except Exception:
        body = {}

    module = (body.get("module") or "").strip()
    if module not in _V4_KNOWN_MODULES:
        return {
            "error": "unknown_module",
            "module": module,
            "known": sorted(_V4_KNOWN_MODULES),
        }

    desired = body.get("enabled")
    # Snapshot pre-toggle so we know the current state if we need to flip.
    current_status = app.hippocampe.v4_status()
    if desired is None:
        # Flip: read current state from the snapshot.
        if module in {"affect_modulates_consolidation", "predictive_coding_apply_gating"}:
            cur = bool(current_status.get(
                module if module != "predictive_coding_apply_gating"
                else "predictive_coding", {}
            ).get(
                "apply_gating" if module == "predictive_coding_apply_gating" else module,
                False,
            )) if module == "predictive_coding_apply_gating" else bool(
                current_status.get("affect_modulates_consolidation", False)
            )
        else:
            cur = bool(current_status.get(module, {}).get("enabled", False))
        new_state = not cur
    else:
        new_state = bool(desired)

    return app.hippocampe.v4_set_module(module, new_state)


# ── Document upload (V4.2) ─────────────────────────────────────────────

# Cap de taille volontairement large (50 MB) — un PDF de thèse fait
# quelques MB. Le vrai garde-fou côté coût n'est pas la taille du
# fichier mais la longueur du texte extrait, gérée plus bas.
_DOCUMENT_MAX_BYTES: int = 50 * 1024 * 1024


@router.post("/api/upload/document")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    mode: str = "attach",
    tag: str | None = None,
):
    """Upload + traitement d'un document (PDF, txt, md, docx, rst).

    Deux modes :

    - ``mode=attach`` — extrait le texte et le retourne au client.
      Le texte sera injecté en contexte du prochain message.
    - ``mode=ingest`` — extrait + chunke + ingère dans ChromaDB +
      extrait les entités vers le KG. Le document devient mémoire
      long-terme de Rune.
    """
    from rune.server.document_ingest import (
        extract_uploaded_document,
        ingest_document_to_memory,
        is_supported,
    )

    if mode not in ("attach", "ingest"):
        raise HTTPException(
            status_code=400,
            detail=f"mode invalide : {mode!r} (attendu 'attach' ou 'ingest')",
        )

    if not is_supported(file.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=f"Extension non supportée : {file.filename!r}",
        )

    data = await file.read()
    if len(data) > _DOCUMENT_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop gros ({len(data) // 1024} KB > "
                   f"{_DOCUMENT_MAX_BYTES // 1024} KB)",
        )

    app = _lythea(request)

    # ── Validation taille texte vs contexte du modèle ─────────────────
    # On extrait d'abord le texte, puis on vérifie qu'il tient dans le
    # budget upload du modèle chargé. Le rejet est explicite avec
    # un message qui dit la limite — l'utilisateur peut alors soit
    # charger un modèle avec plus de contexte, soit découper son doc.
    try:
        text, n_chars = extract_uploaded_document(data, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Document extraction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if app.model.is_loaded:
        _CHARS_PER_TOKEN = 4
        _UPLOAD_BUDGET_RATIO = 0.25
        context_tokens = app.model.context_length or 4096
        max_doc_chars = int(
            context_tokens * _CHARS_PER_TOKEN * _UPLOAD_BUDGET_RATIO
        )
        if n_chars > max_doc_chars:
            model_label = app.model.model_id or "modèle chargé"
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Document trop long pour {model_label} : "
                    f"{n_chars:,} caractères extraits, max {max_doc_chars:,} "
                    f"(25 % d'un contexte de {context_tokens:,} tokens). "
                    f"Charge un modèle avec plus de contexte, ou découpe "
                    f"le fichier en plusieurs morceaux."
                ).replace(",", " "),
            )

    try:
        if mode == "attach":
            return {
                "filename": file.filename,
                "mode": "attach",
                "text": text,
                "n_chars": n_chars,
            }
        else:
            result = ingest_document_to_memory(
                data, file.filename,
                hippocampe=app.hippocampe,
                tag=tag,
            )
            result["mode"] = "ingest"
            return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Document upload failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── V6.0.0-α2 — Workspace endpoints ──────────────────────────────────
#
# The workspace is a shared directory between the user (via these
# HTTP endpoints, called from the sidebar UI) and Lythéa (via the
# filesystem MCP server, scoped to the same dir). It's the bridge
# between the browser file picker and Lythéa's tools.
#
# All paths in request bodies are relative to the sandbox root.
# Absolute paths and ``..`` traversal attempts are rejected.

def _workspace_manager(request: Request):
    """Lazily build (or fetch cached) the workspace manager for this app."""
    app = _lythea(request)
    manager = getattr(app, "workspace_manager", None)
    if manager is not None:
        return manager
    # Build from settings if not yet attached
    from pathlib import Path
    from rune.settings import get_settings
    from rune.server.workspace import WorkspaceManager
    s = get_settings()
    sandbox_str = (
        getattr(s, "mcp_sandbox_dir", "")
        or str(Path.home() / ".lythea" / "sandbox")
    )
    manager = WorkspaceManager(
        sandbox_dir=Path(sandbox_str),
        max_file_bytes=getattr(s, "mcp_workspace_max_file_mb", 20) * 1024 * 1024,
        max_total_bytes=getattr(s, "mcp_workspace_max_total_mb", 200) * 1024 * 1024,
    )
    app.workspace_manager = manager  # cache it
    return manager


@router.get("/api/workspace/files")
async def workspace_list_files(request: Request) -> dict:
    """List the workspace as a tree, plus quota stats.

    Returns
    -------
    {
      "tree": {<root FileEntry>},
      "stats": {"total_files": N, "total_size_bytes": ..., ...}
    }
    """
    manager = _workspace_manager(request)
    return {
        "tree": manager.list_tree().to_dict(),
        "stats": manager.stats().to_dict(),
    }


@router.post("/api/workspace/upload")
async def workspace_upload(
    request: Request,
    file: UploadFile = File(...),
    target_dir: str = Form(""),
) -> dict:
    """Upload a file into the workspace.

    Body : multipart with ``file`` and optional ``target_dir`` form field
    (relative subdirectory inside the sandbox; empty = root).

    Returns the metadata of the saved file or an HTTP error.

    Note V6.0.0-α2 fix : target_dir DOIT être annoté avec Form(),
    pas `str = ""`. Sinon FastAPI le traite comme un query param et
    le multipart parsing peut échouer avec une erreur 422 obscure.
    """
    from rune.server.workspace import (
        WorkspacePathError, WorkspaceSizeError, WorkspaceTypeError,
    )
    manager = _workspace_manager(request)
    data = await file.read()
    try:
        entry = manager.save_upload(
            filename=file.filename or "upload",
            data=data,
            target_dir=target_dir,
        )
    except WorkspaceTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except WorkspaceSizeError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Workspace upload failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "file": entry.to_dict()}


@router.get("/api/workspace/download")
async def workspace_download(request: Request, path: str) -> StreamingResponse:
    """Stream a file from the workspace.

    Query : ``?path=relative/path/in/sandbox.csv``
    """
    from rune.server.workspace import WorkspacePathError
    manager = _workspace_manager(request)
    try:
        abs_path, mime = manager.open_for_download(path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # URL-quote the filename in the Content-Disposition header so
    # accented filenames don't get mangled by HTTP transport.
    from urllib.parse import quote
    disposition = (
        f"attachment; filename*=UTF-8''{quote(abs_path.name)}"
    )
    return StreamingResponse(
        manager.iter_chunks(abs_path),
        media_type=mime,
        headers={
            "Content-Disposition": disposition,
            "Content-Length": str(abs_path.stat().st_size),
        },
    )


@router.delete("/api/workspace/files")
async def workspace_delete(request: Request) -> dict:
    """Delete a file or directory from the workspace.

    Body : ``{"path": "relative/path.txt"}``
    """
    from rune.server.workspace import WorkspacePathError
    body = await request.json()
    rel_path = body.get("path", "")
    if not rel_path:
        raise HTTPException(status_code=400, detail="path required")
    manager = _workspace_manager(request)
    try:
        manager.delete(rel_path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception("Workspace delete failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True}


@router.post("/api/workspace/rename")
async def workspace_rename(request: Request) -> dict:
    """Rename a file or directory in place (same parent).

    Body : ``{"path": "old/path.txt", "new_name": "newname.txt"}``
    """
    from rune.server.workspace import (
        WorkspacePathError, WorkspaceTypeError,
    )
    body = await request.json()
    rel_path = body.get("path", "")
    new_name = body.get("new_name", "")
    if not rel_path or not new_name:
        raise HTTPException(
            status_code=400, detail="path and new_name required"
        )
    manager = _workspace_manager(request)
    try:
        entry = manager.rename(rel_path, new_name)
    except WorkspaceTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except WorkspacePathError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception("Workspace rename failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "file": entry.to_dict()}


@router.post("/api/workspace/ingest")
async def workspace_ingest(request: Request) -> dict:
    """V6.0.0-rc — Ingère un fichier du workspace dans la mémoire long-terme.

    Body : ``{"path": "relative/path/in/sandbox.pdf"}``

    Le fichier est extrait, chunké, vectorisé dans ChromaDB, et ses
    entités sont poussées dans le KG via le pipeline existant de
    /api/upload/document?mode=ingest. La différence : ici le fichier
    est DÉJÀ dans le workspace, on ne re-uploade rien.

    Returns
    -------
    {"ok": True, "n_chars": N, "n_chunks": N, "n_entities": N, ...}
    """
    from rune.server.workspace import WorkspacePathError
    body = await request.json()
    rel_path = body.get("path", "")
    if not rel_path:
        raise HTTPException(status_code=400, detail="path required")

    manager = _workspace_manager(request)
    try:
        abs_path, _mime = manager.open_for_download(rel_path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Lire le contenu binaire
    try:
        data = abs_path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Lecture impossible : {exc}")

    # Réutilise le pipeline ingest existant
    app = _lythea(request)
    if not hasattr(app, "hippocampe") or app.hippocampe is None:
        raise HTTPException(
            status_code=503,
            detail="Hippocampe non initialisé — impossible d'ingérer en mémoire.",
        )

    from rune.server.document_ingest import ingest_document_to_memory, is_supported
    if not is_supported(abs_path.name):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Format non supporté pour l'ingestion : {abs_path.suffix}. "
                f"Formats acceptés : PDF, TXT, MD, DOCX, RST."
            ),
        )

    try:
        result = ingest_document_to_memory(
            data, abs_path.name,
            hippocampe=app.hippocampe,
            tag=f"workspace:{rel_path}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Workspace ingest failed for %s", rel_path)
        raise HTTPException(status_code=500, detail=str(exc))

    result["ok"] = True
    result["mode"] = "ingest"
    return result


# ══════════════════════════════════════════════════════════════════════
# Codegen — multi-file extraction from an answer (V6)
# ══════════════════════════════════════════════════════════════════════
# Rune emits one fenced block per file (marqueur « # file: chemin »).
# These routes turn the raw answer Markdown into individual downloads,
# a project .zip, or files written into the workspace sandbox. Parsing
# is centralised in rune.server.codegen (single source of truth).

@router.post("/api/codegen/extract")
async def codegen_extract(body: CodegenRequest, request: Request) -> dict:
    """Parse an answer and return the declared code files (no side effect).

    Returns ``{"count": N, "files": [{"path","lang","content"}, ...]}``.
    """
    from rune.server.codegen import extract_code_files
    files = extract_code_files(body.text)
    return {"count": len(files), "files": [f.to_dict() for f in files]}


@router.post("/api/codegen/zip")
async def codegen_zip(body: CodegenRequest, request: Request) -> StreamingResponse:
    """Build and stream a ``.zip`` of all code files in the answer."""
    from rune.server.codegen import build_zip, extract_code_files
    files = extract_code_files(body.text)
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Aucun fichier détecté (marqueur « # file: » absent).",
        )
    blob = build_zip(files)
    headers = {
        "Content-Disposition": "attachment; filename*=UTF-8''projet-rune.zip",
        "Content-Length": str(len(blob)),
    }
    return StreamingResponse(
        iter([blob]), media_type="application/zip", headers=headers
    )


@router.post("/api/codegen/commit")
async def codegen_commit(body: CodegenCommitRequest, request: Request) -> dict:
    """Write the answer's code files into the workspace sandbox.

    Optional ``subdir`` namespaces the project. Returns the written paths,
    any per-file errors, and the refreshed tree/stats so the sidebar can
    update without a second round-trip.
    """
    from rune.server.codegen import extract_code_files
    from rune.server.workspace import WorkspaceError

    files = extract_code_files(body.text)
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Aucun fichier détecté (marqueur « # file: » absent).",
        )

    manager = _workspace_manager(request)
    subdir = (body.subdir or "").strip().strip("/")
    written: list[dict] = []
    skipped: list[dict] = []
    for cf in files:
        rel = f"{subdir}/{cf.path}" if subdir else cf.path
        try:
            entry = manager.write_text_file(rel, cf.content)
            written.append(entry.to_dict())
        except WorkspaceError as exc:
            skipped.append({"path": rel, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — surface per-file, keep going
            log.exception("codegen commit failed for %s", rel)
            skipped.append({"path": rel, "error": str(exc)})

    return {
        "ok": not skipped,
        "written": written,
        "skipped": skipped,
        "tree": manager.list_tree().to_dict(),
        "stats": manager.stats().to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════
# Agentic mode — bounded plan/act/critique loop (V6 Phase 1)
# ══════════════════════════════════════════════════════════════════════
# The AgentOrchestrator is a sibling of the chat Hippocampe: it composes
# the SAME model + memory (shared by reference) and adds a multi-step,
# interruptible loop over a worker pool. The chat path is untouched.

def _agent_orchestrator(request: Request):
    """Lazily build (or fetch cached) the agent orchestrator for this app."""
    app = _lythea(request)
    ao = getattr(app, "agent_orchestrator", None)
    if ao is not None:
        return ao
    from rune.agentic import AgentOrchestrator, WorkerPool
    from rune.settings import get_settings

    hippo = getattr(app, "hippocampe", None)
    if hippo is None or getattr(hippo, "model", None) is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    settings = get_settings()
    pool = WorkerPool.from_settings(hippo.model, settings)
    ao = AgentOrchestrator(
        hippocampe=hippo,
        worker_pool=pool,
        workspace_manager=_workspace_manager(request),
        settings=settings,
        # Sandboxed test execution + reactive install. Default ON (per-mission
        # venv keeps the pod env safe); override with agent_execution=false.
        execution_enabled=bool(getattr(settings, "agent_execution", True)),
        # Model-driven tool-calling loop (Hermès). Off by default until the
        # execution foundation is validated live; flip with agent_react=true.
        react_enabled=bool(getattr(settings, "agent_react", False)),
    )
    app.agent_orchestrator = ao
    return ao


@router.post("/api/agent/run")
async def agent_run(body: AgentRunRequest, request: Request) -> StreamingResponse:
    """Launch an agentic run; stream steps as SSE (event:/data: pairs)."""
    import json as _json

    ao = _agent_orchestrator(request)
    app = _lythea(request)
    sid = (body.session_id or "").strip()

    # Persist the conversation so it survives a reload or a session switch
    # (agent bubbles are otherwise DOM-only and vanish). The task is stored
    # up-front; the synthesis is appended when the run ends.
    if sid:
        try:
            app.sessions.add_message(sid, Message(role="user", content=body.task))
        except Exception:  # noqa: BLE001
            log.debug("agent: user message persist failed", exc_info=True)

    async def stream():
        # Le 14B génère lentement : une seule génération (réflexion/critique/
        # synthèse) peut dépasser 100 s SANS émettre d'octet, et un proxy
        # (Cloudflare trycloudflare) coupe alors la connexion SSE. Parade : on
        # fait tourner l'agent dans une TÂCHE qui pousse ses events dans une
        # file, et on lit la file avec un court timeout pour émettre un
        # heartbeat SSE (commentaire « : ... » ignoré par le client) qui garde
        # la connexion vivante pendant les longues générations.
        #
        # FIRE-AND-FORGET : la tâche n'est PAS annulée si le client part. Elle
        # va au bout côté serveur et persiste elle-même sa synthèse (donc tu
        # peux fermer l'onglet et récupérer le résultat plus tard dans la
        # conversation ; seul le déroulé en direct est perdu).
        q: asyncio.Queue = asyncio.Queue()

        async def _producer():
            synth = ""
            try:
                atts = [{"filename": a.filename, "content": a.content}
                        for a in (body.attachments or [])]
                async for ev in ao.run(body.task, subdir=body.subdir,
                                        react=body.react, attachments=atts):
                    if ev.get("type") == "synthesis":
                        synth = ev.get("text", "") or synth
                    await q.put(("ev", ev))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("agent_run failed")
                await q.put(("err", str(exc)))
            finally:
                # Persistance ICI (pas dans le stream) : le résultat est
                # enregistré même si le client s'est déconnecté entre-temps.
                if sid and synth.strip():
                    try:
                        app.sessions.add_message(
                            sid, Message(role="assistant", content=synth))
                    except Exception:  # noqa: BLE001
                        log.debug("agent: synthesis persist failed", exc_info=True)
                await q.put(("done", None))

        task = asyncio.create_task(_producer())
        _AGENT_BG_TASKS.add(task)            # référence forte → survit au client
        task.add_done_callback(_AGENT_BG_TASKS.discard)
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"   # garde la connexion ouverte
                    continue
                if kind == "done":
                    break
                if kind == "err":
                    yield "event: run_error\n"
                    yield f"data: {_json.dumps({'error': payload})}\n\n"
                    break
                ev = payload
                yield f"event: {ev.get('type', 'message')}\n"
                yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            # FIRE-AND-FORGET : on NE tue PAS la mission si le client se
            # déconnecte. _producer continue en arrière-plan et persiste le
            # résultat. (Le bouton stop, lui, arrête bien la mission.)
            pass

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/api/agent/interject")
async def agent_interject(body: AgentInterjectRequest, request: Request) -> dict:
    """Inject a new instruction into a running agent (absorbed next step)."""
    ao = _agent_orchestrator(request)
    ok = ao.interject(body.run_id, body.text)
    if not ok:
        raise HTTPException(status_code=404, detail="Run introuvable ou texte vide.")
    return {"ok": True}


@router.post("/api/agent/stop")
async def agent_stop(body: AgentStopRequest, request: Request) -> dict:
    """Ask a running agent to stop after the current step."""
    ao = _agent_orchestrator(request)
    ok = ao.stop(body.run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Run introuvable.")
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════
# Steering — CAA activation steering on the core model (V6 BETA)
# ══════════════════════════════════════════════════════════════════════
# Computes contrastive vectors on the loaded model, auto-selects layers,
# caches per model, injects alpha*scale*v̂ via forward hooks. Affects the
# in-process core only (chat + agent share it). Style/tone axes only —
# never bypasses inhibition.

def _steering_engine(request: Request):
    """Lazily build (or fetch cached) the steering engine for this app."""
    app = _lythea(request)
    eng = getattr(app, "steering_engine", None)
    if eng is not None:
        return eng
    from rune.steering import SteeringEngine

    hippo = getattr(app, "hippocampe", None)
    if hippo is None or getattr(hippo, "model", None) is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    eng = SteeringEngine(hippo.model)
    app.steering_engine = eng
    return eng


@router.get("/api/steering")
async def steering_state(request: Request) -> dict:
    """List axes + current steering state (active axis, alpha)."""
    return _steering_engine(request).state()


@router.post("/api/steering")
async def steering_apply(body: SteeringApplyRequest, request: Request) -> dict:
    """Attach an axis at alpha; alpha==0 detaches. Lazy-calibrates if needed."""
    eng = _steering_engine(request)
    try:
        if body.alpha == 0.0:
            eng.detach()
            return eng.state()
        return eng.attach(body.axis, body.alpha)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("steering apply failed")
        raise HTTPException(status_code=500, detail=f"Steering: {exc}")


@router.post("/api/steering/mix")
async def steering_mix(body: SteeringMixRequest, request: Request) -> dict:
    """Engage several axes at once. Empty mix detaches. Lazy-calibrates each."""
    eng = _steering_engine(request)
    try:
        return eng.set_mix(body.mix)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("steering mix failed")
        raise HTTPException(status_code=500, detail=f"Steering: {exc}")


@router.post("/api/steering/calibrate")
async def steering_calibrate(body: SteeringCalibrateRequest, request: Request) -> dict:
    """Force (re)calibration of an axis on the loaded model."""
    eng = _steering_engine(request)
    try:
        result = eng.calibrate(body.axis)
        return {"ok": True, "axis": result["axis"], "layers": result["layers"]}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("steering calibrate failed")
        raise HTTPException(status_code=500, detail=f"Calibration: {exc}")
