"""Unified logging configuration for Lythéa."""
from __future__ import annotations

import logging
import os


# Endpoints called repeatedly by polling clients (launch.sh boot watcher,
# UI status refresh, etc.). Their access lines flood the logs without
# adding signal. We silently filter them out from uvicorn.access.
_POLLING_ENDPOINTS = (
    "/api/boot/status",
    "/api/health",
    "/api/config/v4",
)


class _PollingAccessFilter(logging.Filter):
    """Drop uvicorn.access lines that hit any polling endpoint.

    The filter inspects both ``record.args`` (uvicorn's structured form
    where the request path appears as args[2]) and the rendered message
    string (fallback for custom formatters or when args isn't a tuple).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 1) Structured form: uvicorn passes (client, method, path, ...,
        #    status). Match on args[2].
        try:
            args = record.args
            if isinstance(args, tuple) and len(args) >= 3:
                path = args[2]
                if isinstance(path, str):
                    for ep in _POLLING_ENDPOINTS:
                        if ep in path:
                            return False
        except Exception:
            pass

        # 2) Fallback: render the full message and match substring.
        try:
            msg = record.getMessage()
            for ep in _POLLING_ENDPOINTS:
                if ep in msg:
                    return False
        except Exception:
            pass

        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger and silence noisy third-party modules.

    Parameters
    ----------
    level : str
        Default log level, overridable via ``LYTHEA_LOG_LEVEL`` env var.
    """
    effective = os.environ.get("LYTHEA_LOG_LEVEL", level).upper()

    logging.basicConfig(
        level=getattr(logging, effective, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Silence noisy third-party at WARNING level
    noisy = (
        "httpx", "httpcore", "chromadb", "transformers",
        "urllib3", "filelock", "fsspec",
        "accelerate", "sentence_transformers",
    )
    for mod in noisy:
        logging.getLogger(mod).setLevel(logging.WARNING)

    # Fully silence HF hub warnings (auth, deprecation, etc.)
    hf_silent = ("huggingface_hub", "huggingface_hub.utils._http",
                 "huggingface_hub.utils._validators")
    for mod in hf_silent:
        logging.getLogger(mod).setLevel(logging.ERROR)

    # Attach polling filter to uvicorn.access so the boot watcher and
    # UI refresh don't flood the console.
    uv_access = logging.getLogger("uvicorn.access")
    # Avoid attaching the filter twice on hot-reload.
    if not any(isinstance(f, _PollingAccessFilter) for f in uv_access.filters):
        uv_access.addFilter(_PollingAccessFilter())


def build_uvicorn_log_config() -> dict:
    """Return uvicorn's log_config dict with the polling filter wired in.

    uvicorn replaces the root logging config when started without a
    custom log_config. We pass this dict to ``uvicorn.run(log_config=...)``
    so our filter survives uvicorn's own setup.

    Structure mirrors uvicorn.config.LOGGING_CONFIG but adds:
    - ``filters.polling`` referencing our ``_PollingAccessFilter``
    - ``loggers.uvicorn.access.filters`` listing ``polling``
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "polling": {
                "()": "lythea.logging_setup._PollingAccessFilter",
            },
        },
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                "datefmt": "%H:%M:%S",
            },
            "access": {
                # uvicorn fournit record.args = (client, method, path,
                # http_ver, status). Le format doit utiliser %(message)s
                # qui interpolera args via le format msg standard
                # ('%s - "%s %s HTTP/%s" %d').
                "format": "%(asctime)s [%(levelname)s] %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["polling"],
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
