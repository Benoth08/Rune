"""Passerelle Telegram pour Lythéa — parle à Rune depuis ton téléphone.

Deux modes :
1. INTÉGRÉ : démarre automatiquement avec le serveur web quand un token est
   configuré (``telegram_bot_token`` dans les réglages ou env
   ``LYTHEA_TELEGRAM_TOKEN``). Le navigateur n'est plus nécessaire.
2. STANDALONE (« sans le framework ») : ``python -m lythea.telegram_bot
   [--model <id>]`` boote le cœur cognitif (LytheaApp + préchargement) SANS
   uvicorn ni interface web — Telegram devient la seule interface.

Sécurité : long-polling sortant uniquement (aucun port à ouvrir — adapté à un
pod RunPod) ; allowlist de chat IDs (``telegram_allowed_chat_ids``). Tant que
l'allowlist est vide, le bot répond ton chat_id et REFUSE de traiter quoi que
ce soit — ajoute l'ID aux réglages pour appairer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request

log = logging.getLogger("lythea.telegram")

_API = "https://api.telegram.org/bot{token}/{method}"
_HISTORY_MAX = 24          # tours conservés par chat (continuité de session)
_REPLY_CHUNK = 3900        # limite Telegram ≈ 4096


def _call(token: str, method: str, http_timeout: int = 65, **params) -> dict:
    data = urllib.parse.urlencode(
        {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
         for k, v in params.items() if v is not None}).encode()
    req = urllib.request.Request(_API.format(token=token, method=method),
                                 data=data)
    with urllib.request.urlopen(req, timeout=http_timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", errors="replace"))


class TelegramBot(threading.Thread):
    """Thread de long-polling. Route le texte vers le pipeline cognitif
    complet (``hippocampe.process_message``) — même chemin que le chat web,
    donc mémoire, KG, contraintes vitales et steering s'appliquent."""

    def __init__(self, lythea_app, boot_state=None, token: str = "",
                 allowed: list | None = None) -> None:
        super().__init__(daemon=True, name="telegram-bot")
        self.app = lythea_app
        self.boot = boot_state
        s = getattr(lythea_app, "settings", None)
        self.token = (token or os.environ.get("LYTHEA_TELEGRAM_TOKEN", "")
                      or str(getattr(s, "telegram_bot_token", "") or ""))
        self.allowed = set(int(x) for x in (
            allowed if allowed is not None
            else list(getattr(s, "telegram_allowed_chat_ids", []) or [])))
        self._stop = threading.Event()
        self._hist: dict[int, list[dict]] = {}
        self._sess_ts: dict[int, float] = {}
        self._last_ts: dict[int, float] = {}

    # ── envoi ────────────────────────────────────────────────────────
    def _send(self, chat_id: int, text: str) -> None:
        text = (text or "").strip() or "…"
        for i in range(0, len(text), _REPLY_CHUNK):
            try:
                _call(self.token, "sendMessage", chat_id=chat_id,
                      text=text[i:i + _REPLY_CHUNK])
            except Exception:  # noqa: BLE001
                log.warning("telegram send failed", exc_info=True)

    def _typing(self, chat_id: int) -> None:
        try:
            _call(self.token, "sendChatAction", http_timeout=10,
                  chat_id=chat_id, action="typing")
        except Exception:  # noqa: BLE001
            pass

    # ── readiness ───────────────────────────────────────────────────
    def _ready(self) -> str:
        """'' si prêt, sinon message d'attente."""
        b = self.boot
        if b is not None and not bool(getattr(b, "ready", True)):
            return "⏳ Démarrage en cours (modèles en préchargement) — réessaie dans un instant."
        mm = getattr(self.app, "model", None)
        if mm is not None and not bool(getattr(mm, "is_loaded", False)):
            return ("⚠️ Aucun LLM chargé. Charge un modèle (UI web, ou lance le "
                    "mode standalone avec --model <id>).")
        return ""

    # ── traitement d'un message ─────────────────────────────────────
    def _handle_text(self, chat_id: int, text: str) -> None:
        wait = self._ready()
        if wait:
            self._send(chat_id, wait)
            return
        if text.startswith("/start") or text.startswith("/aide"):
            self._send(chat_id,
                       "Rune en ligne. Envoie-moi du texte (pipeline cognitif "
                       "complet), ou « /agent <tâche> » pour une mission "
                       "agentique (code + tests en sandbox).")
            return
        if text.startswith("/agent"):
            task = text[len("/agent"):].strip()
            if not task:
                self._send(chat_id, "Usage : /agent <tâche>")
                return
            self._run_agent(chat_id, task)
            return
        self._run_chat(chat_id, text)

    def _run_chat(self, chat_id: int, text: str) -> None:
        hist = self._hist.setdefault(chat_id, [])
        now = time.time()
        self._typing(chat_id)
        final, err = "", ""
        try:
            for ev in self.app.hippocampe.process_message(
                    text, list(hist), None, threading.Event(),
                    last_message_ts=self._last_ts.get(chat_id),
                    session_created_ts=self._sess_ts.setdefault(chat_id, now)):
                if not isinstance(ev, dict):
                    continue
                if ev.get("type") in ("final", "answer", "message"):
                    final = ev.get("text") or ev.get("content") or final
                elif ev.get("type") == "token":
                    final += ev.get("text", "")
                elif ev.get("type") == "error":
                    err = str(ev.get("error") or ev.get("text") or "")
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            log.exception("telegram chat failed")
        self._last_ts[chat_id] = time.time()
        if final:
            hist.append({"role": "user", "content": text})
            hist.append({"role": "assistant", "content": final})
            del hist[:-2 * _HISTORY_MAX]
            self._send(chat_id, final)
        else:
            self._send(chat_id, f"⚠️ Pas de réponse ({err or 'inconnu'}).")

    def _run_agent(self, chat_id: int, task: str) -> None:
        ao = getattr(self.app, "agent_orchestrator", None)
        if ao is None:
            self._send(chat_id, "⚠️ Orchestrateur agentique indisponible.")
            return
        self._send(chat_id, f"🛠️ Mission lancée : {task[:200]}\nJe te tiens au courant…")
        import asyncio

        def _worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            synth, ok, steps, warn = "", None, 0, 0
            try:
                async def _consume():
                    nonlocal synth, ok, steps, warn
                    async for ev in ao.run(task, react=True):
                        t = ev.get("type")
                        if t == "tool_call":
                            steps += 1
                        elif t == "agent_warning":
                            warn += 1
                        elif t == "synthesis":
                            synth = ev.get("text", synth)
                        elif t == "exec_result":
                            ok = bool(ev.get("ok"))
                        elif t == "run_done":
                            ok = bool(ev.get("ok", ok))
                loop.run_until_complete(_consume())
            except Exception as exc:  # noqa: BLE001
                synth = synth or f"erreur: {exc}"
            finally:
                loop.close()
            badge = "✅" if ok else ("⚠️" if ok is not None else "ℹ️")
            extra = f" · {warn} ⚠" if warn else ""
            self._send(chat_id,
                       f"{badge} Mission terminée ({steps} étapes{extra}).\n\n"
                       f"{synth or '(pas de synthèse)'}")

        threading.Thread(target=_worker, daemon=True,
                         name="telegram-agent").start()

    # ── boucle ──────────────────────────────────────────────────────
    def run(self) -> None:  # noqa: D102
        if not self.token:
            log.info("telegram: pas de token — passerelle inactive")
            return
        log.info("telegram: passerelle démarrée (long-polling)")
        offset = None
        while not self._stop.is_set():
            try:
                resp = _call(self.token, "getUpdates", http_timeout=65,
                             offset=offset, timeout=50)
            except Exception:  # noqa: BLE001
                time.sleep(5)
                continue
            for upd in resp.get("result", []) or []:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message") or {}
                chat_id = (msg.get("chat") or {}).get("id")
                text = (msg.get("text") or "").strip()
                if chat_id is None or not text:
                    continue
                if self.allowed and chat_id not in self.allowed:
                    self._send(chat_id, "⛔ Non appairé.")
                    continue
                if not self.allowed:
                    self._send(chat_id, (
                        "🔐 Appairage requis. Ajoute ce chat_id dans les "
                        f"réglages (telegram_allowed_chat_ids) : {chat_id}"))
                    continue
                try:
                    self._handle_text(chat_id, text)
                except Exception:  # noqa: BLE001
                    log.exception("telegram handler failed")

    def stop(self) -> None:
        self._stop.set()


def start_if_configured(lythea_app, boot_state=None) -> "TelegramBot | None":
    """Démarre la passerelle si un token est présent ET que le toggle
    ``telegram_enabled`` n'est pas à False (désactivation explicite)."""
    s = getattr(lythea_app, "settings", None)
    if not bool(getattr(s, "telegram_enabled", True)):
        log.info("telegram gateway disabled (telegram_enabled=False)")
        return None
    bot = TelegramBot(lythea_app, boot_state)
    if not bot.token:
        return None
    bot.start()
    return bot


def main() -> None:
    """Mode STANDALONE : Telegram seul, sans uvicorn ni UI web."""
    import argparse

    from rune.boot import BootRunner, BootState
    from rune.server.app import LytheaApp

    ap = argparse.ArgumentParser(description="Lythéa — passerelle Telegram standalone")
    ap.add_argument("--model", default="", help="model_id du catalogue à charger")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    app = LytheaApp()
    state = BootState()
    BootRunner(app, state).start()
    while not bool(getattr(state, "ready", False)):
        time.sleep(1)
    log.info("standalone: préchargement terminé")
    if args.model:
        log.info("standalone: chargement du modèle %s…", args.model)
        app.model.load(args.model)
    bot = TelegramBot(app, state)
    if not bot.token:
        raise SystemExit(
            "Aucun token Telegram (réglage telegram_bot_token ou env "
            "LYTHEA_TELEGRAM_TOKEN).")
    bot.start()
    log.info("standalone: passerelle active — Ctrl-C pour arrêter")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    main()
