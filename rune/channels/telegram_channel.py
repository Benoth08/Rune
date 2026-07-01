"""Adaptateur Telegram — via python-telegram-bot.

L'adaptateur gère le polling long (getUpdates) et l'envoi de messages.
Pas de webhook (plus simple à déployer sur un pod sans HTTPS public).

Pour activer :
    1. Créer un bot via @BotFather, récupérer le token
    2. Configurer HERMES_LYTHEA_TELEGRAM_TOKEN=...
    3. channel = TelegramChannel(token=...)
    4. channel.start()  # blocking

Si python-telegram-bot n'est pas installé, l'adaptateur logge un
warning et se met en mode no-op (utile pour les tests).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .base import ChannelAdapter, IncomingMessage, OutgoingMessage

log = logging.getLogger("rune.channels.telegram")


class TelegramChannel(ChannelAdapter):
    """Adaptateur Telegram via python-telegram-bot.

    Parameters
    ----------
    token : str | None
        Bot token. Si None, lit HERMES_LYTHEA_TELEGRAM_TOKEN.
    message_handler : Callable | None
        Handler de messages entrants.
    """

    def __init__(
        self,
        token: str | None = None,
        message_handler: Any = None,
    ) -> None:
        super().__init__(message_handler)
        self._token = token or os.environ.get("HERMES_LYTHEA_TELEGRAM_TOKEN", "")
        self._application: Any = None
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "telegram"

    def start(self) -> None:
        """Démarre le polling Telegram dans un thread séparé."""
        if not self._token:
            log.warning(
                "TelegramChannel: no token — set HERMES_LYTHEA_TELEGRAM_TOKEN. "
                "Channel will not start."
            )
            return

        try:
            from telegram.ext import ApplicationBuilder, MessageHandler, filters
        except ImportError:
            log.warning(
                "python-telegram-bot not installed. "
                "Install with: pip install 'rune[channels]'"
            )
            return

        self._application = (
            ApplicationBuilder().token(self._token).build()
        )

        async def _handle(update: Any, context: Any) -> None:
            if update.message is None or update.message.text is None:
                return
            msg = IncomingMessage(
                channel="telegram",
                channel_user_id=str(update.message.chat_id),
                text=update.message.text,
                metadata={
                    "username": update.message.from_user.username if update.message.from_user else None,
                    "chat_id": update.message.chat_id,
                },
            )
            response = self._dispatch(msg)
            if response is not None:
                await context.bot.send_message(
                    chat_id=update.message.chat_id,
                    text=response.truncated_text(),
                )

        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, _handle)
        )

        self._running = True
        self._thread = threading.Thread(
            target=self._run_polling, daemon=True, name="telegram-channel"
        )
        self._thread.start()
        log.info("Telegram channel started")

    def _run_polling(self) -> None:
        """Lance le polling Telegram. Blocking."""
        try:
            self._application.run_polling()
        except Exception:
            log.exception("Telegram polling failed")

    def stop(self) -> None:
        self._running = False
        if self._application is not None:
            try:
                self._application.stop_running()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("Telegram channel stopped")

    def send(
        self,
        channel_user_id: str,
        message: OutgoingMessage,
    ) -> bool:
        """Envoie un message à un chat Telegram."""
        if self._application is None:
            log.warning("Telegram not started — cannot send")
            return False
        try:
            import asyncio
            asyncio.run(self._application.bot.send_message(
                chat_id=int(channel_user_id),
                text=message.truncated_text(),
            ))
            return True
        except Exception:
            log.exception("Telegram send failed")
            return False
