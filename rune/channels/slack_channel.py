"""Adaptateur Slack — via slack-sdk (Socket Mode pour usage local).

Socket Mode permet de recevoir des messages sans exposer un endpoint
HTTPS public — Slack se connecte à un WebSocket sortant. Idéal pour
un pod RunPod.

Pour activer :
    1. Créer une Slack App sur api.slack.com
    2. Activer Socket Mode, récupérer le xapp token
    3. Donner le bot token (xoxb-...) avec chat:write scope
    4. Configurer :
       HERMES_LYTHEA_SLACK_APP_TOKEN=xapp:...
       HERMES_LYTHEA_SLACK_BOT_TOKEN=xoxb:...
    5. channel = SlackChannel()
    6. channel.start()
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .base import ChannelAdapter, IncomingMessage, OutgoingMessage

log = logging.getLogger("rune.channels.slack")


class SlackChannel(ChannelAdapter):
    """Adaptateur Slack via slack-sdk (Socket Mode).

    Parameters
    ----------
    app_token : str | None
        xapp token (Socket Mode). Si None, lit env.
    bot_token : str | None
        xoxb token. Si None, lit env.
    """

    def __init__(
        self,
        app_token: str | None = None,
        bot_token: str | None = None,
        message_handler: Any = None,
    ) -> None:
        super().__init__(message_handler)
        self._app_token = app_token or os.environ.get(
            "HERMES_LYTHEA_SLACK_APP_TOKEN", ""
        )
        self._bot_token = bot_token or os.environ.get(
            "HERMES_LYTHEA_SLACK_BOT_TOKEN", ""
        )
        self._socket_client: Any = None
        self._web_client: Any = None
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "slack"

    def start(self) -> None:
        if not self._app_token or not self._bot_token:
            log.warning(
                "SlackChannel: missing tokens — set "
                "HERMES_LYTHEA_SLACK_APP_TOKEN and "
                "HERMES_LYTHEA_SLACK_BOT_TOKEN."
            )
            return

        try:
            import slack_sdk
            from slack_bolt.adapter.socket_mode import SocketModeHandler
            from slack_bolt import App
        except ImportError:
            log.warning(
                "slack-sdk / slack-bolt not installed. "
                "Install with: pip install 'rune[channels]'"
            )
            return

        app = App(token=self._bot_token)

        @app.message("")
        def _handle(message: Any, say: Any) -> None:
            # Ignore les messages du bot lui-même
            if message.get("bot_id") or message.get("subtype"):
                return
            msg = IncomingMessage(
                channel="slack",
                channel_user_id=message.get("channel", ""),
                text=message.get("text", ""),
                metadata={
                    "user": message.get("user"),
                    "ts": message.get("ts"),
                    "channel": message.get("channel"),
                },
            )
            response = self._dispatch(msg)
            if response is not None:
                say(text=response.truncated_text())

        self._web_client = app.client
        self._running = True
        self._thread = threading.Thread(
            target=self._run_socket, args=(app,), daemon=True,
            name="slack-channel",
        )
        self._thread.start()
        log.info("Slack channel started")

    def _run_socket(self, app: Any) -> None:
        try:
            from slack_bolt.adapter.socket_mode import SocketModeHandler
            SocketModeHandler(app, self._app_token).start()
        except Exception:
            log.exception("Slack socket mode failed")

    def stop(self) -> None:
        self._running = False
        # SocketModeHandler n'a pas d'API stop propre — on attend juste
        # que le thread se termine (daemon).
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("Slack channel stopped")

    def send(
        self,
        channel_user_id: str,
        message: OutgoingMessage,
    ) -> bool:
        if self._web_client is None:
            log.warning("Slack not started — cannot send")
            return False
        try:
            self._web_client.chat_postMessage(
                channel=channel_user_id,
                text=message.truncated_text(),
            )
            return True
        except Exception:
            log.exception("Slack send failed")
            return False
