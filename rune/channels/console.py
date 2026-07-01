"""Console channel — lit stdin, écrit stdout. Pour le CLI et les tests.

C'est l'adaptateur le plus simple — pas de polling réseau, pas
d'authentification. Juste input() / print(). Idéal pour le mode
``rune chat`` interactif.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Callable

from .base import ChannelAdapter, IncomingMessage, OutgoingMessage

log = logging.getLogger("rune.channels.console")


class ConsoleChannel(ChannelAdapter):
    """Adaptateur console — readline sur stdin.

    Usage :

        channel = ConsoleChannel(handler=my_handler)
        channel.start()  # blocking jusqu'à EOF ou /quit
    """

    @property
    def name(self) -> str:
        return "console"

    def start(self) -> None:
        """Boucle principale — blocking. Sort sur EOF ou '/quit'."""
        print("Rune — mode console (tape /quit pour sortir)")
        print()
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAu revoir.")
                break
            if not line:
                continue
            if line.lower() in {"/quit", "/exit", "/q"}:
                print("Au revoir.")
                break
            msg = IncomingMessage(
                channel="console",
                channel_user_id="console_user",
                text=line,
            )
            response = self._dispatch(msg)
            if response is not None:
                print(f"rune> {response.truncated_text()}")
                print()

    def stop(self) -> None:
        # Rien à faire — start() est blocking et sort sur /quit
        pass

    def send(
        self,
        channel_user_id: str,
        message: OutgoingMessage,
    ) -> bool:
        """Envoie un message — juste un print."""
        print(message.truncated_text())
        return True
