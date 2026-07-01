"""Adaptateur de canal abstrait — interface commune à tous les canaux.

Format unifié
-------------
Tout message entrant (depuis n'importe quel canal) est converti en
:class:`IncomingMessage`. Tout message sortant est un
:class:`OutgoingMessage` que l'adaptateur traduit vers le format
spécifique du canal.

On garde délibérément le format minimal — pas de boutons, pas de
carrousels, pas de médias. Rune est headless et text-first.
Si on veut du riche plus tard, on étendra OutgoingMessage.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("rune.channels.base")


@dataclass
class IncomingMessage:
    """Message normalisé entrant depuis un canal."""
    channel: str  # "telegram" | "slack" | "console" | ...
    channel_user_id: str  # ID utilisateur côté canal
    text: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Pour les canaux qui supportent les images (rare en headless)
    image_url: str | None = None


@dataclass
class OutgoingMessage:
    """Message normalisé sortant vers un canal."""
    text: str
    # Pour les réponses longues, on peut tronquer
    max_chars: int = 4000
    metadata: dict[str, Any] = field(default_factory=dict)

    def truncated_text(self) -> str:
        if len(self.text) <= self.max_chars:
            return self.text
        return self.text[: self.max_chars - 1] + "…"


class ChannelAdapter(ABC):
    """Adaptateur de canal — interface commune.

    Tout adaptateur implémente :
    - ``start()`` : démarre la boucle de réception (polling ou webhook)
    - ``stop()`` : arrête proprement
    - ``send()`` : envoie un message sortant vers un utilisateur
    - ``name`` : nom du canal (pour logs et routing)
    """

    def __init__(
        self,
        message_handler: Callable[[IncomingMessage], OutgoingMessage] | None = None,
    ) -> None:
        self._message_handler = message_handler

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def start(self) -> None:
        """Démarre l'adaptateur. Blocking ou lance un thread."""

    @abstractmethod
    def stop(self) -> None:
        """Arrête l'adaptateur."""

    @abstractmethod
    def send(
        self,
        channel_user_id: str,
        message: OutgoingMessage,
    ) -> bool:
        """Envoie un message à un utilisateur. Retourne True si succès."""

    def set_handler(
        self,
        handler: Callable[[IncomingMessage], OutgoingMessage],
    ) -> None:
        """Branche le handler de messages entrants."""
        self._message_handler = handler

    def _dispatch(self, message: IncomingMessage) -> OutgoingMessage | None:
        """Passe le message au handler. Retourne la réponse ou None."""
        if self._message_handler is None:
            log.warning(
                "No handler set for channel %s — dropping message", self.name
            )
            return None
        try:
            return self._message_handler(message)
        except Exception:
            log.exception("Handler failed for channel %s", self.name)
            return None
