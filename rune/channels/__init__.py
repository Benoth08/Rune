"""Couche channels — adaptateurs omnicanal.

Inspiré d'OpenClaw (routage omnicanal vers 50+ canaux). On fournit
une abstraction :class:`ChannelAdapter` et deux implémentations
concrètes : Telegram et Slack.

Chaque adaptateur :
1. Reçoit un message depuis le canal (webhook ou polling)
2. Le normalise au format :class:`IncomingMessage`
3. Le passe au CognitiveLoop
4. Renvoie la réponse au canal

Tout est pluggable — ajouter un canal = implémenter ChannelAdapter.
"""
from __future__ import annotations

from .base import ChannelAdapter, IncomingMessage, OutgoingMessage
from .console import ConsoleChannel
from .telegram_channel import TelegramChannel
from .slack_channel import SlackChannel

__all__ = [
    "ChannelAdapter",
    "IncomingMessage",
    "OutgoingMessage",
    "ConsoleChannel",
    "TelegramChannel",
    "SlackChannel",
]
