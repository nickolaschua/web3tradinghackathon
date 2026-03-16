"""
TelegramAlerter — real HTTP implementation for Telegram Bot API.

Contract: send() is fire-and-forget. If the Telegram API is down or
credentials are wrong, the bot continues unaffected. Never let alerting
kill the trading loop.
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT_SECONDS = 5
LEVEL_EMOJIS = {
    "INFO": "ℹ️",
    "WARN": "⚠️",
    "ERROR": "🔴",
    "CRITICAL": "🚨",
}


class TelegramAlerter:
    """
    Sends Telegram messages to the ops channel.

    Contract: send() is fire-and-forget. If the Telegram API is down or
    credentials are wrong, the bot continues unaffected. Never let alerting
    kill the trading loop.
    """

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id
        self._enabled = bool(token and chat_id)
        if not self._enabled:
            logger.warning("TelegramAlerter disabled: token or chat_id missing")

    def send(self, message: str, level: str = "INFO") -> bool:
        """
        Send a message to the Telegram channel. Never raises.

        Args:
            message: The message text to send
            level: The alert level (INFO, WARN, ERROR, CRITICAL) — defaults to INFO

        Returns:
            True if message was sent successfully, False otherwise.
        """
        try:
            if not self._enabled:
                logger.debug("[Telegram disabled] %s", message)
                return False

            emoji = LEVEL_EMOJIS.get(level, "")
            text = f"{emoji} {message}" if emoji else message

            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }

            response = requests.post(
                API_URL.format(token=self.token),
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return True
        except Exception:
            logger.debug("TelegramAlerter.send() failed for message: %s", message)
            return False
