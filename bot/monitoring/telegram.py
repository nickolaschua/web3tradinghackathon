"""
TelegramAlerter — interface stub.

Full implementation delivered in Phase 3 (03-01-PLAN.md).
CRITICAL contract: send() must NEVER raise — all calls must be wrapped in try/except.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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

    def send(self, message: str) -> None:
        """
        Send a message to the Telegram channel. Never raises.

        Phase 3 implementation wraps in try/except and logs failures.
        This stub logs the message at INFO level so startup output is visible.
        """
        try:
            logger.info("[Telegram stub] %s", message)
        except Exception:
            pass  # never raise
