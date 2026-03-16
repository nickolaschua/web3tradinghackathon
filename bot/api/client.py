"""
RoostooClient — HMAC-signed HTTP client for the Roostoo mock exchange API.

Phase 2 (02-01): HMAC signing and all 6 endpoints.
Phase 2 (02-02): Rate limiter + exponential backoff (added on top).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse

import requests

logger = logging.getLogger(__name__)


class RoostooClient:
    """
    HTTP client for the Roostoo mock exchange API.

    Endpoints used by main.py:
      - get_balance()        → dict with 'total_usd' and per-asset balances
      - get_open_orders()    → list of open order dicts
      - get_ticker(pair)     → dict with 'LastPrice' (str)
      - place_order(...)     → dict with order result
      - pending_count()      → dict with 'TotalPending' (int)
      - cancel_order(id)     → dict with cancellation result

    Auth: HMAC SHA256 — implemented in Phase 2.
    Rate limiter: 30 calls/min sliding window — implemented in Phase 2.
    Backoff: 3 retries, 2s/4s/8s — implemented in Phase 2.
    """

    def __init__(self, api_key: str, secret: str, base_url: str = "https://mock-api.roostoo.com") -> None:
        self.api_key = api_key
        self.secret = secret
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sign(self, params: dict) -> str:
        """HMAC-SHA256 sign: sort params alphabetically, urlencode, sign with secret."""
        msg = urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Make a signed HTTP request. GET uses query string; POST uses form-encoded body."""
        if params is None:
            params = {}
        signature = self._sign(params)
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        url = self.base_url + path
        if method == "GET":
            resp = requests.get(url, params=params, headers=headers, timeout=10)
        else:
            resp = requests.post(url, data=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        """Return portfolio balances. Keys: 'total_usd', 'BTC', 'USD'."""
        raise NotImplementedError("RoostooClient.get_balance() — implement in Phase 2")

    def get_open_orders(self) -> list:
        """Return list of open order dicts."""
        raise NotImplementedError("RoostooClient.get_open_orders() — implement in Phase 2")

    def get_ticker(self, pair: str) -> dict:
        """Return ticker dict with 'LastPrice' for the given pair."""
        raise NotImplementedError("RoostooClient.get_ticker() — implement in Phase 2")

    def place_order(self, pair: str, side: str, quantity: float) -> dict:
        """Submit a market order. side must be 'BUY' or 'SELL'."""
        raise NotImplementedError("RoostooClient.place_order() — implement in Phase 2")

    def pending_count(self) -> dict:
        """Return {'TotalPending': int}. Success=false is normal for 0 pending."""
        raise NotImplementedError("RoostooClient.pending_count() — implement in Phase 2")

    def cancel_order(self, order_id: int) -> dict:
        """Cancel an order by ID."""
        raise NotImplementedError("RoostooClient.cancel_order() — implement in Phase 2")
