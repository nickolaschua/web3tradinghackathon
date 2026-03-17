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

from bot.api.rate_limiter import _rate_limiter, _trade_cooldown

logger = logging.getLogger(__name__)

BACKOFF_DELAYS = [2, 4, 8]  # seconds; 3 retries = 4 total attempts (FAQ Q19)


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
        """Signed HTTP request with rate limiting and exponential backoff."""
        if params is None:
            params = {}
        params["timestamp"] = str(int(time.time() * 1000))
        signature = self._sign(params)
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        url = self.base_url + path
        last_exc: Exception | None = None
        for attempt in range(len(BACKOFF_DELAYS) + 1):  # attempts 0, 1, 2, 3
            if attempt > 0:
                delay = BACKOFF_DELAYS[attempt - 1]
                logger.warning("retry %d/3 for %s %s in %ds", attempt, method, path, delay)
                time.sleep(delay)
            _rate_limiter.acquire()
            try:
                if method == "GET":
                    resp = requests.get(url, params=params, headers=headers, timeout=10)
                else:
                    resp = requests.post(url, data=params, headers=headers, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                logger.warning("request %s %s failed (attempt %d/4): %s", method, path, attempt + 1, exc)
        raise RuntimeError(f"request {method} {path} failed after 4 attempts") from last_exc

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_ticker(self, pair: str) -> dict:
        """Return ticker dict with 'LastPrice' for the given pair."""
        return self._request("GET", "/v3/ticker", {"pair": pair})

    def get_balance(self) -> dict:
        """Return portfolio balances. Keys: 'total_usd', 'BTC', 'USD'."""
        return self._request("GET", "/v3/balance", {})

    def place_order(self, pair: str, side: str, quantity: float) -> dict:
        """Submit a market order. Enforces 65s trade cooldown before dispatching."""
        _trade_cooldown.acquire()
        return self._request("POST", "/v3/place_order", {
            "pair": pair,
            "side": side,
            "quantity": str(quantity),
            "type": "MARKET",
        })

    def pending_count(self) -> dict:
        """Return raw response containing 'TotalPending'.

        Success=false is the normal response for zero pending orders; check TotalPending only.
        """
        return self._request("GET", "/v3/pending_count", {})

    def cancel_order(self, order_id: int) -> dict:
        """Cancel an order by ID."""
        return self._request("POST", "/v3/cancel_order", {"order_id": str(order_id)})

    def get_open_orders(self) -> list:
        """Return list of open order dicts.

        pending_only MUST be string 'TRUE' not Python bool True.
        """
        resp = self._request("GET", "/v3/order", {"pending_only": "TRUE"})
        return resp.get("Data", [])
