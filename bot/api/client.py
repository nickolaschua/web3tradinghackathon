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
        self._time_offset_ms: int = 0  # corrected by sync_time() on startup

    # ------------------------------------------------------------------
    # Time sync
    # ------------------------------------------------------------------

    def sync_time(self) -> int:
        """Sync local clock against server time. Call once on startup.

        Computes _time_offset_ms = server_time_ms - local_time_ms and stores it.
        All subsequent timestamp generation adds this offset, keeping signatures
        within the 60 000 ms acceptance window even on clock-skewed hosts.

        Returns the measured offset in milliseconds (negative = local clock ahead).
        """
        local_ms = int(time.time() * 1000)
        try:
            resp = requests.get(
                self.base_url + "/v3/serverTime",
                timeout=10,
            )
            resp.raise_for_status()
            server_ms = int(resp.json().get("serverTime", local_ms))
            self._time_offset_ms = server_ms - local_ms
            logger.info(
                "sync_time: server=%d local=%d offset=%+d ms",
                server_ms, local_ms, self._time_offset_ms,
            )
        except Exception as exc:
            logger.warning("sync_time failed — using zero offset: %s", exc)
            self._time_offset_ms = 0
        return self._time_offset_ms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sign(self, params: dict) -> tuple[str, str]:
        """HMAC-SHA256 sign: sort params alphabetically, join without URL-encoding, sign.

        Returns (signature_hex, total_params_string).
        total_params is the exact string sent as the POST body (or derived from for GETs).
        Values are NOT URL-encoded — the server expects the raw f-string join, matching
        the reference implementation in docs/13_roostoo_api_reference.md.
        """
        sorted_keys = sorted(params.keys())
        total_params = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        sig = hmac.new(
            self.secret.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig, total_params

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Signed HTTP request with rate limiting and exponential backoff."""
        if params is None:
            params = {}
        params["timestamp"] = str(int(time.time() * 1000) + self._time_offset_ms)
        signature, total_params = self._sign(params)
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
                    # POST body MUST be the exact sorted string that was signed.
                    # Passing a dict would encode in insertion order and break signature
                    # verification. Content-Type must be set explicitly when body is a string.
                    post_headers = {**headers, "Content-Type": "application/x-www-form-urlencoded"}
                    resp = requests.post(url, data=total_params, headers=post_headers, timeout=10)
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
        """Return normalised portfolio balances.

        Normalises the API response (SpotWallet or Wallet key) into a flat dict:
          total_usd  – float, USD Free + Lock
          USD        – {"Free": float, "Lock": float}
          BTC        – {"Free": float, "Lock": float}
          Data       – {asset: {"Free", "Lock", "Total"}} for order_manager reconcile
          Success    – bool
        """
        resp = self._request("GET", "/v3/balance", {})
        wallet = resp.get("SpotWallet") or resp.get("Wallet") or {}
        usd = wallet.get("USD", {})
        btc = wallet.get("BTC", {})
        usd_free = float(usd.get("Free", 0))
        usd_lock = float(usd.get("Lock", 0))
        btc_free = float(btc.get("Free", 0))
        btc_lock = float(btc.get("Lock", 0))
        return {
            "Success": resp.get("Success", False),
            "ErrMsg": resp.get("ErrMsg", ""),
            "total_usd": usd_free + usd_lock,
            "USD": usd,
            "BTC": btc,
            "Data": {
                "USD": {"Free": usd_free, "Lock": usd_lock, "Total": str(usd_free + usd_lock)},
                "BTC": {"Free": btc_free, "Lock": btc_lock, "Total": str(btc_free + btc_lock)},
            },
        }

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
        resp = self._request("POST", "/v3/query_order", {"pending_only": "TRUE"})
        return resp.get("OrderMatched", [])
