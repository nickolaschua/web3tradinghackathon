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
          total_usd      – float, USD cash only (Free + Lock)
          holdings       – dict of non-USD assets: {asset: total_qty} (e.g. {"BTC": 0.001})
          USD            – {"Free": float, "Lock": float}
          Data           – {asset: {"Free", "Lock", "Total"}} for order_manager reconcile
          Success        – bool
        """
        resp = self._request("GET", "/v3/balance", {})
        wallet = resp.get("SpotWallet") or resp.get("Wallet") or {}
        usd = wallet.get("USD", {})
        usd_free = float(usd.get("Free", 0))
        usd_lock = float(usd.get("Lock", 0))

        # Build Data dict and holdings for all assets in the wallet
        data: dict = {
            "USD": {"Free": usd_free, "Lock": usd_lock, "Total": str(usd_free + usd_lock)},
        }
        holdings: dict[str, float] = {}
        for asset, balances in wallet.items():
            if asset == "USD":
                continue
            a_free = float(balances.get("Free", 0))
            a_lock = float(balances.get("Lock", 0))
            total = a_free + a_lock
            data[asset] = {"Free": a_free, "Lock": a_lock, "Total": str(total)}
            if total > 0:
                holdings[asset] = total

        return {
            "Success": resp.get("Success", False),
            "ErrMsg": resp.get("ErrMsg", ""),
            "total_usd": usd_free + usd_lock,
            "holdings": holdings,
            "USD": usd,
            "Data": data,
        }

    def place_order(
        self, pair: str, side: str, quantity: float, skip_cooldown: bool = False,
    ) -> dict:
        """Submit a MARKET order.

        Args:
            skip_cooldown: If True, bypass the 65s trade cooldown. Used for
                           stop-loss exits where speed is critical — a 65s delay
                           while price is moving against you can be costly.
                           The 30-call/min rate limiter is still enforced.
        """
        if not skip_cooldown:
            _trade_cooldown.acquire()
        return self._request("POST", "/v3/place_order", {
            "pair": pair,
            "side": side,
            "quantity": str(quantity),
            "type": "MARKET",
        })

    def place_limit_with_fallback(
        self, pair: str, side: str, quantity: float, price: float,
        poll_secs: float = 5.0,
    ) -> dict:
        """Place a LIMIT order for maker rate; fall back to MARKET if not filled.

        Acquires the 65s trade cooldown ONCE for the entire sequence:
          1. Place LIMIT at `price`
          2. Wait `poll_secs`, then check status
          3. If FILLED → return (saved ~50% on commission)
          4. If still PENDING → cancel, place MARKET fallback

        Returns the same response shape as place_order() regardless of path taken.
        """
        _trade_cooldown.acquire()

        # Step 1: Try LIMIT
        resp = self._request("POST", "/v3/place_order", {
            "pair": pair,
            "side": side,
            "quantity": str(quantity),
            "type": "LIMIT",
            "price": str(round(price, 2)),
        })

        if not resp.get("Success"):
            logger.warning("LIMIT order rejected for %s — immediate MARKET fallback", pair)
            return self._request("POST", "/v3/place_order", {
                "pair": pair,
                "side": side,
                "quantity": str(quantity),
                "type": "MARKET",
            })

        detail = resp.get("OrderDetail", {})
        order_id = detail.get("OrderID")

        # Rarely, a LIMIT can fill synchronously
        if detail.get("Status") == "FILLED":
            logger.info("LIMIT %s %s filled immediately (maker rate)", side, pair)
            return resp

        # Step 2: Wait and poll
        if order_id is None:
            logger.warning("LIMIT response missing OrderID — MARKET fallback")
            return self._request("POST", "/v3/place_order", {
                "pair": pair,
                "side": side,
                "quantity": str(quantity),
                "type": "MARKET",
            })

        time.sleep(poll_secs)

        try:
            query_resp = self._request(
                "POST", "/v3/query_order", {"order_id": str(order_id)}
            )
            matched = query_resp.get("OrderMatched", [])
            if matched:
                order_info = matched[0] if isinstance(matched, list) else matched
                inner = order_info.get("OrderDetail", order_info)
                if inner.get("Status") == "FILLED":
                    logger.info(
                        "LIMIT %s %s filled after %.0fs poll (maker rate)", side, pair, poll_secs
                    )
                    return {"Success": True, "OrderDetail": inner, "OrderId": order_id}
        except Exception as exc:
            logger.warning("LIMIT poll failed for %s: %s", pair, exc)

        # Step 3: Cancel pending LIMIT and fall back to MARKET
        try:
            self._request("POST", "/v3/cancel_order", {"order_id": str(order_id)})
            logger.info("Cancelled pending LIMIT %s — MARKET fallback", order_id)
        except Exception as exc:
            logger.warning("LIMIT cancel failed for order %s: %s", order_id, exc)

        logger.info("LIMIT->MARKET fallback: %s %s qty=%.6f", side, pair, quantity)
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
