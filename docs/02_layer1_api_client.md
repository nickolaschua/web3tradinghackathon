# Layer 1 — API Client

## What This Layer Does

The API Client is the only part of the system that communicates with the Roostoo exchange. Every other layer that needs market data or needs to place an order goes through this layer. It handles authentication, request signing, time synchronisation, rate limiting, precision enforcement, and error handling.

Everything else in the system is only as reliable as this layer. If signing breaks, nothing works. If time sync drifts, all signed requests are rejected. If rate limiting is wrong, you get locked out. This is the highest-stakes code in the project — write it carefully, test it exhaustively, and never modify it once it works.

**This layer is deployed on EC2.** It is part of the live trading system.

---

## What This Layer Is Trying to Achieve

1. Provide a clean Python interface to every Roostoo endpoint so that all other layers can call simple methods without knowing anything about HTTP, authentication, or signing
2. Guarantee that every signed request is correctly formed and will be accepted by the server
3. Protect the system from rate limit violations that could cause lockout
4. Enforce correct decimal precision on every order to prevent silent rejections
5. Ensure the system's clock stays synchronised with the exchange server clock

---

## How It Contributes to the Bigger Picture

Every single action your bot takes — fetching prices, checking balances, placing orders, querying order status, cancelling orders — is routed through this layer. It is the adapter between your logic and the exchange. If this layer is solid, the rest of the system can be built with confidence. If it has bugs, no amount of strategy sophistication will save you.

---

## Files in This Layer

```
api/
├── client.py           Core API wrapper with all endpoints
├── rate_limiter.py     Thread-safe trade cooldown enforcement
└── exchange_info.py    Pair precision rules cache and order validation
```

---

## `api/client.py`

### Authentication: How HMAC Signing Works

The Roostoo API uses HMAC SHA256 signatures. For every signed endpoint, you must:

1. Build a parameter dictionary including a 13-digit millisecond timestamp
2. Sort the dictionary keys alphabetically
3. Join them as `key1=val1&key2=val2&...` — this is `totalParams`
4. Compute `HMAC-SHA256(secretKey, totalParams)` as a hex digest
5. Send `RST-API-KEY` and `MSG-SIGNATURE` as HTTP headers

**The most common source of bugs:** parameter sort order, URL-encoding issues, or trailing whitespace in the secret key. Test against the exact known values in the API documentation before writing any other code.

```python
# sign_test.py — run this first, before anything else
import hmac, hashlib

SECRET = "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep"
PARAMS = "pair=BNB/USD&quantity=2000&side=BUY&timestamp=1580774512000&type=MARKET"
EXPECTED = "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"

result = hmac.new(SECRET.encode('utf-8'), PARAMS.encode('utf-8'), hashlib.sha256).hexdigest()
assert result == EXPECTED, f"SIGNING BROKEN: got {result}"
print("Signing verified correctly")
```

Run this before writing any other code. If this fails, nothing else will work.

### Time Synchronisation

The server rejects any request where `abs(serverTime - requestTimestamp) > 60,000ms`. On EC2, the OS clock is typically accurate, but application-level drift can accumulate. Implement dual-layer sync:

**Layer 1 (OS):** Configure chrony on EC2 to poll Amazon's Time Sync Service at `169.254.169.123`.

**Layer 2 (Application):** On client initialisation and every 5 minutes during operation, call `/v3/serverTime` and compute an offset:

```python
def _sync_time(self):
    t_before = int(time.time() * 1000)
    resp = self.session.get(f"{self.BASE_URL}/v3/serverTime").json()
    t_after = int(time.time() * 1000)
    latency_ms = (t_after - t_before)
    self.time_offset_ms = resp["ServerTime"] - t_before - (latency_ms // 2)

def _get_timestamp(self):
    return str(int(time.time() * 1000) + self.time_offset_ms)
```

Alert via Telegram if the computed offset ever exceeds 10 seconds — that is a warning that something is wrong with the clock.

### Complete Client Implementation

```python
import requests
import time
import hmac
import hashlib
import threading
import logging

logger = logging.getLogger(__name__)

class RoostooClient:
    BASE_URL = "https://mock-api.roostoo.com"

    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.session = requests.Session()
        self.time_offset_ms = 0
        self._last_sync_time = 0
        self._sync_time()

    # ── Time sync ──────────────────────────────────────────────────────────

    def _sync_time(self):
        t_before = int(time.time() * 1000)
        resp = self.session.get(f"{self.BASE_URL}/v3/serverTime", timeout=5).json()
        t_after = int(time.time() * 1000)
        self.time_offset_ms = resp["ServerTime"] - t_before - ((t_after - t_before) // 2)
        self._last_sync_time = time.time()
        logger.debug(f"Time offset: {self.time_offset_ms}ms")

    def sync_if_stale(self, max_age_seconds: int = 300):
        if time.time() - self._last_sync_time > max_age_seconds:
            self._sync_time()

    def _get_timestamp(self) -> str:
        return str(int(time.time() * 1000) + self.time_offset_ms)

    # ── Signing ────────────────────────────────────────────────────────────

    def _sign(self, params: dict) -> tuple[dict, str]:
        """
        Returns (headers, total_params_string).
        Mutates params by adding timestamp — pass a copy if you need the original.
        """
        params["timestamp"] = self._get_timestamp()
        sorted_keys = sorted(params.keys())
        total_params = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        return headers, total_params

    # ── Public endpoints ───────────────────────────────────────────────────

    def get_server_time(self) -> dict:
        return self._get(f"{self.BASE_URL}/v3/serverTime")

    def get_exchange_info(self) -> dict:
        return self._get(f"{self.BASE_URL}/v3/exchangeInfo")

    def get_ticker(self, pair: str = None) -> dict:
        params = {"timestamp": self._get_timestamp()}
        if pair:
            params["pair"] = pair
        return self._get(f"{self.BASE_URL}/v3/ticker", params=params)

    # ── Signed GET endpoints ───────────────────────────────────────────────

    def get_balance(self) -> dict:
        headers, payload = self._sign({})
        params = dict(p.split("=", 1) for p in payload.split("&"))
        return self._get(f"{self.BASE_URL}/v3/balance", headers=headers, params=params)

    def get_pending_count(self) -> dict:
        headers, payload = self._sign({})
        params = dict(p.split("=", 1) for p in payload.split("&"))
        return self._get(f"{self.BASE_URL}/v3/pending_count", headers=headers, params=params)

    # ── Signed POST endpoints ──────────────────────────────────────────────

    def place_order(self, pair: str, side: str, quantity: float,
                    price: float = None, order_type: str = None) -> dict:
        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"
        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if price is not None:
            payload["price"] = str(price)
        headers, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._post(f"{self.BASE_URL}/v3/place_order",
                          headers=headers, data=total_params)

    def query_order(self, order_id: str = None, pair: str = None,
                    pending_only: bool = None) -> dict:
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
        headers, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._post(f"{self.BASE_URL}/v3/query_order",
                          headers=headers, data=total_params)

    def cancel_order(self, order_id: str = None, pair: str = None) -> dict:
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        headers, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._post(f"{self.BASE_URL}/v3/cancel_order",
                          headers=headers, data=total_params)

    # ── HTTP helpers with retry ────────────────────────────────────────────

    def _get(self, url, headers=None, params=None) -> dict:
        return self._request("GET", url, headers=headers, params=params)

    def _post(self, url, headers=None, data=None) -> dict:
        return self._request("POST", url, headers=headers, data=data)

    def _request(self, method, url, **kwargs) -> dict:
        for attempt in range(3):
            try:
                resp = self.session.request(method, url, timeout=10, **kwargs)
                resp.raise_for_status()
                result = resp.json()
                # CRITICAL: always check Success field, never trust HTTP 200 alone
                if "Success" in result and not result["Success"]:
                    logger.warning(f"API returned Success=false: {result.get('ErrMsg')} | URL: {url}")
                return result
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                logger.error(f"Request failed (attempt {attempt+1}/3): {e}. Retrying in {wait}s")
                if attempt < 2:
                    time.sleep(wait)
        logger.critical(f"All retries exhausted for {url}")
        return {"Success": False, "ErrMsg": "All retries exhausted"}
```

---

## `api/rate_limiter.py`

The rate limiter enforces the 65-second minimum between trade submissions. This is separate from the polling rate — you can call `get_ticker()` freely. Only `place_order()` is gated.

```python
import time
import threading
import logging

logger = logging.getLogger(__name__)

class TradingRateLimiter:
    """
    Enforces minimum 65 seconds between place_order calls.
    Thread-safe for multi-threaded environments.
    On rate-limit rejection: hard 120-second sleep before allowing retry.
    """

    TRADE_COOLDOWN_SECONDS = 65

    def __init__(self):
        self._lock = threading.Lock()
        self._last_trade_time = 0.0

    def wait_for_trade_slot(self):
        with self._lock:
            elapsed = time.time() - self._last_trade_time
            if elapsed < self.TRADE_COOLDOWN_SECONDS:
                wait = self.TRADE_COOLDOWN_SECONDS - elapsed
                logger.debug(f"Rate limiter: waiting {wait:.1f}s before next trade")
                time.sleep(wait)

    def record_trade_submitted(self):
        with self._lock:
            self._last_trade_time = time.time()

    def handle_rate_limit_rejection(self):
        """Call this when the server returns a rate-limit error. Hard sleep."""
        logger.warning("Rate limit rejection received. Sleeping 120 seconds.")
        time.sleep(120)

    def seconds_until_next_trade(self) -> float:
        elapsed = time.time() - self._last_trade_time
        return max(0.0, self.TRADE_COOLDOWN_SECONDS - elapsed)
```

Usage pattern in the order manager:
```python
rate_limiter.wait_for_trade_slot()
result = client.place_order(...)
rate_limiter.record_trade_submitted()
if not result.get("Success"):
    if "rate" in result.get("ErrMsg", "").lower():
        rate_limiter.handle_rate_limit_rejection()
```

---

## `api/exchange_info.py`

Exchange pairs have different precision rules. Getting precision wrong causes silent order rejection — the server returns `Success: false` and you think you're in a position you're not.

```python
import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

@dataclass
class PairInfo:
    coin: str
    price_precision: int
    amount_precision: int
    mini_order: float
    can_trade: bool

class ExchangeInfoCache:
    """
    Loads pair precision rules from /v3/exchangeInfo on startup.
    Provides order validation and precision rounding.
    """

    def __init__(self, client):
        self._pairs: Dict[str, PairInfo] = {}
        self._load(client)

    def _load(self, client):
        info = client.get_exchange_info()
        for pair, data in info.get("TradePairs", {}).items():
            self._pairs[pair] = PairInfo(
                coin=data["Coin"],
                price_precision=data["PricePrecision"],
                amount_precision=data["AmountPrecision"],
                mini_order=data["MiniOrder"],
                can_trade=data["CanTrade"],
            )
        logger.info(f"Loaded exchange info for {len(self._pairs)} pairs")

    def get_tradeable_pairs(self) -> list[str]:
        return [p for p, info in self._pairs.items() if info.can_trade]

    def round_quantity(self, pair: str, quantity: float) -> float:
        info = self._pairs.get(pair)
        if not info:
            raise ValueError(f"Unknown pair: {pair}")
        return round(quantity, info.amount_precision)

    def round_price(self, pair: str, price: float) -> float:
        info = self._pairs.get(pair)
        if not info:
            raise ValueError(f"Unknown pair: {pair}")
        return round(price, info.price_precision)

    def validate_order(self, pair: str, price: float, quantity: float) -> tuple[bool, str]:
        """Returns (is_valid, error_message)."""
        info = self._pairs.get(pair)
        if not info:
            return False, f"Unknown pair: {pair}"
        if not info.can_trade:
            return False, f"Pair {pair} is not tradeable"
        order_value = price * quantity
        if order_value < info.mini_order:
            return False, f"Order value {order_value:.4f} below MiniOrder {info.mini_order}"
        return True, ""
```

---

## Testing This Layer

Before moving to any other layer, verify the following:

1. **Signing test** — run `sign_test.py` against the known hash from the API docs. Must pass.
2. **Server time** — call `get_server_time()`, confirm you receive a valid timestamp.
3. **Exchange info** — call `get_exchange_info()`, confirm you see BTC/USD, ETH/USD, and other pairs.
4. **Ticker** — call `get_ticker("BTC/USD")`, confirm you receive live price data.
5. **Balance** — call `get_balance()`, confirm you see the $50,000 starting balance.
6. **Place order** — place a minimum-size MARKET BUY order. Confirm `Success: true` and `Status: FILLED`.
7. **Query order** — call `query_order(pair="BTC/USD")`, confirm the order appears.
8. **Cancel test** — place a LIMIT order far from market price, then cancel it. Confirm cancellation.
9. **Rate limiter test** — attempt two orders in rapid succession. The second should block for ~65 seconds.
10. **Precision test** — attempt an order with 8 decimal places on a pair with AmountPrecision=2. Confirm it is rounded before submission.

Do not proceed to Layer 2 until all 10 tests pass.

---

## Failure Modes This Layer Prevents

| Failure | How This Layer Prevents It |
|---|---|
| HMAC Signing Bug | sign_test.py validates against known output before anything else |
| Timestamp Drift Rejection | Dual-layer sync: chrony on OS, _sync_time() in application |
| Rate Limit Cascade | 65s cooldown; 120s hard sleep on rejection; no immediate retry |
| Precision/Quantity Rejection | ExchangeInfoCache rounds all orders before submission |
| Silent Order Failure | Every response checked for Success field, not just HTTP status |
| All Retries Exhausted | Returns structured error dict; never raises exception to caller |
