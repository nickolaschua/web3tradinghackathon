# Roostoo Mock Exchange API Reference

## Overview

- **Base URL:** `https://mock-api.roostoo.com`
- **Starting Capital:** $50,000 USD (virtual)
- **Protocol:** REST over HTTPS
- **Total Endpoints:** 8 (no WebSocket, no OHLCV/candlestick endpoint)
- **Only tradeable pair in competition:** `BTC/USD`

---

## Authentication Security Levels

| Level | What It Requires |
|---|---|
| `RCL_NoVerification` | Nothing — open endpoint |
| `RCL_TSCheck` | `timestamp` parameter only |
| `RCL_TopLevelCheck` | `RST-API-KEY` + `MSG-SIGNATURE` headers + `timestamp` param |

---

## Signing (HMAC SHA256)

All `RCL_TopLevelCheck` endpoints require signed headers.

### Signing Process

1. Build your parameter dict (include `timestamp` as 13-digit ms integer)
2. Sort keys alphabetically
3. Join as `key=value&key=value` — this is `totalParams`
4. Compute: `hmac.new(secret.encode(), total_params.encode(), sha256).hexdigest()`
5. Send headers: `RST-API-KEY` and `MSG-SIGNATURE`

### Required Headers

```
RST-API-KEY: <your API key>
MSG-SIGNATURE: <hex digest>
```

For POST requests, also add:
```
Content-Type: application/x-www-form-urlencoded
```

### Signing Example

Parameters: `pair=BNB/USD`, `quantity=2000`, `side=BUY`, `type=MARKET`, `timestamp=1580774512000`

```
totalParams = "pair=BNB/USD&quantity=2000&side=BUY&timestamp=1580774512000&type=MARKET"
signature   = "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"
```

### Reference Implementation

```python
import hmac
import hashlib
import time

def get_signed_headers(payload: dict, secret: str, api_key: str):
    payload['timestamp'] = str(int(time.time() * 1000))
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)
    signature = hmac.new(
        secret.encode('utf-8'),
        total_params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    headers = {
        'RST-API-KEY': api_key,
        'MSG-SIGNATURE': signature,
    }
    return headers, total_params
```

---

## Timing Security

The server rejects requests where the timestamp deviates by more than 60 seconds:

```
abs(serverTime - timestamp) > 60000ms → REJECTED
```

- Always use a 13-digit millisecond timestamp: `int(time.time() * 1000)`
- Use `GET /v3/serverTime` to detect clock drift and apply an offset
- The Layer 1 client already implements this — do not bypass it

---

## Endpoints

### 1. Check Server Time

```
GET /v3/serverTime
Auth: RCL_NoVerification
```

No parameters.

**Response:**
```json
{
  "ServerTime": 1570083944052
}
```

**Use:** Compute clock offset. Store `offset = server_time - local_time` and add to all subsequent timestamps.

---

### 2. Exchange Information

```
GET /v3/exchangeInfo
Auth: RCL_NoVerification
```

No parameters.

**Response:**
```json
{
  "IsRunning": true,
  "InitialWallet": {
    "USD": 50000
  },
  "TradePairs": {
    "BTC/USD": {
      "Coin": "BTC",
      "CoinFullName": "Bitcoin",
      "Unit": "USD",
      "UnitFullName": "US Dollar",
      "CanTrade": true,
      "PricePrecision": 2,
      "AmountPrecision": 6,
      "MiniOrder": 1.0
    }
  }
}
```

**Key fields:**
- `PricePrecision`: decimal places for price (BTC/USD = 2, i.e. $0.01 steps)
- `AmountPrecision`: decimal places for quantity (BTC/USD = 6, i.e. 0.000001 BTC steps)
- `MiniOrder`: minimum order value in USD — **order is rejected if `price × quantity ≤ 1.0`**
- `CanTrade`: must be `true` before placing orders

---

### 3. Market Ticker

```
GET /v3/ticker
Auth: RCL_TSCheck
```

| Parameter | Type | Mandatory | Description |
|---|---|---|---|
| `timestamp` | STRING | YES | 13-digit ms timestamp |
| `pair` | STRING | NO | e.g. `BTC/USD`. Omit for all pairs |

**Response (single pair):**
```json
{
  "Success": true,
  "ErrMsg": "",
  "ServerTime": 1580762734517,
  "Data": {
    "BTC/USD": {
      "MaxBid": 9318.45,
      "MinAsk": 9319.42,
      "LastPrice": 9319.35,
      "Change": -0.0132,
      "CoinTradeValue": 53001.931315,
      "UnitTradeValue": 496450629.05850565
    }
  }
}
```

**Key fields:**
- `LastPrice`: current mid-market price — use this for all sizing and signal calculations
- `Change`: 24-hour price change as a decimal (e.g. `-0.0132` = -1.32%)
- `CoinTradeValue`: 24-hour trading volume in BTC (coin units)
- `UnitTradeValue`: 24-hour trading volume in USD — usable as a volume proxy for OBV when no per-bar volume exists

**Notes:**
- This is the only price source. There is **no OHLCV/candlestick endpoint**.
- All ATR, ADX, and OBV computed in live trading use synthetic bars (H=L=C=O=LastPrice, volume=CoinTradeValue delta or 0).

---

### 4. Balance Information

```
GET /v3/balance
Auth: RCL_TopLevelCheck
```

| Parameter | Type | Mandatory |
|---|---|---|
| `timestamp` | STRING | YES |

**Response:**
```json
{
  "Success": true,
  "ErrMsg": "",
  "Wallet": {
    "BTC": {
      "Free": 0.454878,
      "Lock": 0.555
    },
    "USD": {
      "Free": 98389854.15,
      "Lock": 1601798.20
    }
  }
}
```

**Key fields:**
- `Free`: available for new orders
- `Lock`: reserved for pending LIMIT orders
- Total portfolio value = `USD.Free + USD.Lock + (BTC.Free + BTC.Lock) × current_price`

---

### 5. Pending Order Count

```
GET /v3/pending_count
Auth: RCL_TopLevelCheck
```

| Parameter | Type | Mandatory |
|---|---|---|
| `timestamp` | STRING | YES |

**Response (orders exist):**
```json
{
  "Success": true,
  "ErrMsg": "",
  "TotalPending": 3,
  "OrderPairs": {
    "BTC/USD": 2,
    "BAT/USD": 1
  }
}
```

**Response (no pending orders):**
```json
{
  "Success": false,
  "ErrMsg": "no pending order under this account",
  "TotalPending": 0,
  "OrderPairs": {}
}
```

> **CRITICAL GOTCHA:** `Success: false` with this specific `ErrMsg` is **NOT an error** — it means zero pending orders. Do not treat this as a failure. Check `TotalPending` instead of `Success` for this endpoint.

---

### 6. Place Order

```
POST /v3/place_order
Auth: RCL_TopLevelCheck
Content-Type: application/x-www-form-urlencoded
```

| Parameter | Type | Mandatory | Description |
|---|---|---|---|
| `pair` | STRING | YES | e.g. `BTC/USD` |
| `side` | STRING | YES | `BUY` or `SELL` |
| `type` | STRING | YES | `MARKET` or `LIMIT` |
| `quantity` | STRING | YES | Coin amount (not USD) |
| `timestamp` | STRING | YES | 13-digit ms timestamp |
| `price` | DECIMAL | Conditional | Required for LIMIT orders |

**Response (MARKET order — fills synchronously):**
```json
{
  "Success": true,
  "ErrMsg": "",
  "OrderDetail": {
    "Pair": "BTC/USD",
    "OrderID": 81,
    "Status": "FILLED",
    "Role": "TAKER",
    "CreateTimestamp": 1570224271550,
    "FinishTimestamp": 1570224271590,
    "Side": "BUY",
    "Type": "MARKET",
    "Price": 68420.50,
    "Quantity": 0.052341,
    "FilledQuantity": 0.052341,
    "FilledAverPrice": 68420.50,
    "CoinChange": 0.052341,
    "UnitChange": 3581.04,
    "CommissionCoin": "USD",
    "CommissionChargeValue": 0.4297,
    "CommissionPercent": 0.00012
  }
}
```

**Response (LIMIT order — async, pending):**
```json
{
  "Success": true,
  "ErrMsg": "",
  "OrderDetail": {
    "OrderID": 83,
    "Status": "PENDING",
    "Role": "MAKER",
    "FilledQuantity": 0,
    "FilledAverPrice": 0,
    "CommissionPercent": 0.00008
  }
}
```

**Key behavior:**
- **MARKET orders** fill immediately. `Status: "FILLED"` and `FilledAverPrice` are always present in the response. Commission = 0.012% (TAKER rate).
- **LIMIT orders** are async. `Status: "PENDING"`, must poll with `query_order`. Commission = 0.008% (MAKER rate) when eventually filled.
- `quantity` is in **coin units** (BTC), not USD. Calculate as: `quantity = usd_to_spend / current_price`
- Order rejected if `price × quantity ≤ MiniOrder (1.0 USD)` — minimum order value is $1

**Order validation before placing:**
```python
min_qty = 1.0 / current_price  # MiniOrder / price
if quantity < min_qty:
    raise ValueError(f"Order too small: {quantity} < {min_qty}")
```

---

### 7. Query Order

```
POST /v3/query_order
Auth: RCL_TopLevelCheck
Content-Type: application/x-www-form-urlencoded
```

| Parameter | Type | Mandatory | Description |
|---|---|---|---|
| `timestamp` | STRING | YES | |
| `order_id` | STRING | NO | If sent, no other optional params allowed |
| `pair` | STRING | NO | e.g. `BTC/USD` |
| `offset` | STRING_INT | NO | Pagination offset |
| `limit` | STRING_INT | NO | Default 100 |
| `pending_only` | STRING_BOOL | NO | `"TRUE"` or `"FALSE"` |

> **CRITICAL GOTCHA:** `pending_only` must be the **string** `"TRUE"` or `"FALSE"`, not Python booleans `True`/`False`. Python booleans serialize as `"True"`/`"False"` (wrong case) and may cause incorrect filtering. Always use string literals.

**Response (orders found):**
```json
{
  "Success": true,
  "ErrMsg": "",
  "OrderMatched": [
    {
      "Pair": "BTC/USD",
      "OrderID": 81,
      "Status": "FILLED",
      "FilledAverPrice": 68420.50,
      "FilledQuantity": 0.052341,
      "CommissionChargeValue": 0.4297
    }
  ]
}
```

**Response (no orders matched):**
```json
{
  "Success": false,
  "ErrMsg": "no order matched"
}
```

**Order status values:** `PENDING`, `FILLED`, `CANCELED`

**Usage patterns:**
```python
# Query specific order
query_order(order_id="81")

# Query all pending orders for a pair
query_order(pair="BTC/USD", pending_only="TRUE")

# Query all orders for account
query_order()  # no optional params
```

---

### 8. Cancel Order

```
POST /v3/cancel_order
Auth: RCL_TopLevelCheck
Content-Type: application/x-www-form-urlencoded
```

| Parameter | Type | Mandatory | Description |
|---|---|---|---|
| `timestamp` | STRING | YES | |
| `order_id` | STRING | NO | Cancel specific order |
| `pair` | STRING | NO | Cancel all pending for this pair |

- `order_id` and `pair` are mutually exclusive — send at most one
- If neither is sent, **all pending orders are cancelled**
- Only `PENDING` orders can be cancelled

**Response:**
```json
{
  "Success": true,
  "ErrMsg": "",
  "CanceledList": [20, 35]
}
```

---

## Response Envelope

All responses include:

| Field | Type | Description |
|---|---|---|
| `Success` | BOOL | `true` if request succeeded |
| `ErrMsg` | STRING | Empty `""` on success, error description on failure |
| `ServerTime` | INT | Server timestamp at response time (on some endpoints) |

Always check `response.get("Success")` before using response data.

---

## Known Quirks and Gotchas

### 1. No OHLCV Endpoint (Permanent Limitation)
There is no candlestick or OHLCV data endpoint. All high/low/open/close values in live trading must be synthesized from polling `LastPrice`. This means ATR, ADX, and OBV computed live will diverge from backtested values. This cannot be fixed.

**Implication:** Backtest indicator values are computed on real OHLCV bars; live values are not. Walk-forward optimization reduces but does not eliminate this gap.

### 2. `pending_count` Returns `Success: false` for Zero Pending
Normal behavior, not an error. Handle specifically:
```python
result = client.get_pending_count()
# Do NOT: if not result.get("Success"): raise_error()
# DO: count = result.get("TotalPending", 0)
```

### 3. `pending_only` Must Be a String
```python
# WRONG — Python bool serializes as "True" (capital T only)
payload["pending_only"] = True

# CORRECT
payload["pending_only"] = "TRUE"
```

### 4. Market Order Fill Price Always Available
For MARKET orders, `FilledAverPrice` is always populated in the immediate response. The `Price` field in `OrderDetail` reflects the fill price as well. Never fall back to computing fill price from other fields.

### 5. `quantity` Is in Coin Units, Not USD
The API takes coin quantity, not USD value:
```python
# To buy $1000 worth of BTC at $68,420:
quantity = 1000 / 68420  # = 0.014618 BTC
```

### 6. No Rate Limit Documentation
The API docs contain no mention of rate limits. Observed safe polling: once per minute per endpoint. Do not hammer the API — exponential backoff on errors is already implemented in Layer 1.

### 7. POST Body Must Be Form-Encoded
POST requests (`place_order`, `query_order`, `cancel_order`) must send `total_params` as the raw form-encoded body string, not as JSON and not as a dict. The signature covers this exact string:
```python
requests.post(url, headers=headers, data=total_params)
# NOT: requests.post(url, headers=headers, json=payload)
# NOT: requests.post(url, headers=headers, data=payload)
```

### 8. LIMIT Orders Lock Funds
When a LIMIT BUY order is placed, the USD cost is moved to `USD.Lock` until filled or cancelled. Account for this when computing available capital.

---

## Commission Rates

| Role | Rate | When |
|---|---|---|
| TAKER | 0.012% | MARKET orders, immediately-fillable LIMIT orders |
| MAKER | 0.008% | LIMIT orders that rest in the order book |

Commission is always charged in USD.

---

## Complete Signing Implementation

```python
import hmac
import hashlib
import time
import requests

BASE_URL = "https://mock-api.roostoo.com"

class RoostooClient:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self._time_offset_ms = 0  # Set via sync_time()

    def sync_time(self):
        """Synchronise local clock with server. Call once on startup."""
        resp = requests.get(f"{BASE_URL}/v3/serverTime", timeout=5).json()
        server_ms = resp["ServerTime"]
        local_ms = int(time.time() * 1000)
        self._time_offset_ms = server_ms - local_ms

    def _ms_timestamp(self) -> str:
        return str(int(time.time() * 1000) + self._time_offset_ms)

    def _sign(self, payload: dict) -> tuple[dict, str]:
        payload["timestamp"] = self._ms_timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)
        sig = hmac.new(
            self.secret_key.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": sig,
        }
        return headers, total_params

    def get_balance(self) -> dict:
        headers, params = self._sign({})
        return requests.get(
            f"{BASE_URL}/v3/balance", headers=headers, params=params, timeout=10
        ).json()

    def get_ticker(self, pair: str = None) -> dict:
        params = {"timestamp": self._ms_timestamp()}
        if pair:
            params["pair"] = pair
        return requests.get(
            f"{BASE_URL}/v3/ticker", params=params, timeout=10
        ).json()

    def place_market_order(self, pair: str, side: str, quantity: float) -> dict:
        payload = {"pair": pair, "side": side, "type": "MARKET", "quantity": str(quantity)}
        headers, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return requests.post(
            f"{BASE_URL}/v3/place_order", headers=headers, data=total_params, timeout=10
        ).json()

    def cancel_all(self, pair: str = None) -> dict:
        payload = {}
        if pair:
            payload["pair"] = pair
        headers, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return requests.post(
            f"{BASE_URL}/v3/cancel_order", headers=headers, data=total_params, timeout=10
        ).json()

    def get_pending_count(self) -> int:
        """Returns number of pending orders (0 if none — never raises on Success=false)."""
        headers, params = self._sign({})
        result = requests.get(
            f"{BASE_URL}/v3/pending_count", headers=headers, params=params, timeout=10
        ).json()
        return result.get("TotalPending", 0)

    def query_orders(self, pair: str = None, pending_only: bool = False) -> list:
        payload = {}
        if pair:
            payload["pair"] = pair
            payload["pending_only"] = "TRUE" if pending_only else "FALSE"
        headers, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        result = requests.post(
            f"{BASE_URL}/v3/query_order", headers=headers, data=total_params, timeout=10
        ).json()
        return result.get("OrderMatched", [])
```

---

## Endpoint Summary Table

| Endpoint | Method | Auth | Use |
|---|---|---|---|
| `/v3/serverTime` | GET | None | Clock sync |
| `/v3/exchangeInfo` | GET | None | Pair metadata, MiniOrder |
| `/v3/ticker` | GET | TSCheck | Current price, 24h volume |
| `/v3/balance` | GET | Signed | USD and coin balances |
| `/v3/pending_count` | GET | Signed | Count of PENDING orders |
| `/v3/place_order` | POST | Signed | Submit MARKET or LIMIT order |
| `/v3/query_order` | POST | Signed | Order history and status |
| `/v3/cancel_order` | POST | Signed | Cancel pending orders |
