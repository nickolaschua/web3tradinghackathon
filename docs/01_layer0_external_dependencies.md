# Layer 0 — External Dependencies

## What This Layer Is

Layer 0 is everything outside your codebase that you depend on. You do not build any of it — you consume it. Understanding its characteristics, limitations, and failure modes is critical because every other layer depends on it being available and correct.

There are four external dependencies:

1. **Binance Public Data Archive** — historical OHLCV data for backtesting
2. **Roostoo Mock Exchange API** — live price data and order execution during the competition
3. **Alternative.me Fear & Greed Index API** — sentiment data for regime filtering
4. **Telegram Bot API** — outbound alerting and monitoring

---

## Dependency 1: Binance Public Data Archive

**URL:** `https://data.binance.vision`

**What it provides:** Free bulk downloads of historical spot OHLCV candlestick data for every Binance trading pair, going back to 2017. Organized as monthly zip files containing CSVs.

**URL pattern:**
```
https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YEAR}-{MONTH}.zip
```

Example:
```
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-2024-01.zip
```

**What you need to download:**

| Pair | Intervals | Years |
|---|---|---|
| BTCUSDT | 1h, 4h, 1d | 2020–2025 |
| ETHUSDT | 1h, 4h, 1d | 2020–2025 |
| SOLUSDT | 1h, 4h, 1d | 2020–2025 |
| BNBUSDT | 1h, 4h, 1d | 2020–2025 |

**CSV column order:** `open_time, open, high, low, close, volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore`

**Important nuance — USDT vs USD:** Binance uses USDT (Tether) pairs, not USD pairs. Roostoo uses USD pairs. In normal conditions USDT trades within 0.05–0.1% of USD. This discrepancy is acceptable for backtesting. Do not spend time trying to correct for it.

**Fastest download method:** Use the `binance-historical-data` PyPI package:
```bash
pip install binance-historical-data
```
```python
from binance_historical_data import BinanceDataDumper

dumper = BinanceDataDumper(
    path_dir_where_to_dump="./raw_data",
    asset_class="spot",
    data_type="klines",
    data_frequency="4h",
)
dumper.dump_data(
    tickers=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
    date_start=datetime.date(2020, 1, 1),
    date_end=datetime.date(2025, 12, 31),
)
```

**Failure modes:**
- Some monthly files have missing candles during exchange downtime. Your `gap_detector.py` (Layer 1) handles this.
- The download server is occasionally slow. Download everything before the hackathon starts and store locally as Parquet.
- Always verify the downloaded data visually — plot BTC 4H from 2020–2025 and confirm it shows the 2021 bull run peak (~$69k), the 2022 crash to ~$16k, and the 2024 recovery.

---

## Dependency 2: Roostoo Mock Exchange API

**Base URL:** `https://mock-api.roostoo.com`

**What it provides:** A simulated cryptocurrency exchange that mirrors real market prices. You receive API keys before the competition starts. Every team starts with $50,000 USD in virtual capital. Orders are executed against real-time price feeds.

**Complete endpoint inventory:**

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/v3/serverTime` | GET | None | Get server timestamp for sync |
| `/v3/exchangeInfo` | GET | None | Get all trading pairs + precision rules |
| `/v3/ticker` | GET | Timestamp | Get live prices for one or all pairs |
| `/v3/balance` | GET | Signed | Get current wallet holdings |
| `/v3/pending_count` | GET | Signed | Count of open pending orders |
| `/v3/place_order` | POST | Signed | Submit a new order |
| `/v3/query_order` | POST | Signed | Look up order history or pending orders |
| `/v3/cancel_order` | POST | Signed | Cancel one or all pending orders |

**Authentication levels:**
- `RCL_NoVerification` — no auth needed (serverTime, exchangeInfo)
- `RCL_TSCheck` — timestamp parameter required (ticker)
- `RCL_TopLevelCheck` — full HMAC signature required (all trading endpoints)

**Rate limits:** The exact rate limit is not published in the docs, but the competition is explicitly designed for ~1 trade per minute. Enforce 65 seconds minimum between `place_order` calls internally. Polling endpoints (ticker, balance) have more headroom — poll every 60 seconds with ±5 seconds of jitter.

**Key behavioural details:**
- MARKET orders execute immediately as taker orders (0.012% commission, `Status: FILLED` in response)
- LIMIT orders become maker orders (0.008% commission, `Status: PENDING` until filled)
- POST requests must set `Content-Type: application/x-www-form-urlencoded`
- All parameters must be sorted alphabetically by key before signing
- Timestamp must be a 13-digit millisecond timestamp, within 60 seconds of server time

**What to do when the API behaves unexpectedly:** The hackathon provides a WhatsApp channel with Roostoo engineers. Use it immediately if you observe behaviour that contradicts the documentation. Do not spend hours debugging something that might be a server-side issue.

**Failure modes:**
- API downtime: build exponential backoff (2s, 4s, 8s, max 3 retries). Alert via Telegram after 3 consecutive failures.
- `Success: false` in response body with HTTP 200: always parse the JSON body, never trust HTTP status code alone.
- Timestamp rejection: implement application-level time sync (see Layer 1).

---

## Dependency 3: Alternative.me Fear & Greed Index

**URL:** `https://api.alternative.me/fng/`

**What it provides:** A daily sentiment index for the crypto market, scored 0–100. Scores below 20 indicate "Extreme Fear" (historically a buy signal). Scores above 80 indicate "Extreme Greed" (historically a warning to reduce exposure).

**API call:**
```python
import requests
response = requests.get("https://api.alternative.me/fng/?limit=30")
data = response.json()
# Returns last 30 daily values with timestamps
```

**Response structure:**
```json
{
  "data": [
    {"value": "45", "value_classification": "Fear", "timestamp": "1234567890"},
    ...
  ]
}
```

**How to use it:** This is a regime filter, not a trading signal. Use it as a secondary confirmation:
- FGI < 20 (Extreme Fear) + price at support → increases conviction to enter long
- FGI > 80 (Extreme Greed) + price extended → reduces conviction, tighten stops

Do not use FGI as a standalone signal. It updates once per day and is too coarse to time entries precisely. Its value is in filtering out bad entries during euphoric markets.

**Failure modes:** The API is free and occasionally returns errors. Treat FGI as optional — if the call fails, proceed without it. Never block trading on FGI unavailability.

---

## Dependency 4: Telegram Bot API

**URL:** `https://api.telegram.org/bot{TOKEN}/sendMessage`

**What it provides:** A simple HTTP endpoint that sends a message to a Telegram chat from your bot. This is your monitoring lifeline during the competition.

**Setup (one-time, 15 minutes):**
1. Open Telegram, search for @BotFather
2. Send `/newbot`, follow prompts to get a `BOT_TOKEN`
3. Start a chat with your new bot, then visit `https://api.telegram.org/bot{TOKEN}/getUpdates` to get your `CHAT_ID`
4. Store both in your `.env` file

**Sending a message:**
```python
import requests

def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass  # Never let a Telegram failure propagate to the main loop
```

**What to alert on:**
- Every trade executed (pair, side, price, quantity, commission)
- Every `Success: false` response from Roostoo
- Every regime transition (BULL → BEAR, etc.)
- Portfolio circuit breaker triggered
- Bot startup and shutdown
- Heartbeat every 10 minutes (proof of life)
- 3 consecutive API failures
- Timestamp offset exceeding 10 seconds

**Critical design rule:** The Telegram call must always be wrapped in its own try/except. A Telegram API failure must never propagate upward and kill the main trading loop. Monitoring is secondary to execution — if Telegram is down, the bot keeps trading.

**Failure modes:** Telegram rate-limits bots to ~30 messages/second. You will never hit this. Occasional Telegram API slowness is harmless — set a 5-second timeout and move on if it fails.

---

## How Layer 0 Relates to the Rest of the System

Layer 0 is purely consumed, never modified. But its availability and quality determine the validity of everything built on top of it.

- If Binance data has gaps, your feature calculations will produce NaN, which will corrupt signal generation. Layer 1's gap detector exists specifically for this.
- If the Roostoo API changes behaviour, every layer that calls it is affected. This is why the API client (Layer 1) is a single isolated module — if the API changes, you change one file.
- If Telegram goes down during competition, you lose visibility but not trading capability. This is the correct failure mode — alerting is important but not critical-path.

The single most important thing about Layer 0: **download and validate all historical data before the hackathon starts**. Every other pre-competition task can be done under time pressure. Data download cannot — a corrupted or incomplete dataset discovered on competition day cannot be fixed in time.
