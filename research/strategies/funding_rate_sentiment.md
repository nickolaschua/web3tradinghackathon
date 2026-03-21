# Strategy 3: Funding Rate Sentiment Features

## What it is

Binance perpetual futures have a funding rate that settles every 8 hours (00:00, 08:00, 16:00 UTC).
When the perpetual trades at a premium to spot, longs pay shorts (positive funding). When it trades
at a discount, shorts pay longs (negative funding).

The key insight is behavioral: **extreme positive funding means the long side is crowded and
overleveraged — a contrarian sell signal.** Extreme negative funding means the short side is
crowded — a contrarian buy signal. This is not about collecting the rate itself (that's the
carry trade, which is effectively dead post-ETF), but about using the rate as a *sentiment
thermometer*.

The implementation adds 3 features to the XGBoost model:
1. `funding_rate_change_24h` — how aggressively funding is shifting (momentum in sentiment)
2. `funding_rate_cross_zscore` — where this coin's funding stands vs. the rest of the universe
3. `funding_rate_extreme` — binary flag for truly extreme readings (|z-score| > 2)

Note: the Roostoo API is spot-only. We are not executing a carry trade or interacting with
perpetual futures markets. These features are purely informational inputs to the spot XGBoost
model.

---

## How to implement in this codebase

### Data source

Binance USDT-M Futures API endpoint — **no authentication required** for market data:

```
GET https://fapi.binance.com/fapi/v1/fundingRate
```

Parameters:
- `symbol`: e.g. `BTCUSDT`
- `limit`: up to 1000 (each record = 1 funding period = 8 hours)
- `startTime` / `endTime`: optional, in milliseconds

Response:
```json
[
  {
    "symbol": "BTCUSDT",
    "fundingTime": 1678886400000,
    "fundingRate": "0.00035620"
  }
]
```

The existing `bot/api/client.py` targets `mock-api.roostoo.com`. A separate, lightweight fetcher
is needed for Binance market data. Add `bot/data/funding_fetcher.py`:

```python
"""
Fetch Binance perpetual futures funding rates (public endpoint — no auth).
Used as sentiment features for the spot XGBoost model.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Map Roostoo pair format → Binance USDT-M symbol
# Only coins that exist as USDT perps on Binance
PAIR_TO_SYMBOL = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "BNB/USD": "BNBUSDT",
    "XRP/USD": "XRPUSDT",
    "ADA/USD": "ADAUSDT",
    "AVAX/USD": "AVAXUSDT",
    "DOGE/USD": "DOGEUSDT",
    "LINK/USD": "LINKUSDT",
    "DOT/USD": "DOTUSDT",
    "MATIC/USD": "MATICUSDT",
    "LTC/USD": "LTCUSDT",
    # Add remaining coins from hackathon universe here
}


def fetch_funding_rates(
    symbol: str,
    limit: int = 50,         # 50 × 8h = ~16 days of history
    start_time_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch historical funding rates for a single symbol.

    Returns a DataFrame with columns:
      - fundingTime (datetime, UTC)
      - fundingRate (float)
    Sorted ascending by fundingTime.
    """
    params = {"symbol": symbol, "limit": limit}
    if start_time_ms:
        params["startTime"] = start_time_ms

    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("Funding rate fetch failed for %s: %s", symbol, e)
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    if not data:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    df = pd.DataFrame(data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df.sort_values("fundingTime").reset_index(drop=True)


def fetch_all_funding_rates(
    pairs: list[str],
    limit: int = 50,
) -> dict[str, pd.DataFrame]:
    """
    Fetch funding rates for all pairs in the universe.
    Returns dict mapping pair → funding rate DataFrame.
    """
    result = {}
    for pair in pairs:
        symbol = PAIR_TO_SYMBOL.get(pair)
        if symbol is None:
            logger.debug("No Binance perp symbol for %s — skipping funding rate", pair)
            continue
        result[pair] = fetch_funding_rates(symbol, limit=limit)
        time.sleep(0.1)   # polite rate limiting (no auth = shared rate limit bucket)
    return result
```

### Feature computation

Add `compute_funding_features` to `bot/data/features.py` or a new `funding_features.py`:

```python
def compute_funding_features(
    funding_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    all_funding_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute funding rate sentiment features and align them to the OHLCV 4H bar index.

    Features computed:
      funding_rate_latest       : most recent settled funding rate
      funding_rate_ma_24h       : mean of last 3 settlements (24h)
      funding_rate_change_24h   : latest - 24h-ago (sentiment momentum)
      funding_rate_cross_zscore : z-score of this coin's rate vs. universe median
      funding_rate_extreme      : 1 if |z-score| > 2, else 0

    All features are forward-filled across 4H bars (funding rate only changes every 8h)
    and shifted 1 bar to prevent look-ahead.

    Args:
        funding_df:      Funding rate DataFrame for this coin (from fetch_funding_rates).
        ohlcv_df:        4H OHLCV DataFrame for this coin (must have DatetimeIndex, UTC).
        all_funding_dfs: Funding DataFrames for all coins in universe (for cross-z-score).

    Returns:
        ohlcv_df with funding feature columns appended.
    """
    out = ohlcv_df.copy()

    if funding_df.empty:
        # No funding data: fill with zeros (neutral values)
        out["funding_rate_latest"]       = 0.0
        out["funding_rate_ma_24h"]       = 0.0
        out["funding_rate_change_24h"]   = 0.0
        out["funding_rate_cross_zscore"] = 0.0
        out["funding_rate_extreme"]      = 0
        return out

    # Set funding time as index for reindex
    funding = funding_df.set_index("fundingTime")["fundingRate"]

    # Align to 4H bar index: reindex + forward fill (rate holds until next settlement)
    funding_aligned = funding.reindex(
        out.index, method="ffill"
    )

    # 24h moving average (last 3 × 8h settlements = 24h)
    funding_ma_24h = funding.rolling(3, min_periods=1).mean().reindex(
        out.index, method="ffill"
    )

    # Change over last 24h (24h ago = 3 settlements back)
    funding_3_ago = funding.shift(3).reindex(out.index, method="ffill")
    funding_change_24h = funding_aligned - funding_3_ago

    # Cross-sectional z-score: where does this coin's latest funding rate stand
    # relative to the universe?
    latest_rates = {}
    for pair, df in all_funding_dfs.items():
        if not df.empty:
            latest_rates[pair] = df["fundingRate"].iloc[-1]
    if len(latest_rates) > 1:
        rates_series = pd.Series(latest_rates)
        universe_mean = rates_series.mean()
        universe_std  = rates_series.std()
        this_coin_rate = funding_aligned.iloc[-1] if not funding_aligned.empty else 0.0
        current_zscore = (this_coin_rate - universe_mean) / (universe_std + 1e-10)
    else:
        current_zscore = 0.0

    # Fill the cross_zscore as a constant for now (updated each 4H cycle)
    # In live, this recalculates every 4H; in backtest use a rolling approximation
    out["funding_rate_latest"]       = funding_aligned.shift(1)
    out["funding_rate_ma_24h"]       = funding_ma_24h.shift(1)
    out["funding_rate_change_24h"]   = funding_change_24h.shift(1)
    out["funding_rate_cross_zscore"] = current_zscore   # scalar; replace with rolling in backtest
    out["funding_rate_extreme"]      = (abs(out["funding_rate_cross_zscore"]) > 2).astype(int)

    return out
```

### Calling frequency

Fetch funding rates **once per 4H bar close** (same cadence as OHLCV). Funding settles every 8h,
so you'll see 0-1 new settlements per 4H cycle. Caching the last 50 settlements and doing a
lightweight incremental fetch each cycle is the right pattern:

```python
# In main loop, before model inference:
if bars_since_funding_refresh >= 2:   # refresh every 2 bars = every 8h
    funding_dfs = fetch_all_funding_rates(UNIVERSE)
    bars_since_funding_refresh = 0
bars_since_funding_refresh += 1
```

---

## How to check for correctness

### Settlement time alignment check

Funding rates settle at fixed times (00:00, 08:00, 16:00 UTC). The 4H bar that CLOSES at 00:00
UTC contains the settled rate for that period. Verify alignment:

```python
btc_funding = fetch_funding_rates("BTCUSDT", limit=10)
# Settlement times should be exactly on 8h boundaries
expected_hours = {0, 8, 16}
actual_hours = set(btc_funding["fundingTime"].dt.hour.unique())
assert actual_hours.issubset(expected_hours), f"Unexpected settlement hours: {actual_hours}"
```

### Forward-fill sanity check

After aligning to the 4H bar index, the funding rate should be constant for 2 consecutive bars
(since 8h = 2 × 4H), then step to a new value:

```python
funding_aligned = compute_funding_features(btc_funding_df, btc_ohlcv_df, all_funding_dfs)
# Check that consecutive changes are either 0 (forward-fill) or real step changes
changes = funding_aligned["funding_rate_latest"].diff().dropna()
n_steps = (changes != 0).sum()
n_total = len(changes)
# Expect roughly n_total / 2 changes (one per 8h = every 2 bars)
print(f"Steps: {n_steps}, Total bars: {n_total}, Expected: {n_total//2}")
```

### Cross-universe z-score range

```python
# After computing funding features for all coins:
zscores = {
    pair: feature_dfs[pair]["funding_rate_cross_zscore"].iloc[-1]
    for pair in UNIVERSE
}
z_series = pd.Series(zscores).dropna()
print(z_series.describe())
# Should have mean ≈ 0, std ≈ 1 by construction
# Values outside [-5, 5] are genuine extremes, not errors
```

---

## Maximizing value

### Use change, not level

`funding_rate_change_24h` is more informative than `funding_rate_latest`. Funding at a constant
0.05% is neutral — the market has priced it in. Funding spiking from 0.05% to 0.15% in 24h
signals rapid sentiment shift. The XGBoost model will find this split, but you can verify
it by checking feature importance post-training.

### Sentiment extremes as pre-filter

When `funding_rate_extreme == 1` (|z-score| > 2) and the coin has very high positive funding,
the research (Presto Research, BIS WP1087) consistently shows elevated reversal risk. Use this
as a *signal moderator*: reduce position sizing by 50% for coins in the extreme positive bucket,
regardless of model score.

### Negative funding as a long bias amplifier

Negative funding means shorts are paying longs. When a coin you're considering buying has
negative funding (shorts paying you to hold it), the expected carry is in your favour. This is
a mild but consistent enhancement: all else equal, prefer coins with `funding_rate_latest < 0`
for long entries.

---

## Common pitfalls

### Pitfall 1: Using the accruing rate, not the settled rate

Binance shows an "estimated next funding rate" that updates in real-time. This is different from
the settled historical rate returned by `GET /fapi/v1/fundingRate`. The `premiumIndex` endpoint
returns `fundingRate` (current accruing, not yet settled) and `lastFundingRate` (last settled).
**Use `lastFundingRate` from `premiumIndex` or historical records from `fundingRate` endpoint.**
Using the accruing rate before settlement introduces look-ahead bias in backtests.

### Pitfall 2: Not all hackathon coins have liquid perps

Some coins in the hackathon's 39-coin universe may not have USDT-M perpetual contracts on Binance
(or may have very low open interest). For those coins, `funding_df` will be empty. The code above
handles this by filling with 0.0 (neutral), which is correct — missing funding data is not the
same as neutral funding, but it's the least harmful imputation.

### Pitfall 3: Backtest misalignment

When backtesting, computing the cross-sectional z-score requires knowing all coins' funding rates
at each historical timestamp — not just their latest rate. This requires building a full historical
funding rate matrix (fetch 1000 records per coin = ~333 days of 8h settlements). The live
implementation using the "current scalar" z-score is correct for live trading but wrong for
historical backtests. For the backtest:

```python
# Historical z-score: compute cross-sectionally at each timestamp
funding_panel = pd.DataFrame({
    pair: funding_history[pair].set_index("fundingTime")["fundingRate"]
    for pair in UNIVERSE
})
funding_zscores = funding_panel.sub(
    funding_panel.mean(axis=1), axis=0
).div(funding_panel.std(axis=1) + 1e-10, axis=0)
```

### Pitfall 4: 8h settlement creates pseudo-3-category variable

The forward-filled funding rate only changes every 2 bars (8h / 4h = 2). XGBoost may not split
on a feature that's constant for 2-bar stretches as efficiently as on a feature that changes every
bar. The `funding_rate_change_24h` feature partially resolves this by showing when the step
changed. Consider also adding `bars_since_last_settlement` (0, 1, 2, ..., up to 2) as a feature
to help XGBoost understand the timing within the settlement cycle.
