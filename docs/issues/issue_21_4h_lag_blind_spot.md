# Issue 21: 4H Candle Lag Creates Blind Spot During Intra-Bar Crashes

## Layer
Layer 5 — Strategy Engine / Layer 3 — Data Pipeline

## Description
Signal generation only fires at 4H epoch boundaries (when a new completed candle is appended
to `_buffers`). If BTC crashes 20-30% within a single 4H window (e.g., similar to Luna/USDC
depeg, FTX collapse), no new signal is generated until the next epoch close.

The 60s `check_stops()` cycle *does* fire using live `_last_prices`, so existing positions are
still protected by ATR trailing stops and hard stops. However, there is a critical dependency:

- If the Roostoo API becomes unavailable mid-crash (which is likely during peak chaos),
  `poll_ticker()` will fail, `_last_prices` will stop updating, and stops will never
  re-evaluate until the API recovers.
- There is no staleness guard on `_last_prices`. The stop check loop will silently
  use the last known price indefinitely.

## Impact
**High** — In a fast crash with API degradation, the bot could hold a losing position for
hours past where the stop should have fired.

## Fix Required
1. Add a timestamp to each `_last_prices` entry when it is written.
2. In the stop-check loop, if `_last_prices[pair]` is older than `N` seconds (e.g., 120s),
   skip the stop check and log a critical warning rather than silently using stale price.
3. Consider triggering an emergency flat-all on repeated API failure (configurable).
