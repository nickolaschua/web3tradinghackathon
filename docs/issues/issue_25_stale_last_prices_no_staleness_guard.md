# Issue 25: _last_prices Has No Staleness Guard — Stale Price Used Indefinitely

## Layer
Layer 1 — API Client / Layer 6 — Risk Management

## Description
`LiveFetcher._last_prices` is updated by `poll_ticker()` every 60s cycle. The stop-check
loop in `main.py` reads from `_last_prices` to evaluate ATR trailing stops and hard stops.

There is no timestamp attached to `_last_prices` entries and no check for how old the price is.
If `poll_ticker()` silently fails (connection timeout, Roostoo API hiccup, 429 rate limit),
`_last_prices` retains the last known value indefinitely. The stop-check loop proceeds
as if the price is current.

Practical failure:
- BTC price is $50,000, recorded in `_last_prices`
- Roostoo API goes down for 30 minutes during a flash crash
- BTC actually moves to $42,000 (below hard stop at $47,500)
- `check_stops()` sees $50,000 (stale) → no exit
- When API recovers, `_last_prices` updates to $42,000 → stop fires 30 min late

## Impact
**High** — During API outages that coincide with crashes, stop-losses fail to fire.
This is the most likely real failure mode in production.

## Fix Required
1. Change `_last_prices` to store `(price, timestamp)` tuples.
2. In the stop-check loop (`_run_one_cycle`), check:
   ```python
   price, ts = live_fetcher.get_last_price_with_ts(pair)
   if (now - ts).seconds > 180:
       logger.critical(f"STALE PRICE for {pair}: {(now-ts).seconds}s old, skipping stop check")
       continue
   ```
3. Add a metric/alert for consecutive poll failures.
