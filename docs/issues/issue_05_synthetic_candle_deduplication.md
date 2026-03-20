# Issue 05: Buffer Deduplication Produces Incorrect Candle Data

## Layer
Layer 2 — Data Pipeline (`data/live_fetcher.py`)

## Description
The `LiveFetcher` appends a new synthetic bar every 60 seconds tick. For a 4H candle, this means 240 ticks are appended to the buffer before the candle boundary advances. `get_dataframe()` deduplicates by keeping only one entry per candle timestamp boundary.

The problem: the deduplication keeps only the LAST tick within each candle period. This means the "close" of a 4H candle in the live buffer reflects only the most recent poll price before deduplication, not a properly constructed OHLCV bar with open/high/low/close derived across all 240 ticks.

In practice, H=L=O=C for all ticks (because each synthetic bar copies the current last price), so OHLCV accuracy is lost. Volume accumulation across the candle period is also lost.

## Code Location
`data/live_fetcher.py` → `get_dataframe()` deduplication logic

## Related
This compounds Issue 06 (no OHLCV from Roostoo). Even if we could compute proper OHLCV bars, the current deduplication logic wouldn't preserve them.

## Fix Required
Track open/high/low/close/volume within each candle period explicitly:
- `open`: first price seen in this candle period
- `high`: max price seen
- `low`: min price seen
- `close`: most recent price (current behavior)
- `volume`: accumulated volume (will be 0 from Roostoo, but structure should be correct)

## Impact
**Medium** — close price is correct (last price seen), but high/low/open are wrong. ATR and Bollinger Bands will be slightly distorted.
