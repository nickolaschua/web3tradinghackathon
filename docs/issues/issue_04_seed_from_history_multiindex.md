# Issue 04: `_seed_from_history()` Fails on Flat Parquet Files

## Layer
Layer 2 — Data Pipeline (`data/live_fetcher.py`)

## Description
`LiveFetcher._seed_from_history()` accesses the loaded Parquet DataFrame using `seed_df[pair]` where `pair` is a string like `"BTC/USD"`. This assumes a multi-level column index (e.g. `(pair, "close")`), which is how vectorbt typically stores multi-asset data.

However, the Binance downloader saves per-pair Parquet files with a flat column index (`open`, `high`, `low`, `close`, `volume`). Accessing `seed_df["BTC/USD"]` on a flat DataFrame will raise `KeyError`.

## Code Location
`data/live_fetcher.py` → `_seed_from_history()` method

## Reproduction
```python
seed_df = pd.read_parquet("data/parquet/BTCUSDT_4h.parquet")
# seed_df.columns = ["open", "high", "low", "close", "volume"]
seed_df["BTC/USD"]  # KeyError: "BTC/USD"
```

## Fix Required
Either:
1. Load the correct per-pair file by mapping `pair → filename` (e.g. `BTC/USD → BTCUSDT_4h.parquet`), or
2. Use the flat column directly: `seed_df["close"]`

## Impact
**Critical** — the bot will fail to seed its buffer on startup, meaning it starts with an empty buffer and cannot generate features until 500+ polling cycles have passed (~8 hours at 60s intervals).
