# Issue 07: Cross-Asset NaN Columns Dropped Before They Can Be Filled

## Layer
Layer 3 — Feature Engineering (`data/features.py`)

## Description
`compute_features()` initialises cross-asset feature columns (`btc_return_lag1`, `btc_return_lag2`, etc.) as `np.nan` placeholders and then calls `dropna()`. This `dropna()` removes ALL rows that have NaN in any column — including the rows where cross-asset features are `np.nan` because `compute_cross_asset_features()` hasn't been called yet.

By the time `compute_cross_asset_features()` runs, the DataFrame may be empty or significantly shorter than expected, causing the cross-asset fill to produce no useful data.

## Code Location
`data/features.py` → `compute_features()` (line where `dropna()` is called before cross-asset features are populated)

## Reproduction
1. `compute_features(df, pair="ETH/USD")` initializes `btc_return_lag1 = np.nan`
2. `dropna()` removes all rows (since `btc_return_lag1` is NaN everywhere)
3. `compute_cross_asset_features()` receives empty DataFrame
4. Returns DataFrame with 0 rows

## Fix Required
Option 1: Move `dropna()` to AFTER `compute_cross_asset_features()` is called.
Option 2: Only drop NaN on the indicator columns that should never be NaN (RSI, MACD, etc.) and separately handle cross-asset columns.
Option 3: Use `dropna(subset=[...])` with an explicit list of required columns, excluding cross-asset placeholders.

## Impact
**High** — when computing features for non-BTC pairs (ETH, SOL), the feature DataFrame will be empty, causing every signal computation to fail.
