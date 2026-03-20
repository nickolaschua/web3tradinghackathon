# Issue 11: Regime Detector EMA/ADX Thresholds Calibrated for Daily, Applied to 4H

## Layer
Layer 5 — Strategy Engine (`execution/regime.py`)

## Description
The `RegimeDetector` documentation explicitly notes it was "designed for daily data but used on 4H candles." The EMA periods (20, 50) and ADX threshold (25) are standard values for **daily** timeframe analysis. On a 4H chart:

- EMA(20) on 4H ≈ 80 hours of data ≈ 3.3 days — much shorter lookback than intended
- EMA(50) on 4H ≈ 200 hours ≈ 8.3 days — still very short
- ADX(25) threshold is the same on both timeframes, but ADX behaviour on 4H intraday is noisier than on daily

This means:
1. Regime changes will occur far more frequently than intended (EMA crossovers happen faster on shorter timeframe)
2. The hysteresis buffer (2 confirmation bars) represents only 8 hours rather than 2 days — insufficient smoothing
3. BULL/BEAR regimes will be shorter-lived, causing the strategy to flip position sizing multipliers more often

## Code Location
`execution/regime.py` → `RegimeDetector.__init__()` — EMA periods and ADX threshold config values

## Fix Required
Either:
1. Use daily data (resample 4H bars to daily) for regime detection while keeping 4H for signal generation, or
2. Recalibrate: EMA periods of ~60/150 on 4H approximate the same lookback as 20/50 on daily, and increase hysteresis to 6 bars (24 hours)

## Impact
**Medium** — strategy will overtrade regime transitions. BEAR regimes will be declared too quickly, causing unnecessary cash-holding periods during normal pullbacks.
