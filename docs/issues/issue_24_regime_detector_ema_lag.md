# Issue 24: RegimeDetector EMA Crossover Lags Crash Onset by 5-10 Days

## Layer
Layer 5 — Strategy Engine / RegimeDetector

## Description
`RegimeDetector` uses an EMA(20)/EMA(50) crossover on **daily-resampled** 4H candles to
classify market regime as BULL/SIDEWAYS/BEAR. With 2-bar confirmation hysteresis, the regime
transition from BULL to BEAR requires:

1. EMA(20) to cross below EMA(50) — slow signals (EMA(50) is slow)
2. Confirmed for 2 consecutive daily bars
3. Daily resampling means confirmation happens at daily close

In a crash that begins on day 0, EMA(50) will not reflect the move until 5-10 trading days
later depending on the magnitude. During this lag window, `regime_multiplier = 1.0` (BULL)
and the bot continues entering full-size positions even as the market falls.

## Impact
**High for new entries** — The first week of a bear market, the bot trades as if it's still
a bull market. Existing positions are protected by stop-losses, but new BUY signals can
be generated and sized at full 1.0x during the early stages of a crash.

## Fix Required
Options (in increasing complexity):
1. **Short-term**: Lower `regime_warmup_bars` comment to emphasize this is a slow signal.
   Add a warning log when regime changes after a sustained trend.
2. **Medium-term**: Add a fast circuit — e.g., if the 4H close is more than 2× ATR below
   the 4H EMA(20), suppress BUY signals regardless of daily regime.
3. **Long-term**: Add a separate fast regime check on 4H data directly (EMA(8)/EMA(21))
   to catch intraday trend breaks earlier.
