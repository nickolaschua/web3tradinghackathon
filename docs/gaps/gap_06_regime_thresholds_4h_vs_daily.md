# Gap 06: Correct EMA and ADX Thresholds for 4H Regime Detection

## Why This Is a Gap
Issue 11 identifies that the regime detector uses EMA(20)/EMA(50) and ADX(25) values calibrated for daily data, but runs on 4H data. I need to know the correct equivalent parameters for a 4H regime detector.

## What I Need to Know

1. **What EMA periods on 4H approximate the same "medium-term" and "long-term" trend signals as EMA(20)/EMA(50) on daily?**
   - 1 daily bar ≈ 6 four-hour bars, so EMA(20d) ≈ EMA(120 4H bars)?
   - This seems very long — verify with backtesting whether EMA(60)/EMA(150) works better on 4H

2. **What ADX threshold correctly separates trending from ranging on 4H BTC data?**
   - ADX(25) is a common daily threshold
   - On 4H, intraday noise may require a higher threshold (30?) or longer ADX period
   - Empirically test: what is the distribution of 4H ADX values for BTC during clear bull/bear trends?

3. **How many confirmation bars (hysteresis) are appropriate for 4H?**
   - Current: 2 bars = 8 hours
   - For noise reduction comparable to 2 daily bars, need ≈12 four-hour bars (48 hours)
   - Is 48-hour hysteresis too slow to catch regime changes in a competition context?

4. **Alternative: Run regime detector on daily-resampled bars**
   - Resample 4H buffer to daily, run EMA(20)/EMA(50) on daily — preserves original calibration
   - Is there enough 4H data to produce reliable daily bars?

## Research Approach
Backtest regime labels from EMA(20/50) daily vs EMA(60/150) 4H on 2020-2025 BTC data, compare regime classification agreement rate.

## Priority
**Medium** — regime detection affects position sizing but the system trades at all. Unoptimized thresholds mean suboptimal sizing, not system failure.

---

## Research Findings (Domain Knowledge + pandas, 2026-03-12)

### EMA Period Equivalence: Daily → 4H

The relationship between daily EMA and 4H EMA is not strictly linear because
EMA uses exponential smoothing, but the **half-life** of the decay is what
matters for equivalence:

EMA(N) half-life ≈ N * ln(2) bars

For equal half-lives:
```
EMA(20 daily) → 20 * 6 = EMA(120 4H)   # 20 trading days = 120 × 4H bars
EMA(50 daily) → 50 * 6 = EMA(300 4H)   # 50 trading days = 300 × 4H bars
```

However, EMA(300) on 4H is extremely slow-moving and adds ~50 bars of lag.
A practical compromise:
```
EMA(20 daily) ≈ EMA(60 4H)   # 10-day period (moderate trend)
EMA(50 daily) ≈ EMA(150 4H)  # 25-day period (longer trend)
```

### Recommended Fix: Resample to Daily in Regime Detector

The cleanest solution (Option 4 from original gap):

```python
def detect_regime(df_4h: pd.DataFrame) -> str:
    """Resample 4H data to daily, then apply original EMA(20/50) logic."""
    # Need at least 50 daily bars → 300 4H bars minimum
    if len(df_4h) < 300:
        return "neutral"

    # Resample to daily OHLCV
    daily = df_4h.resample('1D', on='timestamp').agg({
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'open': 'first',
        'volume': 'sum'
    }).dropna()

    # Apply original calibrated EMA(20/50) logic
    ema_fast = daily["close"].ewm(span=20).mean()
    ema_slow = daily["close"].ewm(span=50).mean()

    last_fast = ema_fast.iloc[-1]
    last_slow = ema_slow.iloc[-1]
    last_close = daily["close"].iloc[-1]

    if last_fast > last_slow and last_close > last_fast:
        return "bull"
    elif last_fast < last_slow and last_close < last_fast:
        return "bear"
    else:
        return "neutral"
```

**Advantages of resample approach**:
1. Preserves original EMA(20/50) calibration — no re-tuning needed
2. Daily bars smooth out 4H noise naturally
3. With 500 bar warmup (500 × 4H = ~83 days), we get ~83 daily bars — enough
   for EMA(50) to stabilize

### ADX on Synthetic Candles

As noted in Gap 02 and Gap 05: ADX is near-zero on synthetic Roostoo candles
because it requires high-low directional movement. The regime detector should
**not** use ADX in live trading.

**Replacement**: Use RSI as a trend-strength proxy:
- RSI > 60 for 3+ consecutive 4H bars → trending regime
- RSI < 40 for 3+ consecutive 4H bars → trending (bear) regime
- RSI between 40-60 → ranging/neutral

Or use the EMA slope as a trend-strength indicator:
```python
# EMA slope as ADX replacement
ema_14 = df["close"].ewm(span=14).mean()
ema_slope = ema_14.diff(3) / df["close"] * 100  # slope as % of price
trending = abs(ema_slope.iloc[-1]) > 0.5  # >0.5% per 3 bars = trending
```

### Hysteresis for 4H Data

With resampled daily bars, the original 2-bar daily hysteresis (2 days = 48 hours)
is preserved naturally. No change needed if using the resample approach.

If keeping native 4H regime detection, scale hysteresis:
- 2 daily bars → 12 × 4H bars = 48 hours of confirmation
- In a short competition, 48 hours may be too slow — reduce to 6 × 4H bars (24 hours)
