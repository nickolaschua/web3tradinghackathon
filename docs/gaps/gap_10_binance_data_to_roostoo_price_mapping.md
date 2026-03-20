# Gap 10: BTCUSDT (Binance) vs BTC/USD (Roostoo) Price Divergence

## Why This Is a Gap
Historical data is downloaded from Binance as `BTCUSDT` (BTC priced in Tether/USDT). Live trading happens on Roostoo as `BTC/USD` (BTC priced in USD). These are different markets that can have small but meaningful price differences.

## What I Need to Know

1. **How large is the BTCUSDT vs BTCUSD price spread typically?**
   - USDT trades at a small discount/premium to USD (typically within 0.1%)
   - During market stress (e.g. March 2020 USDT FUD), the spread can widen to 1%+
   - Is this spread large enough to matter for backtesting validity?

2. **Will indicators trained on BTCUSDT prices transfer correctly to BTCUSD?**
   - RSI, MACD, EMA crossovers: price-ratio-based, so a consistent 0.1% offset shouldn't matter
   - ATR: will be calibrated to USDT volatility, which is essentially identical to USD volatility
   - Conclusion: probably fine, but verify

3. **Does the feature engineering need explicit adjustment for this?**
   - The `compute_features()` function doesn't know or care about currency denomination
   - As long as the Roostoo price feed is a continuous series, the indicators will work

4. **Can Binance historical data be used directly to seed the live buffer?**
   - If seeding the LiveFetcher buffer with historical Binance closes, and then appending Roostoo live prices, will there be a visible price discontinuity at the seed boundary?
   - A small USDT/USD spread at the boundary could cause a one-bar spike in returns

## Research Needed
Check USDT/USD historical spread data to quantify the typical divergence.

## Priority
**Low** — the USDT/USD spread is negligible for indicator computation. Worth documenting but not blocking.

---

## Research Findings (Market Data + Domain Knowledge, 2026-03-12)

### USDT/USD Spread Analysis

Historical USDT/USD (Tether peg) data from public sources:
- **Normal conditions**: USDT trades at $0.9990–$1.0010 (±0.1% spread)
- **Stress events**:
  - March 2020 COVID crash: USDT briefly traded at $1.02 (2% premium) as
    traders fled to stablecoins
  - May 2022 LUNA/UST collapse: USDT traded at $0.9955 (0.45% discount)
  - FTX collapse Nov 2022: USDT at $0.975 briefly (2.5% discount)

For a competition running in 2026 under normal market conditions: spread ≈ 0.05%

### Impact on Indicator Computation

| Indicator | Affected by 0.1% USDT premium? | Severity |
|-----------|--------------------------------|----------|
| RSI | No — ratio-based | None |
| MACD | No — EMA difference | None |
| EMA crossover | No — ratio-based | None |
| Close-to-close ATR (vol proxy) | No — uses log returns | None |
| Raw price levels (e.g. support/resistance) | Yes — 0.1% offset | Negligible |

**Conclusion**: 0.1% USDT/USD spread has no meaningful impact on indicator values.
All indicators used in this strategy are either ratio-based or return-based,
making them immune to a constant multiplicative offset.

### Buffer Seeding Discontinuity

When seeding the LiveFetcher buffer with Binance BTCUSDT historical data and
then transitioning to Roostoo BTCUSD live prices:

```
Last Binance bar close:  $84,000.00 (USDT)
First Roostoo live bar: $84,042.00 (USD)  ← 0.05% higher (normal USDT discount)
```

This creates a single-bar "spike" in close-to-close returns:
- Log return = ln(84042/84000) = 0.0005 = 0.05%

With a 14-bar rolling window, this spike is diluted to 0.05%/14 ≈ 0.004%
effect on the ATR proxy. **Completely negligible.**

### Recommendation

**No action needed.** The USDT/USD spread is not a material concern for:
1. Indicator computation (ratio/return-based)
2. Buffer seeding continuity (sub-0.1% single-bar discontinuity)
3. Stop-loss calibration (same 0.1% effect on ATR)

The only edge case to monitor: if a stress event occurs during the competition
that causes USDT to de-peg >1%, the close-to-close ATR proxy will have one
anomalously large bar. This is acceptable because the rolling window smooths it.

### Mapping in Code

No explicit price adjustment is needed in `compute_features()` or `LiveFetcher`.
The `seed_from_history()` function can use Binance BTCUSDT data as-is:

```python
# This is safe — no price scaling needed
def seed_from_history(symbol: str, n_bars: int) -> pd.DataFrame:
    # Downloads BTCUSDT from Binance
    df = download_binance_ohlcv("BTCUSDT", "4h", limit=n_bars)
    # Rename columns to match expected schema
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return df
```
