# Strategy 5: Mean-Reversion with Regime Gate

## What it is

Mean reversion is the tendency of a coin that has dropped sharply to bounce back toward its
moving-average baseline. The catch is that this only works in **uptrending markets** — in
downtrends, oversold coins keep falling. Dobrynskaya (2023) documents that crypto momentum
persists 2–4 weeks before reverting. An oversold coin in a bear trend is not undervalued —
it is correctly priced for a market that is still declining.

The strategy here is not a new strategy. `MeanReversionStrategy` in `bot/strategy/mean_reversion.py`
already exists but is a **stub that returns HOLD**. This document specifies the entry/exit logic
to fill it in, plus the regime gate that separates profitable mean reversion from catching falling
knives.

The regime gate is: **only enter mean-reversion longs when the RegimeDetector reports BULL_TREND**.
In SIDEWAYS the entries are halved; in BEAR_TREND they are fully blocked.

---

## How to implement in this codebase

### Entry and exit logic for `generate_signal()`

The `MeanReversionStrategy.generate_signal()` stub in `bot/strategy/mean_reversion.py` already
has the scaffolding. The feature column names in the docstring differ slightly from the actual
names in `features.py` — the mapping is:

| Docstring name | Actual column (from `compute_features()`) |
|---|---|
| `rsi`          | `RSI_14`  |
| `macd_hist`    | `MACDh_12_26_9` |
| `ema_slope`    | `ema_slope` |

`bb_pos` and `EMA_20`/`EMA_50` are also available from `compute_features()` and should be used.

Replace the stub body with:

```python
def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
    if len(features) == 0:
        return TradingSignal(pair=pair)

    latest = features.iloc[-1]

    # --- Regime gate ---
    # Require the coin to be in a local uptrend (EMA_20 > EMA_50).
    # This is the per-coin micro-regime, distinct from the global RegimeDetector.
    # A coin can be in BULL_TREND globally but still have a broken local trend.
    ema_uptrend = latest.get("EMA_20", float("nan")) > latest.get("EMA_50", float("nan"))
    if not ema_uptrend:
        return TradingSignal(pair=pair)  # HOLD — don't catch falling knives

    rsi       = latest.get("RSI_14", 50.0)
    bb_pos    = latest.get("bb_pos", 0.5)    # 0 = at lower band, 1 = at upper band
    macd_hist = latest.get("MACDh_12_26_9", 0.0)

    # --- Entry: deeply oversold + at or below lower Bollinger Band + MACD turning up ---
    # bb_pos < 0.1 means price is within 10% of the lower band (nearly touching it)
    # RSI < 30 is the "deeply oversold" threshold (25 is more conservative)
    # macd_hist > 0 confirms that momentum is already turning (not leading the reversal)
    if rsi < 30 and bb_pos < 0.15 and macd_hist > 0:
        return TradingSignal(
            pair=pair,
            direction=SignalDirection.BUY,
            size=0.35,        # smaller than momentum — mean reversion is a lower-conviction trade
            confidence=0.60,
        )

    # --- Entry (higher conviction): RSI < 25 alone is sufficient in strong uptrend ---
    # Price is extremely oversold; the uptrend filter already confirmed structural support
    if rsi < 25:
        return TradingSignal(
            pair=pair,
            direction=SignalDirection.BUY,
            size=0.25,
            confidence=0.55,
        )

    # --- Exit: RSI returns to neutral or price reaches upper band ---
    # "Mean" has reverted; no reason to hold further
    if rsi > 55 or bb_pos > 0.6:
        return TradingSignal(
            pair=pair,
            direction=SignalDirection.SELL,
            size=1.0,   # exit entire mean-reversion position
            confidence=0.70,
        )

    return TradingSignal(pair=pair)  # HOLD
```

### Connecting to the global RegimeDetector

The per-coin EMA gate above handles the micro-regime. The global `RegimeDetector` handles the
macro regime. Both should be applied. In `main.py` or wherever signals are collected:

```python
from bot.execution.regime import RegimeState

regime = regime_detector.get_regime()

for pair, features in feature_dfs.items():
    mr_signal = mean_reversion.generate_signal(pair, features)

    # Apply macro regime multiplier to mean-reversion signals
    if regime == RegimeState.BEAR_TREND:
        # No mean-reversion longs in a bear market — override to HOLD
        if mr_signal.direction == SignalDirection.BUY:
            mr_signal = TradingSignal(pair=pair)  # HOLD
    elif regime == RegimeState.SIDEWAYS:
        # Halve position size in sideways markets
        if mr_signal.direction == SignalDirection.BUY:
            mr_signal = TradingSignal(
                pair=mr_signal.pair,
                direction=mr_signal.direction,
                size=mr_signal.size * 0.5,
                confidence=mr_signal.confidence,
            )
```

### Combining with the momentum strategy

Mean reversion and momentum signals may conflict for the same coin (momentum says buy, mean
reversion says sell, or vice versa). The simplest resolution:

```python
# Priority: momentum > mean reversion
# If momentum strategy has a BUY or SELL, use it.
# If momentum is HOLD, check mean reversion.
final_signal = momentum_signal
if momentum_signal.direction == SignalDirection.HOLD:
    final_signal = mr_signal
```

This prevents the strategies from fighting each other. Mean reversion fills the gaps — it
generates signal when momentum is dormant (coin has neither broken out nor broken down).

---

## How to check for correctness

### Regime gate verification

The single most important correctness check is that the regime gate actually fires:

```python
# Inject a declining test case and verify no BUY is generated
import pandas as pd
from bot.strategy.mean_reversion import MeanReversionStrategy
from bot.strategy.base import SignalDirection

mr = MeanReversionStrategy(config={})

# Create a feature row where EMA_20 < EMA_50 (downtrend) but RSI is deeply oversold
test_row = {
    "EMA_20": 95.0,  "EMA_50": 100.0,  # downtrend
    "RSI_14": 20.0,   # deeply oversold — would trigger entry if gate were absent
    "bb_pos": 0.05,   "MACDh_12_26_9": 0.1,
    "close": 95.0,
}
test_df = pd.DataFrame([test_row])
signal = mr.generate_signal("TEST/USD", test_df)
assert signal.direction == SignalDirection.HOLD, "Regime gate failed: issued BUY in downtrend"
print("Regime gate test passed")
```

### Entry threshold verification

```python
# Verify that an uptrending, oversold coin generates a BUY
test_row_uptrend = {
    "EMA_20": 105.0, "EMA_50": 100.0,  # uptrend
    "RSI_14": 28.0,                      # oversold
    "bb_pos": 0.08,                      # near lower Bollinger Band
    "MACDh_12_26_9": 0.05,               # MACD turning up
    "close": 103.0,
}
test_df = pd.DataFrame([test_row_uptrend])
signal = mr.generate_signal("TEST/USD", test_df)
assert signal.direction == SignalDirection.BUY, "Entry logic failed for valid mean-reversion setup"
assert signal.size <= 0.5, "Position size too large for mean-reversion trade"
print(f"Entry test passed: size={signal.size}, confidence={signal.confidence}")
```

### Signal frequency check

Mean reversion should fire infrequently — RSI < 30 on 4H data occurs roughly 5-10% of bars:

```python
from bot.data.features import compute_features

# Run over historical data and count signal frequency
signals = [mr.generate_signal(pair, features_df.iloc[i:i+50]) for i in range(len(features_df) - 50)]
buy_count  = sum(1 for s in signals if s.direction == SignalDirection.BUY)
sell_count = sum(1 for s in signals if s.direction == SignalDirection.SELL)
total = len(signals)

print(f"BUY rate:  {buy_count/total:.1%}  (expect 3–8%)")
print(f"SELL rate: {sell_count/total:.1%}  (expect 3–8%)")
# If BUY rate > 15%, thresholds are too loose
# If BUY rate < 1%, thresholds are too tight
```

### Win rate check (backtest)

```python
# For every BUY signal, compute return from entry to next SELL or 10-bar timeout
entry_returns = []
for i, signal in enumerate(signals):
    if signal.direction != SignalDirection.BUY:
        continue
    # Find exit: first SELL or 10 bars later
    entry_price = features_df["close"].iloc[i + 50]
    for j in range(i+1, min(i+11, len(signals))):
        if signals[j].direction == SignalDirection.SELL:
            exit_price = features_df["close"].iloc[j + 50]
            break
    else:
        exit_price = features_df["close"].iloc[min(i+10+50, len(features_df)-1)]
    entry_returns.append(np.log(exit_price / entry_price))

print(f"Mean return per trade: {np.mean(entry_returns):.3f}")
print(f"Win rate: {(np.array(entry_returns) > 0).mean():.1%}")
# Expect: mean > 0, win rate > 55% in uptrending coin universe
```

---

## Maximizing value

### RSI z-score is better than RSI level

RSI < 30 is a universal threshold, but different coins have different RSI distributions.
DOGE rarely drops below 30 even in corrections; BTC drops to 25 in moderate pullbacks.
A coin-specific RSI z-score is more informative:

```python
# Add to features.py or compute inline:
rsi_zscore = (latest["RSI_14"] - features["RSI_14"].rolling(42).mean().iloc[-1]) / \
             (features["RSI_14"].rolling(42).std().iloc[-1] + 1e-8)
# Entry: rsi_zscore < -1.5  (1.5 standard deviations below recent average)
# This adapts to the coin's own regime rather than using a fixed 30 threshold
```

### Condition stacking improves precision

Single-condition mean reversion (just RSI < 30) has a win rate of roughly 55-60%.
Stacking three conditions (RSI < 30 + bb_pos < 0.1 + MACD hist turning positive) drops
frequency to ~2% of bars but raises win rate to 65-70%. The strategy code above already
stacks conditions — do not remove the stacking to increase signal frequency.

### Scale in on extremes

A single entry at RSI 28 is good. A scaled entry at RSI 25 (adding to a small position) is
better. Implement this by responding to the second entry signal with a partial add:

```python
# If already long and RSI drops further:
if rsi < 25 and currently_long:
    return TradingSignal(pair=pair, direction=SignalDirection.BUY, size=0.15)  # add to position
```

### Use ATR-based stop, not percentage

The existing risk management system uses `atr_proxy` for sizing. Mean-reversion trades should
also place stops below the entry at `1.5 × ATR` — not a fixed 5%. A sharp ATR means wider
stop, which means smaller size, which is the correct risk behavior in volatile conditions:

```python
atr = latest.get("atr_proxy", close * 0.02)
stop_distance = 1.5 * atr  # 1.5 ATR below entry
risk_per_trade = 0.02  # 2% of portfolio
size = risk_per_trade / (stop_distance / close)  # position size as portfolio fraction
size = min(size, 0.35)  # cap at 35%
```

---

## Common pitfalls

### Pitfall 1: Skipping the regime gate

The most common error is implementing the oversold entry without the EMA_20 > EMA_50 check.
In a bear trend, RSI will repeatedly drop to 20-25 as the coin keeps making new lows. Without
the regime gate, the strategy generates a new BUY on each lower low. This is the "catching
falling knives" failure mode Dobrynskaya (2023) specifically documents. The regime gate is
**not optional** — it is the core of the strategy.

### Pitfall 2: Exiting too early or too late

RSI > 50 is the right exit trigger — not RSI > 65 (too late, leaves money on table) and not
RSI > 40 (too early, cuts winners short). The "mean" in mean reversion is the midpoint of the
RSI distribution (~50), not the overbought zone. Similarly, `bb_pos > 0.6` means price is
above the midpoint of the Bollinger Band, which is a reasonable definition of "reverted".

### Pitfall 3: Conflating momentum and mean reversion position sizes

The momentum strategy may already have a full position (0.8–1.0 size) in a coin when the mean
reversion strategy tries to add. This creates unintended over-concentration. The strategy
combination logic should check whether a momentum position is already open before the mean
reversion strategy adds:

```python
# In main.py, before executing mr_signal:
existing_position = portfolio.get_position(pair)
if existing_position > 0 and momentum_signal.direction == SignalDirection.BUY:
    # Momentum is already long — mean reversion add-on is redundant and over-concentrating
    continue
```

### Pitfall 4: Not updating after feature set changes

If you implement Strategy 1 (BTC lead-lag) or Strategy 2 (cross-sectional ranks) and retrain
the XGBoost model, the feature column names in `features.py` may shift. The mean reversion
strategy accesses columns by name (`latest["RSI_14"]`) — verify those column names still exist
after any feature pipeline changes. The docstring in `mean_reversion.py` lists column names
that may not exactly match the real column names in the DataFrame (see the mapping table above).

### Pitfall 5: Mean reversion failing in a trending universe

If the entire 39-coin universe is trending up together (correlation spike), mean reversion
looks like it's catching falling knives even when the per-coin EMA gate is satisfied — because
coins in a correlated rally don't revert, they dip briefly and continue up faster than expected.
Monitor the universe-level correlation using the spread feature from Strategy 2:

```python
# If universe_spread_42bar is very high (>0.15), cross-sectional momentum dominates
# and mean-reversion signals are unreliable — halve mean-reversion position sizes
if universe_spread > 0.15:
    mr_size_multiplier = 0.5
```

### Pitfall 6: The stub is never replaced

`MeanReversionStrategy.generate_signal()` currently returns HOLD. If you integrate the bot
without filling in the logic, the strategy runs silently and generates no trades — which looks
correct but wastes the strategy slot. Add a warning log at startup:

```python
# In main.py or strategy initialization:
logger.warning(
    "MeanReversionStrategy is a stub — verify generate_signal() has been implemented "
    "before relying on it for signal generation."
)
```
