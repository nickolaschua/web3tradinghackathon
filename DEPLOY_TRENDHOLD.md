# Deploying TrendHold Strategy to Live Bot

This document tells a future agent exactly how to switch `main.py` from the
currently deployed strategy (XGB BTC + SOL + MR fallback) to the TrendHold
strategy (ML-gated conviction holds + BTC filler).

See `STRATEGY_COMPARISON.md` for the full strategy assessment and backtest results.

---

## What Changes

| Aspect | Current (deployed) | TrendHold (proposed) |
|--------|-------------------|---------------------|
| **Signal logic** | XGB BTC→XGB SOL→MR fallback, fires on ALL tradeable pairs | XGB gates entries/exits for ETH+SOL only; BTC is cycled filler |
| **Models used** | `xgb_btc_15m.pkl`, `xgb_sol_15m.pkl` | `xgb_btc_15m.pkl`, `xgb_eth_15m.pkl`, `xgb_sol_15m.pkl` |
| **Entry threshold** | BTC: 0.65, SOL: 0.75 | All: 0.55 |
| **Exit threshold** | 0.10 (same) | 0.10 (same) |
| **Tradeable pairs** | 39 pairs (full universe) | 3 pairs: BTC/USD, ETH/USD, SOL/USD |
| **Allocations** | Equal weight, max 5 positions | ETH 35%, SOL 30%, BTC filler 10%, 25% cash |
| **Hard stop** | 5% | 12% |
| **ATR multiplier** | 10x | 25x |
| **CB halt threshold** | 30% | 15% |
| **BTC filler** | None | Cycled every 96 bars (24h) for activity requirement |
| **MR fallback** | Active | Removed |

---

## Step-by-Step Instructions

### 1. Update `bot/config/config.yaml`

Replace the risk and universe settings:

```yaml
tradeable_pairs:
  - BTC/USD
  - ETH/USD
  - SOL/USD

feature_pairs:
  - BTC/USD
  - ETH/USD
  - SOL/USD

max_positions: 4

# TrendHold risk params
hard_stop_pct: 0.12
atr_stop_multiplier: 25.0
trailing_stop_multiplier: 25.0

circuit_breaker:
  halt_threshold: 0.15
  reduce_heavy_threshold: 0.12
  reduce_light_threshold: 0.10

risk_per_trade_pct: 0.025

# TrendHold-specific
trendhold:
  eth_pct: 0.35
  sol_pct: 0.30
  btc_filler_pct: 0.10
  xgb_entry_threshold: 0.55
  xgb_exit_threshold: 0.10
  filler_cycle_bars: 96  # 96 x 15min = 24h
```

### 2. Update strategy initialization in `main.py` (around line 747)

Replace the current strategy setup:

```python
# --- CURRENT (remove this) ---
strategy = XGBoostStrategy(threshold=0.65, exit_threshold=0.10)
sol_strategy = XGBoostStrategy(
    model_path="models/xgb_sol_15m.pkl",
    threshold=0.75,
    pair="SOL/USD",
    exit_threshold=0.10,
)
mean_reversion_strategy = MeanReversionStrategy()
```

With:

```python
# --- TRENDHOLD ---
trendhold_cfg = config.get("trendhold", {})
entry_thresh = trendhold_cfg.get("xgb_entry_threshold", 0.55)
exit_thresh = trendhold_cfg.get("xgb_exit_threshold", 0.10)

btc_strategy = XGBoostStrategy(
    model_path="models/xgb_btc_15m.pkl",
    threshold=entry_thresh,
    pair="BTC/USD",
    exit_threshold=exit_thresh,
)
eth_strategy = XGBoostStrategy(
    model_path="models/xgb_eth_15m.pkl",
    threshold=entry_thresh,
    pair="ETH/USD",
    exit_threshold=exit_thresh,
)
sol_strategy = XGBoostStrategy(
    model_path="models/xgb_sol_15m.pkl",
    threshold=entry_thresh,
    pair="SOL/USD",
    exit_threshold=exit_thresh,
)

# Map pair -> (strategy, allocation_pct, role)
trendhold_strategies = {
    "ETH/USD": (eth_strategy, trendhold_cfg.get("eth_pct", 0.35), "hold"),
    "SOL/USD": (sol_strategy, trendhold_cfg.get("sol_pct", 0.30), "hold"),
    "BTC/USD": (btc_strategy, trendhold_cfg.get("btc_filler_pct", 0.10), "filler"),
}
filler_cycle_bars = trendhold_cfg.get("filler_cycle_bars", 96)
```

### 3. Replace signal generation in `_run_one_cycle` (around line 420-495)

The current logic iterates all tradeable pairs and cascades BTC XGB → SOL XGB → MR.
Replace Phase A + Phase B with TrendHold logic:

```python
# ── Phase A: TrendHold signals ──
signals: dict[str, TradingSignal] = {}
regime_mults: dict[str, float] = {}

for pair, (strat, alloc_pct, role) in trendhold_strategies.items():
    try:
        df = live_fetcher._to_dataframe(pair)
        if len(df) < warmup_bars:
            continue

        features_df = live_fetcher.get_feature_matrix(pair)
        if features_df.empty:
            continue

        features_cache[pair] = features_df.iloc[-1].to_dict()
        regime = regime_detector.update(df)
        regime_mults[pair] = regime.size_multiplier

        signal = strat.generate_signal(pair, features_df)

        if role == "filler":
            # BTC filler: always buy if not holding, cycle after filler_cycle_bars
            pos = order_manager.get_all_positions().get(pair)
            if pos is None:
                # Force a BUY signal for filler
                signal = TradingSignal(
                    pair=pair,
                    direction=SignalDirection.BUY,
                    confidence=0.55,
                    strategy_name="trendhold_filler",
                )
            else:
                # Check if held long enough to cycle
                bars_held = _bars_since_entry(pos, current_15m_epoch)
                if bars_held >= filler_cycle_bars:
                    signal = TradingSignal(
                        pair=pair,
                        direction=SignalDirection.SELL,
                        confidence=0.55,
                        strategy_name="trendhold_filler_cycle",
                    )
                else:
                    signal = TradingSignal(
                        pair=pair,
                        direction=SignalDirection.HOLD,
                        confidence=0.5,
                        strategy_name="trendhold_filler",
                    )

        signals[pair] = signal

    except Exception as exc:
        logger.error("Step 4A: error processing %s: %s", pair, exc, exc_info=True)
```

### 4. Add a helper function for filler bar counting

Add near the top of `main.py`:

```python
def _bars_since_entry(position: dict, current_epoch: int) -> int:
    """Count 15-min bars since position entry."""
    entry_epoch = position.get("entry_epoch", current_epoch)
    return current_epoch - entry_epoch
```

This requires that `order_manager` stores `entry_epoch` when opening positions.
If it doesn't, an alternative is to track filler entry time in `loop_state`:

```python
loop_state.setdefault("filler_entry_epoch", 0)
```

And update it when the filler BUY executes.

### 5. Update position sizing in Phase D (around line 496)

The current code uses `portfolio_allocator.compute_weights()` for sizing.
For TrendHold, override the weight with the fixed allocation:

```python
# In Phase D, when sizing a BUY:
_, alloc_pct, role = trendhold_strategies.get(pair, (None, 0.10, "hold"))
# Pass alloc_pct as portfolio_weight to risk_manager.size_new_position()
```

### 6. Remove dead code

After the switch, remove:
- `pairs_ml_strategy` variable and all Phase B code (lines 470-491)
- `mean_reversion_strategy` initialization
- The `from bot.strategy.mean_reversion import MeanReversionStrategy` import
  (only if you're sure you won't fall back to it)

---

## Verification Checklist

After making changes, verify:

- [ ] `python -c "import main"` succeeds (no import errors)
- [ ] All three models load: `xgb_btc_15m.pkl`, `xgb_eth_15m.pkl`, `xgb_sol_15m.pkl`
- [ ] Signals fire only for BTC/USD, ETH/USD, SOL/USD
- [ ] BTC filler cycles every ~24h (check logs for SELL/BUY pairs on BTC)
- [ ] ETH/SOL entries only fire when XGB P(BUY) >= 0.55
- [ ] ETH/SOL exits fire when XGB P(BUY) <= 0.10 OR stop triggers
- [ ] Hard stop is 12%, ATR trail is 25x, CB halts at 15% drawdown
- [ ] Active days target: at least 8/10 (BTC filler guarantees this)

---

## Rollback

To revert to the old strategy, `git checkout` the previous commit's versions of
`main.py` and `bot/config/config.yaml`. The old models and strategy files are
still in git history.

```bash
git log --oneline -5  # find the pre-TrendHold commit hash
git checkout <hash> -- main.py bot/config/config.yaml
```
