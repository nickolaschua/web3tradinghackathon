# Cross-Sectional Momentum Rotation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a cross-sectional momentum rotation strategy that ranks 20 coins every 4H, holds the top 4 with inverse-vol weighting, and uses buffer zones + 4-layer crash protection to optimize for the competition scoring formula (0.4×Sortino + 0.3×Sharpe + 0.3×Calmar).

**Architecture:** Portfolio-level strategy that operates on 4H-resampled data across all coins simultaneously. Unlike the existing per-pair signal cascade, this strategy makes holistic allocation decisions — which coins to hold, how much capital in each, when to cut exposure. Pure signal functions are separated from the strategy class for testability and backtest reuse.

**Tech Stack:** Python, pandas, numpy, scipy.stats (Spearman), quantstats

**Spec:** `docs/superpowers/specs/2026-03-22-momentum-rotation-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `bot/strategy/momentum_signals.py` | Pure functions: sharpe_mom, nearness, residual_mom, composite score, regime flag. No state, no I/O. Used by both backtest and live. |
| Create | `bot/strategy/momentum_rotation.py` | `MomentumRotationStrategy` class: buffer zones, crash protection layers, EMA smoothing state, rebalancing decisions. Takes all-coin data, returns target portfolio. |
| Create | `scripts/ic_analysis.py` | IC analysis: Spearman rank correlation per signal component. Determines weights (gates the backtest). |
| Create | `scripts/backtest_momentum_rotation.py` | 10-day sliding window backtest with full strategy vs BTC hold comparison, parameter sweeps. |
| Modify | `main.py` | Wire momentum rotation at 4H boundaries. Replaces per-pair signal cascade with portfolio-level rebalancing. |
| Modify | `bot/config/config.yaml` | Add momentum rotation parameters. |

---

## Task 1: Momentum signal functions

**Files:**
- Create: `bot/strategy/momentum_signals.py`

These are pure functions — no state, no I/O, no strategy logic. Compute each signal component on a DataFrame of 4H OHLCV data.

- [ ] **Step 1: Create the signal functions module**

```python
"""
Pure signal functions for cross-sectional momentum rotation.

All functions operate on 4H OHLCV DataFrames. No state, no I/O.
Used by both backtest and live strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def resample_to_4h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample 15M OHLCV to 4H. Input must have DatetimeIndex and lowercase columns."""
    return df_15m.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["close"])


def sharpe_momentum(closes: pd.Series, lookback: int, skip: int = 1) -> float:
    """
    Volatility-adjusted momentum score.

    Args:
        closes: 4H close prices (DatetimeIndex).
        lookback: Number of 4H bars for return calculation (12 = 48H, 42 = 168H).
        skip: Bars to skip from the end (avoid sub-daily reversal). Default 1.

    Returns:
        Sharpe momentum score (return / vol). Higher = stronger momentum.
    """
    if len(closes) < lookback + skip:
        return 0.0
    end = len(closes) - skip
    start = end - lookback
    if start < 0:
        return 0.0
    log_ret = np.log(closes.iloc[end] / closes.iloc[start])
    log_rets = np.log(closes.iloc[start:end] / closes.iloc[start:end].shift(1)).dropna()
    vol = log_rets.std()
    if vol < 1e-10:
        return 0.0
    return float(log_ret / vol)


def nearness_to_high(closes: pd.Series, window: int = 180) -> float:
    """
    Price anchoring signal: current price / rolling max over window.

    Args:
        closes: 4H close prices.
        window: Lookback for rolling max (180 bars ≈ 30 days at 4H).

    Returns:
        Nearness ratio 0-1. Higher = closer to recent high = bullish.
    """
    if len(closes) < window:
        return 0.5  # neutral
    recent_high = closes.iloc[-window:].max()
    if recent_high <= 0:
        return 0.5
    return float(closes.iloc[-1] / recent_high)


def residual_momentum(
    coin_returns: pd.Series,
    btc_returns: pd.Series,
    beta_window: int = 42,
    signal_window: int = 12,
) -> float:
    """
    BTC-beta-stripped momentum.

    Rolling OLS of coin returns on BTC returns over beta_window.
    Cumulate residuals over last signal_window bars, divide by residual vol.

    Args:
        coin_returns: 4H log returns for this coin.
        btc_returns: 4H log returns for BTC (aligned index).
        beta_window: Bars for rolling OLS beta estimation (42 = 7 days).
        signal_window: Bars to accumulate residuals over (12 = 48H).

    Returns:
        Residual momentum score. Higher = coin outperforming vs BTC.
    """
    if len(coin_returns) < beta_window + signal_window:
        return 0.0

    # Rolling OLS: coin = alpha + beta * btc + epsilon
    # Use the most recent beta_window bars for the regression
    y = coin_returns.iloc[-beta_window:].values
    x = btc_returns.iloc[-beta_window:].values

    # Handle NaN
    valid = ~(np.isnan(y) | np.isnan(x))
    if valid.sum() < 10:
        return 0.0

    y_v, x_v = y[valid], x[valid]
    x_mean = x_v.mean()
    y_mean = y_v.mean()
    cov = ((x_v - x_mean) * (y_v - y_mean)).mean()
    var = ((x_v - x_mean) ** 2).mean()
    beta = cov / (var + 1e-10)
    alpha = y_mean - beta * x_mean

    # Compute residuals for the last signal_window bars
    recent_y = coin_returns.iloc[-signal_window:].values
    recent_x = btc_returns.iloc[-signal_window:].values
    residuals = recent_y - alpha - beta * recent_x

    valid_r = ~np.isnan(residuals)
    if valid_r.sum() < 3:
        return 0.0

    residuals = residuals[valid_r]
    resid_vol = residuals.std()
    if resid_vol < 1e-10:
        return 0.0

    return float(residuals.sum() / resid_vol)


def compute_composite_score(
    closes: pd.Series,
    btc_closes: pd.Series,
    weights: dict[str, float],
) -> float:
    """
    Compute the composite momentum score for one coin.

    Args:
        closes: 4H close prices for this coin.
        btc_closes: 4H close prices for BTC (for residual momentum).
        weights: Dict with keys 'sharpe_48h', 'nearness', 'sharpe_168h', 'residual'.
                 Values are IC-derived weights (sum to 1). Missing keys treated as 0.

    Returns:
        Composite score (higher = stronger momentum signal).
    """
    components = {}

    if weights.get("sharpe_48h", 0) > 0:
        components["sharpe_48h"] = sharpe_momentum(closes, lookback=12, skip=1)

    if weights.get("nearness", 0) > 0:
        components["nearness"] = nearness_to_high(closes, window=180)

    if weights.get("sharpe_168h", 0) > 0:
        components["sharpe_168h"] = sharpe_momentum(closes, lookback=42, skip=1)

    if weights.get("residual", 0) > 0:
        coin_ret = np.log(closes / closes.shift(1)).dropna()
        btc_ret = np.log(btc_closes / btc_closes.shift(1)).dropna()
        # Align
        common = coin_ret.index.intersection(btc_ret.index)
        if len(common) > 60:
            components["residual"] = residual_momentum(
                coin_ret.reindex(common), btc_ret.reindex(common)
            )
        else:
            components["residual"] = 0.0

    score = sum(weights.get(k, 0) * v for k, v in components.items())
    return score


def compute_regime_flag(
    btc_closes_4h: pd.Series,
    median_vol: float | None = None,
) -> str:
    """
    Compute market regime from BTC 30-day return and 7-day volatility.

    Args:
        btc_closes_4h: BTC 4H close prices (at least 180 bars).
        median_vol: Precomputed median vol for threshold. If None, uses 0.03.

    Returns:
        One of: "HIGH_VOL_TREND", "LOW_VOL_TREND", "BEARISH", "SIDEWAYS"
    """
    if len(btc_closes_4h) < 180:
        return "SIDEWAYS"

    ret_30d = np.log(btc_closes_4h.iloc[-1] / btc_closes_4h.iloc[-180])
    rets_7d = np.log(btc_closes_4h.iloc[-42:] / btc_closes_4h.iloc[-42:].shift(1)).dropna()
    vol_7d = rets_7d.std()

    if median_vol is None:
        median_vol = 0.03  # ~48% annualized, reasonable crypto default

    if ret_30d > 0.05 and vol_7d > median_vol:
        return "HIGH_VOL_TREND"
    elif ret_30d > 0 and vol_7d <= median_vol:
        return "LOW_VOL_TREND"
    elif ret_30d < -0.05:
        return "BEARISH"
    else:
        return "SIDEWAYS"


def adjust_weights_for_regime(
    base_weights: dict[str, float],
    regime: str,
    boost_factor: float = 1.3,
) -> dict[str, float]:
    """
    Apply regime-based weight adjustment (multiplicative + re-normalize).

    Args:
        base_weights: IC-derived weights (sum to 1).
        regime: Output of compute_regime_flag().
        boost_factor: Multiplier for the favored component (default 1.3).

    Returns:
        Adjusted weights (sum to 1).
    """
    w = dict(base_weights)

    boost_key = {
        "HIGH_VOL_TREND": "sharpe_48h",
        "LOW_VOL_TREND": "nearness",
        "BEARISH": None,   # no boost, crash protection handles this
        "SIDEWAYS": None,  # use base weights
    }.get(regime)

    if boost_key and boost_key in w:
        w[boost_key] *= boost_factor

    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    return w


def compute_ic(
    scores: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """
    Spearman rank IC between scores and subsequent returns.

    Args:
        scores: Signal scores (one per coin per timestamp).
        forward_returns: Realized returns over the next period.

    Returns:
        Spearman correlation (IC). Positive = signal has predictive power.
    """
    valid = scores.notna() & forward_returns.notna()
    if valid.sum() < 10:
        return 0.0
    corr, _ = stats.spearmanr(scores[valid], forward_returns[valid])
    return float(corr) if not np.isnan(corr) else 0.0
```

- [ ] **Step 2: Smoke test the signal functions**

```bash
cd C:/Users/nicko/Desktop/web3tradinghackathon/web3tradinghackathon
python -c "
import pandas as pd, numpy as np
from bot.strategy.momentum_signals import (
    resample_to_4h, sharpe_momentum, nearness_to_high,
    residual_momentum, compute_composite_score, compute_regime_flag,
    adjust_weights_for_regime, compute_ic
)
btc = pd.read_parquet('data/BTCUSDT_15m.parquet')
eth = pd.read_parquet('data/ETHUSDT_15m.parquet')
btc.columns = btc.columns.str.lower()
eth.columns = eth.columns.str.lower()
btc.index = pd.to_datetime(btc.index)
eth.index = pd.to_datetime(eth.index)
btc_4h = resample_to_4h(btc)
eth_4h = resample_to_4h(eth)
print(f'BTC 4H bars: {len(btc_4h)}')
print(f'sharpe_mom_48h BTC: {sharpe_momentum(btc_4h[\"close\"], 12):.4f}')
print(f'sharpe_mom_168h BTC: {sharpe_momentum(btc_4h[\"close\"], 42):.4f}')
print(f'nearness BTC: {nearness_to_high(btc_4h[\"close\"]):.4f}')
btc_ret = np.log(btc_4h['close']/btc_4h['close'].shift(1)).dropna()
eth_ret = np.log(eth_4h['close']/eth_4h['close'].shift(1)).dropna()
common = btc_ret.index.intersection(eth_ret.index)
print(f'residual_mom ETH: {residual_momentum(eth_ret.reindex(common), btc_ret.reindex(common)):.4f}')
w = {'sharpe_48h': 0.30, 'nearness': 0.25, 'sharpe_168h': 0.25, 'residual': 0.20}
print(f'composite ETH: {compute_composite_score(eth_4h[\"close\"], btc_4h[\"close\"], w):.4f}')
print(f'regime: {compute_regime_flag(btc_4h[\"close\"])}')
print('All OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add bot/strategy/momentum_signals.py
git commit -m "feat: add pure momentum signal functions (sharpe, nearness, residual, composite)"
```

---

## Task 2: IC analysis script

**Files:**
- Create: `scripts/ic_analysis.py`

This script determines signal weights by measuring each component's predictive power. **Gates Task 3** — components with IC ≤ 0 are dropped.

- [ ] **Step 1: Create the IC analysis script**

```python
#!/usr/bin/env python3
"""
IC analysis for momentum signal components.

Computes Spearman rank IC for each signal component across 20 coins
on formation period (2024-01 to 2025-06). Determines which components
survive and their weights for the composite score.

Usage:
  python scripts/ic_analysis.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.strategy.momentum_signals import (
    resample_to_4h, sharpe_momentum, nearness_to_high, residual_momentum,
)

COINS = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT",
         "LTC","UNI","NEAR","SUI","APT","PEPE","ARB","SHIB","FIL","HBAR"]

FORMATION_START = pd.Timestamp("2024-01-01", tz="UTC")
FORMATION_END = pd.Timestamp("2025-06-01", tz="UTC")


def main():
    data_dir = Path("data")

    # Load and resample all coins to 4H
    print("Loading and resampling to 4H...", flush=True)
    coin_4h = {}
    for coin in COINS:
        df = pd.read_parquet(data_dir / f"{coin}USDT_15m.parquet")
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()
        coin_4h[coin] = resample_to_4h(df)

    btc_4h = coin_4h["BTC"]
    btc_ret = np.log(btc_4h["close"] / btc_4h["close"].shift(1)).dropna()

    # For each rebalancing point in formation period, compute scores and forward returns
    # Rebalancing every 4H = every bar in the 4H data
    formation_idx = btc_4h.index[
        (btc_4h.index >= FORMATION_START) & (btc_4h.index < FORMATION_END)
    ]

    print(f"Formation period: {FORMATION_START.date()} to {FORMATION_END.date()}")
    print(f"Rebalancing points: {len(formation_idx)}", flush=True)

    # Collect cross-sectional scores at each rebalancing point
    records = []
    for i, ts in enumerate(formation_idx):
        if i % 100 == 0:
            print(f"  Processing bar {i}/{len(formation_idx)}...", flush=True)

        for coin in COINS:
            c4h = coin_4h[coin]
            if ts not in c4h.index:
                continue

            loc = c4h.index.get_loc(ts)
            if loc < 200:  # need enough history
                continue

            closes = c4h["close"].iloc[:loc + 1]

            # Forward 4H return (the thing we're predicting)
            if loc + 1 >= len(c4h):
                continue
            fwd_ret = np.log(c4h["close"].iloc[loc + 1] / c4h["close"].iloc[loc])

            # Compute each signal component
            sm_48h = sharpe_momentum(closes, lookback=12, skip=1)
            sm_168h = sharpe_momentum(closes, lookback=42, skip=1)
            near = nearness_to_high(closes, window=180)

            # Residual momentum
            coin_ret = np.log(c4h["close"].iloc[:loc + 1] / c4h["close"].iloc[:loc + 1].shift(1)).dropna()
            common = coin_ret.index.intersection(btc_ret.index)
            common = common[common <= ts]
            if len(common) > 60:
                res_mom = residual_momentum(
                    coin_ret.reindex(common), btc_ret.reindex(common)
                )
            else:
                res_mom = np.nan

            records.append({
                "ts": ts, "coin": coin,
                "sharpe_48h": sm_48h, "nearness": near,
                "sharpe_168h": sm_168h, "residual": res_mom,
                "fwd_return": fwd_ret,
            })

    df = pd.DataFrame(records)
    print(f"\nTotal observations: {len(df):,}", flush=True)

    # Compute IC per component (cross-sectional Spearman at each timestamp, then average)
    components = ["sharpe_48h", "nearness", "sharpe_168h", "residual"]

    print("\n" + "=" * 70)
    print("IC ANALYSIS (Spearman rank correlation with forward 4H return)")
    print("=" * 70)

    ics = {}
    for comp in components:
        # Per-timestamp cross-sectional IC
        per_ts_ic = []
        for ts, group in df.groupby("ts"):
            valid = group[[comp, "fwd_return"]].dropna()
            if len(valid) < 5:
                continue
            corr, _ = stats.spearmanr(valid[comp], valid["fwd_return"])
            if not np.isnan(corr):
                per_ts_ic.append(corr)

        if per_ts_ic:
            mean_ic = np.mean(per_ts_ic)
            std_ic = np.std(per_ts_ic)
            t_stat = mean_ic / (std_ic / np.sqrt(len(per_ts_ic))) if std_ic > 0 else 0
            pct_positive = sum(1 for x in per_ts_ic if x > 0) / len(per_ts_ic) * 100
        else:
            mean_ic = std_ic = t_stat = 0.0
            pct_positive = 0.0

        ics[comp] = mean_ic
        status = "KEEP" if mean_ic > 0 else "DROP"
        print(f"  {comp:>15}: IC={mean_ic:>+.4f}  std={std_ic:.4f}  "
              f"t={t_stat:>5.2f}  %pos={pct_positive:>4.0f}%  [{status}]", flush=True)

    # Derive weights from positive-IC components
    positive = {k: v for k, v in ics.items() if v > 0}
    if not positive:
        print("\n*** ALL COMPONENTS HAVE IC <= 0. Strategy has no edge. ***")
        return

    total_ic = sum(positive.values())
    weights = {k: round(v / total_ic, 3) for k, v in positive.items()}

    print(f"\n  Surviving components: {list(positive.keys())}")
    print(f"  IC-derived weights: {weights}")
    print(f"  Dropped components: {[k for k in ics if k not in positive]}")

    # Output as Python dict for copy-paste into config
    print(f"\n  # Copy into config or strategy:")
    print(f"  SIGNAL_WEIGHTS = {weights}")

    print("\n" + "=" * 70)
    print("Done. Use these weights in the backtest and strategy.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the IC analysis**

```bash
cd C:/Users/nicko/Desktop/web3tradinghackathon/web3tradinghackathon
python scripts/ic_analysis.py
```

Expected output: IC values per component, weights for surviving components. Record the output — it gates Task 3.

**Decision gate:** If all ICs are ≤ 0, the strategy has no edge on this data and we need to reconsider. If 1-2 components are dropped, proceed with the survivors.

- [ ] **Step 3: Commit**

```bash
git add scripts/ic_analysis.py
git commit -m "feat: add IC analysis script for momentum signal weight determination"
```

---

## Task 3: Backtest script with 10-day window analysis

**Files:**
- Create: `scripts/backtest_momentum_rotation.py`

This is the primary validation tool. Simulates the full strategy across sliding 10-day windows.

- [ ] **Step 1: Create the backtest script**

```python
#!/usr/bin/env python3
"""
Cross-sectional momentum rotation backtest with 10-day window analysis.

Simulates: rank 20 coins → hold top 4 → inverse-vol weight → buffer zones →
4-layer crash protection → rebalance every 4H.

Reports: distribution of 10-day Sortino/Sharpe/Calmar vs BTC buy-and-hold.

Usage:
  python scripts/backtest_momentum_rotation.py
  python scripts/backtest_momentum_rotation.py --sweep-alpha
  python scripts/backtest_momentum_rotation.py --sweep-buffer
"""
import argparse
import sys
import datetime
import numpy as np
import pandas as pd
import quantstats as qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.strategy.momentum_signals import (
    resample_to_4h, sharpe_momentum, nearness_to_high,
    residual_momentum, compute_regime_flag, adjust_weights_for_regime,
)

PERIODS_4H = 365.25 * 6  # annualization for 4H data
COINS = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT",
         "LTC","UNI","NEAR","SUI","APT","PEPE","ARB","SHIB","FIL","HBAR"]
WINDOW_BARS = 60  # 10 days at 4H = 60 bars


def load_4h_data():
    """Load all 20 coins, resample to 4H, return dict."""
    print("Loading data...", flush=True)
    coin_4h = {}
    for coin in COINS:
        df = pd.read_parquet(f"data/{coin}USDT_15m.parquet")
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()
        coin_4h[coin] = resample_to_4h(df)
    return coin_4h


def compute_all_scores(coin_4h, idx, weights, btc_ret):
    """Compute composite scores for all coins at a given bar index."""
    scores = {}
    vols = {}
    for coin in COINS:
        c4h = coin_4h[coin]
        if idx not in c4h.index:
            continue
        loc = c4h.index.get_loc(idx)
        if loc < 200:
            continue

        closes = c4h["close"].iloc[:loc + 1]
        score = 0.0

        if weights.get("sharpe_48h", 0) > 0:
            score += weights["sharpe_48h"] * sharpe_momentum(closes, 12, 1)
        if weights.get("nearness", 0) > 0:
            score += weights["nearness"] * nearness_to_high(closes, 180)
        if weights.get("sharpe_168h", 0) > 0:
            score += weights["sharpe_168h"] * sharpe_momentum(closes, 42, 1)
        if weights.get("residual", 0) > 0:
            coin_ret = np.log(c4h["close"].iloc[:loc+1] / c4h["close"].iloc[:loc+1].shift(1)).dropna()
            common = coin_ret.index.intersection(btc_ret.index)
            common = common[common <= idx]
            if len(common) > 60:
                score += weights["residual"] * residual_momentum(
                    coin_ret.reindex(common), btc_ret.reindex(common)
                )

        # Volatility for inverse-vol weighting
        rets = np.log(closes / closes.shift(1)).dropna()
        vol = rets.iloc[-42:].std() if len(rets) >= 42 else rets.std()

        scores[coin] = score
        vols[coin] = max(vol, 1e-10)

    return scores, vols


def run_momentum_backtest(
    coin_4h, weights,
    start_date="2025-06-01",
    initial=1_000_000.0,
    n_holdings=4,
    sell_rank=6,
    ema_alpha=0.4,
    fee_bps=5,
    dd_flat=0.12,
    dd_half=0.08,
    dd_quarter=0.05,
):
    """Run full momentum rotation backtest. Returns (returns_series, stats_dict)."""
    btc_4h = coin_4h["BTC"]
    btc_ret = np.log(btc_4h["close"] / btc_4h["close"].shift(1)).dropna()

    start_ts = pd.Timestamp(start_date, tz="UTC")
    timeline = btc_4h.index[btc_4h.index >= start_ts]
    n = len(timeline)
    fee_rate = fee_bps / 10_000

    # State
    free = initial
    hwm = initial
    positions = {}  # coin -> {"units": float, "value": float}
    smoothed_scores = {}  # coin -> float (EMA smoothed)
    port_vals = np.zeros(n)
    port_vals[0] = initial
    rets = np.zeros(n)
    n_trades = 0
    active_dates = set()

    for i, ts in enumerate(timeline):
        # Mark to market
        port_val = free
        for coin, pos in positions.items():
            if ts in coin_4h[coin].index:
                price = coin_4h[coin].loc[ts, "close"]
                port_val += pos["units"] * price
        hwm = max(hwm, port_val)

        # Compute scores
        raw_scores, vols = compute_all_scores(coin_4h, ts, weights, btc_ret)

        # EMA smooth
        for coin, score in raw_scores.items():
            prev = smoothed_scores.get(coin, score)
            smoothed_scores[coin] = ema_alpha * score + (1 - ema_alpha) * prev

        # Rank by smoothed score
        ranked = sorted(smoothed_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_coins = [c for c, _ in ranked if c in raw_scores]

        # Crash protection
        drawdown = (hwm - port_val) / hwm if hwm > 0 else 0
        if drawdown > dd_flat:
            dd_scalar = 0.0
        elif drawdown > dd_half:
            dd_scalar = 0.50
        elif drawdown > dd_quarter:
            dd_scalar = 0.75
        else:
            dd_scalar = 1.0

        # BTC TSMOM
        btc_7d = btc_ret.loc[btc_ret.index <= ts].iloc[-42:]
        btc_ema_ret = btc_7d.ewm(span=7).mean().iloc[-1] if len(btc_7d) > 7 else 0
        if btc_ema_ret < -0.05:
            tsmom = 0.0
        elif btc_ema_ret < 0:
            tsmom = 0.5
        else:
            tsmom = 1.0

        # Dispersion
        period_rets = {}
        for coin in COINS:
            if ts in coin_4h[coin].index:
                loc = coin_4h[coin].index.get_loc(ts)
                if loc > 0:
                    period_rets[coin] = np.log(
                        coin_4h[coin]["close"].iloc[loc] /
                        coin_4h[coin]["close"].iloc[loc - 1]
                    )
        dispersion_scalar = 1.0
        if len(period_rets) >= 10:
            disp = np.std(list(period_rets.values()))
            # Simple threshold: if dispersion < 0.005 (0.5%), reduce
            if disp < 0.005:
                dispersion_scalar = 0.5

        exposure = dd_scalar * min(tsmom, dispersion_scalar)

        # Target portfolio
        if exposure == 0:
            # Go flat
            for coin in list(positions.keys()):
                if ts in coin_4h[coin].index:
                    price = coin_4h[coin].loc[ts, "close"]
                    free += positions[coin]["units"] * price * (1 - fee_rate)
                    n_trades += 1
                    active_dates.add(ts.date())
            positions.clear()
        else:
            # Determine target holdings using buffer zones
            current_held = set(positions.keys())
            target_coins = []

            for coin in ranked_coins:
                if len(target_coins) >= n_holdings:
                    break
                rank = ranked_coins.index(coin) + 1
                if coin in current_held:
                    if rank <= sell_rank:  # hold zone
                        target_coins.append(coin)
                    # else: sell (below sell_rank)
                else:
                    if rank <= n_holdings:  # buy zone
                        target_coins.append(coin)

            # Fill remaining slots from top-ranked non-held
            if len(target_coins) < n_holdings:
                for coin in ranked_coins:
                    if coin not in target_coins and len(target_coins) < n_holdings:
                        target_coins.append(coin)

            # Inverse-vol weights
            target_vols = {c: vols.get(c, 0.01) for c in target_coins}
            inv_vol = {c: 1.0 / v for c, v in target_vols.items()}
            total_inv = sum(inv_vol.values())
            target_weights = {c: (inv_vol[c] / total_inv) * exposure for c in target_coins}

            # Sell coins no longer in target
            for coin in list(positions.keys()):
                if coin not in target_coins:
                    if ts in coin_4h[coin].index:
                        price = coin_4h[coin].loc[ts, "close"]
                        free += positions[coin]["units"] * price * (1 - fee_rate)
                        n_trades += 1
                        active_dates.add(ts.date())
                    del positions[coin]

            # Buy/adjust target coins
            total_val = free + sum(
                positions[c]["units"] * coin_4h[c].loc[ts, "close"]
                for c in positions if ts in coin_4h[c].index
            )

            for coin in target_coins:
                target_usd = total_val * target_weights[coin]
                if ts not in coin_4h[coin].index:
                    continue
                price = coin_4h[coin].loc[ts, "close"]

                current_usd = 0
                if coin in positions:
                    current_usd = positions[coin]["units"] * price

                diff = target_usd - current_usd
                if abs(diff) < max(5000, total_val * 0.005):
                    continue  # min trade threshold

                if diff > 0:  # buy more
                    buy_usd = min(diff, free * 0.95)
                    if buy_usd > 10:
                        units = buy_usd / price
                        cost = buy_usd * (1 + fee_rate)
                        free -= cost
                        if coin in positions:
                            positions[coin]["units"] += units
                        else:
                            positions[coin] = {"units": units}
                        n_trades += 1
                        active_dates.add(ts.date())
                elif diff < 0:  # sell some
                    sell_usd = min(-diff, current_usd)
                    sell_units = sell_usd / price
                    positions[coin]["units"] -= sell_units
                    free += sell_usd * (1 - fee_rate)
                    if positions[coin]["units"] < 0.0001:
                        del positions[coin]
                    n_trades += 1
                    active_dates.add(ts.date())

        # Record
        port_val = free
        for coin, pos in positions.items():
            if ts in coin_4h[coin].index:
                port_val += pos["units"] * coin_4h[coin].loc[ts, "close"]
        port_vals[i] = port_val
        if i > 0:
            rets[i] = port_vals[i] / port_vals[i - 1] - 1.0

    ret_s = pd.Series(rets, index=timeline)
    total_days = (timeline[-1] - timeline[0]).days

    return ret_s, {
        "sharpe": float(qs.stats.sharpe(ret_s, periods=PERIODS_4H)),
        "sortino": float(qs.stats.sortino(ret_s, periods=PERIODS_4H)),
        "maxdd": float(qs.stats.max_drawdown(ret_s)) * 100,
        "ret": (port_vals[-1] - initial) / initial * 100,
        "trades": n_trades,
        "active_pct": len(active_dates) / max(total_days, 1) * 100,
    }


def run_10d_windows(ret_s, btc_4h, start_date="2025-06-01"):
    """Compute 10-day window statistics and compare to BTC hold."""
    start_ts = pd.Timestamp(start_date, tz="UTC")
    btc_rets = np.log(btc_4h["close"] / btc_4h["close"].shift(1)).dropna()
    btc_rets = btc_rets[btc_rets.index >= start_ts]

    strat_windows = []
    btc_windows = []

    for start in range(0, len(ret_s) - WINDOW_BARS, 6):  # step 1 day = 6 bars
        end = start + WINDOW_BARS
        w_ret = ret_s.iloc[start:end]
        std = w_ret.std()
        if std < 1e-10:
            continue

        sharpe = float(qs.stats.sharpe(w_ret, periods=PERIODS_4H))
        sortino = float(qs.stats.sortino(w_ret, periods=PERIODS_4H))
        total_ret = (1 + w_ret).prod() - 1
        maxdd = float(qs.stats.max_drawdown(w_ret)) * 100
        calmar = abs(total_ret / (maxdd / 100)) if maxdd != 0 else 0
        score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

        strat_windows.append({
            "ret": total_ret * 100, "sharpe": sharpe, "sortino": sortino,
            "calmar": calmar, "maxdd": maxdd, "score": score,
        })

        # BTC hold for same window
        b_idx = btc_rets.index[start:end] if start + end <= len(btc_rets) else []
        if len(b_idx) >= WINDOW_BARS:
            b_ret = btc_rets.iloc[start:end]
            b_sharpe = float(qs.stats.sharpe(b_ret, periods=PERIODS_4H))
            b_sortino = float(qs.stats.sortino(b_ret, periods=PERIODS_4H))
            b_total = (1 + b_ret).prod() - 1
            b_maxdd = float(qs.stats.max_drawdown(b_ret)) * 100
            b_calmar = abs(b_total / (b_maxdd / 100)) if b_maxdd != 0 else 0
            b_score = 0.4 * b_sortino + 0.3 * b_sharpe + 0.3 * b_calmar
            btc_windows.append({"score": b_score, "ret": b_total * 100})

    return strat_windows, btc_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="JSON dict of IC-derived weights, e.g. '{\"sharpe_48h\":0.4,\"nearness\":0.3,\"sharpe_168h\":0.3}'")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--sweep-alpha", action="store_true")
    parser.add_argument("--sweep-buffer", action="store_true")
    args = parser.parse_args()

    coin_4h = load_4h_data()

    # Default weights (placeholder — use IC-derived)
    import json
    if args.weights:
        weights = json.loads(args.weights)
    else:
        weights = {"sharpe_48h": 0.30, "nearness": 0.25, "sharpe_168h": 0.25, "residual": 0.20}

    print(f"Weights: {weights}", flush=True)

    if args.sweep_alpha:
        print("\n=== EMA Alpha Sweep ===", flush=True)
        for alpha in [0.2, 0.3, 0.4, 0.5, 0.6]:
            _, s = run_momentum_backtest(coin_4h, weights, args.start, ema_alpha=alpha)
            print(f"  alpha={alpha}: Sharpe={s['sharpe']:.3f} Sortino={s['sortino']:.3f} "
                  f"Trades={s['trades']} Ret={s['ret']:+.1f}% DD={s['maxdd']:.1f}%", flush=True)
        return

    if args.sweep_buffer:
        print("\n=== Buffer Zone Sweep ===", flush=True)
        for sell in [5, 6, 7, 8]:
            _, s = run_momentum_backtest(coin_4h, weights, args.start, sell_rank=sell)
            print(f"  sell_rank={sell}: Sharpe={s['sharpe']:.3f} Sortino={s['sortino']:.3f} "
                  f"Trades={s['trades']} Ret={s['ret']:+.1f}% DD={s['maxdd']:.1f}%", flush=True)
        return

    # Main run
    print("\nRunning momentum rotation backtest...", flush=True)
    ret_s, stats = run_momentum_backtest(coin_4h, weights, args.start)

    print("\n" + "=" * 70)
    print(f"  MOMENTUM ROTATION BACKTEST ({args.start} onwards)")
    print("=" * 70)
    print(f"  Sharpe:  {stats['sharpe']:.3f}")
    print(f"  Sortino: {stats['sortino']:.3f}")
    print(f"  MaxDD:   {stats['maxdd']:.1f}%")
    print(f"  Return:  {stats['ret']:+.1f}%")
    print(f"  Trades:  {stats['trades']}")
    print(f"  Active:  {stats['active_pct']:.0f}%")

    # 10-day window analysis
    print("\n--- 10-Day Window Analysis ---", flush=True)
    strat_w, btc_w = run_10d_windows(ret_s, coin_4h["BTC"], args.start)

    if strat_w:
        scores = [w["score"] for w in strat_w]
        rets_list = [w["ret"] for w in strat_w]
        print(f"  Windows analyzed: {len(strat_w)}")
        print(f"  Median score:  {np.median(scores):.3f}")
        print(f"  Median return: {np.median(rets_list):+.2f}%")
        print(f"  P5 return:     {np.percentile(rets_list, 5):+.2f}%")
        print(f"  P95 return:    {np.percentile(rets_list, 95):+.2f}%")
        print(f"  % positive:    {sum(1 for r in rets_list if r > 0)/len(rets_list)*100:.0f}%")

        if btc_w:
            btc_scores = [w["score"] for w in btc_w]
            beat_btc = sum(1 for s, b in zip(scores, btc_scores) if s > b) / len(scores) * 100
            print(f"  % beat BTC:    {beat_btc:.0f}%")
            print(f"  BTC med score: {np.median(btc_scores):.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the backtest with IC-derived weights (from Task 2)**

```bash
cd C:/Users/nicko/Desktop/web3tradinghackathon/web3tradinghackathon
python scripts/backtest_momentum_rotation.py --weights '{"sharpe_48h": 0.30, "nearness": 0.25, "sharpe_168h": 0.25, "residual": 0.20}'
```

Replace the weights JSON with the actual IC-derived weights from Task 2.

- [ ] **Step 3: Run parameter sweeps**

```bash
python scripts/backtest_momentum_rotation.py --sweep-alpha
python scripts/backtest_momentum_rotation.py --sweep-buffer
```

- [ ] **Step 4: Commit**

```bash
git add scripts/backtest_momentum_rotation.py
git commit -m "feat: add momentum rotation backtest with 10-day window analysis"
```

---

## Task 4: Validate against gates

**Files:** None (analysis only — run backtests from Task 3)

- [ ] **Step 1: Check primary gate — full strategy vs BTC hold**

Review output from Task 3 Step 2. Must pass:
- Median competition score > BTC hold median score
- P5 10-day return > -8%
- \>60% of windows beat BTC

- [ ] **Step 2: Check secondary gates**

Review sweep outputs from Task 3 Step 3:
- EMA alpha: verify 0.4 is optimal or pick best
- Buffer zone: verify sell_rank=6 is optimal

- [ ] **Step 3: Document results and chosen parameters**

Record the final configuration:
- Signal weights (from IC analysis)
- EMA alpha (from sweep)
- Buffer sell rank (from sweep)
- Overall Sharpe/Sortino/active days

---

## Task 5: MomentumRotationStrategy class

**Files:**
- Create: `bot/strategy/momentum_rotation.py`

**Only proceed if Task 4 validation gates pass.**

This class manages the stateful parts: EMA smoothing, buffer zone tracking, crash protection layers. It calls the pure signal functions from `momentum_signals.py`.

- [ ] **Step 1: Create the strategy class**

The strategy has a different interface from BaseStrategy — it operates at the portfolio level, not per-pair. It takes all coin data and returns target allocations.

```python
"""
Cross-sectional momentum rotation strategy.

Portfolio-level strategy: ranks 20 coins, holds top 4 with inverse-vol
weighting, buffer zones for turnover control, 4-layer crash protection.

Unlike per-pair strategies (BaseStrategy), this operates on all coins
simultaneously and returns target portfolio allocations.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from bot.strategy.momentum_signals import (
    resample_to_4h,
    compute_composite_score,
    compute_regime_flag,
    adjust_weights_for_regime,
)

logger = logging.getLogger(__name__)


@dataclass
class TargetAllocation:
    """Target portfolio allocation from momentum rotation."""
    coin: str
    weight: float        # fraction of portfolio (0-1)
    target_usd: float    # dollar amount
    action: str          # "BUY", "SELL", "HOLD", "ADJUST"


@dataclass
class MomentumRotationStrategy:
    """
    Cross-sectional momentum rotation with crash protection.

    Call rebalance() every 4H with all coin data to get target allocations.
    """
    weights: dict = field(default_factory=lambda: {
        "sharpe_48h": 0.30, "nearness": 0.25,
        "sharpe_168h": 0.25, "residual": 0.20,
    })
    n_holdings: int = 4
    sell_rank: int = 6
    ema_alpha: float = 0.4
    fee_bps: int = 5
    dd_flat: float = 0.12
    dd_half: float = 0.08
    dd_quarter: float = 0.05
    min_trade_usd: float = 5000.0
    min_trade_pct: float = 0.005

    # State
    _smoothed_scores: dict = field(default_factory=dict)
    _hwm: float = 0.0
    _regime: str = "SIDEWAYS"
    _last_regime_date: object = None

    def rebalance(
        self,
        coin_closes_4h: dict[str, pd.Series],
        btc_closes_4h: pd.Series,
        current_positions: dict[str, float],
        portfolio_value: float,
        free_cash: float,
        current_date: object = None,
    ) -> list[TargetAllocation]:
        """
        Compute target portfolio allocations.

        Args:
            coin_closes_4h: Dict of coin -> 4H close price Series.
            btc_closes_4h: BTC 4H close prices (for residual momentum).
            current_positions: Dict of coin -> current USD value held.
            portfolio_value: Total portfolio value (positions + cash).
            free_cash: Available cash for new positions.
            current_date: Current date for regime recomputation.

        Returns:
            List of TargetAllocation objects describing desired portfolio changes.
        """
        self._hwm = max(self._hwm, portfolio_value)

        # Recompute regime daily
        active_weights = dict(self.weights)
        if current_date and current_date != self._last_regime_date:
            self._regime = compute_regime_flag(btc_closes_4h)
            active_weights = adjust_weights_for_regime(active_weights, self._regime)
            self._last_regime_date = current_date
            logger.info("Regime: %s, adjusted weights: %s", self._regime, active_weights)

        # Compute and smooth scores
        raw_scores = {}
        vols = {}
        for coin, closes in coin_closes_4h.items():
            if len(closes) < 200:
                continue
            raw_scores[coin] = compute_composite_score(closes, btc_closes_4h, active_weights)
            rets = np.log(closes / closes.shift(1)).dropna()
            vols[coin] = max(rets.iloc[-42:].std() if len(rets) >= 42 else 0.01, 1e-10)

        for coin, score in raw_scores.items():
            prev = self._smoothed_scores.get(coin, score)
            self._smoothed_scores[coin] = self.ema_alpha * score + (1 - self.ema_alpha) * prev

        # Rank
        ranked = sorted(self._smoothed_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_coins = [c for c, _ in ranked if c in raw_scores]

        # Crash protection
        drawdown = (self._hwm - portfolio_value) / self._hwm if self._hwm > 0 else 0
        if drawdown > self.dd_flat:
            exposure = 0.0
        elif drawdown > self.dd_half:
            exposure = 0.50
        elif drawdown > self.dd_quarter:
            exposure = 0.75
        else:
            exposure = 1.0

        if exposure == 0:
            # Go flat
            return [
                TargetAllocation(coin=c, weight=0, target_usd=0, action="SELL")
                for c in current_positions if current_positions[c] > 0
            ]

        # Buffer zone selection
        current_held = set(c for c, v in current_positions.items() if v > 0)
        target_coins = []
        for coin in ranked_coins:
            if len(target_coins) >= self.n_holdings:
                break
            rank = ranked_coins.index(coin) + 1
            if coin in current_held and rank <= self.sell_rank:
                target_coins.append(coin)
            elif coin not in current_held and rank <= self.n_holdings:
                target_coins.append(coin)

        # Fill remaining
        for coin in ranked_coins:
            if coin not in target_coins and len(target_coins) < self.n_holdings:
                target_coins.append(coin)

        # Inverse-vol weights
        target_vols = {c: vols.get(c, 0.01) for c in target_coins}
        inv_vol = {c: 1.0 / v for c, v in target_vols.items()}
        total_inv = sum(inv_vol.values()) or 1.0
        target_weights = {c: (inv_vol[c] / total_inv) * exposure for c in target_coins}

        # Build allocations
        allocations = []
        for coin in set(list(current_positions.keys()) + target_coins):
            target_usd = portfolio_value * target_weights.get(coin, 0)
            current_usd = current_positions.get(coin, 0)
            diff = target_usd - current_usd

            if coin not in target_coins and current_usd > 0:
                allocations.append(TargetAllocation(coin, 0, 0, "SELL"))
            elif abs(diff) > max(self.min_trade_usd, portfolio_value * self.min_trade_pct):
                action = "BUY" if diff > 0 else "SELL" if target_usd == 0 else "ADJUST"
                allocations.append(TargetAllocation(coin, target_weights.get(coin, 0), target_usd, action))

        return allocations

    def dump_state(self) -> dict:
        return {
            "smoothed_scores": dict(self._smoothed_scores),
            "hwm": self._hwm,
            "regime": self._regime,
        }

    def load_state(self, state: dict) -> None:
        self._smoothed_scores = state.get("smoothed_scores", {})
        self._hwm = state.get("hwm", 0.0)
        self._regime = state.get("regime", "SIDEWAYS")
```

- [ ] **Step 2: Verify import**

```bash
python -c "from bot.strategy.momentum_rotation import MomentumRotationStrategy, TargetAllocation; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add bot/strategy/momentum_rotation.py
git commit -m "feat: add MomentumRotationStrategy with buffer zones and crash protection"
```

---

## Task 6: Wire into main.py

**Files:**
- Modify: `main.py`
- Modify: `bot/config/config.yaml`

**Only proceed if Task 4 validation gates pass.**

- [ ] **Step 1: Add momentum rotation config to config.yaml**

Add after the existing `intraday` section:

```yaml
# Momentum Rotation Strategy
momentum_rotation:
  enabled: true
  rebalance_interval_hours: 4
  n_holdings: 4
  sell_rank: 6
  ema_alpha: 0.4       # from sweep
  fee_bps: 5           # limit orders
  # Weights from IC analysis (update after running scripts/ic_analysis.py)
  weights:
    sharpe_48h: 0.30
    nearness: 0.25
    sharpe_168h: 0.25
    residual: 0.20
  # Crash protection
  dd_flat: 0.12
  dd_half: 0.08
  dd_quarter: 0.05
```

- [ ] **Step 2: Add momentum rotation import and initialization in main.py**

Add import at top of main.py:
```python
from bot.strategy.momentum_rotation import MomentumRotationStrategy
from bot.strategy.momentum_signals import resample_to_4h
```

In the strategy initialization section (around line 760), add:
```python
    # Momentum rotation strategy (portfolio-level, replaces per-pair cascade)
    momentum_config = config.get("momentum_rotation", {})
    momentum_strategy = MomentumRotationStrategy(
        weights=momentum_config.get("weights", {}),
        n_holdings=momentum_config.get("n_holdings", 4),
        sell_rank=momentum_config.get("sell_rank", 6),
        ema_alpha=momentum_config.get("ema_alpha", 0.4),
        fee_bps=momentum_config.get("fee_bps", 5),
    ) if momentum_config.get("enabled", False) else None
```

- [ ] **Step 3: Add 4H rebalancing trigger in the main loop**

In Step 4 of the main loop (around line 399), add a 4H boundary check that triggers momentum rotation rebalancing. This runs alongside the existing 15M signal generation but only fires every 16th bar (4H):

```python
        # ── Phase A-MOM: Momentum rotation (every 4H boundary) ──────────
        _4H_BARS = 16  # 4H = 16 × 15M
        if momentum_strategy is not None and current_15m_epoch % _4H_BARS == 0:
            logger.info("Step 4A-MOM: 4H boundary — running momentum rotation")

            # Build 4H close series for all feature pairs
            coin_closes_4h = {}
            for pair in feature_pairs:
                df = live_fetcher._to_dataframe(pair)
                if len(df) >= 200:
                    df_4h = resample_to_4h(df)
                    coin_symbol = pair.split("/")[0]
                    coin_closes_4h[coin_symbol] = df_4h["close"]

            if "BTC" in coin_closes_4h:
                # Get current positions
                open_pos = order_manager.get_all_positions()
                current_pos_usd = {}
                for p, pos_obj in open_pos.items():
                    price = prices.get(p) or live_fetcher._last_prices.get(p, 0)
                    current_pos_usd[p.split("/")[0]] = pos_obj.quantity * price

                allocations = momentum_strategy.rebalance(
                    coin_closes_4h=coin_closes_4h,
                    btc_closes_4h=coin_closes_4h["BTC"],
                    current_positions=current_pos_usd,
                    portfolio_value=total_usd,
                    free_cash=free_usd,
                    current_date=pd.Timestamp.now(tz="UTC").date(),
                )

                for alloc in allocations:
                    pair = f"{alloc.coin}/USD"
                    if alloc.action == "SELL":
                        if pair in open_pos:
                            order_manager.close_position(pair)
                            logger.info("MOM: SELL %s", pair)
                    elif alloc.action in ("BUY", "ADJUST"):
                        # Use limit order with market fallback
                        current_price = prices.get(pair, 0)
                        if current_price > 0 and alloc.target_usd > 0:
                            target_qty = alloc.target_usd / current_price
                            logger.info("MOM: %s %s target=$%.0f",
                                       alloc.action, pair, alloc.target_usd)
                            # Submit order (limit preferred)
                            order_manager.submit_order(pair, target_qty, "limit")
```

Note: The exact order submission API depends on `order_manager`'s interface. Adapt the `submit_order` call to match.

- [ ] **Step 4: Commit**

```bash
git add main.py bot/config/config.yaml
git commit -m "feat: wire momentum rotation into main loop at 4H boundaries"
```
