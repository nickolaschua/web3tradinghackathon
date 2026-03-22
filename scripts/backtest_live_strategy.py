#!/usr/bin/env python3
"""
Backtest the EXACT live strategy from main.py.

Signal cascade:
  1. XGBoost BTC (threshold=0.65, exit=0.08) — BTC/USD only
  2. XGBoost SOL (threshold=0.70, exit=0.08) — SOL/USD only
  3. MeanReversion fallback (RSI<30 + bb<0.15 + MACD>0, or RSI<25) — all pairs
  4. No relaxed MR, no pairs ML

Risk management (from bot/execution/risk.py):
  - ATR trailing stop: 10x atr_proxy
  - Hard stop: 5%
  - Equal dollar risk sizing: 2% risk per trade
  - Kelly criterion gate
  - Max 5 concurrent positions
  - Max 40% concentration per position
  - Tiered circuit breaker: 10%->0.5x, 20%->0.25x, 30%->halt
  - Commission: 0.1%

Regime detection:
  - EMA_20 > EMA_50 spread > 0.1%: BULL (1.0x)
  - EMA_20 < EMA_50: BEAR (0.35x)
  - Sideways: 0.5x

Usage:
  python scripts/backtest_live_strategy.py
  python scripts/backtest_live_strategy.py --start 2025-01-01
  python scripts/backtest_live_strategy.py --window 10  # 10-day rolling windows
"""
import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features, compute_cross_asset_features, compute_btc_context_features
from bot.execution.portfolio import PortfolioAllocator

# ── Config (matches config.yaml + main.py) ──────────────────────────────────
CAPITAL = 1_000_000
COMMISSION_PCT = 0.001  # 0.1%
HARD_STOP_PCT = 0.05
ATR_STOP_MULT = 10.0
TRAILING_STOP_MULT = 10.0
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 5
MAX_SINGLE_PCT = 0.40
EXPECTED_WIN_LOSS = 1.5
WARMUP_BARS = 35
TRAIN_CUTOFF = "2024-01-01"

# Regime
REGIME_EMA_FAST = 20
REGIME_EMA_SLOW = 50

# Circuit breaker thresholds
CB_LIGHT = 0.10   # 0.5x
CB_HEAVY = 0.20   # 0.25x
CB_HALT = 0.30    # full halt

# XGBoost
BTC_MODEL = "models/xgb_btc_15m_iter5.pkl"
SOL_MODEL = "models/xgb_sol_15m.pkl"
BTC_THRESHOLD = 0.65
SOL_THRESHOLD = 0.70
EXIT_THRESHOLD = 0.08

PERIODS_15M = 35_040


def load_data(pair_name: str, data_dir: str = "data") -> pd.DataFrame:
    """Load 15m parquet for a given pair."""
    sym_map = {
        "BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT", "SOL/USD": "SOLUSDT",
        "BNB/USD": "BNBUSDT", "ADA/USD": "ADAUSDT", "AVAX/USD": "AVAXUSDT",
        "DOGE/USD": "DOGEUSDT", "LINK/USD": "LINKUSDT", "DOT/USD": "DOTUSDT",
        "UNI/USD": "UNIUSDT", "XRP/USD": "XRPUSDT", "LTC/USD": "LTCUSDT",
        "AAVE/USD": "AAVEUSDT", "CRV/USD": "CRVUSDT", "NEAR/USD": "NEARUSDT",
        "FIL/USD": "FILUSDT", "FET/USD": "FETUSDT", "HBAR/USD": "HBARUSDT",
        "ZEC/USD": "ZECUSDT", "ZEN/USD": "ZENUSDT",
    }
    sym = sym_map.get(pair_name)
    if not sym:
        return pd.DataFrame()
    path = Path(data_dir) / f"{sym}_15m.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def compute_regime(df: pd.DataFrame) -> pd.Series:
    """Compute regime multiplier series matching regime.py logic."""
    ema_fast = df["close"].ewm(span=REGIME_EMA_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=REGIME_EMA_SLOW, adjust=False).mean()
    spread_pct = (ema_fast - ema_slow).abs() / ema_slow

    regime = pd.Series(0.5, index=df.index)  # default SIDEWAYS
    regime[spread_pct >= 0.001] = np.where(
        ema_fast[spread_pct >= 0.001] > ema_slow[spread_pct >= 0.001],
        1.0,   # BULL
        0.35,  # BEAR
    )
    return regime


def get_cb_multiplier(drawdown: float) -> float:
    """Tiered circuit breaker multiplier."""
    if drawdown >= CB_HALT:
        return 0.0
    if drawdown >= CB_HEAVY:
        return 0.25
    if drawdown >= CB_LIGHT:
        return 0.5
    return 1.0


def run_backtest(
    start: str = TRAIN_CUTOFF,
    end: str | None = None,
) -> pd.DataFrame:
    """Run the full multi-asset backtest mirroring main.py."""

    # Load models
    with open(BTC_MODEL, "rb") as f:
        btc_model = pickle.load(f)
    btc_features = list(btc_model.feature_names_in_)

    sol_model = None
    sol_features = []
    if Path(SOL_MODEL).exists():
        with open(SOL_MODEL, "rb") as f:
            sol_model = pickle.load(f)
        sol_features = list(sol_model.feature_names_in_)

    # Load price data
    btc_raw = load_data("BTC/USD")
    eth_raw = load_data("ETH/USD")
    sol_raw = load_data("SOL/USD")

    if btc_raw.empty:
        print("ERROR: BTC data not found")
        sys.exit(1)

    # Compute features for BTC (primary)
    btc_feat = compute_features(btc_raw)
    if not eth_raw.empty and not sol_raw.empty:
        btc_feat = compute_cross_asset_features(btc_feat, {"ETH/USD": eth_raw, "SOL/USD": sol_raw})
    btc_ctx = compute_btc_context_features(btc_raw, eth_raw, sol_raw)
    btc_feat = btc_feat.join(btc_ctx, how="left", rsuffix="_ctx")

    # SOL features
    sol_feat = pd.DataFrame()
    if not sol_raw.empty:
        sol_feat = compute_features(sol_raw)
        if not eth_raw.empty:
            sol_feat = compute_cross_asset_features(sol_feat, {"ETH/USD": eth_raw, "BTC/USD": btc_raw})
        sol_ctx = compute_btc_context_features(btc_raw, eth_raw, sol_raw)
        sol_feat = sol_feat.join(sol_ctx, how="left", rsuffix="_ctx")

    # Load additional pairs for MR
    mr_pairs = ["ETH/USD", "BNB/USD", "ADA/USD", "AVAX/USD", "DOGE/USD",
                "LINK/USD", "DOT/USD", "UNI/USD", "XRP/USD", "LTC/USD",
                "AAVE/USD", "CRV/USD", "NEAR/USD", "FIL/USD", "FET/USD",
                "HBAR/USD", "ZEC/USD", "ZEN/USD"]
    mr_data = {}
    for p in mr_pairs:
        raw = load_data(p)
        if not raw.empty:
            feat = compute_features(raw)
            mr_data[p] = feat

    # Filter to OOS period
    btc_feat = btc_feat[start:end] if end else btc_feat[start:]
    sol_feat = sol_feat[start:end] if end and not sol_feat.empty else (sol_feat[start:] if not sol_feat.empty else sol_feat)
    for p in list(mr_data.keys()):
        mr_data[p] = mr_data[p][start:end] if end else mr_data[p][start:]

    # Regime for BTC
    btc_regime = compute_regime(btc_feat)

    # ── Simulation state ─────────────────────────────────────────────────────
    portfolio_value = float(CAPITAL)
    cash = float(CAPITAL)
    hwm = float(CAPITAL)
    positions = {}  # pair -> {qty, entry, stop, trailing_stop, source}
    trades = []
    equity_curve = []

    btc_close = btc_feat["close"].values
    btc_idx = btc_feat.index

    # Pre-compute XGBoost probabilities for BTC
    btc_row_df = btc_feat.copy()
    for col in btc_features:
        if col not in btc_row_df.columns:
            btc_row_df[col] = np.nan
    btc_probas = btc_model.predict_proba(btc_row_df[btc_features])[:, 1]

    # Pre-compute XGBoost probabilities for SOL
    sol_probas = np.full(len(btc_idx), 0.5)
    if sol_model is not None and not sol_feat.empty:
        common_idx = btc_idx.intersection(sol_feat.index)
        sol_row_df = sol_feat.loc[common_idx].copy()
        for col in sol_features:
            if col not in sol_row_df.columns:
                sol_row_df[col] = np.nan
        sol_p = sol_model.predict_proba(sol_row_df[sol_features])[:, 1]
        sol_proba_series = pd.Series(sol_p, index=common_idx)
        sol_probas_aligned = sol_proba_series.reindex(btc_idx).fillna(0.5).values
        sol_probas = sol_probas_aligned

    # Pre-compute MR features at each timestamp
    # Index MR data by btc_idx for aligned access
    mr_features_aligned = {}
    for p, feat_df in mr_data.items():
        aligned = feat_df.reindex(btc_idx)
        mr_features_aligned[p] = aligned

    # SOL close prices aligned
    sol_close_aligned = pd.Series(dtype=float)
    if not sol_raw.empty:
        sol_close = sol_raw["close"].reindex(btc_idx, method="ffill")
        sol_close_aligned = sol_close

    # MR close prices aligned
    mr_close = {}
    for p in mr_data:
        raw = load_data(p)
        if not raw.empty:
            mr_close[p] = raw["close"].reindex(btc_idx, method="ffill")

    print(f"Backtesting {len(btc_idx):,} bars from {btc_idx[0]} to {btc_idx[-1]}")
    print(f"Capital: ${CAPITAL:,.0f} | Commission: {COMMISSION_PCT*100:.1f}%")
    print(f"Models: BTC(th={BTC_THRESHOLD}) SOL(th={SOL_THRESHOLD}) + MR fallback")

    # Portfolio allocator (HRP + CVaR blended weights, matching live bot)
    allocator_config = {"cvar_beta": 0.95, "hrp_blend": 0.5}
    allocator = PortfolioAllocator(allocator_config)
    n_feature_pairs = 20  # matches config.yaml feature_pairs count

    # Preload all raw OHLCV for portfolio weight computation
    all_feature_pairs = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "AVAX/USD",
                         "DOGE/USD", "LINK/USD", "DOT/USD", "UNI/USD", "XRP/USD",
                         "LTC/USD", "NEAR/USD", "SUI/USD", "APT/USD", "PEPE/USD",
                         "ARB/USD", "SHIB/USD", "FIL/USD", "HBAR/USD", "BNB/USD"]
    feature_raw = {}
    for p in all_feature_pairs:
        raw = load_data(p)
        if not raw.empty:
            feature_raw[p] = raw

    # Precompute weights ONCE using all available history (fast, stable)
    price_history = {p: df for p, df in feature_raw.items() if len(df) > 60}
    if len(price_history) >= 2:
        allocator.compute_weights(price_history)
    print()

    for i in range(WARMUP_BARS, len(btc_idx)):
        ts = btc_idx[i]
        price_btc = btc_close[i]

        # ── Update portfolio value ───────────────────────────────────────────
        portfolio_value = cash
        for pair, pos in list(positions.items()):
            if pair == "BTC/USD":
                cur_price = price_btc
            elif pair == "SOL/USD" and not sol_close_aligned.empty:
                cur_price = sol_close_aligned.iloc[i] if i < len(sol_close_aligned) else pos["entry"]
            elif pair in mr_close:
                cur_price = mr_close[pair].iloc[i] if i < len(mr_close[pair]) else pos["entry"]
            else:
                cur_price = pos["entry"]
            portfolio_value += pos["qty"] * cur_price

        # ── Circuit breaker ──────────────────────────────────────────────────
        hwm = max(hwm, portfolio_value)
        drawdown = (hwm - portfolio_value) / hwm if hwm > 0 else 0
        cb_mult = get_cb_multiplier(drawdown)
        cb_halt = cb_mult == 0.0

        # ── Check stops for all positions ────────────────────────────────────
        for pair in list(positions.keys()):
            pos = positions[pair]
            if pair == "BTC/USD":
                cur_price = price_btc
            elif pair == "SOL/USD" and not sol_close_aligned.empty:
                cur_price = sol_close_aligned.iloc[i] if i < len(sol_close_aligned) else pos["entry"]
            elif pair in mr_close:
                cur_price = mr_close[pair].iloc[i] if i < len(mr_close[pair]) else pos["entry"]
            else:
                continue

            # Update trailing stop (only moves up)
            atr = pos.get("atr", cur_price * 0.02)
            new_trail = cur_price - TRAILING_STOP_MULT * atr
            if new_trail > pos["trailing_stop"]:
                pos["trailing_stop"] = new_trail

            # Check hard stop and trailing stop
            exit_reason = None
            if cur_price <= pos["stop"]:
                exit_reason = "hard_stop"
            elif cur_price <= pos["trailing_stop"]:
                exit_reason = "trailing_stop"

            if exit_reason:
                pnl = (cur_price - pos["entry"]) * pos["qty"]
                commission = cur_price * pos["qty"] * COMMISSION_PCT
                cash += cur_price * pos["qty"] - commission
                trades.append({
                    "pair": pair, "side": "SELL", "price": cur_price,
                    "qty": pos["qty"], "pnl": pnl - commission,
                    "reason": exit_reason, "source": pos["source"],
                    "timestamp": ts,
                })
                del positions[pair]

        # ── Signal generation (Phase 4A cascade) ────────────────────────────
        regime_mult_btc = btc_regime.iloc[i] if i < len(btc_regime) else 0.5

        signals = {}  # pair -> (direction, confidence, size, source)

        # 1. XGBoost BTC
        p_btc = btc_probas[i]
        if p_btc >= BTC_THRESHOLD:
            signals["BTC/USD"] = ("BUY", p_btc, 1.0, "xgb_btc")
        elif p_btc <= EXIT_THRESHOLD:
            signals["BTC/USD"] = ("SELL", p_btc, 1.0, "xgb_btc")

        # 2. XGBoost SOL
        p_sol = sol_probas[i]
        if p_sol >= SOL_THRESHOLD:
            signals["SOL/USD"] = ("BUY", p_sol, 1.0, "xgb_sol")
        elif p_sol <= EXIT_THRESHOLD:
            signals["SOL/USD"] = ("SELL", p_sol, 1.0, "xgb_sol")

        # 3. MR fallback for all pairs without XGB signal
        all_pairs = ["BTC/USD", "SOL/USD"] + list(mr_features_aligned.keys())
        for pair in all_pairs:
            if pair in signals:
                continue

            if pair == "BTC/USD":
                feat_row = btc_feat.iloc[i] if i < len(btc_feat) else None
            elif pair == "SOL/USD":
                feat_row = sol_feat.iloc[i] if not sol_feat.empty and i < len(sol_feat) else None
            elif pair in mr_features_aligned:
                feat_row = mr_features_aligned[pair].iloc[i]
            else:
                continue

            if feat_row is None or (hasattr(feat_row, 'isna') and feat_row.get("close", np.nan) != feat_row.get("close", np.nan)):
                continue

            ema20 = feat_row.get("EMA_20", np.nan)
            ema50 = feat_row.get("EMA_50", np.nan)
            if not (ema20 > ema50):  # regime gate
                continue

            rsi = feat_row.get("RSI_14", 50.0)
            bb_pos = feat_row.get("bb_pos", 0.5)
            macd_hist = feat_row.get("MACDh_12_26_9", 0.0)

            if rsi < 30 and bb_pos < 0.15 and macd_hist > 0:
                signals[pair] = ("BUY", 0.60, 0.35, "mr")
            elif rsi < 25:
                signals[pair] = ("BUY", 0.55, 0.25, "mr")
            elif rsi > 55 or bb_pos > 0.6:
                signals[pair] = ("SELL", 0.70, 1.0, "mr")

        # ── Execute signals (Phase 4D) ───────────────────────────────────────
        for pair, (direction, confidence, sig_size, source) in signals.items():
            if pair == "BTC/USD":
                cur_price = price_btc
            elif pair == "SOL/USD" and not sol_close_aligned.empty:
                cur_price = sol_close_aligned.iloc[i] if i < len(sol_close_aligned) else 0
            elif pair in mr_close:
                cur_price = mr_close[pair].iloc[i] if i < len(mr_close[pair]) else 0
            else:
                continue

            if cur_price <= 0:
                continue

            if direction == "BUY":
                if cb_halt:
                    continue
                if pair in positions:
                    continue
                if len(positions) >= MAX_POSITIONS:
                    continue

                # Regime multiplier
                if pair == "BTC/USD":
                    r_mult = regime_mult_btc
                elif pair in mr_features_aligned:
                    pair_raw = load_data(pair)
                    if not pair_raw.empty:
                        pair_regime = compute_regime(pair_raw)
                        r_mult_s = pair_regime.reindex(btc_idx).ffill()
                        r_mult = r_mult_s.iloc[i] if i < len(r_mult_s) else 0.5
                    else:
                        r_mult = 0.5
                else:
                    r_mult = 0.5

                if r_mult < 0.10:
                    continue

                # ATR
                if pair == "BTC/USD":
                    atr = btc_feat.iloc[i].get("atr_proxy", cur_price * 0.02)
                elif pair == "SOL/USD" and not sol_feat.empty:
                    atr = sol_feat.iloc[i].get("atr_proxy", cur_price * 0.02) if i < len(sol_feat) else cur_price * 0.02
                elif pair in mr_features_aligned:
                    atr_val = mr_features_aligned[pair].iloc[i].get("atr_proxy", np.nan)
                    atr = atr_val if not np.isnan(atr_val) else cur_price * 0.02
                else:
                    atr = cur_price * 0.02

                # Stop levels
                hard_stop = cur_price * (1 - HARD_STOP_PCT)
                atr_stop = cur_price - ATR_STOP_MULT * atr if atr > 0 else hard_stop
                initial_stop = max(hard_stop, atr_stop)
                stop_distance = cur_price - initial_stop
                if stop_distance <= 0:
                    stop_distance = cur_price * HARD_STOP_PCT

                # Kelly gate
                b = EXPECTED_WIN_LOSS
                p = max(min(confidence, 1.0), 0.0)
                kelly = (p * b - (1 - p)) / b
                if kelly <= 0:
                    continue

                # Effective multiplier
                eff_mult = r_mult * cb_mult

                # Equal dollar risk sizing
                total_port = portfolio_value
                risk_usd = total_port * RISK_PER_TRADE * confidence * eff_mult
                # Portfolio weight from HRP+CVaR allocator (matches live bot)
                pw = allocator.get_pair_weight(pair, n_active_pairs=n_feature_pairs)
                pw *= sig_size  # scale by signal size (MR uses 0.25-0.35)
                risk_usd *= pw

                qty = risk_usd / stop_distance
                target_usd = qty * cur_price

                # Caps
                usable = cash * 0.95
                target_usd = min(target_usd, total_port * MAX_SINGLE_PCT, usable)
                if target_usd < 100:
                    continue
                qty = target_usd / cur_price

                # Commission
                commission = target_usd * COMMISSION_PCT
                cash -= target_usd + commission

                positions[pair] = {
                    "qty": qty, "entry": cur_price,
                    "stop": hard_stop, "trailing_stop": initial_stop,
                    "atr": atr, "source": source,
                }
                trades.append({
                    "pair": pair, "side": "BUY", "price": cur_price,
                    "qty": qty, "pnl": -commission,
                    "reason": "signal", "source": source,
                    "timestamp": ts,
                })

            elif direction == "SELL":
                if pair not in positions:
                    continue
                pos = positions[pair]
                pnl = (cur_price - pos["entry"]) * pos["qty"]
                commission = cur_price * pos["qty"] * COMMISSION_PCT
                cash += cur_price * pos["qty"] - commission
                trades.append({
                    "pair": pair, "side": "SELL", "price": cur_price,
                    "qty": pos["qty"], "pnl": pnl - commission,
                    "reason": "signal_exit", "source": pos["source"],
                    "timestamp": ts,
                })
                del positions[pair]

        # Record equity
        pv = cash
        for pair, pos in positions.items():
            if pair == "BTC/USD":
                pv += pos["qty"] * price_btc
            elif pair == "SOL/USD" and not sol_close_aligned.empty:
                pv += pos["qty"] * (sol_close_aligned.iloc[i] if i < len(sol_close_aligned) else pos["entry"])
            elif pair in mr_close:
                pv += pos["qty"] * (mr_close[pair].iloc[i] if i < len(mr_close[pair]) else pos["entry"])
            else:
                pv += pos["qty"] * pos["entry"]

        equity_curve.append({"timestamp": ts, "equity": pv})

    # ── Results ──────────────────────────────────────────────────────────────
    eq = pd.DataFrame(equity_curve).set_index("timestamp")["equity"]
    returns = eq.pct_change().dropna()

    total_ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    n_trades = len([t for t in trades if t["side"] == "BUY"])
    wins = len([t for t in trades if t["side"] == "SELL" and t["pnl"] > 0])
    losses = len([t for t in trades if t["side"] == "SELL" and t["pnl"] <= 0])
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    # Annualized metrics
    ann_ret = returns.mean() * PERIODS_15M
    ann_vol = returns.std() * np.sqrt(PERIODS_15M)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(PERIODS_15M)
    sortino = ann_ret / downside if downside > 0 else 0

    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_dd = drawdown.min() * 100

    calmar = (ann_ret / abs(max_dd / 100)) if max_dd != 0 else 0

    # Competition score
    comp_score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    print("=" * 60)
    print("BACKTEST RESULTS — LIVE STRATEGY")
    print("=" * 60)
    print(f"Period:         {btc_idx[WARMUP_BARS]} -> {btc_idx[-1]}")
    print(f"Total Return:   {total_ret:+.2f}%")
    print(f"Final Equity:   ${eq.iloc[-1]:,.2f}")
    print(f"Sharpe:         {sharpe:.3f}")
    print(f"Sortino:        {sortino:.3f}")
    print(f"Calmar:         {calmar:.3f}")
    print(f"Max Drawdown:   {max_dd:.2f}%")
    print(f"Comp Score:     {comp_score:.3f} (0.4×Sortino + 0.3×Sharpe + 0.3×Calmar)")
    print(f"Trades:         {n_trades} entries")
    print(f"Win Rate:       {win_rate:.1f}% ({wins}W / {losses}L)")
    print()

    # Source breakdown
    source_counts = {}
    for t in trades:
        if t["side"] == "BUY":
            s = t["source"]
            source_counts[s] = source_counts.get(s, 0) + 1
    print("Signal sources:")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c} trades")
    print()

    # Stop vs signal exits
    stop_exits = len([t for t in trades if t["side"] == "SELL" and "stop" in t["reason"]])
    sig_exits = len([t for t in trades if t["side"] == "SELL" and "stop" not in t["reason"]])
    print(f"Exits: {stop_exits} stops / {sig_exits} signals")

    # 10-day rolling windows
    print()
    print("10-DAY ROLLING WINDOWS (competition-relevant):")
    print("-" * 60)
    bars_10d = 10 * 24 * 4  # 10 days at 15M
    if len(eq) > bars_10d:
        for offset in range(0, min(5, len(eq) // bars_10d)):
            end_idx = len(eq) - 1 - offset * bars_10d
            start_idx = end_idx - bars_10d
            if start_idx < 0:
                break
            window_eq = eq.iloc[start_idx:end_idx]
            w_ret = (window_eq.iloc[-1] / window_eq.iloc[0] - 1) * 100
            w_returns = window_eq.pct_change().dropna()
            w_sharpe = (w_returns.mean() / w_returns.std() * np.sqrt(PERIODS_15M)) if w_returns.std() > 0 else 0
            w_dd = ((window_eq - window_eq.cummax()) / window_eq.cummax()).min() * 100
            print(f"  {window_eq.index[0].strftime('%Y-%m-%d')} -> {window_eq.index[-1].strftime('%Y-%m-%d')}: "
                  f"ret={w_ret:+.2f}% sharpe={w_sharpe:.2f} dd={w_dd:.2f}%")

    return eq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=TRAIN_CUTOFF)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    run_backtest(start=args.start, end=args.end)
