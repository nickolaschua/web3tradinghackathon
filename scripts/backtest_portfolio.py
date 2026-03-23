#!/usr/bin/env python3
"""
Multi-model portfolio backtest: run N XGBoost models simultaneously.

Simulates portfolio with independent position slots per coin, shared capital,
full risk stack (ATR trailing stops, circuit breaker, Kelly gate).

Usage:
  # BTC + XRP only (current live config)
  python scripts/backtest_portfolio.py --coins btc,xrp

  # All 5 models
  python scripts/backtest_portfolio.py --coins btc,xrp,bnb,sol,doge

  # Custom thresholds
  python scripts/backtest_portfolio.py --coins btc,xrp,bnb,sol,doge \
    --thresholds 0.65,0.65,0.70,0.70,0.65
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_btc_context_features, compute_cross_asset_features, compute_features

PERIODS_15M = 35_040
TRAIN_CUTOFF = "2024-01-01"
CORR_WINDOW = 2880


# -- Feature pipelines per coin ------------------------------------------------

def _load_and_clean(path):
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower()
    return df


def prepare_btc_features(btc, eth, sol, start=None, end=None):
    """BTC feature matrix (matches train_model_15m.py)."""
    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})

    for asset, df in [("btc", btc), ("eth", eth), ("sol", sol)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    feat = compute_btc_context_features(feat, eth, sol, window=CORR_WINDOW)
    feat = feat.dropna()

    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]
    return feat


def prepare_xrp_features(btc, eth, sol, xrp, start=None, end=None):
    """XRP feature matrix. Model uses eth_btc_corr/eth_btc_beta (same as BTC model).
    Features: compute_features(xrp) + ETH/SOL cross-asset lags + eth_btc context."""
    feat = compute_features(xrp)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})

    # ETH and SOL 4H/1D return lags (matches model feature_names_in_)
    for asset, df in [("eth", eth), ("sol", sol)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    # eth_btc_corr and eth_btc_beta (same as BTC model — NOT xrp_btc)
    feat = compute_btc_context_features(feat, eth, sol, window=CORR_WINDOW)

    feat = feat.dropna()
    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]
    return feat


def prepare_alt_features(target, btc, eth, sol, target_df, start=None, end=None):
    """Generic alt-coin feature matrix (BNB, SOL, DOGE, ETH).
    Uses ETH+SOL as cross-asset lags (not BTC), target_btc_corr/beta for context."""
    feat = compute_features(target_df)
    cross_dfs = {"BTC/USD": btc, "ETH/USD": eth, "SOL/USD": sol}
    # Remove target from cross_dfs if present
    _pair_map = {"eth": "ETH/USD", "sol": "SOL/USD"}
    if target in _pair_map and _pair_map[target] in cross_dfs:
        del cross_dfs[_pair_map[target]]
    feat = compute_cross_asset_features(feat, cross_dfs)

    # Cross-asset lags: use ETH + SOL (for BNB/DOGE) or BTC + other (for ETH/SOL)
    # Determine from feature cols which lags are needed
    lag_sources = {"bnb": [("eth", eth), ("sol", sol)],
                   "doge": [("eth", eth), ("sol", sol)],
                   "sol": [("btc", btc), ("eth", eth)],
                   "eth": [("btc", btc), ("sol", sol)]}

    for asset, df in lag_sources.get(target, [("eth", eth), ("sol", sol)]):
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    # Target-BTC correlation and beta
    target_ret = np.log(target_df["close"] / target_df["close"].shift(1)).reindex(feat.index)
    btc_ret = np.log(btc["close"] / btc["close"].shift(1)).reindex(feat.index)
    feat[f"{target}_btc_corr"] = target_ret.rolling(CORR_WINDOW).corr(btc_ret).shift(1)
    cov = target_ret.rolling(CORR_WINDOW).cov(btc_ret)
    var_btc = btc_ret.rolling(CORR_WINDOW).var()
    feat[f"{target}_btc_beta"] = (cov / (var_btc + 1e-10)).shift(1)

    feat = feat.dropna()
    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]
    return feat


# -- Model loading -------------------------------------------------------------

def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def batch_predict(model, feat_df):
    cols = list(model.feature_names_in_)
    missing = [c for c in cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing features for model: {missing}")
    X = feat_df[cols].values
    valid = ~np.isnan(X).any(axis=1)
    probas = np.full(len(X), np.nan)
    if valid.any():
        probas[valid] = model.predict_proba(X[valid])[:, 1]
    return probas


# -- Circuit breaker -----------------------------------------------------------

def cb_multiplier(drawdown, halt=0.30, heavy=0.20, light=0.10):
    if drawdown >= halt:
        return 0.0
    elif drawdown >= heavy:
        return 0.25
    elif drawdown >= light:
        return 0.5
    return 1.0


# -- Combined N-model backtest -------------------------------------------------

def run_portfolio_backtest(
    coin_data,  # dict: coin_name -> {feat_df, probas, threshold, exit_threshold}
    initial_capital=1_000_000,
    risk_per_trade=0.02,
    hard_stop_pct=0.05,
    atr_mult=10.0,
    max_single_pct=0.40,
    expected_wl=1.5,
    fee_bps=10,
):
    """Bar-by-bar portfolio backtest with N independent position slots."""
    fee = fee_bps / 10_000.0

    # Find common timeline across all coins
    common_idx = None
    for cd in coin_data.values():
        idx = cd["feat_df"].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    # Align all data to common index
    for name, cd in coin_data.items():
        mask = cd["feat_df"].index.isin(common_idx)
        cd["feat_aligned"] = cd["feat_df"].loc[common_idx]
        cd["probas_aligned"] = cd["probas"][mask][:len(common_idx)]
        cd["closes"] = cd["feat_aligned"]["close"].values
        cd["atrs"] = cd["feat_aligned"]["atr_proxy"].values

    timestamps = common_idx
    n = len(common_idx)
    coin_names = list(coin_data.keys())

    # Portfolio state
    free_balance = initial_capital
    portfolio_hwm = initial_capital

    slots = {name: {"units": 0.0, "entry_price": 0.0, "trail_stop": 0.0}
             for name in coin_names}

    portfolio_values = np.zeros(n)
    portfolio_values[0] = initial_capital
    returns = np.zeros(n)
    closed_trades = []
    active_days = set()
    gate_stats = {"kelly_blocked": 0, "cb_halted": 0, "total_signals": 0}

    for i in range(n):
        ts = timestamps[i]

        # Mark to market
        total = free_balance
        for name in coin_names:
            total += slots[name]["units"] * coin_data[name]["closes"][i]

        portfolio_hwm = max(portfolio_hwm, total)
        dd = (portfolio_hwm - total) / portfolio_hwm if portfolio_hwm > 0 else 0.0
        cb_mult = cb_multiplier(dd)

        # Process each coin
        for name in coin_names:
            cd = coin_data[name]
            slot = slots[name]
            c = cd["closes"][i]
            atr = cd["atrs"][i]
            p = cd["probas_aligned"][i]
            threshold = cd["threshold"]
            exit_thresh = cd["exit_threshold"]

            # Trailing stop update
            if slot["units"] > 0 and not np.isnan(atr) and atr > 0:
                new_stop = c - atr_mult * atr
                slot["trail_stop"] = max(slot["trail_stop"], new_stop)

            # Check exits
            just_exited = False
            if slot["units"] > 0:
                stop_hit = c <= slot["trail_stop"]
                sell_signal = (not np.isnan(p)) and p <= exit_thresh

                if stop_hit or sell_signal:
                    proceeds = slot["units"] * c * (1.0 - fee)
                    net_exit = c * (1.0 - fee)
                    pnl_pct = (net_exit - slot["entry_price"]) / slot["entry_price"]

                    closed_trades.append({
                        "asset": name.upper(),
                        "exit_bar": ts,
                        "entry_price": slot["entry_price"],
                        "exit_price": net_exit,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "stop" if stop_hit else "signal",
                    })

                    free_balance += proceeds
                    slot["units"] = 0.0
                    just_exited = True
                    active_days.add(ts.date() if hasattr(ts, "date") else str(ts)[:10])

                    # Refresh total
                    total = free_balance
                    for n2 in coin_names:
                        total += slots[n2]["units"] * coin_data[n2]["closes"][i]

            # Check entries
            if slot["units"] == 0 and not just_exited:
                if np.isnan(p) or p < threshold:
                    continue

                gate_stats["total_signals"] += 1

                # CB gate
                if cb_mult == 0.0:
                    gate_stats["cb_halted"] += 1
                    continue

                # Kelly gate
                kelly = (p * expected_wl - (1.0 - p)) / expected_wl
                if kelly <= 0:
                    gate_stats["kelly_blocked"] += 1
                    continue

                # Compute stop
                hard_stop = c * (1.0 - hard_stop_pct)
                if not np.isnan(atr) and atr > 0:
                    atr_stop = c - atr_mult * atr
                    initial_stop = max(hard_stop, atr_stop)
                else:
                    initial_stop = hard_stop

                stop_dist = min(c - initial_stop, c * hard_stop_pct)
                if stop_dist <= 0:
                    stop_dist = c * hard_stop_pct

                # Size position
                effective_mult = cb_mult
                risk_usd = total * risk_per_trade * p * effective_mult
                qty = risk_usd / stop_dist
                target_usd = qty * c

                usable = free_balance * 0.95
                target_usd = min(target_usd, total * max_single_pct, usable)

                if target_usd >= 10.0:
                    position_units = target_usd / c
                    entry_fee = target_usd * fee
                    free_balance -= (target_usd + entry_fee)
                    slot["units"] = position_units
                    slot["entry_price"] = c * (1.0 + fee)
                    slot["trail_stop"] = initial_stop
                    active_days.add(ts.date() if hasattr(ts, "date") else str(ts)[:10])

                    # Refresh total
                    total = free_balance
                    for n2 in coin_names:
                        total += slots[n2]["units"] * coin_data[n2]["closes"][i]

        # Record portfolio value
        pv = free_balance
        for name in coin_names:
            pv += slots[name]["units"] * coin_data[name]["closes"][i]
        portfolio_values[i] = pv
        if i > 0:
            returns[i] = portfolio_values[i] / portfolio_values[i - 1] - 1.0

    return (
        pd.Series(returns, index=timestamps),
        pd.Series(portfolio_values, index=timestamps),
        closed_trades,
        gate_stats,
        active_days,
    )


# -- Stats ---------------------------------------------------------------------

def compute_and_print_stats(returns, portfolio, trades, gate_stats, active_days, label=""):
    total_ret = portfolio.iloc[-1] / portfolio.iloc[0] - 1
    years = len(returns) / PERIODS_15M
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    sharpe = (returns.mean() / returns.std()) * np.sqrt(PERIODS_15M) if returns.std() > 0 else 0
    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std()) * np.sqrt(PERIODS_15M) if len(downside) > 0 and downside.std() > 0 else 0
    running_max = portfolio.cummax()
    drawdowns = (portfolio - running_max) / running_max
    max_dd = drawdowns.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Competition score
    comp_score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_pnl = np.mean([t["pnl_pct"] for t in trades]) * 100 if trades else 0

    # By asset
    assets = {}
    for t in trades:
        a = t["asset"]
        if a not in assets:
            assets[a] = {"count": 0, "wins": 0, "pnl_sum": 0}
        assets[a]["count"] += 1
        assets[a]["pnl_sum"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            assets[a]["wins"] += 1

    stop_exits = sum(1 for t in trades if t["exit_reason"] == "stop")

    print(f"\n{'=' * 70}")
    print(f"  PORTFOLIO BACKTEST — {label}")
    print(f"{'=' * 70}")
    print(f"  Initial capital   : ${portfolio.iloc[0]:,.0f}")
    print(f"  Final portfolio   : ${portfolio.iloc[-1]:,.2f}")
    print(f"  Total return      : {total_ret:+.2%}")
    print(f"  CAGR              : {cagr:+.2%}")
    print(f"  Sharpe (15M ann.) : {sharpe:.3f}")
    print(f"  Sortino (15M ann.): {sortino:.3f}")
    print(f"  Calmar            : {calmar:.3f}")
    print(f"  Max drawdown      : {max_dd:.2%}")
    print(f"  Competition score : {comp_score:.3f}  (0.4*Sortino + 0.3*Sharpe + 0.3*Calmar)")
    print(f"{'=' * 70}")
    print(f"  # Trades          : {n_trades}")
    print(f"    Stop exits      : {stop_exits}")
    print(f"    Signal exits    : {n_trades - stop_exits}")
    print(f"  Win rate          : {win_rate:.1f}%")
    print(f"  Avg trade PnL     : {avg_pnl:+.2f}%")
    if trades:
        print(f"  Best trade        : {max(t['pnl_pct'] for t in trades):+.2%}")
        print(f"  Worst trade       : {min(t['pnl_pct'] for t in trades):+.2%}")
    print(f"{'=' * 70}")
    print(f"  By asset:")
    for asset, stats in sorted(assets.items()):
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        avg = stats["pnl_sum"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"    {asset:8s}: {stats['count']:4d} trades | {wr:5.1f}% win | {avg:+.2f}% avg")
    print(f"{'=' * 70}")
    print(f"  Active trading days: {len(active_days)}")
    total_days = (portfolio.index[-1] - portfolio.index[0]).days
    print(f"  Calendar days      : {total_days}")
    if total_days > 0:
        print(f"  Activity rate      : {len(active_days) / total_days:.1%}")
    print(f"{'=' * 70}")
    print(f"  Gates:")
    print(f"    Total signals    : {gate_stats['total_signals']}")
    print(f"    CB halted        : {gate_stats['cb_halted']}")
    print(f"    Kelly blocked    : {gate_stats['kelly_blocked']}")
    print(f"{'=' * 70}")

    return {
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "comp_score": comp_score, "total_return": total_ret,
        "max_dd": max_dd, "n_trades": n_trades, "win_rate": win_rate,
        "active_days": len(active_days),
    }


# -- Main ----------------------------------------------------------------------

MODEL_PATHS = {
    "btc": "models/xgb_btc_15m_iter5.pkl",
    "xrp": "models/xgb_xrp_15m.pkl",
    "bnb": "models/xgb_bnb_15m.pkl",
    "sol": "models/xgb_sol_15m.pkl",
    "doge": "models/xgb_doge_15m.pkl",
    "ada": "models/xgb_ada_15m.pkl",
    "fet": "models/xgb_fet_15m.pkl",
    "hbar": "models/xgb_hbar_15m.pkl",
    "avax": "models/xgb_avax_15m.pkl",
    "ltc": "models/xgb_ltc_15m.pkl",
    "link": "models/xgb_link_15m.pkl",
    "dot": "models/xgb_dot_15m.pkl",
    "near": "models/xgb_near_15m.pkl",
    "uni": "models/xgb_uni_15m.pkl",
    "sui": "models/xgb_sui_15m.pkl",
    "trx": "models/xgb_trx_15m.pkl",
    "xlm": "models/xgb_xlm_15m.pkl",
    "shib": "models/xgb_shib_15m.pkl",
    "cake": "models/xgb_cake_15m.pkl",
    "floki": "models/xgb_floki_15m.pkl",
    "crv": "models/xgb_crv_15m.pkl",
    "icp": "models/xgb_icp_15m.pkl",
    "zen": "models/xgb_zen_15m.pkl",
}

DATA_PATHS = {
    "btc": "data/BTCUSDT_15m.parquet",
    "eth": "data/ETHUSDT_15m.parquet",
    "sol": "data/SOLUSDT_15m.parquet",
    "xrp": "data/XRPUSDT_15m.parquet",
    "bnb": "data/BNBUSDT_15m.parquet",
    "doge": "data/DOGEUSDT_15m.parquet",
    "ada": "data/ADAUSDT_15m.parquet",
    "fet": "data/FETUSDT_15m.parquet",
    "hbar": "data/HBARUSDT_15m.parquet",
    "avax": "data/AVAXUSDT_15m.parquet",
    "ltc": "data/LTCUSDT_15m.parquet",
    "link": "data/LINKUSDT_15m.parquet",
    "dot": "data/DOTUSDT_15m.parquet",
    "near": "data/NEARUSDT_15m.parquet",
    "uni": "data/UNIUSDT_15m.parquet",
    "sui": "data/SUIUSDT_15m.parquet",
    "trx": "data/TRXUSDT_15m.parquet",
    "xlm": "data/XLMUSDT_15m.parquet",
    "shib": "data/SHIBUSDT_15m.parquet",
    "cake": "data/CAKEUSDT_15m.parquet",
    "floki": "data/FLOKIUSDT_15m.parquet",
    "crv": "data/CRVUSDT_15m.parquet",
    "icp": "data/ICPUSDT_15m.parquet",
    "zen": "data/ZENUSDT_15m.parquet",
}

DEFAULT_THRESHOLDS = {
    "btc": 0.65,
    "xrp": 0.65,
    "bnb": 0.70,
    "sol": 0.70,
    "doge": 0.65,
    "ada": 0.70,
    "fet": 0.65,
    "hbar": 0.70,
    "avax": 0.75,
    "ltc": 0.70,
    "link": 0.65,
    "dot": 0.75,
    "near": 0.75,
    "uni": 0.75,
    "sui": 0.75,
    "trx": 0.75,
    "xlm": 0.70,
    "shib": 0.70,
    "cake": 0.75,
    "floki": 0.65,
    "crv": 0.75,
    "icp": 0.85,
    "zen": 0.65,
}


def parse_args():
    p = argparse.ArgumentParser(description="Multi-model portfolio backtest")
    p.add_argument("--coins", required=True,
                   help="Comma-separated list of coins (e.g., btc,xrp,bnb,sol,doge)")
    p.add_argument("--thresholds", default=None,
                   help="Comma-separated thresholds (same order as --coins). Uses defaults if omitted.")
    p.add_argument("--exit-threshold", type=float, default=0.10,
                   help="Exit threshold (default: 0.10, matching live bot)")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--atr-mult", type=float, default=10.0)
    p.add_argument("--risk-per-trade", type=float, default=0.02)
    p.add_argument("--fee-bps", type=int, default=10)
    p.add_argument("--max-single-pct", type=float, default=0.40)
    return p.parse_args()


def main():
    args = parse_args()
    coins = [c.strip().lower() for c in args.coins.split(",")]

    # Parse thresholds
    if args.thresholds:
        thresh_list = [float(t) for t in args.thresholds.split(",")]
        if len(thresh_list) != len(coins):
            print(f"ERROR: {len(coins)} coins but {len(thresh_list)} thresholds")
            sys.exit(1)
        thresholds = dict(zip(coins, thresh_list))
    else:
        thresholds = {c: DEFAULT_THRESHOLDS.get(c, 0.65) for c in coins}

    print(f"Portfolio: {', '.join(c.upper() for c in coins)}")
    print(f"Thresholds: {thresholds}")
    print()

    # Load raw data
    print("Loading raw data...")
    raw = {}
    needed = set(coins) | {"btc", "eth", "sol"}  # always need BTC/ETH/SOL for cross-asset
    for coin in needed:
        if coin in DATA_PATHS:
            raw[coin] = _load_and_clean(DATA_PATHS[coin])
            print(f"  {coin.upper()}: {len(raw[coin]):,} bars")

    # Prepare features for each coin
    print("\nPreparing features...")
    features = {}
    for coin in coins:
        if coin == "btc":
            features[coin] = prepare_btc_features(
                raw["btc"], raw["eth"], raw["sol"], args.start, args.end
            )
        elif coin == "xrp":
            features[coin] = prepare_xrp_features(
                raw["btc"], raw["eth"], raw["sol"], raw["xrp"], args.start, args.end
            )
        else:
            features[coin] = prepare_alt_features(
                coin, raw["btc"], raw["eth"], raw["sol"], raw[coin], args.start, args.end
            )
        print(f"  {coin.upper()}: {features[coin].shape[0]:,} bars x {features[coin].shape[1]} cols")

    # Load models and predict
    print("\nLoading models and predicting...")
    coin_data = {}
    for coin in coins:
        model = load_model(MODEL_PATHS[coin])
        probas = batch_predict(model, features[coin])
        exit_thresh = args.exit_threshold

        coin_data[coin] = {
            "feat_df": features[coin],
            "probas": probas,
            "threshold": thresholds[coin],
            "exit_threshold": exit_thresh,
        }
        valid = (~np.isnan(probas)).sum()
        above = (probas >= thresholds[coin]).sum()
        print(f"  {coin.upper()}: {valid:,} valid probas, {above:,} above threshold {thresholds[coin]}")

    # Run backtest
    print("\nRunning portfolio backtest...")
    returns, portfolio, trades, gates, active_days = run_portfolio_backtest(
        coin_data,
        initial_capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        atr_mult=args.atr_mult,
        max_single_pct=args.max_single_pct,
        fee_bps=args.fee_bps,
    )

    label = " + ".join(c.upper() for c in coins) + f"  (capital=${args.capital:,.0f})"
    stats = compute_and_print_stats(returns, portfolio, trades, gates, active_days, label)

    return stats


if __name__ == "__main__":
    main()
