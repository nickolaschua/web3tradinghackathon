#!/usr/bin/env python3
"""
Multi-coin frequency backtest: MeanReversion + Momentum across all tradeable coins
alongside XGBoost BTC + SOL models.

Answers:
  1. How many trades/day does the combined strategy produce?
  2. How many unique active trading days out of the OOS period?
  3. Sharpe, Sortino, Calmar, and composite score?
  4. Breakdown by signal source (xgb_btc, xgb_sol, mr, momentum)?
  5. Does it meet the 8/10 active-day competition requirement?

Signal priority chain (per coin, per 15M bar):
  BTC/USD:  XGB_BTC → MR → Momentum
  SOL/USD:  XGB_SOL → MR → Momentum
  Others:   MR → Momentum

Risk stack (matches backtest_15m.py):
  ATR trailing stop (10x), 2% risk/trade, Kelly gate, tiered CB, 5% hard stop,
  max 5 concurrent positions, 10bps fee, 40% concentration cap.

BACKTEST-ONLY. No live trading, no Roostoo API calls.
"""

import json
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import (
    compute_btc_context_features,
    compute_cross_asset_features,
    compute_features,
)
from bot.strategy.base import SignalDirection
from bot.strategy.mean_reversion import MeanReversionStrategy
from bot.strategy.momentum import MomentumStrategy
from bot.config.unlock_screen import should_exclude

# ── Configuration ─────────────────────────────────────────────────────────────

PERIODS_15M = 35_040  # 365.25 × 24 × 4

CONFIG = {
    "max_positions": 5,
    "risk_per_trade_pct": 0.02,
    "hard_stop_pct": 0.05,
    "atr_stop_multiplier": 10.0,
    "expected_win_loss_ratio": 1.5,
    "max_single_position_pct": 0.40,
    "circuit_breaker": {
        "halt_threshold": 0.30,
        "reduce_heavy_threshold": 0.20,
        "reduce_light_threshold": 0.10,
    },
}
INITIAL_CAPITAL = 1_000_000.0
FEE_BPS = 10
BTC_THRESHOLD = 0.60
BTC_EXIT_THRESHOLD = 0.10
SOL_THRESHOLD = 0.75
SOL_EXIT_THRESHOLD = 0.10
OOS_START = "2024-01-01"

CORR_WINDOW = 2880  # 30 days at 15M

_BINANCE_TO_ROOSTOO: dict[str, str] = {
    "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "BNBUSDT": "BNB/USD",
    "SOLUSDT": "SOL/USD", "ADAUSDT": "ADA/USD", "AVAXUSDT": "AVAX/USD",
    "DOGEUSDT": "DOGE/USD", "LINKUSDT": "LINK/USD", "DOTUSDT": "DOT/USD",
    "UNIUSDT": "UNI/USD", "XRPUSDT": "XRP/USD", "LTCUSDT": "LTC/USD",
    "AAVEUSDT": "AAVE/USD", "CRVUSDT": "CRV/USD", "NEARUSDT": "NEAR/USD",
    "FILUSDT": "FIL/USD", "FETUSDT": "FET/USD", "HBARUSDT": "HBAR/USD",
    "ZECUSDT": "ZEC/USD", "ZENUSDT": "ZEN/USD", "CAKEUSDT": "CAKE/USD",
    "PAXGUSDT": "PAXG/USD", "XLMUSDT": "XLM/USD", "TRXUSDT": "TRX/USD",
    "CFXUSDT": "CFX/USD", "SHIBUSDT": "SHIB/USD", "ICPUSDT": "ICP/USD",
    "APTUSDT": "APT/USD", "ARBUSDT": "ARB/USD", "SUIUSDT": "SUI/USD",
    "FLOKIUSDT": "FLOKI/USD", "PEPEUSDT": "PEPE/USD",
    "PENDLEUSDT": "PENDLE/USD", "WLDUSDT": "WLD/USD", "SEIUSDT": "SEI/USD",
    "BONKUSDT": "BONK/USD", "WIFUSDT": "WIF/USD", "ENAUSDT": "ENA/USD",
    "TAOUSDT": "TAO/USD",
}

DATA_DIR = Path("data")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_15m_data() -> dict[str, pd.DataFrame]:
    """Load all available 15M parquets, return dict mapping Roostoo pair -> DataFrame."""
    loaded = {}
    for binance_sym, roostoo_pair in _BINANCE_TO_ROOSTOO.items():
        path = DATA_DIR / f"{binance_sym}_15m.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.columns = df.columns.str.lower()
            loaded[roostoo_pair] = df
            print(f"  Loaded {roostoo_pair}: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as exc:
            print(f"  WARNING: Failed to load {path}: {exc}")
    return loaded


# ── Feature Computation ───────────────────────────────────────────────────────

def compute_btc_features(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame, sol_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full BTC feature pipeline matching train_model_15m.py prepare_features().
    19 features: standard + cross-asset lags + BTC context.
    """
    feat = compute_features(btc_df)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth_df, "SOL/USD": sol_df})

    for asset, df in [("eth", eth_df), ("sol", sol_df)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    feat = compute_btc_context_features(feat, eth_df, sol_df, window=CORR_WINDOW)
    feat = feat.dropna()
    return feat


def compute_sol_features(
    sol_df: pd.DataFrame, btc_df: pd.DataFrame, eth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full SOL feature pipeline matching train_alt_15m.py prepare_features(target="sol").
    19 features: standard + cross-asset lags + SOL/BTC corr/beta.
    """
    feat = compute_features(sol_df)
    feat = compute_cross_asset_features(feat, {"BTC/USD": btc_df, "ETH/USD": eth_df})

    for asset, df in [("btc", btc_df), ("eth", eth_df)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    sol_ret = np.log(sol_df["close"] / sol_df["close"].shift(1)).reindex(feat.index)
    btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1)).reindex(feat.index)

    corr = sol_ret.rolling(CORR_WINDOW).corr(btc_ret)
    cov = sol_ret.rolling(CORR_WINDOW).cov(btc_ret)
    var_btc = btc_ret.rolling(CORR_WINDOW).var()

    feat["sol_btc_corr"] = corr.shift(1)
    feat["sol_btc_beta"] = (cov / (var_btc + 1e-10)).shift(1)

    feat = feat.dropna()
    return feat


def compute_coin_features(
    coin_df: pd.DataFrame, btc_df: pd.DataFrame, eth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute features for any non-BTC/SOL coin using BTC/ETH as cross-asset references."""
    feat = compute_features(coin_df)
    feat = compute_cross_asset_features(feat, {"BTC/USD": btc_df, "ETH/USD": eth_df})
    feat = feat.dropna()
    return feat


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Model at {model_path} has no predict_proba()")
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(f"Model at {model_path} has no feature_names_in_")
    return model


def batch_predict(model, feat_df: pd.DataFrame) -> np.ndarray:
    feature_cols = list(model.feature_names_in_)
    X = feat_df.reindex(columns=feature_cols)
    return model.predict_proba(X)[:, 1]


# ── Circuit Breaker ───────────────────────────────────────────────────────────

def _cb_multiplier(drawdown: float) -> float:
    cb = CONFIG["circuit_breaker"]
    if drawdown >= cb["halt_threshold"]:
        return 0.0
    if drawdown >= cb["reduce_heavy_threshold"]:
        return 0.25
    if drawdown >= cb["reduce_light_threshold"]:
        return 0.50
    return 1.0


# ── Position State ────────────────────────────────────────────────────────────

@dataclass
class PositionState:
    pair: str
    entry_price: float
    units: float
    trail_stop: float
    source: str  # "xgb_btc" | "xgb_sol" | "mr" | "momentum"
    entry_bar: pd.Timestamp = None


# ── Signal Candidate ──────────────────────────────────────────────────────────

@dataclass
class SignalCandidate:
    pair: str
    direction: SignalDirection
    source: str
    confidence: float
    size: float
    close: float
    atr: float


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_multicoin_backtest(
    all_features: dict[str, pd.DataFrame],
    btc_probas: np.ndarray | None,
    sol_probas: np.ndarray | None,
    btc_feat_index: pd.DatetimeIndex | None,
    sol_feat_index: pd.DatetimeIndex | None,
) -> tuple[pd.Series, pd.Series, list[dict], dict]:
    """
    Bar-by-bar multi-coin backtest with the full risk stack.

    Returns:
        (returns_series, portfolio_series, closed_trades, gate_stats)
    """
    fee_rate = FEE_BPS / 10_000.0
    max_positions = CONFIG["max_positions"]
    risk_per_trade_pct = CONFIG["risk_per_trade_pct"]
    hard_stop_pct = CONFIG["hard_stop_pct"]
    atr_stop_multiplier = CONFIG["atr_stop_multiplier"]
    max_single_position_pct = CONFIG["max_single_position_pct"]
    expected_win_loss_ratio = CONFIG["expected_win_loss_ratio"]

    mr_strategy = MeanReversionStrategy()
    mom_strategy = MomentumStrategy()

    # Build BTC/SOL proba lookup dicts for O(1) bar access
    btc_proba_map: dict[pd.Timestamp, float] = {}
    if btc_probas is not None and btc_feat_index is not None:
        for ts, p in zip(btc_feat_index, btc_probas):
            btc_proba_map[ts] = p

    sol_proba_map: dict[pd.Timestamp, float] = {}
    if sol_probas is not None and sol_feat_index is not None:
        for ts, p in zip(sol_feat_index, sol_probas):
            sol_proba_map[ts] = p

    # Build common timestamp index across all coins (union of OOS timestamps)
    oos_ts = pd.Timestamp(OOS_START, tz="UTC")
    all_timestamps = set()
    for pair, feat_df in all_features.items():
        ts_oos = feat_df.index[feat_df.index >= oos_ts]
        all_timestamps.update(ts_oos)
    common_index = pd.DatetimeIndex(sorted(all_timestamps))

    if len(common_index) == 0:
        print("ERROR: No OOS timestamps found.")
        return pd.Series(dtype=float), pd.Series(dtype=float), [], {}

    n = len(common_index)
    print(f"  Simulation: {n:,} bars from {common_index[0]} to {common_index[-1]}")
    print(f"  Coins in universe: {len(all_features)}")

    # Pre-index: for each coin, build a dict of ts -> row index for O(1) lookup
    coin_row_lookup: dict[str, dict[pd.Timestamp, int]] = {}
    for pair, feat_df in all_features.items():
        oos_df = feat_df[feat_df.index >= oos_ts]
        coin_row_lookup[pair] = {ts: i for i, ts in enumerate(oos_df.index)}

    # Pre-slice OOS feature matrices
    coin_oos_features: dict[str, pd.DataFrame] = {}
    for pair, feat_df in all_features.items():
        coin_oos_features[pair] = feat_df[feat_df.index >= oos_ts]

    # Portfolio state
    free_balance = INITIAL_CAPITAL
    portfolio_hwm = INITIAL_CAPITAL
    positions: dict[str, PositionState] = {}

    portfolio_values = np.zeros(n)
    portfolio_values[0] = INITIAL_CAPITAL
    returns = np.zeros(n)
    closed_trades: list[dict] = []
    gate_stats = {
        "kelly_blocked": 0, "cb_halted": 0, "cb_reduced": 0,
        "unlock_blocked": 0, "max_pos_blocked": 0,
    }

    t_start = time.time()

    for bar_idx in range(n):
        ts = common_index[bar_idx]

        # ── 1. Update trailing stops for all open positions ──────────────
        for pair, pos in list(positions.items()):
            feat_df = coin_oos_features.get(pair)
            if feat_df is None or ts not in coin_row_lookup.get(pair, {}):
                continue
            row_idx = coin_row_lookup[pair][ts]
            row = feat_df.iloc[row_idx]
            c = row["close"]
            atr = row.get("atr_proxy", np.nan)

            if not np.isnan(atr) and atr > 0:
                new_atr_stop = c - atr_stop_multiplier * atr
                pos.trail_stop = max(pos.trail_stop, new_atr_stop)

        # ── 2. Check exits ───────────────────────────────────────────────
        pairs_just_exited = set()
        for pair, pos in list(positions.items()):
            feat_df = coin_oos_features.get(pair)
            if feat_df is None or ts not in coin_row_lookup.get(pair, {}):
                continue
            row_idx = coin_row_lookup[pair][ts]
            row = feat_df.iloc[row_idx]
            c = row["close"]

            stop_hit = c <= pos.trail_stop

            # Source-specific signal exit
            sell_signal = False
            if pos.source == "xgb_btc":
                p = btc_proba_map.get(ts, 0.5)
                sell_signal = p <= BTC_EXIT_THRESHOLD
            elif pos.source == "xgb_sol":
                p = sol_proba_map.get(ts, 0.5)
                sell_signal = p <= SOL_EXIT_THRESHOLD
            elif pos.source == "mr":
                mr_sig = mr_strategy.generate_signal(pair, feat_df.iloc[row_idx:row_idx+1])
                sell_signal = mr_sig.direction == SignalDirection.SELL
            elif pos.source == "momentum":
                mom_sig = mom_strategy.generate_signal(pair, feat_df.iloc[row_idx:row_idx+1])
                sell_signal = mom_sig.direction == SignalDirection.SELL

            if stop_hit or sell_signal:
                proceeds = pos.units * c * (1.0 - fee_rate)
                net_exit = c * (1.0 - fee_rate)
                pnl_pct = (net_exit - pos.entry_price) / pos.entry_price

                closed_trades.append({
                    "pair": pair,
                    "entry_bar": pos.entry_bar,
                    "exit_bar": ts,
                    "entry_price": pos.entry_price,
                    "exit_price": net_exit,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "stop" if stop_hit else "signal",
                    "source": pos.source,
                })

                free_balance += proceeds
                del positions[pair]
                pairs_just_exited.add(pair)

        # ── 3. Mark to market ────────────────────────────────────────────
        position_value = 0.0
        for pair, pos in positions.items():
            feat_df = coin_oos_features.get(pair)
            if feat_df is not None and ts in coin_row_lookup.get(pair, {}):
                row_idx = coin_row_lookup[pair][ts]
                c = feat_df.iloc[row_idx]["close"]
                position_value += pos.units * c

        total_portfolio = free_balance + position_value
        portfolio_hwm = max(portfolio_hwm, total_portfolio)
        drawdown = (portfolio_hwm - total_portfolio) / portfolio_hwm if portfolio_hwm > 0 else 0.0
        cb_mult = _cb_multiplier(drawdown)

        # ── 4. Collect new BUY signals across all coins ──────────────────
        candidates: list[SignalCandidate] = []

        for pair, feat_df in coin_oos_features.items():
            if pair in positions or pair in pairs_just_exited:
                continue
            if ts not in coin_row_lookup.get(pair, {}):
                continue

            row_idx = coin_row_lookup[pair][ts]
            row = feat_df.iloc[row_idx]
            c = row["close"]
            atr = row.get("atr_proxy", np.nan)
            single_row = feat_df.iloc[row_idx:row_idx+1]

            signal_source = None
            signal_confidence = 0.0
            signal_size = 0.0
            signal_direction = SignalDirection.HOLD

            # Priority chain
            if pair == "BTC/USD" and btc_proba_map:
                p = btc_proba_map.get(ts)
                if p is not None and p >= BTC_THRESHOLD:
                    signal_source = "xgb_btc"
                    signal_confidence = p
                    signal_size = 1.0
                    signal_direction = SignalDirection.BUY

            if signal_direction == SignalDirection.HOLD and pair == "SOL/USD" and sol_proba_map:
                p = sol_proba_map.get(ts)
                if p is not None and p >= SOL_THRESHOLD:
                    signal_source = "xgb_sol"
                    signal_confidence = p
                    signal_size = 1.0
                    signal_direction = SignalDirection.BUY

            if signal_direction == SignalDirection.HOLD:
                mr_sig = mr_strategy.generate_signal(pair, single_row)
                if mr_sig.direction == SignalDirection.BUY:
                    signal_source = "mr"
                    signal_confidence = mr_sig.confidence
                    signal_size = mr_sig.size
                    signal_direction = SignalDirection.BUY

            if signal_direction == SignalDirection.HOLD:
                mom_sig = mom_strategy.generate_signal(pair, single_row)
                if mom_sig.direction == SignalDirection.BUY:
                    signal_source = "momentum"
                    signal_confidence = mom_sig.confidence
                    signal_size = mom_sig.size
                    signal_direction = SignalDirection.BUY

            if signal_direction == SignalDirection.BUY:
                candidates.append(SignalCandidate(
                    pair=pair,
                    direction=signal_direction,
                    source=signal_source,
                    confidence=signal_confidence,
                    size=signal_size,
                    close=c,
                    atr=atr,
                ))

        # ── 5. Rank candidates by confidence (highest first) and fill ────
        candidates.sort(key=lambda s: s.confidence, reverse=True)

        for cand in candidates:
            if len(positions) >= max_positions:
                gate_stats["max_pos_blocked"] += 1
                continue

            if should_exclude(cand.pair):
                gate_stats["unlock_blocked"] += 1
                continue

            c = cand.close
            atr = cand.atr

            if cand.source in ("xgb_btc", "xgb_sol"):
                # XGBoost entry: equal dollar risk sizing
                kelly = (cand.confidence * expected_win_loss_ratio - (1.0 - cand.confidence)) / expected_win_loss_ratio
                if kelly <= 0:
                    gate_stats["kelly_blocked"] += 1
                    continue
                if cb_mult == 0.0:
                    gate_stats["cb_halted"] += 1
                    continue
                if cb_mult < 1.0:
                    gate_stats["cb_reduced"] += 1

                hard_stop_price = c * (1.0 - hard_stop_pct)
                if not np.isnan(atr) and atr > 0:
                    atr_stop_price = c - atr_stop_multiplier * atr
                    initial_stop = max(hard_stop_price, atr_stop_price)
                else:
                    initial_stop = hard_stop_price

                stop_distance = c - initial_stop
                stop_distance = min(stop_distance, c * hard_stop_pct)
                if stop_distance <= 0:
                    stop_distance = c * hard_stop_pct

                risk_usd = total_portfolio * risk_per_trade_pct * cand.confidence * cb_mult
                quantity = risk_usd / stop_distance
                target_usd = quantity * c

                usable = free_balance * 0.95
                target_usd = min(target_usd, total_portfolio * max_single_position_pct, usable)

            else:
                # MR / Momentum entry: portfolio weight sizing
                if cb_mult == 0.0:
                    gate_stats["cb_halted"] += 1
                    continue
                if cb_mult < 1.0:
                    gate_stats["cb_reduced"] += 1

                hard_stop_price = c * (1.0 - hard_stop_pct)
                if not np.isnan(atr) and atr > 0:
                    atr_stop_price = c - atr_stop_multiplier * atr
                    initial_stop = max(hard_stop_price, atr_stop_price)
                else:
                    initial_stop = hard_stop_price

                target_usd = total_portfolio * cand.size * cb_mult
                usable = free_balance * 0.95
                target_usd = min(target_usd, usable)

            if target_usd >= 10.0:
                entry_units = target_usd / c
                entry_fee = target_usd * fee_rate
                free_balance -= (target_usd + entry_fee)
                entry_effective_price = c * (1.0 + fee_rate)

                positions[cand.pair] = PositionState(
                    pair=cand.pair,
                    entry_price=entry_effective_price,
                    units=entry_units,
                    trail_stop=initial_stop,
                    source=cand.source,
                    entry_bar=ts,
                )

                # Recalculate portfolio value after entry
                position_value = sum(
                    pos.units * coin_oos_features[pos.pair].iloc[
                        coin_row_lookup[pos.pair][ts]
                    ]["close"]
                    for pos in positions.values()
                    if ts in coin_row_lookup.get(pos.pair, {})
                )
                total_portfolio = free_balance + position_value

        # ── 6. Record bar-end portfolio value ────────────────────────────
        position_value = 0.0
        for pair, pos in positions.items():
            feat_df = coin_oos_features.get(pair)
            if feat_df is not None and ts in coin_row_lookup.get(pair, {}):
                row_idx = coin_row_lookup[pair][ts]
                position_value += pos.units * feat_df.iloc[row_idx]["close"]

        portfolio_values[bar_idx] = free_balance + position_value
        if bar_idx > 0:
            returns[bar_idx] = portfolio_values[bar_idx] / portfolio_values[bar_idx - 1] - 1.0

        # Progress reporting
        if bar_idx > 0 and bar_idx % 10_000 == 0:
            elapsed = time.time() - t_start
            pct = bar_idx / n * 100
            print(f"    [{pct:5.1f}%] bar {bar_idx:,}/{n:,}  trades={len(closed_trades)}  "
                  f"open={len(positions)}  elapsed={elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"  Simulation complete: {len(closed_trades)} closed trades, "
          f"{len(positions)} still open, {elapsed:.1f}s")

    returns_series = pd.Series(returns, index=common_index)
    portfolio_series = pd.Series(portfolio_values, index=common_index)
    return returns_series, portfolio_series, closed_trades, gate_stats


# ── Reporting ─────────────────────────────────────────────────────────────────

def compute_report(
    returns: pd.Series,
    portfolio: pd.Series,
    closed_trades: list[dict],
    gate_stats: dict,
) -> dict:
    """Compute all metrics for the final report."""
    n_trades = len(closed_trades)

    # Per-source breakdown
    source_counts = defaultdict(int)
    for t in closed_trades:
        source_counts[t["source"]] += 1

    # Per-coin breakdown
    coin_counts = defaultdict(int)
    for t in closed_trades:
        coin_counts[t["pair"]] += 1

    # Win/loss stats
    if n_trades > 0:
        winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        win_rate = winners / n_trades
        avg_pnl = sum(t["pnl_pct"] for t in closed_trades) / n_trades
        stop_exits = sum(1 for t in closed_trades if t["exit_reason"] == "stop")
    else:
        win_rate = avg_pnl = 0.0
        stop_exits = 0

    # Active trading days
    trade_dates = set()
    for t in closed_trades:
        entry_date = pd.Timestamp(t["entry_bar"]).date()
        exit_date = pd.Timestamp(t["exit_bar"]).date()
        trade_dates.add(entry_date)
        trade_dates.add(exit_date)

    total_oos_days = (portfolio.index[-1] - portfolio.index[0]).days + 1
    active_days = len(trade_dates)

    # 10-day sliding window check
    if trade_dates:
        sorted_dates = sorted(trade_dates)
        all_dates = pd.date_range(portfolio.index[0].date(), portfolio.index[-1].date(), freq="D")
        min_active_in_10 = total_oos_days  # worst case
        for i in range(len(all_dates) - 9):
            window = set(all_dates[i:i+10].date)
            active_in_window = len(window & trade_dates)
            min_active_in_10 = min(min_active_in_10, active_in_window)
    else:
        min_active_in_10 = 0

    # Risk metrics
    returns_clean = returns[returns.index.notna()]
    sharpe = float(qs.stats.sharpe(returns_clean, periods=PERIODS_15M))
    sortino = float(qs.stats.sortino(returns_clean, periods=PERIODS_15M))
    calmar = float(qs.stats.calmar(returns_clean))
    max_dd = float(qs.stats.max_drawdown(returns_clean)) * 100
    total_return = (portfolio.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    # Trades per day
    trades_per_day = n_trades / total_oos_days if total_oos_days > 0 else 0.0

    return {
        "n_trades": n_trades,
        "trades_per_day": trades_per_day,
        "active_days": active_days,
        "total_oos_days": total_oos_days,
        "active_pct": active_days / total_oos_days * 100 if total_oos_days > 0 else 0.0,
        "source_counts": dict(source_counts),
        "coin_counts": dict(coin_counts),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "composite": composite,
        "max_drawdown_pct": max_dd,
        "total_return_pct": total_return,
        "win_rate_pct": win_rate * 100,
        "avg_pnl_pct": avg_pnl * 100,
        "stop_exits": stop_exits,
        "signal_exits": n_trades - stop_exits,
        "min_active_in_10day_window": min_active_in_10,
        "meets_8_of_10": min_active_in_10 >= 8,
        "gate_stats": gate_stats,
        "final_portfolio": portfolio.iloc[-1],
    }


def print_report(report: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  MULTI-COIN FREQUENCY BACKTEST")
    print(f"  OOS Period: {OOS_START} to present")
    print(sep)
    print(f"  Total trades:        {report['n_trades']}")
    print(f"  Trades/day:          {report['trades_per_day']:.2f}")
    print(f"  Active trading days: {report['active_days']} / {report['total_oos_days']} ({report['active_pct']:.1f}%)")
    print()
    print("  Trades by source:")
    for src in ["xgb_btc", "xgb_sol", "mr", "momentum"]:
        cnt = report["source_counts"].get(src, 0)
        pct = cnt / report["n_trades"] * 100 if report["n_trades"] > 0 else 0
        print(f"    {src:12s}: {cnt:>4d} ({pct:5.1f}%)")
    print()

    print("  Trades by coin (top 10):")
    sorted_coins = sorted(report["coin_counts"].items(), key=lambda x: x[1], reverse=True)
    for pair, cnt in sorted_coins[:10]:
        print(f"    {pair:12s}: {cnt:>4d}")
    print()

    print("  Risk-Adjusted Metrics:")
    print(f"    Sharpe:     {report['sharpe']:.3f}")
    print(f"    Sortino:    {report['sortino']:.3f}")
    print(f"    Calmar:     {report['calmar']:.3f}")
    print(f"    Composite:  {report['composite']:.3f} (0.4*Sortino + 0.3*Sharpe + 0.3*Calmar)")
    print()
    print(f"  Total return:   {report['total_return_pct']:+.2f}%")
    print(f"  Final portfolio: ${report['final_portfolio']:,.2f}")
    print(f"  Max drawdown:   {report['max_drawdown_pct']:.2f}%")
    print(f"  Win rate:       {report['win_rate_pct']:.1f}%")
    print(f"  Avg trade PnL:  {report['avg_pnl_pct']:+.2f}%")
    print(f"  Stop exits:     {report['stop_exits']}  Signal exits: {report['signal_exits']}")
    print()
    print("  Gate stats:")
    for k, v in report["gate_stats"].items():
        print(f"    {k:20s}: {v}")
    print()
    print("  Competition feasibility:")
    print(f"    Min active days in any 10-day window: {report['min_active_in_10day_window']}/10")
    print(f"    Meets 8/10 requirement: {'YES' if report['meets_8_of_10'] else 'NO'}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MULTI-COIN FREQUENCY BACKTEST")
    print("=" * 60)

    # Step 1: Load all available parquet data
    print("\nStep 1: Loading 15M parquet data...")
    raw_data = load_all_15m_data()
    if not raw_data:
        print("ERROR: No parquet data found in data/")
        sys.exit(1)

    btc_df = raw_data.get("BTC/USD")
    eth_df = raw_data.get("ETH/USD")
    sol_df = raw_data.get("SOL/USD")

    if btc_df is None or eth_df is None:
        print("ERROR: BTC and ETH parquets are required (cross-asset reference)")
        sys.exit(1)

    # Step 2: Compute features for all coins
    print(f"\nStep 2: Computing features for {len(raw_data)} coins...")
    all_features: dict[str, pd.DataFrame] = {}

    print("  Computing BTC features (full pipeline)...")
    btc_feat = compute_btc_features(btc_df, eth_df, sol_df)
    all_features["BTC/USD"] = btc_feat
    print(f"    BTC/USD: {len(btc_feat):,} bars, {btc_feat.shape[1]} columns")

    if sol_df is not None:
        print("  Computing SOL features (full pipeline)...")
        sol_feat = compute_sol_features(sol_df, btc_df, eth_df)
        all_features["SOL/USD"] = sol_feat
        print(f"    SOL/USD: {len(sol_feat):,} bars, {sol_feat.shape[1]} columns")

    for pair, coin_df in raw_data.items():
        if pair in ("BTC/USD", "SOL/USD"):
            continue
        print(f"  Computing features for {pair}...")
        feat = compute_coin_features(coin_df, btc_df, eth_df)
        all_features[pair] = feat
        print(f"    {pair}: {len(feat):,} bars, {feat.shape[1]} columns")

    # Step 3: Load XGBoost models
    print("\nStep 3: Loading XGBoost models...")
    btc_model_path = "models/xgb_btc_15m_iter5.pkl"
    sol_model_path = "models/xgb_sol_15m.pkl"

    btc_model = None
    sol_model = None
    btc_probas = None
    sol_probas = None
    btc_feat_index = None
    sol_feat_index = None

    if Path(btc_model_path).exists():
        btc_model = load_model(btc_model_path)
        print(f"  BTC model: {len(btc_model.feature_names_in_)} features, threshold={BTC_THRESHOLD}")

        oos_ts = pd.Timestamp(OOS_START, tz="UTC")
        btc_oos = btc_feat[btc_feat.index >= oos_ts]
        btc_probas = batch_predict(btc_model, btc_oos)
        btc_feat_index = btc_oos.index
        print(f"  BTC OOS predictions: {len(btc_probas):,} bars, "
              f"mean P(BUY)={btc_probas.mean():.3f}, "
              f">={BTC_THRESHOLD}: {(btc_probas >= BTC_THRESHOLD).sum():,}")
    else:
        print(f"  WARNING: BTC model not found at {btc_model_path}")

    if Path(sol_model_path).exists() and "SOL/USD" in all_features:
        sol_model = load_model(sol_model_path)
        print(f"  SOL model: {len(sol_model.feature_names_in_)} features, threshold={SOL_THRESHOLD}")

        oos_ts = pd.Timestamp(OOS_START, tz="UTC")
        sol_oos = sol_feat[sol_feat.index >= oos_ts]
        sol_probas = batch_predict(sol_model, sol_oos)
        sol_feat_index = sol_oos.index
        print(f"  SOL OOS predictions: {len(sol_probas):,} bars, "
              f"mean P(BUY)={sol_probas.mean():.3f}, "
              f">={SOL_THRESHOLD}: {(sol_probas >= SOL_THRESHOLD).sum():,}")
    else:
        print(f"  WARNING: SOL model not found at {sol_model_path}")

    # Step 4: Run backtest
    print("\nStep 4: Running multi-coin backtest simulation...")
    returns, portfolio, closed_trades, gate_stats = run_multicoin_backtest(
        all_features=all_features,
        btc_probas=btc_probas,
        sol_probas=sol_probas,
        btc_feat_index=btc_feat_index,
        sol_feat_index=sol_feat_index,
    )

    if returns.empty:
        print("ERROR: No returns generated.")
        sys.exit(1)

    # Step 5: Compute and display report
    print("\nStep 5: Computing report...")
    report = compute_report(returns, portfolio, closed_trades, gate_stats)
    print_report(report)

    # Step 6: Save results to JSON
    output_path = Path("research_results/multicoin_frequency_backtest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_report = {k: v for k, v in report.items()}
    json_report["oos_start"] = OOS_START
    json_report["initial_capital"] = INITIAL_CAPITAL
    json_report["config"] = CONFIG
    json_report["btc_threshold"] = BTC_THRESHOLD
    json_report["sol_threshold"] = SOL_THRESHOLD
    json_report["fee_bps"] = FEE_BPS
    json_report["coins_loaded"] = list(all_features.keys())
    json_report["n_coins"] = len(all_features)

    # Convert non-serializable types
    for k, v in json_report.items():
        if isinstance(v, (np.integer, np.int64)):
            json_report[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            json_report[k] = float(v)

    with open(output_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
