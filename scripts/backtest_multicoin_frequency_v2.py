#!/usr/bin/env python3
"""
Multi-Coin Frequency Backtest v2
=================================
Changes from v1:
  - SOL threshold lowered: 0.75 → 0.55 (only 2 trades at 0.75)
  - Added RELAXED MR/Momentum signal generators (backtest-only overrides)
    that bypass the strict strategy module conditions. These do NOT modify
    the strategy files — they're inline functions in this script.
  - Designed for full 39-coin universe (run download_all_15m.py first)
  - Added per-source PnL breakdown and rolling active-day analysis

Run:
    python scripts/download_all_15m.py          # step 1: get data
    python scripts/backtest_multicoin_frequency_v2.py  # step 2: backtest

Signal priority chain (per coin, per 15M bar):
  BTC/USD:  XGB_BTC(0.60) → RelaxedMR → RelaxedMomentum
  SOL/USD:  XGB_SOL(0.55) → RelaxedMR → RelaxedMomentum
  Others:   RelaxedMR → RelaxedMomentum

Relaxed MR BUY:   RSI<40  AND  bb_pos<0.30  (was RSI<30 AND bb_pos<0.15 AND MACD_hist>0)
Relaxed MR SELL:  RSI>55  OR   bb_pos>0.60
Relaxed Mom BUY:  RSI<55  AND  MACD_hist>0   (was RSI<50 AND MACD_hist>0 AND EMA regime)
Relaxed Mom SELL: RSI>60  OR   MACD_hist<0   (was RSI>65)
"""
import json
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
SOL_THRESHOLD = 0.55       # ← lowered from 0.75
SOL_EXIT_THRESHOLD = 0.10
OOS_START = "2024-01-01"
CORR_WINDOW = 2880

# ── Relaxed strategy thresholds (backtest-only, does not modify strategy files) ──

RELAXED_MR_RSI_BUY = 40       # was 30
RELAXED_MR_BB_POS_BUY = 0.30  # was 0.15
RELAXED_MR_RSI_SELL = 55
RELAXED_MR_BB_POS_SELL = 0.60

RELAXED_MOM_RSI_BUY = 55      # was 50
RELAXED_MOM_RSI_SELL = 60     # was 65


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
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            loaded[roostoo_pair] = df
            print(f"  {roostoo_pair}: {len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as exc:
            print(f"  WARNING: {path}: {exc}")
    return loaded


# ── Feature Computation ───────────────────────────────────────────────────────

def compute_btc_features(btc_df, eth_df, sol_df):
    feat = compute_features(btc_df)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth_df, "SOL/USD": sol_df})
    for asset, df in [("eth", eth_df), ("sol", sol_df)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)
    feat = compute_btc_context_features(feat, eth_df, sol_df, window=CORR_WINDOW)
    return feat.dropna()


def compute_sol_features(sol_df, btc_df, eth_df):
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
    return feat.dropna()


def compute_coin_features(coin_df, btc_df, eth_df):
    feat = compute_features(coin_df)
    feat = compute_cross_asset_features(feat, {"BTC/USD": btc_df, "ETH/USD": eth_df})
    return feat.dropna()


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict_proba") and hasattr(model, "feature_names_in_")
    return model


def batch_predict(model, feat_df):
    X = feat_df.reindex(columns=list(model.feature_names_in_))
    return model.predict_proba(X)[:, 1]


# ── Relaxed Signal Generators (backtest-only) ────────────────────────────────

def relaxed_mr_signal(row) -> tuple[str, float]:
    """
    Relaxed mean-reversion BUY/SELL using looser thresholds.
    Returns (direction, confidence).

    Original MR required: RSI<30 AND bb_pos<0.15 AND MACD_hist>0
    Relaxed:             RSI<40 AND bb_pos<0.30
    """
    rsi = row.get("RSI_14", np.nan)
    bb_pos = row.get("bb_pos", np.nan)

    if pd.isna(rsi) or pd.isna(bb_pos):
        return "HOLD", 0.0

    if rsi < RELAXED_MR_RSI_BUY and bb_pos < RELAXED_MR_BB_POS_BUY:
        conf = min(0.9, 0.5 + (RELAXED_MR_RSI_BUY - rsi) / 40)
        return "BUY", conf

    if rsi > RELAXED_MR_RSI_SELL or bb_pos > RELAXED_MR_BB_POS_SELL:
        return "SELL", 0.5

    return "HOLD", 0.0


def relaxed_momentum_signal(row) -> tuple[str, float]:
    """
    Relaxed momentum BUY/SELL.

    Original Mom required: RSI<50 AND MACD_hist>0 AND EMA_20>EMA_50
    Relaxed:              RSI<55 AND MACD_hist>0
    Exit:                 RSI>60 OR MACD_hist<0 (was RSI>65)
    """
    rsi = row.get("RSI_14", np.nan)
    macd_hist = row.get("MACDh_12_26_9", np.nan)

    if pd.isna(rsi) or pd.isna(macd_hist):
        return "HOLD", 0.0

    if rsi < RELAXED_MOM_RSI_BUY and macd_hist > 0:
        conf = min(0.8, 0.4 + macd_hist * 100)
        return "BUY", max(conf, 0.3)

    if rsi > RELAXED_MOM_RSI_SELL or macd_hist < 0:
        return "SELL", 0.5

    return "HOLD", 0.0


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
    source: str
    entry_bar: pd.Timestamp = None


@dataclass
class SignalCandidate:
    pair: str
    direction: str  # "BUY"
    source: str
    confidence: float
    size: float
    close: float
    atr: float


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest(
    all_features: dict[str, pd.DataFrame],
    btc_probas: np.ndarray | None,
    sol_probas: np.ndarray | None,
    btc_feat_index: pd.DatetimeIndex | None,
    sol_feat_index: pd.DatetimeIndex | None,
    use_relaxed: bool = True,
):
    fee_rate = FEE_BPS / 10_000.0
    max_positions = CONFIG["max_positions"]
    risk_per_trade_pct = CONFIG["risk_per_trade_pct"]
    hard_stop_pct = CONFIG["hard_stop_pct"]
    atr_stop_multiplier = CONFIG["atr_stop_multiplier"]
    max_single_position_pct = CONFIG["max_single_position_pct"]
    expected_win_loss_ratio = CONFIG["expected_win_loss_ratio"]

    mr_strategy = MeanReversionStrategy()
    mom_strategy = MomentumStrategy()

    btc_proba_map = {}
    if btc_probas is not None and btc_feat_index is not None:
        btc_proba_map = dict(zip(btc_feat_index, btc_probas))
    sol_proba_map = {}
    if sol_probas is not None and sol_feat_index is not None:
        sol_proba_map = dict(zip(sol_feat_index, sol_probas))

    oos_ts = pd.Timestamp(OOS_START, tz="UTC")
    all_timestamps = set()
    for feat_df in all_features.values():
        all_timestamps.update(feat_df.index[feat_df.index >= oos_ts])
    common_index = pd.DatetimeIndex(sorted(all_timestamps))

    if len(common_index) == 0:
        print("ERROR: No OOS timestamps found.")
        return pd.Series(dtype=float), pd.Series(dtype=float), [], {}

    n = len(common_index)
    print(f"  Simulation: {n:,} bars, {len(all_features)} coins")
    print(f"  Period: {common_index[0].date()} → {common_index[-1].date()}")
    print(f"  Mode: {'RELAXED' if use_relaxed else 'ORIGINAL'} MR/Momentum thresholds")
    print(f"  SOL threshold: {SOL_THRESHOLD}")

    coin_row_lookup: dict[str, dict[pd.Timestamp, int]] = {}
    coin_oos_features: dict[str, pd.DataFrame] = {}
    for pair, feat_df in all_features.items():
        oos_df = feat_df[feat_df.index >= oos_ts]
        coin_row_lookup[pair] = {ts: i for i, ts in enumerate(oos_df.index)}
        coin_oos_features[pair] = oos_df

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

        # ── 1. Update trailing stops ─────────────────────────────────────
        for pair, pos in list(positions.items()):
            if ts not in coin_row_lookup.get(pair, {}):
                continue
            row_idx = coin_row_lookup[pair][ts]
            row = coin_oos_features[pair].iloc[row_idx]
            c = row["close"]
            atr = row.get("atr_proxy", np.nan)
            if not np.isnan(atr) and atr > 0:
                pos.trail_stop = max(pos.trail_stop, c - atr_stop_multiplier * atr)

        # ── 2. Check exits ───────────────────────────────────────────────
        pairs_just_exited = set()
        for pair, pos in list(positions.items()):
            if ts not in coin_row_lookup.get(pair, {}):
                continue
            row_idx = coin_row_lookup[pair][ts]
            row = coin_oos_features[pair].iloc[row_idx]
            c = row["close"]

            stop_hit = c <= pos.trail_stop

            sell_signal = False
            if pos.source == "xgb_btc":
                p = btc_proba_map.get(ts, 0.5)
                sell_signal = p <= BTC_EXIT_THRESHOLD
            elif pos.source == "xgb_sol":
                p = sol_proba_map.get(ts, 0.5)
                sell_signal = p <= SOL_EXIT_THRESHOLD
            elif pos.source == "mr":
                if use_relaxed:
                    sig, _ = relaxed_mr_signal(row)
                    sell_signal = sig == "SELL"
                else:
                    single_row = coin_oos_features[pair].iloc[row_idx:row_idx+1]
                    sell_signal = mr_strategy.generate_signal(pair, single_row).direction == SignalDirection.SELL
            elif pos.source == "momentum":
                if use_relaxed:
                    sig, _ = relaxed_momentum_signal(row)
                    sell_signal = sig == "SELL"
                else:
                    single_row = coin_oos_features[pair].iloc[row_idx:row_idx+1]
                    sell_signal = mom_strategy.generate_signal(pair, single_row).direction == SignalDirection.SELL

            if stop_hit or sell_signal:
                proceeds = pos.units * c * (1.0 - fee_rate)
                net_exit = c * (1.0 - fee_rate)
                pnl_pct = (net_exit - pos.entry_price) / pos.entry_price
                closed_trades.append({
                    "pair": pair, "entry_bar": pos.entry_bar, "exit_bar": ts,
                    "entry_price": pos.entry_price, "exit_price": net_exit,
                    "pnl_pct": pnl_pct, "exit_reason": "stop" if stop_hit else "signal",
                    "source": pos.source,
                })
                free_balance += proceeds
                del positions[pair]
                pairs_just_exited.add(pair)

        # ── 3. Mark to market ────────────────────────────────────────────
        position_value = 0.0
        for pair, pos in positions.items():
            if ts in coin_row_lookup.get(pair, {}):
                c = coin_oos_features[pair].iloc[coin_row_lookup[pair][ts]]["close"]
                position_value += pos.units * c
        total_portfolio = free_balance + position_value
        portfolio_hwm = max(portfolio_hwm, total_portfolio)
        drawdown = (portfolio_hwm - total_portfolio) / portfolio_hwm if portfolio_hwm > 0 else 0.0
        cb_mult = _cb_multiplier(drawdown)

        # ── 4. Collect new BUY signals ───────────────────────────────────
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
            got_buy = False

            # ── Priority: XGBoost BTC ────────────────────────────────────
            if pair == "BTC/USD" and btc_proba_map:
                p = btc_proba_map.get(ts)
                if p is not None and p >= BTC_THRESHOLD:
                    signal_source = "xgb_btc"
                    signal_confidence = p
                    signal_size = 1.0
                    got_buy = True

            # ── Priority: XGBoost SOL ────────────────────────────────────
            if not got_buy and pair == "SOL/USD" and sol_proba_map:
                p = sol_proba_map.get(ts)
                if p is not None and p >= SOL_THRESHOLD:
                    signal_source = "xgb_sol"
                    signal_confidence = p
                    signal_size = 1.0
                    got_buy = True

            # ── Priority: Mean Reversion ─────────────────────────────────
            if not got_buy:
                if use_relaxed:
                    sig, conf = relaxed_mr_signal(row)
                    if sig == "BUY":
                        signal_source = "mr"
                        signal_confidence = conf
                        signal_size = 0.20
                        got_buy = True
                else:
                    mr_sig = mr_strategy.generate_signal(pair, single_row)
                    if mr_sig.direction == SignalDirection.BUY:
                        signal_source = "mr"
                        signal_confidence = mr_sig.confidence
                        signal_size = mr_sig.size
                        got_buy = True

            # ── Priority: Momentum ───────────────────────────────────────
            if not got_buy:
                if use_relaxed:
                    sig, conf = relaxed_momentum_signal(row)
                    if sig == "BUY":
                        signal_source = "momentum"
                        signal_confidence = conf
                        signal_size = 0.15
                        got_buy = True
                else:
                    mom_sig = mom_strategy.generate_signal(pair, single_row)
                    if mom_sig.direction == SignalDirection.BUY:
                        signal_source = "momentum"
                        signal_confidence = mom_sig.confidence
                        signal_size = mom_sig.size
                        got_buy = True

            if got_buy:
                candidates.append(SignalCandidate(
                    pair=pair, direction="BUY", source=signal_source,
                    confidence=signal_confidence, size=signal_size,
                    close=c, atr=atr,
                ))

        # ── 5. Rank and fill ─────────────────────────────────────────────
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

            if cand.source in ("xgb_btc", "xgb_sol"):
                stop_distance = c - initial_stop
                stop_distance = min(stop_distance, c * hard_stop_pct)
                if stop_distance <= 0:
                    stop_distance = c * hard_stop_pct
                risk_usd = total_portfolio * risk_per_trade_pct * cand.confidence * cb_mult
                quantity = risk_usd / stop_distance
                target_usd = quantity * c
            else:
                target_usd = total_portfolio * cand.size * cb_mult

            usable = free_balance * 0.95
            target_usd = min(target_usd, total_portfolio * max_single_position_pct, usable)

            if target_usd >= 10.0:
                entry_units = target_usd / c
                entry_fee = target_usd * fee_rate
                free_balance -= (target_usd + entry_fee)
                positions[cand.pair] = PositionState(
                    pair=cand.pair,
                    entry_price=c * (1.0 + fee_rate),
                    units=entry_units,
                    trail_stop=initial_stop,
                    source=cand.source,
                    entry_bar=ts,
                )

        # ── 6. Record bar-end portfolio ──────────────────────────────────
        position_value = 0.0
        for pair, pos in positions.items():
            if ts in coin_row_lookup.get(pair, {}):
                position_value += pos.units * coin_oos_features[pair].iloc[coin_row_lookup[pair][ts]]["close"]
        portfolio_values[bar_idx] = free_balance + position_value
        if bar_idx > 0:
            returns[bar_idx] = portfolio_values[bar_idx] / portfolio_values[bar_idx - 1] - 1.0

        if bar_idx > 0 and bar_idx % 10_000 == 0:
            elapsed = time.time() - t_start
            pct = bar_idx / n * 100
            print(f"    [{pct:5.1f}%] bar {bar_idx:,}/{n:,}  trades={len(closed_trades)}  "
                  f"open={len(positions)}  equity=${portfolio_values[bar_idx]:,.0f}  "
                  f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"  Done: {len(closed_trades)} closed, {len(positions)} open, {elapsed:.1f}s")

    return (
        pd.Series(returns, index=common_index),
        pd.Series(portfolio_values, index=common_index),
        closed_trades,
        gate_stats,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def compute_report(returns, portfolio, closed_trades, gate_stats):
    n_trades = len(closed_trades)
    source_counts = defaultdict(int)
    source_pnl = defaultdict(float)
    coin_counts = defaultdict(int)

    for t in closed_trades:
        source_counts[t["source"]] += 1
        source_pnl[t["source"]] += t["pnl_pct"]
        coin_counts[t["pair"]] += 1

    winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0) if n_trades else 0
    win_rate = winners / n_trades if n_trades else 0.0
    avg_pnl = sum(t["pnl_pct"] for t in closed_trades) / n_trades if n_trades else 0.0
    stop_exits = sum(1 for t in closed_trades if t["exit_reason"] == "stop")

    trade_dates = set()
    for t in closed_trades:
        trade_dates.add(pd.Timestamp(t["entry_bar"]).date())
        trade_dates.add(pd.Timestamp(t["exit_bar"]).date())

    total_oos_days = (portfolio.index[-1] - portfolio.index[0]).days + 1
    active_days = len(trade_dates)

    if trade_dates:
        all_dates = pd.date_range(portfolio.index[0].date(), portfolio.index[-1].date(), freq="D")
        min_active_in_10 = total_oos_days
        for i in range(len(all_dates) - 9):
            window = set(all_dates[i:i+10].date)
            active_in_window = len(window & trade_dates)
            min_active_in_10 = min(min_active_in_10, active_in_window)
    else:
        min_active_in_10 = 0

    eq = portfolio.values
    rets = np.diff(eq) / eq[:-1]
    total_return = (eq[-1] / eq[0]) - 1.0
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-10
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(PERIODS_15M)

    down_rets = rets[rets < 0]
    down_std = np.std(down_rets, ddof=1) if len(down_rets) > 1 else 1e-10
    sortino = (mean_ret / (down_std + 1e-10)) * np.sqrt(PERIODS_15M)

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    calmar = (mean_ret * PERIODS_15M) / (abs(max_dd) + 1e-10) if max_dd < 0 else 0.0

    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    return {
        "n_trades": n_trades,
        "trades_per_day": n_trades / max(total_oos_days, 1),
        "active_days": active_days,
        "total_oos_days": total_oos_days,
        "active_pct": active_days / max(total_oos_days, 1) * 100,
        "source_counts": dict(source_counts),
        "source_avg_pnl": {k: v / max(source_counts[k], 1) * 100 for k, v in source_pnl.items()},
        "coin_counts": dict(coin_counts),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "composite": round(composite, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_pnl_pct": round(avg_pnl * 100, 2),
        "stop_exits": stop_exits,
        "signal_exits": n_trades - stop_exits,
        "min_active_in_10day_window": min_active_in_10,
        "meets_8_of_10": min_active_in_10 >= 8,
        "gate_stats": gate_stats,
        "final_portfolio": float(portfolio.iloc[-1]),
    }


def print_report(report):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  MULTI-COIN FREQUENCY BACKTEST v2")
    print(f"  OOS Period: {OOS_START} → present")
    print(f"  SOL threshold: {SOL_THRESHOLD}  (was 0.75)")
    print(f"  MR: relaxed (RSI<{RELAXED_MR_RSI_BUY}, bb<{RELAXED_MR_BB_POS_BUY})")
    print(f"  Mom: relaxed (RSI<{RELAXED_MOM_RSI_BUY}, exit RSI>{RELAXED_MOM_RSI_SELL})")
    print(sep)
    print(f"  Total trades:        {report['n_trades']}")
    print(f"  Trades/day:          {report['trades_per_day']:.2f}")
    print(f"  Active trading days: {report['active_days']} / {report['total_oos_days']} ({report['active_pct']:.1f}%)")
    print()
    print("  Trades by source:")
    for src in ["xgb_btc", "xgb_sol", "mr", "momentum"]:
        cnt = report["source_counts"].get(src, 0)
        pct = cnt / max(report["n_trades"], 1) * 100
        avg = report["source_avg_pnl"].get(src, 0)
        print(f"    {src:12s}: {cnt:>4d} ({pct:5.1f}%)  avg PnL: {avg:+.2f}%")
    print()
    print("  Trades by coin (top 15):")
    sorted_coins = sorted(report["coin_counts"].items(), key=lambda x: x[1], reverse=True)
    for pair, cnt in sorted_coins[:15]:
        print(f"    {pair:12s}: {cnt:>4d}")
    print()
    print("  Risk-Adjusted Metrics:")
    print(f"    Sharpe:     {report['sharpe']:.3f}")
    print(f"    Sortino:    {report['sortino']:.3f}")
    print(f"    Calmar:     {report['calmar']:.3f}")
    print(f"    Composite:  {report['composite']:.3f} (0.4*Sortino + 0.3*Sharpe + 0.3*Calmar)")
    print()
    print(f"  Total return:    {report['total_return_pct']:+.2f}%")
    print(f"  Final portfolio: ${report['final_portfolio']:,.2f}")
    print(f"  Max drawdown:    {report['max_drawdown_pct']:.2f}%")
    print(f"  Win rate:        {report['win_rate_pct']:.1f}%")
    print(f"  Avg trade PnL:   {report['avg_pnl_pct']:+.2f}%")
    print(f"  Stop exits:      {report['stop_exits']}  Signal exits: {report['signal_exits']}")
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
    print("  MULTI-COIN FREQUENCY BACKTEST v2")
    print("  (relaxed MR/Momentum + SOL threshold 0.55)")
    print("=" * 60)

    print("\nStep 1: Loading 15M data...")
    raw_data = load_all_15m_data()
    if not raw_data:
        print("ERROR: No data found in data/. Run download_all_15m.py first.")
        sys.exit(1)

    btc_df = raw_data.get("BTC/USD")
    eth_df = raw_data.get("ETH/USD")
    sol_df = raw_data.get("SOL/USD")
    if btc_df is None or eth_df is None:
        print("ERROR: BTC and ETH parquets are required.")
        sys.exit(1)

    print(f"\n  Loaded {len(raw_data)} coins")

    print(f"\nStep 2: Computing features for {len(raw_data)} coins...")
    all_features = {}

    print("  BTC features (full pipeline)...")
    all_features["BTC/USD"] = compute_btc_features(btc_df, eth_df, sol_df)
    print(f"    → {len(all_features['BTC/USD']):,} bars")

    if sol_df is not None:
        print("  SOL features (full pipeline)...")
        all_features["SOL/USD"] = compute_sol_features(sol_df, btc_df, eth_df)
        print(f"    → {len(all_features['SOL/USD']):,} bars")

    for pair, coin_df in raw_data.items():
        if pair in ("BTC/USD", "SOL/USD"):
            continue
        print(f"  {pair}...", end="", flush=True)
        try:
            feat = compute_coin_features(coin_df, btc_df, eth_df)
            all_features[pair] = feat
            print(f" {len(feat):,} bars")
        except Exception as e:
            print(f" FAILED: {e}")

    print(f"\n  Features computed for {len(all_features)} coins")

    print("\nStep 3: Loading XGBoost models...")
    btc_model_path = "models/xgb_btc_15m_iter5.pkl"
    sol_model_path = "models/xgb_sol_15m.pkl"

    btc_probas = sol_probas = None
    btc_feat_index = sol_feat_index = None
    oos_ts = pd.Timestamp(OOS_START, tz="UTC")

    if Path(btc_model_path).exists():
        btc_model = load_model(btc_model_path)
        btc_oos = all_features["BTC/USD"][all_features["BTC/USD"].index >= oos_ts]
        btc_probas = batch_predict(btc_model, btc_oos)
        btc_feat_index = btc_oos.index
        buys_above = (btc_probas >= BTC_THRESHOLD).sum()
        print(f"  BTC: {len(btc_probas):,} bars, mean P(BUY)={btc_probas.mean():.3f}, "
              f"≥{BTC_THRESHOLD}: {buys_above:,}")
    else:
        print(f"  WARNING: {btc_model_path} not found")

    if Path(sol_model_path).exists() and "SOL/USD" in all_features:
        sol_model = load_model(sol_model_path)
        sol_oos = all_features["SOL/USD"][all_features["SOL/USD"].index >= oos_ts]
        sol_probas = batch_predict(sol_model, sol_oos)
        sol_feat_index = sol_oos.index
        buys_above = (sol_probas >= SOL_THRESHOLD).sum()
        print(f"  SOL: {len(sol_probas):,} bars, mean P(BUY)={sol_probas.mean():.3f}, "
              f"≥{SOL_THRESHOLD}: {buys_above:,}")
    else:
        print(f"  WARNING: {sol_model_path} not found")

    print("\nStep 4: Running backtest...")
    returns, portfolio, closed_trades, gate_stats = run_backtest(
        all_features, btc_probas, sol_probas,
        btc_feat_index, sol_feat_index,
        use_relaxed=True,
    )

    if returns.empty:
        print("ERROR: No returns.")
        sys.exit(1)

    print("\nStep 5: Report...")
    report = compute_report(returns, portfolio, closed_trades, gate_stats)
    print_report(report)

    output_path = Path("research_results/multicoin_frequency_backtest_v2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_report = {**report}
    json_report["config"] = CONFIG
    json_report["sol_threshold"] = SOL_THRESHOLD
    json_report["btc_threshold"] = BTC_THRESHOLD
    json_report["relaxed_mr"] = {"rsi_buy": RELAXED_MR_RSI_BUY, "bb_buy": RELAXED_MR_BB_POS_BUY}
    json_report["relaxed_mom"] = {"rsi_buy": RELAXED_MOM_RSI_BUY, "rsi_sell": RELAXED_MOM_RSI_SELL}
    json_report["n_coins"] = len(all_features)
    json_report["coins"] = list(all_features.keys())

    for k, v in json_report.items():
        if isinstance(v, (np.integer, np.int64)):
            json_report[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            json_report[k] = float(v)

    with open(output_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
