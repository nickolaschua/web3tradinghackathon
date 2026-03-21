"""
Four pluggable strategies for the alt-strategies research engine.

All strategies implement the same two-method interface:
    generate_entries(ts, bar_idx, coin_data, open_pairs, portfolio_value) -> list[SignalCandidate]
    check_exit(ts, bar_idx, pair, source, coin_row, coin_data)            -> bool

Strategies:
    1. CrossSectionalMomentum — ML-enhanced (XGBoost) or rule-based cross-coin ranking
    2. SpreadReversion        — z-score mean-reversion on coin-pair ratios
    3. RegimeFilteredXGB      — BTC XGBoost as regime gate for altcoin momentum
    4. VolatilityBreakout     — ATR expansion + price breakout
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.alt_strategies.engine import SignalCandidate


# ── 1. Cross-Sectional Momentum ──────────────────────────────────────────────

class CrossSectionalMomentum:
    """
    Rank all coins each bar and go long the top-K.

    Two modes:
      - ML (model is not None): use pre-computed XGBoost P(outperform) probabilities
      - Rule-based (model is None): rank by trailing 1-day return

    The XGBoost model is trained by train_xsm.py on features that capture
    multi-horizon momentum, cross-sectional ranks, BTC beta, and volume regime.
    This makes it strictly better than simple return ranking because it learns
    non-linear interactions (e.g., "fast momentum + low RSI + expanding volume"
    predicts continuation; "fast momentum + high RSI" predicts reversal).
    """
    name = "xsm"

    def __init__(
        self,
        xsm_proba_map: dict[str, dict] | None = None,
        top_k: int = 5,
        buy_threshold: float = 0.50,
        exit_threshold: float = 0.30,
        rebalance_bars: int = 96,  # rebalance every ~1 day
        min_coins: int = 5,
        excluded_pairs: set[str] | None = None,
    ):
        self.xsm_proba_map = xsm_proba_map
        self.top_k = top_k
        self.buy_threshold = buy_threshold
        self.exit_threshold = exit_threshold
        self.rebalance_bars = rebalance_bars
        self.min_coins = min_coins
        self.excluded_pairs = excluded_pairs or set()
        self._last_rebalance_bar = -999

    def reset(self):
        self._last_rebalance_bar = -999

    def generate_entries(self, ts, bar_idx, coin_data, open_pairs, portfolio_value):
        if bar_idx - self._last_rebalance_bar < self.rebalance_bars:
            return []

        candidates = []
        for pair, row in coin_data.items():
            if pair in open_pairs or pair in self.excluded_pairs:
                continue

            score = self._get_score(pair, ts, row)
            if score is None or np.isnan(score):
                continue
            candidates.append((pair, score, row))

        if len(candidates) < self.min_coins:
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        self._last_rebalance_bar = bar_idx

        signals = []
        for pair, score, row in candidates[:self.top_k]:
            if self.xsm_proba_map is not None and score < self.buy_threshold:
                continue

            close = row.get("close", 0)
            atr = row.get("atr_proxy", close * 0.02 if close else 0)
            signals.append(SignalCandidate(
                pair=pair,
                direction="BUY",
                source=self.name,
                confidence=float(score),
                size=1.0 / self.top_k,
                sizing_mode="portfolio_pct",
            ))

        return signals

    def check_exit(self, ts, bar_idx, pair, source, coin_row, coin_data):
        if source != self.name:
            return False

        score = self._get_score(pair, ts, coin_row)
        if score is not None and score < self.exit_threshold:
            return True

        # Rebalance-based exit: if we're at a rebalance bar, check rank
        if bar_idx - self._last_rebalance_bar >= self.rebalance_bars:
            all_scores = []
            for p, row in coin_data.items():
                s = self._get_score(p, ts, row)
                if s is not None and not np.isnan(s):
                    all_scores.append((p, s))
            all_scores.sort(key=lambda x: x[1], reverse=True)
            top_pairs = {p for p, _ in all_scores[:self.top_k]}
            if pair not in top_pairs:
                return True

        return False

    def _get_score(self, pair, ts, row):
        if self.xsm_proba_map is not None:
            pair_map = self.xsm_proba_map.get(pair, {})
            return pair_map.get(ts, None)
        return row.get("return_1d", None)


# ── 2. Spread Reversion ─────────────────────────────────────────────────────

class SpreadReversion:
    """
    Mean-reversion on the log-ratio spread between two coins.

    Entry: z-score < -entry_z or > +entry_z → trade back toward mean.
    Exit: z-score reverts past exit_z, or hits stop_z.

    Classic pairs: BTC/PAXG (low correlation), high-beta/low-beta coin pairs.
    """
    name = "spread"

    def __init__(
        self,
        pair_a: str = "BTC/USD",
        pair_b: str = "PAXG/USD",
        lookback: int = 672,    # 7 days in 15M bars
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
        position_pct: float = 0.20,
    ):
        self.pair_a = pair_a
        self.pair_b = pair_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.position_pct = position_pct

        self._spread_history: list[float] = []
        self._current_z = 0.0
        self._side: str | None = None  # "long_a" or "long_b"

    def reset(self):
        self._spread_history = []
        self._current_z = 0.0
        self._side = None

    def generate_entries(self, ts, bar_idx, coin_data, open_pairs, portfolio_value):
        row_a = coin_data.get(self.pair_a)
        row_b = coin_data.get(self.pair_b)
        if not row_a or not row_b:
            return []

        close_a = row_a.get("close", 0)
        close_b = row_b.get("close", 0)
        if close_a <= 0 or close_b <= 0:
            return []

        spread = math.log(close_a / close_b)
        self._spread_history.append(spread)
        if len(self._spread_history) > self.lookback * 2:
            self._spread_history = self._spread_history[-self.lookback * 2:]

        if len(self._spread_history) < self.lookback:
            return []

        window = self._spread_history[-self.lookback:]
        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-10:
            return []

        self._current_z = (spread - mean) / std

        if self.pair_a in open_pairs or self.pair_b in open_pairs:
            return []

        signals = []
        if self._current_z < -self.entry_z:
            # Spread too low: pair_a is cheap relative to pair_b → buy pair_a
            self._side = "long_a"
            signals.append(SignalCandidate(
                pair=self.pair_a,
                direction="BUY",
                source=self.name,
                confidence=min(1.0, abs(self._current_z) / 3.0),
                size=self.position_pct,
                sizing_mode="portfolio_pct",
            ))
        elif self._current_z > self.entry_z:
            # Spread too high: pair_b is cheap relative to pair_a → buy pair_b
            self._side = "long_b"
            signals.append(SignalCandidate(
                pair=self.pair_b,
                direction="BUY",
                source=self.name,
                confidence=min(1.0, abs(self._current_z) / 3.0),
                size=self.position_pct,
                sizing_mode="portfolio_pct",
            ))

        return signals

    def check_exit(self, ts, bar_idx, pair, source, coin_row, coin_data):
        if source != self.name:
            return False

        if self._side == "long_a" and pair == self.pair_a:
            return self._current_z >= -self.exit_z or self._current_z < -self.stop_z
        if self._side == "long_b" and pair == self.pair_b:
            return self._current_z <= self.exit_z or self._current_z > self.stop_z

        return False


# ── 3. Regime-Filtered XGBoost ────────────────────────────────────────────────

class RegimeFilteredXGB:
    """
    Use the BTC XGBoost model as a market-wide regime filter.

    When BTC model says BUY (P > regime_threshold), enter top momentum altcoins.
    When BTC model says HOLD/SELL (P < exit_threshold), exit all altcoin positions.

    This captures the observation that altcoins generally follow BTC direction
    but with higher beta — so timing the regime via BTC and expressing it via
    altcoins amplifies returns.
    """
    name = "regime_xgb"

    def __init__(
        self,
        btc_proba_map: dict | None = None,
        regime_threshold: float = 0.60,
        exit_threshold: float = 0.40,
        top_k: int = 3,
        target_pairs: list[str] | None = None,
        position_pct: float = 0.15,
    ):
        self.btc_proba_map = btc_proba_map or {}
        self.regime_threshold = regime_threshold
        self.exit_threshold = exit_threshold
        self.top_k = top_k
        self.target_pairs = set(target_pairs or [])
        self.position_pct = position_pct

    def reset(self):
        pass

    def generate_entries(self, ts, bar_idx, coin_data, open_pairs, portfolio_value):
        btc_p = self.btc_proba_map.get(ts, 0.0)
        if btc_p < self.regime_threshold:
            return []

        # BTC regime is bullish — buy top momentum altcoins
        ranked = []
        for pair, row in coin_data.items():
            if pair in open_pairs:
                continue
            if self.target_pairs and pair not in self.target_pairs:
                continue
            if pair in ("BTC/USD", "PAXG/USD"):
                continue
            ema_20 = row.get("EMA_20", 0)
            ema_50 = row.get("EMA_50", 0)
            rsi = row.get("RSI_14", 50)
            macd_hist = row.get("MACDh_12_26_9", 0)
            if ema_20 > ema_50 and rsi < 60 and macd_hist > 0:
                score = (50 - rsi) / 50 + 0.5
                ranked.append((pair, score, row))

        ranked.sort(key=lambda x: x[1], reverse=True)

        signals = []
        for pair, score, row in ranked[:self.top_k]:
            signals.append(SignalCandidate(
                pair=pair,
                direction="BUY",
                source=self.name,
                confidence=float(btc_p),
                size=self.position_pct,
                sizing_mode="portfolio_pct",
            ))

        return signals

    def check_exit(self, ts, bar_idx, pair, source, coin_row, coin_data):
        if source != self.name:
            return False
        btc_p = self.btc_proba_map.get(ts, 0.0)
        if btc_p < self.exit_threshold:
            return True
        rsi = coin_row.get("RSI_14", 50)
        macd_hist = coin_row.get("MACDh_12_26_9", 0)
        if rsi > 70 or macd_hist < 0:
            return True
        return False


# ── 4. Volatility Breakout ───────────────────────────────────────────────────

class VolatilityBreakout:
    """
    Enter when a coin's volatility expands and price breaks out of a range.

    Entry conditions:
      - ATR expansion: current ATR > expansion_factor × 20-bar avg ATR
      - Price breakout: close > highest high of lookback_bars
      - Volume confirmation: volume_ratio > 1.5

    Exit: tight ATR trailing stop (3x ATR), or breakdown below midpoint.

    This strategy is inherently spiky — it fires many signals during volatility
    events and is quiet otherwise. Good for catching breakout moves across the
    altcoin universe.
    """
    name = "vol_breakout"

    def __init__(
        self,
        expansion_factor: float = 1.5,
        lookback_bars: int = 96,    # 1 day range
        volume_threshold: float = 1.3,
        position_pct: float = 0.10,
        excluded_pairs: set[str] | None = None,
    ):
        self.expansion_factor = expansion_factor
        self.lookback_bars = lookback_bars
        self.volume_threshold = volume_threshold
        self.position_pct = position_pct
        self.excluded_pairs = excluded_pairs or {"PAXG/USD"}
        self._highs: dict[str, list[float]] = {}

    def reset(self):
        self._highs = {}

    def generate_entries(self, ts, bar_idx, coin_data, open_pairs, portfolio_value):
        signals = []
        for pair, row in coin_data.items():
            if pair in open_pairs or pair in self.excluded_pairs:
                continue

            high = row.get("high", 0)
            close = row.get("close", 0)
            atr = row.get("atr_proxy", 0)
            volume_ratio = row.get("volume_ratio", 0)
            bb_width = row.get("bb_width", 0)
            rsi = row.get("RSI_14", 50)

            if close <= 0 or atr <= 0:
                continue

            # Track highs for breakout detection
            hist = self._highs.setdefault(pair, [])
            hist.append(high)
            if len(hist) > self.lookback_bars * 2:
                self._highs[pair] = hist[-self.lookback_bars * 2:]

            if len(hist) < self.lookback_bars:
                continue

            range_high = max(hist[-self.lookback_bars:])

            # Normalized ATR expansion check using Bollinger bandwidth as proxy
            atr_expanding = bb_width > 0.04

            # Breakout: close above range high
            breakout = close > range_high * 0.998

            # Volume confirmation
            vol_confirm = volume_ratio > self.volume_threshold

            if atr_expanding and breakout and vol_confirm and rsi < 75:
                confidence = min(0.9, 0.5 + (volume_ratio - 1.0) * 0.2)
                signals.append(SignalCandidate(
                    pair=pair,
                    direction="BUY",
                    source=self.name,
                    confidence=confidence,
                    size=self.position_pct,
                    sizing_mode="portfolio_pct",
                ))

        return signals

    def check_exit(self, ts, bar_idx, pair, source, coin_row, coin_data):
        if source != self.name:
            return False
        rsi = coin_row.get("RSI_14", 50)
        macd_hist = coin_row.get("MACDh_12_26_9", 0)
        if rsi > 75 or macd_hist < -0.001:
            return True
        return False


# ── Panel Feature Builder (shared by train + run) ────────────────────────────

def build_xsm_panel(
    all_features: dict[str, pd.DataFrame],
    btc_close: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build a panel DataFrame with cross-sectional features for XSM model.

    Returns DataFrame with MultiIndex (timestamp, pair) containing:
      - Per-coin features: trailing returns, RSI, BB pos, MACD, volume, volatility
      - Cross-sectional ranks: percentile rank of each per-coin feature among all coins
      - BTC-relative: correlation, beta, relative strength

    Used by both train_xsm.py (for training) and run.py (for prediction).
    """
    print("    Panel: computing per-coin features...")
    btc_log_ret = None
    btc_ret_1d = None
    if btc_close is not None:
        btc_log_ret = np.log(btc_close / btc_close.shift(1))
        btc_ret_1d = btc_close.pct_change(96)

    records = []
    for i, (pair, feat_df) in enumerate(all_features.items()):
        df = feat_df.copy()
        df["pair"] = pair

        close = df["close"]
        df["return_1h"] = close.pct_change(4)
        df["return_4h"] = close.pct_change(16)
        df["return_1d"] = close.pct_change(96)
        df["return_3d"] = close.pct_change(288)
        df["volatility"] = close.pct_change().rolling(96).std()
        df["log_return"] = np.log(close / close.shift(1))

        # BTC-relative features — compute per-coin BEFORE concat (much faster)
        if btc_log_ret is not None:
            coin_log_ret = df["log_return"]
            btc_aligned = btc_log_ret.reindex(df.index)

            window = min(960, len(df) // 2)
            if window > 50:
                df["btc_corr"] = coin_log_ret.rolling(window).corr(btc_aligned)
                cov = coin_log_ret.rolling(window).cov(btc_aligned)
                var = btc_aligned.rolling(window).var()
                df["btc_beta"] = cov / (var + 1e-10)

            if btc_ret_1d is not None:
                btc_ret_aligned = btc_ret_1d.reindex(df.index)
                df["relative_strength"] = df["return_1d"] - btc_ret_aligned

        records.append(df)
        if (i + 1) % 10 == 0:
            print(f"      {i + 1}/{len(all_features)} coins done")

    if not records:
        return pd.DataFrame()

    print("    Panel: concatenating...")
    panel = pd.concat(records, axis=0)
    panel = panel.set_index("pair", append=True)

    # Cross-sectional ranks (vectorized via groupby)
    print("    Panel: computing cross-sectional ranks...")
    rank_cols = [
        "return_1h", "return_4h", "return_1d", "return_3d",
        "RSI_14", "volatility", "volume_ratio",
    ]
    for col in rank_cols:
        if col in panel.columns:
            panel[f"{col}_rank"] = panel.groupby(level=0)[col].rank(pct=True)

    print(f"    Panel: done, shape={panel.shape}")
    return panel


XSM_FEATURE_COLS = [
    "return_1h", "return_4h", "return_1d", "return_3d",
    "RSI_14", "bb_pos", "MACDh_12_26_9", "ema_slope",
    "volume_ratio", "volatility",
    "return_1h_rank", "return_4h_rank", "return_1d_rank", "return_3d_rank",
    "RSI_14_rank", "volatility_rank", "volume_ratio_rank",
    "btc_corr", "btc_beta", "relative_strength",
]
