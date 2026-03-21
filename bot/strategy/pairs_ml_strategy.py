"""
ML-based pairs trading strategy using XGBoost spread-reversion classifier.

Replaces rule-based Engle-Granger cointegration approach with a trained model
that predicts P(spread reverts within 32 bars) given current spread features.

Entry: abs(zscore) > ENTRY_ZSCORE AND P(reversion) >= threshold
Exit:  abs(zscore) < EXIT_ZSCORE (reverted) OR abs(zscore) > STOP_ZSCORE (stopped out)
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bot.strategy.base import SignalDirection, TradingSignal
from bot.data.pairs_features import compute_pairs_features

logger = logging.getLogger(__name__)

# Entry and exit thresholds (in standard deviations) — timeframe-agnostic
ENTRY_ZSCORE = 1.5  # minimum abs(zscore) to consider entry
EXIT_ZSCORE = 0.5   # exit when spread reverts this close to mean
STOP_ZSCORE = 3.0   # emergency exit if spread blows out this far

# Minimum bars before the model can fire.
# ols_window(2880) + zscore_window(672) = 3552 bars required for the first valid zscore.
MIN_BARS = 3552

# Default P(reversion) threshold for BUY signal
DEFAULT_THRESHOLD = 0.60


@dataclass
class PairState:
    """Tracks the live state of a single ML-based pairs strategy position."""
    pair_a: str          # e.g. "BTC/USD"
    pair_b: str          # e.g. "ETH/USD"
    long_pair: Optional[str] = None   # which pair we're currently long (or None)
    last_zscore: float = 0.0          # zscore at entry (for exit logic and direction tracking)


class PairsMLStrategy:
    """
    Long-only ML-based pairs trading strategy.

    Uses an XGBoost classifier to predict P(spread reverts within 32 bars)
    and enters only when the reversion probability exceeds the configured threshold.

    Usage:
        strategy = PairsMLStrategy(config={}, model_path="models/pairs_btc_eth_15m.pkl")
        strategy.add_candidate_pair("BTC/USD", "ETH/USD")

        # In the main loop, once per 15M bar:
        for pair_state in strategy.pair_states:
            signals = strategy.update(pair_state, coin_dfs, bar_index)
            # signals is a list of TradingSignal objects (0, 1, or 2 items)
    """

    def __init__(
        self,
        config: dict,
        model_path: str = "models/pairs_btc_eth_15m.pkl",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Initialize the ML pairs strategy.

        Args:
            config: Configuration dict (currently unused but required for interface compatibility).
            model_path: Path to the XGBoost model pickle file.
            threshold: P(reversion) threshold for entry (0.0-1.0).
        """
        self.config = config
        self._threshold = threshold
        self._model = None
        self._feature_cols: list[str] = []

        # Try to load the model
        model_file = Path(model_path)
        if model_file.exists():
            try:
                with open(model_file, "rb") as f:
                    self._model = pickle.load(f)
                if not hasattr(self._model, "predict_proba"):
                    raise ValueError("Model missing predict_proba() — wrong object type?")
                if not hasattr(self._model, "feature_names_in_"):
                    raise ValueError("Model missing feature_names_in_ — retrain with XGBoost >= 1.6")
                self._feature_cols = list(self._model.feature_names_in_)
                logger.info(
                    "PairsMLStrategy: loaded model from %s (threshold=%.2f, %d features)",
                    model_path, threshold, len(self._feature_cols),
                )
            except Exception as e:
                logger.warning(
                    "PairsMLStrategy: failed to load model from %s: %s (strategy will be no-op)",
                    model_path, e,
                )
                self._model = None
                self._feature_cols = []
        else:
            logger.warning(
                "PairsMLStrategy: model file not found at %s (strategy will be no-op)",
                model_path,
            )

        self.candidate_pairs: list[tuple[str, str]] = []
        self.pair_states: list[PairState] = []

    def add_candidate_pair(self, pair_a: str, pair_b: str) -> None:
        """
        Register a candidate pair for ML-based pairs trading.

        Args:
            pair_a: First asset (e.g. "BTC/USD").
            pair_b: Second asset (e.g. "ETH/USD").
        """
        self.candidate_pairs.append((pair_a, pair_b))
        self.pair_states.append(PairState(pair_a=pair_a, pair_b=pair_b))

    def update(
        self,
        state: PairState,
        coin_dfs: dict[str, pd.DataFrame],
        bar_index: int,
    ) -> list:
        """
        Update a pair's ML state and generate trading signals.

        Entry logic:
          - abs(zscore) > ENTRY_ZSCORE (spread is sufficiently wide)
          - Model predicts P(reversion) >= threshold

        Exit logic:
          - abs(zscore) < EXIT_ZSCORE (spread has reverted to mean)
          - abs(zscore) > STOP_ZSCORE (spread has blown out — stop loss)
          - zscore has flipped sign (different regime)

        Args:
            state:     PairState for this pair.
            coin_dfs:  Dict mapping pair symbol → OHLCV DataFrame (with DatetimeIndex).
            bar_index: Current bar counter (currently unused but maintained for interface).

        Returns:
            List of TradingSignal objects. Empty list = no action.
        """
        try:
            # No-op if model not loaded
            if self._model is None:
                return []

            # Fetch price data
            df_a = coin_dfs.get(state.pair_a)
            df_b = coin_dfs.get(state.pair_b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                return []

            # Check minimum bars
            if len(df_a) < MIN_BARS or len(df_b) < MIN_BARS:
                return []

            # Compute pairs features
            feat_df = compute_pairs_features(df_a["close"], df_b["close"])
            if feat_df.empty:
                return []

            # Get the last row (current bar)
            last_row = feat_df.iloc[-1]
            current_zscore = float(last_row["zscore"])

            # Handle NaN zscore (insufficient data for rolling windows)
            if np.isnan(current_zscore):
                return []

            signals = []

            # --- Exit conditions (if we're currently in a position) ---
            if state.long_pair is not None:
                should_exit = False
                exit_reason = ""

                # Exit 1: spread has reverted to mean
                if abs(current_zscore) < EXIT_ZSCORE:
                    should_exit = True
                    exit_reason = f"reversion (zscore={current_zscore:.2f})"

                # Exit 2: spread has blown out (stop loss)
                elif abs(current_zscore) > STOP_ZSCORE:
                    should_exit = True
                    exit_reason = f"stop loss (zscore={current_zscore:.2f})"

                # Exit 3: zscore has flipped sign (different regime)
                elif np.sign(current_zscore) != np.sign(state.last_zscore) and state.last_zscore != 0.0:
                    should_exit = True
                    exit_reason = f"sign flip (entry={state.last_zscore:.2f}, now={current_zscore:.2f})"

                if should_exit:
                    signals.append(
                        TradingSignal(
                            pair=state.long_pair,
                            direction=SignalDirection.SELL,
                            size=1.0,
                            confidence=0.65,
                        )
                    )
                    logger.info(
                        "Pairs ML exit: %s (%s)",
                        state.long_pair, exit_reason,
                    )
                    state.long_pair = None
                    state.last_zscore = 0.0

                return signals  # don't generate new entries while managing an exit

            # --- Entry conditions (if we're not in a position) ---
            if abs(current_zscore) > ENTRY_ZSCORE:
                # Build feature row for the model
                feature_row = pd.DataFrame([last_row[self._feature_cols]])

                # Handle missing columns (fill with NaN, model should handle it)
                for col in self._feature_cols:
                    if col not in feature_row.columns:
                        feature_row[col] = np.nan

                # Reorder columns to match model's expected order
                feature_row = feature_row[self._feature_cols]

                # Get probability of reversion (class 1)
                try:
                    proba = self._model.predict_proba(feature_row)[0, 1]
                except Exception as e:
                    logger.warning(
                        "Pairs ML: predict_proba failed for %s/%s: %s",
                        state.pair_a, state.pair_b, e,
                    )
                    return []

                # Check threshold
                if proba >= self._threshold:
                    # Determine laggard: positive zscore means pair_a is expensive, so buy pair_b
                    laggard = state.pair_b if current_zscore > 0 else state.pair_a

                    signals.append(
                        TradingSignal(
                            pair=laggard,
                            direction=SignalDirection.BUY,
                            size=0.25,  # modest size — pairs trade is a secondary strategy
                            confidence=proba,
                        )
                    )
                    state.long_pair = laggard
                    state.last_zscore = current_zscore

                    logger.info(
                        "Pairs ML entry: long %s (zscore=%.2f, P(revert)=%.3f)",
                        laggard, current_zscore, proba,
                    )

            return signals

        except Exception as e:
            logger.error(
                "Pairs ML strategy error for %s/%s: %s",
                state.pair_a, state.pair_b, e, exc_info=True,
            )
            return []
