"""
Long-only pairs trading strategy using Engle-Granger cointegration.

Because Roostoo is spot-only (no shorting), this strategy takes only the LONG leg:
- Buy the laggard when spread widens beyond +1.5 standard deviations
- Exit when spread reverts to within 0.5 standard deviations of mean

The "spread" is the residual of the OLS regression:
    log(price_A) = alpha + beta * log(price_B) + epsilon
When epsilon (the residual) is large and positive (A is expensive relative to B), buy B.
When epsilon is large and negative (A is cheap relative to B), buy A.

Reference: research/strategies/pairs_trading.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Engle-Granger test p-value threshold for cointegration
COINT_PVALUE_MAX = 0.10   # accept up to 10% — looser than the usual 5% to get more pairs

# Spread entry and exit thresholds (in standard deviations)
ENTRY_ZSCORE = 1.5    # enter long laggard when spread is >1.5 std from mean
EXIT_ZSCORE  = 0.5    # exit when spread reverts to within 0.5 std of mean
STOP_ZSCORE  = 3.0    # emergency exit if spread widens further to 3.0 std (spread breaking down)

# Minimum bars for cointegration test
MIN_BARS = 100         # ~16 days of 4H data

# How often to re-run the cointegration test (in bars)
RETEST_INTERVAL = 42   # re-test every 7 days


@dataclass
class PairState:
    """Tracks the live state of a single cointegrated pair."""
    pair_a: str          # e.g. "ETH/USD"
    pair_b: str          # e.g. "BNB/USD"
    beta: float = 1.0    # OLS coefficient: log(A) = alpha + beta * log(B) + eps
    alpha: float = 0.0   # OLS intercept
    spread_mean: float = 0.0
    spread_std: float = 1.0
    last_bar_tested: int = 0   # bar index of last cointegration test
    is_cointegrated: bool = False
    long_pair: Optional[str] = None    # which pair we're currently long (or None)


class PairsTradingStrategy:
    """
    Long-only cointegration-based pairs trading strategy.

    Usage:
        strategy = PairsTradingStrategy(config={})
        strategy.add_candidate_pair("ETH/USD", "BNB/USD")

        # In the main loop, once per 4H bar:
        for pair_state in strategy.pair_states:
            signals = strategy.update(pair_state, coin_dfs, bar_index)
            # signals is a list of TradingSignal objects (0, 1, or 2 items)
    """

    def __init__(self, config: dict):
        self.config = config
        self.candidate_pairs: list[tuple[str, str]] = []
        self.pair_states: list[PairState] = []

    def add_candidate_pair(self, pair_a: str, pair_b: str) -> None:
        """Register a candidate pair for cointegration testing."""
        self.candidate_pairs.append((pair_a, pair_b))
        self.pair_states.append(PairState(pair_a=pair_a, pair_b=pair_b))

    def _test_cointegration(
        self, log_price_a: pd.Series, log_price_b: pd.Series
    ) -> tuple[bool, float, float]:
        """
        Run Engle-Granger cointegration test and compute OLS hedge ratio.

        Returns:
            (is_cointegrated, beta, alpha)
        """
        from statsmodels.tsa.stattools import coint
        from statsmodels.api import OLS, add_constant

        if len(log_price_a) < MIN_BARS or len(log_price_b) < MIN_BARS:
            return False, 1.0, 0.0

        # Align on common index
        aligned = pd.concat([log_price_a, log_price_b], axis=1).dropna()
        if len(aligned) < MIN_BARS:
            return False, 1.0, 0.0

        la = aligned.iloc[:, 0]
        lb = aligned.iloc[:, 1]

        # Engle-Granger test: is la - beta*lb stationary?
        try:
            _, pvalue, _ = coint(la, lb)
        except Exception as e:
            logger.warning("Cointegration test failed: %s", e)
            return False, 1.0, 0.0

        if pvalue > COINT_PVALUE_MAX:
            return False, 1.0, 0.0

        # OLS: la = alpha + beta * lb + residuals
        X = add_constant(lb)
        try:
            ols = OLS(la, X).fit()
            alpha, beta = ols.params
        except Exception as e:
            logger.warning("OLS regression failed: %s", e)
            return False, 1.0, 0.0

        return True, float(beta), float(alpha)

    def _compute_spread(
        self, log_price_a: pd.Series, log_price_b: pd.Series, beta: float, alpha: float
    ) -> pd.Series:
        """
        Compute the spread (OLS residual): log(A) - alpha - beta * log(B).

        A positive spread means A is overpriced relative to B → laggard is B (buy B).
        A negative spread means A is underpriced relative to B → laggard is A (buy A).
        """
        aligned = pd.concat([log_price_a, log_price_b], axis=1).dropna()
        la = aligned.iloc[:, 0]
        lb = aligned.iloc[:, 1]
        return la - alpha - beta * lb

    def update(
        self,
        state: PairState,
        coin_dfs: dict[str, pd.DataFrame],
        bar_index: int,
    ) -> list:
        """
        Update a pair's cointegration state and generate trading signals.

        Args:
            state:     PairState for this pair.
            coin_dfs:  Dict mapping pair symbol → OHLCV DataFrame (with DatetimeIndex).
            bar_index: Current bar counter (used to throttle cointegration re-testing).

        Returns:
            List of TradingSignal objects. Empty list = no action.
        """
        from bot.strategy.base import SignalDirection, TradingSignal

        df_a = coin_dfs.get(state.pair_a)
        df_b = coin_dfs.get(state.pair_b)
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            return []

        log_a = np.log(df_a["close"])
        log_b = np.log(df_b["close"])

        signals = []

        # Re-test cointegration periodically.
        # Pitfall 4 fix: close any open position before refitting — new beta changes
        # the spread definition and would instantly invalidate the z-score.
        bars_since_test = bar_index - state.last_bar_tested
        if bars_since_test >= RETEST_INTERVAL or not state.is_cointegrated:
            if bars_since_test >= RETEST_INTERVAL and state.long_pair is not None:
                signals.append(TradingSignal(
                    pair=state.long_pair,
                    direction=SignalDirection.SELL,
                    size=1.0,
                    confidence=0.50,
                ))
                logger.info(
                    "Pairs pre-refit exit: %s (bar %d, refit due)",
                    state.long_pair, bar_index,
                )
                state.long_pair = None

            is_coint, beta, alpha = self._test_cointegration(log_a, log_b)
            state.is_cointegrated = is_coint
            state.beta = beta
            state.alpha = alpha
            state.last_bar_tested = bar_index

            if is_coint:
                # Recompute spread statistics on the full history
                spread = self._compute_spread(log_a, log_b, beta, alpha)
                state.spread_mean = float(spread.mean())
                state.spread_std  = float(spread.std() + 1e-10)
                logger.info(
                    "Pair %s/%s: cointegrated (p<%.2f), beta=%.3f, "
                    "spread_mean=%.4f, spread_std=%.4f",
                    state.pair_a, state.pair_b, COINT_PVALUE_MAX,
                    beta, state.spread_mean, state.spread_std,
                )
            else:
                logger.debug(
                    "Pair %s/%s: NOT cointegrated at bar %d",
                    state.pair_a, state.pair_b, bar_index,
                )
                return signals  # return any pre-refit exit signal

        if not state.is_cointegrated:
            return signals

        # Current spread z-score (use only the last bar — no look-ahead)
        current_spread = float(log_a.iloc[-2] - state.alpha - state.beta * log_b.iloc[-2])
        zscore = (current_spread - state.spread_mean) / state.spread_std

        # --- Exit conditions ---
        if state.long_pair is not None:
            should_exit = (
                abs(zscore) < EXIT_ZSCORE or    # spread has reverted
                abs(zscore) > STOP_ZSCORE        # spread breaking down (stop loss)
            )
            if should_exit:
                signals.append(TradingSignal(
                    pair=state.long_pair,
                    direction=SignalDirection.SELL,
                    size=1.0,
                    confidence=0.65,
                ))
                logger.info(
                    "Pairs exit: %s (zscore=%.2f)", state.long_pair, zscore
                )
                state.long_pair = None
            return signals  # don't generate new entries while managing an exit

        # --- Entry conditions ---
        if abs(zscore) > ENTRY_ZSCORE:
            # Positive spread: A is overpriced vs B → laggard is B
            # Negative spread: A is underpriced vs B → laggard is A
            laggard = state.pair_b if zscore > 0 else state.pair_a

            signals.append(TradingSignal(
                pair=laggard,
                direction=SignalDirection.BUY,
                size=0.25,    # modest size — pairs trade is a secondary strategy
                confidence=0.60,
            ))
            state.long_pair = laggard
            logger.info(
                "Pairs entry: long %s (zscore=%.2f)", laggard, zscore
            )

        return signals
