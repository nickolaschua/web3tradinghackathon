import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeState(Enum):
    """Market regime classification with size multipliers."""

    BULL_TREND = "bull"
    SIDEWAYS = "sideways"
    BEAR_TREND = "bear"

    @property
    def size_multiplier(self) -> float:
        """Return the position size multiplier for this regime."""
        return REGIME_MULTIPLIERS.get(self, 0.50)


REGIME_MULTIPLIERS = {
    RegimeState.BULL_TREND: 1.00,
    RegimeState.SIDEWAYS:   1.00,
    RegimeState.BEAR_TREND: 1.00,
}


class RegimeDetector:
    """
    Detects market regime (bull/sideways/bear) using EMA crossover on daily-resampled 15M data.
    Uses hysteresis to prevent thrashing at crossover boundaries.
    """

    def __init__(self, config: dict):
        """
        Initialize RegimeDetector.

        Args:
            config: Configuration dictionary (may contain regime-specific settings)
        """
        self.config = config
        self.MIN_BARS = 4800  # 50 days * 96 bars/day at 15M resolution
        self.CONFIRMATION_BARS = 2

        self._current_regime = RegimeState.SIDEWAYS
        self._pending_regime: Optional[RegimeState] = None
        self._pending_count: int = 0

    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 15M OHLCV data to daily.

        Args:
            df: DataFrame with 'timestamp' (or as index), 'open', 'high', 'low', 'close', 'volume'

        Returns:
            Daily OHLCV DataFrame with 'timestamp' column
        """
        # Set timestamp as index if it's a column
        if "timestamp" in df.columns:
            df_work = df.set_index("timestamp").copy()
        else:
            df_work = df.copy()

        # Resample to daily
        daily = df_work.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )

        # Drop rows with NaN close
        daily = daily.dropna(subset=["close"])

        # Reset index and ensure timestamp is a column
        daily = daily.reset_index()
        daily = daily.rename(columns={"timestamp": "timestamp"}) if "timestamp" in daily.columns else daily

        return daily

    def is_warmed_up(self, df: pd.DataFrame) -> bool:
        """
        Check if we have enough 15M bars for regime detection.

        Args:
            df: DataFrame with 15M candles

        Returns:
            True if we have at least MIN_BARS bars
        """
        return len(df) >= self.MIN_BARS

    def get_regime(self) -> RegimeState:
        """Get the current regime state."""
        return self._current_regime

    def get_size_multiplier(self) -> float:
        """Get the position size multiplier for the current regime."""
        return self._current_regime.size_multiplier

    def _compute_ema_crossover(self, daily_df: pd.DataFrame) -> Optional[RegimeState]:
        """
        Compute EMA(20)/EMA(50) crossover signal on daily data.

        Args:
            daily_df: Daily OHLCV DataFrame

        Returns:
            RegimeState if enough data, None otherwise
        """
        if len(daily_df) < 50:
            return None

        close = daily_df["close"]
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        e20 = ema20.iloc[-1]
        e50 = ema50.iloc[-1]

        # Dead zone: if spread < 0.1%, consider sideways
        if abs(e20 - e50) / e50 < 0.001:
            return RegimeState.SIDEWAYS

        # Crossover signal
        if e20 > e50:
            return RegimeState.BULL_TREND
        else:
            return RegimeState.BEAR_TREND

    def update(self, df: pd.DataFrame) -> RegimeState:
        """
        Update regime based on 15M data, applying hysteresis.

        Args:
            df: DataFrame with 15M OHLCV data

        Returns:
            Current regime state after update
        """
        # Check warmup
        if not self.is_warmed_up(df):
            logger.debug(
                f"RegimeDetector: not warmed up ({len(df)}/{self.MIN_BARS} bars)"
            )
            return self._current_regime

        # Resample to daily and compute signal
        daily_df = self._resample_to_daily(df)
        raw_signal = self._compute_ema_crossover(daily_df)

        if raw_signal is None:
            return self._current_regime

        # Hysteresis logic
        if raw_signal != self._current_regime:
            if raw_signal == self._pending_regime:
                self._pending_count += 1
            else:
                self._pending_regime = raw_signal
                self._pending_count = 1

            if self._pending_count >= self.CONFIRMATION_BARS:
                old = self._current_regime
                self._current_regime = raw_signal
                self._pending_regime = None
                self._pending_count = 0
                logger.info(f"Regime: {old.name} -> {self._current_regime.name}")
        else:
            self._pending_regime = None
            self._pending_count = 0

        return self._current_regime

    def dump_state(self) -> dict:
        """
        Serialize regime state for crash recovery.

        Returns:
            Dictionary with current_regime, pending_regime, pending_count
        """
        return {
            "current_regime": self._current_regime.name,
            "pending_regime": self._pending_regime.name if self._pending_regime else None,
            "pending_count": self._pending_count,
        }

    def load_state(self, state: dict) -> None:
        """
        Restore regime state from serialized dict.

        Args:
            state: Dictionary from dump_state()
        """
        self._current_regime = RegimeState[state["current_regime"]]
        self._pending_regime = (
            RegimeState[state["pending_regime"]] if state.get("pending_regime") else None
        )
        self._pending_count = state.get("pending_count", 0)
