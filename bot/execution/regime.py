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
        if self == RegimeState.BULL_TREND:
            return 1.0
        elif self == RegimeState.SIDEWAYS:
            return 0.5
        elif self == RegimeState.BEAR_TREND:
            return 0.0
        else:
            return 0.5  # Fallback


class RegimeDetector:
    """
    Detects market regime (bull/sideways/bear) using EMA crossover on daily-resampled 4H data.
    Uses hysteresis to prevent thrashing at crossover boundaries.
    """

    def __init__(self, config: dict):
        """
        Initialize RegimeDetector.

        Args:
            config: Configuration dictionary (may contain regime-specific settings)
        """
        self.config = config
        self.MIN_4H_BARS = 300
        self.CONFIRMATION_BARS = 2

        self._current_regime = RegimeState.SIDEWAYS
        self._pending_regime: Optional[RegimeState] = None
        self._pending_count: int = 0

    def _resample_to_daily(self, df_4h: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 4H OHLCV data to daily.

        Args:
            df_4h: DataFrame with 'timestamp' (or as index), 'open', 'high', 'low', 'close', 'volume'

        Returns:
            Daily OHLCV DataFrame with 'timestamp' column
        """
        # Set timestamp as index if it's a column
        if "timestamp" in df_4h.columns:
            df_work = df_4h.set_index("timestamp").copy()
        else:
            df_work = df_4h.copy()

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

    def is_warmed_up(self, df_4h: pd.DataFrame) -> bool:
        """
        Check if we have enough 4H bars for regime detection.

        Args:
            df_4h: DataFrame with 4H candles

        Returns:
            True if we have at least MIN_4H_BARS bars
        """
        return len(df_4h) >= self.MIN_4H_BARS

    def get_regime(self) -> RegimeState:
        """Get the current regime state."""
        return self._current_regime

    def get_size_multiplier(self) -> float:
        """Get the position size multiplier for the current regime."""
        return self._current_regime.size_multiplier
