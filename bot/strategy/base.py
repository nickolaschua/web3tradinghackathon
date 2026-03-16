"""
Strategy base types for the Roostoo trading bot.

Defines the contract that all strategy implementations must follow:
- SignalDirection enum: BUY / SELL / HOLD
- TradingSignal dataclass: pair is a required positional field with no default
- BaseStrategy ABC: generate_signal(pair, features) -> TradingSignal
"""
from __future__ import annotations

import enum
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass


class SignalDirection(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """
    A trading signal produced by a strategy.

    Attributes
    ----------
    pair : str
        The tradeable pair, e.g. "BTC/USD".  REQUIRED — no default.
        An empty or missing pair causes all order submissions to silently fail;
        this field being required prevents that bug at construction time.
    direction : SignalDirection
        BUY, SELL, or HOLD.  Default: HOLD (safe no-op).
    size : float
        Position size as a fraction of the portfolio, 0.0–1.0.  Default: 0.0.
    confidence : float
        Signal confidence 0.0–1.0, reserved for future sizing/filtering logic.
        Default: 0.0.
    """

    pair: str  # required — NO default
    direction: SignalDirection = SignalDirection.HOLD
    size: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.pair, str) or not self.pair:
            raise ValueError(
                f"TradingSignal.pair must be a non-empty string, got {self.pair!r}"
            )
        if not isinstance(self.direction, SignalDirection):
            raise ValueError(
                f"TradingSignal.direction must be a SignalDirection, got {self.direction!r}"
            )
        if not (0.0 <= self.size <= 1.0):
            raise ValueError(
                f"TradingSignal.size must be between 0.0 and 1.0, got {self.size!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"TradingSignal.confidence must be between 0.0 and 1.0, got {self.confidence!r}"
            )


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate a trading signal from the latest features.

        Parameters
        ----------
        pair : str
            The tradeable pair symbol, e.g. "BTC/USD".  Must match the pair
            passed to the strategy at runtime.
        features : pd.DataFrame
            DataFrame of computed indicators.  All indicator columns are
            already shifted by 1 bar (shift(1)) to prevent look-ahead bias.
            Columns include: close, atr_proxy, rsi, macd, macd_signal,
            macd_hist, ema_slope, eth_btc_corr, sol_btc_corr, eth_return,
            sol_return.

        Returns
        -------
        TradingSignal
            Signal with pair matching the input ``pair`` argument.
            ``direction`` defaults to HOLD (safe no-op if no conditions met).
        """
