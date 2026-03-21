"""
RSI Divergence Strategy

Detects bullish divergences where price makes lower lows but RSI makes higher lows,
signaling weakening downward momentum and potential reversal.

Entry Logic:
- Price makes new low over lookback window
- RSI at that low is HIGHER than RSI at previous low (divergence)
- RSI is oversold (< 40)
- MACD histogram turning positive (momentum shift confirmation)

Exit Logic:
- RSI crosses above 60 (overbought, take profit)
- MACD histogram turns negative (momentum reversal)

No hardcoded stop-loss or sizing - uses RiskManager + PortfolioAllocator framework.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI divergence strategy.

    Captures reversals when price and RSI momentum diverge, indicating
    exhaustion of the prevailing trend.
    """

    def __init__(
        self,
        lookback_window: int = 20,
        rsi_oversold: float = 40.0,
        rsi_overbought: float = 60.0,
    ):
        """
        Initialize strategy with tunable parameters.

        Args:
            lookback_window: Period to look for divergences (default 20)
            rsi_oversold: RSI threshold for entry consideration (default 40)
            rsi_overbought: RSI threshold for exit (default 60)
        """
        self.lookback_window = lookback_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate RSI divergence signal.

        Required features:
        - close: Current close price
        - rsi: RSI indicator (14-period)
        - macd_hist: MACD histogram

        Args:
            pair: Trading pair (e.g., "BTC/USD")
            features: DataFrame with required RSI and price features

        Returns:
            TradingSignal with BUY/SELL/HOLD direction
        """
        if len(features) < self.lookback_window + 5:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]
        rsi = latest.get("rsi", 50.0)
        macd_hist = latest.get("macd_hist", 0.0)
        close = latest.get("close", 0.0)

        # Handle missing data
        if pd.isna(rsi) or pd.isna(macd_hist) or pd.isna(close):
            return TradingSignal(pair=pair)

        # Get recent window
        recent = features.iloc[-self.lookback_window:]
        recent_prices = recent["close"].values
        recent_rsi = recent["rsi"].values

        # Find local price lows (simple: lowest point in window)
        price_low_idx = recent_prices.argmin()
        price_low = recent_prices[price_low_idx]
        rsi_at_price_low = recent_rsi[price_low_idx]

        # Find previous low before the current low
        if price_low_idx > 5:  # Need some history
            prev_window_prices = recent_prices[:price_low_idx - 2]
            prev_window_rsi = recent_rsi[:price_low_idx - 2]

            if len(prev_window_prices) > 0:
                prev_low_idx = prev_window_prices.argmin()
                prev_price_low = prev_window_prices[prev_low_idx]
                rsi_at_prev_low = prev_window_rsi[prev_low_idx]

                # BULLISH DIVERGENCE: price lower, RSI higher
                price_making_lower_low = close <= prev_price_low
                rsi_making_higher_low = rsi_at_price_low > rsi_at_prev_low

                # Additional confirmation
                rsi_oversold = rsi < self.rsi_oversold
                macd_turning_positive = macd_hist > 0

                # --- ENTRY CONDITIONS ---
                if price_making_lower_low and rsi_making_higher_low and rsi_oversold and macd_turning_positive:
                    # Confidence based on divergence strength
                    divergence_strength = (rsi_at_price_low - rsi_at_prev_low) / 10.0
                    confidence = min(0.9, 0.6 + divergence_strength * 0.1)

                    return TradingSignal(
                        pair=pair,
                        direction=SignalDirection.BUY,
                        size=0.35,
                        confidence=confidence,
                    )

        # --- EXIT CONDITIONS ---
        # Take profit when RSI reaches overbought
        if rsi > self.rsi_overbought:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.7,
            )

        # Exit when MACD momentum reverses
        if macd_hist < 0 and rsi > 50:  # Only exit if we had some gains
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.6,
            )

        # Default: HOLD
        return TradingSignal(pair=pair)
