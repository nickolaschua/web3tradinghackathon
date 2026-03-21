"""
Relative Strength Rotation Strategy

Long-only rotation strategy that holds whichever asset (BTC/ETH/SOL) shows
the strongest volume-weighted momentum. Since shorting is not available,
this strategy rotates capital between assets rather than using traditional
pairs trading.

Logic:
- Calculate volume-weighted momentum for BTC, ETH, SOL
- Hold the asset with strongest momentum
- Exit BTC when alts show significantly stronger momentum
- Enter BTC when it shows strongest momentum

Rotation threshold (momentum gap) prevents excessive whipsawing.

No hardcoded stop-loss or sizing - uses RiskManager + PortfolioAllocator framework.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class RelativeStrengthRotationStrategy(BaseStrategy):
    """
    Relative strength rotation strategy for long-only portfolios.

    Rotates between BTC and strongest altcoin based on volume-weighted
    momentum scores. Only trades BTC pair (rotation to alts handled externally).
    """

    def __init__(
        self,
        momentum_window: int = 12,
        rotation_threshold: float = 0.015,
        min_hold_bars: int = 6,
    ):
        """
        Initialize strategy with tunable parameters.

        Args:
            momentum_window: Lookback period for momentum calculation (default 12 bars = 3h on 15m)
            rotation_threshold: Minimum momentum gap to trigger rotation (default 0.015 = 1.5%)
            min_hold_bars: Minimum bars to hold before rotating (default 6 = 90min on 15m)
        """
        self.momentum_window = momentum_window
        self.rotation_threshold = rotation_threshold
        self.min_hold_bars = min_hold_bars
        self._bars_since_entry = 0

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate relative strength rotation signal.

        Required features:
        - btc_momentum_12: 12-bar momentum for BTC (pct_change(12))
        - eth_momentum_12: 12-bar momentum for ETH
        - sol_momentum_12: 12-bar momentum for SOL
        - btc_volume: Current BTC volume
        - btc_volume_ma: BTC 20-period volume MA
        - eth_volume: Current ETH volume
        - eth_volume_ma: ETH 20-period volume MA
        - sol_volume: Current SOL volume
        - sol_volume_ma: SOL 20-period volume MA

        Args:
            pair: Trading pair (e.g., "BTC/USD")
            features: DataFrame with required momentum and volume features

        Returns:
            TradingSignal with BUY/SELL/HOLD direction for BTC rotation
        """
        if len(features) == 0:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]

        # Extract momentum features
        btc_momentum = latest.get("btc_momentum_12", 0.0)
        eth_momentum = latest.get("eth_momentum_12", 0.0)
        sol_momentum = latest.get("sol_momentum_12", 0.0)

        # Extract volume features
        btc_volume = latest.get("btc_volume", 1.0)
        btc_volume_ma = latest.get("btc_volume_ma", 1.0)
        eth_volume = latest.get("eth_volume", 1.0)
        eth_volume_ma = latest.get("eth_volume_ma", 1.0)
        sol_volume = latest.get("sol_volume", 1.0)
        sol_volume_ma = latest.get("sol_volume_ma", 1.0)

        # Handle missing data
        if pd.isna(btc_momentum) or pd.isna(eth_momentum) or pd.isna(sol_momentum):
            return TradingSignal(pair=pair)

        # Calculate volume-weighted momentum scores
        # Higher volume relative to MA indicates stronger conviction
        btc_vol_weight = btc_volume / btc_volume_ma if btc_volume_ma > 0 else 1.0
        eth_vol_weight = eth_volume / eth_volume_ma if eth_volume_ma > 0 else 1.0
        sol_vol_weight = sol_volume / sol_volume_ma if sol_volume_ma > 0 else 1.0

        btc_score = btc_momentum * btc_vol_weight
        eth_score = eth_momentum * eth_vol_weight
        sol_score = sol_momentum * sol_vol_weight

        # Find strongest asset
        strongest_score = max(btc_score, eth_score, sol_score)
        btc_is_strongest = (strongest_score == btc_score)

        # Calculate momentum gap (how much stronger is the leader)
        momentum_gap = abs(strongest_score - btc_score)

        # Track holding period
        self._bars_since_entry += 1

        # Only trade BTC pair
        if pair != "BTC/USD":
            return TradingSignal(pair=pair)

        # --- ENTRY CONDITIONS ---
        # BUY BTC when it shows strongest momentum with significant gap
        if btc_is_strongest and momentum_gap > self.rotation_threshold:
            self._bars_since_entry = 0  # Reset hold counter
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.5,  # Request 50% of portfolio for rotation
                confidence=0.6,
            )

        # --- EXIT CONDITIONS ---
        # SELL BTC when an alt shows significantly stronger momentum
        # Only rotate if we've held minimum period (avoid whipsaws)
        if (
            not btc_is_strongest
            and momentum_gap > self.rotation_threshold
            and self._bars_since_entry >= self.min_hold_bars
        ):
            self._bars_since_entry = 0  # Reset hold counter
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,  # Exit full BTC position (to rotate to alt)
                confidence=0.6,
            )

        # Default: HOLD current position
        return TradingSignal(pair=pair)
