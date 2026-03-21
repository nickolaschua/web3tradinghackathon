"""
BTC Correlation Divergence Strategy

Trades BTC based on divergences from correlated assets (ETH) combined with
funding rate extremes. Long-only strategy that captures mean-reversion when
BTC underperforms despite high correlation.

Entry Logic:
- High correlation between BTC and ETH (>0.7)
- ETH outperforms BTC by significant margin (>0.5%)
- Funding rate shows extreme pessimism (z-score < -1.5)

Exit Logic:
- Correlation breaks down (<0.5) - relationship no longer valid
- Let RiskManager handle stop-loss via ATR trailing stops

No hardcoded stop-loss or sizing - uses RiskManager + PortfolioAllocator framework.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class BTCCorrelationDivergenceStrategy(BaseStrategy):
    """
    BTC correlation divergence strategy.

    Detects when BTC underperforms highly correlated assets (ETH) while
    funding rates show extreme pessimism, signaling a mean-reversion opportunity.
    """

    def __init__(
        self,
        corr_threshold_entry: float = 0.7,
        corr_threshold_exit: float = 0.5,
        outperformance_threshold: float = 0.005,
        funding_zscore_threshold: float = -1.5,
    ):
        """
        Initialize strategy with tunable parameters.

        Args:
            corr_threshold_entry: Minimum BTC-ETH correlation for entry (default 0.7)
            corr_threshold_exit: Correlation below which to exit (default 0.5)
            outperformance_threshold: ETH outperformance required for entry (default 0.005 = 0.5%)
            funding_zscore_threshold: Funding z-score threshold for entry (default -1.5)
        """
        self.corr_threshold_entry = corr_threshold_entry
        self.corr_threshold_exit = corr_threshold_exit
        self.outperformance_threshold = outperformance_threshold
        self.funding_zscore_threshold = funding_zscore_threshold

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate correlation divergence signal.

        Required features:
        - btc_eth_corr: Rolling 20-bar correlation between BTC and ETH returns
        - eth_return: Latest ETH return (1 bar)
        - btc_return: Latest BTC return (1 bar)
        - btc_funding_zscore: Z-score of BTC funding rate (90-period)

        Args:
            pair: Trading pair (e.g., "BTC/USD")
            features: DataFrame with required correlation and funding features

        Returns:
            TradingSignal with BUY/SELL/HOLD direction, size request, and confidence
        """
        if len(features) == 0:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]

        # Extract required features
        btc_eth_corr = latest.get("btc_eth_corr", 0.0)
        eth_return = latest.get("eth_return", 0.0)
        btc_return = latest.get("btc_return", 0.0)
        funding_zscore = latest.get("btc_funding_zscore", 0.0)

        # Handle missing data
        if pd.isna(btc_eth_corr) or pd.isna(eth_return) or pd.isna(btc_return):
            return TradingSignal(pair=pair)

        # Calculate ETH outperformance
        eth_outperformance = eth_return - btc_return

        # --- ENTRY CONDITIONS ---
        # BUY when: high correlation + ETH outperforms + funding extreme
        if (
            btc_eth_corr > self.corr_threshold_entry
            and eth_outperformance > self.outperformance_threshold
            and funding_zscore < self.funding_zscore_threshold
        ):
            # Higher confidence when conditions are more extreme
            # Base confidence 0.5, add 0.1 for each unit of extreme funding
            confidence = min(0.9, 0.5 + abs(funding_zscore - self.funding_zscore_threshold) * 0.1)

            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.3,  # Request 30% of portfolio (RiskManager will adjust via Kelly/confidence)
                confidence=confidence,
            )

        # --- EXIT CONDITIONS ---
        # SELL when correlation breaks down (mean reversion complete or relationship invalid)
        if btc_eth_corr < self.corr_threshold_exit:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,  # Exit full position
                confidence=0.6,
            )

        # Default: HOLD current position
        return TradingSignal(pair=pair)
