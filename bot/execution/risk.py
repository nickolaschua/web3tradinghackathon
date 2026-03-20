"""
RiskManager: ATR-based stop-loss, tiered circuit breaker, position sizing.

Provides:
- check_stops: Monitors existing positions for hard % and ATR trailing stops
- check_circuit_breaker: Detects drawdown tiers and enforces position sizing restrictions
- size_new_position: Computes entry stop levels and sizes positions through all gates
- dump_state / load_state: Crash-safe persistence of trailing stops, entry prices, HWM, CB status
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskDecision(Enum):
    """Sizing decision outcome for a new position."""
    APPROVED = "APPROVED"
    BLOCKED_CIRCUIT_BREAKER = "BLOCKED_CIRCUIT_BREAKER"
    BLOCKED_MAX_POSITIONS = "BLOCKED_MAX_POSITIONS"
    BLOCKED_CONCENTRATION = "BLOCKED_CONCENTRATION"
    BLOCKED_INSUFFICIENT_BALANCE = "BLOCKED_INSUFFICIENT_BALANCE"
    BLOCKED_ZERO_REGIME_MULTIPLIER = "BLOCKED_ZERO_REGIME_MULTIPLIER"
    BLOCKED_NEGATIVE_KELLY = "BLOCKED_NEGATIVE_KELLY"


@dataclass
class SizingResult:
    """Outcome of size_new_position() gate checks."""
    decision: RiskDecision
    approved_quantity: float
    approved_usd_value: float
    stop_price: float
    trailing_stop_price: float
    reason: str


@dataclass
class StopCheckResult:
    """Outcome of check_stops() for an existing position."""
    should_exit: bool
    exit_reason: str
    exit_type: str


class RiskManager:
    """
    Gate all new positions through ATR-based stop-loss, tiered circuit breaker, and sizing logic.

    Must persist full state (trailing_stops, entry_prices, portfolio_hwm, circuit_breaker_active)
    across restarts so trailing stops and circuit breaker status survive a crash.
    """

    def __init__(self, config: dict):
        """
        Initialize RiskManager with config defaults.

        Args:
            config: Dict with optional keys:
                - hard_stop_pct: Hard stop loss % (default 0.08)
                - atr_stop_multiplier: ATR trailing stop multiplier (default 2.0)
                - circuit_breaker_drawdown: CB activation drawdown threshold (default 0.30)
                - max_positions: Maximum concurrent positions (default 1)
                - max_single_position_pct: Max position size as % of portfolio (default 0.40)
        """
        self.config = config
        self._trailing_stops: dict[str, float] = {}
        self._entry_prices: dict[str, float] = {}
        self._portfolio_hwm: float = 0.0
        self._circuit_breaker_active: bool = False

    def check_stops(self, pair: str, current_price: float, current_atr: float) -> StopCheckResult:
        """
        Check hard % and ATR trailing stops for an existing position.

        Called every 60s cycle independently of signal generation.
        Trailing stop only moves UP, never down.

        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            current_price: Current market price
            current_atr: Current ATR value (may be NaN or 0)

        Returns:
            StopCheckResult with should_exit flag and exit type
        """
        if pair not in self._entry_prices:
            return StopCheckResult(False, "No entry price", "")

        entry_price = self._entry_prices[pair]
        hard_stop_pct = self.config.get("hard_stop_pct", 0.08)
        hard_stop_price = entry_price * (1 - hard_stop_pct)

        # Check hard stop
        if current_price <= hard_stop_price:
            logger.warning(
                f"HARD STOP triggered for {pair}: price={current_price:.4f} <= stop={hard_stop_price:.4f}"
            )
            return StopCheckResult(
                True,
                f"HARD STOP: price={current_price:.4f} <= stop={hard_stop_price:.4f}",
                "hard_pct"
            )

        # Check ATR trailing stop
        if not pd.isna(current_atr) and current_atr > 0:
            atr_mult = self.config.get("atr_stop_multiplier", 2.0)
            new_atr_stop = current_price - atr_mult * current_atr
            current_trail = self._trailing_stops.get(pair, hard_stop_price)
            updated_trail = max(current_trail, new_atr_stop)  # Only moves UP
            self._trailing_stops[pair] = updated_trail

            if current_price <= updated_trail:
                logger.warning(
                    f"ATR TRAILING STOP triggered for {pair}: price={current_price:.4f} <= stop={updated_trail:.4f}"
                )
                return StopCheckResult(
                    True,
                    f"ATR TRAILING STOP: price={current_price:.4f} <= stop={updated_trail:.4f}",
                    "atr_trailing"
                )

        return StopCheckResult(False, "Within stop bounds", "")

    def record_entry(self, pair: str, entry_price: float, initial_stop: float) -> None:
        """
        Record a new position entry.

        Args:
            pair: Trading pair
            entry_price: Entry price
            initial_stop: Initial stop loss price
        """
        self._entry_prices[pair] = entry_price
        self._trailing_stops[pair] = initial_stop
        logger.info(f"Recorded entry for {pair} at {entry_price:.4f}, initial stop {initial_stop:.4f}")

    def record_exit(self, pair: str) -> None:
        """
        Record a position exit (remove from tracking).

        Args:
            pair: Trading pair
        """
        self._entry_prices.pop(pair, None)
        self._trailing_stops.pop(pair, None)
        logger.info(f"Recorded exit for {pair}")

    def get_current_stop(self, pair: str) -> Optional[float]:
        """
        Get current trailing stop for a position.

        Args:
            pair: Trading pair

        Returns:
            Current stop price or None if no position
        """
        return self._trailing_stops.get(pair)

    def initialize_hwm(self, initial_portfolio_value: float) -> None:
        """
        Initialize or update portfolio high water mark.

        Args:
            initial_portfolio_value: Initial portfolio value in USD
        """
        self._portfolio_hwm = max(self._portfolio_hwm, initial_portfolio_value)

    def _get_cb_size_multiplier(self, drawdown: float) -> float:
        """
        Get tiered circuit breaker size multiplier based on drawdown.

        Tiering:
        - drawdown < 10%: 1.0x (full)
        - drawdown 10-20%: 0.5x (half)
        - drawdown 20-30%: 0.25x (quarter)
        - drawdown >= 30%: 0.0x (no new positions)

        Args:
            drawdown: Drawdown ratio (0.0 = no loss, 1.0 = 100% loss)

        Returns:
            Size multiplier (0.0 to 1.0)
        """
        if drawdown >= 0.30:
            return 0.0
        if drawdown >= 0.20:
            return 0.25
        if drawdown >= 0.10:
            return 0.50
        return 1.0

    def check_circuit_breaker(self, current_portfolio_value: float) -> bool:
        """
        Check for drawdown-based circuit breaker activation.

        Activates at 30% drawdown, deactivates when recovering to HWM.

        Args:
            current_portfolio_value: Current total portfolio value in USD

        Returns:
            True if circuit breaker is active
        """
        # Update HWM if recovering
        if current_portfolio_value > self._portfolio_hwm:
            old_hwm = self._portfolio_hwm
            self._portfolio_hwm = current_portfolio_value
            if self._circuit_breaker_active:
                self._circuit_breaker_active = False
                logger.info(
                    f"Circuit breaker DEACTIVATED: portfolio recovered from {old_hwm:.0f} to {current_portfolio_value:.0f}"
                )

        if self._portfolio_hwm == 0:
            return False

        drawdown = (self._portfolio_hwm - current_portfolio_value) / self._portfolio_hwm
        cb_threshold = self.config.get("circuit_breaker_drawdown", 0.30)

        if drawdown >= cb_threshold and not self._circuit_breaker_active:
            self._circuit_breaker_active = True
            logger.critical(
                f"CIRCUIT BREAKER ACTIVATED: drawdown {drawdown*100:.1f}% (threshold {cb_threshold*100:.1f}%), "
                f"HWM={self._portfolio_hwm:.0f}, current={current_portfolio_value:.0f}"
            )

        return self._circuit_breaker_active

    def get_cb_size_multiplier(self, current_portfolio_value: float) -> float:
        """
        Get circuit breaker size multiplier for current portfolio state.

        If CB is active (hard stop at 30%), returns 0.0.
        Otherwise, returns tiered multiplier based on current drawdown.

        Args:
            current_portfolio_value: Current portfolio value in USD

        Returns:
            Size multiplier (0.0 to 1.0)
        """
        if self._circuit_breaker_active:
            return 0.0

        if self._portfolio_hwm == 0:
            return 1.0

        drawdown = (self._portfolio_hwm - current_portfolio_value) / self._portfolio_hwm
        return self._get_cb_size_multiplier(max(drawdown, 0.0))

    def size_new_position(
        self,
        pair: str,
        current_price: float,
        current_atr: float,
        free_balance_usd: float,
        open_positions: dict,
        regime_multiplier: float,
        confidence: float = 0.7,
        portfolio_weight: float = 1.0,
    ) -> SizingResult:
        """
        Size a new position through all gate checks using equal dollar risk sizing.

        Gates (in order):
        1. Circuit breaker active
        2. Regime multiplier == 0.0 (BEAR regime)
        3. Max positions limit
        4. Insufficient balance
        5. Combined multiplier == 0.0 (regime × CB)
        6. Negative Kelly criterion (no edge)

        Sizing formula (equal dollar risk):
            risk_usd = total_portfolio × risk_per_trade_pct × portfolio_weight
                       × confidence × effective_multiplier
            quantity  = risk_usd / stop_distance

        This risks the same dollar amount on every trade regardless of volatility.
        Stop distance naturally scales quantity: wide stops → smaller position.

        Args:
            pair: Trading pair
            current_price: Current market price
            current_atr: Current ATR (may be NaN)
            free_balance_usd: Free balance in USD (never total wallet)
            open_positions: Dict of open positions {pair: value_usd}
            regime_multiplier: Regime size multiplier (1.0/0.5/0.0)
            confidence: Signal confidence in [0, 1] (from strategy, default 0.7)
            portfolio_weight: Pair weight from PortfolioAllocator (default 1.0)

        Returns:
            SizingResult with decision, approved quantity, and stop levels
        """
        # Gate 1: Circuit breaker
        if self._circuit_breaker_active:
            return SizingResult(
                RiskDecision.BLOCKED_CIRCUIT_BREAKER,
                0.0, 0.0, 0.0, 0.0,
                "Circuit breaker active"
            )

        # Gate 2: Regime multiplier
        if regime_multiplier == 0.0:
            return SizingResult(
                RiskDecision.BLOCKED_ZERO_REGIME_MULTIPLIER,
                0.0, 0.0, 0.0, 0.0,
                "BEAR regime"
            )

        # Gate 3: Max positions check
        max_positions = self.config.get("max_positions", 1)
        active_positions = sum(1 for v in open_positions.values() if v > 0)
        if active_positions >= max_positions:
            return SizingResult(
                RiskDecision.BLOCKED_MAX_POSITIONS,
                0.0, 0.0, 0.0, 0.0,
                f"Max positions ({max_positions}) reached"
            )

        # Gate 4: Insufficient balance
        usable_balance = free_balance_usd * 0.95
        if usable_balance < 100:
            return SizingResult(
                RiskDecision.BLOCKED_INSUFFICIENT_BALANCE,
                0.0, 0.0, 0.0, 0.0,
                f"Usable balance ${usable_balance:.0f} < $100 minimum"
            )

        # Compute tiered circuit breaker size multiplier
        total_portfolio = free_balance_usd + sum(open_positions.values())
        if self._portfolio_hwm > 0:
            drawdown = max((self._portfolio_hwm - total_portfolio) / self._portfolio_hwm, 0.0)
            cb_mult = self._get_cb_size_multiplier(drawdown)
        else:
            cb_mult = 1.0

        # Gate 5: Combined multiplier (regime × circuit breaker)
        effective_multiplier = regime_multiplier * cb_mult
        if effective_multiplier == 0.0:
            return SizingResult(
                RiskDecision.BLOCKED_ZERO_REGIME_MULTIPLIER,
                0.0, 0.0, 0.0, 0.0,
                "Combined multiplier is zero"
            )

        # Compute stop levels (needed before sizing to get stop_distance)
        hard_stop_pct = self.config.get("hard_stop_pct", 0.08)
        hard_stop = current_price * (1 - hard_stop_pct)
        atr_mult = self.config.get("atr_stop_multiplier", 2.0)
        atr_stop = (
            (current_price - atr_mult * current_atr)
            if (not math.isnan(current_atr) and current_atr > 0)
            else hard_stop
        )
        initial_stop = max(hard_stop, atr_stop)

        # Stop distance must be positive; fallback to hard_stop_pct of price
        stop_distance = current_price - initial_stop
        if stop_distance <= 0:
            stop_distance = current_price * hard_stop_pct

        # Gate 6: Half-Kelly criterion — block if no positive edge
        # p = confidence (win probability proxy), b = avg_win / avg_loss
        b = self.config.get("expected_win_loss_ratio", 1.5)
        p = max(min(confidence, 1.0), 0.0)
        kelly = (p * b - (1.0 - p)) / b  # full Kelly fraction
        if kelly <= 0:
            return SizingResult(
                RiskDecision.BLOCKED_NEGATIVE_KELLY,
                0.0, 0.0, hard_stop, initial_stop,
                f"Negative Kelly edge (p={p:.2f}, b={b:.2f}, kelly={kelly:.3f})"
            )

        # Equal dollar risk sizing
        # risk_usd = how many dollars we're willing to lose on this trade
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.02)
        risk_usd = (
            total_portfolio
            * risk_per_trade_pct
            * portfolio_weight
            * confidence
            * effective_multiplier
        )

        quantity = risk_usd / stop_distance
        target_usd = quantity * current_price

        # Apply concentration and liquidity caps
        max_single_pct = self.config.get("max_single_position_pct", 0.40)
        target_usd = min(target_usd, total_portfolio * max_single_pct, usable_balance)
        quantity = target_usd / current_price

        return SizingResult(
            RiskDecision.APPROVED,
            quantity,
            target_usd,
            hard_stop,
            initial_stop,
            f"Approved ${target_usd:.0f} (risk=${risk_usd:.0f}) mult={effective_multiplier:.2f} "
            f"conf={confidence:.2f} wt={portfolio_weight:.2f}"
        )

    def dump_state(self) -> dict:
        """
        Serialize RiskManager state for persistence.

        Must include: trailing_stops, entry_prices, portfolio_hwm, circuit_breaker_active

        Returns:
            Dict with full state snapshot
        """
        return {
            "trailing_stops": self._trailing_stops,
            "entry_prices": self._entry_prices,
            "portfolio_hwm": self._portfolio_hwm,
            "circuit_breaker_active": self._circuit_breaker_active,
        }

    def load_state(self, state: dict) -> None:
        """
        Restore RiskManager state from persistence.

        Args:
            state: Dict from dump_state()
        """
        self._trailing_stops = state.get("trailing_stops", {})
        self._entry_prices = state.get("entry_prices", {})
        self._portfolio_hwm = state.get("portfolio_hwm", 0.0)
        self._circuit_breaker_active = state.get("circuit_breaker_active", False)
        logger.info(
            f"RiskManager state loaded: HWM=${self._portfolio_hwm:.0f}, "
            f"CB={'active' if self._circuit_breaker_active else 'inactive'}, "
            f"positions={list(self._entry_prices.keys())}"
        )
