"""
RiskManager optimized for HIGH DAILY COVERAGE

Modified version of risk.py with looser parameters to keep positions open longer.
Goal: Achieve 80-95% daily coverage while maintaining positive returns.

Key Changes from Standard RiskManager:
- Wider hard stop (15% vs 8%)
- Looser ATR trailing stop (0.8x vs 2x)
- Minimum hold time (4 hours on 15m = 16 bars)
- Higher max drawdown tolerance (40% vs 30%)
- More aggressive position sizing

Trade-off: Higher drawdowns but much better daily coverage.
"""

import logging
import math
from dataclasses import dataclass
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
    BLOCKED_MIN_HOLD_TIME = "BLOCKED_MIN_HOLD_TIME"


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


class RiskManagerHighCoverage:
    """
    Risk manager optimized for maximum daily coverage.

    Keeps positions open longer by using looser stops and minimum hold times.
    """

    def __init__(self, config: dict):
        """
        Initialize with high-coverage parameters.

        Default config (optimized for coverage):
            - hard_stop_pct: 0.15 (15% vs standard 8%)
            - atr_stop_multiplier: 0.8 (looser vs standard 2.0)
            - min_hold_bars: 16 (4 hours on 15m vs no minimum)
            - circuit_breaker_drawdown: 0.40 (40% vs 30%)
            - max_positions: 1
            - max_single_position_pct: 0.50 (50% vs 40%)
            - risk_per_trade_pct: 0.03 (3% vs 2% - more aggressive)
        """
        # Override defaults for high coverage
        self.config = {
            "hard_stop_pct": 0.15,
            "atr_stop_multiplier": 0.8,
            "min_hold_bars": 16,
            "circuit_breaker_drawdown": 0.40,
            "max_positions": 1,
            "max_single_position_pct": 0.50,
            "risk_per_trade_pct": 0.03,
            "expected_win_loss_ratio": 1.5,
        }
        # Override with user config
        self.config.update(config)

        self._trailing_stops: dict[str, float] = {}
        self._entry_prices: dict[str, float] = {}
        self._entry_bars: dict[str, int] = {}  # Track when we entered
        self._current_bar: int = 0
        self._portfolio_hwm: float = 0.0
        self._circuit_breaker_active: bool = False

    def check_stops(self, pair: str, current_price: float, current_atr: float) -> StopCheckResult:
        """
        Check stops with minimum hold time enforcement.

        Will NOT exit due to stops until minimum hold time is met.
        """
        self._current_bar += 1

        if pair not in self._entry_prices:
            return StopCheckResult(False, "No entry price", "")

        # Check if we've held for minimum time
        entry_bar = self._entry_bars.get(pair, 0)
        bars_held = self._current_bar - entry_bar
        min_hold = self.config.get("min_hold_bars", 16)

        if bars_held < min_hold:
            # Still within minimum hold period - don't exit
            return StopCheckResult(False, f"Minimum hold time not met ({bars_held}/{min_hold} bars)", "")

        entry_price = self._entry_prices[pair]
        hard_stop_pct = self.config.get("hard_stop_pct", 0.15)
        hard_stop_price = entry_price * (1 - hard_stop_pct)

        # Check hard stop
        if current_price <= hard_stop_price:
            logger.warning(
                f"HARD STOP triggered for {pair}: price={current_price:.4f} <= stop={hard_stop_price:.4f} "
                f"(held {bars_held} bars)"
            )
            return StopCheckResult(
                True,
                f"HARD STOP: price={current_price:.4f} <= stop={hard_stop_price:.4f}",
                "hard_pct"
            )

        # Check ATR trailing stop (much looser)
        if not pd.isna(current_atr) and current_atr > 0:
            atr_mult = self.config.get("atr_stop_multiplier", 0.8)
            new_atr_stop = current_price - atr_mult * current_atr
            current_trail = self._trailing_stops.get(pair, hard_stop_price)
            updated_trail = max(current_trail, new_atr_stop)
            self._trailing_stops[pair] = updated_trail

            if current_price <= updated_trail:
                logger.warning(
                    f"ATR TRAILING STOP triggered for {pair}: price={current_price:.4f} <= stop={updated_trail:.4f} "
                    f"(held {bars_held} bars)"
                )
                return StopCheckResult(
                    True,
                    f"ATR TRAILING STOP: price={current_price:.4f} <= stop={updated_trail:.4f}",
                    "atr_trailing"
                )

        return StopCheckResult(False, "Within stop bounds", "")

    def record_entry(self, pair: str, entry_price: float, initial_stop: float) -> None:
        """Record entry with bar counter for min hold time."""
        self._entry_prices[pair] = entry_price
        self._trailing_stops[pair] = initial_stop
        self._entry_bars[pair] = self._current_bar
        logger.info(
            f"Recorded entry for {pair} at {entry_price:.4f}, initial stop {initial_stop:.4f}, "
            f"bar {self._current_bar}"
        )

    def record_exit(self, pair: str) -> None:
        """Record exit and clean up tracking."""
        self._entry_prices.pop(pair, None)
        self._trailing_stops.pop(pair, None)
        self._entry_bars.pop(pair, None)
        logger.info(f"Recorded exit for {pair}")

    def get_current_stop(self, pair: str) -> Optional[float]:
        """Get current trailing stop for a position."""
        return self._trailing_stops.get(pair)

    def initialize_hwm(self, initial_portfolio_value: float) -> None:
        """Initialize portfolio high water mark."""
        self._portfolio_hwm = max(self._portfolio_hwm, initial_portfolio_value)

    def _get_cb_size_multiplier(self, drawdown: float) -> float:
        """
        Tiered circuit breaker with higher thresholds for coverage.

        Tiering (more lenient than standard):
        - drawdown < 15%: 1.0x (full)
        - drawdown 15-25%: 0.75x (3/4)
        - drawdown 25-40%: 0.5x (half)
        - drawdown >= 40%: 0.0x (no new positions)
        """
        if drawdown >= 0.40:
            return 0.0
        if drawdown >= 0.25:
            return 0.5
        if drawdown >= 0.15:
            return 0.75
        return 1.0

    def check_circuit_breaker(self, current_portfolio_value: float) -> bool:
        """Check circuit breaker with higher threshold (40% vs 30%)."""
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
        cb_threshold = self.config.get("circuit_breaker_drawdown", 0.40)

        if drawdown >= cb_threshold and not self._circuit_breaker_active:
            self._circuit_breaker_active = True
            logger.critical(
                f"CIRCUIT BREAKER ACTIVATED: drawdown {drawdown*100:.1f}% (threshold {cb_threshold*100:.1f}%), "
                f"HWM={self._portfolio_hwm:.0f}, current={current_portfolio_value:.0f}"
            )

        return self._circuit_breaker_active

    def get_cb_size_multiplier(self, current_portfolio_value: float) -> float:
        """Get circuit breaker size multiplier."""
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
        Size position with more aggressive parameters for coverage.

        Higher risk per trade (3% vs 2%) and larger max position size (50% vs 40%).
        """
        # Gate 1: Circuit breaker
        if self._circuit_breaker_active:
            return SizingResult(
                RiskDecision.BLOCKED_CIRCUIT_BREAKER,
                0.0, 0.0, 0.0, 0.0,
                "Circuit breaker active (40% drawdown)"
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

        # Compute tiered CB multiplier
        total_portfolio = free_balance_usd + sum(open_positions.values())
        if self._portfolio_hwm > 0:
            drawdown = max((self._portfolio_hwm - total_portfolio) / self._portfolio_hwm, 0.0)
            cb_mult = self._get_cb_size_multiplier(drawdown)
        else:
            cb_mult = 1.0

        # Combined multiplier
        effective_multiplier = regime_multiplier * cb_mult
        if effective_multiplier == 0.0:
            return SizingResult(
                RiskDecision.BLOCKED_ZERO_REGIME_MULTIPLIER,
                0.0, 0.0, 0.0, 0.0,
                "Combined multiplier is zero"
            )

        # Compute stop levels
        hard_stop_pct = self.config.get("hard_stop_pct", 0.15)
        hard_stop = current_price * (1 - hard_stop_pct)
        atr_mult = self.config.get("atr_stop_multiplier", 0.8)
        atr_stop = (
            (current_price - atr_mult * current_atr)
            if (not math.isnan(current_atr) and current_atr > 0)
            else hard_stop
        )
        initial_stop = max(hard_stop, atr_stop)

        stop_distance = current_price - initial_stop
        if stop_distance <= 0:
            stop_distance = current_price * hard_stop_pct

        # Kelly criterion
        b = self.config.get("expected_win_loss_ratio", 1.5)
        p = max(min(confidence, 1.0), 0.0)
        kelly = (p * b - (1.0 - p)) / b
        if kelly <= 0:
            return SizingResult(
                RiskDecision.BLOCKED_NEGATIVE_KELLY,
                0.0, 0.0, hard_stop, initial_stop,
                f"Negative Kelly edge (p={p:.2f}, b={b:.2f}, kelly={kelly:.3f})"
            )

        # More aggressive sizing (3% risk vs 2%)
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.03)
        risk_usd = (
            total_portfolio
            * risk_per_trade_pct
            * portfolio_weight
            * confidence
            * effective_multiplier
        )

        quantity = risk_usd / stop_distance
        target_usd = quantity * current_price

        # Higher concentration limit (50% vs 40%)
        max_single_pct = self.config.get("max_single_position_pct", 0.50)
        target_usd = min(target_usd, total_portfolio * max_single_pct, usable_balance)
        quantity = target_usd / current_price

        return SizingResult(
            RiskDecision.APPROVED,
            quantity,
            target_usd,
            hard_stop,
            initial_stop,
            f"Approved ${target_usd:.0f} (risk=${risk_usd:.0f}) mult={effective_multiplier:.2f} "
            f"conf={confidence:.2f} wt={portfolio_weight:.2f} [HIGH COVERAGE MODE]"
        )

    def dump_state(self) -> dict:
        """Serialize state for persistence."""
        return {
            "trailing_stops": self._trailing_stops,
            "entry_prices": self._entry_prices,
            "entry_bars": self._entry_bars,
            "current_bar": self._current_bar,
            "portfolio_hwm": self._portfolio_hwm,
            "circuit_breaker_active": self._circuit_breaker_active,
        }

    def load_state(self, state: dict) -> None:
        """Restore state from persistence."""
        self._trailing_stops = state.get("trailing_stops", {})
        self._entry_prices = state.get("entry_prices", {})
        self._entry_bars = state.get("entry_bars", {})
        self._current_bar = state.get("current_bar", 0)
        self._portfolio_hwm = state.get("portfolio_hwm", 0.0)
        self._circuit_breaker_active = state.get("circuit_breaker_active", False)
        logger.info(
            f"RiskManagerHighCoverage state loaded: HWM=${self._portfolio_hwm:.0f}, "
            f"CB={'active' if self._circuit_breaker_active else 'inactive'}, "
            f"positions={list(self._entry_prices.keys())}"
        )
