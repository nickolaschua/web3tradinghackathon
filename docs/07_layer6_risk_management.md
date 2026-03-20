# Layer 6 — Risk Management

## What This Layer Does

Risk Management is a hard gate between the strategy signal and the order submission. Every signal from the Strategy Engine must pass through this layer before anything is sent to the exchange. It enforces stop-losses on open positions, validates position sizing, applies the regime multiplier, checks portfolio-level limits, and triggers a circuit breaker when overall drawdown becomes unacceptable.

This layer operates on two timescales simultaneously:

1. **Every polling cycle (60 seconds):** Check all open positions for stop-loss breach, regardless of whether a new signal was generated
2. **On new signal:** Validate sizing, enforce concentration limits, apply regime multiplier, confirm the portfolio isn't in circuit-breaker mode

**This layer is deployed on EC2.** It is part of the live trading system.

---

## What This Layer Is Trying to Achieve

1. Ensure no single position can cause catastrophic portfolio damage — stops are the primary defence
2. Ensure position sizing is rational and consistent with validated backtest parameters
3. Prevent the bot from trading into a compounding loss spiral (circuit breaker)
4. Act as a sanity check that catches any edge case the strategy logic missed

---

## How It Contributes to the Bigger Picture

The Strategy Engine tells you what you want to do. The Risk layer tells you what you're allowed to do. Without this layer, a single bad trade — a flash crash, a news event, a strategy bug — can wipe out enough capital to make recovery mathematically impossible within the competition window.

The asymmetry is stark: a 30% loss requires a 43% gain to recover. A 50% loss requires a 100% gain. In a competition where you may have days to weeks, not months, recovery from a catastrophic drawdown is effectively impossible. The circuit breaker exists to prevent this scenario.

---

## Files in This Layer

```
execution/
└── risk.py     All risk management: stops, sizing, portfolio limits, circuit breaker
```

---

## `execution/risk.py`

```python
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class RiskDecision(Enum):
    APPROVED = "APPROVED"
    BLOCKED_CIRCUIT_BREAKER = "BLOCKED_CIRCUIT_BREAKER"
    BLOCKED_MAX_POSITIONS = "BLOCKED_MAX_POSITIONS"
    BLOCKED_CONCENTRATION = "BLOCKED_CONCENTRATION"
    BLOCKED_INSUFFICIENT_BALANCE = "BLOCKED_INSUFFICIENT_BALANCE"
    BLOCKED_ZERO_REGIME_MULTIPLIER = "BLOCKED_ZERO_REGIME_MULTIPLIER"

@dataclass
class SizingResult:
    decision: RiskDecision
    approved_quantity: float     # In units of the coin (0 if blocked)
    approved_usd_value: float    # In USD
    stop_price: float            # Hard stop price level
    trailing_stop_price: float   # ATR trailing stop level
    reason: str

@dataclass
class StopCheckResult:
    should_exit: bool
    exit_reason: str
    exit_type: str  # "atr_trailing", "hard_pct", "circuit_breaker"

class RiskManager:
    """
    All risk logic for the trading bot.
    
    Responsibilities:
    - ATR trailing stop management (updated each cycle)
    - Hard percentage stop as backstop
    - Portfolio circuit breaker
    - Position sizing with regime multiplier
    - Concentration limits
    """

    def __init__(self, config: dict):
        self.config = config
        # Track trailing stops per pair {pair: stop_price}
        self._trailing_stops: dict[str, float] = {}
        # Track entry prices per pair {pair: entry_price}
        self._entry_prices: dict[str, float] = {}
        # Portfolio high-water mark for drawdown calculation
        self._portfolio_hwm: float = 0.0
        self._circuit_breaker_active: bool = False

    # ── Public interface ───────────────────────────────────────────────────

    def check_stops(self, pair: str, current_price: float,
                    current_atr: float) -> StopCheckResult:
        """
        Called every polling cycle for every open position.
        Returns whether the position should be exited and why.
        
        This runs INDEPENDENTLY of signal generation.
        Do not wait for a strategy signal to check stops.
        """
        if pair not in self._entry_prices:
            return StopCheckResult(False, "No entry price recorded", "")

        entry_price = self._entry_prices[pair]
        hard_stop_pct = self.config.get("hard_stop_pct", 0.08)

        # ── Hard percentage stop ───────────────────────────────────────────
        hard_stop_price = entry_price * (1 - hard_stop_pct)
        if current_price <= hard_stop_price:
            reason = (f"HARD STOP: price={current_price:.4f} <= "
                      f"stop={hard_stop_price:.4f} ({hard_stop_pct:.1%} from entry)")
            logger.warning(reason)
            return StopCheckResult(True, reason, "hard_pct")

        # ── ATR trailing stop ──────────────────────────────────────────────
        if not pd.isna(current_atr) and current_atr > 0:
            atr_mult = self.config.get("atr_stop_multiplier", 2.0)
            new_atr_stop = current_price - atr_mult * current_atr

            # Trailing stop only moves UP, never down
            current_trail = self._trailing_stops.get(pair, hard_stop_price)
            updated_trail = max(current_trail, new_atr_stop)
            self._trailing_stops[pair] = updated_trail

            if current_price <= updated_trail:
                reason = (f"ATR TRAILING STOP: price={current_price:.4f} <= "
                          f"stop={updated_trail:.4f} (ATR={current_atr:.4f}×{atr_mult})")
                logger.warning(reason)
                return StopCheckResult(True, reason, "atr_trailing")

        return StopCheckResult(False, "Within stop bounds", "")

    def check_circuit_breaker(self, current_portfolio_value: float) -> bool:
        """
        Check and update the portfolio circuit breaker.
        Returns True if circuit breaker is active (no new trades allowed).
        """
        # Update high-water mark
        if current_portfolio_value > self._portfolio_hwm:
            self._portfolio_hwm = current_portfolio_value
            if self._circuit_breaker_active:
                # Only deactivate circuit breaker when portfolio fully recovers
                # Conservative: require full recovery before resuming
                logger.info(f"Circuit breaker reset: new HWM = ${self._portfolio_hwm:,.0f}")
                self._circuit_breaker_active = False

        if self._portfolio_hwm == 0:
            return False

        drawdown = (self._portfolio_hwm - current_portfolio_value) / self._portfolio_hwm

        cb_threshold = self.config.get("circuit_breaker_drawdown", 0.30)
        if drawdown >= cb_threshold and not self._circuit_breaker_active:
            self._circuit_breaker_active = True
            logger.critical(
                f"CIRCUIT BREAKER TRIGGERED: drawdown={drawdown:.1%} "
                f"(threshold={cb_threshold:.1%}), HWM=${self._portfolio_hwm:,.0f}, "
                f"current=${current_portfolio_value:,.0f}. NO NEW POSITIONS."
            )

        return self._circuit_breaker_active

    def size_new_position(self, pair: str, current_price: float,
                          current_atr: float, free_balance_usd: float,
                          open_positions: dict, regime_multiplier: float,
                          signal_size_pct: float) -> SizingResult:
        """
        Calculate approved position size for a new entry.
        
        Args:
            pair: Trading pair (e.g., "BTC/USD")
            current_price: Current price of the asset
            current_atr: Current ATR for stop calculation
            free_balance_usd: Available USD from /v3/balance (Free, not total)
            open_positions: {pair: usd_value} of current open positions
            regime_multiplier: From regime detection (0.0, 0.5, or 1.0)
            signal_size_pct: Suggested size from strategy (e.g., 0.25 = 25% of capital)
        
        Returns:
            SizingResult with approved quantity or block reason
        """
        # ── Gate: circuit breaker ──────────────────────────────────────────
        if self._circuit_breaker_active:
            return SizingResult(
                RiskDecision.BLOCKED_CIRCUIT_BREAKER, 0, 0, 0, 0,
                "Circuit breaker active — no new positions until portfolio recovers"
            )

        # ── Gate: regime multiplier ────────────────────────────────────────
        if regime_multiplier == 0.0:
            return SizingResult(
                RiskDecision.BLOCKED_ZERO_REGIME_MULTIPLIER, 0, 0, 0, 0,
                "Regime is BEAR — no new positions, hold cash"
            )

        # ── Gate: max concurrent positions ────────────────────────────────
        max_positions = self.config.get("max_positions", 3)
        current_position_count = len([p for p, v in open_positions.items() if v > 0])
        if current_position_count >= max_positions:
            return SizingResult(
                RiskDecision.BLOCKED_MAX_POSITIONS, 0, 0, 0, 0,
                f"Already at max {max_positions} concurrent positions"
            )

        # ── Gate: concentration limit ──────────────────────────────────────
        total_portfolio = free_balance_usd + sum(open_positions.values())
        max_single_pct = self.config.get("max_single_position_pct", 0.40)
        max_usd_in_pair = total_portfolio * max_single_pct

        # ── Gate: minimum free balance buffer ─────────────────────────────
        # Keep 5% as permanent buffer for fees + rounding
        usable_balance = free_balance_usd * 0.95

        if usable_balance < 100:  # Less than $100 free — too small to trade
            return SizingResult(
                RiskDecision.BLOCKED_INSUFFICIENT_BALANCE, 0, 0, 0, 0,
                f"Insufficient free balance: ${free_balance_usd:.2f}"
            )

        # ── Calculate position size ────────────────────────────────────────
        target_usd = total_portfolio * signal_size_pct * regime_multiplier
        # Apply concentration cap
        target_usd = min(target_usd, max_usd_in_pair)
        # Cannot exceed available free balance (minus buffer)
        target_usd = min(target_usd, usable_balance)

        quantity = target_usd / current_price

        # ── Calculate stops ────────────────────────────────────────────────
        hard_stop = current_price * (1 - self.config.get("hard_stop_pct", 0.08))
        atr_mult = self.config.get("atr_stop_multiplier", 2.0)
        atr_stop = current_price - atr_mult * current_atr if not pd.isna(current_atr) else hard_stop
        # Initial stop is the higher (less aggressive) of the two
        initial_stop = max(hard_stop, atr_stop)

        return SizingResult(
            decision=RiskDecision.APPROVED,
            approved_quantity=quantity,
            approved_usd_value=target_usd,
            stop_price=hard_stop,
            trailing_stop_price=initial_stop,
            reason=f"Approved: {target_usd:.0f} USD, regime_mult={regime_multiplier}",
        )

    def record_entry(self, pair: str, entry_price: float, initial_stop: float):
        """Call this after a successful order fill to record stop levels."""
        self._entry_prices[pair] = entry_price
        self._trailing_stops[pair] = initial_stop
        logger.info(f"Entry recorded: {pair} @ {entry_price:.4f}, stop @ {initial_stop:.4f}")

    def record_exit(self, pair: str):
        """Call this after a position is fully closed."""
        self._entry_prices.pop(pair, None)
        self._trailing_stops.pop(pair, None)
        logger.info(f"Exit recorded: {pair} cleared from risk tracking")

    def get_current_stop(self, pair: str) -> Optional[float]:
        return self._trailing_stops.get(pair)

    def initialize_hwm(self, initial_portfolio_value: float):
        """Call on startup with current portfolio value."""
        self._portfolio_hwm = max(self._portfolio_hwm, initial_portfolio_value)
```

---

## Stop-Loss Architecture

The system uses two stops in combination — they are not alternatives, they work together:

**ATR Trailing Stop (primary)**
- Set at entry: `entry_price - (atr_multiplier × ATR_14)`
- Updated every polling cycle: `stop = max(current_stop, current_price - atr_multiplier × ATR)`
- The stop moves up as price rises, locking in profit. It never moves down.
- When price falls and crosses the stop, exit immediately.
- The ATR multiplier (default 2.0) should match the value used in your walk-forward backtest.

**Hard Percentage Stop (backstop)**
- Fixed at entry: `entry_price × (1 - hard_stop_pct)`
- Default 8%. Does not trail.
- This catches catastrophic gap-downs where the price jumps below the ATR stop in a single candle.
- Think of it as insurance against flash crashes or news events.

Both are checked independently every 60 seconds. If either is breached, exit the full position immediately.

**Why both:**
- ATR trailing stop: rides winners and cuts losers proportionally to current volatility
- Hard percentage stop: absolute maximum loss guarantee per position

---

## Portfolio Circuit Breaker

The circuit breaker is a last resort that prevents a single bad strategy phase from compounding into total ruin.

When portfolio value falls 30% from its high-water mark (e.g., from $50,000 to $35,000), the circuit breaker activates. While active:
- No new BUY orders are submitted, regardless of signal strength
- Existing positions continue to be monitored for stop-loss exits
- The situation is logged at CRITICAL level and a Telegram alert is sent

The circuit breaker deactivates only when the portfolio fully recovers to the previous high-water mark. This is intentionally conservative — if you've lost 30%, the market conditions that caused the loss have not necessarily resolved.

In a competition context, the correct action when the circuit breaker fires is to manually SSH into EC2 and assess whether the strategy logic is working correctly or whether there is a bug. This is one of the three scenarios defined in your pre-competition runbook where manual intervention is appropriate.

---

## Half-Kelly Position Sizing

The base position size (in `config.yaml` as `base_position_size`) should be derived from your backtest statistics using the Kelly Criterion, halved for safety:

```
Kelly fraction = (win_rate × avg_win - loss_rate × avg_loss) / avg_win
Half-Kelly = Kelly fraction / 2
```

From typical momentum backtest results (60% win rate, 1.5:1 reward/risk):
```
Kelly = (0.60 × 1.5 - 0.40 × 1.0) / 1.5 = (0.90 - 0.40) / 1.5 = 0.333
Half-Kelly = 0.167 ≈ 17%
```

With the regime multiplier at 1.0x (BULL), a 17% position in BTC on $50,000 capital is ~$8,500. Comfortable, leaves room for 3 simultaneous positions, and won't be ruinous if the stop is hit.

Run your specific backtest statistics through this formula and use the result as your `base_position_size` value. Do not guess.

---

## Failure Modes This Layer Prevents

| Failure | Prevention |
|---|---|
| Single position causing portfolio ruin | ATR trailing stop + hard 8% stop, checked every cycle |
| Holding through bear regime losses | BEAR_TREND → regime_multiplier=0 → no new positions |
| Portfolio drawdown spiral | 30% circuit breaker prevents compounding losses |
| Over-concentration in one pair | 40% maximum per pair enforced on every sizing request |
| Sizing from locked balance | Only `Free` balance used, never total wallet balance |
| Stops not checked between signals | Stop check runs every 60-second poll, not just on signal |
| Stop levels lost on crash/restart | Stop prices written to state.json and reloaded on startup |
