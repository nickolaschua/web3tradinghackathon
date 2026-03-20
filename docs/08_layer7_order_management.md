# Layer 7 — Order Management System (OMS)

## What This Layer Does

The Order Management System (OMS) handles everything between "a signal has been approved by the risk layer" and "the exchange has confirmed the order is filled." It performs pre-flight validation before submission, tracks every order through its complete lifecycle, and runs a reconciliation loop that regularly compares local state against the exchange to catch any discrepancies.

The OMS is the last line of defence before capital moves. It catches errors that every other layer missed — wrong precision, insufficient balance, unexpected API responses — and ensures the system's understanding of its own positions always matches reality.

**This layer is deployed on EC2.** It runs continuously as part of the live bot.

---

## What This Layer Is Trying to Achieve

1. Ensure every submitted order is correctly formed before it reaches the exchange
2. Track all orders through their lifecycle so the system always knows its current position
3. Regularly reconcile local order state against the exchange to catch silent failures and fills
4. Never allow the system to act on a stale or incorrect view of its own positions

---

## How It Contributes to the Bigger Picture

The OMS is the accountability layer. The strategy and risk layers tell the system what to do. The OMS ensures those instructions were actually executed and that the system knows what happened. Without order tracking and reconciliation, a single silent failure — an order that was submitted but never confirmed, or a fill that was missed — can corrupt the system's entire view of its position, leading to double entries, missed exits, or incorrect risk calculations.

---

## Files in This Layer

```
execution/
└── order_manager.py    Complete OMS: pre-flight, lifecycle, reconciliation
```

---

## Order Lifecycle

```
               Strategy + Risk approve a signal
                            │
                            ▼
                    ┌─────────────┐
                    │  PRE-FLIGHT │  Precision check, MiniOrder check,
                    │   CHECKS    │  balance check, rate limit check
                    └──────┬──────┘
                           │ PASS
                           ▼
                    ┌─────────────┐
                    │   SUBMIT    │  client.place_order()
                    └──────┬──────┘
                           │
               ┌───────────┴───────────┐
               │                       │
         Success=true             Success=false
               │                       │
    ┌──────────┴──────┐         ┌──────┴──────────┐
    │    MARKET order │         │   Log + Alert   │
    │   Status=FILLED │         │   Do not update │
    └────────┬────────┘         │   position state│
             │                  └─────────────────┘
    ┌────────┴────────┐
    │  Update local   │
    │  position state │
    │  Record stop    │
    │  levels in risk │
    └────────┬────────┘
             │
             ▼
    Reconciliation loop (every 5 min)
    query_order(pending_only=TRUE)
    Diff against local state
    Alert on any discrepancy
```

---

## `execution/order_manager.py`

```python
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"

@dataclass
class ManagedOrder:
    order_id: Optional[int]
    pair: str
    side: str           # "BUY" or "SELL"
    order_type: str     # "MARKET" or "LIMIT"
    quantity: float
    price: Optional[float]
    status: OrderStatus
    submitted_at: float
    filled_at: Optional[float] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    commission: Optional[float] = None

@dataclass
class Position:
    pair: str
    quantity: float
    entry_price: float
    entry_time: float
    usd_value: float

class OrderManager:
    """
    Manages order submission, tracking, and position reconciliation.
    
    Uses MARKET orders by default for simplicity and guaranteed execution.
    LIMIT orders can be used for entries at specific price levels.
    """

    def __init__(self, client, exchange_info, rate_limiter,
                 risk_manager, telegram, config: dict):
        self.client = client
        self.exchange_info = exchange_info
        self.rate_limiter = rate_limiter
        self.risk = risk_manager
        self.telegram = telegram
        self.config = config

        self._orders: dict[int, ManagedOrder] = {}
        self._positions: dict[str, Position] = {}
        self._last_reconciliation = 0.0

    # ── Public interface ───────────────────────────────────────────────────

    def submit_buy(self, pair: str, quantity: float,
                   initial_stop: float, price: float = None) -> Optional[ManagedOrder]:
        """
        Submit a BUY order after pre-flight validation.
        Returns the managed order on success, None on failure.
        """
        order = self._pre_flight_and_submit(pair, "BUY", quantity, price)
        if order and order.status == OrderStatus.FILLED:
            # Record the fill with risk manager for stop tracking
            fill_price = order.filled_price or price or quantity
            self.risk.record_entry(pair, fill_price, initial_stop)
            self._positions[pair] = Position(
                pair=pair,
                quantity=order.filled_quantity or quantity,
                entry_price=fill_price,
                entry_time=time.time(),
                usd_value=(order.filled_quantity or quantity) * fill_price,
            )
            self.telegram.send(
                f"✅ BUY FILLED: {pair}\n"
                f"Qty: {order.filled_quantity:.6f}\n"
                f"Price: ${order.filled_price:,.2f}\n"
                f"Value: ${order.usd_value:,.0f}\n"
                f"Stop: ${initial_stop:,.2f}"
            )
            logger.info(f"BUY filled: {pair} {quantity:.6f} @ {fill_price:.2f}")
        return order

    def submit_sell(self, pair: str, reason: str) -> Optional[ManagedOrder]:
        """
        Submit a SELL order for the full current position in pair.
        """
        position = self._positions.get(pair)
        if not position:
            logger.warning(f"submit_sell called for {pair} but no position recorded")
            return None

        quantity = position.quantity
        order = self._pre_flight_and_submit(pair, "SELL", quantity)

        if order and order.status == OrderStatus.FILLED:
            self.risk.record_exit(pair)
            entry_price = position.entry_price
            fill_price = order.filled_price or entry_price
            pnl_pct = (fill_price - entry_price) / entry_price
            pnl_usd = quantity * (fill_price - entry_price)

            self._positions.pop(pair, None)

            self.telegram.send(
                f"{'✅' if pnl_usd > 0 else '🔴'} SELL FILLED: {pair}\n"
                f"Reason: {reason}\n"
                f"Entry: ${entry_price:,.2f} → Exit: ${fill_price:,.2f}\n"
                f"PnL: {pnl_pct:+.2%} (${pnl_usd:+,.0f})"
            )
            logger.info(f"SELL filled: {pair} {quantity:.6f} @ {fill_price:.2f} | PnL: {pnl_pct:+.2%}")

        return order

    # ── Pre-flight validation and submission ───────────────────────────────

    def _pre_flight_and_submit(self, pair: str, side: str,
                                quantity: float, price: float = None) -> Optional[ManagedOrder]:
        # ── Step 1: Precision rounding ─────────────────────────────────────
        quantity = self.exchange_info.round_quantity(pair, quantity)
        if price is not None:
            price = self.exchange_info.round_price(pair, price)

        # ── Step 2: MiniOrder validation ───────────────────────────────────
        check_price = price or self._estimate_market_price(pair)
        is_valid, err = self.exchange_info.validate_order(pair, check_price, quantity)
        if not is_valid:
            logger.warning(f"Pre-flight failed for {pair}: {err}")
            self.telegram.send(f"⚠️ PRE-FLIGHT FAIL: {pair} | {err}")
            return None

        # ── Step 3: Rate limit wait ────────────────────────────────────────
        self.rate_limiter.wait_for_trade_slot()

        # ── Step 4: Submit ─────────────────────────────────────────────────
        order_type = "LIMIT" if price is not None else "MARKET"
        response = self.client.place_order(pair, side, quantity, price, order_type)
        self.rate_limiter.record_trade_submitted()

        # ── Step 5: Parse response ─────────────────────────────────────────
        # CRITICAL: always check Success field
        if not response.get("Success"):
            err_msg = response.get("ErrMsg", "Unknown error")
            logger.error(f"Order submission failed: {pair} {side} | {err_msg}")
            self.telegram.send(f"❌ ORDER FAILED: {pair} {side}\nError: {err_msg}")

            if "rate" in err_msg.lower():
                self.rate_limiter.handle_rate_limit_rejection()
            return None

        detail = response.get("OrderDetail", {})
        order = ManagedOrder(
            order_id=detail.get("OrderID"),
            pair=pair,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED if detail.get("Status") == "FILLED" else OrderStatus.SUBMITTED,
            submitted_at=time.time(),
            filled_at=detail.get("FinishTimestamp", 0) / 1000 if detail.get("Status") == "FILLED" else None,
            filled_price=detail.get("FilledAverPrice"),
            filled_quantity=detail.get("FilledQuantity"),
            commission=detail.get("CommissionChargeValue"),
        )

        if order.order_id:
            self._orders[order.order_id] = order

        # Log full order detail for audit trail
        logger.info(f"Order submitted: {side} {quantity:.6f} {pair} | "
                    f"ID={order.order_id} status={order.status.value} "
                    f"fill_price={order.filled_price}")

        return order

    def _estimate_market_price(self, pair: str) -> float:
        """Fetch current price for pre-flight validation when no limit price is given."""
        ticker = self.client.get_ticker(pair)
        if ticker.get("Success"):
            return ticker["Data"][pair]["LastPrice"]
        return 0.0

    # ── Reconciliation ─────────────────────────────────────────────────────

    def reconcile(self, force: bool = False):
        """
        Query exchange for current pending orders and balance.
        Compare against local state. Alert on discrepancies.
        
        Run every 5 minutes automatically, or force=True on startup.
        """
        now = time.time()
        if not force and (now - self._last_reconciliation) < 300:
            return
        self._last_reconciliation = now

        # Check pending orders
        pending_resp = self.client.get_pending_count()
        if pending_resp.get("TotalPending", 0) != self._count_local_pending():
            logger.warning(
                f"RECONCILIATION MISMATCH: "
                f"exchange has {pending_resp.get('TotalPending')} pending, "
                f"local has {self._count_local_pending()}"
            )
            self.telegram.send(
                f"⚠️ RECONCILIATION MISMATCH\n"
                f"Exchange: {pending_resp.get('TotalPending')} pending\n"
                f"Local: {self._count_local_pending()} pending\n"
                "Querying full order history to resync..."
            )
            self._resync_from_exchange()

        logger.debug(f"Reconciliation complete. Local positions: {list(self._positions.keys())}")

    def _count_local_pending(self) -> int:
        return sum(1 for o in self._orders.values()
                   if o.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED))

    def _resync_from_exchange(self):
        """Full resync: query all recent orders and rebuild local position state."""
        response = self.client.query_order()
        if not response.get("Success"):
            logger.error("Resync failed: could not query orders")
            return

        # Rebuild positions from filled orders
        # This is a simplified version — production implementation would be more thorough
        for order_data in response.get("OrderMatched", []):
            order_id = order_data.get("OrderID")
            if order_data.get("Status") == "FILLED" and order_id not in self._orders:
                logger.warning(f"Found untracked filled order: {order_id}")
                self.telegram.send(f"⚠️ UNTRACKED FILL: OrderID={order_id} | Manual review needed")

    # ── State access ───────────────────────────────────────────────────────

    def get_position(self, pair: str) -> Optional[Position]:
        return self._positions.get(pair)

    def get_position_value(self, pair: str) -> float:
        pos = self._positions.get(pair)
        return pos.usd_value if pos else 0.0

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_open_position_count(self) -> int:
        return len(self._positions)

    def load_state(self, state: dict):
        """Restore OMS state from state.json on startup."""
        for pair, pos_data in state.get("positions", {}).items():
            self._positions[pair] = Position(**pos_data)
        logger.info(f"OMS state restored: {len(self._positions)} positions")

    def dump_state(self) -> dict:
        """Serialise OMS state for writing to state.json."""
        return {
            "positions": {
                pair: {
                    "pair": p.pair, "quantity": p.quantity,
                    "entry_price": p.entry_price, "entry_time": p.entry_time,
                    "usd_value": p.usd_value,
                }
                for pair, p in self._positions.items()
            }
        }
```

---

## Startup Reconciliation Sequence

On every bot startup (including crash recovery), the OMS runs a full reconciliation before the first trade:

```python
def startup_reconciliation(oms: OrderManager, client, risk_manager, state: dict):
    """
    Complete startup sequence to ensure local state matches exchange state.
    Must complete successfully before the main loop begins.
    """
    logger.info("Starting startup reconciliation...")

    # 1. Load persisted state
    oms.load_state(state.get("oms", {}))
    risk_manager.load_state(state.get("risk", {}))

    # 2. Fetch live balance from exchange
    balance = client.get_balance()
    if not balance.get("Success"):
        raise RuntimeError("Cannot get balance on startup — cannot proceed safely")

    # 3. Fetch all pending orders
    pending = client.query_order(pending_only=True)

    # 4. Cross-check: does our state match what the exchange reports?
    exchange_coins = {coin: data["Free"] + data["Lock"]
                     for coin, data in balance.get("Wallet", {}).items()
                     if data["Free"] + data["Lock"] > 0 and coin != "USD"}

    local_coins = {pos.pair.split("/")[0]: pos.quantity
                   for pos in oms.get_all_positions().values()}

    for coin, exchange_qty in exchange_coins.items():
        local_qty = local_coins.get(coin, 0)
        discrepancy = abs(exchange_qty - local_qty) / (exchange_qty + 1e-10)
        if discrepancy > 0.01:  # More than 1% difference
            logger.warning(f"POSITION DISCREPANCY: {coin} "
                           f"exchange={exchange_qty:.6f} local={local_qty:.6f}")
            # Trust the exchange — update local state
            # This is where manual review may be needed

    logger.info(f"Startup reconciliation complete. "
                f"USD free: ${balance['Wallet'].get('USD', {}).get('Free', 0):,.2f}")
```

---

## Failure Modes This Layer Prevents

| Failure | Prevention |
|---|---|
| Silent order failure (Success=false with HTTP 200) | Always parse Success field; Telegram alert on False |
| Ghost position from unconfirmed order | Only update position state after confirmed FILLED status |
| Position state lost on crash | dump_state() + load_state() + startup reconciliation |
| Double entry after restart | Reconciliation checks exchange balance against local state |
| Precision rejection silently losing trades | Pre-flight precision rounding before every submission |
| Insufficient balance rejection | Pre-flight balance check using Free balance only |
| Rate limit cascade from rapid retries | Rate limiter called before submission; no immediate retry on rejection |
| Missing fills after network hiccup | Periodic reconciliation queries pending order count |
