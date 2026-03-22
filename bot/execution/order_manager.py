import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Enum of order statuses throughout lifecycle."""
    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"


@dataclass
class ManagedOrder:
    """A single order tracked by OrderManager."""
    pair: str
    side: str
    quantity: float
    submitted_at: float
    order_id: Optional[int] = None
    status: OrderStatus = OrderStatus.PENDING_SUBMIT
    fill_price: Optional[float] = None
    filled_quantity: float = 0.0


@dataclass
class Position:
    """A single open position."""
    pair: str
    quantity: float
    entry_price: float
    current_value_usd: float = 0.0


class OrderManager:
    """
    Order Management System (OMS) for the trading bot.

    Responsibilities:
    - Submit orders to exchange (pre-flight validation)
    - Track order lifecycle (PENDING -> FILLED -> open position)
    - Reconcile positions against live exchange state (every 5 min)
    - Crash-safe state persistence (dump_state/load_state)
    - Position sizing & stop-loss management
    """

    def __init__(self, client, risk_manager, config: dict):
        """
        Initialize OrderManager.

        Args:
            client: RoostooClient instance (API client)
            risk_manager: RiskManager instance
            config: Configuration dict
        """
        self.client = client
        self.risk_manager = risk_manager
        self.config = config

        self._open_orders: Dict[int, ManagedOrder] = {}
        self._positions: Dict[str, Position] = {}
        self._last_reconcile_ts: float = 0.0

        # Quantity step sizes per pair (Roostoo exchange requirements)
        # Rounds DOWN to avoid exceeding available balance
        self._step_sizes: Dict[str, int] = {
            "BTC/USD": 3,    # 0.001
            "ETH/USD": 2,    # 0.01
            "BNB/USD": 2,    # 0.01
            "SOL/USD": 2,    # 0.01
            "XRP/USD": 0,    # 1
            "DOGE/USD": 0,   # 1
            "ADA/USD": 0,    # 1
            "AVAX/USD": 1,   # 0.1
            "LINK/USD": 1,   # 0.1
            "DOT/USD": 1,    # 0.1
            "LTC/USD": 2,    # 0.01
            "UNI/USD": 1,    # 0.1
            "NEAR/USD": 1,   # 0.1
            "SUI/USD": 1,    # 0.1
            "APT/USD": 1,    # 0.1
            "PEPE/USD": 0,   # 1 (priced in fractions of a cent)
            "ARB/USD": 0,    # 1
            "SHIB/USD": 0,   # 1
            "FIL/USD": 1,    # 0.1
            "HBAR/USD": 0,   # 1
            "AAVE/USD": 2,   # 0.01
            "CRV/USD": 0,    # 1
            "FET/USD": 0,    # 1
            "ZEC/USD": 2,    # 0.01
            "ZEN/USD": 1,    # 0.1
            "CAKE/USD": 1,   # 0.1
            "PAXG/USD": 3,   # 0.001
            "XLM/USD": 0,    # 1
            "TRX/USD": 0,    # 1
            "CFX/USD": 0,    # 1
            "ICP/USD": 1,    # 0.1
            "PENDLE/USD": 1, # 0.1
            "WLD/USD": 1,    # 0.1
            "SEI/USD": 0,    # 1
            "BONK/USD": 0,   # 1
            "WIF/USD": 0,    # 1
            "ENA/USD": 0,    # 1
            "TAO/USD": 2,    # 0.01
            "FLOKI/USD": 0,  # 1
        }
        self.RECONCILE_INTERVAL = 300  # 5 minutes

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        entry_price: float,
        initial_stop: float
    ) -> Optional[ManagedOrder]:
        """
        Submit a MARKET order to the exchange and track it.

        Args:
            pair: Trading pair (e.g., "BTC/USD")
            side: "BUY" or "SELL"
            quantity: Quantity in coin units (e.g., BTC, not USD)
            entry_price: Expected entry price (used if fill_price is missing)
            initial_stop: Initial stop loss price

        Returns:
            ManagedOrder if successful, None otherwise
        """
        try:
            # Round quantity DOWN to exchange step size (avoid exceeding balance)
            import math
            decimals = self._step_sizes.get(pair, 3)  # default 0.001
            factor = 10 ** decimals
            quantity = math.floor(quantity * factor) / factor

            # Validate quantity
            if quantity <= 0:
                logger.warning(f"place_order: quantity<=0 for {pair}, skipping")
                return None

            # Create managed order tracking object
            managed_order = ManagedOrder(
                order_id=None,
                pair=pair,
                side=side,
                quantity=quantity,
                submitted_at=time.time()
            )

            # Submit to exchange
            response = self.client.place_order(
                pair=pair,
                side=side,
                quantity=quantity
            )

            # Check if submission was successful
            if not response.get("Success", False):
                logger.error(f"place_order failed for {pair}: {response}")
                return None

            # Extract order ID (handle multiple response formats)
            order_id = response.get("OrderId") or response.get("order_id")
            # Some endpoints nest details inside OrderDetail
            order_detail = response.get("OrderDetail", {})
            if order_id is None and order_detail:
                order_id = order_detail.get("OrderID") or order_detail.get("OrderId")
            if order_id is None:
                logger.error(
                    f"place_order: response missing OrderId for {pair}: {response}"
                )
                return None
            try:
                order_id = int(order_id)
            except (TypeError, ValueError):
                logger.error(
                    f"place_order: invalid OrderId '{order_id}' for {pair}: {response}"
                )
                return None
            managed_order.order_id = order_id
            managed_order.status = OrderStatus.SUBMITTED

            # Extract fill price with explicit None check (CRITICAL)
            fill_price_raw = (
                response.get("FilledPrice")
                or response.get("fill_price")
                or order_detail.get("FilledAverPrice")
            )
            if fill_price_raw is None:
                logger.warning(
                    f"fill_price is None for order {order_id}, using entry_price={entry_price} as fallback"
                )
                managed_order.fill_price = entry_price
            else:
                managed_order.fill_price = float(fill_price_raw)

            # Update order tracking
            managed_order.filled_quantity = quantity
            managed_order.status = OrderStatus.FILLED

            # Update position tracking
            if pair in self._positions:
                self._positions[pair].quantity += quantity
                self._positions[pair].entry_price = managed_order.fill_price
            else:
                self._positions[pair] = Position(
                    pair=pair,
                    quantity=quantity,
                    entry_price=managed_order.fill_price
                )

            # Record with risk manager
            self.risk_manager.record_entry(pair, managed_order.fill_price, initial_stop)

            # Store in order tracking
            self._open_orders[order_id] = managed_order

            logger.info(
                f"Order filled: {side} {quantity:.6f} {pair} @ {managed_order.fill_price:.4f}"
            )

            return managed_order

        except Exception as e:
            logger.error(f"place_order exception: {e}")
            return None

    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get shallow copy of all open positions.

        Returns:
            dict[pair_str, Position]
        """
        return dict(self._positions)

    def close_position(
        self,
        pair: str,
        quantity: float,
        current_price: float
    ) -> Optional[ManagedOrder]:
        """
        Close (sell) an open position.

        Args:
            pair: Trading pair
            quantity: Quantity to close
            current_price: Current market price

        Returns:
            ManagedOrder if successful, None otherwise
        """
        result = self.place_order(
            pair=pair,
            side="SELL",
            quantity=quantity,
            entry_price=current_price,
            initial_stop=0.0
        )

        if result:
            if pair in self._positions:
                del self._positions[pair]
            self.risk_manager.record_exit(pair)

        return result

    def _resync_from_exchange(self, force: bool = False) -> None:
        """
        Periodically reconcile positions against live exchange balances.

        Corrects local position state if exchange has different balance.

        Args:
            force: Skip interval check, force immediate reconciliation
        """
        # Check reconciliation interval
        if not force and (time.time() - self._last_reconcile_ts) < self.RECONCILE_INTERVAL:
            return

        self._last_reconcile_ts = time.time()

        try:
            # Fetch balances from exchange
            balance_resp = self.client.get_balance()

            if not balance_resp.get("Success", False):
                logger.warning("Reconcile: balance fetch failed")
                return

            balances = balance_resp.get("Data", {})  # dict of {asset: {Free, Locked, Total}}

            # Check each tracked position
            for pair in list(self._positions.keys()):
                # Extract asset from pair (e.g., "BTC" from "BTC/USD")
                asset = pair.split("/")[0]
                exchange_qty_str = balances.get(asset, {}).get("Total", "0")
                exchange_qty = float(exchange_qty_str)
                local_qty = self._positions[pair].quantity

                # Detect meaningful discrepancy
                if abs(exchange_qty - local_qty) > 1e-8:
                    logger.warning(
                        f"Reconcile discrepancy for {pair}: "
                        f"local={local_qty:.8f} exchange={exchange_qty:.8f}"
                    )

                    # CRITICAL: write back to self._positions
                    if exchange_qty > 1e-8:
                        self._positions[pair].quantity = exchange_qty
                        logger.info(
                            f"Reconcile: updated {pair} position to {exchange_qty:.8f}"
                        )
                    else:
                        del self._positions[pair]
                        self.risk_manager.record_exit(pair)
                        logger.info(
                            f"Reconcile: closed {pair} (exchange shows zero balance)"
                        )

            logger.debug(f"Reconcile complete: {len(self._positions)} positions")

        except Exception as e:
            logger.error(f"_resync_from_exchange exception: {e}")

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.cancel_order(order_id=order_id)

            if response.get("Success", False):
                logger.info(f"cancel_order {order_id}: success")
                if order_id in self._open_orders:
                    self._open_orders[order_id].status = OrderStatus.CANCELED
                return True
            else:
                logger.warning(f"cancel_order {order_id} failed: {response}")
                return False

        except Exception as e:
            logger.error(f"cancel_order exception: {e}")
            return False

    def dump_state(self) -> dict:
        """
        Serialize OrderManager state for crash recovery.

        Returns:
            dict with "open_orders" and "positions" keys
        """
        return {
            "open_orders": {
                str(oid): {
                    "order_id": o.order_id,
                    "pair": o.pair,
                    "side": o.side,
                    "quantity": o.quantity,
                    "submitted_at": o.submitted_at,
                    "status": o.status.name,
                    "fill_price": o.fill_price,
                    "filled_quantity": o.filled_quantity,
                }
                for oid, o in self._open_orders.items()
            },
            "positions": {
                pair: {
                    "pair": p.pair,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_value_usd": p.current_value_usd,
                }
                for pair, p in self._positions.items()
            },
        }

    def load_state(self, state: dict) -> None:
        """
        Restore OrderManager state from dump.

        Args:
            state: dict from dump_state()
        """
        # Restore open orders
        self._open_orders = {}
        for oid_str, order_data in state.get("open_orders", {}).items():
            oid = int(oid_str)
            managed_order = ManagedOrder(
                order_id=order_data["order_id"],
                pair=order_data["pair"],
                side=order_data["side"],
                quantity=order_data["quantity"],
                submitted_at=order_data["submitted_at"],
                status=OrderStatus[order_data["status"]],
                fill_price=order_data["fill_price"],
                filled_quantity=order_data["filled_quantity"],
            )
            self._open_orders[oid] = managed_order

        # Restore positions
        self._positions = {}
        for pair, pos_data in state.get("positions", {}).items():
            position = Position(
                pair=pos_data["pair"],
                quantity=pos_data["quantity"],
                entry_price=pos_data["entry_price"],
                current_value_usd=pos_data["current_value_usd"],
            )
            self._positions[pair] = position

        logger.info(
            f"OrderManager state loaded: "
            f"{len(self._open_orders)} orders, {len(self._positions)} positions"
        )

    def maybe_reconcile(self) -> None:
        """Convenience method for main loop to trigger periodic reconciliation."""
        self._resync_from_exchange(force=False)
