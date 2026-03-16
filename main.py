from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from bot.api.client import RoostooClient
from bot.execution.order_manager import OrderManager
from bot.execution.regime import RegimeDetector
from bot.execution.risk import RiskManager
from bot.monitoring.telegram import TelegramAlerter
from bot.persistence.state_manager import StateManager
from bot.strategy.mean_reversion import MeanReversionStrategy
from bot.strategy.momentum import MomentumStrategy


def _setup_logging() -> None:
    """Configure rotating file handler + stdout handler."""
    from logging.handlers import RotatingFileHandler

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = RotatingFileHandler(logs_dir / "bot.log", maxBytes=10 * 1024 * 1024, backupCount=10)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


def _load_config() -> dict:
    """Load config.yaml and merge with environment variables."""
    import yaml  # type: ignore[import]

    config_path = Path("bot/config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def startup_reconciliation(
    client: RoostooClient,
    risk_manager: RiskManager,
    order_manager: OrderManager,
    telegram: TelegramAlerter,
    state_manager: StateManager,
) -> None:
    """
    Reconcile persisted state against live exchange state on startup.

    Steps:
      1. Load RiskManager state from disk (trailing_stops, entry_prices, portfolio_hwm, circuit_breaker_active)
      2. Load OrderManager state from disk (positions, managed_orders)
      3. Fetch live balance from exchange
      4. Fetch live open orders from exchange
      5. Compare live positions vs persisted positions — WARN on discrepancy
      6. Compare live balance vs persisted portfolio_hwm — WARN if delta > 1%

    Discrepancies do NOT abort startup — the bot continues with live state
    taking precedence, but the operator is alerted via Telegram.
    """
    logger = logging.getLogger(__name__)
    logger.info("=== startup_reconciliation() begin ===")

    # Step 1 & 2: Load persisted state
    saved_state = state_manager.read()
    if saved_state:
        logger.info("Found persisted state (keys: %s)", list(saved_state.keys()))
        risk_state = saved_state.get("risk_manager", {})
        order_state = saved_state.get("order_manager", {})
        if risk_state:
            risk_manager.load_state(risk_state)
            logger.info(
                "RiskManager state loaded: hwm=%.2f cb_active=%s",
                risk_manager._portfolio_hwm,
                risk_manager._circuit_breaker_active,
            )
        if order_state:
            order_manager.load_state(order_state)
            logger.info(
                "OrderManager state loaded: %d positions",
                len(order_manager.get_all_positions()),
            )
    else:
        logger.info("No persisted state found — cold start")

    # Step 3 & 4: Fetch live exchange state
    try:
        live_balance = client.get_balance()
        live_orders = client.get_open_orders()
        logger.info("Live balance fetched: total_usd=%.2f", live_balance.get("total_usd", 0.0))
        logger.info("Live open orders: %d", len(live_orders))
    except Exception as exc:
        logger.error("Failed to fetch live state from exchange: %s", exc)
        telegram.send(f"\u26a0\ufe0f WARN: startup_reconciliation() could not fetch live state: {exc}")
        return

    # Step 5: Reconcile positions
    persisted_positions = order_manager.get_all_positions()
    live_order_pairs = {o.get("pair") or o.get("Symbol", "") for o in live_orders}

    discrepancies = []
    for pair, pos in persisted_positions.items():
        if pair not in live_order_pairs and pos.quantity > 0:
            discrepancies.append(
                f"Position {pair} qty={pos.quantity:.6f} in state but no open order on exchange"
            )

    for pair in live_order_pairs:
        if pair and pair not in persisted_positions:
            discrepancies.append(
                f"Open order for {pair} on exchange but no position in persisted state"
            )

    # Step 6: Reconcile portfolio value
    live_usd = live_balance.get("total_usd", 0.0)
    hwm = getattr(risk_manager, "_portfolio_hwm", live_usd)
    if hwm > 0 and abs(live_usd - hwm) / hwm > 0.01:
        discrepancies.append(
            f"Portfolio value mismatch: live={live_usd:.2f} USD vs persisted_hwm={hwm:.2f} USD "
            f"(delta {abs(live_usd - hwm) / hwm * 100:.1f}%)"
        )

    if discrepancies:
        msg = "\u26a0\ufe0f STARTUP RECONCILIATION DISCREPANCY:\n" + "\n".join(
            f"  \u2022 {d}" for d in discrepancies
        )
        logger.warning(msg)
        telegram.send(msg)
    else:
        logger.info("Reconciliation OK — persisted state matches exchange")

    logger.info("=== startup_reconciliation() complete ===")


def _register_shutdown_handler(
    state_manager: StateManager,
    risk_manager: RiskManager,
    order_manager: OrderManager,
    telegram: TelegramAlerter,
) -> None:
    """
    Register SIGTERM and SIGINT handlers that flush state before exit.

    systemd sends SIGTERM on service stop. Ctrl-C sends SIGINT.
    Both must write state to disk so the next startup can reconcile correctly.
    """
    logger = logging.getLogger(__name__)

    def _handle_shutdown(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — flushing state before exit", sig_name)
        try:
            state = {
                "risk_manager": risk_manager.dump_state(),
                "order_manager": order_manager.dump_state(),
                "shutdown_signal": sig_name,
                "shutdown_time": time.time(),
            }
            state_manager.write(state)
            logger.info("State flushed successfully")
        except Exception as exc:
            logger.error("Failed to flush state on shutdown: %s", exc)
        telegram.send(f"\U0001f6d1 Bot shutdown ({sig_name}) — state flushed")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    logger.info("Shutdown handler registered (SIGTERM + SIGINT)")


def main() -> None:
    """Entry point — startup sequence for the trading bot."""
    load_dotenv()
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Roostoo Trading Bot starting up")

    config = _load_config()

    # Initialise components
    api_key = os.environ["ROOSTOO_API_KEY"]
    secret = os.environ["ROOSTOO_SECRET"]
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")

    client = RoostooClient(api_key=api_key, secret=secret)
    telegram = TelegramAlerter(token=tg_token, chat_id=tg_chat)
    state_manager = StateManager(path=Path(config.get("state_path", "state.json")))
    risk_manager = RiskManager(config=config)
    order_manager = OrderManager(client=client, risk_manager=risk_manager, config=config)

    # Register shutdown handler BEFORE reconciliation (handles crashes during startup)
    _register_shutdown_handler(state_manager, risk_manager, order_manager, telegram)

    # Startup reconciliation
    startup_reconciliation(
        client=client,
        risk_manager=risk_manager,
        order_manager=order_manager,
        telegram=telegram,
        state_manager=state_manager,
    )

    telegram.send("\u2705 Bot started — reconciliation complete")
    logger.info("Startup complete — main loop not yet implemented (07-02)")
    # TODO(07-02): main loop goes here


if __name__ == "__main__":
    main()
