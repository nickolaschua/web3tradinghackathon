# Layer 10 — Orchestration (main.py)

## What This Layer Does

The Orchestration layer is the main loop that sequences all other layers into a coherent, repeating cycle. It initialises every component on startup, runs the startup reconciliation, then executes a defined sequence of operations every 60 seconds until the process is stopped.

It is intentionally thin. `main.py` contains no strategy logic, no risk calculations, no signing code, and no indicator computation. It only calls other layers in the correct order, handles the top-level exception boundary, manages the timing of the loop, and coordinates shutdown.

**This is the only entry point for the live bot.** Everything that runs on EC2 starts here.

---

## What This Layer Is Trying to Achieve

1. Sequence all layers in the correct dependency order every cycle
2. Ensure no unhandled exception ever terminates the bot — the catch-all handler is the last safety net
3. Manage timing so the loop runs at the right cadence without accumulating drift
4. Provide a clean shutdown path that closes positions if configured to do so

---

## How It Contributes to the Bigger Picture

Every other layer does one well-defined job. This layer knows about all of them and calls them in the right sequence. The complexity of the system — data fetching, feature computation, regime detection, signal generation, risk checks, order submission, state writing — is managed here by sequencing, not by coupling.

The single most important design choice in this layer is the catch-all exception handler that wraps the entire loop body. Any bug, any unexpected API response, any transient network error — if it reaches this handler, the bot logs it, sends a Telegram alert, sleeps for 60 seconds, and continues. The bot should never die from a runtime error.

---

## Files in This Layer

```
main.py         The only file in this layer
config.yaml     Read on startup; watched for changes during competition
```

---

## Complete `main.py`

```python
#!/usr/bin/env python3
"""
Roostoo Trading Bot — Main Entry Point
"""

import os
import sys
import time
import signal
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv

# ── Local imports ──────────────────────────────────────────────────────────
from api.client import RoostooClient
from api.rate_limiter import TradingRateLimiter
from api.exchange_info import ExchangeInfoCache
from data.live_fetcher import LiveFetcher
from data.features import compute_features, compute_cross_asset_features
from execution.regime import RegimeDetector, MarketRegime
from execution.risk import RiskManager
from execution.order_manager import OrderManager
from persistence.state import StateManager
from monitoring.logger import setup_logging, TradeLogger
from monitoring.telegram import TelegramAlerter
from monitoring.healthcheck import HealthChecker
from strategy.momentum import MultiTimeframeMomentumStrategy
from strategy.mean_reversion import BollingerMeanReversionStrategy
import pandas as pd

# ── Startup ────────────────────────────────────────────────────────────────

load_dotenv()
setup_logging()
logger = logging.getLogger("main")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_components(config: dict):
    """Instantiate all components. Called once at startup."""
    telegram = TelegramAlerter(
        token=os.environ["TELEGRAM_BOT_TOKEN"],
        chat_id=os.environ["TELEGRAM_CHAT_ID"],
    )
    client = RoostooClient(
        api_key=os.environ["ROOSTOO_API_KEY"],
        secret_key=os.environ["ROOSTOO_SECRET_KEY"],
    )
    rate_limiter = TradingRateLimiter()
    exchange_info = ExchangeInfoCache(client)
    risk_manager = RiskManager(config["risk"])
    oms = OrderManager(client, exchange_info, rate_limiter,
                       risk_manager, telegram, config)
    state_manager = StateManager()
    trade_logger = TradeLogger()
    health_checker = HealthChecker(telegram, oms, client)
    regime_detector = RegimeDetector()
    momentum_strategy = MultiTimeframeMomentumStrategy(config["momentum"])
    mr_strategy = BollingerMeanReversionStrategy(config["mean_reversion"])

    return {
        "telegram": telegram, "client": client, "rate_limiter": rate_limiter,
        "exchange_info": exchange_info, "risk": risk_manager, "oms": oms,
        "state": state_manager, "trade_logger": trade_logger,
        "health": health_checker, "regime": regime_detector,
        "momentum": momentum_strategy, "mr": mr_strategy,
    }


def startup_sequence(c: dict, config: dict) -> bool:
    """
    Full startup: load state, reconcile with exchange, seed data buffer.
    Returns True if safe to start trading.
    """
    logger.info("=" * 60)
    logger.info("TRADING BOT STARTING UP")
    logger.info("=" * 60)

    # Sync clock before anything else
    c["client"].sync_if_stale(max_age_seconds=0)

    # Load persisted state
    persisted = c["state"].read()
    c["oms"].load_state(persisted.get("oms", {}))
    c["risk"].load_state(persisted.get("risk", {}))

    # Live balance check
    balance = c["client"].get_balance()
    if not balance.get("Success"):
        logger.critical("Cannot get balance on startup — aborting")
        c["telegram"].send("🚨 STARTUP FAILED: Cannot get balance", level="CRITICAL")
        return False

    free_usd = balance["Wallet"].get("USD", {}).get("Free", 0)
    logger.info(f"Free USD on startup: ${free_usd:,.2f}")

    # Initialize risk high-water mark
    position_values = sum(
        c["oms"].get_position_value(pair)
        for pair in c["oms"].get_all_positions()
    )
    total_portfolio = free_usd + position_values
    c["risk"].initialize_hwm(
        persisted.get("risk", {}).get("portfolio_hwm", total_portfolio)
    )

    # Reconcile OMS against exchange
    c["oms"].reconcile(force=True)

    # Seed live data buffer from historical Parquet
    pairs = config["pairs"]["primary"]
    seed_dfs = {}
    for pair_roostoo in pairs:
        # BTC/USD → BTCUSDT
        binance_symbol = pair_roostoo.replace("/", "").replace("USD", "USDT")
        parquet_path = Path(f"data/parquet/{binance_symbol}_4h.parquet")
        if parquet_path.exists():
            seed_dfs[pair_roostoo] = pd.read_parquet(parquet_path).tail(500)

    c["fetcher"] = LiveFetcher(
        client=c["client"],
        pairs=pairs,
        primary_interval="4h",
        seed_dfs=seed_dfs,
    )

    # Verify all buffers are warmed up
    for pair in pairs:
        if not c["fetcher"].is_warmed_up(pair):
            logger.warning(f"Buffer not warmed up for {pair} — first signals may be delayed")

    c["telegram"].send(
        f"🤖 Bot started\n"
        f"Free USD: ${free_usd:,.2f}\n"
        f"Positions: {list(c['oms'].get_all_positions().keys()) or 'none'}\n"
        f"Pairs: {pairs}"
    )
    logger.info("Startup complete")
    return True


# ── Signal selection ───────────────────────────────────────────────────────

def select_strategy(regime_state, momentum_strategy, mr_strategy):
    if regime_state.regime == MarketRegime.BULL_TREND:
        return momentum_strategy
    elif regime_state.regime == MarketRegime.SIDEWAYS:
        return mr_strategy
    return None  # BEAR_TREND → cash


# ── Main loop ──────────────────────────────────────────────────────────────

def run_main_loop(c: dict, config: dict):
    """
    The central trading loop. Runs every 61 seconds.
    
    Sequence per cycle:
    1.  Sync time if stale
    2.  Poll ticker → update live buffer
    3.  Health checks (memory, disk, heartbeat)
    4.  If new candle closed → recompute features + regime
    5.  Check all open position stops (EVERY cycle, not just on new candle)
    6.  Reconcile OMS if interval elapsed
    7.  If new candle AND active strategy → generate signal
    8.  Risk check → OMS pre-flight → place order
    9.  Write state to disk
    10. Sleep (61s - time_used + jitter)
    """
    pairs = config["pairs"]["primary"]
    btc_pair = "BTC/USD"

    # Per-cycle state
    feature_cache: dict[str, pd.DataFrame] = {}
    current_regime_state = None
    active_strategy = None
    last_regime_log = ""

    logger.info("Main loop starting")

    while True:
        cycle_start = time.time()

        try:
            # ── 1. Time sync ───────────────────────────────────────────────
            c["client"].sync_if_stale(max_age_seconds=300)

            # ── 2. Poll ticker ─────────────────────────────────────────────
            dirty = c["fetcher"].poll_and_update()

            # ── 3. Health checks ───────────────────────────────────────────
            c["health"].check_all()

            # ── 4. Recompute features on new candle ────────────────────────
            if any(dirty.values()):
                # Compute BTC features first (needed for cross-asset features)
                btc_df = c["fetcher"].get_dataframe(btc_pair)
                if len(btc_df) > 200:
                    btc_features = compute_features(btc_df)
                    feature_cache[btc_pair] = btc_features

                    # Compute daily BTC features for regime detection
                    # (Use 6× 4H bars as proxy for daily, or load daily separately)
                    current_regime_state = c["regime"].detect(btc_features)

                    # Log regime change
                    regime_str = current_regime_state.regime.value
                    if regime_str != last_regime_log:
                        c["telegram"].send_regime_change(
                            last_regime_log, regime_str,
                            current_regime_state.ema20,
                            current_regime_state.ema50,
                            current_regime_state.adx,
                        )
                        last_regime_log = regime_str

                    active_strategy = select_strategy(
                        current_regime_state, c["momentum"], c["mr"]
                    )

                    # Compute features for all pairs
                    for pair in pairs:
                        if pair == btc_pair:
                            feature_cache[pair] = btc_features
                            continue
                        pair_df = c["fetcher"].get_dataframe(pair)
                        if len(pair_df) > 200:
                            pair_features = compute_features(pair_df)
                            pair_features = compute_cross_asset_features(
                                btc_df, pair_features
                            )
                            feature_cache[pair] = pair_features

            # ── 5. Check stops for ALL open positions ──────────────────────
            # This runs EVERY cycle, not just on new candle
            for pair, position in list(c["oms"].get_all_positions().items()):
                features = feature_cache.get(pair)
                if features is None or len(features) == 0:
                    continue

                current_price = c["fetcher"].get_latest_price(pair)
                current_atr = float(features.iloc[-1].get("atr_14", float("nan")))

                stop_check = c["risk"].check_stops(pair, current_price, current_atr)
                if stop_check.should_exit:
                    logger.warning(f"Stop triggered for {pair}: {stop_check.exit_reason}")
                    c["oms"].submit_sell(pair, reason=stop_check.exit_reason)

            # ── 6. Periodic OMS reconciliation ────────────────────────────
            c["oms"].reconcile()

            # ── 7. Generate signals on new candle ─────────────────────────
            if any(dirty.values()) and active_strategy is not None:

                # Circuit breaker check
                balance = c["client"].get_balance()
                if balance.get("Success"):
                    free_usd = balance["Wallet"].get("USD", {}).get("Free", 0)
                    position_usd = sum(
                        c["oms"].get_position_value(p) for p in c["oms"].get_all_positions()
                    )
                    total_portfolio = free_usd + position_usd
                    if c["risk"].check_circuit_breaker(total_portfolio):
                        logger.warning("Circuit breaker active — skipping signal generation")
                    else:
                        # Generate signals for each pair
                        for pair in pairs:
                            features = feature_cache.get(pair)
                            if features is None or len(features) == 0:
                                continue
                            if not dirty.get(pair, False):
                                continue

                            current_price = c["fetcher"].get_latest_price(pair)
                            current_position = c["oms"].get_position_value(pair)

                            signal = active_strategy.generate_signal(
                                features, current_price, current_position
                            )
                            logger.info(f"Signal for {pair}: {signal.action.value} | {signal.reason}")

                            # ── 8. Risk check + submit ─────────────────────
                            if signal.action.value == "BUY" and current_position == 0:
                                current_atr = float(features.iloc[-1].get("atr_14", 0))
                                open_positions_usd = {
                                    p: c["oms"].get_position_value(p)
                                    for p in c["oms"].get_all_positions()
                                }
                                sizing = c["risk"].size_new_position(
                                    pair=pair,
                                    current_price=current_price,
                                    current_atr=current_atr,
                                    free_balance_usd=free_usd,
                                    open_positions=open_positions_usd,
                                    regime_multiplier=current_regime_state.size_multiplier,
                                    signal_size_pct=signal.suggested_size_pct,
                                )
                                if sizing.decision.value == "APPROVED":
                                    c["oms"].submit_buy(
                                        pair=pair,
                                        quantity=sizing.approved_quantity,
                                        initial_stop=sizing.trailing_stop_price,
                                    )
                                else:
                                    logger.info(f"Risk blocked BUY {pair}: {sizing.reason}")

                            elif signal.action.value == "SELL" and current_position > 0:
                                c["oms"].submit_sell(pair, reason=signal.reason)

            # ── 9. Write state ─────────────────────────────────────────────
            c["state"].write(
                oms_state=c["oms"].dump_state(),
                risk_state=c["risk"].dump_state(),
                regime_state={
                    "current_regime": last_regime_log,
                    "size_multiplier": current_regime_state.size_multiplier if current_regime_state else 0,
                    "last_detected_at": time.time(),
                },
                strategy_state={
                    "active_strategy": active_strategy.__class__.__name__ if active_strategy else "none",
                    "last_candle_boundary": c["fetcher"].get_candle_boundaries(),
                },
                execution_state={
                    "last_trade_time": c["rate_limiter"]._last_trade_time,
                    "last_reconciliation_time": c["oms"]._last_reconciliation,
                },
            )

        except Exception as e:
            # ── CATCH-ALL: nothing should ever reach the top level ─────────
            logger.exception(f"Unhandled exception in main loop: {e}")
            c["telegram"].send_error("Main loop exception", str(e))
            # Sleep and continue — do not exit
            time.sleep(60)
            continue

        # ── 10. Sleep until next cycle ─────────────────────────────────────
        cycle_duration = time.time() - cycle_start
        sleep_time = max(1.0, 61.0 - cycle_duration)  # Target 61-second cadence
        logger.debug(f"Cycle took {cycle_duration:.2f}s. Sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)


# ── Shutdown handler ───────────────────────────────────────────────────────

def setup_shutdown_handler(c: dict):
    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received")
        c["telegram"].send("🛑 Bot shutting down gracefully")
        # Write final state
        # (Components may already be partially torn down — best effort)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config()
    components = build_components(config)
    setup_shutdown_handler(components)

    if not startup_sequence(components, config):
        logger.critical("Startup failed — exiting")
        sys.exit(1)

    run_main_loop(components, config)
```

---

## Loop Timing

The loop targets a 61-second cadence. The extra second beyond the 60-second minimum provides buffer against clock imprecision. Actual sleep time is `61 - cycle_duration` so that if a cycle takes 3 seconds (e.g., due to an API call being slow), the next cycle still starts approximately 61 seconds after the previous one started.

If a cycle takes longer than 61 seconds (e.g., waiting on rate limiter), the next cycle starts immediately with no sleep. The rate limiter itself enforces the trading cadence independently.

---

## Configuration Hot-Swap

During the competition you can modify strategy parameters without redeploying code:

```bash
# Edit config.yaml on EC2
nano config.yaml

# Restart the service (takes ~5 seconds)
sudo systemctl restart tradingbot

# Verify it started correctly
journalctl -u tradingbot -n 50 -f
```

The startup sequence runs again after every restart, including reconciliation against the exchange. This is safe to do even with open positions — the reconciliation will restore the correct state.

---

## What NOT to Put in main.py

- Strategy logic (goes in `strategy/`)
- Risk calculations (goes in `execution/risk.py`)
- Indicator computation (goes in `data/features.py`)
- HTTP requests (goes in `api/client.py`)
- Logging configuration (goes in `monitoring/logger.py`)

The rule: if you find yourself writing domain logic in `main.py`, it belongs in a dedicated module. `main.py` is a coordinator, not an implementer.

---

## Failure Modes This Layer Prevents

| Failure | Prevention |
|---|---|
| Any unhandled exception killing the bot | Catch-all in loop body: log, alert, sleep, continue |
| Stop-loss missed because no new signal | Stop check runs every cycle independently of signal generation |
| Time sync drift causing auth failures | `sync_if_stale()` called at the top of every cycle |
| State not written after a trade | State write at end of every cycle, unconditionally |
| Startup with stale or inconsistent state | Full startup reconciliation before first trade cycle |
| Config changes requiring redeployment | config.yaml loaded fresh on every restart |
| Process dies and doesn't restart | systemd `Restart=always` handles this outside main.py |
