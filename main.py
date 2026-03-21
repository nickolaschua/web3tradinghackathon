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
from bot.data.live_fetcher import LiveFetcher
from bot.execution.order_manager import OrderManager
from bot.execution.portfolio import PortfolioAllocator
from bot.execution.regime import RegimeDetector
from bot.execution.risk import RiskManager
from bot.monitoring.telegram import TelegramAlerter
from bot.persistence.state_manager import StateManager
from bot.strategy.base import SignalDirection
from bot.config.unlock_screen import apply_unlock_screen
from bot.strategy.mean_reversion import MeanReversionStrategy
from bot.strategy.pairs_trading import PairsTradingStrategy
from bot.strategy.xgboost_strategy import XGBoostStrategy

_CANDLE_15M_SECONDS = 15 * 60  # seconds per 15M candle period

_BINANCE_TO_ROOSTOO: dict[str, str] = {
    "BTCUSDT": "BTC/USD",
    "ETHUSDT": "ETH/USD",
    "SOLUSDT": "SOL/USD",
}


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


def _run_one_cycle(
    *,
    client: RoostooClient,
    live_fetcher: LiveFetcher,
    risk_manager: RiskManager,
    order_manager: OrderManager,
    portfolio_allocator: PortfolioAllocator,
    telegram: TelegramAlerter,
    state_manager: StateManager,
    strategy: Any,
    mean_reversion_strategy: MeanReversionStrategy,
    pairs_strategy: PairsTradingStrategy,
    regime_detector: RegimeDetector,
    tradeable_pairs: list[str],
    feature_pairs: list[str],
    loop_state: dict,
    config: dict,
) -> None:
    """
    Execute one iteration of the 7-step main loop (PROJECT.md Issue 08).

    Steps in mandatory order:
      1. Poll ticker for all feature_pairs → update live_fetcher buffers
      2. Fetch balance + check circuit breaker
      3. Check stops for each open position (ATR from features_cache)
      4. If new 4H candle epoch: compute features → signal → size → submit
      5. Write state.json atomically
      6. Heartbeat log
      7. Boundary-aligned sleep to next 60s boundary

    Args:
        loop_state: Mutable dict persisted across cycles.
                    Keys: "last_signal_epoch" (int), "features_cache" (dict[str, dict])
    """
    logger = logging.getLogger(__name__)
    warmup_bars = config.get("warmup_bars", 35)

    # ── Step 1: Poll ticker for all pairs ────────────────────────────────────
    prices: dict[str, float] = {}
    for pair in feature_pairs:
        try:
            ticker = client.get_ticker(pair)
            price = float(ticker.get("Data", {}).get(pair, {}).get("LastPrice", 0.0))
            if price > 0.0:
                live_fetcher.poll_ticker(pair, price)
                prices[pair] = price
        except Exception as exc:
            logger.warning("Step 1: poll_ticker failed for %s: %s", pair, exc)

    # ── Step 2: Balance + circuit breaker check ───────────────────────────────
    total_usd = 0.0
    cb_active = False
    try:
        balance_resp = client.get_balance()
        total_usd = float(balance_resp.get("total_usd", 0.0))
        cb_active = risk_manager.check_circuit_breaker(total_usd)
        if cb_active:
            logger.warning("Step 2: circuit breaker active (total_usd=%.2f)", total_usd)
    except Exception as exc:
        logger.error("Step 2: balance check failed: %s", exc)

    # ── Step 3: Check stops per open position ────────────────────────────────
    features_cache: dict[str, dict] = loop_state.setdefault("features_cache", {})
    positions = order_manager.get_all_positions()

    for pair, pos in list(positions.items()):
        current_price = prices.get(pair)
        if current_price is None or current_price <= 0.0:
            continue
        # Use ATR from last cycle's feature computation; fallback = 2% of price
        cached_atr = features_cache.get(pair, {}).get("atr_proxy", current_price * 0.02)
        try:
            stop_result = risk_manager.check_stops(pair, current_price, cached_atr)
            if stop_result.should_exit:
                logger.info(
                    "Step 3: stop triggered for %s: %s (%s)",
                    pair, stop_result.exit_reason, stop_result.exit_type,
                )
                order_manager.close_position(pair, pos.quantity, current_price)
                telegram.send(
                    f"\U0001f534 Stop triggered: {pair} | {stop_result.exit_type} | "
                    f"price={current_price:.4f}"
                )
        except Exception as exc:
            logger.error("Step 3: stop check failed for %s: %s", pair, exc)

    # ── Step 4: New 15M candle — features + signal + size + submit ───────────
    now = time.time()
    current_15m_epoch = int(now) // _CANDLE_15M_SECONDS
    last_signal_epoch = loop_state.get("last_signal_epoch", 0)

    if current_15m_epoch > last_signal_epoch:
        loop_state["last_signal_epoch"] = current_15m_epoch
        logger.info("Step 4: new 15M candle epoch %d — computing signals", current_15m_epoch)

        # Extend feature buffers with the just-completed 15M candle.
        # MUST run before the tradeable_pairs loop so get_feature_matrix() sees
        # the latest bar for cross-asset lag features (eth_return_lag1 etc.).
        epoch_ts = current_15m_epoch * _CANDLE_15M_SECONDS
        for fpair in feature_pairs:
            if fpair in prices:
                live_fetcher.append_epoch_candle(fpair, prices[fpair], epoch_ts)
                logger.debug("Step 4: appended epoch candle for %s at epoch %d", fpair, epoch_ts)
            else:
                logger.warning("Step 4: no live price for %s — epoch candle skipped", fpair)

        # Recompute portfolio weights once per 4H boundary using full price history
        feature_price_history = {
            fpair: live_fetcher._to_dataframe(fpair)
            for fpair in feature_pairs
        }
        portfolio_allocator.compute_weights(feature_price_history)

        # ── Phase A: collect per-pair signals (momentum → mean-reversion fallback) ──
        from bot.strategy.base import TradingSignal
        signals: dict[str, TradingSignal] = {}
        regime_mults: dict[str, float] = {}

        for pair in tradeable_pairs:
            try:
                df = live_fetcher._to_dataframe(pair)

                if len(df) < warmup_bars:
                    logger.info(
                        "Step 4A: %s has %d bars (need %d) — skipping (warmup)",
                        pair, len(df), warmup_bars,
                    )
                    continue

                features_df = live_fetcher.get_feature_matrix(pair)
                if features_df.empty:
                    logger.warning("Step 4A: get_feature_matrix empty for %s", pair)
                    continue

                # Update cache — ATR available for step 3 in next cycle
                features_cache[pair] = features_df.iloc[-1].to_dict()

                # Regime detection
                regime = regime_detector.update(df)
                regime_mults[pair] = regime.size_multiplier
                logger.debug(
                    "Step 4A: regime=%s (mult=%.1f) for %s",
                    regime.name, regime.size_multiplier, pair,
                )

                # Primary: momentum strategy
                signal = strategy.generate_signal(pair, features_df)

                # Fallback: mean reversion when momentum says HOLD
                if signal.direction == SignalDirection.HOLD:
                    signal = mean_reversion_strategy.generate_signal(pair, features_df)

                signals[pair] = signal

            except Exception as exc:
                logger.error(
                    "Step 4A: error processing %s: %s", pair, exc, exc_info=True
                )

        # ── Phase B: pairs trading signals ───────────────────────────────────
        bar_index: int = loop_state.get("bar_index", 0)
        loop_state["bar_index"] = bar_index + 1

        coin_dfs = {fpair: live_fetcher._to_dataframe(fpair) for fpair in feature_pairs}
        for pair_state in pairs_strategy.pair_states:
            try:
                pt_signals = pairs_strategy.update(pair_state, coin_dfs, bar_index)
                for pt_sig in pt_signals:
                    existing = signals.get(pt_sig.pair)
                    if existing is None or existing.direction == SignalDirection.HOLD:
                        signals[pt_sig.pair] = pt_sig
                        logger.info(
                            "Step 4B: pairs signal %s for %s",
                            pt_sig.direction.name, pt_sig.pair,
                        )
            except Exception as exc:
                logger.error(
                    "Step 4B: pairs error %s/%s: %s",
                    pair_state.pair_a, pair_state.pair_b, exc, exc_info=True,
                )

        # ── Phase C: token unlock screen ─────────────────────────────────────
        signals = apply_unlock_screen(signals)

        # ── Phase D: execute signals ──────────────────────────────────────────
        for pair, signal in signals.items():
            if signal.direction == SignalDirection.HOLD:
                logger.debug("Step 4D: HOLD signal for %s", pair)
                continue

            try:
                current_price = prices.get(pair, 0.0)
                if current_price <= 0.0:
                    logger.warning("Step 4D: no price for %s, skipping", pair)
                    continue

                regime_mult = regime_mults.get(pair, 1.0)

                if signal.direction == SignalDirection.BUY:
                    if cb_active:
                        logger.info("Step 4D: CB active, skipping BUY for %s", pair)
                        continue

                    if pair in order_manager.get_all_positions():
                        logger.debug(
                            "Step 4D: already in position for %s, skipping BUY", pair
                        )
                        continue

                    atr = features_cache.get(pair, {}).get("atr_proxy", current_price * 0.02)
                    confidence = signal.confidence if signal.confidence > 0.0 else 0.5
                    portfolio_weight = portfolio_allocator.get_pair_weight(
                        pair, n_active_pairs=len(feature_pairs)
                    )

                    open_pos = order_manager.get_all_positions()
                    open_pos_usd: dict[str, float] = {
                        p: pos_obj.quantity * prices.get(p, current_price)
                        for p, pos_obj in open_pos.items()
                    }
                    free_usd = max(0.0, total_usd - sum(open_pos_usd.values()))

                    sizing = risk_manager.size_new_position(
                        pair=pair,
                        current_price=current_price,
                        current_atr=atr,
                        free_balance_usd=free_usd,
                        open_positions=open_pos_usd,
                        regime_multiplier=regime_mult,
                        confidence=confidence,
                        portfolio_weight=portfolio_weight,
                    )

                    if sizing.decision.name == "APPROVED":
                        managed = order_manager.place_order(
                            pair=pair,
                            side="BUY",
                            quantity=sizing.approved_quantity,
                            entry_price=current_price,
                            initial_stop=sizing.stop_price,
                        )
                        if managed:
                            telegram.send(
                                f"\U0001f7e2 BUY {pair}: qty={sizing.approved_quantity:.6f} "
                                f"@ ~{current_price:.4f} | stop={sizing.stop_price:.4f}"
                            )
                    else:
                        logger.info(
                            "Step 4D: BUY blocked for %s: %s (%s)",
                            pair, sizing.decision.name, sizing.reason,
                        )

                elif signal.direction == SignalDirection.SELL:
                    pos = order_manager.get_all_positions().get(pair)
                    if pos is None:
                        logger.debug(
                            "Step 4D: SELL signal for %s but no open position", pair
                        )
                        continue
                    managed = order_manager.close_position(pair, pos.quantity, current_price)
                    if managed:
                        telegram.send(
                            f"\U0001f534 SELL {pair}: qty={pos.quantity:.6f} @ ~{current_price:.4f}"
                        )

            except Exception as exc:
                logger.error(
                    "Step 4D: error executing %s: %s", pair, exc, exc_info=True
                )

    # Periodic reconciliation (every 5 min — OrderManager tracks interval internally)
    order_manager.maybe_reconcile()

    # ── Step 5: Write state.json ──────────────────────────────────────────────
    try:
        state = {
            "risk_manager": risk_manager.dump_state(),
            "order_manager": order_manager.dump_state(),
            "last_signal_epoch": loop_state.get("last_signal_epoch", 0),
            "timestamp": time.time(),
        }
        state_manager.write(state)
    except Exception as exc:
        logger.error("Step 5: state write failed: %s", exc)

    # ── Step 6: Heartbeat log ─────────────────────────────────────────────────
    logger.info(
        "Heartbeat: positions=%d cb_active=%s 15m_epoch=%d prices_polled=%d",
        len(order_manager.get_all_positions()),
        cb_active,
        current_15m_epoch,
        len(prices),
    )

    # ── Step 7: Boundary-aligned sleep ───────────────────────────────────────
    sleep_secs = max(0.0, 60.0 - (time.time() % 60.0))
    logger.debug("Step 7: sleeping %.1fs to next 60s boundary", sleep_secs)
    time.sleep(sleep_secs)


def _load_seed_data(config: dict) -> dict[str, Any]:
    """
    Load historical Parquet seed data for the configured feature pairs.

    Tries multiple filename conventions for each pair. Missing files are silently
    skipped — the bot starts with an empty buffer and warms up from live ticks.

    Filename patterns tried (in order):
      1. {data_dir}/{BINANCE_SYM}_15m.parquet          (e.g. BTCUSDT_15m.parquet)
      2. {data_dir}/{BINANCE_SYM}-15m.parquet           (e.g. BTCUSDT-15m.parquet)
      3. {data_dir}/{BINANCE_SYM.lower()}_15m.parquet
      4. {data_dir}/{BINANCE_SYM}-15m-*.parquet (glob)  (from binance_historical_data)

    Returns:
        dict[str, pd.DataFrame] with Roostoo pair names (e.g. "BTC/USD") as keys.
        Pairs with no data are omitted — LiveFetcher handles missing pairs gracefully.
    """
    import glob as glob_mod

    import pandas as pd

    logger = logging.getLogger(__name__)
    data_dir = Path(config.get("data_dir", "data"))

    roostoo_to_binance = {v: k for k, v in _BINANCE_TO_ROOSTOO.items()}
    feature_pairs: list[str] = config.get("feature_pairs", ["BTC/USD"])

    seed_dfs: dict[str, pd.DataFrame] = {}

    for pair in feature_pairs:
        binance_sym = roostoo_to_binance.get(pair)
        if not binance_sym:
            logger.warning("_load_seed_data: no Binance symbol for pair %s, skipping", pair)
            continue

        candidates = [
            data_dir / f"{binance_sym}_15m.parquet",
            data_dir / f"{binance_sym}-15m.parquet",
            data_dir / f"{binance_sym.lower()}_15m.parquet",
        ]

        loaded = False
        for path in candidates:
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    logger.info("Seed data for %s: %d rows from %s", pair, len(df), path)
                    seed_dfs[pair] = df
                    loaded = True
                    break
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", path, exc)

        if not loaded:
            # Try glob for date-suffixed files: BTCUSDT-15m-2024-01-01.parquet etc.
            pattern = str(data_dir / f"{binance_sym}-15m-*.parquet")
            matches = sorted(glob_mod.glob(pattern))
            if matches:
                try:
                    dfs = [pd.read_parquet(m) for m in matches]
                    df = pd.concat(dfs).sort_index()
                    logger.info(
                        "Seed data for %s: %d rows from %d glob files",
                        pair, len(df), len(dfs),
                    )
                    seed_dfs[pair] = df
                    loaded = True
                except Exception as exc:
                    logger.warning("Glob seed load failed for %s: %s", pair, exc)

        if not loaded:
            logger.info(
                "No seed data found for %s (tried %s) — cold start",
                pair, [str(c) for c in candidates],
            )

    return seed_dfs


def main() -> None:
    """Entry point — full trading bot with startup reconciliation and main loop."""
    load_dotenv()
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Roostoo Trading Bot starting up")

    config = _load_config()

    # ── Component initialisation ──────────────────────────────────────────────
    api_key = os.environ.get("ROOSTOO_API_KEY") or os.environ["ROOSTOO_API_KEY_TEST"]
    secret = os.environ.get("ROOSTOO_SECRET") or os.environ["ROOSTOO_SECRET_TEST"]
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")

    client = RoostooClient(api_key=api_key, secret=secret)
    telegram = TelegramAlerter(token=tg_token, chat_id=tg_chat)
    state_manager = StateManager(path=Path(config.get("state_path", "state.json")))
    risk_manager = RiskManager(config=config)
    order_manager = OrderManager(client=client, risk_manager=risk_manager, config=config)
    regime_detector = RegimeDetector(config=config)
    portfolio_allocator = PortfolioAllocator(config=config)

    # Load seed data and initialise LiveFetcher
    # maxlen=3200 (~33 days at 15M) — required for window=2880 in compute_btc_context_features
    seed_dfs = _load_seed_data(config)
    live_fetcher = LiveFetcher(seed_dfs=seed_dfs, maxlen=3200)
    logger.info("LiveFetcher initialised: %s", live_fetcher)

    # Primary strategy: XGBoost 15M model (xgb_btc_15m_iter5.pkl, threshold=0.65)
    strategy = XGBoostStrategy(threshold=0.65)
    # Fallback strategy: mean reversion (fires when momentum returns HOLD)
    mean_reversion_strategy = MeanReversionStrategy()
    # Pairs trading (ETH/SOL and BTC/ETH — only feature_pairs with seeded data)
    pairs_strategy = PairsTradingStrategy(config=config)
    pairs_strategy.add_candidate_pair("ETH/USD", "SOL/USD")
    pairs_strategy.add_candidate_pair("BTC/USD", "ETH/USD")
    logger.info(
        "Strategies: primary=%s fallback=%s pairs=%d candidate pairs",
        strategy.__class__.__name__,
        mean_reversion_strategy.__class__.__name__,
        len(pairs_strategy.candidate_pairs),
    )

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

    telegram.send("\u2705 Bot started — entering main loop")
    logger.info("Startup complete — entering main loop")

    # Restore loop_state from persisted state (preserves last_signal_epoch across restarts)
    saved_state = state_manager.read() or {}
    loop_state: dict[str, Any] = {
        "last_signal_epoch": saved_state.get("last_signal_epoch", 0),
        "features_cache": {},
        "bar_index": 0,
    }

    tradeable_pairs: list[str] = config.get("tradeable_pairs", ["BTC/USD"])
    feature_pairs: list[str] = config.get("feature_pairs", ["BTC/USD"])

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            _run_one_cycle(
                client=client,
                live_fetcher=live_fetcher,
                risk_manager=risk_manager,
                order_manager=order_manager,
                portfolio_allocator=portfolio_allocator,
                telegram=telegram,
                state_manager=state_manager,
                strategy=strategy,
                mean_reversion_strategy=mean_reversion_strategy,
                pairs_strategy=pairs_strategy,
                regime_detector=regime_detector,
                tradeable_pairs=tradeable_pairs,
                feature_pairs=feature_pairs,
                loop_state=loop_state,
                config=config,
            )
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — saving state and exiting")
            state = {
                "risk_manager": risk_manager.dump_state(),
                "order_manager": order_manager.dump_state(),
                "last_signal_epoch": loop_state.get("last_signal_epoch", 0),
                "timestamp": time.time(),
            }
            state_manager.write(state)
            telegram.send("\U0001f6d1 Bot stopped (KeyboardInterrupt) — state saved")
            sys.exit(0)
        except Exception as exc:
            logger.error("Main loop exception: %s", exc, exc_info=True)
            telegram.send(f"\u26a0\ufe0f WARN: main loop exception — retrying in 10s: {exc}")
            time.sleep(10)


if __name__ == "__main__":
    main()
