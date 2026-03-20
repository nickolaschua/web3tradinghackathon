"""
Comprehensive test suite for the trading bot framework.
Covers: feature pipeline, risk manager, portfolio allocator, crisis scenarios.
"""
import sys
import math
import traceback
import numpy as np
import pandas as pd

PASS = 0
FAIL = 0
ERRORS = []

def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")

def fail(name, reason):
    global FAIL
    FAIL += 1
    ERRORS.append(f"{name}: {reason}")
    print(f"  FAIL  {name}: {reason}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_ohlcv(n=200, base_price=50000.0, vol=0.01, seed=42):
    """Synthetic 4H OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h")
    close = base_price * np.exp(np.cumsum(rng.normal(0, vol, n)))
    # Roostoo synthetic: H=L=O=C (worst case)
    df = pd.DataFrame({
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": rng.uniform(1e6, 1e7, n),
    }, index=idx)
    return df


def make_realistic_ohlcv(n=200, base_price=50000.0, vol=0.01, seed=42):
    """More realistic OHLCV with H > O/C and L < O/C."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h")
    close = base_price * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low  = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(1e6, 1e7, n),
    }, index=idx)
    return df


# ─────────────────────────────────────────────────────────────
# Section 1: Feature Pipeline
# ─────────────────────────────────────────────────────────────
section("1. Feature Pipeline")

try:
    from bot.data.features import compute_features, compute_cross_asset_features

    # 1a: compute_features returns required columns
    df_raw = make_ohlcv(n=100)
    df_feat = compute_features(df_raw.copy())
    required_cols = ["atr_proxy", "RSI_14", "MACD_12_26_9", "EMA_20", "EMA_50", "ema_slope"]
    missing = [c for c in required_cols if c not in df_feat.columns]
    if missing:
        fail("feature_cols_present", f"Missing: {missing}")
    else:
        ok("feature_cols_present")

    # 1b: features are shifted by 1 bar (no look-ahead)
    df2 = make_ohlcv(n=50)
    df2_feat = compute_features(df2.copy())
    # After shift(1), first row of features should be NaN
    if not pd.isna(df2_feat["RSI_14"].iloc[0]):
        fail("feature_shift_no_lookahead", "RSI_14 row 0 is not NaN — shift(1) may be missing")
    else:
        ok("feature_shift_no_lookahead")

    # 1c: cross-asset features inject eth/sol columns
    btc_df = make_ohlcv(n=100, base_price=50000, seed=1)
    eth_df = make_ohlcv(n=100, base_price=3000, seed=2)
    sol_df = make_ohlcv(n=100, base_price=150, seed=3)
    btc_feat = compute_features(btc_df.copy())
    cross_feat_dfs = {"ETH/USD": eth_df, "SOL/USD": sol_df}
    btc_cross = compute_cross_asset_features(btc_feat, cross_feat_dfs)
    cross_cols = ["eth_return_lag1", "eth_return_lag2", "sol_return_lag1", "sol_return_lag2"]
    missing_cross = [c for c in cross_cols if c not in btc_cross.columns]
    if missing_cross:
        fail("cross_asset_cols", f"Missing: {missing_cross}")
    else:
        ok("cross_asset_cols")

    # 1d: dropna after cross-asset doesn't empty the DataFrame
    btc_clean = btc_cross.dropna()
    if len(btc_clean) < 10:
        fail("cross_asset_dropna", f"Only {len(btc_clean)} rows after dropna (expected 10+)")
    else:
        ok("cross_asset_dropna")

    # 1e: ATR proxy is positive for synthetic candles (H=L=O=C)
    df_synth = make_ohlcv(n=60)
    df_synth_feat = compute_features(df_synth.copy())
    atr_vals = df_synth_feat["atr_proxy"].dropna()
    if len(atr_vals) == 0:
        fail("atr_proxy_positive_synthetic", "No non-NaN ATR values")
    elif (atr_vals <= 0).any():
        fail("atr_proxy_positive_synthetic", f"Some ATR values <= 0: {atr_vals[atr_vals <= 0].values[:3]}")
    else:
        ok("atr_proxy_positive_synthetic")

except Exception as e:
    fail("feature_pipeline_import", traceback.format_exc())


# ─────────────────────────────────────────────────────────────
# Section 2: RiskManager
# ─────────────────────────────────────────────────────────────
section("2. RiskManager")

try:
    from bot.execution.risk import RiskManager, RiskDecision

    cfg = {
        "hard_stop_pct": 0.05,
        "atr_stop_multiplier": 2.0,
        "circuit_breaker_drawdown": 0.30,
        "max_positions": 5,
        "max_single_position_pct": 0.40,
        "risk_per_trade_pct": 0.02,
        "expected_win_loss_ratio": 1.5,
    }

    # 2a: Normal approved trade
    rm = RiskManager(cfg)
    rm.initialize_hwm(10000.0)
    result = rm.size_new_position(
        "BTC/USD", current_price=50000.0, current_atr=1000.0,
        free_balance_usd=10000.0, open_positions={},
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    if result.decision != RiskDecision.APPROVED:
        fail("approved_normal", f"Expected APPROVED, got {result.decision}: {result.reason}")
    elif result.approved_quantity <= 0:
        fail("approved_normal", f"Quantity is {result.approved_quantity}")
    else:
        ok("approved_normal")

    # 2b: Kelly gate — very low confidence (p=0.3, b=1.5 → kelly = (0.3*1.5 - 0.7)/1.5 = -0.167)
    rm2 = RiskManager(cfg)
    rm2.initialize_hwm(10000.0)
    result2 = rm2.size_new_position(
        "BTC/USD", 50000.0, 1000.0, 10000.0, {},
        regime_multiplier=1.0, confidence=0.3, portfolio_weight=1.0
    )
    if result2.decision != RiskDecision.BLOCKED_NEGATIVE_KELLY:
        fail("kelly_gate_low_confidence", f"Expected BLOCKED_NEGATIVE_KELLY, got {result2.decision}")
    else:
        ok("kelly_gate_low_confidence")

    # 2c: Bear regime blocks trade
    rm3 = RiskManager(cfg)
    rm3.initialize_hwm(10000.0)
    result3 = rm3.size_new_position(
        "BTC/USD", 50000.0, 1000.0, 10000.0, {},
        regime_multiplier=0.0, confidence=0.7, portfolio_weight=1.0
    )
    if result3.decision != RiskDecision.BLOCKED_ZERO_REGIME_MULTIPLIER:
        fail("bear_regime_blocked", f"Expected BLOCKED_ZERO_REGIME_MULTIPLIER, got {result3.decision}")
    else:
        ok("bear_regime_blocked")

    # 2d: Circuit breaker blocks at 30% drawdown
    rm4 = RiskManager(cfg)
    rm4.initialize_hwm(10000.0)
    rm4.check_circuit_breaker(7000.0)  # 30% drawdown → activates CB
    result4 = rm4.size_new_position(
        "BTC/USD", 50000.0, 1000.0, 7000.0, {},
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    if result4.decision != RiskDecision.BLOCKED_CIRCUIT_BREAKER:
        fail("circuit_breaker_blocks", f"Expected BLOCKED_CIRCUIT_BREAKER, got {result4.decision}")
    else:
        ok("circuit_breaker_blocks")

    # 2e: Max positions reached
    rm5 = RiskManager({**cfg, "max_positions": 2})
    rm5.initialize_hwm(10000.0)
    open_pos = {"BTC/USD": 2000.0, "ETH/USD": 2000.0}
    result5 = rm5.size_new_position(
        "SOL/USD", 150.0, 5.0, 6000.0, open_pos,
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    if result5.decision != RiskDecision.BLOCKED_MAX_POSITIONS:
        fail("max_positions_blocked", f"Expected BLOCKED_MAX_POSITIONS, got {result5.decision}")
    else:
        ok("max_positions_blocked")

    # 2f: Low balance blocks
    rm6 = RiskManager(cfg)
    rm6.initialize_hwm(10000.0)
    result6 = rm6.size_new_position(
        "BTC/USD", 50000.0, 1000.0, free_balance_usd=50.0, open_positions={},
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    if result6.decision != RiskDecision.BLOCKED_INSUFFICIENT_BALANCE:
        fail("low_balance_blocked", f"Expected BLOCKED_INSUFFICIENT_BALANCE, got {result6.decision}")
    else:
        ok("low_balance_blocked")

    # 2g: NaN ATR falls back to hard stop
    rm7 = RiskManager(cfg)
    rm7.initialize_hwm(10000.0)
    result7 = rm7.size_new_position(
        "BTC/USD", 50000.0, float("nan"), 10000.0, {},
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    if result7.decision != RiskDecision.APPROVED:
        fail("nan_atr_fallback", f"Expected APPROVED with NaN ATR, got {result7.decision}: {result7.reason}")
    else:
        ok("nan_atr_fallback")

    # 2h: Zero ATR falls back to hard stop
    rm8 = RiskManager(cfg)
    rm8.initialize_hwm(10000.0)
    result8 = rm8.size_new_position(
        "BTC/USD", 50000.0, 0.0, 10000.0, {},
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    if result8.decision != RiskDecision.APPROVED:
        fail("zero_atr_fallback", f"Expected APPROVED with zero ATR, got {result8.decision}: {result8.reason}")
    else:
        ok("zero_atr_fallback")

    # 2i: Hard stop triggers on price drop
    rm9 = RiskManager(cfg)
    rm9.record_entry("BTC/USD", entry_price=50000.0, initial_stop=47500.0)
    stop_check = rm9.check_stops("BTC/USD", current_price=47000.0, current_atr=float("nan"))
    if not stop_check.should_exit or stop_check.exit_type != "hard_pct":
        fail("hard_stop_triggers", f"Expected hard_pct exit, got: should_exit={stop_check.should_exit} type={stop_check.exit_type}")
    else:
        ok("hard_stop_triggers")

    # 2j: Tiered CB multiplier — 15% drawdown → 0.5x
    rm10 = RiskManager(cfg)
    rm10.initialize_hwm(10000.0)
    mult = rm10.get_cb_size_multiplier(8500.0)  # 15% drawdown
    if not math.isclose(mult, 0.5, rel_tol=1e-6):
        fail("tiered_cb_15pct", f"Expected 0.5x at 15% drawdown, got {mult}")
    else:
        ok("tiered_cb_15pct")

    # 2k: Tiered CB multiplier — 25% drawdown → 0.25x
    rm11 = RiskManager(cfg)
    rm11.initialize_hwm(10000.0)
    mult2 = rm11.get_cb_size_multiplier(7500.0)  # 25% drawdown
    if not math.isclose(mult2, 0.25, rel_tol=1e-6):
        fail("tiered_cb_25pct", f"Expected 0.25x at 25% drawdown, got {mult2}")
    else:
        ok("tiered_cb_25pct")

    # 2l: Equal dollar risk sizing — wider stops produce smaller positions
    rm12 = RiskManager(cfg)
    rm12.initialize_hwm(10000.0)
    r_narrow = rm12.size_new_position(
        "BTC/USD", 50000.0, 500.0, 10000.0, {},  # ATR=500 → stop_distance~1000
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    r_wide = rm12.size_new_position(
        "BTC/USD", 50000.0, 2000.0, 10000.0, {},  # ATR=2000 → stop_distance~4000
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    if r_narrow.approved_quantity <= r_wide.approved_quantity:
        fail("equal_dollar_risk_sizing",
             f"Narrow stop should produce larger position: narrow={r_narrow.approved_quantity:.6f} wide={r_wide.approved_quantity:.6f}")
    else:
        ok("equal_dollar_risk_sizing")

    # 2m: dump_state / load_state roundtrip
    rm13 = RiskManager(cfg)
    rm13.initialize_hwm(10000.0)
    rm13.record_entry("BTC/USD", 50000.0, 47500.0)
    rm13.check_circuit_breaker(9000.0)
    state = rm13.dump_state()
    rm14 = RiskManager(cfg)
    rm14.load_state(state)
    if rm14._entry_prices != rm13._entry_prices:
        fail("state_roundtrip", "Entry prices mismatch after load_state")
    elif rm14._portfolio_hwm != rm13._portfolio_hwm:
        fail("state_roundtrip", "HWM mismatch after load_state")
    else:
        ok("state_roundtrip")

except Exception as e:
    fail("risk_manager_section", traceback.format_exc())


# ─────────────────────────────────────────────────────────────
# Section 3: PortfolioAllocator
# ─────────────────────────────────────────────────────────────
section("3. PortfolioAllocator")

try:
    from bot.execution.portfolio import PortfolioAllocator, _build_returns_df

    pa_cfg = {"cvar_beta": 0.95, "hrp_blend": 0.5}

    # 3a: Insufficient history → empty returns df
    short_hist = {
        "BTC/USD": make_ohlcv(n=30, seed=1),
        "ETH/USD": make_ohlcv(n=30, seed=2),
    }
    ret_df = _build_returns_df(short_hist)
    if not ret_df.empty:
        fail("short_history_empty", f"Expected empty DataFrame for <60 bars, got {len(ret_df)} rows")
    else:
        ok("short_history_empty")

    # 3b: Sufficient history → non-empty returns df
    long_hist = {
        "BTC/USD": make_ohlcv(n=120, seed=1),
        "ETH/USD": make_ohlcv(n=120, seed=2),
        "SOL/USD": make_ohlcv(n=120, seed=3),
    }
    ret_df2 = _build_returns_df(long_hist)
    if ret_df2.empty:
        fail("sufficient_history_nonempty", "Expected non-empty DataFrame for 120 bars")
    else:
        ok("sufficient_history_nonempty")

    # 3c: compute_weights → weights sum to ~1.0
    pa = PortfolioAllocator(pa_cfg)
    pa.compute_weights(long_hist)
    weights = {p: pa.get_pair_weight(p) for p in long_hist}
    total_weight = sum(weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=0.01):
        fail("weights_sum_to_one", f"Weights sum to {total_weight:.4f}, expected ~1.0")
    else:
        ok("weights_sum_to_one")

    # 3d: All weights non-negative
    if any(w < 0 for w in weights.values()):
        fail("weights_nonnegative", f"Negative weights: {weights}")
    else:
        ok("weights_nonnegative")

    # 3e: Fallback before compute_weights → equal weight
    pa_fresh = PortfolioAllocator(pa_cfg)
    w = pa_fresh.get_pair_weight("BTC/USD", n_active_pairs=3)
    expected = 1.0 / 3
    if not math.isclose(w, expected, rel_tol=0.01):
        fail("fallback_equal_weight", f"Expected 1/3={expected:.4f}, got {w:.4f}")
    else:
        ok("fallback_equal_weight")

    # 3f: Single pair → returns empty (need >=2 pairs)
    single_hist = {"BTC/USD": make_ohlcv(n=120, seed=1)}
    ret_single = _build_returns_df(single_hist)
    if not ret_single.empty:
        fail("single_pair_empty_returns", "Expected empty returns df for single pair")
    else:
        ok("single_pair_empty_returns")

except Exception as e:
    fail("portfolio_allocator_section", traceback.format_exc())


# ─────────────────────────────────────────────────────────────
# Section 4: Crisis Scenarios
# ─────────────────────────────────────────────────────────────
section("4. Crisis Scenarios")

try:
    from bot.execution.risk import RiskManager, RiskDecision
    from bot.data.features import compute_features

    crisis_cfg = {
        "hard_stop_pct": 0.05,
        "atr_stop_multiplier": 2.0,
        "circuit_breaker_drawdown": 0.30,
        "max_positions": 5,
        "max_single_position_pct": 0.40,
        "risk_per_trade_pct": 0.02,
        "expected_win_loss_ratio": 1.5,
    }

    # 4a: Flash crash — ATR spikes to 20% of price → very wide stop → tiny position
    rm_flash = RiskManager(crisis_cfg)
    rm_flash.initialize_hwm(10000.0)
    flash_atr = 50000.0 * 0.20  # ATR = 20% of price
    r_flash = rm_flash.size_new_position(
        "BTC/USD", 50000.0, flash_atr, 10000.0, {},
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    if r_flash.decision == RiskDecision.APPROVED:
        # Position should be very small due to wide stop
        max_normal_usd = 10000.0 * 0.40  # 40% cap
        if r_flash.approved_usd_value > max_normal_usd:
            fail("flash_crash_tiny_position", f"Flash crash position ${r_flash.approved_usd_value:.0f} > cap ${max_normal_usd:.0f}")
        else:
            ok("flash_crash_tiny_position")
    elif r_flash.decision == RiskDecision.BLOCKED_NEGATIVE_KELLY:
        ok("flash_crash_tiny_position")  # Kelly blocked is also acceptable
    else:
        fail("flash_crash_tiny_position", f"Unexpected decision: {r_flash.decision}")

    # 4b: Flash crash hard stop triggers on -6% price move
    rm_stop = RiskManager(crisis_cfg)
    entry_price = 50000.0
    rm_stop.record_entry("BTC/USD", entry_price, entry_price * 0.95)
    crashed_price = entry_price * 0.93  # -7% (below 5% hard stop)
    stop_result = rm_stop.check_stops("BTC/USD", crashed_price, float("nan"))
    if not stop_result.should_exit:
        fail("flash_crash_stop_trigger", f"Hard stop should trigger at -7%, got should_exit=False")
    else:
        ok("flash_crash_stop_trigger")

    # 4c: 30% drawdown triggers circuit breaker
    rm_cb = RiskManager(crisis_cfg)
    rm_cb.initialize_hwm(100000.0)
    active = rm_cb.check_circuit_breaker(70000.0)  # exactly -30%
    if not active:
        fail("cb_at_30pct", "Circuit breaker should activate at exactly 30% drawdown")
    else:
        ok("cb_at_30pct")

    # 4d: Circuit breaker deactivates on recovery
    rm_cb2 = RiskManager(crisis_cfg)
    rm_cb2.initialize_hwm(100000.0)
    rm_cb2.check_circuit_breaker(69000.0)  # 31% drawdown → CB active
    assert rm_cb2._circuit_breaker_active, "CB should be active before recovery"
    rm_cb2.check_circuit_breaker(101000.0)  # New high → CB deactivates
    if rm_cb2._circuit_breaker_active:
        fail("cb_deactivates_on_recovery", "Circuit breaker should deactivate when portfolio recovers to new HWM")
    else:
        ok("cb_deactivates_on_recovery")

    # 4e: Correlated crash — BTC and ETH crash together, HRP rebalances next cycle
    # (We can only test that compute_weights still runs and produces valid output)
    from bot.execution.portfolio import PortfolioAllocator
    rng = np.random.default_rng(99)
    idx = pd.date_range("2023-01-01", periods=150, freq="4h")
    # Highly correlated crash scenario: both fall ~50% in last 30 bars
    btc_prices = np.concatenate([
        50000 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, 120))),
        50000 * np.exp(np.cumsum(rng.normal(-0.02, 0.02, 30))),
    ])
    eth_prices = btc_prices * 0.06 + rng.normal(0, 10, 150)  # highly correlated with BTC
    sol_prices = btc_prices * 0.003 + rng.normal(0, 1, 150)
    crash_hist = {
        "BTC/USD": pd.DataFrame({"close": btc_prices}, index=idx),
        "ETH/USD": pd.DataFrame({"close": eth_prices}, index=idx),
        "SOL/USD": pd.DataFrame({"close": sol_prices}, index=idx),
    }
    pa_crash = PortfolioAllocator({"cvar_beta": 0.95, "hrp_blend": 0.5})
    pa_crash.compute_weights(crash_hist)
    crash_weights = {p: pa_crash.get_pair_weight(p) for p in crash_hist}
    total_cw = sum(crash_weights.values())
    if not math.isclose(total_cw, 1.0, rel_tol=0.05):
        fail("crash_weights_valid", f"Crash scenario weights sum {total_cw:.4f}, expected ~1.0")
    else:
        ok("crash_weights_valid")

    # 4f: High volatility scenario — position size should be <= concentration cap
    rm_vol = RiskManager(crisis_cfg)
    rm_vol.initialize_hwm(10000.0)
    # Very high ATR (10% of price) → stop_distance is large → small quantity
    high_vol_result = rm_vol.size_new_position(
        "BTC/USD", 50000.0, 5000.0, 10000.0, {},
        regime_multiplier=1.0, confidence=0.8, portfolio_weight=1.0
    )
    if high_vol_result.decision == RiskDecision.APPROVED:
        max_cap = 10000.0 * 0.40  # 40% concentration cap
        if high_vol_result.approved_usd_value > max_cap * 1.01:
            fail("high_vol_size_capped", f"Position ${high_vol_result.approved_usd_value:.0f} exceeds concentration cap ${max_cap:.0f}")
        else:
            ok("high_vol_size_capped")
    else:
        ok("high_vol_size_capped")  # Any block is also acceptable

    # 4g: Synthetic candles (H=L=O=C) — ATR proxy stays positive
    df_synthetic = make_ohlcv(n=80)  # H=L=O=C
    df_syn_feat = compute_features(df_synthetic.copy())
    atr_syn = df_syn_feat["atr_proxy"].dropna()
    if len(atr_syn) == 0 or (atr_syn <= 0).any():
        fail("synthetic_candle_atr", f"ATR proxy has zero/negative values with H=L=O=C candles")
    else:
        ok("synthetic_candle_atr")

    # 4h: Absurdly large ATR (> price) → stop_distance floor guard prevents division error
    rm_absurd = RiskManager(crisis_cfg)
    rm_absurd.initialize_hwm(10000.0)
    try:
        r_absurd = rm_absurd.size_new_position(
            "BTC/USD", 100.0, 1e6, 10000.0, {},  # ATR=1M when price=100 → atr_stop << 0
            regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
        )
        # Should either be approved with tiny position or blocked by Kelly
        if r_absurd.approved_quantity < 0:
            fail("absurd_atr_guard", f"Negative quantity {r_absurd.approved_quantity}")
        else:
            ok("absurd_atr_guard")
    except Exception as exc:
        fail("absurd_atr_guard", f"Raised exception with absurd ATR: {exc}")

    # 4i: Sideways regime (0.5x) — position half size of bull
    rm_sideways_bull = RiskManager(crisis_cfg)
    rm_sideways_bull.initialize_hwm(10000.0)
    r_bull = rm_sideways_bull.size_new_position(
        "BTC/USD", 50000.0, 1000.0, 10000.0, {},
        regime_multiplier=1.0, confidence=0.7, portfolio_weight=1.0
    )
    rm_sideways = RiskManager(crisis_cfg)
    rm_sideways.initialize_hwm(10000.0)
    r_sideways = rm_sideways.size_new_position(
        "BTC/USD", 50000.0, 1000.0, 10000.0, {},
        regime_multiplier=0.5, confidence=0.7, portfolio_weight=1.0
    )
    if r_bull.decision == RiskDecision.APPROVED and r_sideways.decision == RiskDecision.APPROVED:
        ratio = r_sideways.approved_usd_value / r_bull.approved_usd_value
        if not math.isclose(ratio, 0.5, rel_tol=0.05):
            fail("sideways_half_size", f"Sideways/bull size ratio={ratio:.3f}, expected 0.5")
        else:
            ok("sideways_half_size")
    else:
        fail("sideways_half_size", f"Bull={r_bull.decision} Sideways={r_sideways.decision}")

except Exception as e:
    fail("crisis_section", traceback.format_exc())


# ─────────────────────────────────────────────────────────────
# Section 5: Configuration & Integration Checks
# ─────────────────────────────────────────────────────────────
section("5. Configuration & Integration")

try:
    import yaml
    import os
    config_path = os.path.join(os.path.dirname(__file__), "bot", "config", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 5a: Required risk config keys present
    required_keys = [
        "hard_stop_pct", "atr_stop_multiplier",
        "risk_per_trade_pct", "expected_win_loss_ratio",
        "cvar_beta", "hrp_blend", "max_positions",
    ]
    missing_cfg = [k for k in required_keys if k not in config]
    if missing_cfg:
        fail("config_risk_keys", f"Missing config keys: {missing_cfg}")
    else:
        ok("config_risk_keys")

    # 5b: Circuit breaker nested config
    if "circuit_breaker" not in config:
        fail("config_circuit_breaker", "Missing 'circuit_breaker' section")
    else:
        cb = config["circuit_breaker"]
        cb_keys = ["halt_threshold", "reduce_heavy_threshold", "reduce_light_threshold"]
        missing_cb = [k for k in cb_keys if k not in cb]
        if missing_cb:
            fail("config_circuit_breaker", f"Missing CB keys: {missing_cb}")
        else:
            ok("config_circuit_breaker")

    # 5c: Feature pairs are a subset of tradeable pairs
    tradeable = set(config.get("tradeable_pairs", []))
    feature = set(config.get("feature_pairs", []))
    if not feature.issubset(tradeable):
        fail("feature_subset_tradeable", f"Feature pairs not in tradeable: {feature - tradeable}")
    else:
        ok("feature_subset_tradeable")

    # 5d: hard_stop_pct matches risk.py default behavior
    hsp = config.get("hard_stop_pct", 0)
    if not (0 < hsp <= 0.20):
        fail("hard_stop_pct_sane", f"hard_stop_pct={hsp} seems out of range (0, 0.20]")
    else:
        ok("hard_stop_pct_sane")

    # 5e: risk_per_trade_pct is sane
    rpt = config.get("risk_per_trade_pct", 0)
    if not (0 < rpt <= 0.05):
        fail("risk_per_trade_pct_sane", f"risk_per_trade_pct={rpt} seems out of range (0, 0.05]")
    else:
        ok("risk_per_trade_pct_sane")

except Exception as e:
    fail("config_section", traceback.format_exc())


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
print(f"{'='*60}")
if ERRORS:
    print("\nFailed tests:")
    for e in ERRORS:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
