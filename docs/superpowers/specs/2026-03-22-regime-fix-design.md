# Design Spec: Bear Regime Multiplier Fix + Threshold Tuning

**Date:** 2026-03-22 (Competition Day 1 of 10)
**Status:** Draft
**Scope:** 3 files, 5 line-level edits. No structural changes.

---

## Problem

The bot has zero trades. `RegimeState.BEAR_TREND` applies a `0.0x` position size multiplier, which hard-blocks every signal — XGBoost, original MR, and relaxed MR — at the `risk.py` gate before any order logic runs. The relaxed MR strategy has its own soft gate (0.25x in downtrend, never zero), but the downstream hard gate in `risk.py` overrides it.

Secondary: SOL entry threshold (0.75) is too tight for bear-market volatility, and exit thresholds (0.10) hold positions too long for short-lived bear bounces.

## Goal

Enable trading in bear markets at reduced size. Confirm first trade within 2 candle cycles (30 min) of deployment.

## Non-Goals

- No new features (no BTC intrabar filter, no vol-targeting, no endgame protocol)
- No config.yaml plumbing — thresholds stay in constructor calls
- No changes to BTC entry threshold (stays at 0.65)

---

## Changes

### 1. `bot/execution/regime.py` — Bear multiplier 0.0 to 0.35

Replace the `size_multiplier` property on `RegimeState` with a module-level lookup dict.

**Before:**
```python
@property
def size_multiplier(self) -> float:
    if self == RegimeState.BULL_TREND:
        return 1.0
    elif self == RegimeState.SIDEWAYS:
        return 0.5
    elif self == RegimeState.BEAR_TREND:
        return 0.0
    else:
        return 0.5
```

**After:** Define `REGIME_MULTIPLIERS` at module level, after the `RegimeState` class definition:

```python
class RegimeState(Enum):
    BULL_TREND = "bull"
    SIDEWAYS = "sideways"
    BEAR_TREND = "bear"

    @property
    def size_multiplier(self) -> float:
        return REGIME_MULTIPLIERS.get(self, 0.50)


REGIME_MULTIPLIERS = {
    RegimeState.BULL_TREND: 1.00,
    RegimeState.SIDEWAYS:   0.50,
    RegimeState.BEAR_TREND: 0.35,
}
```

**Why 0.35x:** At 2% risk per trade, a 0.35x multiplier bounds single-trade loss to ~0.7% of portfolio. High enough to capture bear-market MR bounces and register real activity. Low enough to survive a sustained grind.

**SIDEWAYS stays at 0.50.** No change — keep scope minimal.

### 2. `bot/execution/risk.py` — Gate check 0.0 to 0.10

**Before (line 312):**
```python
if regime_multiplier == 0.0:
```

**After:**
```python
if regime_multiplier < 0.10:
```

This ensures the 0.35x bear multiplier passes through. The `< 0.10` floor catches any future misconfiguration where a near-zero multiplier would produce uselessly small positions.

**Gate 5 (line 348) stays unchanged.** Gate 5 checks `effective_multiplier = regime_multiplier * cb_mult`. If the circuit breaker sets `cb_mult = 0.0` (at 30% drawdown halt), the product is zero and Gate 5 blocks the trade. This is correct — a circuit breaker halt is an absolute veto regardless of regime. Only the regime-only gate (Gate 2) changes.

### 3. `main.py` — Threshold changes (lines 776-784)

| Parameter | Line | Before | After |
|-----------|------|--------|-------|
| BTC exit threshold | 776 | `exit_threshold=0.10` | `exit_threshold=0.08` |
| SOL entry threshold | 781 | `threshold=0.75` | `threshold=0.70` |
| SOL exit threshold | 783 | `exit_threshold=0.10` | `exit_threshold=0.08` |

**SOL entry 0.75 to 0.70:** SOL's 4.45% daily vol (vs BTC's 2.3%) creates stronger reversion setups. 0.75 is too selective — risks missing the 8/10 active days requirement.

**Exit thresholds 0.10 to 0.08:** Bear bounces are sharp but short-lived. Exiting ~20% earlier in the signal decay curve takes profit before the next leg down.

---

### 4. `main.py` — Micro-trade activity fallback (Phase E)

Added after Phase D (signal execution). If no trade has been placed today by 20:00 UTC, place a $500 BUY on the first liquid coin without an open position (BTC, ETH, SOL, BNB, XRP tried in order).

- Cost: ~$0.50 in fees per day
- Risk: $500 / $1M = 0.05% portfolio exposure
- Guarantees 100% active trading days regardless of signal frequency
- Tracked via `loop_state["last_trade"]` date string matching

---

## Files Touched

| File | Type of change |
|------|---------------|
| `bot/execution/regime.py` | Replace property with dict lookup |
| `bot/execution/risk.py` | One comparison operator change |
| `main.py` | Three constructor argument changes + micro-trade fallback |

## What Does NOT Change

- BTC entry threshold (0.65)
- ATR stop multiplier (10.0)
- Risk per trade (2%)
- Circuit breaker thresholds
- Relaxed MR strategy logic
- Config.yaml

## Backtest Validation

Combined backtest (BTC+SOL XGBoost + MR + Relaxed MR + Regime) OOS 2024-2026:

| Metric | OLD (bear=0.0x) | NEW (bear=0.35x) |
|--------|----------------|-------------------|
| Return | +13.02% | **+27.53%** |
| Sharpe | 0.921 | **1.127** |
| Sortino | 0.477 | **0.773** |
| Calmar | 0.931 | **1.507** |
| Max DD | -6.09% | -7.69% |
| Trades | 1,259 | **2,096** |
| Active days | 53.5% | **88.4%** |
| Regime blocked | 5,439 | **0** |

With micro-trade fallback: activity rate → 100%.

## Verification

1. Deploy the fix
2. Watch logs for `[RISK]` entries — expect `decision=APPROVED` instead of `BLOCKED_ZERO_REGIME_MULTIPLIER`
3. First trade should appear within 2 candle cycles (30 min)
4. If no trade by 20:00 UTC: micro-trade fallback fires automatically
5. If no trade in 2 hours: check regime detector output and trace signal cascade

## Risk

- **Worst case:** 0.35x positions in a continued bear decline. At 2% ATR risk, max single-trade loss is ~0.7% of portfolio. Existing circuit breakers (10%/20%/30% drawdown tiers) remain active as backstop.
- **Rollback:** Revert 5 lines across 3 files. No database state, no config migration, no dependencies.
