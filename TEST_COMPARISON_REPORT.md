# Shared-Engine Test Report
**Scope:** Unified shared backtest engine only  
**Period:** 2024-01-01 to 2026-03-20 (4H, ~4,851 bars)  
**Core score:** `0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar`  
**Fees:** 10 bps per side (0.10% entry + 0.10% exit)

---

## Strategy Set (Shared Engine)
Framework used:
- shared features from `bot.data.features`
- shared risk/stop/sizing from `bot.execution.risk.RiskManager`
- shared allocation from `bot.execution.portfolio.PortfolioAllocator`

Compared strategies:
- `Base Indicators Only`
- `Oil + DXY`
- `Funding + Oil + DXY`
- `Funding Only`
- `Funding Only + Macro Filter` (BUY only when `oil_return_1d > 0` and `dxy_return_1d < 0`)

---

## Results

| Strategy | Composite | Sharpe | Sortino | Calmar | Return | Max DD | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Funding Only + Macro Filter** 🏆 | **-0.143** | -0.174 | -0.207 | -0.026 | -0.27% | -2.61% | 27 |
| Funding Only | -0.267 | -0.414 | -0.378 | 0.029 | -1.26% | n/a | 175 |
| Oil + DXY | -0.600 | -0.724 | -0.904 | -0.070 | -3.84% | -10.40% | 138 |
| Base Indicators Only | -0.745 | -0.919 | -1.134 | -0.055 | -4.54% | n/a | 143 |
| Funding + Oil + DXY | -0.782 | -0.907 | -1.186 | -0.119 | -4.60% | n/a | 128 |

---

## Key Reading
- Best variant in this regime is **Funding Only + Macro Filter**.
- The macro filter appears to improve quality by reducing churn (27 vs 175 trades).
- All strategies remain negative in return for this test window, but filtered funding is closest to breakeven with the shallowest observed drawdown.

---

## Weighted Momentum Score Tuning (40 tests)
Rule tested:
- Base momentum gate: `EMA_20 > EMA_50` and `RSI_14 < 50` and `MACDh_12_26_9 > 0`
- Score starts at `1.0`, then:
  - `+funding_bonus` if `btc_funding_zscore < funding_z_threshold`
  - `+macro_bonus` if `oil_return_1d > macro_threshold` and `dxy_return_1d < -macro_threshold`
  - `+vol_bonus` if `volume_ratio > vol_threshold`
- BUY when `score >= score_threshold`

Fixed bonuses during sweeps:
- `funding_bonus = 0.5`
- `macro_bonus = 0.5`
- `vol_bonus = 0.3`

Sweeps run (10 values each):
- `score_threshold`
- `funding_z_threshold`
- `vol_threshold`
- `macro_threshold`

### Best parameter per sweep

| Sweep | Best value | Composite | Sharpe | Sortino | Calmar | Trades | Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| `score_threshold` | `1.2` | 1.733 | 1.650 | 2.739 | 0.476 | 16 | +4.02% |
| `funding_z_threshold` | `-1.25` | 1.155 | 1.006 | 1.927 | 0.274 | 4 | +1.38% |
| `vol_threshold` | `1.4` | 1.024 | 0.863 | 1.755 | 0.211 | 2 | +1.00% |
| `macro_threshold` | `0.0` | 0.755 | 0.617 | 1.303 | 0.161 | 3 | +0.71% |

### Combined best-threshold set (applied together)

| Params | Composite | Sharpe | Sortino | Calmar | Trades | Return | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `funding_z=-1.25`, `macro=0.0`, `vol=1.4`, `score=1.2` | **1.752** | 1.645 | 2.803 | 0.456 | 14 | +3.71% | -1.90% |

Notes:
- This outperformed prior shared-engine variants in this sample window.
- Trade count is still small, so treat this as promising but not production-final; verify with additional periods/walk-forward windows.

---

## Weekly Constraint Re-Test (1-week windows, must trade daily)
To align with your live requirement, testing was updated to:
- evaluate on rolling 7-day windows (`evaluation_mode=weekly`)
- enforce feasibility targets:
  - `min_daily_coverage = 1.0` (activity every day)
  - `min_trades_per_day = 1.0`

Outcome from the same 40-test threshold sweep:

| Sweep | Best value (by composite) | Composite | Trades | Avg day coverage | Avg trades/day | Feasible? |
|---|---:|---:|---:|---:|---:|---:|
| `score_threshold` | 1.2 | 1.153 | 11 | 3.9% | 0.02 | No |
| `funding_z_threshold` | -1.25 | 0.380 | 3 | 1.0% | 0.00 | No |
| `vol_threshold` | 1.4 | 0.206 | 2 | 0.6% | 0.00 | No |
| `macro_threshold` | 0.0 | 0.103 | 3 | 0.8% | 0.00 | No |

Interpretation:
- Under a strict 1-week "trade all days" objective, this rule family is too selective.
- The previously strong tuned settings are not robust for mandatory daily activity.
- Next iteration should optimize directly for activity-constrained objectives (or use intraday/shorter-horizon triggers).

---

## Size-Overlay Momentum Tuning (new approach)
Updated method (as requested):
- Keep an always-on momentum base signal for entries.
- Convert funding/macro/vol from entry filters to multiplicative position scaling.
- Optimize on shared engine with weekly windows + feasibility constraints:
  - `daily_coverage >= 1.0`
  - `trades_per_day >= 1.0`

Rule form:
```python
if base_signal:
    size = base_size
    if funding_z < funding_z_threshold:
        size *= funding_mult
    if macro_condition:
        size *= macro_mult
    if vol_spike > vol_threshold:
        size *= vol_mult
    BUY(size)
```

Tuning summary (6 sweeps x 10 tests = 60 tests):

| Sweep | Best value (by composite) | Composite | Trades | Avg day coverage | Avg trades/day | Avg utilization | Feasible? |
|---|---:|---:|---:|---:|---:|---:|---:|
| `funding_z_threshold` | 0.0 | 1.084 | 22 | 5.0% | 0.03 | 0.66% | No |
| `funding_mult` | 1.2 | 1.084 | 22 | 5.0% | 0.03 | 0.66% | No |
| `macro_threshold` | 0.0 | 1.084 | 22 | 5.0% | 0.03 | 0.66% | No |
| `macro_mult` | 1.9 | 1.084 | 22 | 5.0% | 0.03 | 0.66% | No |
| `vol_threshold` | 1.0 | 1.084 | 22 | 5.0% | 0.03 | 0.66% | No |
| `vol_mult` | 1.7 | 1.086 | 22 | 5.0% | 0.03 | 0.67% | No |

Combined best-threshold set:
- `funding_z_threshold=0.0`
- `funding_mult=1.2`
- `macro_threshold=0.0`
- `macro_mult=1.9`
- `vol_threshold=1.0`
- `vol_mult=1.7`

Combined performance:
- Composite: **1.086**
- Trades: **22**
- Avg day coverage: **5.0%**
- Avg trades/day: **0.03**
- Avg utilization: **0.68%**
- **Feasible under constraints:** No

### Overfit diagnostics (performed)
1) **Tune vs holdout split over windows (60/40):**
- Tune composite: `0.430`
- Holdout composite: `2.038`
- Large dispersion indicates unstable estimate under sparse-activity regime.

2) **Local sensitivity around best params (9-neighbor grid):**
- Mean composite: `1.0859`
- Std composite: `0.000004`
- Very flat local surface: objective barely changes because strategy hardly trades.

Conclusion:
- Switching to size overlays improved conceptual robustness, but still fails your operational requirement (trade every day in 1-week window).
- Main bottleneck is not score optimization; it is **insufficient signal frequency/activity** at this 4H setup.

---

## Intraday Hybrid Layer (5m/15m) with 4H Bias Sizing
New architecture added:
- **Slow layer (4H):** directional bias score from trend/funding/macro (0..3)
- **Fast layer (5m/15m):** intraday trigger generation
- **Execution:** shared `RiskManager.size_new_position()` with bias-driven size multiplier

Design rule (as requested):
- Trigger always controls entry timing.
- 4H bias **does not filter** entries; it scales position size.

Tuning mode:
- Weekly windows (7-day blocks)
- Frequency-first objective:
  1) satisfy feasibility
     - `daily_coverage >= 1.0`
     - `trades_per_day >= 1.0`
  2) maximize utilization/composite among feasible sets
- Search run: 72 candidates per interval (5m and 15m), then validated on 12 most recent weekly windows.

### Frequency feasibility outcome

| Interval | Best trigger family | Feasible all validation windows? | Feasible ratio | Avg trades/day | Avg daily coverage | Avg utilization |
|---|---|---:|---:|---:|---:|---:|
| `5m` | Mean-reversion (`zscore=0.5`) | **Yes** | 1.00 | 81.23 | 100.0% | 15.2% |
| `15m` | Mean-reversion (`zscore=0.3`) | **Yes** | 1.00 | 30.44 | 100.0% | 18.2% |

Best validated parameter sets:
- `5m`: `trigger_mode=mean_reversion`, `zscore_threshold=0.5`, `base_size=0.12`, `bias_weight=0.1`, `max_hold_bars=2`
- `15m`: `trigger_mode=mean_reversion`, `zscore_threshold=0.3`, `base_size=0.12`, `bias_weight=0.1`, `max_hold_bars=2`

### Quality check (important)
- These configurations satisfy the **activity** mandate robustly (every validation week).
- But current risk-adjusted quality is poor (very negative Sharpe/composite), indicating hyper-active churn under current fee/risk settings.
- Practical implication: intraday layer solves frequency; next iteration must optimize **quality under frequency constraints** (e.g., stronger microstructure filter, cooldown, trade clustering control, and fee-aware trigger gating).

Artifacts:
- tuner output: `research_results/intraday_hybrid_tuning.json`
- backtest outputs: `research_results/intraday_hybrid_5m.json`, `research_results/intraday_hybrid_15m.json`

---

## Intraday Quality Pass-2 (requested six fixes)
Scope:
- optimize **quality under frequency constraints** using weekly windows
- constraints held fixed: `daily_coverage >= 1.0`, `trades_per_day >= 1.0`
- report metrics: Sharpe, Sortino, Calmar, composite (`0.4*Sortino + 0.3*Sharpe + 0.3*Calmar`)

Fixes tested individually:
1. higher z-score threshold (`1.0`, `1.2`, `1.5`)
2. trend confirmation (`EMA_5 > EMA_20`)
3. volume confirmation (`volume_ratio > 1.1/1.2/1.3`)
4. cooldown bars (`3/4/6`)
5. minimum expected edge (`0.002/0.003/0.004`)
6. max trades/day cap (`10/15/20`)

### 5m results (12 weekly windows)

| Fix | Best parameter | Feasible all windows? | Trades/day | Coverage | Sharpe | Sortino | Calmar | Composite |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Z-score threshold | `1.5` | Yes | 19.06 | 100.0% | -66.42 | -72.67 | -0.130 | -49.03 |
| Trend confirmation | `off` (better than `on`) | Yes | 26.05 | 100.0% | -78.78 | -85.12 | -0.132 | -57.72 |
| Volume confirmation | `on`, `volume_ratio > 1.3` | Yes | 13.37 | 100.0% | -51.78 | -58.08 | -0.127 | -38.81 |
| Cooldown | `6` bars | Yes | 13.61 | 100.0% | -57.27 | -62.71 | -0.128 | -42.30 |
| Min expected edge | `0.004` | **No** | 4.51 | 71.4% | -20.55 | -23.58 | -0.113 | -15.63 |
| Max trades/day | `10` | Yes | 10.00 | 100.0% | -54.40 | -57.84 | -0.127 | -39.49 |

Pass-2 composite trial (all strict fixes together):
- best combined set became **infeasible** (`coverage ~9.5%`, `tpd ~0.10`)
- signal quality improved materially when active, but activity collapsed under full stacking.

### 15m results (12 weekly windows)

| Fix | Best parameter | Feasible all windows? | Trades/day | Coverage | Sharpe | Sortino | Calmar | Composite |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Z-score threshold | `1.5` | Yes | 6.10 | 100.0% | -21.47 | -24.48 | -0.303 | -16.33 |
| Trend confirmation | `off` (better than `on`) | Yes | 8.71 | 100.0% | -29.72 | -33.89 | -0.358 | -22.58 |
| Volume confirmation | `on`, `volume_ratio > 1.3` | No (11/12) | 4.70 | 98.8% | -17.96 | -20.69 | -0.321 | -13.76 |
| Cooldown | `6` bars | Yes | 4.52 | 100.0% | -20.65 | -23.03 | -0.324 | -15.50 |
| Min expected edge | `0.004` | **No** | 4.18 | 81.0% | -12.73 | -15.15 | -0.284 | -9.97 |
| Max trades/day | `10` | Yes | 7.70 | 100.0% | -28.64 | -32.56 | -0.355 | -21.72 |

Pass-2 composite trial (all strict fixes together):
- best combined set also **infeasible** (`coverage ~13.1%`, `tpd ~0.13`)
- same pattern as 5m: better per-trade profile, but insufficient activity when all gates are active together.

### Overfit diagnostics (performed)
1) Tune vs holdout split:
- **5m** composite candidate: tune composite `-2.470` vs holdout `-4.991` (degrades on holdout)
- **15m** composite candidate: tune `-2.336` vs holdout `+3.056` (high dispersion due to sparse active trades)

2) Local sensitivity around best composite:
- **5m**: composite std `0.624` (moderate sensitivity)
- **15m**: composite std `1.134` (higher instability)

Interpretation:
- Requested fixes clearly improve quality metrics and reduce churn **individually**.
- Stacking all strict constraints simultaneously over-constrains entries and breaks the daily-activity mandate.
- Best practical next step is a **soft-gating model**: keep z-score/cooldown/trade-cap hard, but make trend/volume/edge affect size (or confidence) rather than binary entry.

Artifacts:
- pass-2 tuning output: `research_results/intraday_quality_pass2.json`

---

Generated: 2026-03-20
