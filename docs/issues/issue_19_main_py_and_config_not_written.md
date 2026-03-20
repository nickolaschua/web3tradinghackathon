# Issue 19: `main.py` and `config.yaml` Do Not Exist

## Layer
Layer 10 — Orchestration

## Description
The entire system is documented but the two most critical files for running the bot do not exist:

1. **`main.py`**: The orchestration entry point that wires all layers together. Layer 10 documents describe its structure in detail, but the file itself is not written. The bot cannot run at all without `main.py`.

2. **`config.yaml`**: Every layer references configuration values from `config.yaml` (e.g. `self.config.get("hard_stop_pct", 0.08)`, `atr_stop_multiplier`, `circuit_breaker_drawdown`, `max_positions`, etc.). None of these have validated values — they all use hardcoded defaults that have not been walk-forward optimized (because `backtest_fold()` is unimplemented, see Issue 08).

## What `config.yaml` Needs
Based on references across all layers, `config.yaml` must contain at minimum:
- `hard_stop_pct` (default 0.08)
- `atr_stop_multiplier` (default 2.0)
- `circuit_breaker_drawdown` (default 0.30)
- `max_positions` (default 3)
- `max_single_position_pct` (default 0.40)
- `base_position_size` (default ~0.17 from half-Kelly)
- `trade_cooldown_seconds` (default 65)
- `candle_interval` ("4h")
- `pairs` list (["BTC/USD"])
- `regime.ema_fast`, `regime.ema_slow`, `regime.adx_threshold`

## Impact
**Critical** — the bot cannot start without `main.py`. Configuration defaults exist in the code but are unvalidated.
