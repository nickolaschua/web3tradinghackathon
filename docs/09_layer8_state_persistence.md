# Layer 8 — State & Persistence

## What This Layer Does

The State & Persistence layer writes the bot's complete operational state to disk after every trading cycle and reads it back on startup. It is the mechanism that makes the system crash-safe: a process kill, EC2 reboot, or deployment restart should lose nothing beyond the cycle it was interrupted in.

State includes: open positions and their entry prices, stop levels managed by the risk layer, pending order IDs, the current regime classification, the active strategy mode, the portfolio high-water mark for the circuit breaker, and the timestamp of the last successful trade.

**This layer is deployed on EC2.** It runs at the end of every main loop iteration.

---

## What This Layer Is Trying to Achieve

1. Ensure a crash or restart loses zero meaningful context
2. Provide a startup reconciliation foundation so the bot can verify its state against the exchange before resuming
3. Give you a human-readable audit trail you can inspect after any incident
4. Keep write operations simple and atomic enough that a mid-write crash doesn't corrupt the state file

---

## How It Contributes to the Bigger Picture

Every other layer operates in memory. This layer is the bridge between the current process lifetime and the next. Without it, every restart is a cold start: the bot doesn't know it holds BTC, doesn't know its stop levels, doesn't know its portfolio high-water mark, and can't distinguish between "I have no positions" and "I just crashed mid-trade."

In a 24/7 competition system, crashes are not an edge case — they are a certainty over a multi-week timeframe. This layer is the difference between a crash being a brief interruption and a crash being a catastrophic loss event.

---

## Files in This Layer

```
persistence/
└── state.py    State serialisation, deserialisation, atomic write
```

---

## State Schema

The complete `state.json` structure:

```json
{
  "version": 1,
  "written_at": 1712345678.123,
  "written_at_iso": "2025-04-05T12:34:38.123Z",

  "oms": {
    "positions": {
      "BTC/USD": {
        "pair": "BTC/USD",
        "quantity": 0.052341,
        "entry_price": 68420.50,
        "entry_time": 1712340000.0,
        "usd_value": 3581.04
      }
    }
  },

  "risk": {
    "trailing_stops": {
      "BTC/USD": 65800.00
    },
    "entry_prices": {
      "BTC/USD": 68420.50
    },
    "portfolio_hwm": 52400.00,
    "circuit_breaker_active": false
  },

  "regime": {
    "current_regime": "BULL_TREND",
    "last_detected_at": 1712340000.0
  },

  "strategy": {
    "active_strategy": "momentum",
    "last_signal_time": 1712340000.0,
    "last_candle_boundary": {
      "BTC/USD": 1712332800,
      "ETH/USD": 1712332800
    }
  },

  "execution": {
    "last_trade_time": 1712340000.0,
    "last_reconciliation_time": 1712341800.0
  }
}
```

---

## `persistence/state.py`

```python
import json
import os
import time
import logging
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STATE_FILE = Path("state.json")
STATE_BACKUP_FILE = Path("state.json.bak")

class StateManager:
    """
    Atomic read/write of bot state to disk.
    
    Write strategy:
    1. Serialise state to JSON string
    2. Write to state.json.tmp
    3. Atomically rename tmp → state.json
    
    This ensures state.json is always either the previous complete write
    or the new complete write — never a partial write.
    """

    VERSION = 1

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.backup_file = state_file.with_suffix(".json.bak")
        self.tmp_file = state_file.with_suffix(".json.tmp")

    def write(self, oms_state: dict, risk_state: dict, regime_state: dict,
              strategy_state: dict, execution_state: dict) -> bool:
        """
        Write complete bot state atomically.
        Returns True on success, False on failure (non-fatal — log and continue).
        """
        state = {
            "version": self.VERSION,
            "written_at": time.time(),
            "written_at_iso": datetime.now(timezone.utc).isoformat(),
            "oms": oms_state,
            "risk": risk_state,
            "regime": regime_state,
            "strategy": strategy_state,
            "execution": execution_state,
        }

        try:
            json_str = json.dumps(state, indent=2, default=self._json_serialiser)

            # Write to temp file first
            self.tmp_file.write_text(json_str, encoding="utf-8")

            # Backup existing state before overwriting
            if self.state_file.exists():
                import shutil
                shutil.copy2(self.state_file, self.backup_file)

            # Atomic rename: tmp → state.json
            os.replace(self.tmp_file, self.state_file)

            logger.debug(f"State written: {len(json_str)} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to write state: {e}")
            return False

    def read(self) -> dict:
        """
        Read state from disk.
        Falls back to backup if primary is corrupted.
        Returns empty dict if neither file exists (fresh start).
        """
        for file in [self.state_file, self.backup_file]:
            if not file.exists():
                continue
            try:
                content = file.read_text(encoding="utf-8")
                state = json.loads(content)
                if state.get("version") != self.VERSION:
                    logger.warning(f"State version mismatch: {state.get('version')} != {self.VERSION}")
                logger.info(f"State loaded from {file}: written at {state.get('written_at_iso')}")
                return state
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse {file}: {e}. Trying backup...")
                continue

        logger.info("No valid state file found — fresh start")
        return {}

    def get_age_seconds(self) -> float:
        """How old is the current state file? Used to detect stale state."""
        if not self.state_file.exists():
            return float("inf")
        state = self.read()
        written_at = state.get("written_at", 0)
        return time.time() - written_at

    @staticmethod
    def _json_serialiser(obj: Any) -> Any:
        """Handle non-serialisable types."""
        if hasattr(obj, "value"):
            return obj.value  # Enum
        if hasattr(obj, "__dict__"):
            return obj.__dict__  # Dataclass fallback
        raise TypeError(f"Not serialisable: {type(obj)}")
```

---

## Integration in `main.py`

State is written at the end of every loop iteration, unconditionally:

```python
# At end of every main loop cycle:
state_manager.write(
    oms_state=oms.dump_state(),
    risk_state=risk_manager.dump_state(),
    regime_state={
        "current_regime": current_regime.regime.value,
        "last_detected_at": time.time(),
    },
    strategy_state={
        "active_strategy": active_strategy.__class__.__name__ if active_strategy else "none",
        "last_signal_time": last_signal_time,
        "last_candle_boundary": fetcher.get_candle_boundaries(),
    },
    execution_state={
        "last_trade_time": rate_limiter._last_trade_time,
        "last_reconciliation_time": oms._last_reconciliation,
    },
)
```

---

## Startup Sequence Using State

```python
def startup(state_manager: StateManager, oms, risk_manager,
            regime_detector, fetcher, client) -> bool:
    """
    Full startup sequence. Returns True if safe to begin trading.
    """
    logger.info("Bot starting up...")

    # 1. Check state file age
    age = state_manager.get_age_seconds()
    if age < 3600:
        logger.info(f"Loading recent state ({age:.0f}s old)")
        state = state_manager.read()
    elif age < 86400:
        logger.warning(f"State is {age/3600:.1f} hours old — loading with caution")
        state = state_manager.read()
    else:
        logger.warning("State is more than 24 hours old or missing — cold start")
        state = {}

    # 2. Restore component state from disk
    oms.load_state(state.get("oms", {}))
    risk_manager.load_state(state.get("risk", {}))

    # 3. Live reconciliation against exchange
    # (Full reconciliation implementation in Layer 7)
    live_balance = client.get_balance()
    if not live_balance.get("Success"):
        logger.critical("Cannot get balance — aborting startup")
        return False

    # 4. Recompute stop levels if state is stale
    if age > 300 and state.get("oms", {}).get("positions"):
        logger.info("State is >5 min old — recalculating stop levels from current price")
        # Re-derive ATR trailing stops from current ATR
        # (Implementation depends on feature availability at startup)

    # 5. Seed live data buffer
    # (Live fetcher pre-seeded from Parquet, see Layer 2)

    # 6. Final check: portfolio value sanity
    free_usd = live_balance["Wallet"].get("USD", {}).get("Free", 0)
    logger.info(f"Free USD: ${free_usd:,.2f} | Positions: {list(oms.get_all_positions().keys())}")

    # 7. Initialize risk HWM from current portfolio value
    total_portfolio = free_usd + sum(
        oms.get_position_value(pair) for pair in oms.get_all_positions()
    )
    risk_manager.initialize_hwm(
        state.get("risk", {}).get("portfolio_hwm", total_portfolio)
    )

    logger.info("Startup complete — ready to trade")
    return True
```

---

## What Gets Persisted and Why

| State Item | Why It Must Be Persisted |
|---|---|
| Open positions (pair, quantity, entry_price) | Without this, bot doesn't know it holds crypto on restart |
| ATR trailing stop levels | Without this, stops reset to entry on restart, potentially at wrong levels |
| Hard stop entry prices | Required to recalculate hard percentage stop |
| Portfolio high-water mark | Without this, circuit breaker resets on restart, allowing fresh losses |
| Circuit breaker status | If CB was active, must remain active until recovery |
| Last candle boundary per pair | Without this, bot may re-evaluate signals mid-candle on restart |
| Last trade time | Rate limiter needs this to enforce cooldown across restarts |
| Active regime | Context for the strategy selector on restart |

---

## What State Does NOT Track

- Full order history: this lives in `trades.log` (Layer 9) and the exchange itself
- Feature values or indicators: these are always recomputed from fresh data
- Strategy internal counters: strategies are stateless (see Layer 5)
- API keys or configuration: these come from `.env` and `config.yaml`

---

## Failure Modes This Layer Prevents

| Failure | Prevention |
|---|---|
| Crash causes double entry on restart | Positions persisted; reconciliation detects any discrepancy |
| Stop levels reset incorrectly after restart | Trailing and hard stops persisted and restored exactly |
| Circuit breaker resets on crash | CB state and HWM persisted; cannot be circumvented by restarting |
| Mid-write crash corrupts state file | Atomic write: tmp → rename; backup preserved |
| Stale state misrepresenting positions | State age check; reconciliation against live balance on startup |
| Strategy re-entering signals already traded | Candle boundaries and last signal times persisted |
