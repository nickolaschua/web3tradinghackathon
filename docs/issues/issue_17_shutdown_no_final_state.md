# Issue 17: Shutdown Handler Does Not Write Final State

## Layer
Layer 10 — Orchestration (`main.py`)

## Description
The signal handler for SIGTERM/SIGINT in the main loop does not write the current state to `state.json` before exiting. When systemd stops the service (e.g. for a deployment), the last written state may be from up to 60 seconds before shutdown (the previous main loop cycle).

If the bot placed a trade in the final 60 seconds before shutdown, the state.json will not contain that trade, and on restart the bot will not know it has an open position.

## Code Location
`main.py` → signal handler / shutdown sequence (Layer 10 design docs)

## Fix Required
The shutdown handler must call `state_manager.write(...)` before returning:
```python
def _handle_shutdown(signum, frame):
    logger.info(f"Signal {signum} received — writing final state and shutting down")
    state_manager.write(
        oms_state=oms.dump_state(),
        risk_state=risk_manager.dump_state(),
        ...
    )
    sys.exit(0)
```

## Impact
**High** — any trade placed in the last ~60 seconds before a planned shutdown will be "invisible" on restart. The reconciliation on startup should catch this, but only if reconciliation is implemented (see Issue 16).
