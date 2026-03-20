# Issue 02: Rate Limiter Holds Lock for Entire 65-Second Sleep

## Layer
Layer 1 — API Client (`api/rate_limiter.py`)

## Description
The `RateLimiter.wait_if_needed()` method acquires a threading lock and then calls `time.sleep()` for up to 65 seconds while the lock is held. This means any other thread that needs to check the rate limiter (e.g. a stop-loss exit trying to place a SELL order) will block for up to 65 seconds waiting for the lock to release.

In the current single-threaded architecture this is not immediately fatal, but if the main loop ever becomes multi-threaded (or if emergency stop logic runs in a signal handler), this will cause deadlocks.

## Code Location
`api/rate_limiter.py` → `wait_if_needed()` method

## Reproduction
1. Bot places a BUY order
2. 30 seconds later, ATR stop fires
3. Stop-exit SELL calls `wait_if_needed()`, blocks 35 more seconds waiting for the 65s cooldown lock
4. Stop is effectively delayed by up to 65 seconds

## Fix Required
Release the lock before sleeping:
```python
with self._lock:
    wait_time = max(0, self._min_interval - elapsed)
# Lock released BEFORE sleeping
if wait_time > 0:
    time.sleep(wait_time)
```

## Impact
**Medium** — in single-threaded bot, causes delayed stop exits. In any multi-threaded scenario, causes deadlock.
