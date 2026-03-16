"""
Global rate limiter and trade cooldown for RoostooClient.

CRITICAL: threading.Lock is ALWAYS released before time.sleep() to prevent
blocking emergency SELL orders during a rate-limit wait (Issue 02).
"""
import threading
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """30-call/minute sliding-window rate limiter.

    The lock is released BEFORE sleep so other threads (e.g. emergency SELL)
    can check the window without waiting for the sleeping thread to wake.
    """

    def __init__(self, max_calls: int = 30, window_secs: float = 60.0) -> None:
        self._max_calls = max_calls
        self._window = window_secs
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a rate-limit slot is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps that have fallen outside the rolling window
                while self._timestamps and self._timestamps[0] < now - self._window:
                    self._timestamps.popleft()

                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return  # slot acquired; lock released by exiting `with` block

                # Window is full — compute how long to sleep WHILE still holding lock
                sleep_until = self._timestamps[0] + self._window
                sleep_for = sleep_until - now

            # LOCK IS RELEASED HERE (exited `with` block) — sleep outside the lock
            if sleep_for > 0:
                logger.debug("rate limiter: sleeping %.2fs (window full)", sleep_for)
                time.sleep(sleep_for)
            # Loop: re-check after waking (another thread may have taken the slot)


class TradeCooldown:
    """65-second minimum gap between place_order calls.

    Layered on top of RateLimiter — does not count against the 30/min window.
    Lock released before sleep for the same reason as RateLimiter.
    """

    def __init__(self, cooldown_secs: float = 65.0) -> None:
        self._cooldown = cooldown_secs
        self._last_trade = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until cooldown_secs have elapsed since the last trade."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_trade
                if elapsed >= self._cooldown:
                    self._last_trade = now
                    return  # cooldown satisfied; lock released
                sleep_for = self._cooldown - elapsed

            # Lock released before sleep
            logger.debug("trade cooldown: sleeping %.2fs", sleep_for)
            time.sleep(sleep_for)
            # Loop: re-check (defensive against clock drift / spurious wakeups)


# Module-level singletons — one shared window across all RoostooClient instances
_rate_limiter = RateLimiter()
_trade_cooldown = TradeCooldown()
