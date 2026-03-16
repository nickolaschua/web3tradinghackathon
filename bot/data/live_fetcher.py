"""LiveFetcher: seed from Binance Parquet history + live Roostoo ticker polling."""
from __future__ import annotations

import time
from collections import deque
from typing import Dict

import numpy as np
import pandas as pd

from bot.data.features import compute_features, compute_cross_asset_features


class LiveFetcher:
    """
    Maintains per-pair OHLCV ring buffers seeded from historical Parquet data
    and updated via live Roostoo ticker polls.

    Args:
        seed_dfs: Dict mapping Roostoo pair symbols (e.g. "BTC/USD") to
                  historical DataFrames with columns: open, high, low, close, volume.
                  Keys MUST be in Roostoo format ("BTC/USD", "ETH/USD", "SOL/USD"),
                  NOT Binance format ("BTCUSDT").
        maxlen: Maximum candles to retain per pair (default 500).
    """

    def __init__(
        self,
        seed_dfs: Dict[str, pd.DataFrame],
        maxlen: int = 500,
    ) -> None:
        self._maxlen = maxlen
        # Initialize one deque per pair — O(1) append, auto-drops oldest on overflow
        self._buffers: Dict[str, deque] = {
            pair: deque(maxlen=maxlen) for pair in seed_dfs
        }
        # Seed each buffer from historical data
        for pair, df in seed_dfs.items():
            self._seed_from_history(pair, df)

    def _seed_from_history(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Load historical Parquet data into the ring buffer for `symbol`.

        CRITICAL: lowercase all column names immediately — Binance raw CSVs use
        capitalized headers (Open, High, Low, Close, Volume); Parquet conversion
        may or may not have renamed them. Lowercase guard costs nothing and
        prevents an 8-hour startup failure on column KeyError.

        Args:
            symbol: Roostoo pair symbol (e.g. "BTC/USD").
            df: DataFrame with OHLCV columns (any capitalization).
        """
        df = df.copy()
        df.columns = df.columns.str.lower()  # CRITICAL: Binance CSVs are capitalized

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Seed DataFrame for {symbol} is missing columns: {missing}. "
                f"Got: {list(df.columns)}"
            )

        # Load last maxlen rows only — avoids holding 5000 rows in memory
        for ts, row in df.tail(self._maxlen).iterrows():
            timestamp = (
                int(ts.timestamp())
                if hasattr(ts, "timestamp")
                else int(ts)
            )
            self._buffers[symbol].append({
                "open":      float(row["open"]),
                "high":      float(row["high"]),
                "low":       float(row["low"]),
                "close":     float(row["close"]),
                "volume":    float(row["volume"]),
                "timestamp": timestamp,
            })

    def _to_dataframe(self, pair: str) -> pd.DataFrame:
        """
        Convert the ring buffer for `pair` to a DataFrame with DatetimeIndex.

        Returns an empty DataFrame with correct columns if buffer is empty.
        """
        if pair not in self._buffers or not self._buffers[pair]:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        rows = list(self._buffers[pair])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.pop("timestamp"), unit="s", utc=True)
        df.index.name = "timestamp"
        return df

    def poll_ticker(self, pair: str, last_price: float) -> None:
        """
        Append a synthetic candle from a Roostoo ticker poll.

        Roostoo /v3/ticker returns only LastPrice — no OHLCV. All four price
        fields are set to last_price; volume is 0. This is intentional and
        expected. Do NOT try to synthesize H/L from multiple polls — that adds
        complexity without improving feature quality for the close-to-close ATR proxy.

        Args:
            pair: Roostoo pair symbol (e.g. "BTC/USD"). Created in buffer if absent.
            last_price: The LastPrice field from /v3/ticker response.
        """
        if pair not in self._buffers:
            self._buffers[pair] = deque(maxlen=self._maxlen)

        self._buffers[pair].append({
            "open":      last_price,
            "high":      last_price,   # H = L = O = C — Roostoo has no OHLCV
            "low":       last_price,
            "close":     last_price,
            "volume":    0.0,          # Roostoo provides no volume
            "timestamp": int(time.time()),
        })

    def get_latest_price(self, pair: str) -> float:
        """
        Return the most recent close price for `pair`.

        Required by main.py for position sizing and stop-loss calculations.

        Raises:
            KeyError: If pair has no buffer.
            ValueError: If buffer is empty (not yet seeded or polled).
        """
        if pair not in self._buffers:
            raise KeyError(f"No buffer for pair '{pair}'. Known pairs: {list(self._buffers)}")
        if not self._buffers[pair]:
            raise ValueError(f"Buffer for '{pair}' is empty — not yet seeded or polled.")
        return float(self._buffers[pair][-1]["close"])

    def get_candle_boundaries(self) -> Dict[str, int]:
        """
        Return the last candle close timestamp (Unix seconds) for each pair.

        Required by main.py to detect when a new 4H candle has completed.
        Returns 0 for pairs with empty buffers.

        Returns:
            Dict mapping pair symbol → last candle timestamp (Unix seconds, int).
            Example: {"BTC/USD": 1742000000, "ETH/USD": 1742000000}
        """
        return {
            pair: (buf[-1]["timestamp"] if buf else 0)
            for pair, buf in self._buffers.items()
        }

    def __repr__(self) -> str:
        summary = {p: len(b) for p, b in self._buffers.items()}
        return f"LiveFetcher(buffers={summary}, maxlen={self._maxlen})"

    def is_warmed_up(self, pair: str = "BTC/USD") -> bool:
        """
        Return True when the buffer for `pair` has at least 35 candles.

        Threshold: MACD(12,26,9) requires 35 bars to stabilise. Below this
        threshold, indicator values are unreliable and should not drive trades.

        Args:
            pair: Pair to check (defaults to "BTC/USD", the primary tradeable pair).

        Returns:
            True if buffer contains >= 35 candles, False otherwise.
        """
        return len(self._buffers.get(pair, [])) >= 35

    def get_feature_matrix(self, pair: str = "BTC/USD") -> pd.DataFrame:
        """
        Return the fully-processed feature matrix for `pair`, ready for strategy input.

        Ordering is CRITICAL and must not be changed:
        1. compute_features(df)                          — indicators + shift(1)
        2. compute_cross_asset_features(df, other_dfs)  — ETH/SOL lag columns injected
        3. df.dropna()                                   — warmup rows removed LAST

        If dropna() runs before step 2, the ETH/SOL columns do not yet exist and
        pandas silently drops every row, returning an empty DataFrame.

        Returns an empty DataFrame (same columns, 0 rows) if not yet warmed up.
        Callers should check is_warmed_up() before calling this method.

        Args:
            pair: Primary pair to build feature matrix for (default "BTC/USD").

        Returns:
            DataFrame with OHLCV + indicator + cross-asset columns, no NaN rows.
            DatetimeIndex aligned to pair's candle boundaries.
        """
        if not self.is_warmed_up(pair):
            return pd.DataFrame()

        # Step 1: Raw OHLCV → compute indicators → shift(1)
        df = self._to_dataframe(pair)
        df = compute_features(df)

        # Step 2: Inject cross-asset lag features from other pairs in buffer
        # MUST be before dropna — otherwise empty-column dropna removes all rows
        other_dfs = {
            p: self._to_dataframe(p)
            for p in self._buffers
            if p != pair
        }
        df = compute_cross_asset_features(df, other_dfs)

        # Step 3: Drop warmup rows (NaN from rolling windows and shift)
        df = df.dropna()

        return df
