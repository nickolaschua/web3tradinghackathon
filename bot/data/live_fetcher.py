"""LiveFetcher: seed from Binance Parquet history + live Roostoo ticker polling."""
from __future__ import annotations

import time
from collections import deque
from typing import Dict

import numpy as np
import pandas as pd

from bot.data.features import (
    compute_features,
    compute_cross_asset_features,
    compute_btc_context_features,
)


class LiveFetcher:
    """
    Maintains per-pair OHLCV ring buffers seeded from historical Parquet data
    and extended via completed 4H epoch candles.

    IMPORTANT — two distinct data flows:
    - _buffers: completed 4H candles only (seeded from Parquet, extended via
      append_epoch_candle() at each 4H boundary). Used for ALL feature computation.
    - _last_prices: latest live price from ticker polls (updated by poll_ticker()).
      Used only for current-price lookups (stop checks, position sizing).

    poll_ticker() does NOT write to _buffers. This is intentional: adding 60-second
    tick "candles" to the feature buffer would cause RSI/MACD/EMA to be computed on
    minute-scale bars rather than 4H bars, completely invalidating model predictions.

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
        # Completed 4H candle buffer — seeded from Parquet, extended at epoch boundaries.
        # This is the ONLY buffer used for feature computation.
        self._buffers: Dict[str, deque] = {
            pair: deque(maxlen=maxlen) for pair in seed_dfs
        }
        # Latest live price from ticker polls — separate from feature buffer.
        self._last_prices: Dict[str, float] = {}
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
        tail = df.tail(self._maxlen)
        for ts, row in tail.iterrows():
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

        # Seed _last_prices so get_latest_price() works immediately after init
        if not tail.empty:
            self._last_prices[symbol] = float(tail.iloc[-1]["close"])

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
        Record the latest live price from a Roostoo ticker poll.

        IMPORTANT: this method does NOT write to _buffers. Writing 60-second
        tick prices into the feature buffer would corrupt RSI/MACD/EMA — those
        indicators are trained on 4H bars and must only ever see 4H candles.

        Use append_epoch_candle() at 4H boundaries to extend _buffers.

        Args:
            pair: Roostoo pair symbol (e.g. "BTC/USD").
            last_price: The LastPrice field from /v3/ticker response.
        """
        self._last_prices[pair] = float(last_price)

    def append_epoch_candle(
        self,
        pair: str,
        price: float,
        timestamp: int | None = None,
    ) -> None:
        """
        Append a completed 4H epoch candle to the feature buffer for `pair`.

        Called by main.py exactly once per 4H boundary, after all feature pairs
        have been polled. Uses the last known ticker price as a close-price proxy
        (O=H=L=C=price, volume=0 — same limitation as Roostoo has no OHLCV).

        Args:
            pair: Roostoo pair symbol (e.g. "BTC/USD").
            price: Close price for this completed 4H bar (from _last_prices).
            timestamp: Unix seconds for the candle. Defaults to now if omitted.
        """
        if pair not in self._buffers:
            self._buffers[pair] = deque(maxlen=self._maxlen)

        ts = timestamp if timestamp is not None else int(time.time())
        self._buffers[pair].append({
            "open":      price,
            "high":      price,
            "low":       price,
            "close":     price,
            "volume":    0.0,
            "timestamp": ts,
        })

    def get_latest_price(self, pair: str) -> float:
        """
        Return the most recent live price for `pair`.

        Reads from _last_prices (updated by poll_ticker every 60 s), NOT from
        the 4H candle buffer. This gives a sub-4H current price for stop-loss
        and position sizing without polluting the feature buffer.

        Falls back to the last candle close if poll_ticker has not yet been
        called (e.g. immediately after seeding from history).

        Raises:
            KeyError: If pair is unknown (never seeded or polled).
        """
        if pair in self._last_prices:
            return self._last_prices[pair]
        # Fallback: use last historical close (seeded in _seed_from_history)
        if pair not in self._buffers or not self._buffers[pair]:
            raise KeyError(
                f"No price data for pair '{pair}'. Known pairs: {list(self._buffers)}"
            )
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

        # Step 2.5: 15M-specific features for BTC/USD (shift-16=4H, shift-96=1D returns
        # and rolling 2880-bar corr/beta). Mirrors prepare_features() in train_model_15m.py.
        if pair == "BTC/USD":
            eth_raw = other_dfs.get("ETH/USD", pd.DataFrame())
            sol_raw = other_dfs.get("SOL/USD", pd.DataFrame())
            for prefix, raw_df in [("eth", eth_raw), ("sol", sol_raw)]:
                if not raw_df.empty:
                    log_ret = np.log(raw_df["close"] / raw_df["close"].shift(1))
                    df[f"{prefix}_return_4h"] = log_ret.shift(16).reindex(df.index)
                    df[f"{prefix}_return_1d"] = log_ret.shift(96).reindex(df.index)
            if not eth_raw.empty and not sol_raw.empty:
                df = compute_btc_context_features(df, eth_raw, sol_raw, window=2880)

        # Step 3: Drop warmup rows (NaN from rolling windows and shift)
        df = df.dropna()

        return df
