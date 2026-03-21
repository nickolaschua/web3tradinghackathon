"""
Universe-level feature computation.

Functions here require all coins simultaneously and cannot live in
compute_features() (which only sees one coin at a time).

All rank/zscore columns are shifted 1 bar before injection — identical
convention to compute_features().
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cross_sectional_ranks(
    coin_dfs: dict[str, pd.DataFrame],
    lookbacks: list[int] = [42, 84, 168],   # 7d, 14d, 28d in 4H bars
) -> dict[str, pd.DataFrame]:
    """
    Compute cross-sectional return rank features for each coin.

    For each lookback, ranks each coin's cumulative log return as a percentile
    (0-1) within the universe at each bar, and also computes a z-score across
    the universe. Both columns are shifted 1 bar to prevent look-ahead bias.

    Args:
        coin_dfs: Dict mapping pair symbol -> OHLCV DataFrame.
                  All DataFrames should share a compatible DatetimeIndex.
                  Column names must be lowercase (run df.columns.str.lower() first).
        lookbacks: Bar lookback windows. 42 bars = 7d at 4H cadence.

    Returns:
        Same dict as input but with new columns per lookback injected into
        each DataFrame:
          ret_{n}bar_rank    -- percentile rank within universe (0=worst, 1=best)
          ret_{n}bar_zscore  -- z-score within universe at each bar

    Notes:
        - NaN propagates naturally through rank: a coin with missing data at a
          timestamp simply gets NaN rank for that bar, which dropna() removes later.
        - The universe must be identical in train and live to avoid rank distribution
          shift (see research/iteration_log.md, Iteration 2).
    """
    # Step 1: compute cumulative log returns for each coin at each lookback
    return_matrices: dict[int, pd.DataFrame] = {}
    for n in lookbacks:
        ret_matrix = pd.DataFrame(
            {
                pair: np.log(df["close"] / df["close"].shift(n))
                for pair, df in coin_dfs.items()
                if not df.empty and "close" in df.columns
            }
        )
        return_matrices[n] = ret_matrix

    # Step 2: rank and z-score across coins at each timestamp
    rank_matrices: dict[int, pd.DataFrame] = {}
    zscore_matrices: dict[int, pd.DataFrame] = {}
    for n, ret_matrix in return_matrices.items():
        rank_matrices[n] = ret_matrix.rank(axis=1, pct=True)
        mean = ret_matrix.mean(axis=1)
        std  = ret_matrix.std(axis=1)
        zscore_matrices[n] = ret_matrix.sub(mean, axis=0).div(std + 1e-10, axis=0)

    # Step 3: inject back into per-coin DataFrames with 1-bar shift
    result: dict[str, pd.DataFrame] = {}
    for pair, df in coin_dfs.items():
        out = df.copy()
        for n in lookbacks:
            if pair in rank_matrices[n].columns:
                out[f"ret_{n}bar_rank"]   = rank_matrices[n][pair].shift(1)
                out[f"ret_{n}bar_zscore"] = zscore_matrices[n][pair].shift(1)
        result[pair] = out

    return result


def compute_universe_spread(
    coin_dfs: dict[str, pd.DataFrame],
    lookback: int = 42,
) -> pd.Series:
    """
    Compute the return spread between top and bottom tercile of the universe.

    High spread = cross-sectional momentum is 'hot' (CS signals more reliable).
    Low spread  = universe is highly correlated (CS signals noisier).

    Returns a Series aligned to the union of coin indices, shifted 1 bar.
    Inject into every coin's feature DataFrame as a regime indicator.
    """
    ret_matrix = pd.DataFrame(
        {
            pair: np.log(df["close"] / df["close"].shift(lookback))
            for pair, df in coin_dfs.items()
            if not df.empty and "close" in df.columns
        }
    )
    top = ret_matrix.apply(lambda row: row[row >= row.quantile(0.67)].mean(), axis=1)
    bot = ret_matrix.apply(lambda row: row[row <= row.quantile(0.33)].mean(), axis=1)
    return (top - bot).shift(1).rename(f"universe_spread_{lookback}bar")


__all__ = ["compute_cross_sectional_ranks", "compute_universe_spread"]
