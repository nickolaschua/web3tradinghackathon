"""
Fetch Binance perpetual futures funding rates (public endpoint — no auth).

Used to build sentiment features for the spot XGBoost model.
Funding rates settle every 8H (00:00, 08:00, 16:00 UTC).

NOTE: these are informational features only. The Roostoo API is spot-only;
we are not executing carry trades or interacting with perpetual markets.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"


def fetch_funding_rates_paginated(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch all funding rate records for a symbol between start_ms and end_ms.

    Paginates automatically: each Binance call returns up to 1000 records
    (1000 × 8h ≈ 333 days). Calls are chained using the last record's
    fundingTime as the next startTime until end_ms is reached.

    Args:
        symbol:    Binance USDT-M symbol, e.g. "BTCUSDT".
        start_ms:  Start timestamp in milliseconds (UTC).
        end_ms:    End timestamp in milliseconds (UTC).
        limit:     Max records per API call (default 1000, max 1000).

    Returns:
        DataFrame with columns: fundingTime (datetime UTC), fundingRate (float).
        Empty DataFrame on error or no data.
    """
    all_records: list[dict] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Funding rate fetch failed for %s: %s", symbol, exc)
            break

        if not data:
            break

        all_records.extend(data)
        last_time = int(data[-1]["fundingTime"])

        if len(data) < limit:
            break  # last page

        current_start = last_time + 1  # advance 1 ms past last record
        time.sleep(0.1)  # polite rate limiting

    if not all_records:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    df = pd.DataFrame(all_records)[["fundingTime", "fundingRate"]]
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return (
        df.drop_duplicates("fundingTime")
        .sort_values("fundingTime")
        .reset_index(drop=True)
    )


def load_or_fetch_funding(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return funding rate history for symbol between start_date and end_date.

    Loads from a parquet cache if the cached data already covers the requested
    range; otherwise fetches from Binance and saves to cache.

    Args:
        symbol:     Binance symbol (e.g. "BTCUSDT").
        start_date: ISO date string, inclusive (e.g. "2021-03-01").
        end_date:   ISO date string, exclusive upper bound (e.g. "2026-04-01").
        cache_dir:  Directory for parquet cache. Defaults to project data/funding/.

    Returns:
        DataFrame with columns: fundingTime (datetime UTC), fundingRate (float).
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "funding"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}_funding.parquet"

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts   = pd.Timestamp(end_date, tz="UTC")

    # Try cache first
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["fundingTime"] = pd.to_datetime(cached["fundingTime"], utc=True)
        cached_start = cached["fundingTime"].min()
        cached_end   = cached["fundingTime"].max()
        needs_refresh = (
            cached_start > start_ts
            or cached_end < end_ts - pd.Timedelta(days=1)
        )
        if not needs_refresh:
            logger.info(
                "Funding cache hit: %s — %d records (%s to %s)",
                symbol, len(cached), cached_start.date(), cached_end.date(),
            )
            return cached[
                (cached["fundingTime"] >= start_ts)
                & (cached["fundingTime"] <= end_ts)
            ].reset_index(drop=True)
        logger.info("Cache incomplete for %s — refetching from Binance.", symbol)

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms   = int(end_ts.timestamp()   * 1000)
    df = fetch_funding_rates_paginated(symbol, start_ms, end_ms)

    if not df.empty:
        df.to_parquet(cache_path, index=False)
        logger.info("Saved %d funding records to %s", len(df), cache_path.name)

    return df


__all__ = ["fetch_funding_rates_paginated", "load_or_fetch_funding"]
