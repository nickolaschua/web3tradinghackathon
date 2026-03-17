#!/usr/bin/env python3
"""
Historical OHLCV data downloader for BTC/ETH/SOL 4H candles from Binance.

Downloads from Binance public klines API and saves as Parquet with UTC DatetimeIndex.
No API key required (public endpoint).

Usage:
  python scripts/download_data.py --start 2022-01-01 --end 2024-03-17 --output-dir data
  python scripts/download_data.py --help
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Binance public klines API endpoint (no auth required)
BINANCE_API_BASE = "https://api.binance.com/api/v3"

# Symbols to download (Binance format)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Interval (4-hour candles)
INTERVAL = "4h"

# API rate limiting
REQUEST_SLEEP = 0.1  # seconds between requests (very conservative; limit is 1200 weight/min)
BATCH_SIZE = 1000  # max rows per API call


def timestamp_to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD to milliseconds since epoch (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def download_klines(symbol: str, start_ms: int, end_ms: int, max_retries: int = 3) -> list:
    """
    Download klines for a symbol from start_ms to end_ms.

    Returns list of [open_time, open, high, low, close, volume, ...] rows.
    Handles pagination automatically with retry logic.
    """
    all_rows = []
    current_start_ms = start_ms

    while current_start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current_start_ms,
            "endTime": end_ms,
            "limit": BATCH_SIZE,
        }

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    f"{BINANCE_API_BASE}/klines",
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
                batch = resp.json()
                break
            except (requests.RequestException, requests.Timeout) as e:
                if attempt == max_retries - 1:
                    raise
                backoff = 2 ** attempt
                print(f"    Retry {attempt + 1}/{max_retries} after {backoff}s (error: {e})")
                time.sleep(backoff)

        if not batch:
            break  # No more data

        all_rows.extend(batch)

        print(f"  {symbol}: {len(all_rows)} rows fetched...")

        if len(batch) < BATCH_SIZE:
            # Last batch was incomplete — we've reached the end
            break

        # Advance to next batch: last_row_open_time + 1ms
        current_start_ms = int(batch[-1][0]) + 1

        # Rate limit: sleep before next request
        time.sleep(REQUEST_SLEEP)

    return all_rows


def klines_to_dataframe(rows: list, symbol: str) -> pd.DataFrame:
    """
    Convert Binance klines response to a DataFrame.

    Args:
        rows: List of [open_time, open, high, low, close, volume, ...] rows
        symbol: Symbol name (for error messages)

    Returns:
        DataFrame with columns [open, high, low, close, volume] and UTC DatetimeIndex
    """
    if not rows:
        raise ValueError(f"No data downloaded for {symbol}")

    # Extract relevant columns from Binance response
    # Index 0: open_time (ms)
    # Index 1-5: open, high, low, close, volume
    open_times = [int(row[0]) for row in rows]
    opens = [float(row[1]) for row in rows]
    highs = [float(row[2]) for row in rows]
    lows = [float(row[3]) for row in rows]
    closes = [float(row[4]) for row in rows]
    volumes = [float(row[7]) for row in rows]  # Quote asset volume at index 7

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # Set UTC DatetimeIndex from open_time (milliseconds)
    df.index = pd.to_datetime(open_times, unit="ms", utc=True)
    df.index.name = "timestamp"

    # Ensure all columns are float64
    for col in df.columns:
        df[col] = df[col].astype("float64")

    return df


def download_and_save(
    symbol: str,
    start_ms: int,
    end_ms: int,
    output_dir: Path,
) -> None:
    """Download klines for a symbol and save as Parquet."""
    try:
        print(f"Downloading {symbol} 4H candles...")
        rows = download_klines(symbol, start_ms, end_ms)

        df = klines_to_dataframe(rows, symbol)

        output_path = output_dir / f"{symbol}_4h.parquet"
        df.to_parquet(output_path)

        print(f"{symbol}: {len(df)} rows saved to {output_path}")

    except Exception as e:
        print(f"ERROR: {symbol} download failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance 4H OHLCV data for BTC/ETH/SOL and save as Parquet."
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Start date (YYYY-MM-DD, default 2022-01-01)",
    )
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD, default today)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default 'data')",
    )

    args = parser.parse_args()

    # Parse dates
    start_ms = timestamp_to_ms(args.start)
    end_ms = timestamp_to_ms(args.end)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {', '.join(SYMBOLS)} from {args.start} to {args.end}")
    print(f"Output directory: {output_dir}")
    print()

    # Download each symbol
    for symbol in SYMBOLS:
        download_and_save(symbol, start_ms, end_ms, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
