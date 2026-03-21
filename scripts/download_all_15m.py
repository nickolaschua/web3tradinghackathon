#!/usr/bin/env python3
"""
Download 15M OHLCV data for all 39 tradeable coins from Binance.

Usage (run locally, NOT in sandboxed env):
    python scripts/download_all_15m.py
    python scripts/download_all_15m.py --start 2022-01-01
    python scripts/download_all_15m.py --symbols DOGEUSDT LINKUSDT  # specific coins only

Saves to data/{SYMBOL}_15m.parquet
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT",
    "DOGEUSDT", "LINKUSDT", "DOTUSDT", "UNIUSDT", "XRPUSDT", "LTCUSDT",
    "AAVEUSDT", "CRVUSDT", "NEARUSDT", "FILUSDT", "FETUSDT", "HBARUSDT",
    "ZECUSDT", "ZENUSDT", "CAKEUSDT", "PAXGUSDT", "XLMUSDT", "TRXUSDT",
    "CFXUSDT", "SHIBUSDT", "ICPUSDT", "APTUSDT", "ARBUSDT", "SUIUSDT",
    "FLOKIUSDT", "PEPEUSDT", "PENDLEUSDT", "WLDUSDT", "SEIUSDT",
    "BONKUSDT", "WIFUSDT", "ENAUSDT", "TAOUSDT",
]

BASE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path("data")


def download_15m(symbol: str, start: str, end_ms: int) -> pd.DataFrame:
    """Download all 15M klines for a single symbol from Binance."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        for attempt in range(3):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 30))
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                batch = resp.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    ERROR after 3 retries: {e}")
                    return _rows_to_df(all_rows) if all_rows else pd.DataFrame()
                time.sleep(2 ** attempt)
                continue
        else:
            break

        if not batch:
            break

        all_rows.extend(batch)
        if len(batch) < 1000:
            break
        current_start = int(batch[-1][0]) + 1
        time.sleep(0.12)  # ~8 req/s, well under Binance 1200/min limit

    return _rows_to_df(all_rows)


def _rows_to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df.index = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df.index.name = "timestamp"
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume", "quote_volume"]]


def main():
    parser = argparse.ArgumentParser(description="Download 15M data for all tradeable coins")
    parser.add_argument("--start", default="2022-01-01", help="Start date (default: 2022-01-01)")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to download (default: all 39)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip symbols that already have parquets")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbols = args.symbols if args.symbols else SYMBOLS
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    print(f"Downloading 15M data for {len(symbols)} symbols")
    print(f"  Start: {args.start}")
    print(f"  Output: {DATA_DIR}/")
    print()

    results = {}
    for i, sym in enumerate(symbols, 1):
        out_path = DATA_DIR / f"{sym}_15m.parquet"

        if args.skip_existing and out_path.exists():
            existing = pd.read_parquet(out_path)
            print(f"[{i}/{len(symbols)}] {sym}: SKIP (exists, {len(existing):,} bars)")
            results[sym] = len(existing)
            continue

        print(f"[{i}/{len(symbols)}] {sym}: downloading...", end="", flush=True)
        df = download_15m(sym, args.start, end_ms)

        if df.empty:
            print(f" FAILED (no data)")
            results[sym] = 0
            continue

        df.to_parquet(out_path)
        print(f" {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")
        results[sym] = len(df)

    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    success = sum(1 for v in results.values() if v > 0)
    print(f"  Success: {success}/{len(symbols)}")
    failed = [s for s, v in results.items() if v == 0]
    if failed:
        print(f"  Failed:  {', '.join(failed)}")
    total_bars = sum(results.values())
    print(f"  Total bars: {total_bars:,}")
    print(f"  Est. disk:  {total_bars * 48 / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
