#!/usr/bin/env python3
"""
Historical OHLCV data downloader for all Roostoo-listed coins (4H candles from Binance).

Downloads from Binance public klines API and saves as Parquet with UTC DatetimeIndex.
No API key required (public endpoint).

Usage:
  python scripts/download_data.py                          # all 67 pairs, 5yr, skip existing
  python scripts/download_data.py --start 2022-01-01      # custom start
  python scripts/download_data.py --symbols BTC ETH SOL   # specific coins only
  python scripts/download_data.py --workers 8             # parallel downloads
  python scripts/download_data.py --no-skip-existing      # re-download all
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BINANCE_API_BASE = "https://api.binance.com/api/v3"
INTERVAL = "4h"
REQUEST_SLEEP = 0.15   # seconds between paginated requests per symbol
BATCH_SIZE = 1000      # max rows per API call (Binance limit)
DEFAULT_WORKERS = 5    # parallel symbols (conservative; Binance limit: 1200 weight/min)

# All 67 pairs available on Roostoo as of 2026-03-18
# Binance symbol = coin + "USDT"  (e.g. BTC/USD → BTCUSDT)
ROOSTOO_COINS = [
    "1000CHEEMS", "AAVE", "ADA", "APT", "ARB", "ASTER", "AVAX", "AVNT",
    "BIO", "BMT", "BNB", "BONK", "BTC", "CAKE", "CFX", "CRV",
    "DOGE", "DOT", "EDEN", "EIGEN", "ENA", "ETH", "FET", "FIL",
    "FLOKI", "FORM", "HBAR", "HEMI", "ICP", "LINEA", "LINK", "LISTA",
    "LTC", "MIRA", "NEAR", "OMNI", "ONDO", "OPEN", "PAXG", "PENDLE",
    "PENGU", "PEPE", "PLUME", "POL", "PUMP", "S", "SEI", "SHIB",
    "SOL", "SOMI", "STO", "SUI", "TAO", "TON", "TRUMP", "TRX",
    "TUT", "UNI", "VIRTUAL", "WIF", "WLD", "WLFI", "XLM", "XPL",
    "XRP", "ZEC", "ZEN",
]


def coin_to_binance_symbol(coin: str) -> str:
    return f"{coin}USDT"


def timestamp_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def download_klines(symbol: str, start_ms: int, end_ms: int, max_retries: int = 3) -> list:
    """Download all klines for a symbol between start_ms and end_ms with pagination."""
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
                time.sleep(backoff)

        if not batch:
            break

        all_rows.extend(batch)

        if len(batch) < BATCH_SIZE:
            break

        current_start_ms = int(batch[-1][0]) + 1
        time.sleep(REQUEST_SLEEP)

    return all_rows


def klines_to_dataframe(rows: list) -> pd.DataFrame:
    df = pd.DataFrame({
        "open":   [float(r[1]) for r in rows],
        "high":   [float(r[2]) for r in rows],
        "low":    [float(r[3]) for r in rows],
        "close":  [float(r[4]) for r in rows],
        "volume": [float(r[7]) for r in rows],  # quote asset volume
    })
    df.index = pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True)
    df.index.name = "timestamp"
    return df.astype("float64")


def download_and_save(
    coin: str,
    start_ms: int,
    end_ms: int,
    output_dir: Path,
    skip_existing: bool,
) -> tuple[str, str, int]:
    """
    Download one coin and save to parquet.
    Returns (coin, status, n_rows).
    status: 'saved' | 'skipped' | 'error:<msg>'
    """
    symbol = coin_to_binance_symbol(coin)
    output_path = output_dir / f"{symbol}_4h.parquet"

    if skip_existing and output_path.exists():
        existing = pd.read_parquet(output_path)
        return coin, "skipped", len(existing)

    try:
        rows = download_klines(symbol, start_ms, end_ms)
        if not rows:
            return coin, "error:no data returned from Binance", 0

        df = klines_to_dataframe(rows)
        df.to_parquet(output_path)
        return coin, "saved", len(df)

    except requests.HTTPError as e:
        # 400 = symbol not found on Binance
        return coin, f"error:{e.response.status_code} {e.response.reason}", 0
    except Exception as e:
        return coin, f"error:{e}", 0


def main():
    five_years_ago = (datetime.now(timezone.utc) - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="Download Binance 4H OHLCV data for all Roostoo-listed coins."
    )
    parser.add_argument("--start", default=five_years_ago,
                        help=f"Start date YYYY-MM-DD (default: 5 years ago = {five_years_ago})")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--symbols", nargs="+", metavar="COIN",
                        help="Download only these coins (e.g. --symbols BTC ETH SOL)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-download even if parquet already exists")
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    coins = args.symbols if args.symbols else ROOSTOO_COINS
    # Validate user-supplied coins
    valid = set(ROOSTOO_COINS)
    unknown = [c for c in coins if c not in valid]
    if unknown:
        print(f"WARNING: unknown coins (not in Roostoo list): {unknown}")

    start_ms = timestamp_to_ms(args.start)
    end_ms = timestamp_to_ms(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Coins: {len(coins)} | Range: {args.start} to {args.end} | Workers: {args.workers}")
    print(f"Output: {output_dir} | Skip existing: {args.skip_existing}")
    print()

    saved = skipped = errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_and_save, coin, start_ms, end_ms, output_dir, args.skip_existing): coin
            for coin in coins
        }
        for fut in as_completed(futures):
            coin, status, n_rows = fut.result()
            symbol = coin_to_binance_symbol(coin)
            if status == "saved":
                print(f"  [OK]  {symbol:<20} {n_rows:>5} bars saved")
                saved += 1
            elif status == "skipped":
                print(f"  [--]  {symbol:<20} {n_rows:>5} bars (already exists, skipped)")
                skipped += 1
            else:
                msg = status.replace("error:", "")
                print(f"  [FAIL] {symbol:<20} {msg}")
                errors += 1

    print(f"\nDone. Saved: {saved} | Skipped: {skipped} | Failed: {errors}")
    if errors:
        print("  (Failed coins are not on Binance or are too new to have history)")


if __name__ == "__main__":
    main()
