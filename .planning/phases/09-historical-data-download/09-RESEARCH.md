---
phase: 09-historical-data-download
researched: 2026-03-17
status: complete
---

# Phase 9 Research: Historical Data Download

## Executive Summary

Direct Binance REST API (`/api/v3/klines`) is the right approach. The alternative
(`binance-historical-data` pip package, S3-based) is more complex, harder to parse,
and adds a dependency for a one-time script. Plain `requests` is sufficient.

**Key finding:** Nothing in this phase requires research-driven decisions — the
Binance klines API is stable, well-documented, and the pagination pattern is
straightforward. The main deliverable is confirming the exact request/response
format and pyarrow Parquet compatibility.

---

## Domain 1: Binance Public Klines API

### Endpoint

```
GET https://api.binance.com/api/v3/klines
```

No API key required. The `/api/v3/klines` endpoint is public (no auth headers).
Same endpoint works from any IP, no geo-blocking on public data.

### Parameters

| Parameter  | Type   | Required | Notes                                          |
|------------|--------|----------|------------------------------------------------|
| symbol     | STRING | YES      | e.g. `BTCUSDT`                                 |
| interval   | ENUM   | YES      | e.g. `4h`                                      |
| startTime  | LONG   | NO       | Unix milliseconds. Use to paginate forward.    |
| endTime    | LONG   | NO       | Unix milliseconds. Optional upper bound.       |
| limit      | INT    | NO       | Default 500, max **1000**                      |

### Response Format

Each kline is a 12-element array:

```
[
  0:  open_time          (ms, UTC Unix timestamp)
  1:  open               (string → cast to float64)
  2:  high               (string → cast to float64)
  3:  low                (string → cast to float64)
  4:  close              (string → cast to float64)
  5:  volume             (string → cast to float64)
  6:  close_time         (ms) — DISCARD
  7:  quote_asset_volume — DISCARD
  8:  number_of_trades   — DISCARD
  9:  taker_buy_base_volume — DISCARD
 10:  taker_buy_quote_volume — DISCARD
 11:  ignore             — DISCARD
]
```

**All numeric fields are returned as strings** — must be cast to float64.

### Rate Limits

- **IP weight limit:** 6000 weight/minute (Spot API, as of late 2024; was 1200)
- **klines weight:** 2 per request (regardless of limit parameter)
- **Safe cadence:** 0.1s sleep between requests → 600 calls/min = 1200 weight/min
  (well within even the old 1200 limit; same sleep works for both limits)
- **Estimated calls for 2022-01-01→today at 4H interval:**
  - ~1,500 bars per pair ÷ 1000 per request = 2 calls per pair × 3 pairs = 6 calls
  - With 0.1s sleep: completes in under 5 seconds total
  - Full download is trivially fast; 0.1s sleep is conservative but appropriate

### Pagination Pattern

```python
start_ms = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
end_ms   = int(pd.Timestamp.utcnow().timestamp() * 1000)

while start_ms < end_ms:
    resp = requests.get(url, params={..., "startTime": start_ms})
    rows = resp.json()
    if not rows:
        break
    all_rows.extend(rows)
    if len(rows) < 1000:   # final page
        break
    start_ms = rows[-1][0] + 1   # advance past last open_time
```

**Critical:** Advance `startTime` to `last_open_time + 1ms` to avoid re-fetching
the last bar. Binance uses inclusive startTime.

---

## Domain 2: pyarrow + pandas Parquet

### Why pyarrow (not fastparquet)

- `fastparquet` has edge cases with tz-aware DatetimeIndex serialization — it can
  strip or mis-encode the UTC timezone, requiring manual re-attachment on read.
- `pyarrow>=14.0` correctly round-trips `pd.DatetimeIndex` with UTC timezone.
- `pd.read_parquet()` auto-detects engine; pyarrow is the pandas-recommended default.

### UTC DatetimeIndex → Parquet Round-trip

```python
# Write
df.index = pd.to_datetime(open_times, unit="ms", utc=True)
df.index.name = "timestamp"
df.to_parquet(path, engine="pyarrow")

# Read (LiveFetcher compatibility check)
df2 = pd.read_parquet(path)
assert str(df2.index.tz) == "UTC"   # ✓ preserved by pyarrow
ts = df2.index[0]
ts.timestamp()                       # ✓ .timestamp() available on tz-aware Timestamp
```

### LiveFetcher Compatibility Requirements (from Phase 4)

LiveFetcher._seed_from_history() needs:
1. `df.columns` — any capitalization (lowercased internally with `.str.lower()`)
2. `df.index` — must have `.timestamp()` method (tz-aware `pd.Timestamp` has it)
3. Required columns: `open`, `high`, `low`, `close`, `volume`

The UTC DatetimeIndex from `pd.to_datetime(..., utc=True)` satisfies requirement 2.
pyarrow preserves it on round-trip. No special handling needed.

### pyarrow Version

- `pyarrow>=14.0` — released Oct 2023, widely available, stable UTC timezone support
- Do NOT use fastparquet (tz edge cases) or omit engine (defaults vary by env)

---

## Domain 3: binance-historical-data Package (Alternative — Rejected)

The `binance-historical-data` pip package downloads raw data from Binance's public
S3 bucket (`s3://data.binance.vision` / `https://data.binance.vision`).

**Why we're NOT using it:**

1. Downloads ZIP files, extracts CSV, requires multi-step parsing
2. CSV headers differ from API response format (different column names)
3. Adds an extra dependency for a one-time 15-line script
4. S3 bucket data is organized by day/month → requires more file management
5. Direct REST API gives identical data with simpler code

**When you'd use it:** Very large historical downloads (10+ years, tick data) where
rate limits are a concern. For 3 years of 4H data (~4,500 rows total), REST API
is strictly better.

---

## Domain 4: Data Directory & gitignore

- `data/` directory does not need to exist before writing — `mkdir(parents=True, exist_ok=True)` handles creation
- `.gitignore` already has `*.parquet` — covers all three output files
- Files are gitignored but present locally; bot reads them on startup
- EC2 deployment: files must be manually `scp`'d or re-downloaded on EC2 (Phase 9 is local only; EC2 seeding is out of scope)

---

## Standard Stack Decision

| Decision | Choice | Rejected | Reason |
|----------|--------|----------|--------|
| HTTP client | `requests` (already installed) | `httpx`, `aiohttp` | No async needed; requests is already a dep |
| Parquet engine | `pyarrow>=14.0` | `fastparquet` | UTC timezone round-trip reliability |
| Data source | Binance REST API | `binance-historical-data` S3 | Simpler, no extra deps |
| CLI args | `argparse` (stdlib) | `click`, `typer` | No extra deps for a one-time script |

---

## Common Pitfalls

1. **Strings not floats** — Binance returns all OHLCV as strings. Must `float(x)` each.
2. **Inclusive startTime** — not advancing by +1ms causes infinite loop re-fetching last bar.
3. **UTC timezone missing** — using `pd.to_datetime(unit="ms")` without `utc=True` gives tz-naive index; LiveFetcher's `ts.timestamp()` still works on tz-naive Timestamps, but downstream timezone checks may fail.
4. **fastparquet UTC loss** — if pyarrow unavailable and fastparquet used, UTC may be stripped on write.
5. **EC2 data files** — data files are local-only; must be re-downloaded or `scp`'d to EC2 separately.

---

## Conclusion

Phase 9 has no research-blocking unknowns. The PLAN.md created in parallel accurately
reflects the correct implementation approach. Key confirmations:

- ✅ Binance `/api/v3/klines` is public, no auth, max 1000 rows/request
- ✅ All OHLCV fields returned as strings — cast to float64
- ✅ Pagination: advance `startTime = last_open_time + 1ms`, stop on `< 1000` rows
- ✅ 0.1s sleep is safe (well under rate limits)
- ✅ pyarrow>=14.0 correctly round-trips UTC DatetimeIndex
- ✅ `binance-historical-data` package correctly rejected (over-engineered for this use case)
- ✅ `.gitignore` already covers `*.parquet`
