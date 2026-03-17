# Phase 9: Historical Data Download - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<vision>
## How This Should Work

A script I run once locally. It hits Binance, pulls 4H candles for BTC/USD, ETH/USD, and SOL/USD, and saves them as Parquet files in the `data/` directory. After that, the bot can seed its LiveFetcher from those files on every startup instead of doing a cold start.

The end state: run the script once, files appear in `data/`, bot starts up with a full history and enters warmup immediately rather than waiting 5+ days of live polling.

</vision>

<essential>
## What Must Be Nailed

- **Correct column names** — LiveFetcher._seed_from_history() expects flat lowercase columns: `open`, `high`, `low`, `close`, `volume`. The script must output exactly this — not a multi-index, not capitalised headers.
- **Correct filename pattern** — main.py._load_seed_data() looks for `data/BTCUSDT_4h.parquet`, `data/ETHUSDT_4h.parquet`, `data/SOLUSDT_4h.parquet`. Script must match these exactly.
- **Enough history** — at least 500 bars (the LiveFetcher maxlen) so warmup completes instantly. Ideally 2+ years for training data later.

</essential>

<boundaries>
## What's Out of Scope

- Not cleaning or engineering features — raw OHLCV only; features are computed at runtime by `bot/data/features.py`
- Not running on EC2 — local only; data gets committed or copied to EC2 separately
- Not real-time updates — one-time download, not a scheduled refresh
- ML training data prep — that's Phase 11

</boundaries>

<specifics>
## Specific Ideas

- Target pairs: BTCUSDT, ETHUSDT, SOLUSDT (4H interval)
- Should be runnable with no API key (Binance public endpoint)
- Enough history to cover 2022–2024 for future ML training (Phase 11 will train on 2022–2023, validate on 2024)

</specifics>

<notes>
## Additional Context

- The `data/` directory should be in `.gitignore` (Parquet files are large binary blobs)
- LiveFetcher keys are in Roostoo format ("BTC/USD") but filename uses Binance format ("BTCUSDT") — main.py handles the mapping via `_BINANCE_TO_ROOSTOO`
- If Binance public API has rate limits or geo-blocks, `binance-historical-data` (pip package) is an alternative that downloads from Binance's public S3 bucket

</notes>

---

*Phase: 09-historical-data-download*
*Context gathered: 2026-03-17*
