# Phase 4: Data Pipeline - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<vision>
## How This Should Work

The pipeline is a quality gate. Raw OHLCV data comes in, clean feature matrices go out. Strategy logic never touches raw data — it only ever sees a validated, gap-free, shift-corrected feature DataFrame.

At startup, the LiveFetcher seeds from historical Binance Parquet files (flat columns) to warm up buffers instantly. During the competition, it transitions seamlessly to live Roostoo ticker polls every 60s, building synthetic candles. The feature output looks identical whether the source is historical or live — strategy logic is never aware of the difference.

</vision>

<essential>
## What Must Be Nailed

- **No NaN ever reaches the strategy** — warmup rows dropped, gaps forward-filled with volume=0, all indicators shifted 1 bar. The feature matrix is always clean.
- **Backtest/live parity** — `features.py` is a single shared library called identically in both environments. Validated parameters transfer directly to live trading with no surprises.
- **Three missing methods on LiveFetcher** — `seed_dfs: dict[str, pd.DataFrame]` constructor, `get_latest_price(pair)`, `get_candle_boundaries()` — these are required by main.py and will cause AttributeError crashes if absent.

</essential>

<boundaries>
## What's Out of Scope

- No backtesting engine (vbt_sweep, walk-forward scripts) — features.py is the shared library, not the research tooling
- No strategy logic — this phase delivers a feature matrix; what to do with it is Phase 6's job
- No IC analysis tooling — ic_analysis.py is research scaffolding, not production code

</boundaries>

<specifics>
## Specific Ideas

- Use close-to-close ATR proxy (`log_returns.rolling(14).std() * close * 1.25`) — NOT pandas-ta ATR, NOT ADX, NOT OBV
- Cross-asset features (`btc_return_lag1`, `btc_return_lag2`) must be injected BEFORE `dropna()` — otherwise they get silently dropped
- `is_warmed_up()` threshold is 35 bars (not 200 as in early docs)
- Parquet columns are flat (`open`, `high`, `low`, `close`, `volume`) — not multi-index
- LiveFetcher buffer is 500 bars max (deque maxlen)

</specifics>

<notes>
## Additional Context

Competition deadline: bot must be deployed and trading before Mar 21, 8PM. Data pipeline is on the critical path — it must be complete before the execution engine (Phase 5) and main loop (Phase 7) can be wired together.

Agent 2 owns only `bot/data/` — zero cross-package imports. Dependencies: `pandas`, `pandas-ta-classic` (not `pandas-ta`), `binance_historical_data`.

Issue 15 is the primary driver: three methods called by other layers that don't exist yet. These must be implemented with correct signatures matching main.py's call sites.

</notes>

---

*Phase: 04-data-pipeline*
*Context gathered: 2026-03-16*
