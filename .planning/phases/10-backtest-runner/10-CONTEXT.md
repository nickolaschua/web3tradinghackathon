# Phase 10: Backtest Runner + Feature Prep - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<vision>
## How This Should Work

A script (`scripts/backtest.py`) that:
1. Loads the historical Parquet files from Phase 9
2. Runs the full feature engineering pipeline (reusing `bot/data/features.py`)
3. Accepts a pre-trained XGBoost model (.pkl file) as input
4. Simulates trading bar-by-bar — the model sees each bar's features and outputs a signal, the runner tracks positions and PnL
5. Prints (and optionally saves) a comprehensive stats report

The model is pre-trained externally (Phase 11). This phase just builds the infrastructure that accepts any model and tells you how it performed.

</vision>

<essential>
## What Must Be Nailed

- **No lookahead bias** — the feature pipeline already shift(1)s all indicators; the bar-by-bar loop must only feed the model features available at that bar's close, never future data
- **Feature prep matches live bot** — the same `compute_features()` + `compute_cross_asset_features()` pipeline used in live trading must be used in backtest, so backtest results reflect what the live bot would have done
- **Comprehensive stats** — must include at minimum: total return %, Sharpe ratio, Sortino ratio, max drawdown, win rate, avg trade PnL. Include any other metrics from the hackathon problem statement.
- **XGBoost model interface** — loads a `.pkl` file, feeds it the feature matrix, gets back BUY/SELL/HOLD signals (or probabilities that get thresholded)

</essential>

<boundaries>
## What's Out of Scope

- Training the XGBoost model — that's Phase 11
- Parameter sweeping / hyperparameter optimisation — that's Phase 11
- Walk-forward validation — Phase 11
- Live trading integration — the runner is offline only; it doesn't connect to Roostoo API
- Plotting/visualisation — stats report is text/CSV output; no charts required for this phase

</boundaries>

<specifics>
## Specific Ideas

- Stats output should include everything the hackathon judges might look at: total return, Sharpe, Sortino, max drawdown, win rate, avg trade, number of trades
- Input: `--model model.pkl --data data/BTCUSDT_4h.parquet --start 2024-01-01 --end 2024-12-31` style CLI args (or similar)
- Feature prep is the bridge between raw Parquet and what the XGBoost model expects — the same feature matrix the live bot computes

</specifics>

<notes>
## Additional Context

- Phase 11 will train on 2022–2023 data and validate on 2024 held-out data — so this backtest runner needs to handle date range filtering
- The hackathon scoring metric is unknown (gap_03 in docs); include Sharpe AND Sortino AND total return so we can optimise for whichever matters
- XGBoost model output: likely a probability (0–1) per bar; runner needs a threshold to convert to BUY/SELL/HOLD
- Existing `bot/execution/risk.py` (RiskManager) and `bot/execution/order_manager.py` (OrderManager) logic can inform the simulation but don't need to be imported directly — a simple backtest sim is fine

</notes>

---

*Phase: 10-backtest-runner*
*Context gathered: 2026-03-17*
