# Pre-Competition Checklist

## Before Round 1 starts (tonight, Mar 21 8PM SGT)

- [ ] **Delete `state.json`** — stale test account data will confuse reconciliation on startup.
  ```
  del state.json
  ```

- [ ] **Add Round 1 API keys to `.env`** — replace test keys with Round 1 competition keys.
  ```
  ROOSTOO_API_KEY=<round1_key>
  ROOSTOO_SECRET=<round1_secret>
  ```
  The bot reads `ROOSTOO_API_KEY` first, then falls back to `ROOSTOO_API_KEY_TEST`.

- [ ] **Verify bot connects with Round 1 keys** — run the smoke test:
  ```
  python scripts/test_order_placement.py
  ```

- [x] **AWS EC2 instance** — already running. SSH in, pull latest code, restart the bot service.

- [x] **Telegram alerts** — already configured and working. Bot sends "✅ Bot started" on startup.

## Notes

- Commission rate is **0.1% (taker)** — API docs say 0.012% but live smoke test confirmed 0.001 (=0.1%). Backtests already use 10bps.
- Pairs trading is **disabled** — backtested at -14.34%, catastrophic.
- Server time sync runs automatically on startup via `client.sync_time()`.
- `state.json` is written atomically every cycle — safe to delete while bot is stopped.

## Round Dates

- Round 1: Mar 21 – Mar 31
- Round 2: Apr 4 – Apr 14
