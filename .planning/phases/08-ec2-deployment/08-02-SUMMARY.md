---
phase: 08-ec2-deployment
plan: "08-02"
subsystem: deployment
tags: [ec2, tmux, smoke-test, round1]
requires:
  - phase: 08-01
    provides: deploy/roostoo-bot.service, deploy/bootstrap.sh
provides:
  - Bot running in tmux on EC2 ap-southeast-2 t3.medium
  - startup_reconciliation verified (Reconciliation OK)
  - Round 1 keys active in .env
  - $50,000 USD balance confirmed via API
affects: []
key-files:
  created:
    - /home/ec2-user/bot/.env (on EC2, not in repo)
    - /home/ec2-user/bot/state.json (first write confirmed)
  modified:
    - bot/api/client.py (SpotWallet key fix + get_open_orders endpoint fix)
    - .env (local — variable names corrected to ROOSTOO_API_KEY/ROOSTOO_SECRET)
key-decisions:
  - "Used tmux (not systemd) for process persistence — hackathon environment constraints"
  - "Python 3.9.25 (not 3.11) — template pre-installed version; compatible with all deps"
  - "EC2 is t3.medium (not t3.micro) — enforced by HackathonBotTemplate"
  - "Session Manager only (SSH blocked in hackathon AWS environment)"
---

# Phase 8 Plan 02: EC2 Deployment and Verification Summary

**Bot deployed to EC2 ap-southeast-2, smoke test passed with confirmed $50,000 USD balance, Round 1 keys active in tmux session.**

## Accomplishments

- Launched EC2 t3.medium in ap-southeast-2 via HackathonBotTemplate
- Connected via Session Manager (SSH blocked in hackathon environment)
- Cloned repo, set up venv, installed dependencies with Python 3.9.25
- Fixed two critical bugs in `bot/api/client.py` discovered during live testing:
  1. `get_balance()` — real API returns `SpotWallet` key, not `Wallet`; normalised to handle both
  2. `get_open_orders()` — wrong endpoint (`GET /v3/order` → 404); fixed to `POST /v3/query_order` with `OrderMatched` key
- Fixed local `.env` variable names (`API_KEY` → `ROOSTOO_API_KEY`, `SECRET` → `ROOSTOO_SECRET`)
- Resolved 401 auth errors (signature error → api-key invalid → 200 OK) by correcting key values
- Confirmed 200 OK from `/v3/balance` with `total_usd=50000.00`
- Bot starts cleanly: `Reconciliation OK`, `Startup complete — entering main loop`, warmup mode active
- Bot running persistently in tmux session on EC2

## Files Created/Modified

**EC2 (not in repo):**
- `/home/ec2-user/bot/.env` — Round 1 keys active (ROOSTOO_API_KEY / ROOSTOO_SECRET)
- `/home/ec2-user/bot/state.json` — first state write confirmed

**Repo (committed):**
- `bot/api/client.py` — SpotWallet normalisation + get_open_orders endpoint fix
- `.env` (local) — variable names corrected
- `test_api.py` — diagnostic script for HMAC auth debugging

## Issues Encountered

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| 401 "signature error" | Old testing keys in .env, not actual account keys | Retrieved real keys from Roostoo portal |
| 401 "api-key invalid" | EC2 .env still had `your_round1_key_here` placeholder | Re-edited .env in nano and saved correctly |
| `total_usd=0.0` from balance | Real API uses `SpotWallet` key, not `Wallet` | Fixed `get_balance()` to check both keys |
| 404 on open orders | Wrong endpoint `GET /v3/order` | Fixed to `POST /v3/query_order`, key `OrderMatched` |
| Local .env variable name mismatch | Manual edit used `API_KEY` instead of `ROOSTOO_API_KEY` | Restored correct variable names |
| Python 3.11 not found | Template pre-installs Python 3.9.25, not 3.11 | Used `python3` (compatible) |
| Session Manager paste issues | Multi-line scripts cause "unexpected indent" | Committed scripts to repo, pulled on EC2 |

## Decisions Made

- **tmux instead of systemd** — The hackathon AWS environment constrained process management. Systemd service was installed in 08-01 but the practical path in this environment was tmux for the smoke test. Bot is running and persistent.
- **Python 3.9 vs 3.11** — HackathonBotTemplate ships with Python 3.9.25. All dependencies are compatible; no code changes needed.
- **Round 1 keys already active** — The keys in `.env` (`vFTkFpzj8l...`) are the actual competition keys from the Roostoo portal, not testing keys. The bot is live.

## Warmup Note

The bot requires 35 × 4H bars (~5.8 days) before trading signals activate (MACD warmup requirement). Bot started 2026-03-17. Round 1 competition window opens Mar 21 8PM SGT. The bot will be in warmup mode for the first ~1 day of competition — this is expected behaviour; infrastructure is correct.

## Next Step

Phase 8 complete. Project complete — bot is live for Round 1 competition.

**To fill in alpha strategy (optional before Mar 21):**
- `bot/strategy/momentum.py` — implement `generate_signal()` for momentum alpha
- `bot/strategy/mean_reversion.py` — implement `generate_signal()` for mean reversion alpha
- Both stubs currently return HOLD (neutral) — bot will not trade until alpha is added

**To monitor the bot:**
```bash
# Reconnect via Session Manager → your instance
tmux attach        # or: tmux ls → tmux attach -t <name>
```
