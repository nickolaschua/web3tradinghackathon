# Handoff: Project Complete — Bot Live on EC2

**Updated:** 2026-03-17 (end of session)
**Status:** All 18 plans complete. Bot running live.

---

## Where things stand right now

**Bot is live on EC2 ap-southeast-2 (t3.medium, HackathonBotTemplate), running in tmux.**

### What's confirmed working:
- ✅ EC2 instance running in ap-southeast-2 via HackathonBotTemplate
- ✅ Repo cloned, venv set up, dependencies installed (Python 3.9.25)
- ✅ Round 1 keys active in `/home/ec2-user/bot/.env`
- ✅ API authentication: 200 OK, $50,000 USD balance confirmed
- ✅ `Reconciliation OK` on startup
- ✅ `Startup complete — entering main loop` logged
- ✅ Bot in warmup mode (correct — needs 35 × 4H bars ≈ 5.8 days)
- ✅ state.json written (crash recovery functional)
- ✅ SIGINT handler registered (Ctrl+C flushes state)

### What the bot is doing right now:
- Polling ticker every 60s
- Building synthetic 4H candles
- In **warmup mode** — no trades executing until 35 bars collected
- Strategies return HOLD (stubs — no alpha logic filled in yet)

---

## Timeline

| Event | Date |
|-------|------|
| Bot started on EC2 | 2026-03-17 |
| Warmup completes (~35 × 4H bars) | ~2026-03-23 |
| Round 1 competition window | Mar 21 8PM SGT → TBD |

**Note:** Bot will be in warmup for ~first 1 day of competition. After warmup, it trades but executes HOLD (neutral) until alpha is added.

---

## To reconnect to the bot

1. Go to `https://d-906625dad1.awsapps.com/start` → sign in
2. AWS Account → your team → Management Console → ap-southeast-2
3. EC2 → Instances → select instance → Connect → Session Manager → Connect
4. In terminal:
```bash
tmux attach        # or: tmux ls → tmux attach -t <session-name>
```

---

## Optional next steps (not required for competition)

### Add alpha strategy (recommended before warmup ends ~Mar 23)

Strategy stubs are in:
- `bot/strategy/momentum.py` — fill in `generate_signal(pair, features)` for momentum signals
- `bot/strategy/mean_reversion.py` — fill in `generate_signal(pair, features)` for mean reversion

Both currently return `TradingSignal(pair=pair, direction="HOLD", size=0.0, confidence=0.0)`.

Available features in the `features` DataFrame:
- `close`, `rsi`, `macd`, `macd_signal`, `ema_slope` (shifted 1 bar, no lookahead)
- `atr_proxy` (close-to-close ATR)
- `lag1_close`, `lag2_close`, `lag3_close` (historical closes)
- Cross-asset: `eth_close`, `sol_close`, `eth_rsi`, `sol_rsi` (if seeded)

### Deploy strategy update to EC2

After editing locally:
```bash
git add bot/strategy/momentum.py bot/strategy/mean_reversion.py
git commit -m "feat: add alpha strategy signals"
git push origin main
```

On EC2 (via Session Manager):
```bash
cd /home/ec2-user/bot
git pull origin main
# restart bot in tmux:
tmux attach
# Ctrl+C to stop, then:
python3 main.py
# Ctrl+B, D to detach
```

---

## For Claude Agent (next session)

**Project is complete.** No planned phases remain.

If user asks about the bot, refer to:
- `.planning/phases/08-ec2-deployment/08-02-SUMMARY.md` — full deployment log
- `.planning/STATE.md` — project status (100% complete)
- `bot/strategy/momentum.py` and `bot/strategy/mean_reversion.py` — where alpha goes

If user wants to add alpha:
- Read `bot/strategy/base.py` for TradingSignal dataclass
- Read `bot/data/live_fetcher.py` for feature column names
- Strategy returns `TradingSignal(pair, direction, size, confidence)` where direction is `"BUY"`, `"SELL"`, or `"HOLD"`
