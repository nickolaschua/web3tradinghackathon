# Infrastructure & Deployment

## What This Covers

The infrastructure layer is everything that the code runs on: the EC2 instance, the operating system configuration, the process manager, the clock synchronisation, the secrets management, and the deployment workflow. None of this appears in the block diagram as a trading system layer, but without it, none of the trading layers run reliably.

---

## EC2 Instance Setup

### Choosing the Right Instance

Use two separate instances with different purposes:

**Research instance (pre-hackathon only):** `t3.medium` (2 vCPU, 4 GB RAM). Used for vectorbt parameter sweeps, walk-forward optimisation with Optuna, and data processing. Shut this down after the competition prep phase — you don't need it during competition.

**Live trading instance (competition):** `t3.micro` (1 vCPU, 1 GB RAM). Sufficient for a single trading bot polling once per minute. The live bot uses minimal CPU and typically under 200 MB of RAM.

Always use **On-Demand** instances for the competition period, not Spot. Spot instances can be reclaimed with 2 minutes notice — acceptable for batch jobs, unacceptable for a live trading bot.

### Initial Server Setup

```bash
# Update OS
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Create project directory
mkdir -p ~/trading-bot
cd ~/trading-bot

# Create virtualenv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install requests pandas numpy pyarrow pandas-ta python-dotenv \
            pyyaml psutil python-telegram-bot

# For research instance only:
pip install vectorbt backtesting optuna quantstats lightgbm scikit-learn
```

### Clock Synchronisation (chrony)

The Roostoo API rejects requests where the timestamp differs from server time by more than 60 seconds. Configure chrony to use Amazon's Time Sync Service, which provides sub-millisecond accuracy on EC2.

```bash
# Install chrony
sudo apt install -y chrony

# Configure to use Amazon Time Sync Service
sudo nano /etc/chrony/chrony.conf
```

Add or replace the server lines with:
```
server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
```

```bash
# Restart chrony and verify
sudo systemctl restart chrony
chronyc tracking
# "System time" should show < 1ms offset
```

The application also implements its own time sync (see Layer 1) as a redundant backup. Both layers together make timestamp rejection essentially impossible.

---

## Secrets Management

API keys must never appear in code, logs, or git history.

### `.env` file

```bash
# Create .env in the project root
cat > ~/trading-bot/.env << 'EOF'
ROOSTOO_API_KEY=your_api_key_here
ROOSTOO_SECRET_KEY=your_secret_key_here
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
EOF

# Restrict permissions
chmod 600 ~/trading-bot/.env
```

### `.gitignore`

```
.env
state.json
state.json.bak
logs/
data/raw/
data/parquet/
__pycache__/
*.pyc
venv/
```

---

## systemd Service

The systemd service ensures the bot:
- Starts automatically when EC2 boots
- Restarts automatically if it crashes (up to 5 times in 5 minutes, then pauses)
- Writes all output to the system journal (queryable with `journalctl`)

```bash
sudo nano /etc/systemd/system/tradingbot.service
```

```ini
[Unit]
Description=Roostoo Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-bot
EnvironmentFile=/home/ubuntu/trading-bot/.env
ExecStart=/home/ubuntu/trading-bot/venv/bin/python main.py
Restart=always
RestartSec=10
StartLimitBurst=5
StartLimitIntervalSec=300
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tradingbot

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable tradingbot
sudo systemctl start tradingbot

# Verify running
sudo systemctl status tradingbot

# View logs
journalctl -u tradingbot -f
journalctl -u tradingbot --since "1 hour ago"
```

### Testing Crash Recovery (do this before competition)

This is mandatory. Do not skip it:

```bash
# Find the process PID
sudo systemctl status tradingbot | grep "Main PID"

# Kill it hard (simulates crash)
sudo kill -9 <PID>

# Wait 15 seconds, then verify it restarted
sleep 15
sudo systemctl status tradingbot
# Should show "active (running)" with a recent start time

# Check logs for startup reconciliation
journalctl -u tradingbot -n 50
# Should show: "Startup complete — ready to trade"
```

---

## Deployment Workflow

All code changes to the live bot should follow this workflow:

```bash
# On your local machine
git add -A
git commit -m "description of change"
git push origin main

# On EC2
cd ~/trading-bot
git pull origin main
sudo systemctl restart tradingbot
journalctl -u tradingbot -f  # Watch startup logs
```

Never edit files directly on EC2 in production without committing them — you'll lose the changes on the next `git pull`.

**Config-only changes** (most common during competition):

```bash
# On EC2 directly
nano config.yaml
# Make changes
sudo systemctl restart tradingbot
# Startup reconciliation runs, bot resumes trading
```

---

## Pre-Competition Deployment Checklist

Run through this list the day before the competition starts:

```
Infrastructure:
□ EC2 On-Demand instance running (NOT Spot)
□ chrony configured and showing < 1ms offset
□ systemd service enabled and running
□ Crash recovery tested (kill -9 + verify restart)

Secrets:
□ .env file present with correct API keys
□ .env permissions are 600
□ API keys tested against live Roostoo endpoint

Code:
□ git pull — latest code deployed
□ config.yaml has walk-forward validated parameters
□ Holdout performance numbers documented

Data:
□ Historical Parquet files on EC2 for seeding live buffer
□ Gap detector run on all files
□ Buffer seeding verified (is_warmed_up() returns True)

Monitoring:
□ Telegram bot configured and responding
□ Test message sent and received
□ Heartbeat fires correctly (wait 10 min after start)
□ Trade alert tested with a minimum-size order

Verification:
□ Bot ran for 24 hours without intervention
□ sign_test.py passes on EC2 environment
□ State.json written correctly after first cycle
□ Reconciliation logs clean (no discrepancies)

Competition prep:
□ Three config variants ready (aggressive, normal, conservative)
□ Runbook written: what to do if circuit breaker fires
□ Runbook written: what to do if API is down > 30 min
□ Runbook written: what to do if equity drops 15% in 24 hours
□ WhatsApp channel with Roostoo engineers joined and notifications on
```

---

## Competition Operations Runbook

### Scenario 1: Circuit breaker triggers

1. Do NOT restart the bot or disable the circuit breaker immediately
2. SSH into EC2 and review `logs/bot.log` for the last 2 hours
3. Check the Roostoo app for current market conditions — is BTC in a major downtrend?
4. If the strategy has a bug causing incorrect stops, fix it and restart
5. If the market is genuinely in a bear regime, let the circuit breaker hold until recovery
6. The circuit breaker resets automatically when portfolio recovers to HWM

### Scenario 2: API is unresponsive for > 30 minutes

1. Check your bot logs for the exact error responses
2. Message the Roostoo WhatsApp channel immediately — this is a systemic issue
3. The bot will keep retrying with exponential backoff — do not restart it
4. If positions are open and you need to exit manually, use the Roostoo app directly
5. Once API recovers, the bot reconciles automatically on its next successful cycle

### Scenario 3: Equity drops 15% in 24 hours

1. SSH into EC2: `journalctl -u tradingbot --since "24 hours ago" | grep -E "TRADE|STOP|REGIME"`
2. Identify whether losses are from stops being hit (expected) or from the circuit breaker not triggering (problem)
3. If stops are working correctly, the loss is within expected parameters — do not override
4. If the strategy is entering positions at the wrong times, consider switching config to conservative variant
5. Never manually close positions based on emotion — only based on clear evidence of a strategy bug

### Scenario 4: Telegram heartbeat goes silent

1. SSH into EC2 immediately: `sudo systemctl status tradingbot`
2. If stopped: `sudo systemctl start tradingbot` and review why it stopped
3. If running but no heartbeat: `journalctl -u tradingbot -n 100` to identify the issue
4. Most likely cause: Telegram API timeout causing healthcheck to hang (non-fatal but needs investigation)

---

## Resource Monitoring

Expected resource usage for a running bot on t3.micro:

| Resource | Expected | Warning threshold |
|---|---|---|
| CPU | 1–5% average | > 90% sustained for 2+ minutes |
| Memory | 150–300 MB | > 800 MB (80% of 1 GB) |
| Disk (logs) | ~10 MB/day | > 85% of total disk |
| Network | < 1 MB/hour | N/A |

If memory usage is unexpectedly high, the most common cause is the feature DataFrame accumulating in memory without being garbage-collected. Fix: ensure `feature_cache` in `main.py` is overwritten each cycle rather than appended to.
