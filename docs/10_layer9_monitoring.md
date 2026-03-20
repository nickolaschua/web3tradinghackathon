# Layer 9 — Monitoring & Alerting

## What This Layer Does

The Monitoring layer gives you visibility into a system that runs autonomously 24/7. It does three things: writes structured logs to disk for post-incident analysis, sends real-time Telegram alerts to your phone for anything requiring attention, and runs periodic health checks on the EC2 instance itself to catch infrastructure problems before they affect trading.

Without this layer, the bot is a black box. You wouldn't know if it was trading, crashing, or silently sitting in an error loop. During a multi-day competition, you will be sleeping, eating, and living your life while the bot operates. This layer is what makes that possible without anxiety.

**This layer is deployed on EC2.** It runs throughout the competition.

---

## What This Layer Is Trying to Achieve

1. Alert you immediately to anything that requires human attention — trade fills, errors, regime changes, infrastructure problems
2. Provide a complete, queryable audit trail of every decision the bot made
3. Keep the alert signal-to-noise ratio high — too many alerts is as bad as too few
4. Never allow a monitoring failure to affect trading (Telegram down ≠ bot stops)

---

## How It Contributes to the Bigger Picture

Every other layer is designed to run without human intervention. This layer is the interface between the autonomous system and you. Its value becomes clear the moment something unexpected happens — a regime change at 3am, a circuit breaker triggering overnight, a server running low on disk space. Without this layer, you find out about these events when you SSH in and look. With this layer, you find out within minutes.

The secondary value of this layer is the post-competition analysis it enables. Every trade, every signal, every API call is logged. After the competition you can replay exactly what happened and understand why.

---

## Files in This Layer

```
monitoring/
├── logger.py       Structured file logging with rotation
├── telegram.py     Alert dispatcher
└── healthcheck.py  System resource monitoring + heartbeat
```

---

## `monitoring/logger.py`

Two log streams run simultaneously:

**Main log (`bot.log`):** Every significant event in the system — API calls, signal generation, regime changes, errors, start/stop. Rotates at 10MB, keeps 10 backups. JSON format for machine parsing.

**Trade log (`trades.log`):** Every order submission and response. One JSON object per line. Never rotated — you want the complete history. Used for PnL analysis after the competition.

```python
import logging
import logging.handlers
import json
import time
from pathlib import Path

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)

    # ── Main rotating log ──────────────────────────────────────────────────
    main_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/bot.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
        encoding="utf-8",
    )
    main_handler.setFormatter(JsonFormatter())

    # ── Console handler for development ───────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(main_handler)
    root_logger.addHandler(console_handler)

    return root_logger


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": time.time(),
            "iso": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


class TradeLogger:
    """
    Append-only log of every order submission and response.
    Never rotated — complete trade history for the competition.
    """

    def __init__(self, log_dir: str = "logs"):
        Path(log_dir).mkdir(exist_ok=True)
        self._file = open(f"{log_dir}/trades.log", "a", encoding="utf-8")

    def log_submission(self, pair: str, side: str, quantity: float,
                       price: float, order_type: str, payload: dict):
        self._write({
            "event": "SUBMISSION",
            "ts": time.time(),
            "pair": pair, "side": side, "quantity": quantity,
            "price": price, "order_type": order_type,
            "payload": payload,
        })

    def log_response(self, success: bool, order_id: int,
                     status: str, fill_price: float,
                     commission: float, err_msg: str = ""):
        self._write({
            "event": "RESPONSE",
            "ts": time.time(),
            "success": success, "order_id": order_id,
            "status": status, "fill_price": fill_price,
            "commission": commission, "err_msg": err_msg,
        })

    def log_stop_exit(self, pair: str, stop_type: str,
                      entry_price: float, exit_price: float):
        self._write({
            "event": "STOP_EXIT",
            "ts": time.time(),
            "pair": pair, "stop_type": stop_type,
            "entry_price": entry_price, "exit_price": exit_price,
            "pnl_pct": (exit_price - entry_price) / entry_price,
        })

    def _write(self, obj: dict):
        self._file.write(json.dumps(obj) + "\n")
        self._file.flush()  # Flush immediately — never buffer trade logs

    def __del__(self):
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()
```

---

## `monitoring/telegram.py`

```python
import requests
import time
import logging
import os
from typing import Optional
from functools import wraps

logger = logging.getLogger(__name__)

class TelegramAlerter:
    """
    Send alerts to a Telegram chat.
    
    CRITICAL DESIGN RULE: This class must NEVER raise an exception to the caller.
    If Telegram is down or slow, the bot continues trading silently.
    Monitoring failure must never cause trading failure.
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    TIMEOUT_SECONDS = 5

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._enabled = bool(token and chat_id)
        if not self._enabled:
            logger.warning("Telegram not configured — alerts disabled")

    def send(self, message: str, level: str = "INFO") -> bool:
        """Send a message. Always returns without raising. Returns True on success."""
        if not self._enabled:
            return False
        try:
            prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "CRITICAL": "🚨"}.get(level, "")
            full_message = f"{prefix} {message}" if prefix else message
            response = requests.post(
                self.API_URL.format(token=self.token),
                json={"chat_id": self.chat_id, "text": full_message, "parse_mode": "HTML"},
                timeout=self.TIMEOUT_SECONDS,
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Telegram send failed (non-fatal): {e}")
            return False

    def send_trade_alert(self, side: str, pair: str, price: float,
                         quantity: float, pnl_pct: Optional[float] = None):
        emoji = "🟢" if side == "BUY" else "🔴"
        msg = (f"{emoji} <b>{side}</b> {pair}\n"
               f"Price: ${price:,.4f}\n"
               f"Qty: {quantity:.6f}")
        if pnl_pct is not None:
            emoji = "✅" if pnl_pct > 0 else "❌"
            msg += f"\nPnL: {emoji} {pnl_pct:+.2%}"
        self.send(msg)

    def send_regime_change(self, old_regime: str, new_regime: str,
                           ema20: float, ema50: float, adx: float):
        self.send(
            f"📊 <b>REGIME CHANGE</b>\n"
            f"{old_regime} → {new_regime}\n"
            f"EMA20={ema20:.0f} EMA50={ema50:.0f} ADX={adx:.1f}",
            level="WARN"
        )

    def send_circuit_breaker(self, drawdown: float, portfolio_value: float, hwm: float):
        self.send(
            f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n"
            f"Drawdown: {drawdown:.1%}\n"
            f"Portfolio: ${portfolio_value:,.0f}\n"
            f"HWM: ${hwm:,.0f}\n"
            f"NO NEW POSITIONS until recovery",
            level="CRITICAL"
        )

    def send_heartbeat(self, free_usd: float, open_positions: list[str],
                       uptime_hours: float):
        pos_str = ", ".join(open_positions) if open_positions else "none"
        self.send(
            f"💓 Heartbeat\n"
            f"Free USD: ${free_usd:,.0f}\n"
            f"Positions: {pos_str}\n"
            f"Uptime: {uptime_hours:.1f}h"
        )

    def send_error(self, context: str, error: str):
        self.send(
            f"❌ <b>ERROR</b>: {context}\n<code>{error[:500]}</code>",
            level="ERROR"
        )
```

---

## Alert Taxonomy

Alerts are categorised by urgency. Tune your phone notifications accordingly:

**Immediate action required (CRITICAL):**
- Circuit breaker triggered
- 3+ consecutive API failures
- Startup reconciliation finds a major discrepancy
- Bot process died and could not restart (you'd get no heartbeat)

**Review when you can (WARN):**
- Regime change (BULL → BEAR or BEAR → BULL)
- Timestamp offset > 10 seconds
- Memory usage > 80%
- Order rejected (Success=false)

**Informational (INFO):**
- Every trade executed (BUY and SELL)
- Bot startup and shutdown
- Heartbeat (every 10 minutes — proves bot is alive)
- Daily PnL summary (once per day at midnight UTC)

**Do not alert on (suppress):**
- HOLD signals (too frequent, no value)
- Every ticker poll (too frequent, no value)
- Every state.json write (too frequent, no value)
- Normal reconciliation completion (no discrepancy)

---

## `monitoring/healthcheck.py`

```python
import psutil
import time
import logging

logger = logging.getLogger(__name__)

class HealthChecker:
    """
    Monitors EC2 instance resources and sends alerts when thresholds are exceeded.
    Also manages the periodic heartbeat.
    """

    MEMORY_WARN_PCT = 80.0
    DISK_WARN_PCT = 85.0
    CPU_WARN_PCT = 90.0
    CPU_SUSTAINED_SECONDS = 120  # Must be high for this long before alerting
    HEARTBEAT_INTERVAL_SECONDS = 600  # 10 minutes

    def __init__(self, telegram, oms, client):
        self.telegram = telegram
        self.oms = oms
        self.client = client
        self._last_heartbeat = 0.0
        self._bot_start_time = time.time()
        self._cpu_high_since = 0.0

    def check_all(self):
        """Run all health checks. Call once per main loop iteration."""
        self._check_memory()
        self._check_disk()
        self._check_cpu()
        self._maybe_send_heartbeat()

    def _check_memory(self):
        mem = psutil.virtual_memory()
        if mem.percent > self.MEMORY_WARN_PCT:
            logger.warning(f"High memory usage: {mem.percent:.1f}%")
            self.telegram.send(
                f"⚠️ High memory: {mem.percent:.1f}% "
                f"({mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB)",
                level="WARN"
            )

    def _check_disk(self):
        disk = psutil.disk_usage("/")
        if disk.percent > self.DISK_WARN_PCT:
            logger.warning(f"High disk usage: {disk.percent:.1f}%")
            self.telegram.send(
                f"⚠️ High disk: {disk.percent:.1f}% used",
                level="WARN"
            )

    def _check_cpu(self):
        cpu = psutil.cpu_percent(interval=1)
        if cpu > self.CPU_WARN_PCT:
            if self._cpu_high_since == 0:
                self._cpu_high_since = time.time()
            elif time.time() - self._cpu_high_since > self.CPU_SUSTAINED_SECONDS:
                self.telegram.send(
                    f"⚠️ Sustained high CPU: {cpu:.1f}% for "
                    f"{(time.time()-self._cpu_high_since)/60:.1f} min",
                    level="WARN"
                )
                self._cpu_high_since = time.time()  # Reset to avoid spam
        else:
            self._cpu_high_since = 0.0

    def _maybe_send_heartbeat(self):
        if time.time() - self._last_heartbeat < self.HEARTBEAT_INTERVAL_SECONDS:
            return
        self._last_heartbeat = time.time()

        # Get current state for heartbeat message
        balance = self.client.get_balance()
        free_usd = 0.0
        if balance.get("Success"):
            free_usd = balance["Wallet"].get("USD", {}).get("Free", 0)

        positions = list(self.oms.get_all_positions().keys())
        uptime = (time.time() - self._bot_start_time) / 3600

        self.telegram.send_heartbeat(free_usd, positions, uptime)
        logger.debug(f"Heartbeat sent (uptime={uptime:.1f}h)")
```

---

## Systemd Integration for Process Monitoring

The bot runs as a systemd service. This means the OS handles restart on crash, startup on boot, and log collection via journalctl. Systemd is a more reliable process manager than tmux or screen.

`/etc/systemd/system/tradingbot.service`:
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
# Write stdout/stderr to systemd journal
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tradingbot

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable tradingbot
sudo systemctl start tradingbot

# Monitor logs
journalctl -u tradingbot -f

# Test restart recovery (do this before competition)
sudo systemctl stop tradingbot
sleep 15
sudo systemctl status tradingbot  # Should have auto-restarted
```

---

## Daily PnL Summary

Send one summary message per day at midnight UTC. This gives you a daily performance snapshot without requiring manual SSH:

```python
def send_daily_summary(telegram, client, oms, start_of_day_value: float):
    balance = client.get_balance()
    if not balance.get("Success"):
        return
    free_usd = balance["Wallet"].get("USD", {}).get("Free", 0)
    position_values = sum(oms.get_position_value(p) for p in oms.get_all_positions())
    current_total = free_usd + position_values
    day_pnl_pct = (current_total - start_of_day_value) / start_of_day_value

    telegram.send(
        f"📈 <b>Daily Summary</b>\n"
        f"Portfolio: ${current_total:,.0f}\n"
        f"Day PnL: {day_pnl_pct:+.2%}\n"
        f"Open positions: {list(oms.get_all_positions().keys()) or 'none'}"
    )
```

---

## Failure Modes This Layer Prevents

| Failure | Prevention |
|---|---|
| Silent outage going unnoticed for hours | 10-minute heartbeat; silence = problem |
| No debug trail for post-incident analysis | JSON structured logs written for every event |
| Monitoring failure affecting trading | All Telegram calls wrapped in try/except; failures are non-fatal |
| EC2 running out of disk from log buildup | RotatingFileHandler: 10MB × 10 files = 100MB max |
| High memory causing OOM kill | psutil check + alert at 80% gives time to intervene |
| Missing context when reviewing logs | JSON format includes timestamp, logger name, and full message |
| Alert fatigue from too-frequent notifications | Alert taxonomy: only meaningful events trigger Telegram |
