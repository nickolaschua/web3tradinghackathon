"""
Smoke test: verify that order placement works end-to-end against the Roostoo API.

Runs three checks:
  1. GET /v3/balance        — fetch starting USD balance
  2. GET /v3/ticker BTC/USD — get current BTC price
  3. POST /v3/place_order   — submit a tiny BUY (0.001 BTC, ~$70 notional)
  4. GET /v3/balance        — verify balance changed

Usage:
    python scripts/test_order_placement.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from bot.api.client import RoostooClient

# ── Auth ─────────────────────────────────────────────────────────────────────
api_key = os.environ.get("ROOSTOO_API_KEY") or os.environ.get("ROOSTOO_API_KEY_TEST")
secret = os.environ.get("ROOSTOO_SECRET") or os.environ.get("ROOSTOO_SECRET_TEST")
base_url = os.environ.get("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")

if not api_key or not secret:
    print("ERROR: No API credentials found. Set ROOSTOO_API_KEY_TEST / ROOSTOO_SECRET_TEST in .env")
    sys.exit(1)

print(f"Using API key: {api_key[:12]}... | base_url: {base_url}")

client = RoostooClient(api_key=api_key, secret=secret, base_url=base_url)

# ── Step 1: Balance before ────────────────────────────────────────────────────
print("\n[1] Fetching balance...")
balance_before = client.get_balance()
usd_before = balance_before.get("total_usd", 0.0)
print(f"    total_usd before: ${usd_before:,.2f}")
print(f"    USD free:         ${float(balance_before.get('USD', {}).get('Free', 0)):,.2f}")
print(f"    Success: {balance_before.get('Success')}")

# ── Step 2: Ticker ────────────────────────────────────────────────────────────
print("\n[2] Fetching BTC/USD ticker...")
ticker = client.get_ticker("BTC/USD")
price = float(ticker.get("Data", {}).get("BTC/USD", {}).get("LastPrice", 0.0))
print(f"    BTC/USD last price: ${price:,.2f}")
if price <= 0:
    print("ERROR: Could not get a valid price. Aborting.")
    sys.exit(1)

# ── Step 3: Place order ───────────────────────────────────────────────────────
qty = 0.001  # ~$70 notional at $70k BTC
print(f"\n[3] Placing BUY order: {qty} BTC/USD @ market (~${price * qty:,.2f} notional)...")
try:
    order_resp = client.place_order(pair="BTC/USD", side="BUY", quantity=qty)
    print(f"    Raw response: {order_resp}")
    success = order_resp.get("Success", False)
    print(f"    Success: {success}")
    if not success:
        print(f"    ErrMsg: {order_resp.get('ErrMsg', 'unknown')}")
except Exception as exc:
    print(f"ERROR placing order: {exc}")
    sys.exit(1)

# ── Step 4: Balance after ─────────────────────────────────────────────────────
print("\n[4] Fetching balance after order...")
balance_after = client.get_balance()
usd_after = balance_after.get("total_usd", 0.0)
print(f"    total_usd after:  ${usd_after:,.2f}")
print(f"    USD free:         ${float(balance_after.get('USD', {}).get('Free', 0)):,.2f}")
delta = usd_after - usd_before
print(f"    Delta:            ${delta:+,.2f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== RESULT ===")
if success:
    print("PASS — order submitted successfully. Check balance delta above.")
else:
    print("FAIL — order not accepted by exchange.")
