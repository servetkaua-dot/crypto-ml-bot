import json
import os
from datetime import datetime, timezone, timedelta

STATE_FILE = "trade_state.json"
COOLDOWN_MINUTES = int(os.getenv("TRADE_COOLDOWN_MINUTES", "30"))


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def can_trade(signal):
    state = load_state()
    direction = signal.get("direction")
    symbol = signal.get("symbol")
    key = f"{symbol}:{direction}"

    last_time = state.get(key)
    if not last_time:
        return True, "ok"

    last_dt = datetime.fromisoformat(last_time)
    now = datetime.now(timezone.utc)

    if now - last_dt < timedelta(minutes=COOLDOWN_MINUTES):
        return False, "cooldown"

    return True, "ok"


def mark_trade(signal):
    state = load_state()
    direction = signal.get("direction")
    symbol = signal.get("symbol")
    key = f"{symbol}:{direction}"
    state[key] = datetime.now(timezone.utc).isoformat()
    save_state(state)
