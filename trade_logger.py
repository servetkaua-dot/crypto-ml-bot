import json
import os
from datetime import datetime, timezone

TRADES_FILE = "trades_log.json"


def load_trades():
    if not os.path.exists(TRADES_FILE):
        return []

    with open(TRADES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_trades(trades):
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)


def log_signal(signal, trade_result=None):
    trades = load_trades()

    trade = {
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": signal.get("symbol"),
        "timeframe": signal.get("timeframe"),
        "direction": signal.get("direction"),
        "entry": signal.get("entry"),
        "stop_loss": signal.get("stop_loss"),
        "tp1": signal.get("tp1"),
        "tp2": signal.get("tp2"),
        "take_profit": signal.get("take_profit"),
        "confidence": signal.get("confidence"),
        "rsi": signal.get("rsi"),
        "volume_ratio": signal.get("volume_ratio"),
        "volatility": signal.get("volatility"),
        "boll_position": signal.get("boll_position"),
        "result": None,
        "trade_result": trade_result,
    }

    trades.append(trade)
    save_trades(trades)
    return trade


def mark_last_trade_result(result):
    trades = load_trades()

    if not trades:
        return None

    trades[-1]["result"] = result
    trades[-1]["result_time"] = datetime.now(timezone.utc).isoformat()

    save_trades(trades)
    return trades[-1]
