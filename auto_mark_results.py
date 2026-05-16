import json
from datetime import datetime, timezone
from binance.um_futures import UMFutures

TRADES_FILE = "trades_log.json"

client = UMFutures()

def load_trades():
    with open(TRADES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_trades(trades):
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)

def check_trade(trade):

    if trade.get("result"):
        return trade

    symbol = trade["symbol"].replace("/", "")
    interval = "5m"

    klines = client.klines(symbol=symbol, interval=interval, limit=20)

    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]

    tp = float(trade["tp1"] or trade["take_profit"])
    sl = float(trade["stop_loss"])
    direction = trade["direction"]

    result = None

    if direction == "LONG":

        if max(highs) >= tp:
            result = "win"

        elif min(lows) <= sl:
            result = "loss"

    elif direction == "SHORT":

        if min(lows) <= tp:
            result = "win"

        elif max(highs) >= sl:
            result = "loss"

    if result:
        trade["result"] = result
        trade["result_time"] = datetime.now(timezone.utc).isoformat()

    return trade

def main():

    trades = load_trades()

    updated = []

    for trade in trades:
        updated.append(check_trade(trade))

    save_trades(updated)

    print("[OK] Results checked")

if __name__ == "__main__":
    main()
