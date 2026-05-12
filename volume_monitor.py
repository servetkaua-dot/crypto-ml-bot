import os
import time
import subprocess
import ccxt

EXCHANGE = ccxt.binance({"enableRateLimit": True})

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

CHECK_INTERVAL = 30
PRICE_MOVE_LIMIT = 0.003
VOLUME_MULTIPLIER = 2.0
COOLDOWN_SECONDS = 300

last_trigger_time = {}


def get_candles(symbol):
    return EXCHANGE.fetch_ohlcv(symbol, timeframe="1m", limit=25)


def run_ml_signal():
    subprocess.run(["python", "live_signal.py"], check=False)
    subprocess.run(["python", "execute_signal.py"], check=False)


def check_symbol(symbol):
    candles = get_candles(symbol)

    last = candles[-1]
    prev = candles[-2]

    last_close = last[4]
    prev_close = prev[4]

    price_move = (last_close - prev_close) / prev_close

    volumes = [c[5] for c in candles[:-1]]
    avg_volume = sum(volumes) / len(volumes)
    last_volume = last[5]

    volume_ratio = last_volume / avg_volume if avg_volume else 0

    now = time.time()
    last_time = last_trigger_time.get(symbol, 0)

    if now - last_time < COOLDOWN_SECONDS:
        return

    if abs(price_move) >= PRICE_MOVE_LIMIT and volume_ratio >= VOLUME_MULTIPLIER:
        last_trigger_time[symbol] = now

        print(
            f"[TRIGGER] {symbol} "
            f"move={round(price_move * 100, 2)}% "
            f"volume=x{round(volume_ratio, 2)}"
        )

        run_ml_signal()


def main():
    while True:
        for symbol in SYMBOLS:
            try:
                check_symbol(symbol)
            except Exception as e:
                print("[ERROR]", symbol, e)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
