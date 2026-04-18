import json
import os
import requests


BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def load_signal(path="signal.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def send_telegram_message(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        print(text)
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    r = requests.post(url, json=payload, timeout=20)
    print(r.text)


def main():
    signal = load_signal()

    msg = (
        f"🔥 PRO ML SIGNAL\n\n"
        f"Symbol: {signal['symbol']}\n"
        f"Timeframe: {signal['timeframe']}\n"
        f"Direction: {signal['direction']}\n"
        f"Confidence: {signal['confidence']}\n"
        f"Close: {signal['close']}\n"
        f"Returns: {signal['returns']}\n"
        f"Volatility: {signal['volatility']}\n"
        f"Volume Ratio: {signal['volume_ratio']}\n"
        f"Agent: {signal['selected_agent']}\n"
        f"Time: {signal['timestamp']}"
    )

    send_telegram_message(msg)


if __name__ == "__main__":
    main()
