import os
import time
import hmac
import json
import hashlib
import requests
from decimal import Decimal, ROUND_DOWN


BASE_URL = "https://api-testnet.bybit.com" if os.getenv("BYBIT_TESTNET", "false").lower() == "true" else "https://api.bybit.com"

API_KEY = os.getenv("BYBIT_API_KEY", "").strip()
API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()

DRY_RUN = os.getenv("BYBIT_DRY_RUN", "true").lower() == "true"
CATEGORY = os.getenv("BYBIT_CATEGORY", "linear")
SYMBOL = os.getenv("BYBIT_SYMBOL", "BTCUSDT")
TRADE_USDT_SIZE = float(os.getenv("TRADE_USDT_SIZE", "10"))
RECV_WINDOW = "5000"


def _ts():
    return str(int(time.time() * 1000))


def _sign(payload: str, timestamp: str):
    raw = timestamp + API_KEY + RECV_WINDOW + payload
    return hmac.new(
        API_SECRET.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def _headers(payload: str):
    timestamp = _ts()
    return {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-SIGN": _sign(payload, timestamp),
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "Content-Type": "application/json",
    }


def _post(path: str, body: dict):
    payload = json.dumps(body, separators=(",", ":"))
    r = requests.post(
        BASE_URL + path,
        headers=_headers(payload),
        data=payload,
        timeout=20,
    )
    try:
        return r.json()
    except Exception:
        return {"error": r.text}


def _get(path: str, params: dict):
    r = requests.get(BASE_URL + path, params=params, timeout=20)
    try:
        return r.json()
    except Exception:
        return {"error": r.text}


def round_qty(qty: float, step: str = "0.001") -> str:
    q = Decimal(str(qty))
    s = Decimal(step)
    return str((q / s).to_integral_value(rounding=ROUND_DOWN) * s)


def get_last_price():
    data = _get("/v5/market/tickers", {
        "category": CATEGORY,
        "symbol": SYMBOL,
    })
    return float(data["result"]["list"][0]["lastPrice"])


def calc_qty_by_usdt(usdt_size: float, price: float):
    qty = usdt_size / price
    return round_qty(qty, "0.001")


def execute_trade_from_signal(signal: dict):
    direction = signal.get("direction")

    if direction not in ("LONG", "SHORT"):
        return {"skipped": True, "reason": "FLAT_OR_UNKNOWN"}

    if not API_KEY or not API_SECRET:
        return {"skipped": True, "reason": "BYBIT_API_KEY_OR_SECRET_MISSING"}

    entry = signal.get("entry")
    stop_loss = signal.get("stop_loss")
    tp1 = signal.get("tp1")
    tp2 = signal.get("tp2")
    take_profit = signal.get("take_profit") or tp2

    last_price = get_last_price()
    qty = calc_qty_by_usdt(TRADE_USDT_SIZE, last_price)

    side = "Buy" if direction == "LONG" else "Sell"

    order = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "side": side,
        "orderType": "Market",
        "qty": qty,
        "timeInForce": "IOC",
        "positionIdx": 0,
    }

    if stop_loss:
        order["stopLoss"] = str(stop_loss)

    if take_profit:
        order["takeProfit"] = str(take_profit)

    result = {
        "dry_run": DRY_RUN,
        "direction": direction,
        "symbol": SYMBOL,
        "last_price": last_price,
        "entry_from_signal": entry,
        "qty": qty,
        "trade_usdt_size": TRADE_USDT_SIZE,
        "tp1_50_percent": tp1,
        "tp2_50_percent": tp2,
        "order": order,
    }

    if DRY_RUN:
        return result

    response = _post("/v5/order/create", order)
    result["bybit_response"] = response
    return result
