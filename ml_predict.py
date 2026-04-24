import json
from datetime import datetime, timezone

import ccxt
import pandas as pd
from xgboost import XGBClassifier


class MLModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    def prepare_data(self, df: pd.DataFrame):
        df = df.copy()

        df["returns"] = df["close"].pct_change()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["volatility"] = df["returns"].rolling(10).std()
        df["volume_ma10"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma10"]

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

        # EMA
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

        # Bollinger Bands
        df["boll_mid"] = df["close"].rolling(20).mean()
        df["boll_std"] = df["close"].rolling(20).std()
        df["boll_upper"] = df["boll_mid"] + 2 * df["boll_std"]
        df["boll_lower"] = df["boll_mid"] - 2 * df["boll_std"]
        df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]

        df = df.dropna().reset_index(drop=True)

        X = df[
            [
                "returns",
                "ma10",
                "ma20",
                "volatility",
                "volume_ratio",
                "rsi",
                "ema50",
                "ema200",
                "boll_mid",
                "boll_upper",
                "boll_lower",
                "boll_width",
            ]
        ]
        y = (df["close"].shift(-1) > df["close"]).astype(int)

        return X.iloc[:-1], y.iloc[:-1], df

    def train(self, df: pd.DataFrame):
        X, y, _ = self.prepare_data(df)
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame):
        X, _, prepared = self.prepare_data(df)
        last_x = X.tail(1)

        pred = int(self.model.predict(last_x)[0])
        probs = self.model.predict_proba(last_x)[0]
        confidence = float(max(probs))

        last_row = prepared.iloc[-1]

        return {
            "pred": pred,
            "confidence": confidence,
            "close": float(last_row["close"]),
            "returns": float(last_row["returns"]),
            "volatility": float(last_row["volatility"]),
            "volume_ratio": float(last_row["volume_ratio"]),
            "rsi": float(last_row["rsi"]),
            "ema50": float(last_row["ema50"]),
            "ema200": float(last_row["ema200"]),
            "boll_mid": float(last_row["boll_mid"]),
            "boll_upper": float(last_row["boll_upper"]),
            "boll_lower": float(last_row["boll_lower"]),
            "boll_width": float(last_row["boll_width"]),
        }


def fetch_data(symbol: str = "BTC/USDT", timeframe: str = "5m", limit: int = 250) -> pd.DataFrame:
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(
        bars,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    return df


def build_signal(symbol: str = "BTC/USDT", timeframe: str = "5m"):
    df = fetch_data(symbol=symbol, timeframe=timeframe, limit=250)

    model = MLModel()
    model.train(df)
    result = model.predict(df)

    confidence = result["confidence"]
    close_price = result["close"]
    returns = result["returns"]
    volume_ratio = result["volume_ratio"]
    rsi = result["rsi"]
    ema50 = result["ema50"]
    ema200 = result["ema200"]
    boll_mid = result["boll_mid"]
    boll_upper = result["boll_upper"]
    boll_lower = result["boll_lower"]
    boll_width = result["boll_width"]

    boll_position = "inside"
    if close_price >= boll_upper:
        boll_position = "above_upper"
    elif close_price <= boll_lower:
        boll_position = "below_lower"
    elif close_price > boll_mid:
        boll_position = "upper_half"
    else:
        boll_position = "lower_half"

    if confidence < 0.65:
        direction = "FLAT"
    else:
        direction = "LONG" if result["pred"] == 1 else "SHORT"

    # Secondary filters over ML
    if direction == "LONG":
        if volume_ratio < 0.8:
            direction = "FLAT"
        elif returns <= 0:
            direction = "FLAT"
        elif rsi >= 65:
            direction = "FLAT"
        elif close_price <= ema50:
            direction = "FLAT"
        elif boll_position == "above_upper":
            direction = "FLAT"

    elif direction == "SHORT":
        if volume_ratio < 0.8:
            direction = "FLAT"
        elif returns >= 0:
            direction = "FLAT"
        elif rsi <= 35:
            direction = "FLAT"
        elif close_price >= ema50:
            direction = "FLAT"
        elif boll_position == "below_lower":
            direction = "FLAT"
                elif boll_position == "below_lower":
            direction = "FLAT"

                elif boll_position == "below_lower":
            direction = "FLAT"

    entry = close_price
    volatility = abs(result["volatility"])
    price_move = entry * volatility

    stop_loss = None
    take_profit = None

    if direction == "LONG":
        stop_loss = entry - price_move * 1.5
        take_profit = entry + price_move * 3.0
    elif direction == "SHORT":
        stop_loss = entry + price_move * 1.5
        take_profit = entry - price_move * 3.0

    signal = {
        "entry": round(entry, 4),
        "stop_loss": round(stop_loss, 4) if stop_loss is not None else None,
        "take_profit": round(take_profit, 4) if take_profit is not None else None,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "confidence": round(confidence, 4),
        "close": round(result["close"], 4),
        "returns": round(result["returns"], 6),
        "volatility": round(result["volatility"], 6),
        "volume_ratio": round(result["volume_ratio"], 4),
        "rsi": round(result["rsi"], 2),
        "ema50": round(result["ema50"], 4),
        "ema200": round(result["ema200"], 4),
        "boll_mid": round(result["boll_mid"], 4),
        "boll_upper": round(result["boll_upper"], 4),
        "boll_lower": round(result["boll_lower"], 4),
        "boll_width": round(result["boll_width"], 6),
        "boll_position": boll_position,
        "selected_agent": "XGBOOST_PRO_ML",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    entry = signal["close"]
    volatility = abs(signal["volatility"])
    price_move = entry * volatility

    stop_loss = None
    take_profit = None
    rr_ratio = None

    if direction == "LONG":
        stop_loss = entry - price_move * 1.5
        take_profit = entry + price_move * 3.0
        rr_ratio = 2.0
    elif direction == "SHORT":
        stop_loss = entry + price_move * 1.5
        take_profit = entry - price_move * 3.0
        rr_ratio = 2.0

    signal["entry"] = round(entry, 4)
    signal["stop_loss"] = round(stop_loss, 4) if stop_loss is not None else None
    signal["take_profit"] = round(take_profit, 4) if take_profit is not None else None
    signal["risk_reward"] = rr_ratio

    return signal


def save_signal(signal: dict, path: str = "signal.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    signal = build_signal()
    save_signal(signal)
    print(json.dumps(signal, ensure_ascii=False, indent=2))
