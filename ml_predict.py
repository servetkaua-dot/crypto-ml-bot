import json
import os
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

        df = df.dropna().reset_index(drop=True)

        X = df[["returns", "ma10", "ma20", "volatility", "volume_ratio"]]
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

    if confidence < 0.55:
        direction = "FLAT"
    else:
        direction = "LONG" if result["pred"] == 1 else "SHORT"

    signal = {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "confidence": round(confidence, 4),
        "close": round(result["close"], 4),
        "returns": round(result["returns"], 6),
        "volatility": round(result["volatility"], 6),
        "volume_ratio": round(result["volume_ratio"], 4),
        "selected_agent": "XGBOOST_PRO_ML",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return signal


def save_signal(signal: dict, path: str = "signal.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    signal = build_signal()
    save_signal(signal)
    print(json.dumps(signal, ensure_ascii=False, indent=2))
