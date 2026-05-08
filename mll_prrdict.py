# mll_prrdict.py
# Clean PRO ML prediction module:
# - 5m entry timeframe
# - 15m / 30m / 1h / 4h / 1d multi-timeframe confirmation
# - internal liquidity heatmap
# - AUTO market mode: FLAT / TREND / BREAKOUT / WAIT
# - adaptive learning.json filters
# - signals_journal.json logging
# - TP1/TP2 partial targets
#
# Required packages:
# pip install ccxt pandas scikit-learn

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import ccxt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


SIGNAL_LOG_FILE = "signals_journal.json"
LEARNING_FILE = "learning.json"


DEFAULT_LEARNING = {
    "min_confidence": 0.62,
    "min_volume": 0.08,
    "long_rsi_max": 70,
    "short_rsi_min": 30,
    "min_boll_width": 0.0015,
    "breakout_confidence": 0.60,
    "trend_votes_required": 3
}


def load_learning(path: str = LEARNING_FILE) -> Dict[str, Any]:
    """Load adaptive filter settings. Creates defaults if missing."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_LEARNING, f, ensure_ascii=False, indent=2)
        return DEFAULT_LEARNING.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return DEFAULT_LEARNING.copy()

    for k, v in DEFAULT_LEARNING.items():
        data.setdefault(k, v)

    return data


def save_signal_journal(signal: Dict[str, Any], path: str = SIGNAL_LOG_FILE) -> None:
    """Append every generated signal to journal for later analysis/training."""
    logs = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(signal)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def fetch_data(symbol: str = "BTC/USDT", timeframe: str = "5m", limit: int = 250) -> pd.DataFrame:
    """Fetch OHLCV from Binance via ccxt."""
    exchange = ccxt.binance({"enableRateLimit": True})
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(
        bars,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators used by ML and rule engine."""
    df = df.copy()

    df["returns"] = df["close"].pct_change()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["volatility"] = df["returns"].rolling(20).std()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
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

    return df.dropna().reset_index(drop=True)


class MLModel:
    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
        )

    def prepare_data(self, df: pd.DataFrame):
        df = add_indicators(df)

        features = [
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

        x = df[features]
        y = (df["close"].shift(-1) > df["close"]).astype(int)

        return x.iloc[:-1], y.iloc[:-1], df

    def train(self, df: pd.DataFrame) -> None:
        x, y, _ = self.prepare_data(df)
        self.model.fit(x, y)

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        x, _, prepared = self.prepare_data(df)
        last_x = x.tail(1)

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


def tf_trend(df: pd.DataFrame) -> str:
    """Trend for a single timeframe using EMA50/EMA200 structure."""
    data = add_indicators(df)
    close = float(data["close"].iloc[-1])
    ema50 = float(data["ema50"].iloc[-1])
    ema200 = float(data["ema200"].iloc[-1])

    if close > ema50 > ema200:
        return "UP"
    if close < ema50 < ema200:
        return "DOWN"
    return "NEUTRAL"


def build_liquidity_heatmap(df: pd.DataFrame, close_price: float, lookback: int = 120) -> Dict[str, Any]:
    """
    Internal heatmap approximation:
    - recent high = liquidity above
    - recent low = liquidity below
    - bias based on distance + volume factor
    """
    data = df.tail(lookback).copy()

    recent_high = float(data["high"].max())
    recent_low = float(data["low"].min())

    avg_volume = float(data["volume"].mean())
    last_volume = float(data["volume"].iloc[-1])
    volume_factor = last_volume / avg_volume if avg_volume else 1.0

    upper_distance = abs(recent_high - close_price)
    lower_distance = abs(close_price - recent_low)

    if lower_distance < upper_distance and volume_factor > 1.1:
        liq_bias = "DOWN"
    elif upper_distance < lower_distance and volume_factor > 1.1:
        liq_bias = "UP"
    else:
        liq_bias = "NEUTRAL"

    return {
        "liq_above": round(recent_high, 4),
        "liq_below": round(recent_low, 4),
        "liq_bias": liq_bias,
        "target_long": round(recent_high, 4),
        "target_short": round(recent_low, 4),
        "heatmap_volume_factor": round(volume_factor, 4),
    }


def build_signal(symbol: str = "BTC/USDT", timeframe: str = "5m") -> Dict[str, Any]:
    # === 1. DATA ===
    df = fetch_data(symbol=symbol, timeframe="5m", limit=250)

    df_15m = fetch_data(symbol=symbol, timeframe="15m", limit=250)
    df_30m = fetch_data(symbol=symbol, timeframe="30m", limit=250)
    df_1h = fetch_data(symbol=symbol, timeframe="1h", limit=250)
    df_4h = fetch_data(symbol=symbol, timeframe="4h", limit=250)
    df_1d = fetch_data(symbol=symbol, timeframe="1d", limit=250)

    # === 2. ML ===
    model = MLModel()
    model.train(df)
    result = model.predict(df)

    confidence = float(result["confidence"])
    close_price = float(result["close"])
    returns = float(result["returns"])
    volatility = abs(float(result["volatility"]))
    volume_ratio = float(result["volume_ratio"])
    rsi = float(result["rsi"])

    ema50 = float(result["ema50"])
    ema200 = float(result["ema200"])
    boll_mid = float(result["boll_mid"])
    boll_upper = float(result["boll_upper"])
    boll_lower = float(result["boll_lower"])
    boll_width = float(result["boll_width"])

    # === 3. LEARNING / ADAPTIVE FILTERS ===
    learning = load_learning()
    min_confidence = float(learning.get("min_confidence", 0.62))
    min_volume = float(learning.get("min_volume", 0.08))
    long_rsi_max = float(learning.get("long_rsi_max", 70))
    short_rsi_min = float(learning.get("short_rsi_min", 30))
    min_boll_width = float(learning.get("min_boll_width", 0.0015))
    breakout_confidence = float(learning.get("breakout_confidence", 0.60))
    trend_votes_required = int(learning.get("trend_votes_required", 3))

    # === 4. MULTI-TIMEFRAME TREND ===
    trend_5m = tf_trend(df)
    trend_15m = tf_trend(df_15m)
    trend_30m = tf_trend(df_30m)
    trend_1h = tf_trend(df_1h)
    trend_4h = tf_trend(df_4h)
    trend_1d = tf_trend(df_1d)

    tf_trends = [trend_15m, trend_30m, trend_1h, trend_4h, trend_1d]
    tf_up_score = tf_trends.count("UP")
    tf_down_score = tf_trends.count("DOWN")

    if tf_up_score >= trend_votes_required:
        htf_trend = "UP"
    elif tf_down_score >= trend_votes_required:
        htf_trend = "DOWN"
    else:
        htf_trend = "NEUTRAL"

    # === 5. INTERNAL HEATMAP ===
    heatmap = build_liquidity_heatmap(df, close_price)

    liq_above = heatmap["liq_above"]
    liq_below = heatmap["liq_below"]
    liq_bias = heatmap["liq_bias"]
    target_long = heatmap["target_long"]
    target_short = heatmap["target_short"]
    heatmap_volume_factor = heatmap["heatmap_volume_factor"]

    # === 6. STRUCTURE / SWEEP ===
    latest = df.iloc[-1]
    recent_high = float(df["high"].iloc[-21:-1].max())
    recent_low = float(df["low"].iloc[-21:-1].min())

    sweep_high = bool(latest["high"] > recent_high and latest["close"] < recent_high)
    sweep_low = bool(latest["low"] < recent_low and latest["close"] > recent_low)

    # === 7. BOLLINGER POSITION ===
    if close_price >= boll_upper:
        boll_position = "above_upper"
    elif close_price <= boll_lower:
        boll_position = "below_lower"
    elif close_price > boll_mid:
        boll_position = "upper_half"
    else:
        boll_position = "lower_half"

    # === 8. AUTO MARKET MODE ===
    strong_move_up = bool(
        returns > 0.001
        and volume_ratio > min_volume * 1.5
        and close_price > boll_upper
    )

    strong_move_down = bool(
        returns < -0.001
        and volume_ratio > min_volume * 1.5
        and close_price < boll_lower
    )

    if boll_width < min_boll_width:
        market_mode = "FLAT"
    elif strong_move_up or strong_move_down:
        market_mode = "BREAKOUT"
    elif htf_trend in ("UP", "DOWN"):
        market_mode = "TREND"
    else:
        market_mode = "WAIT"

    # === 9. DIRECTION ===
    pred_direction = "LONG" if int(result["pred"]) == 1 else "SHORT"
    direction = "FLAT"

    if market_mode == "TREND":
        if (
            htf_trend == "UP"
            and pred_direction == "LONG"
            and confidence >= min_confidence
            and volume_ratio >= min_volume
            and liq_bias != "DOWN"
        ):
            direction = "LONG"

        elif (
            htf_trend == "DOWN"
            and pred_direction == "SHORT"
            and confidence >= min_confidence
            and volume_ratio >= min_volume
            and liq_bias != "UP"
        ):
            direction = "SHORT"

    elif market_mode == "BREAKOUT":
        if strong_move_up and confidence >= breakout_confidence and htf_trend != "DOWN":
            direction = "LONG"
        elif strong_move_down and confidence >= breakout_confidence and htf_trend != "UP":
            direction = "SHORT"

    # === 10. SECONDARY FILTERS ===
    if direction == "LONG":
        if rsi >= long_rsi_max:
            direction = "FLAT"
        elif returns <= -0.0005:
            direction = "FLAT"
        elif not sweep_low and confidence < 0.80 and market_mode != "BREAKOUT":
            direction = "FLAT"
        elif htf_trend == "DOWN":
            direction = "FLAT"

    elif direction == "SHORT":
        if rsi <= short_rsi_min:
            direction = "FLAT"
        elif returns >= 0.0005:
            direction = "FLAT"
        elif not sweep_high and confidence < 0.80 and market_mode != "BREAKOUT":
            direction = "FLAT"
        elif htf_trend == "UP":
            direction = "FLAT"

    # === 11. SMART ENTRY / SL / TP ===
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None

    if direction == "LONG":
        entry = min(close_price, boll_mid)
        buffer = entry * 0.001

        stop_loss = recent_low - buffer
        risk = entry - stop_loss

        tp1 = boll_upper
        tp2 = target_long

        if tp1 <= entry:
            tp1 = entry + risk
        if tp2 <= entry:
            tp2 = entry + risk * 2

        take_profit = tp2

    elif direction == "SHORT":
        entry = max(close_price, boll_mid)
        buffer = entry * 0.001

        stop_loss = recent_high + buffer
        risk = stop_loss - entry

        tp1 = boll_lower
        tp2 = target_short

        if tp1 >= entry:
            tp1 = entry - risk
        if tp2 >= entry:
            tp2 = entry - risk * 2

        take_profit = tp2

    else:
        entry = close_price

    # === 12. SIGNAL ===
    signal = {
        "entry": round(entry, 4),
        "stop_loss": round(stop_loss, 4) if stop_loss is not None else None,
        "take_profit": round(take_profit, 4) if take_profit is not None else None,
        "tp1": round(tp1, 4) if tp1 is not None else None,
        "tp2": round(tp2, 4) if tp2 is not None else None,

        "partial_close_1": "50%",
        "partial_close_2": "50%",

        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "pred_direction": pred_direction,
        "confidence": round(confidence, 4),

        "close": round(close_price, 4),
        "returns": round(returns, 6),
        "volatility": round(volatility, 6),
        "volume_ratio": round(volume_ratio, 4),
        "rsi": round(rsi, 2),

        "ema50": round(ema50, 4),
        "ema200": round(ema200, 4),

        "boll_mid": round(boll_mid, 4),
        "boll_upper": round(boll_upper, 4),
        "boll_lower": round(boll_lower, 4),
        "boll_width": round(boll_width, 6),
        "boll_position": boll_position,

        "trend_5m": trend_5m,
        "trend_15m": trend_15m,
        "trend_30m": trend_30m,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "trend_1d": trend_1d,
        "tf_up_score": tf_up_score,
        "tf_down_score": tf_down_score,
        "htf_trend": htf_trend,

        "market_mode": market_mode,
        "strong_move_up": strong_move_up,
        "strong_move_down": strong_move_down,

        "sweep_high": sweep_high,
        "sweep_low": sweep_low,

        "liq_above": liq_above,
        "liq_below": liq_below,
        "liq_bias": liq_bias,
        "target_long": target_long,
        "target_short": target_short,
        "heatmap_volume_factor": heatmap_volume_factor,

        "selected_agent": "ML_PREDICT_TRUE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    save_signal_journal(signal)
    return signal


def save_signal(signal: Dict[str, Any], path: str = "signal.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    signal = build_signal()
    save_signal(signal)
    print(json.dumps(signal, ensure_ascii=False, indent=2))
