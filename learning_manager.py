import json
import os

LEARNING_FILE = "learning.json"

DEFAULT_SETTINGS = {
    "min_confidence": 0.62,
    "min_volume": 0.08,
    "long_rsi_max": 70,
    "short_rsi_min": 30,
    "max_boll_width_min": 0.0015
}


def load_learning():
    if not os.path.exists(LEARNING_FILE):
        save_learning(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()

    with open(LEARNING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for k, v in DEFAULT_SETTINGS.items():
        data.setdefault(k, v)

    return data


def save_learning(settings):
    with open(LEARNING_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def adapt_after_trade(trade):
    settings = load_learning()

    result = trade.get("result")
    direction = trade.get("direction")
    rsi = float(trade.get("rsi", 50))
    volume_ratio = float(trade.get("volume_ratio", 0))
    confidence = float(trade.get("confidence", 0))

    if result == "loss":
        if confidence < 0.70:
            settings["min_confidence"] = min(settings["min_confidence"] + 0.01, 0.80)

        if volume_ratio < settings["min_volume"]:
            settings["min_volume"] = min(settings["min_volume"] + 0.01, 0.30)

        if direction == "LONG" and rsi > 60:
            settings["long_rsi_max"] = max(settings["long_rsi_max"] - 1, 58)

        if direction == "SHORT" and rsi < 40:
            settings["short_rsi_min"] = min(settings["short_rsi_min"] + 1, 42)

    elif result == "win":
        settings["min_confidence"] = max(settings["min_confidence"] - 0.005, 0.55)
        settings["min_volume"] = max(settings["min_volume"] - 0.005, 0.03)

    save_learning(settings)
    return settings
