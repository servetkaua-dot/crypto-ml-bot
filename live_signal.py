from ml_predict import build_signal, save_signal


def main():
    signal = build_signal(symbol="BTC/USDT", timeframe="5m")
    save_signal(signal, "signal.json")
    print("[OK] signal written to signal.json")
    print(signal)


if __name__ == "__main__":
    main()
