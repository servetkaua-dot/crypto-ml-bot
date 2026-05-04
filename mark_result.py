import sys
from trade_logger import mark_last_trade_result
from learning_manager import adapt_after_trade

if len(sys.argv) < 2:
    print("Usage: python mark_result.py win|loss|neutral")
    raise SystemExit

result = sys.argv[1].lower()

if result not in ("win", "loss", "neutral"):
    print("Result must be: win, loss, neutral")
    raise SystemExit

trade = mark_last_trade_result(result)

if not trade:
    print("No trades found")
    raise SystemExit

settings = adapt_after_trade(trade)

print("[OK] Trade marked:", result)
print("[LEARNING]", settings)
