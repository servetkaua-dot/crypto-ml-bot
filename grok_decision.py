import requests
import json
import os
from datetime import datetime

GROK_API_KEY = os.getenv("GROK_API_KEY")  # добавишь в .env

def get_grok_decision(market_data: dict, coin: str) -> dict:
    """
    Отправляет текущий рынок + последние новости Grok и получает решение
    """
    prompt = f"""
Ты — опытный крипто-трейдер. Сейчас {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC.

Монета: {coin}
Текущие данные:
- Цена: {market_data.get('close')}
- RSI: {market_data.get('rsi')}
- Trend (HTF): {market_data.get('htf_trend')}
- Bollinger Position: {market_data.get('boll_position')}
- Volume Ratio: {market_data.get('volume_ratio')}
- Market Mode: {market_data.get('market_mode')}
- ML Prediction: {market_data.get('pred_direction')} (confidence {market_data.get('confidence')})

Последние новости / события по {coin} (если есть): {market_data.get('news', 'Нет свежих новостей')}

Экономический календарь сегодня: {market_data.get('calendar', 'Без важных событий')}

Задача:
1. Учти общий рыночный контекст (Bitcoin dominance, настроение рынка).
2. Дай **финальное решение**: LONG, SHORT или FLAT.
3. Укажи силу сигнала (1-10).
4. Кратко объясни почему.

Ответ строго в JSON:
{{
  "decision": "LONG|SHORT|FLAT",
  "strength": 7,
  "reason": "Краткое объяснение",
  "suggested_sl": число,
  "suggested_tp": число
}}
"""

    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-3",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400
            },
            timeout=15
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        else:
            print(f"[GROK] API Error: {response.status_code}")
            return {"decision": "FLAT", "strength": 3, "reason": "API error"}
            
    except Exception as e:
        print(f"[GROK] Error: {e}")
        return {"decision": "FLAT", "strength": 3, "reason": "Exception"}

# Для теста
if __name__ == "__main__":
    test_data = {"close": 2120, "rsi": 62, "htf_trend": "UP", "pred_direction": "LONG", "confidence": 0.68}
    print(get_grok_decision(test_data, "ETH/USDT"))
