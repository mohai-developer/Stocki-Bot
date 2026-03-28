from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import asyncio
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import anthropic
from telegram import Bot
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["https://mohai-developer.github.io"])

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DATA_FILE = "market_data.csv"

def get_stock_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return "No news available"
        news_text = []
        for item in news[:5]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            news_text.append(f"- {title} ({publisher})")
        return "\n".join(news_text)
    except:
        return "Could not fetch news"

def get_market_news():
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {"apiKey": NEWS_API_KEY, "category": "business", "language": "en", "pageSize": 5}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") != "ok":
            return "Could not fetch market news"
        news_text = []
        for article in data.get("articles", [])[:5]:
            title = article.get("title", "")
            source = article.get("source", {}).get("name", "")
            news_text.append(f"- {title} ({source})")
        return "\n".join(news_text)
    except:
        return "Could not fetch market news"

def get_geopolitical_news():
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"apiKey": NEWS_API_KEY, "q": "trade war OR interest rates OR Fed OR inflation OR geopolitical", "language": "en", "sortBy": "publishedAt", "pageSize": 5}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") != "ok":
            return "Could not fetch geopolitical news"
        news_text = []
        for article in data.get("articles", [])[:5]:
            title = article.get("title", "")
            source = article.get("source", {}).get("name", "")
            news_text.append(f"- {title} ({source})")
        return "\n".join(news_text)
    except:
        return "Could not fetch geopolitical news"

def get_latest_data(symbol):
    try:
        if not os.path.exists(DATA_FILE):
            return None
        df = pd.read_csv(DATA_FILE)
        df_symbol = df[df["symbol"] == symbol].tail(5)
        if df_symbol.empty:
            return None
        return df_symbol.to_dict(orient="records")
    except:
        return None

def analyze_general(symbol, report_type, data, stock_news, market_news, geo_news):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        if report_type == "summary":
            format_instruction = """
قدم ملخصاً مختصراً بهذا الشكل فقط:

📊 الوضع: [إيجابي/سلبي/محايد]
🎯 التوصية: [ادخل/انتظر/تجنب]

إذا كانت التوصية الدخول:
💰 سعر الدخول: $X
🎯 الهدف: $X
🛑 وقف الخسارة: $X
📊 R/R: 1:X
✅ مستوى الثقة: X%
⚠️ أهم خطر: [جملة واحدة]
"""
        else:
            format_instruction = """
قدم تقريراً مفصلاً:

1. الوضع الفني: RSI، MACD، Bollinger، الترند، حجم التداول
2. الوضع الأساسي: القطاع، P/E، الأرباح القادمة
3. تأثير الأخبار والأحداث الجيوسياسية
4. إشارات Pre-Market و After-Hours

5. التوصية النهائية:
   - الوضع: إيجابي أم سلبي
   - القرار: ادخل / انتظر / تجنب

   إذا كانت التوصية الدخول:
   💰 سعر الدخول المقترح: $X (مع السبب الفني)
   🎯 الهدف الأول: $X
   🎯 الهدف الثاني: $X
   🛑 وقف الخسارة: $X (مع السبب)
   📊 نسبة R/R: 1:X
   ⏰ توقيت الدخول: [الشرط الفني المطلوب]

6. مستوى الثقة: X% مع 3 أسباب و3 مخاطر
"""

        prompt = f"""
أنت محلل مالي محترف متخصص في الأسهم الأمريكية وخبير في Swing Trading.

حلل السهم {symbol} بناءً على:
البيانات الفنية: {data}
أخبار السهم: {stock_news}
أخبار السوق: {market_news}
الأحداث الجيوسياسية: {geo_news}

{format_instruction}

كن دقيقاً بالأرقام ولا تعطِ إجابات مبهمة.
"""
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_position(symbol, entry_price, data, stock_news, market_news, geo_news):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        prompt = f"""
أنت محلل مالي محترف متخصص في الأسهم الأمريكية وخبير في Swing Trading.

المستخدم داخل صفقة في {symbol} بسعر دخول ${entry_price}.

حلل وضعه بناءً على:
البيانات الفنية: {data}
أخبار السهم: {stock_news}
أخبار السوق: {market_news}
الأحداث الجيوسياسية: {geo_news}

قدم تقريراً مختصراً بهذا الشكل:

💼 سعر الدخول: ${entry_price}
📈 السعر الحالي: $X
📊 الربح/الخسارة الحالية: X%

🔍 تقييم الوضع: [جملتان فقط عن أهم ما يؤثر الآن]

التوصية:
إذا الوضع إيجابي:
✅ ابقَ في الصفقة
🎯 الهدف المقترح: $X
🛑 وقف الخسارة: $X
📊 R/R المتبقي: 1:X

إذا الوضع سلبي:
❌ اخرج من الصفقة
💰 سعر الخروج المقترح: $X
⚠️ السبب: [جملة واحدة واضحة]

✅ مستوى الثقة: X%
"""
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

async def send_telegram_async(chat_id, message):
    bot = Bot(token=TELEGRAM_TOKEN)
    if len(message) > 4000:
        parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for part in parts:
            await bot.send_message(chat_id=chat_id, text=part)
    else:
        await bot.send_message(chat_id=chat_id, text=message)

def send_telegram(chat_id, message):
    asyncio.run(send_telegram_async(chat_id, message))

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "time": datetime.now().strftime("%Y-%m-%d %H:%M")})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        body = request.json
        symbol = body.get("symbol", "").upper()
        analysis_type = body.get("analysis_type", "general")
        report_type = body.get("report_type", "full")
        entry_price = body.get("entry_price", "")

        if not symbol:
            return jsonify({"error": "Symbol required"}), 400

        print(f"Analyzing {symbol} — type: {analysis_type}")

        data = get_latest_data(symbol)
        stock_news = get_stock_news(symbol)
        market_news = get_market_news()
        geo_news = get_geopolitical_news()

        if analysis_type == "position":
            analysis = analyze_position(symbol, entry_price, data, stock_news, market_news, geo_news)
        else:
            analysis = analyze_general(symbol, report_type, data, stock_news, market_news, geo_news)

        return jsonify({"status": "success", "symbol": symbol, "analysis": analysis})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
