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
CORS(app)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DATA_FILE = "market_data.csv"

# ============================================================
# GET STOCK NEWS
# ============================================================
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

# ============================================================
# GET MARKET NEWS
# ============================================================
def get_market_news():
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": NEWS_API_KEY,
            "category": "business",
            "language": "en",
            "pageSize": 5
        }
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

# ============================================================
# GET GEOPOLITICAL NEWS
# ============================================================
def get_geopolitical_news():
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWS_API_KEY,
            "q": "trade war OR interest rates OR Fed OR inflation OR geopolitical OR sanctions",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5
        }
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

# ============================================================
# GET LATEST DATA FROM CSV
# ============================================================
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

# ============================================================
# ANALYZE WITH CLAUDE
# ============================================================
def analyze_with_claude(symbol, report_type, entry, target, stop, data, stock_news, market_news, geo_news):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        summary_note = "أعطِ ملخصاً مختصراً جداً: القرار، سعر الدخول، الهدف، الوقف، ومستوى الثقة فقط بدون تفاصيل." if report_type == "summary" else ""

        entry_note = f"المستخدم دخل بسعر ${entry}. هدفه ${target}. وقف الخسارة ${stop}." if entry else ""

        prompt = f"""
أنت محلل مالي محترف متخصص في الأسهم الأمريكية وخبير في Swing Trading.

{entry_note}

حلل السهم {symbol} بناءً على:

📊 البيانات الفنية:
{data}

📰 أخبار السهم:
{stock_news}

🌍 أخبار السوق:
{market_news}

⚠️ الأحداث الجيوسياسية:
{geo_news}

{summary_note if report_type == 'summary' else '''قدم التقرير بهذا الشكل:

1. الوضع الفني: RSI، MACD، Bollinger، الترند
2. الوضع الأساسي: القطاع، P/E، الأرباح القادمة
3. تأثير الأخبار والأحداث الجيوسياسية
4. إشارات Pre-Market و After-Hours
5. خطة التنفيذ:
   - القرار: ادخل / انتظر / اخرج
   - سعر الدخول المقترح
   - الهدف الأول والثاني
   - وقف الخسارة مع السبب
   - نسبة Risk/Reward
6. توقيت الدخول: ما الشرط الفني المطلوب؟
7. مستوى الثقة: نسبة مئوية + 3 أسباب + 3 مخاطر'''}

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

# ============================================================
# SEND TELEGRAM
# ============================================================
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

# ============================================================
# API ROUTES
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "time": datetime.now().strftime("%Y-%m-%d %H:%M")})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        body = request.json
        symbol = body.get("symbol", "").upper()
        report_type = body.get("report_type", "full")
        entry = body.get("entry", "")
        target = body.get("target", "")
        stop = body.get("stop", "")
        chat_id = body.get("chat_id", "")

        if not symbol:
            return jsonify({"error": "Symbol required"}), 400

        print(f"Analyzing {symbol}...")

        data = get_latest_data(symbol)
        stock_news = get_stock_news(symbol)
        market_news = get_market_news()
        geo_news = get_geopolitical_news()

        analysis = analyze_with_claude(
            symbol, report_type, entry, target, stop,
            data, stock_news, market_news, geo_news
        )

        message = f"""
🤖 تحليل {symbol}
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*35}

{analysis}

{'='*35}
Stocki Bot — AI-Powered Analysis
        """

        if chat_id:
            send_telegram(chat_id, message)

        return jsonify({
            "status": "success",
            "symbol": symbol,
            "analysis": analysis
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
