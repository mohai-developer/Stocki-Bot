import yfinance as yf
import pandas as pd
import os
import asyncio
import requests
from datetime import datetime
import anthropic
from telegram import Bot
from dotenv import load_dotenv
# ============================================================
# SETTINGS
# ============================================================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

STOCKS = ["NVDA", "MSFT", "AMZN", "ARM", "PLUG", "QCOM"]
STOCKS = ["NVDA", "MSFT", "AMZN", "ARM", "PLUG", "QCOM"]
DATA_FILE = "market_data.csv"

# ============================================================
# READ LATEST DATA FROM CSV
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
    except Exception as e:
        print(f"Error reading data for {symbol}: {e}")
        return None

# ============================================================
# GET STOCK NEWS FROM YFINANCE
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
    except Exception as e:
        print(f"News error for {symbol}: {e}")
        return "Could not fetch news"

# ============================================================
# GET MARKET & GEOPOLITICAL NEWS FROM NEWSAPI
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
    except Exception as e:
        print(f"NewsAPI error: {e}")
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
    except Exception as e:
        print(f"Geopolitical news error: {e}")
        return "Could not fetch geopolitical news"

# ============================================================
# ASK CLAUDE FOR ANALYSIS
# ============================================================
def analyze_with_claude(symbol, data, stock_news, market_news, geo_news):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        prompt = f"""
أنت محلل مالي محترف متخصص في الأسهم الأمريكية.

حلل السهم {symbol} بناءً على هذه المعلومات الشاملة:

📊 البيانات الفنية والأساسية:
{data}

📰 أخبار السهم المباشرة:
{stock_news}

🌍 أخبار السوق والاقتصاد:
{market_news}

⚠️ الأحداث الجيوسياسية المؤثرة:
{geo_news}

قدم تحليلاً شاملاً يتضمن:

1. **الوضع الفني الحالي:**
   - قراءة RSI و MFI و Bollinger Bands
   - اتجاه الترند يومي وأسبوعي
   - قوة حجم التداول

2. **الوضع الأساسي:**
   - مقارنة السهم بقطاعه
   - P/E والتقييم
   - تاريخ الأرباح القادمة

3. **تأثير الأخبار:**
   - كيف تؤثر أخبار السهم على حركته؟
   - هل الأحداث الجيوسياسية تشكل خطراً أو فرصة؟
   - تأثير قرارات Fed والتضخم على هذا السهم تحديداً

4. **إشارات Pre-Market و After-Hours:**
   - هل هناك نمط واضح؟
   - ماذا تقول هذه الأرقام عن اليوم القادم؟

5. **التوصية النهائية:**
   - للمتداول الذي لم يدخل: هل يدخل الآن أم ينتظر؟
   - للمتداول الداخل: هل يبقى أم يخرج؟
   - مستوى الثقة: منخفض / متوسط / عالي

6. **مستوى الخطر:**
   - ما هي المخاطر الرئيسية الآن؟
   - أين يضع وقف الخسارة المنطقي؟

كن دقيقاً وصريحاً، لا تعطي إجابات مبهمة.
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text

    except Exception as e:
        print(f"Claude error for {symbol}: {e}")
        return None

# ============================================================
# SEND TELEGRAM
# ============================================================
async def send_telegram(message):
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        if len(message) > 4000:
            parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
            for part in parts:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=part)
        else:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        print("Telegram message sent")
    except Exception as e:
        print(f"Telegram error: {e}")

# ============================================================
# MAIN JOB
# ============================================================
def run_analysis():
    print(f"\n--- Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")

    # Fetch market and geopolitical news once for all stocks
    print("Fetching market news...")
    market_news = get_market_news()
    
    print("Fetching geopolitical news...")
    geo_news = get_geopolitical_news()

    for symbol in STOCKS:
        print(f"\nAnalyzing {symbol}...")

        data = get_latest_data(symbol)
        if not data:
            print(f"No data found for {symbol}")
            continue

        print(f"Fetching news for {symbol}...")
        stock_news = get_stock_news(symbol)

        analysis = analyze_with_claude(symbol, data, stock_news, market_news, geo_news)
        if not analysis:
            continue

        message = f"""
🤖 تحليل {symbol}
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*30}

{analysis}

{'='*30}
        """

        asyncio.run(send_telegram(message))
        print(f"{symbol} analysis sent")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("Bot Advisor started...")
    print(f"Analyzing: {', '.join(STOCKS)}")
    run_analysis()