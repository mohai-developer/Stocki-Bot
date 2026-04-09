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
        news_items = []
        
        # المصدر الأول: NewsAPI بحث مخصص للسهم
        if NEWS_API_KEY:
            url = "https://newsapi.org/v2/everything"
            params = {
                "apiKey": NEWS_API_KEY,
                "q": f"{symbol} stock",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 8,
                "from": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("status") == "ok":
                for a in data.get("articles", [])[:8]:
                    title = a.get("title", "")
                    source = a.get("source", {}).get("name", "")
                    published = a.get("publishedAt", "")[:10]
                    if title and "[Removed]" not in title:
                        news_items.append(f"- {title} ({source}) {published}")
        
        # المصدر الثاني: yfinance كاحتياط
        if not news_items:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if news:
                for item in news[:5]:
                    title = item.get("title", "")
                    publisher = item.get("publisher", "")
                    if title:
                        news_items.append(f"- {title} ({publisher})")
        
        return "\n".join(news_items) if news_items else "No recent news found"
    except Exception as e:
        return f"Could not fetch news: {e}"

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
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="6mo", interval="1d")
        if len(hist) < 14:
            return None

        latest = hist.iloc[-1]
        prev = hist.iloc[-2]

        delta = hist["Close"].diff()
        gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rsi = round(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])), 2)

        ema12 = hist["Close"].ewm(span=12).mean()
        ema26 = hist["Close"].ewm(span=26).mean()
        macd = round((ema12 - ema26).iloc[-1], 4)
        signal_line = round((ema12 - ema26).ewm(span=9).mean().iloc[-1], 4)

        sma20 = hist["Close"].rolling(20).mean().iloc[-1]
        std20 = hist["Close"].rolling(20).std().iloc[-1]
        bb_upper = round(sma20 + 2 * std20, 2)
        bb_lower = round(sma20 - 2 * std20, 2)
        bb_position = round(((latest["Close"] - bb_lower) / (bb_upper - bb_lower)) * 100, 1)

        sma50 = round(hist["Close"].rolling(50).mean().iloc[-1], 2)
        avg_volume = round(hist["Volume"].rolling(20).mean().iloc[-1])
        volume_ratio = round(latest["Volume"] / avg_volume, 2)

        week_ago = hist["Close"].iloc[-6] if len(hist) >= 6 else hist["Close"].iloc[0]
        weekly_change = round(((latest["Close"] - week_ago) / week_ago) * 100, 2)

        return {
            "symbol": symbol,
            "close": round(latest["Close"], 2),
            "prev_close": round(prev["Close"], 2),
            "change_pct": round(((latest["Close"] - prev["Close"]) / prev["Close"]) * 100, 2),
            "weekly_change": weekly_change,
            "pre_market": round(info.get("preMarketPrice", 0) or 0, 2),
            "after_hours": round(info.get("postMarketPrice", 0) or 0, 2),
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal_line,
            "macd_cross": "bullish" if macd > signal_line else "bearish",
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_position": bb_position,
            "sma50": sma50,
            "volume": int(latest["Volume"]),
            "avg_volume": int(avg_volume),
            "volume_ratio": volume_ratio,
            "pe_ratio": round(info.get("trailingPE", 0) or 0, 2),
            "market_cap_b": round((info.get("marketCap", 0) or 0) / 1e9, 1),
            "earnings_date": str(info.get("earningsTimestamp", "N/A")),
        }
    except Exception as e:
        print(f"Error fetching live data for {symbol}: {e}")
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
أنت محلل Smart Money محترف. حلل السهم بهذا الإطار:

1. MARKET STATE (حالة السوق)
هل السوق نشط أم ميت؟ (الفوليوم + نطاق الشموع)
إذا كان ميتاً → القرار: NO TRADE فوراً

2. LOCATION (الموقع)
أين السعر الآن؟
- عند دعم/مقاومة سابقة واضحة؟
- عند منطقة فوليوم مرتفع سابقة؟
- في الفراغ؟ (= ضوضاء، يمنع الدخول)

3. VOLUME LAW (الجهد والنتيجة)
- فوليوم عالي + شمعة قوية = زخم حقيقي
- فوليوم عالي + شمعة صغيرة = امتصاص (انتظر تأكيد)
- فوليوم ضعيف + حركة قوية = حركة وهمية (يمنع)
- فوليوم منخفض + هبوط نحو دعم = Supply Dry-up (تحضير)

4. PATTERN (النمط الحالي)
- Compression: تضيق الشموع + انخفاض الفوليوم = طاقة مخزنة
- Spring: كسر دعم + عودة سريعة = تجميع سيولة
- Upthrust: كسر مقاومة + رفض = تصريف
- Absorption: فوليوم عالي بدون حركة = انتظر

5. STRUCTURE — MSB (كسر الهيكل)
هل حدث كسر هيكل؟
- صعود: كسر آخر Lower High
- هبوط: كسر آخر Higher Low
بدون MSB = لا دخول حتى لو النمط جميل

6. القرار النهائي
يجب توفر الثلاثة:
✓ Location واضح
✓ Pattern محدد
✓ MSB حدث

إذا الثلاثة متوفرة:
💰 سعر الدخول: $X (محافظ: انتظار إعادة اختبار / هجومي: مباشر بعد MSB)
🎯 الهدف الأول: $X
🎯 الهدف الثاني: $X
🛑 وقف الخسارة: $X (تحت/فوق منطقة السيولة)
📊 R/R: 1:X
⏰ شرط الدخول: [المحافظ أم الهجومي مع السبب]

إذا ناقص عنصر واحد:
→ NO TRADE مع ذكر السبب الدقيق

الوضع الأساسي: P/E، الأرباح القادمة، القطاع
تأثير الأخبار والماكرو
مستوى الثقة: X% مع 3 أسباب و3 مخاطر
"""

        # جلب توصية الخيارات
        options_rec = None
        if report_type == "full":
            options_rec = get_options_recommendation(symbol, data.get("close", 0) if data else 0)
        
        if options_rec:
            options_text = f"""

أضف في نهاية تقريرك هذا القسم كما هو:

───────────────────────────
📊 توصية الخيارات:
النوع: {options_rec['type']}
تاريخ الانتهاء: {options_rec['expiry']}
Strike المقترح: ${options_rec['suggested_strike']}
IV الحالي: {options_rec['iv_current']}% — {options_rec['iv_signal']} {options_rec['iv_rating']}
───────────────────────────
"""
        else:
            options_text = """

أضف في نهاية تقريرك هذا السطر كما هو:

───────────────────────────
📊 توصية الخيارات: لا توصية حالياً
───────────────────────────
"""

        prompt = f"""
أنت محلل مالي محترف متخصص في الأسهم الأمريكية وخبير في Swing Trading.

حلل السهم {symbol} بناءً على:
البيانات الفنية: {data}
أخبار السهم: {stock_news}
أخبار السوق: {market_news}
الأحداث الجيوسياسية: {geo_news}

{format_instruction}
{options_text}
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


@app.route("/dashboard", methods=["POST"])
def dashboard():
    try:
        body = request.json
        symbols = body.get("symbols", [])
        results = []
        for symbol in symbols:
            try:
                data = get_latest_data(symbol)
                if not data:
                    continue
                rsi = data.get("rsi", 50)
                macd_cross = data.get("macd_cross", "bearish")
                volume_ratio = data.get("volume_ratio", 1)
                change_pct = data.get("change_pct", 0)

                if rsi < 40 and macd_cross == "bullish" and volume_ratio > 1.2:
                    signal = "buy"
                    conf = min(int(70 + (40 - rsi)), 90)
                elif rsi > 70 and macd_cross == "bearish":
                    signal = "exit"
                    conf = min(int(60 + (rsi - 70)), 85)
                else:
                    signal = "wait"
                    conf = 50

                results.append({
                    "symbol": symbol,
                    "price": data.get("close", 0),
                    "change": data.get("change_pct", 0),
                    "signal": signal,
                    "conf": conf,
                    "rsi": rsi,
                    "pre_market": data.get("pre_market", 0),
                    "after_hours": data.get("after_hours", 0),
                })
            except Exception as e:
                print(f"Dashboard error for {symbol}: {e}")
                continue

        buy_count = len([r for r in results if r["signal"] == "buy"])
        return jsonify({"status": "success", "stocks": results, "buy_count": buy_count})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/council", methods=["POST"])
def council():
    try:
        body = request.json
        symbol = body.get("symbol", "").upper()
        if not symbol:
            return jsonify({"error": "Symbol required"}), 400

        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from council import run_council

        result = run_council(symbol)
        if not result:
            return jsonify({"error": f"Could not analyze {symbol}"}), 500

        data = result["data"]
        decision = result["decision"]
        memory = result.get("log", {})

        return jsonify({
            "status": "success",
            "symbol": symbol,
            "close": float(data["close"]),
            "rsi": float(data["rsi"]),
            "macd": data["macd_cross"],
            "trend": data["trend"],
            "edge_score": data["edge_score"],
            "verdict": f"{decision['Confidence']}%",
            "decision": decision["Decision"],
            "confidence": decision["Confidence"],
            "entry": decision["Plan"]["Entry"],
            "stop": decision["Plan"]["Stop"],
            "target": decision["Plan"]["Target1"],
            "target2": decision["Plan"]["Target2"],
            "rr": decision["Plan"]["RiskReward"],
            "scores": decision.get("Scores", {}),
            "memory_sessions": result.get("memory", {}).get("total_sessions", 0) if result.get("memory") else 0,
            "win_rate": result.get("memory", {}).get("win_rate", 0) if result.get("memory") else 0
        })

    except Exception as e:
        print(f"Council error: {e}")
        return jsonify({"error": str(e)}), 500



def get_options_recommendation(symbol, current_price):
    """جلب بيانات الخيارات وحساب IV Rank تقريبياً"""
    try:
        ticker = yf.Ticker(symbol)
        options_dates = ticker.options
        
        if not options_dates or len(options_dates) < 2:
            return None
        
        # نختار أقرب انتهاء بعد 30 يوم
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=30)
        
        best_date = None
        for d in options_dates:
            exp = datetime.strptime(d, "%Y-%m-%d")
            if exp > target_date:
                best_date = d
                break
        
        if not best_date:
            best_date = options_dates[-1]
        
        chain = ticker.option_chain(best_date)
        calls = chain.calls
        puts = chain.puts
        
        # IV الحالي (median للخيارات قرب السعر الحالي)
        atm_calls = calls[abs(calls['strike'] - current_price) < current_price * 0.05]
        atm_puts = puts[abs(puts['strike'] - current_price) < current_price * 0.05]
        
        if atm_calls.empty or atm_puts.empty:
            atm_calls = calls
            atm_puts = puts
        
        iv_calls = round(atm_calls['impliedVolatility'].median() * 100, 1)
        iv_puts = round(atm_puts['impliedVolatility'].median() * 100, 1)
        iv_current = round((iv_calls + iv_puts) / 2, 1)
        
        # Strike المقترح (أقرب strike فوق السعر للـ Call)
        otm_calls = calls[calls['strike'] > current_price].head(3)
        suggested_strike = round(otm_calls['strike'].iloc[0], 2) if not otm_calls.empty else round(current_price * 1.05, 2)
        
        # تقييم IV
        if iv_current < 30:
            iv_signal = "منخفض — مناسب لشراء الخيارات"
            iv_rating = "✅ جيد"
        elif iv_current < 50:
            iv_signal = "متوسط — مقبول بحذر"
            iv_rating = "⚠️ متوسط"
        else:
            iv_signal = "مرتفع — الخيارات غالية"
            iv_rating = "❌ مرتفع"
        
        return {
            "expiry": best_date,
            "iv_current": iv_current,
            "iv_signal": iv_signal,
            "iv_rating": iv_rating,
            "suggested_strike": suggested_strike,
            "type": "Call Option"
        }
    except Exception as e:
        print(f"Options error for {symbol}: {e}")
        return None


@app.route("/scanner", methods=["GET"])
def scanner():
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from ces_scanner import run_scanner, WATCHLIST_FLAT
        from ces_v5 import get_profile

        results = run_scanner(WATCHLIST_FLAT)

        clean = []
        for r in results:
            if "error" in r:
                continue
            clean.append({
                "symbol":    r.get("symbol", ""),
                "price":     r.get("price", 0),
                "ces":       r.get("ces", 0),
                "threshold": r.get("threshold", 72),
                "distance":  r.get("distance", 0),
                "iv_rank":   r.get("iv_rank", 0),
                "style":     r.get("style", ""),
                "signal":    r.get("signal", "انتظر"),
            })

        enter = [r for r in clean if r["signal"] == "ادخل"]
        near  = [r for r in clean if r["signal"] == "قريب"]
        wait  = [r for r in clean if r["signal"] == "انتظر"]

        return jsonify({
            "status":   "success",
            "date":     datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total":    len(clean),
            "signals":  len(enter),
            "results":  clean,
            "enter":    enter,
            "near":     near,
            "wait":     wait
        })

    except Exception as e:
        print(f"Scanner error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
