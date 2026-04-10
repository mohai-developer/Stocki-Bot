"""
news_bot.py
══════════════════════════════════════════════════════
بوت الأخبار المتخصص — Stocki Bot
يجمع الأخبار من مصادر متعددة ويحللها بـ Claude
══════════════════════════════════════════════════════
"""

import os
import re
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ============================================================
# المصدر 1 — Google News RSS (يغطي كل الأسهم)
# ============================================================
def fetch_google_news(symbol, company_name=""):
    try:
        query = f"{symbol} stock {company_name}".strip()
        query_encoded = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockiBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            return []
        
        root = ET.fromstring(resp.content)
        items = []
        cutoff = datetime.now() - timedelta(days=7)
        
        for item in root.findall(".//item")[:10]:
            title = item.findtext("title", "")
            source = item.findtext("source", "Google News")
            pub_date_str = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            
            # تنظيف العنوان
            title = re.sub(r'\s*-\s*[^-]+$', '', title).strip()
            
            if title:
                items.append({
                    "title": title,
                    "source": source,
                    "date": pub_date_str[:16] if pub_date_str else "",
                    "link": link,
                    "origin": "Google News"
                })
        
        return items
    except Exception as e:
        print(f"Google News error: {e}")
        return []

# ============================================================
# المصدر 2 — NewsAPI
# ============================================================
def fetch_newsapi(symbol, company_name=""):
    try:
        if not NEWS_API_KEY:
            return []
        
        query = f"{symbol}"
        if company_name:
            query = f"{symbol} OR \"{company_name}\""
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWS_API_KEY,
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "from": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        }
        
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get("status") != "ok":
            return []
        
        items = []
        for a in data.get("articles", []):
            title = a.get("title", "")
            source = a.get("source", {}).get("name", "")
            published = a.get("publishedAt", "")[:10]
            
            if title and "[Removed]" not in title:
                items.append({
                    "title": title,
                    "source": source,
                    "date": published,
                    "link": a.get("url", ""),
                    "origin": "NewsAPI"
                })
        
        return items
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []

# ============================================================
# المصدر 3 — SEC EDGAR (إعلانات رسمية)
# ============================================================
def fetch_sec_filings(symbol):
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt={(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}&enddt={datetime.now().strftime('%Y-%m-%d')}&forms=8-K,10-Q,10-K"
        headers = {"User-Agent": "StockiBot research@stockibot.com"}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        items = []
        
        for hit in data.get("hits", {}).get("hits", [])[:5]:
            source = hit.get("_source", {})
            form = source.get("form_type", "")
            filed = source.get("file_date", "")
            entity = source.get("entity_name", symbol)
            description = source.get("period_of_report", "")
            
            if form:
                items.append({
                    "title": f"SEC Filing: {form} — {entity} ({description})",
                    "source": "SEC EDGAR",
                    "date": filed,
                    "link": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={symbol}&type={form}",
                    "origin": "SEC EDGAR"
                })
        
        return items
    except Exception as e:
        print(f"SEC error: {e}")
        return []

# ============================================================
# دمج وتصفية الأخبار
# ============================================================
def get_company_name(symbol):
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        return info.get("shortName", "") or info.get("longName", "")
    except:
        return ""

def deduplicate(items):
    seen = set()
    unique = []
    for item in items:
        title_clean = re.sub(r'[^\w\s]', '', item["title"].lower())[:50]
        if title_clean not in seen:
            seen.add(title_clean)
            unique.append(item)
    return unique

def collect_news(symbol):
    print(f"  جلب أخبار {symbol}...")
    
    company_name = get_company_name(symbol)
    
    # جلب من كل المصادر
    google = fetch_google_news(symbol, company_name)
    newsapi = fetch_newsapi(symbol, company_name)
    sec = fetch_sec_filings(symbol)
    
    # دمج وإزالة التكرار
    all_news = google + newsapi + sec
    unique = deduplicate(all_news)
    
    print(f"  ✅ وجدت {len(unique)} خبر ({len(google)} Google, {len(newsapi)} NewsAPI, {len(sec)} SEC)")
    
    return unique[:15]  # أقصى 15 خبر

# ============================================================
# تحليل الأخبار بـ Claude
# ============================================================
def analyze_news(symbol, news_items, current_price=None):
    if not news_items:
        return {
            "summary": "لا توجد أخبار حديثة متاحة لهذا السهم.",
            "impact": "محايد",
            "impact_score": 5,
            "key_events": [],
            "price_drivers": "لا توجد محفزات واضحة",
            "risk_factors": "لا توجد مخاطر إخبارية محددة"
        }
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    news_text = "\n".join([
        f"- [{n['date']}] {n['title']} ({n['source']})"
        for n in news_items
    ])
    
    price_info = f"السعر الحالي: ${current_price}" if current_price else ""
    
    prompt = f"""أنت محلل أخبار مالي متخصص في الأسهم الأمريكية.

السهم: {symbol}
{price_info}

الأخبار الحديثة:
{news_text}

حلل هذه الأخبار وأجب بالعربية بهذا الـ JSON فقط:
{{
  "summary": "ملخص موجز للوضع الإخباري العام",
  "impact": "إيجابي أو سلبي أو محايد",
  "impact_score": 0-10,
  "key_events": ["أهم حدث 1", "أهم حدث 2", "أهم حدث 3"],
  "price_drivers": "ما الذي يحرك السعر إخبارياً الآن",
  "risk_factors": "المخاطر الإخبارية القادمة",
  "catalyst": "هل يوجد محفز قريب (أرباح/منتج/قرار)؟",
  "recommendation": "هل الأخبار تدعم الشراء أم تحذر منه"
}}

قواعد:
- impact_score: 0=كارثي, 5=محايد, 10=إيجابي جداً
- إذا لم تجد أخباراً مؤثرة قل ذلك بوضوح
- ركز على ما يؤثر على السعر فعلاً
"""
    
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = msg.content[0].text
    
    # استخراج JSON
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    
    return {"summary": text, "impact": "محايد", "impact_score": 5,
            "key_events": [], "price_drivers": "", "risk_factors": "", 
            "catalyst": "", "recommendation": ""}

# ============================================================
# الدالة الرئيسية
# ============================================================
def get_news(symbol, current_price=None):
    """
    الدالة الرئيسية — تُستدعى من server.py و council.py
    تعيد تحليلاً إخبارياً كاملاً
    """
    news_items = collect_news(symbol)
    analysis = analyze_news(symbol, news_items, current_price)
    
    return {
        "symbol": symbol,
        "news_count": len(news_items),
        "raw_news": news_items,
        "analysis": analysis,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

def format_news_for_prompt(news_result):
    """تنسيق الأخبار لإدراجها في prompt التحليل"""
    if not news_result:
        return "لا توجد أخبار متاحة"
    
    analysis = news_result.get("analysis", {})
    raw = news_result.get("raw_news", [])
    
    lines = []
    lines.append(f"عدد الأخبار: {news_result.get('news_count', 0)}")
    lines.append(f"التأثير العام: {analysis.get('impact', 'محايد')} ({analysis.get('impact_score', 5)}/10)")
    lines.append(f"الملخص: {analysis.get('summary', '')}")
    
    if analysis.get('key_events'):
        lines.append("أهم الأحداث:")
        for event in analysis.get('key_events', []):
            lines.append(f"  • {event}")
    
    if analysis.get('catalyst'):
        lines.append(f"المحفز القادم: {analysis.get('catalyst')}")
    
    if analysis.get('price_drivers'):
        lines.append(f"محركات السعر: {analysis.get('price_drivers')}")
    
    if analysis.get('risk_factors'):
        lines.append(f"المخاطر الإخبارية: {analysis.get('risk_factors')}")
    
    return "\n".join(lines)

# ============================================================
# تشغيل مباشر للاختبار
# ============================================================
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    
    print(f"\n{'='*50}")
    print(f"News Bot — {symbol}")
    print(f"{'='*50}")
    
    result = get_news(symbol)
    
    print(f"\nعدد الأخبار: {result['news_count']}")
    print(f"\nالأخبار الخام:")
    for n in result['raw_news'][:5]:
        print(f"  [{n['date']}] {n['title'][:80]} ({n['source']})")
    
    print(f"\nالتحليل:")
    analysis = result['analysis']
    print(f"  التأثير: {analysis.get('impact')} ({analysis.get('impact_score')}/10)")
    print(f"  الملخص: {analysis.get('summary')}")
    print(f"  الأحداث: {analysis.get('key_events')}")
    print(f"  المحفز: {analysis.get('catalyst')}")
    print(f"  التوصية: {analysis.get('recommendation')}")
    
    print(f"\n{'='*50}")
    print("للإدراج في prompt:")
    print(format_news_for_prompt(result))