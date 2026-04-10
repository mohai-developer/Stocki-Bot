"""
dual_analysis.py
══════════════════════════════════════════════════════
تحليل مزدوج — Claude + GPT
كل نموذج يحلل بحرية كاملة ويعطي قراره ونسبة ثقته
══════════════════════════════════════════════════════
"""

import os
import json
import re
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# HELPERS
# ============================================================
def clean_json(text):
    if isinstance(text, dict):
        return text
    text = str(text).strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return "{}"

def safe_json(text):
    try:
        if isinstance(text, dict):
            return text
        return json.loads(clean_json(text))
    except:
        return {}

def call_gpt(prompt, max_tokens=2000):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        print(f"GPT error: {data}")
        return ""
    except Exception as e:
        print(f"GPT exception: {e}")
        return ""

# ============================================================
# FETCH DATA
# ============================================================
def fetch_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="6mo", interval="1d")
        hist_weekly = ticker.history(period="1y", interval="1wk")

        if len(hist) < 20:
            return None

        close = hist["Close"]

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rsi = round(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])), 2)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_val = round((ema12 - ema26).iloc[-1], 4)
        macd_sig = round((ema12 - ema26).ewm(span=9).mean().iloc[-1], 4)
        macd_cross = "bullish" if macd_val > macd_sig else "bearish"

        # Bollinger
        sma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        bb_upper = round(sma20 + 2 * std20, 2)
        bb_lower = round(sma20 - 2 * std20, 2)
        bb_pos = round(((close.iloc[-1] - bb_lower) / (bb_upper - bb_lower)) * 100, 1) if bb_upper != bb_lower else 50

        # MAs
        sma50  = round(close.rolling(50).mean().iloc[-1], 2)
        sma200 = round(close.rolling(min(200, len(close))).mean().iloc[-1], 2)
        ema20  = round(close.ewm(span=20).mean().iloc[-1], 2)

        # Volume
        avg_vol   = round(hist["Volume"].rolling(20).mean().iloc[-1])
        vol_ratio = round(hist["Volume"].iloc[-1] / avg_vol, 2)

        # Trend — 3 layers
        trend_long  = int(close.iloc[-1] > sma200)
        trend_mid   = int(sma50 > sma200)
        trend_short = int(ema20 > sma50)
        trend_score = trend_long + trend_mid + trend_short
        trend_map   = {3:"strong_uptrend", 2:"weak_uptrend", 1:"weak_downtrend", 0:"strong_downtrend"}
        trend       = trend_map[trend_score]

        # Edge Score
        mean20 = close.rolling(20).mean()
        std20s = close.rolling(20).std()
        z = round(((close - mean20) / std20s).iloc[-1], 2)

        vol_mean = hist["Volume"].rolling(20).mean()
        vol_std  = hist["Volume"].rolling(20).std()
        vol_z    = round(((hist["Volume"] - vol_mean) / vol_std).iloc[-1], 2)

        edge = 50
        if z < -2:    edge += 20
        elif z < -1:  edge += 10
        elif z > 2:   edge -= 20
        if vol_z > 2:  edge += 15
        elif vol_z > 1: edge += 5
        if trend_score == 3:   edge += 15
        elif trend_score == 2: edge += 5
        elif trend_score == 0: edge -= 15
        edge = max(0, min(100, round(edge)))

        # Changes
        change_1d = round(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100, 2)
        change_1w = round(((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) * 100, 2) if len(hist) >= 6 else 0
        change_1m = round(((close.iloc[-1] - close.iloc[-22]) / close.iloc[-22]) * 100, 2) if len(hist) >= 22 else 0

        high_52w = round(close.max(), 2)
        low_52w  = round(close.min(), 2)

        return {
            "symbol":      symbol,
            "date":        datetime.now().strftime("%Y-%m-%d"),
            "close":       round(close.iloc[-1], 2),
            "change_1d":   change_1d,
            "change_1w":   change_1w,
            "change_1m":   change_1m,
            "pre_market":  round(info.get("preMarketPrice", 0) or 0, 2),
            "after_hours": round(info.get("postMarketPrice", 0) or 0, 2),
            "rsi":         rsi,
            "macd":        macd_val,
            "macd_signal": macd_sig,
            "macd_cross":  macd_cross,
            "bb_upper":    bb_upper,
            "bb_lower":    bb_lower,
            "bb_position": bb_pos,
            "sma50":       sma50,
            "sma200":      sma200,
            "ema20":       ema20,
            "volume":      int(hist["Volume"].iloc[-1]),
            "avg_volume":  int(avg_vol),
            "volume_ratio": vol_ratio,
            "high_52w":    high_52w,
            "low_52w":     low_52w,
            "trend":       trend,
            "trend_score": trend_score,
            "edge_score":  edge,
            "z_score":     z,
            "volume_z":    vol_z,
            "pe_ratio":    round(info.get("trailingPE", 0) or 0, 2),
            "market_cap_b": round((info.get("marketCap", 0) or 0) / 1e9, 1),
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ============================================================
# FETCH NEWS
# ============================================================
def fetch_news(symbol, current_price=None):
    try:
        from news_bot import get_news, format_news_for_prompt
        result = get_news(symbol, current_price)
        return format_news_for_prompt(result)
    except Exception as e:
        print(f"news_bot error: {e}")
        return "Could not fetch news"

# ============================================================
# CLAUDE ANALYSIS
# ============================================================
def run_claude(symbol, data, news, report_type="full"):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if report_type == "summary":
        instruction = "قدم تحليلاً مختصراً وقراراً واضحاً مع نسبة ثقة."
    else:
        instruction = "قدم تحليلاً مفصلاً شاملاً بكل الأدوات التي تراها مناسبة."

    prompt = f"""أنت محلل مالي متخصص في الأسهم الأمريكية وSwing Trading.
حلل السهم {symbol} بحرية كاملة — استخدم الأدوات والمنهجية التي تراها مناسبة.

{instruction}

البيانات المتاحة:
{json.dumps(data, indent=2)}

الأخبار والمعلومات الإخبارية:
{news}

أجب بهذا الـ JSON فقط:
{{
  "analysis": "تحليلك الكامل هنا",
  "decision": "ادخل أو انتظر أو لا تداول",
  "confidence": 0-100,
  "entry": 0,
  "stop": 0,
  "target1": 0,
  "target2": 0,
  "rr": "1:X",
  "key_reasons": ["سبب 1", "سبب 2", "سبب 3"],
  "risks": ["خطر 1", "خطر 2"],
  "tools_used": ["الأدوات التي استخدمتها"]
}}

إذا القرار انتظر أو لا تداول، اجعل entry وstop وtarget1 وtarget2 = 0
"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return safe_json(msg.content[0].text)

# ============================================================
# GPT ANALYSIS
# ============================================================
def run_gpt_analysis(symbol, data, news, report_type="full"):
    if report_type == "summary":
        instruction = "قدم تحليلاً مختصراً وقراراً واضحاً مع نسبة ثقة."
    else:
        instruction = "قدم تحليلاً مفصلاً شاملاً بكل الأدوات التي تراها مناسبة."

    prompt = f"""أنت محلل مالي متخصص في الأسهم الأمريكية وSwing Trading.
حلل السهم {symbol} بحرية كاملة — استخدم الأدوات والمنهجية التي تراها مناسبة.

{instruction}

البيانات المتاحة:
{json.dumps(data, indent=2)}

الأخبار والمعلومات الإخبارية:
{news}

أجب بهذا الـ JSON فقط:
{{
  "analysis": "تحليلك الكامل هنا",
  "decision": "ادخل أو انتظر أو لا تداول",
  "confidence": 0-100,
  "entry": 0,
  "stop": 0,
  "target1": 0,
  "target2": 0,
  "rr": "1:X",
  "key_reasons": ["سبب 1", "سبب 2", "سبب 3"],
  "risks": ["خطر 1", "خطر 2"],
  "tools_used": ["الأدوات التي استخدمتها"]
}}

إذا القرار انتظر أو لا تداول، اجعل entry وstop وtarget1 وtarget2 = 0
"""

    result = call_gpt(prompt)
    return safe_json(result)

# ============================================================
# COMBINE RESULTS
# ============================================================
def combine_results(claude_result, gpt_result):
    claude_conf = claude_result.get("confidence", 50)
    gpt_conf    = gpt_result.get("confidence", 50)
    avg_conf    = round((claude_conf + gpt_conf) / 2)

    claude_dec = claude_result.get("decision", "انتظر")
    gpt_dec    = gpt_result.get("decision", "انتظر")

    agreement = claude_dec == gpt_dec

    if agreement:
        final_decision = claude_dec
        status = "اتفاق"
    else:
        # الأعلى ثقة يرجح
        if claude_conf > gpt_conf:
            final_decision = claude_dec
            status = f"خلاف — Claude أرجح ({claude_dec})"
        elif gpt_conf > claude_conf:
            final_decision = gpt_dec
            status = f"خلاف — GPT أرجح ({gpt_dec})"
        else:
            final_decision = "انتظر"
            status = "خلاف متساوٍ — انتظر"

    # Entry/Stop/Target من الأعلى ثقة
    if final_decision == "ادخل":
        if claude_conf >= gpt_conf:
            entry   = claude_result.get("entry", 0)
            stop    = claude_result.get("stop", 0)
            target1 = claude_result.get("target1", 0)
            target2 = claude_result.get("target2", 0)
            rr      = claude_result.get("rr", "N/A")
        else:
            entry   = gpt_result.get("entry", 0)
            stop    = gpt_result.get("stop", 0)
            target1 = gpt_result.get("target1", 0)
            target2 = gpt_result.get("target2", 0)
            rr      = gpt_result.get("rr", "N/A")
    else:
        entry = stop = target1 = target2 = 0
        rr = "N/A"

    return {
        "final_decision": final_decision,
        "avg_confidence": avg_conf,
        "agreement": agreement,
        "status": status,
        "entry":   entry,
        "stop":    stop,
        "target1": target1,
        "target2": target2,
        "rr":      rr,
        "claude_confidence": claude_conf,
        "gpt_confidence":    gpt_conf,
    }

# ============================================================
# FORMAT OUTPUT
# ============================================================
def format_dual_report(symbol, data, claude_result, gpt_result, combined):
    lines = []
    lines.append(f"{symbol} — تحليل مزدوج")
    lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("━" * 40)

    # البيانات الأساسية
    lines.append(f"الإغلاق: ${data['close']} | RSI: {data['rsi']} | MACD: {data['macd_cross']}")
    lines.append(f"الاتجاه: {data['trend']} | Edge: {data['edge_score']}/100")
    lines.append("")

    # تحليل Claude
    lines.append("━" * 40)
    lines.append("📊 تحليل Claude")
    lines.append("━" * 40)
    lines.append(claude_result.get("analysis", ""))
    lines.append("")
    lines.append(f"الأدوات: {', '.join(claude_result.get('tools_used', []))}")
    lines.append("")

    reasons = claude_result.get("key_reasons", [])
    if reasons:
        lines.append("الأسباب:")
        for r in reasons:
            lines.append(f"  • {r}")

    risks = claude_result.get("risks", [])
    if risks:
        lines.append("المخاطر:")
        for r in risks:
            lines.append(f"  ⚠ {r}")

    lines.append("")
    lines.append(f"القرار: {claude_result.get('decision', 'انتظر')}")
    lines.append(f"الثقة: {claude_result.get('confidence', 0)}%")

    if claude_result.get("decision") == "ادخل" and claude_result.get("entry", 0) > 0:
        lines.append(f"الدخول: ${claude_result.get('entry')} | الوقف: ${claude_result.get('stop')}")
        lines.append(f"الهدف 1: ${claude_result.get('target1')} | الهدف 2: ${claude_result.get('target2')}")
        lines.append(f"R/R: {claude_result.get('rr')}")

    lines.append("")

    # تحليل GPT
    lines.append("━" * 40)
    lines.append("🤖 تحليل GPT")
    lines.append("━" * 40)
    lines.append(gpt_result.get("analysis", ""))
    lines.append("")
    lines.append(f"الأدوات: {', '.join(gpt_result.get('tools_used', []))}")
    lines.append("")

    reasons = gpt_result.get("key_reasons", [])
    if reasons:
        lines.append("الأسباب:")
        for r in reasons:
            lines.append(f"  • {r}")

    risks = gpt_result.get("risks", [])
    if risks:
        lines.append("المخاطر:")
        for r in risks:
            lines.append(f"  ⚠ {r}")

    lines.append("")
    lines.append(f"القرار: {gpt_result.get('decision', 'انتظر')}")
    lines.append(f"الثقة: {gpt_result.get('confidence', 0)}%")

    if gpt_result.get("decision") == "ادخل" and gpt_result.get("entry", 0) > 0:
        lines.append(f"الدخول: ${gpt_result.get('entry')} | الوقف: ${gpt_result.get('stop')}")
        lines.append(f"الهدف 1: ${gpt_result.get('target1')} | الهدف 2: ${gpt_result.get('target2')}")
        lines.append(f"R/R: {gpt_result.get('rr')}")

    lines.append("")

    # الخلاصة
    lines.append("━" * 40)
    lines.append("⚖️ الخلاصة")
    lines.append("━" * 40)
    lines.append(f"Claude: {combined['claude_confidence']}% | GPT: {combined['gpt_confidence']}%")
    lines.append(f"متوسط الثقة: {combined['avg_confidence']}%")
    lines.append(f"الحالة: {combined['status']}")
    lines.append(f"القرار الموحد: {combined['final_decision']}")

    if combined["final_decision"] == "ادخل" and combined["entry"] > 0:
        lines.append(f"الدخول: ${combined['entry']}")
        lines.append(f"الوقف: ${combined['stop']}")
        lines.append(f"الهدف 1: ${combined['target1']}")
        lines.append(f"الهدف 2: ${combined['target2']}")
        lines.append(f"R/R: {combined['rr']}")

    return "\n".join(lines)

# ============================================================
# MAIN FUNCTION
# ============================================================
def run_dual_analysis(symbol, report_type="full"):
    print(f"\n{'='*50}")
    print(f"Dual Analysis — {symbol}")
    print(f"{'='*50}")

    print(f"جلب البيانات...")
    data = fetch_data(symbol)
    if not data:
        return None

    print(f"✅ Close: ${data['close']} | RSI: {data['rsi']} | Edge: {data['edge_score']}")

    print(f"جلب الأخبار...")
    news = fetch_news(symbol, data["close"])

    print(f"Claude و GPT يحللان بالتوازي...")
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_claude = executor.submit(run_claude, symbol, data, news, report_type)
        future_gpt    = executor.submit(run_gpt_analysis, symbol, data, news, report_type)
        claude_result = future_claude.result(timeout=120)
        gpt_result    = future_gpt.result(timeout=120)

    print(f"✅ Claude: {claude_result.get('decision')} ({claude_result.get('confidence')}%)")
    print(f"✅ GPT: {gpt_result.get('decision')} ({gpt_result.get('confidence')}%)")

    combined = combine_results(claude_result, gpt_result)

    print(f"\n{'='*50}")
    print(f"الخلاصة: {combined['status']}")
    print(f"القرار: {combined['final_decision']} ({combined['avg_confidence']}%)")
    print(f"{'='*50}\n")

    report = format_dual_report(symbol, data, claude_result, gpt_result, combined)

    return {
        "symbol":       symbol,
        "data":         data,
        "claude":       claude_result,
        "gpt":          gpt_result,
        "combined":     combined,
        "report":       report,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    result = run_dual_analysis(symbol)
    if result:
        print(result["report"])
