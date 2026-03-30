import os
import json
import re
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import requests
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

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
    except Exception as e:
        print(f"⚠️ JSON parse error: {e}")
        return {}

def call_gpt(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        print(f"GPT error: {data}")
        return "{}"
    except Exception as e:
        print(f"GPT exception: {e}")
        return "{}"

# ============================================================
# QUANT ENGINE — EDGE SCORE
# ============================================================
def calculate_edge(hist):
    close = hist["Close"]
    volume = hist["Volume"]

    # Z-Score السعر
    mean20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    z = round(((close - mean20) / std20).iloc[-1], 2)

    # Z-Score الحجم
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    vol_z = round(((volume - vol_mean) / vol_std).iloc[-1], 2)

    # الاتجاه — 3 طبقات صحيحة
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(min(200, len(close))).mean()
    ema20 = close.ewm(span=20).mean()

    trend_long  = int(close.iloc[-1] > sma200.iloc[-1])   # السعر فوق SMA200
    trend_mid   = int(sma50.iloc[-1] > sma200.iloc[-1])   # SMA50 فوق SMA200
    trend_short = int(ema20.iloc[-1] > sma50.iloc[-1])    # EMA20 فوق SMA50
    trend_score = trend_long + trend_mid + trend_short     # 0-3

    # تسمية الاتجاه
    trend_labels = {3: "strong_uptrend", 2: "weak_uptrend",
                    1: "weak_downtrend", 0: "strong_downtrend"}
    trend_label = trend_labels[trend_score]

    # حساب EdgeScore
    edge = 50

    if z < -2:    edge += 20
    elif z < -1:  edge += 10
    elif z > 2:   edge -= 20
    elif z > 1:   edge -= 10

    if vol_z > 2:  edge += 15
    elif vol_z > 1: edge += 5
    elif vol_z < -1: edge -= 5

    if trend_score == 3:   edge += 15
    elif trend_score == 2: edge += 5
    elif trend_score == 1: edge -= 5
    elif trend_score == 0: edge -= 15

    edge = max(0, min(100, round(edge)))

    return {
        "EdgeScore": edge,
        "z_score": z,
        "volume_z": vol_z,
        "trend_score": trend_score,
        "trend": trend_label
    }

# ============================================================
# FETCH LIVE DATA — كاملة ودقيقة
# ============================================================
def fetch_live_data(symbol):
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

        # Moving Averages
        sma50  = round(close.rolling(50).mean().iloc[-1], 2)
        sma200 = round(close.rolling(min(200, len(close))).mean().iloc[-1], 2)
        ema20  = round(close.ewm(span=20).mean().iloc[-1], 2)

        # Volume
        avg_vol   = round(hist["Volume"].rolling(20).mean().iloc[-1])
        vol_ratio = round(hist["Volume"].iloc[-1] / avg_vol, 2)

        # 52-week
        high_52w     = round(close.max(), 2)
        low_52w      = round(close.min(), 2)
        pct_from_high = round(((close.iloc[-1] - high_52w) / high_52w) * 100, 2)

        # Changes
        change_1d = round(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100, 2)
        change_1w = round(((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) * 100, 2) if len(hist) >= 6 else 0
        change_1m = round(((close.iloc[-1] - close.iloc[-22]) / close.iloc[-22]) * 100, 2) if len(hist) >= 22 else 0

        # Edge Score
        edge = calculate_edge(hist)

        return {
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "close": round(close.iloc[-1], 2),
            "prev_close": round(close.iloc[-2], 2),
            "change_1d": change_1d,
            "change_1w": change_1w,
            "change_1m": change_1m,
            "pre_market": round(info.get("preMarketPrice", 0) or 0, 2),
            "after_hours": round(info.get("postMarketPrice", 0) or 0, 2),
            "rsi": rsi,
            "macd": macd_val,
            "macd_signal": macd_sig,
            "macd_cross": macd_cross,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_position": bb_pos,
            "sma50": sma50,
            "sma200": sma200,
            "ema20": ema20,
            "volume": int(hist["Volume"].iloc[-1]),
            "avg_volume": int(avg_vol),
            "volume_ratio": vol_ratio,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_from_high": pct_from_high,
            "trend": edge["trend"],
            "trend_score": edge["trend_score"],
            "edge_score": edge["EdgeScore"],
            "z_score": edge["z_score"],
            "volume_z": edge["volume_z"],
            "pe_ratio": round(info.get("trailingPE", 0) or 0, 2),
            "market_cap_b": round((info.get("marketCap", 0) or 0) / 1e9, 1),
        }
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

# ============================================================
# FETCH NEWS
# ============================================================
def fetch_stock_news(symbol):
    try:
        news = yf.Ticker(symbol).news
        if not news:
            return "No recent news"
        return "\n".join([f"- {n.get('title','')} ({n.get('publisher','')})" for n in news[:5]])
    except:
        return "Could not fetch news"

def fetch_macro_news():
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWS_API_KEY,
            "q": "Fed OR inflation OR trade war OR recession OR geopolitical",
            "language": "en", "sortBy": "publishedAt", "pageSize": 5
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return "Could not fetch macro news"
        return "\n".join([f"- {a.get('title','')} ({a.get('source',{}).get('name','')})"
                          for a in data.get("articles", [])[:5]])
    except:
        return "Could not fetch macro news"

# ============================================================
# INSTITUTIONAL MEMORY
# ============================================================
def load_memory(symbol, log_file="shadow_log.csv"):
    if not os.path.exists(log_file):
        return None
    try:
        df = pd.read_csv(log_file)
        df_s = df[df["symbol"] == symbol].tail(10)
        if df_s.empty:
            return None

        # Add missing columns for backward compatibility
        for col in ["trend","edge_score","actual_result","score"]:
            if col not in df_s.columns:
                df_s[col] = ""

        last_5 = df_s.tail(5)[["date","close","rsi","trend","edge_score",
                                "decision","confidence","actual_result","score"]].to_dict(orient="records")

        scored = df_s[df_s["score"] != ""].copy()
        if len(scored) > 0:
            scored["score_num"] = pd.to_numeric(scored["score"], errors="coerce")
            win_rate = round(len(scored[scored["score_num"] > 0]) / len(scored) * 100, 1)
            avg_score = round(scored["score_num"].mean(), 2)
        else:
            win_rate = avg_score = 0

        return {
            "symbol": symbol,
            "total_sessions": len(df_s),
            "win_rate": win_rate,
            "avg_score": avg_score,
            "execute_count": len(df_s[df_s["decision"] == "Execute"]),
            "last_5": last_5
        }
    except Exception as e:
        print(f"Memory error: {e}")
        return None

def format_memory(memory):
    if not memory:
        return "No previous sessions. First analysis."
    lines = [
        f"MEMORY — {memory['symbol']}",
        f"Sessions: {memory['total_sessions']} | Win Rate: {memory['win_rate']}% | Avg Score: {memory['avg_score']}",
        f"Execute count: {memory['execute_count']}",
        "Last 5:"
    ]
    for d in memory["last_5"]:
        lines.append(f"  {d['date']} | Close:{d['close']} | Edge:{d.get('edge_score','?')} | Decision:{d['decision']} | Result:{d.get('actual_result','Pending')}")
    return "\n".join(lines)

# ============================================================
# PHASE 1 — ANALYST (Claude)
# ============================================================
def run_analyst(symbol, data, memory_text):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""
You are a professional technical analyst for US stocks (Swing Trading, Daily timeframe).

Stock: {symbol}
Memory: {memory_text}

LIVE DATA:
{json.dumps(data, indent=2)}

Analyze price action, trend, momentum, and key levels.
Return ONLY this JSON:
{{
  "TechnicalScore": 0-100,
  "Trend": "description",
  "KeySupport": 0,
  "KeyResistance": 0,
  "EntryZone": 0,
  "StopLoss": 0,
  "Target1": 0,
  "Target2": 0,
  "Summary": "clear reasoning"
}}
"""
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return safe_json(msg.content[0].text)

# ============================================================
# PHASE 2 — MARKET INTELLIGENCE (GPT)
# ============================================================
def run_market_intelligence(symbol, data, stock_news, macro_news, memory_text):
    prompt = f"""
You are Market Intelligence for US stock trading.

Stock: {symbol}
Memory: {memory_text}

TECHNICAL CONTEXT (use this to interpret news):
Close: {data['close']} | RSI: {data['rsi']} | MACD: {data['macd_cross']}
Trend: {data['trend']} | EdgeScore: {data['edge_score']} | Z-Score: {data['z_score']}
Volume ratio: {data['volume_ratio']}x

STOCK NEWS:
{stock_news}

MACRO NEWS:
{macro_news}

Interpret how news and macro factors interact with current technical condition.
Return ONLY this JSON:
{{
  "MacroScore": 0-100,
  "MarketBias": "Risk On or Risk Off",
  "KeyCatalyst": "most important factor",
  "NewsImpact": "positive/negative/neutral",
  "Summary": "contextual reasoning"
}}
"""
    return safe_json(call_gpt(prompt))

# ============================================================
# PHASE 3 — CRITIC (SCORING ONLY — NO VETO)
# ============================================================
def run_critic(symbol, analyst, market_intel):
    prompt = f"""
You are a risk evaluator. Your job is SCORING ONLY — you have NO veto power.

Evaluate the consistency between Analyst and Market Intelligence for {symbol}.

Analyst: {json.dumps(analyst)}
Market Intel: {json.dumps(market_intel)}

Return ONLY this JSON:
{{
  "ConsistencyScore": 0-100,
  "RiskLevel": "Low/Medium/High",
  "Risks": ["risk1", "risk2"],
  "Alignment": "aligned/divergent"
}}
"""
    return safe_json(call_gpt(prompt))

# ============================================================
# PHASE 4 — DECISION ENGINE (Weighted Probability)
# ============================================================
def run_decision(symbol, data, analyst, market_intel, critic):
    edge = data.get("edge_score", 50)
    tech  = analyst.get("TechnicalScore", 50)
    macro = market_intel.get("MacroScore", 50)
    cons  = critic.get("ConsistencyScore", 50)

    # Weighted score
    final = round(
        0.40 * edge +
        0.25 * tech  +
        0.20 * macro +
        0.15 * cons
    )

    if final >= 72:
        decision = "Execute"
    elif final >= 58:
        decision = "Conditional"
    else:
        decision = "No Trade"

    # Entry/Stop/Target من Analyst أو محسوبة
    close = data["close"]
    entry   = analyst.get("EntryZone") or round(close * 0.997, 2)
    stop    = analyst.get("StopLoss")  or round(close * 0.97, 2)
    target1 = analyst.get("Target1")   or round(close * 1.05, 2)
    target2 = analyst.get("Target2")   or round(close * 1.10, 2)

    rr = round((target1 - entry) / (entry - stop), 2) if (entry - stop) > 0 else 0

    # لا نعرض الأسعار إلا عند Execute أو Conditional
    if decision == "No Trade":
        plan = {
            "Entry": "N/A",
            "Stop": "N/A",
            "Target1": "N/A",
            "Target2": "N/A",
            "RiskReward": "N/A"
        }
    else:
        plan = {
            "Entry": entry,
            "Stop": stop,
            "Target1": target1,
            "Target2": target2,
            "RiskReward": f"1:{rr}"
        }

    return {
        "Decision": decision,
        "Confidence": final,
        "EdgeScore": edge,
        "FinalScore": final,
        "Scores": {"Edge": edge, "Technical": tech, "Macro": macro, "Consistency": cons},
        "Plan": plan
    }

# ============================================================
# SAVE SHADOW LOG
# ============================================================
def save_shadow_log(symbol, data, decision):
    log_file = "shadow_log.csv"
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "close": data.get("close", 0),
        "rsi": data.get("rsi", 0),
        "macd_cross": data.get("macd_cross", ""),
        "trend": data.get("trend", ""),
        "edge_score": data.get("edge_score", 0),
        "volume_ratio": data.get("volume_ratio", 0),
        "decision": decision.get("Decision", ""),
        "confidence": decision.get("Confidence", 0),
        "entry": decision.get("Plan", {}).get("Entry", "N/A"),
        "stop": decision.get("Plan", {}).get("Stop", "N/A"),
        "target": decision.get("Plan", {}).get("Target1", "N/A"),
        "rr": decision.get("Plan", {}).get("RiskReward", "N/A"),
        "actual_result": "",
        "score": ""
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(log_file, index=False)
    print(f"✅ Log saved for {symbol}")
    return row

# ============================================================
# MAIN
# ============================================================
def run_council(symbol):
    print(f"\n{'='*50}")
    print(f"ADVISORY COUNCIL — {symbol}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    print(f"📡 Fetching data for {symbol}...")
    data = fetch_live_data(symbol)
    if not data:
        print(f"❌ Failed to fetch {symbol}")
        return None

    print(f"✅ Close: ${data['close']} | RSI: {data['rsi']} | MACD: {data['macd_cross']}")
    print(f"   Trend: {data['trend']} | Edge: {data['edge_score']} | Z: {data['z_score']}")

    memory = load_memory(symbol)
    memory_text = format_memory(memory)
    if memory:
        print(f"📚 Memory: {memory['total_sessions']} sessions | Win rate: {memory['win_rate']}%")
    else:
        print(f"📚 First session for {symbol}")

    stock_news = fetch_stock_news(symbol)
    macro_news = fetch_macro_news()

    print("[Phase 1] Analyst (Claude)...")
    analyst = run_analyst(symbol, data, memory_text)
    print(f"✅ TechnicalScore: {analyst.get('TechnicalScore', '?')}")

    print("[Phase 2] Market Intelligence (GPT)...")
    market_intel = run_market_intelligence(symbol, data, stock_news, macro_news, memory_text)
    print(f"✅ MacroScore: {market_intel.get('MacroScore', '?')}")

    print("[Phase 3] Critic (GPT — Scoring Only)...")
    critic = run_critic(symbol, analyst, market_intel)
    print(f"✅ ConsistencyScore: {critic.get('ConsistencyScore', '?')}")

    print("[Phase 4] Decision Engine...")
    decision = run_decision(symbol, data, analyst, market_intel, critic)

    log = save_shadow_log(symbol, data, decision)

    print(f"\n{'='*50}")
    print(f"COUNCIL DECISION — {symbol}")
    print(f"{'='*50}")
    print(f"Edge Score:   {data['edge_score']}/100")
    print(f"Tech Score:   {analyst.get('TechnicalScore','?')}/100")
    print(f"Macro Score:  {market_intel.get('MacroScore','?')}/100")
    print(f"Consistency:  {critic.get('ConsistencyScore','?')}/100")
    print(f"Final Score:  {decision['FinalScore']}/100")
    print(f"─────────────────────────────────")
    print(f"Decision:     {decision['Decision']}")
    print(f"Confidence:   {decision['Confidence']}%")
    print(f"Entry:        ${decision['Plan']['Entry']}")
    print(f"Stop:         ${decision['Plan']['Stop']}")
    print(f"Target 1:     ${decision['Plan']['Target1']}")
    print(f"Target 2:     ${decision['Plan']['Target2']}")
    print(f"R/R:          {decision['Plan']['RiskReward']}")
    print(f"Trend:        {data['trend']}")
    print(f"{'='*50}\n")

    return {
        "symbol": symbol,
        "data": data,
        "analyst": analyst,
        "market_intel": market_intel,
        "critic": critic,
        "decision": decision,
        "log": log,
        "memory": memory
    }

if __name__ == "__main__":
    stocks = ["NVDA", "MSFT", "AMZN", "ARM", "PLUG", "QCOM"]
    for stock in stocks:
        run_council(stock)
