import os
import json
import re
import yfinance as yf
import pandas as pd
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
    text = text.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return text

def safe_json_load(text, fallback="{}"):
    try:
        if isinstance(text, dict):
            return text
        if isinstance(text, str):
            return json.loads(clean_json(text))
        return json.loads(fallback)
    except Exception as e:
        print(f"⚠️ JSON parse fallback: {e}")
        return json.loads(fallback)

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
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        data = resp.json()
        if "choices" in data:
            result = data["choices"][0]["message"]["content"]
            return result
        else:
            print(f"GPT error: {data}")
            return None
    except Exception as e:
        print(f"GPT call exception: {e}")
        return None

# ============================================================
# FETCH LIVE DATA
# ============================================================
def fetch_live_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="6mo", interval="1d")
        hist_weekly = ticker.history(period="1y", interval="1wk")

        if len(hist) < 20:
            return None

        delta = hist["Close"].diff()
        gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rsi = round(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])), 2)

        ema12 = hist["Close"].ewm(span=12).mean()
        ema26 = hist["Close"].ewm(span=26).mean()
        macd = round((ema12 - ema26).iloc[-1], 4)
        macd_signal_line = round((ema12 - ema26).ewm(span=9).mean().iloc[-1], 4)
        macd_cross = "bullish" if macd > macd_signal_line else "bearish"

        sma20 = hist["Close"].rolling(20).mean().iloc[-1]
        std20 = hist["Close"].rolling(20).std().iloc[-1]
        bb_upper = round(sma20 + 2 * std20, 2)
        bb_lower = round(sma20 - 2 * std20, 2)
        bb_position = round(((hist["Close"].iloc[-1] - bb_lower) / (bb_upper - bb_lower)) * 100, 1)

        sma50 = round(hist["Close"].rolling(50).mean().iloc[-1], 2)
        sma200 = round(hist["Close"].rolling(min(200, len(hist))).mean().iloc[-1], 2)
        ema20 = round(hist["Close"].ewm(span=20).mean().iloc[-1], 2)

        avg_vol = round(hist["Volume"].rolling(20).mean().iloc[-1])
        vol_ratio = round(hist["Volume"].iloc[-1] / avg_vol, 2)

        # Trend — 3 layers (short/medium/long)
        short_trend = "uptrend" if hist["Close"].iloc[-1] > ema20 else "downtrend"
        medium_trend = "uptrend" if hist["Close"].iloc[-1] > sma50 else "downtrend"
        weekly_sma20 = hist_weekly["Close"].rolling(20).mean().iloc[-1]
        long_trend = "uptrend" if hist_weekly["Close"].iloc[-1] > weekly_sma20 else "downtrend"

        up_count = [short_trend, medium_trend, long_trend].count("uptrend")
        if up_count == 3:
            weekly_trend = "strong_uptrend"
        elif up_count == 2:
            weekly_trend = "weak_uptrend"
        elif up_count == 1:
            weekly_trend = "weak_downtrend"
        else:
            weekly_trend = "strong_downtrend"

        high_52w = round(hist["Close"].max(), 2)
        low_52w = round(hist["Close"].min(), 2)
        pct_from_high = round(((hist["Close"].iloc[-1] - high_52w) / high_52w) * 100, 2)

        change_1d = round(((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100, 2)
        change_1w = round(((hist["Close"].iloc[-1] - hist["Close"].iloc[-6]) / hist["Close"].iloc[-6]) * 100, 2) if len(hist) >= 6 else 0
        change_1m = round(((hist["Close"].iloc[-1] - hist["Close"].iloc[-22]) / hist["Close"].iloc[-22]) * 100, 2) if len(hist) >= 22 else 0

        return {
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "close": round(hist["Close"].iloc[-1], 2),
            "prev_close": round(hist["Close"].iloc[-2], 2),
            "change_1d": change_1d,
            "change_1w": change_1w,
            "change_1m": change_1m,
            "pre_market": round(info.get("preMarketPrice", 0) or 0, 2),
            "after_hours": round(info.get("postMarketPrice", 0) or 0, 2),
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal_line,
            "macd_cross": macd_cross,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_position": bb_position,
            "sma50": sma50,
            "sma200": sma200,
            "ema20": ema20,
            "volume": int(hist["Volume"].iloc[-1]),
            "avg_volume": int(avg_vol),
            "volume_ratio": vol_ratio,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_from_high": pct_from_high,
            "weekly_trend": weekly_trend,
            "pe_ratio": round(info.get("trailingPE", 0) or 0, 2),
            "market_cap_b": round((info.get("marketCap", 0) or 0) / 1e9, 1),
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# ============================================================
# FETCH NEWS
# ============================================================
def fetch_stock_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return "No recent news"
        items = []
        for n in news[:5]:
            title = n.get("title", "")
            publisher = n.get("publisher", "")
            items.append(f"- {title} ({publisher})")
        return "\n".join(items)
    except:
        return "Could not fetch news"

def fetch_macro_news():
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWS_API_KEY,
            "q": "Fed interest rates OR inflation OR geopolitical OR trade war OR recession",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return "Could not fetch macro news"
        items = []
        for a in data.get("articles", [])[:5]:
            items.append(f"- {a.get('title','')} ({a.get('source',{}).get('name','')})")
        return "\n".join(items)
    except:
        return "Could not fetch macro news"

# ============================================================
# PHASE 1 — ANALYST (Claude)
# ============================================================
def run_analyst(symbol, data, protocol, memory_text=""):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""
{protocol}

You are now acting as the ANALYST role only.
Analyze {symbol} using the provided daily data.
Output ONLY the [Analyst Output] JSON section.
Be precise with numbers. No vague statements.

INSTITUTIONAL MEMORY (use this to improve your analysis):
{memory_text}

LIVE DATA:
{json.dumps(data, indent=2)}

Respond ONLY with valid JSON in the [Analyst Output] format defined in the protocol.
"""
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

# ============================================================
# PHASE 2 — MARKET INTELLIGENCE (GPT)
# ============================================================
def run_market_intelligence(symbol, stock_news, macro_news, protocol, memory_text=""):
    prompt = f"""
{protocol}

You are now acting as the MARKET INTELLIGENCE role only.
Analyze macro environment and news impact for {symbol}.
Output ONLY the [Market Intelligence Output] JSON section.

INSTITUTIONAL MEMORY:
{memory_text}

STOCK NEWS:
{stock_news}

MACRO NEWS:
{macro_news}

Respond ONLY with valid JSON in the [Market Intelligence Output] format defined in the protocol.
"""
    result = call_gpt(prompt)
    if result:
        return result
    return f'{{"Role":"Market Intelligence","Ticker":"{symbol}","MarketBias":"Unknown","ImpactScore":0,"KeyCatalysts":["API error"],"MacroFactors":[],"Notes":"API error"}}'

# ============================================================
# PHASE 3 — CRITIC (GPT)
# ============================================================
def run_critic(symbol, analyst_output, market_output, protocol):
    prompt = f"""
{protocol}

You are now acting as the CRITIC role only.
Evaluate the Analyst and Market Intelligence outputs for {symbol}.

You MUST respond with ONLY this exact JSON structure, no extra nesting:
{{
  "Role": "Critic",
  "Ticker": "{symbol}",
  "EarlyEntryJustified": true or false,
  "ExitTimingValid": true or false,
  "IssuesFound": ["issue1", "issue2"],
  "Contradictions": ["contradiction1"],
  "CriticVerdict": "Pass" or "Conditional" or "Reject",
  "VerdictReason": "reason here"
}}

ANALYST OUTPUT:
{analyst_output}

MARKET INTELLIGENCE OUTPUT:
{market_output}

Respond ONLY with the JSON above. No markdown, no extra text, no nesting.
"""
    result = call_gpt(prompt)
    if result:
        return result
    return '{"Role":"Critic","EarlyEntryJustified":false,"ExitTimingValid":false,"IssuesFound":["API error"],"Contradictions":[],"CriticVerdict":"Reject","VerdictReason":"API error"}'

# ============================================================
# PHASE 4 — DECISION ENGINE (GPT)
# ============================================================
def run_decision_engine(symbol, analyst_output, market_output, critic_output, protocol):
    prompt = f"""
{protocol}

You are now acting as the DECISION ENGINE role only.
Make the final trading decision for {symbol}.

STRICT RULES:
- This system is for LONG positions ONLY (buying stocks or call options)
- SHORT selling is strictly FORBIDDEN
- If the outlook is bearish, the decision must be "No Trade" not a short position
- Entry price must always be BELOW or EQUAL to current price for long entries
- Target price must always be ABOVE entry price

You MUST respond with ONLY this exact JSON structure, no extra nesting:
{{
  "Role": "Decision Engine",
  "Ticker": "{symbol}",
  "InstrumentType": "Stock" or "Call Option" or "Put Option" or "No Trade",
  "InstrumentReason": "reason here",
  "TradePlan": {{
    "Entry": 0,
    "Stop": 0,
    "TargetPartial": 0,
    "TargetFull": 0,
    "TrailingStop": "description",
    "RiskReward": "1:X"
  }},
  "ConfidenceScore": 0,
  "Decision": "Execute" or "Abstain" or "No Trade",
  "AbstentionTriggered": false,
  "MemoryLog": "summary here"
}}

ANALYST OUTPUT:
{analyst_output}

MARKET INTELLIGENCE OUTPUT:
{market_output}

CRITIC OUTPUT:
{critic_output}

Respond ONLY with the JSON above. No markdown, no extra text, no nesting.
"""
    result = call_gpt(prompt)
    if result:
        return result
    return '{"Role":"Decision Engine","Ticker":"","InstrumentType":"No Trade","InstrumentReason":"API error","TradePlan":{"Entry":"N/A","Stop":"N/A","TargetPartial":"N/A","TargetFull":"N/A","TrailingStop":"N/A","RiskReward":"N/A"},"ConfidenceScore":0,"Decision":"No Trade","AbstentionTriggered":false}'

# ============================================================
# SAVE SHADOW LOG
# ============================================================
def save_shadow_log(symbol, analyst, market, critic, decision, data):
    log_file = "shadow_log.csv"

    critic_raw = safe_json_load(critic)
    decision_raw = safe_json_load(decision)

    # Handle nested: {"Critic Output": {...}} or flat {"CriticVerdict": ...}
    critic_json = critic_raw.get("Critic Output", critic_raw)
    decision_json = decision_raw.get("Decision Output", decision_raw)

    verdict = (
        critic_json.get("CriticVerdict") or
        critic_json.get("verdict") or
        critic_json.get("Verdict") or
        "Unknown"
    )

    final_decision = (
        decision_json.get("Decision") or
        decision_json.get("trading_decision") or
        decision_json.get("decision") or
        "Unknown"
    )

    confidence = (
        decision_json.get("ConfidenceScore") or
        decision_json.get("confidence_score") or
        decision_json.get("confidence") or
        0
    )

    trade_plan = (
        decision_json.get("TradePlan") or
        decision_json.get("trade_plan") or
        decision_json.get("trading_plan") or
        {}
    )

    entry = trade_plan.get("Entry") or trade_plan.get("entry") or "N/A"
    stop = trade_plan.get("Stop") or trade_plan.get("stop") or "N/A"
    target = trade_plan.get("TargetFull") or trade_plan.get("target") or trade_plan.get("Target") or "N/A"
    rr = trade_plan.get("RiskReward") or trade_plan.get("risk_reward") or "N/A"

    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "close": data.get("close", 0),
        "rsi": data.get("rsi", 0),
        "macd_cross": data.get("macd_cross", ""),
        "volume_ratio": data.get("volume_ratio", 0),
        "weekly_trend": data.get("weekly_trend", ""),
        "critic_verdict": verdict,
        "decision": final_decision,
        "confidence": confidence,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": rr,
        "actual_result": "",
        "score": ""
    }

    df_new = pd.DataFrame([row])
    if os.path.exists(log_file):
        df_existing = pd.read_csv(log_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(log_file, index=False)
    print(f"✅ Shadow log saved for {symbol}")
    return row


# ============================================================
# INSTITUTIONAL MEMORY LAYER
# ============================================================
def load_memory(symbol, log_file="shadow_log.csv"):
    """Load last 5 decisions for symbol and calculate accuracy"""
    if not os.path.exists(log_file):
        return None

    try:
        df = pd.read_csv(log_file)
        df_symbol = df[df["symbol"] == symbol].tail(10)

        if df_symbol.empty:
            return None

        # Last 5 decisions
        last_5 = df_symbol.tail(5)[["date", "close", "rsi", "macd_cross", 
                                     "critic_verdict", "decision", "confidence",
                                     "entry", "stop", "target", "rr",
                                     "actual_result", "score"]].to_dict(orient="records")

        # Calculate stats
        total = len(df_symbol)
        scored = df_symbol[df_symbol["score"] != ""].copy()
        
        if len(scored) > 0:
            scored["score_num"] = pd.to_numeric(scored["score"], errors="coerce")
            avg_score = round(scored["score_num"].mean(), 2)
            wins = len(scored[scored["score_num"] > 0])
            win_rate = round(wins / len(scored) * 100, 1)
        else:
            avg_score = 0
            win_rate = 0

        # Execute decisions
        execute_count = len(df_symbol[df_symbol["decision"] == "Execute"])
        no_trade_count = len(df_symbol[df_symbol["decision"] == "No Trade"])

        return {
            "symbol": symbol,
            "total_sessions": total,
            "last_5_decisions": last_5,
            "win_rate": win_rate,
            "avg_score": avg_score,
            "execute_count": execute_count,
            "no_trade_count": no_trade_count,
            "note": "Memory loaded from shadow_log.csv"
        }
    except Exception as e:
        print(f"Memory load error: {e}")
        return None

def format_memory(memory):
    """Format memory for prompt injection"""
    if not memory:
        return "No previous sessions for this asset. This is the first analysis."

    lines = [
        f"INSTITUTIONAL MEMORY — {memory['symbol']}",
        f"Total sessions: {memory['total_sessions']}",
        f"Win rate: {memory['win_rate']}%",
        f"Avg score: {memory['avg_score']}",
        f"Execute decisions: {memory['execute_count']}",
        f"No Trade decisions: {memory['no_trade_count']}",
        "",
        "Last 5 decisions:"
    ]

    for d in memory["last_5_decisions"]:
        result = d.get("actual_result", "") or "Pending"
        score = d.get("score", "") or "Pending"
        lines.append(
            f"  {d['date']} | Close: {d['close']} | RSI: {d['rsi']} | "
            f"Decision: {d['decision']} | Confidence: {d['confidence']}% | "
            f"Result: {result} | Score: {score}"
        )

    return "\n".join(lines)

# ============================================================
# MAIN COUNCIL FUNCTION
# ============================================================
def run_council(symbol):
    print(f"\n{'='*50}")
    print(f"ADVISORY COUNCIL — {symbol}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    protocol_file = "council_protocol.txt"
    if os.path.exists(protocol_file):
        with open(protocol_file, "r", encoding="utf-8") as f:
            protocol = f.read()
    else:
        protocol = "Advisory Council v2.1 — enforce all roles strictly."

    print(f"📡 Fetching live data for {symbol}...")
    data = fetch_live_data(symbol)
    if not data:
        print(f"❌ Failed to fetch data for {symbol}")
        return None

    print(f"✅ Close: ${data['close']} | RSI: {data['rsi']} | MACD: {data['macd_cross']} | Vol: {data['volume_ratio']}x")

    # Load institutional memory
    memory = load_memory(symbol)
    memory_text = format_memory(memory)
    if memory:
        print(f"📚 Memory: {memory['total_sessions']} sessions | Win rate: {memory['win_rate']}%")
    else:
        print(f"📚 No previous memory for {symbol} — first session")

    stock_news = fetch_stock_news(symbol)
    macro_news = fetch_macro_news()

    print("\n[Phase 1] Analyst (Claude)...")
    analyst_output = run_analyst(symbol, data, protocol, memory_text)
    print("✅ Analyst done.")

    print("[Phase 2] Market Intelligence (GPT)...")
    market_output = run_market_intelligence(symbol, stock_news, macro_news, protocol, memory_text)
    print("✅ Market Intelligence done.")

    print("[Phase 3] Critic (GPT)...")
    critic_output = run_critic(symbol, analyst_output, market_output, protocol)
    print("✅ Critic done.")

    print("[Phase 4] Decision Engine (GPT)...")
    decision_output = run_decision_engine(symbol, analyst_output, market_output, critic_output, protocol)
    print("✅ Decision Engine done.")

    log = save_shadow_log(symbol, analyst_output, market_output, critic_output, decision_output, data)

    print(f"\n{'='*50}")
    print(f"COUNCIL DECISION — {symbol}")
    print(f"{'='*50}")
    print(f"Close:      ${data['close']}")
    print(f"RSI:        {data['rsi']}")
    print(f"MACD:       {data['macd_cross']}")
    print(f"Weekly:     {data['weekly_trend']}")
    print(f"Verdict:    {log['critic_verdict']}")
    print(f"Decision:   {log['decision']}")
    print(f"Confidence: {log['confidence']}%")
    print(f"Entry:      {log['entry']}")
    print(f"Stop:       {log['stop']}")
    print(f"Target:     {log['target']}")
    print(f"R/R:        {log['rr']}")
    print(f"{'='*50}\n")

    return {
        "symbol": symbol,
        "data": data,
        "analyst": analyst_output,
        "market": market_output,
        "critic": critic_output,
        "decision": decision_output,
        "summary": log
    }

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    stocks = ["NVDA", "MSFT", "AMZN", "ARM", "PLUG", "QCOM"]
    for stock in stocks:
        run_council(stock)
        print(f"Completed: {stock}")
