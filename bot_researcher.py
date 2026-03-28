import yfinance as yf
import pandas as pd
import schedule
import time
import os
from datetime import datetime
import asyncio
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
# Sector ETFs for comparison
SECTORS = {
    "NVDA": "SOXX",
    "MSFT": "QQQ",
    "AMZN": "XLY",
    "ARM": "SOXX",
    "PLUG": "ICLN",
    "QCOM": "SOXX"
}

DATA_FILE = "market_data.csv"

# ============================================================
# FETCH COMPREHENSIVE DATA
# ============================================================
def fetch_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Daily history - 6 months
        hist = ticker.history(period="6mo", interval="1d")
        
        if len(hist) < 14:
            return None

        latest = hist.iloc[-1]
        prev = hist.iloc[-2]

        # ── RSI ──
        delta = hist["Close"].diff()
        gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rsi = round(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])), 2)

        # ── MACD ──
        ema12 = hist["Close"].ewm(span=12).mean()
        ema26 = hist["Close"].ewm(span=26).mean()
        macd = round((ema12 - ema26).iloc[-1], 4)
        signal = round((ema12 - ema26).ewm(span=9).mean().iloc[-1], 4)
        macd_cross = "bullish" if macd > signal else "bearish"

        # ── Bollinger Bands ──
        sma20 = hist["Close"].rolling(20).mean().iloc[-1]
        std20 = hist["Close"].rolling(20).std().iloc[-1]
        bb_upper = round(sma20 + 2 * std20, 2)
        bb_lower = round(sma20 - 2 * std20, 2)
        bb_position = round(((latest["Close"] - bb_lower) / (bb_upper - bb_lower)) * 100, 1)

        # ── MFI ──
        tp = (hist["High"] + hist["Low"] + hist["Close"]) / 3
        mf = tp * hist["Volume"]
        pos_f = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_f = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        mfi = round(100 - (100 / (1 + pos_f.iloc[-1] / neg_f.iloc[-1])), 2)

        # ── ATR (Volatility) ──
        tr = pd.concat([
            hist["High"] - hist["Low"],
            (hist["High"] - hist["Close"].shift()).abs(),
            (hist["Low"] - hist["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr = round(tr.rolling(14).mean().iloc[-1], 2)

        # ── Volume Analysis ──
        avg_volume_20 = round(hist["Volume"].rolling(20).mean().iloc[-1])
        volume_ratio = round(latest["Volume"] / avg_volume_20, 2)

        # ── Moving Averages ──
        sma50 = round(hist["Close"].rolling(50).mean().iloc[-1], 2)
        sma200 = round(hist["Close"].rolling(200).mean().iloc[-1], 2) if len(hist) >= 200 else None
        ema20 = round(hist["Close"].ewm(span=20).mean().iloc[-1], 2)

        # ── Price Position ──
        high_52w = round(hist["Close"].max(), 2)
        low_52w = round(hist["Close"].min(), 2)
        pct_from_high = round(((latest["Close"] - high_52w) / high_52w) * 100, 2)
        pct_from_low = round(((latest["Close"] - low_52w) / low_52w) * 100, 2)

        # ── Weekly Change ──
        week_ago = hist["Close"].iloc[-6] if len(hist) >= 6 else hist["Close"].iloc[0]
        weekly_change = round(((latest["Close"] - week_ago) / week_ago) * 100, 2)

        # ── Monthly Change ──
        month_ago = hist["Close"].iloc[-22] if len(hist) >= 22 else hist["Close"].iloc[0]
        monthly_change = round(((latest["Close"] - month_ago) / month_ago) * 100, 2)

        # ── Sector ETF Comparison ──
        sector_change = 0
        try:
            sector_etf = SECTORS.get(symbol, "QQQ")
            sector_ticker = yf.Ticker(sector_etf)
            sector_hist = sector_ticker.history(period="2d", interval="1d")
            if len(sector_hist) >= 2:
                sector_change = round(
                    ((sector_hist["Close"].iloc[-1] - sector_hist["Close"].iloc[-2])
                     / sector_hist["Close"].iloc[-2]) * 100, 2)
        except:
            sector_change = 0

        # ── Fundamental Data ──
        pe_ratio = round(info.get("trailingPE", 0) or 0, 2)
        earnings_date = str(info.get("earningsTimestamp", "N/A"))
        market_cap = info.get("marketCap", 0)
        market_cap_b = round(market_cap / 1e9, 1) if market_cap else 0

        # ── Trend Direction ──
        if latest["Close"] > sma50 and sma50 > sma200 if sma200 else latest["Close"] > sma50:
            trend = "strong_uptrend"
        elif latest["Close"] > sma50:
            trend = "uptrend"
        elif latest["Close"] < sma50:
            trend = "downtrend"
        else:
            trend = "sideways"

        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "symbol": symbol,

            # Price
            "close": round(latest["Close"], 2),
            "prev_close": round(prev["Close"], 2),
            "change_pct": round(((latest["Close"] - prev["Close"]) / prev["Close"]) * 100, 2),
            "weekly_change": weekly_change,
            "monthly_change": monthly_change,
            "pre_market": round(info.get("preMarketPrice", 0) or 0, 2),
            "after_hours": round(info.get("postMarketPrice", 0) or 0, 2),

            # Technical
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal,
            "macd_cross": macd_cross,
            "mfi": mfi,
            "atr": atr,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_position": bb_position,

            # Moving Averages
            "sma50": sma50,
            "sma200": sma200,
            "ema20": ema20,
            "trend": trend,

            # Volume
            "volume": int(latest["Volume"]),
            "avg_volume_20": int(avg_volume_20),
            "volume_ratio": volume_ratio,

            # 52-Week
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_from_high": pct_from_high,
            "pct_from_low": pct_from_low,

            # Sector
            "sector_etf": SECTORS.get(symbol, "QQQ"),
            "sector_change": sector_change,
            "vs_sector": round(
                ((latest["Close"] - prev["Close"]) / prev["Close"] * 100) - sector_change, 2),

            # Fundamental
            "pe_ratio": pe_ratio,
            "market_cap_b": market_cap_b,
            "earnings_date": earnings_date,
        }
        return data

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ============================================================
# SAVE DATA
# ============================================================
def save_data(data_list):
    df_new = pd.DataFrame(data_list)
    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(DATA_FILE, index=False)
    print(f"Data saved — {len(df_combined)} total records")

# ============================================================
# SEND TELEGRAM
# ============================================================
async def send_telegram(message):
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        print("Telegram message sent")
    except Exception as e:
        print(f"Telegram error: {e}")

# ============================================================
# MAIN JOB
# ============================================================
def run_job():
    print(f"\n--- Job started: {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")

    results = []
    message_lines = ["📊 Market Update\n"]

    for symbol in STOCKS:
        data = fetch_stock_data(symbol)
        if data:
            results.append(data)

            emoji = "🟢" if data["change_pct"] >= 0 else "🔴"
            trend_emoji = "📈" if "up" in data["trend"] else "📉" if "down" in data["trend"] else "➡️"
            volume_emoji = "🔥" if data["volume_ratio"] > 1.5 else ""

            message_lines.append(
                f"{emoji} {symbol}: ${data['close']} ({data['change_pct']:+.2f}%) {trend_emoji}{volume_emoji}\n"
                f"   RSI: {data['rsi']} | MFI: {data['mfi']} | BB: {data['bb_position']}%\n"
                f"   Volume: {data['volume_ratio']}x avg | vs Sector: {data['vs_sector']:+.2f}%\n"
                f"   Pre: ${data['pre_market']} | AH: ${data['after_hours']}"
            )
            print(f"{symbol} fetched — RSI:{data['rsi']} Trend:{data['trend']}")

    if results:
        save_data(results)
        message_lines.append(f"\nRecords saved: {len(results)}")
        asyncio.run(send_telegram("\n".join(message_lines)))

# ============================================================
# SCHEDULER
# ============================================================
def start_bot():
    print("Bot Researcher started...")
    print(f"Monitoring: {', '.join(STOCKS)}")
    print("Schedule: 8:00 AM and 4:30 PM daily")

    run_job()

    schedule.every().day.at("08:00").do(run_job)
    schedule.every().day.at("16:30").do(run_job)

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_bot()