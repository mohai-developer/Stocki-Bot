from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import yfinance as yf
import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATA_FILE = "market_data.csv"

def get_latest_data(symbol):
    try:
        if not os.path.exists(DATA_FILE): return None
        df = pd.read_csv(DATA_FILE)
        df_symbol = df[df["symbol"] == symbol.upper()].tail(5)
        return df_symbol.to_dict(orient="records") if not df_symbol.empty else None
    except: return None

def get_stock_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return "\n".join([f"- {n['title']}" for n in ticker.news[:3]])
    except: return "No news"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        body = request.json
        symbol = body.get("symbol", "").upper()
        mode = body.get("analysis_type", "general")
        report = body.get("report_type", "full")
        entry = body.get("entry_price", "")

        data = get_latest_data(symbol)
        news = get_stock_news(symbol)
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        if mode == "position":
            prompt = f"حلل صفقة {symbol} بسعر دخول {entry}. البيانات: {data}. اخبار: {news}. قرر استمرار او خروج مع هدف ووقف. اللغة: العربية."
        else:
            style = "مفصل جداً" if report == "full" else "مختصر جداً"
            prompt = f"حلل سهم {symbol} تقرير {style}. البيانات: {data}. اخبار: {news}. اقترح دخول وهدف ووقف. اللغة: العربية."

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return jsonify({"status": "success", "analysis": message.content[0].text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
