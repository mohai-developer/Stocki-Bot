"""
ces_scanner.py
══════════════════════════════════════════════════════════════
أداة المسح اليومي — CES v5
تشغّلها كل صباح قبل افتتاح السوق
تعطيك قائمة بالأسهم التي عندها إشارة الآن

الاستخدام:
  python ces_scanner.py              ← مسح القائمة الكاملة
  python ces_scanner.py --all        ← مع كل التفاصيل
  python ces_scanner.py --watchlist  ← الممتازة فقط
══════════════════════════════════════════════════════════════
"""

import sys
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance pandas numpy")
    sys.exit(1)

# استيراد من ces_v5
try:
    from ces_v5 import (
        STOCK_PROFILES, get_profile,
        gaussian_kernel_regression,
        stochastic_rsi,
        yang_zhang_iv_rank,
        fetch_options_data,
        score_kernel_trend, score_stoch_rsi,
        score_put_call_ratio, score_oi_momentum,
        score_iv_skew, score_iv_rank,
        WEIGHTS, ENTRY_THRESHOLD
    )
    print("  ces_v5 imported ✓")
except ImportError as e:
    print(f"  تأكد أن ces_v5.py في نفس المجلد: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# Watchlist
# ══════════════════════════════════════════════════════════════
WATCHLIST_FULL = {
    "ممتاز": ["NVDA", "AMD", "META", "ORCL", "GOOG"],
    "جيد":   ["CRM"],
    "احتياطي": ["MU", "TSLA"],
}

WATCHLIST_FLAT = [s for group in WATCHLIST_FULL.values() for s in group]


# ══════════════════════════════════════════════════════════════
# حساب CES بسيط (بدون Options — للسرعة)
# ══════════════════════════════════════════════════════════════
def quick_ces(symbol: str, profile: dict) -> dict:
    """حساب CES سريع من بيانات السهم فقط"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y", interval="1d", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].copy()

        if len(df) < 60:
            return {"error": "بيانات غير كافية"}

        price = float(df["Close"].iloc[-1])

        # Kernel
        kernel = gaussian_kernel_regression(df["Close"], bandwidth=20)

        # Stochastic RSI
        k_line, d_line = stochastic_rsi(df["Close"])

        # IV Rank
        iv_arr = yang_zhang_iv_rank(df)
        iv_val = float(iv_arr[-1]) if len(iv_arr) > 0 else 50.0

        # الدرجات الفنية
        sc_kernel = score_kernel_trend(df["Close"], kernel)
        sc_srsi   = score_stoch_rsi(k_line, d_line)
        sc_iv     = score_iv_rank(iv_val)

        # أوزان سريعة بدون Options (نعيد توزيع الأوزان)
        w_k  = WEIGHTS["kernel"]  + WEIGHTS["oi_mom"] // 2
        w_sr = WEIGHTS["stoch_rsi"] + WEIGHTS["pcr"] // 2
        w_iv = WEIGHTS["iv_rank"] + WEIGHTS["iv_skew"] // 2
        tw   = w_k + w_sr + w_iv + WEIGHTS["oi_mom"] // 2 + WEIGHTS["pcr"] // 2 + WEIGHTS["iv_skew"] // 2

        ces_raw = (sc_kernel * w_k + sc_srsi * w_sr + sc_iv * w_iv) / (w_k + w_sr + w_iv)
        ces_raw = float(pd.Series([ces_raw]).ewm(span=3, adjust=False).mean().iloc[-1])

        thr = profile["threshold"]

        # مسافة من العتبة
        dist = ces_raw - thr

        # حالة الـ CES قبل يومين للكشف عن الاتجاه
        return {
            "symbol":    symbol,
            "price":     round(price, 2),
            "ces":       round(ces_raw, 1),
            "threshold": thr,
            "distance":  round(dist, 1),
            "iv_rank":   round(iv_val, 1),
            "style":     profile["style"],
            "signal":    "ادخل" if ces_raw >= thr else ("قريب" if dist >= -8 else "انتظر"),
        }

    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# المسح الكامل
# ══════════════════════════════════════════════════════════════
def run_scanner(symbols: list, show_all: bool = False) -> list:
    results = []

    print(f"\n{'═'*58}")
    print(f"  CES Scanner — المسح اليومي")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | {len(symbols)} سهم")
    print(f"{'═'*58}\n")

    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        profile = get_profile(symbol)
        r = quick_ces(symbol, profile)
        results.append(r)
        if "error" not in r:
            icon = "🟢" if r["signal"] == "ادخل" else ("🟡" if r["signal"] == "قريب" else "⚪")
            print(f"{icon} CES={r['ces']} | عتبة={r['threshold']} | {r['signal']}")
        else:
            print(f"❌ {r['error']}")

    return results


# ══════════════════════════════════════════════════════════════
# طباعة التقرير
# ══════════════════════════════════════════════════════════════
def print_report(results: list):
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("  لا توجد نتائج")
        return

    # تصنيف
    enter  = [r for r in ok if r["signal"] == "ادخل"]
    near   = [r for r in ok if r["signal"] == "قريب"]
    wait   = [r for r in ok if r["signal"] == "انتظر"]

    print(f"\n{'═'*58}")
    print(f"  ملخص المسح اليومي — {datetime.datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'═'*58}")

    if enter:
        print(f"\n  🟢 إشارات دخول نشطة ({len(enter)}):")
        print(f"  {'─'*56}")
        print(f"  {'السهم':<7}  {'CES':>5}  {'العتبة':>6}  {'المسافة':>8}  {'IV Rank':>8}  النمط")
        print(f"  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}")
        for r in sorted(enter, key=lambda x: x["distance"], reverse=True):
            print(f"  {r['symbol']:<7}  {r['ces']:>5}  {r['threshold']:>6}  {r['distance']:>+8.1f}  {r['iv_rank']:>7.1f}%  {r['style']}")

    if near:
        print(f"\n  🟡 قريب من الإشارة ({len(near)}) — راقب:")
        print(f"  {'─'*56}")
        print(f"  {'السهم':<7}  {'CES':>5}  {'العتبة':>6}  {'المسافة':>8}  {'IV Rank':>8}  النمط")
        print(f"  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}")
        for r in sorted(near, key=lambda x: x["distance"], reverse=True):
            print(f"  {r['symbol']:<7}  {r['ces']:>5}  {r['threshold']:>6}  {r['distance']:>+8.1f}  {r['iv_rank']:>7.1f}%  {r['style']}")

    if wait:
        print(f"\n  ⚪ انتظر ({len(wait)}):")
        for r in sorted(wait, key=lambda x: x["distance"], reverse=True):
            print(f"  {r['symbol']:<7}  CES={r['ces']:>5}  بعد {abs(r['distance']):.0f} نقطة عن العتبة")

    print(f"\n{'═'*58}")
    print(f"  الملخص: {len(enter)} إشارة | {len(near)} قريب | {len(wait)} انتظار")
    print(f"{'═'*58}\n")

    # تحذير إذا لا توجد إشارات
    if not enter and not near:
        print("  ⚠ السوق لا يعطي إشارات — انتظر أو راجع وضع السوق العام\n")


# ══════════════════════════════════════════════════════════════
# حفظ النتائج
# ══════════════════════════════════════════════════════════════
def save_results(results: list):
    output = {
        "date":      datetime.datetime.now().strftime("%Y-%m-%d"),
        "time":      datetime.datetime.now().strftime("%H:%M"),
        "results":   results,
        "signals":   [r["symbol"] for r in results if r.get("signal") == "ادخل"],
        "near":      [r["symbol"] for r in results if r.get("signal") == "قريب"],
    }
    fname = f"scan_{datetime.datetime.now().strftime('%Y%m%d')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"  محفوظ في: {fname}\n")
    return fname


# ══════════════════════════════════════════════════════════════
# نقطة الدخول
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    show_all    = "--all"       in sys.argv
    only_watch  = "--watchlist" in sys.argv

    if only_watch:
        symbols = WATCHLIST_FULL["ممتاز"]
    else:
        symbols = WATCHLIST_FLAT

    results = run_scanner(symbols, show_all)
    print_report(results)
    save_results(results)