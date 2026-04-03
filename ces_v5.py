"""
ces_v5.py  —  Call Entry Score v5
══════════════════════════════════════════════════════════════
النقلة النوعية: من المؤشرات الفنية لذكاء سوق الأوبشن

المصادر:
  - yfinance: بيانات السهم + بيانات الأوبشن
  - CBOE (اختياري): Put/Call Ratio اليومي

المؤشرات الجديدة:
  1. Kernel Regression (بدل MA) — اتجاه ذكي بلا تأخر
  2. Stochastic RSI — زخم سريع
  3. Put/Call Ratio — ذكاء سوق الأوبشن
  4. OI Momentum — تراكم العقود الذكي
  5. IV Skew — توقع السوق للاتجاه
  6. Yang-Zhang IV Rank — تكلفة البريميوم

الاستخدام:
  pip install yfinance pandas numpy scipy
  python ces_v5.py MU
  python ces_v5.py NVDA --days 5
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
    print("pip install yfinance pandas numpy scipy")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# ملف إعدادات الأسهم — لكل سهم شخصيته
# ══════════════════════════════════════════════════════════════
STOCK_PROFILES = {
    "AMD": {
        "threshold":     80,     # عتبة الدخول المثلى
        "profit_target": 0.15,   # هدف الربح
        "stop_loss":     0.07,   # وقف الخسارة
        "use_squeeze":   True,   # Squeeze يعمل ممتاز
        "regime_filter": True,   # فلتر السوق مفيد
        "description":   "انتقائي — إشارات نادرة وعالية الجودة",
        "style":         "انتقائي",
    },
    "NVDA": {
        "threshold":     78,
        "profit_target": 0.15,
        "stop_loss":     0.07,
        "use_squeeze":   True,   # Squeeze يحسّن النتائج
        "regime_filter": True,   # فلتر السوق مفيد جداً
        "description":   "زخم قوي — يستفيد من فلتر السوق",
        "style":         "زخم",
    },
    "MU": {
        "threshold":     68,
        "profit_target": 0.15,
        "stop_loss":     0.07,
        "use_squeeze":   False,  # Squeeze يخرج مبكراً
        "regime_filter": True,   # فلتر السوق مفيد
        "description":   "متوازن — يحتاج وقتاً للحركة",
        "style":         "متوازن",
    },
    "TSLA": {
        "threshold":     63,
        "profit_target": 0.15,
        "stop_loss":     0.07,
        "use_squeeze":   False,  # Squeeze يضر
        "regime_filter": False,  # يتحرك باستقلالية
        "description":   "متقلب — يتحرك بمنطق مختلف",
        "style":         "متقلب",
    },
    "META": {
        "threshold":     82,
        "profit_target": 0.15,
        "stop_loss":     0.07,
        "use_squeeze":   True,
        "regime_filter": False,
        "description":   "زخم قوي — يتحرك باستقلالية عن السوق",
        "style":         "زخم",
    },
    "ORCL": {
        "threshold":     82,
        "profit_target": 0.15,
        "stop_loss":     0.07,
        "use_squeeze":   True,
        "regime_filter": True,
        "description":   "انتقائي جداً — إشارات نادرة ودقيقة",
        "style":         "انتقائي",
    },
    "GOOG": {
        "threshold":     62,
        "profit_target": 0.15,
        "stop_loss":     0.07,
        "use_squeeze":   True,
        "regime_filter": True,
        "description":   "نشط — إشارات متكررة وعالية الجودة",
        "style":         "نشط",
    },
    "CRM": {
        "threshold":     75,
        "profit_target": 0.15,
        "stop_loss":     0.07,
        "use_squeeze":   True,
        "regime_filter": True,
        "description":   "جيد — يحتاج مراقبة",
        "style":         "متوازن",
    },
    # قالب للأسهم الجديدة
    "DEFAULT": {
        "threshold":     72,
        "profit_target": 0.15,
        "stop_loss":     0.07,
        "use_squeeze":   True,
        "regime_filter": True,
        "description":   "إعدادات افتراضية",
        "style":         "افتراضي",
    },
}

def get_profile(symbol: str) -> dict:
    """يجلب إعدادات السهم أو الافتراضية إذا لم يكن محدداً"""
    profile = STOCK_PROFILES.get(symbol, STOCK_PROFILES["DEFAULT"]).copy()
    profile["symbol"] = symbol
    return profile


# ══════════════════════════════════════════════════════════════
# الإعدادات العامة
# ══════════════════════════════════════════════════════════════
SYMBOL   = sys.argv[1].upper() if len(sys.argv) > 1 else "MU"
OUTPUT   = f"ces_v5_{SYMBOL}.json"
PROFILE  = get_profile(SYMBOL)

# أوزان CES v5 (مبنية على نتائج الـ Optimizer + منطق Options)
WEIGHTS = {
    "kernel":  20,   # الاتجاه الذكي
    "stoch_rsi": 15, # الزخم السريع
    "pcr":     20,   # Put/Call Ratio — ذكاء الأوبشن
    "oi_mom":  20,   # Open Interest Momentum
    "iv_skew": 15,   # IV Skew
    "iv_rank": 10,   # تكلفة البريميوم
}

ENTRY_THRESHOLD = 72   # عتبة الدخول


# ══════════════════════════════════════════════════════════════
# 1. جلب البيانات
# ══════════════════════════════════════════════════════════════

def fetch_stock_data(symbol: str) -> pd.DataFrame:
    """يجلب بيانات السهم التاريخية"""
    print(f"  جارٍ جلب بيانات {symbol}...", end=" ", flush=True)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y", interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    print(f"✓ {len(df)} يوم")
    return df


def fetch_options_data(symbol: str, current_price: float) -> dict:
    """
    يجلب بيانات الأوبشن من Yahoo Finance:
    - Put/Call Ratio
    - Open Interest لكل Strike
    - IV للكول والبوت (Skew)
    """
    print(f"  جارٍ جلب بيانات الأوبشن...", end=" ", flush=True)

    ticker  = yf.Ticker(symbol)
    dates   = ticker.options

    if not dates:
        print("لا توجد عقود")
        return {}

    results = {
        "pcr_list":      [],   # Put/Call Ratio لكل أجل
        "oi_calls":      [],   # OI للكول
        "oi_puts":       [],   # OI للبوت
        "iv_calls_atm":  [],   # IV كول ATM
        "iv_puts_atm":   [],   # IV بوت ATM
        "total_call_oi": 0,
        "total_put_oi":  0,
        "dates_used":    [],
    }

    # نأخذ أقرب 3 آجال
    for date in dates[:3]:
        try:
            chain = ticker.option_chain(date)
            calls = chain.calls.copy()
            puts  = chain.puts.copy()

            # فلتر OI منخفض
            calls = calls[calls["openInterest"] > 10]
            puts  = puts[puts["openInterest"]  > 10]

            if calls.empty or puts.empty:
                continue

            # Put/Call Ratio بالحجم
            total_call_vol = calls["volume"].fillna(0).sum()
            total_put_vol  = puts["volume"].fillna(0).sum()
            if total_call_vol > 0:
                pcr = total_put_vol / total_call_vol
                results["pcr_list"].append(pcr)

            # OI إجمالي
            results["oi_calls"].append(calls["openInterest"].sum())
            results["oi_puts"].append(puts["openInterest"].sum())
            results["total_call_oi"] += calls["openInterest"].sum()
            results["total_put_oi"]  += puts["openInterest"].sum()

            # IV Skew — ATM ±5%
            lo, hi = current_price * 0.95, current_price * 1.05
            atm_calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
            atm_puts  = puts[(puts["strike"]  >= lo) & (puts["strike"]  <= hi)]

            if not atm_calls.empty:
                iv_c = atm_calls["impliedVolatility"].replace(0, np.nan).median()
                if pd.notna(iv_c):
                    results["iv_calls_atm"].append(iv_c * 100)

            if not atm_puts.empty:
                iv_p = atm_puts["impliedVolatility"].replace(0, np.nan).median()
                if pd.notna(iv_p):
                    results["iv_puts_atm"].append(iv_p * 100)

            results["dates_used"].append(date)

        except Exception:
            continue

    print(f"✓ {len(results['dates_used'])} أجل")
    return results


# ══════════════════════════════════════════════════════════════
# 2. حساب المؤشرات الفنية
# ══════════════════════════════════════════════════════════════

def gaussian_kernel_regression(prices: pd.Series, bandwidth: int = 20) -> pd.Series:
    """
    Gaussian Kernel Regression — بديل ذكي للمتوسط المتحرك
    يعطي وزناً أعلى للأسعار الأقرب زمنياً
    النتيجة: اتجاه أنعم وأقل تأخراً من MA
    """
    n = len(prices)
    result = np.zeros(n)
    p = prices.values

    for i in range(n):
        # المسافة الزمنية من كل نقطة للنقطة الحالية
        distances = np.arange(i + 1)[::-1]
        # أوزان Gaussian
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weights = weights / weights.sum()
        result[i] = np.dot(weights, p[:i + 1])

    return pd.Series(result, index=prices.index)


def stochastic_rsi(close: pd.Series, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3) -> tuple:
    """Stochastic RSI — أسرع من RSI في اكتشاف نقاط الانعكاس"""
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/rsi_len, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/rsi_len, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    # Stochastic على RSI
    rsi_min = rsi.rolling(stoch_len).min()
    rsi_max = rsi.rolling(stoch_len).max()
    rng     = rsi_max - rsi_min
    stoch_k = np.where(rng > 0, (rsi - rsi_min) / rng * 100, 50)
    stoch_k = pd.Series(stoch_k, index=close.index)

    k = stoch_k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


def yang_zhang_iv_rank(df: pd.DataFrame, yz_len=20, rank_len=252) -> pd.Series:
    """Yang-Zhang IV Rank المحسوب تلقائياً"""
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]
    n = yz_len
    k = 0.34 / (1.34 + (n+1)/(n-1))

    yz = np.sqrt(
        np.log(c/c.shift(1)).rolling(n).var() +
        k * np.log(c/o).rolling(n).var() +
        (1-k) * (np.log(h/c)*np.log(h/o) + np.log(l/c)*np.log(l/o)).rolling(n).mean()
    ).clip(lower=0) * np.sqrt(252) * 100

    mn = yz.rolling(rank_len).min()
    mx = yz.rolling(rank_len).max()
    return np.where(mx > mn, (yz - mn) / (mx - mn) * 100, 50.0)


# ══════════════════════════════════════════════════════════════
# 3. حساب الدرجات
# ══════════════════════════════════════════════════════════════

def score_kernel_trend(close: pd.Series, kernel: pd.Series) -> float:
    """
    درجة الاتجاه بناءً على Kernel Regression
    - السعر فوق الكيرنل + الكيرنل صاعد = 100
    - السعر فوق الكيرنل فقط = 65
    - السعر تحت الكيرنل = 0
    """
    if len(close) < 10:
        return 50.0

    price_now   = close.iloc[-1]
    kernel_now  = kernel.iloc[-1]
    kernel_prev = kernel.iloc[-5]   # ميل الكيرنل
    kernel_slope = (kernel_now - kernel_prev) / kernel_prev * 100

    if price_now > kernel_now and kernel_slope > 0.5:
        return 100.0
    if price_now > kernel_now and kernel_slope > 0:
        return 80.0
    if price_now > kernel_now:
        return 65.0
    if price_now < kernel_now and kernel_slope < -0.5:
        return 0.0
    return 30.0


def score_stoch_rsi(k: pd.Series, d: pd.Series) -> float:
    """
    درجة Stochastic RSI
    - K صاعد من منطقة ذروة بيع (< 20) = 100
    - K فوق D وكلاهما صاعد = 80
    - K فوق 80 (ذروة شراء) = 10
    """
    if k.isna().iloc[-1] or d.isna().iloc[-1]:
        return 50.0

    k_now, k_prev = k.iloc[-1], k.iloc[-2]
    d_now = d.iloc[-1]
    rising = k_now > k_prev

    if k_prev < 20 and k_now > k_prev:   return 100.0
    if k_now > d_now and rising and k_now < 80: return 80.0
    if k_now > d_now and rising:          return 60.0
    if k_now > d_now:                     return 50.0
    if k_now > 80:                        return 10.0
    return 25.0


def score_put_call_ratio(opts: dict) -> float:
    """
    درجة Put/Call Ratio
    PCR منخفض = السوق متفائل = جيد للكول
    PCR مرتفع = تحوط = تحذير

    نطاقات PCR للأسهم الفردية:
    < 0.5  = تفاؤل مفرط (احذر)
    0.5-0.8 = متوازن صاعد (ممتاز للكول)
    0.8-1.2 = محايد
    > 1.2  = تشاؤم (تجنب الكول)
    """
    if not opts or not opts.get("pcr_list"):
        return 50.0  # لا بيانات — درجة محايدة

    pcr = np.median(opts["pcr_list"])

    # OI Ratio (أكثر موثوقية من Volume PCR)
    total_call_oi = opts.get("total_call_oi", 0)
    total_put_oi  = opts.get("total_put_oi", 0)
    if total_call_oi > 0:
        oi_pcr = total_put_oi / total_call_oi
        # متوسط بين Volume PCR و OI PCR
        pcr = (pcr + oi_pcr) / 2

    if pcr < 0.5:   return 40.0   # تفاؤل مفرط — خطر انعكاس
    if pcr < 0.7:   return 100.0  # صاعد قوي
    if pcr < 0.9:   return 85.0   # صاعد معتدل
    if pcr < 1.1:   return 60.0   # محايد
    if pcr < 1.3:   return 30.0   # تشاؤم
    return 10.0                    # تشاؤم شديد


def score_oi_momentum(opts: dict) -> float:
    """
    درجة Open Interest Momentum
    تراكم OI على الكول = مؤسسات تراهن على الصعود
    """
    if not opts:
        return 50.0

    call_oi = opts.get("oi_calls", [])
    put_oi  = opts.get("oi_puts",  [])

    if not call_oi or not put_oi:
        return 50.0

    total_call = sum(call_oi)
    total_put  = sum(put_oi)

    if total_call + total_put == 0:
        return 50.0

    # نسبة الكول من إجمالي OI
    call_pct = total_call / (total_call + total_put) * 100

    # تحقق من تراكم الكول (الأجل القريب vs البعيد)
    if len(call_oi) >= 2:
        near_call = call_oi[0]
        far_call  = sum(call_oi[1:])
        # تراكم في الأجل القريب = رهان على حركة سريعة
        near_dominance = near_call / (near_call + far_call + 1e-9)
    else:
        near_dominance = 0.5

    if call_pct > 65 and near_dominance > 0.6:   return 100.0  # تراكم قوي جداً
    if call_pct > 60:                             return 85.0
    if call_pct > 55:                             return 70.0
    if call_pct > 50:                             return 60.0
    if call_pct > 45:                             return 40.0
    return 20.0


def score_iv_skew(opts: dict) -> float:
    """
    درجة IV Skew
    IV الكول > IV البوت = السوق يدفع أكثر للكول = توقع صعود
    IV البوت > IV الكول = السوق يشتري حماية = توقع هبوط
    """
    if not opts:
        return 50.0

    iv_calls = opts.get("iv_calls_atm", [])
    iv_puts  = opts.get("iv_puts_atm",  [])

    if not iv_calls or not iv_puts:
        return 50.0

    avg_call_iv = np.mean(iv_calls)
    avg_put_iv  = np.mean(iv_puts)

    if avg_put_iv == 0:
        return 50.0

    # Skew = فرق IV (إيجابي = كول أغلى = صاعد)
    skew = (avg_call_iv - avg_put_iv) / avg_put_iv * 100

    if skew > 10:    return 100.0  # سوق يدفع كثيراً للكول
    if skew > 5:     return 85.0
    if skew > 0:     return 70.0
    if skew > -5:    return 50.0
    if skew > -10:   return 30.0
    return 10.0                     # سوق يشتري حماية = تجنب الكول


def score_iv_rank(iv_rank_val: float) -> float:
    """درجة IV Rank — نفس منطق CES v4"""
    if iv_rank_val < 20: return 100.0
    if iv_rank_val < 35: return 85.0
    if iv_rank_val < 50: return 65.0
    if iv_rank_val < 70: return 25.0
    return 0.0


# ══════════════════════════════════════════════════════════════
# 4. CES v5 الرئيسي
# ══════════════════════════════════════════════════════════════

def compute_ces_v5(symbol: str) -> dict:
    print(f"\n{'═'*52}")
    print(f"  CES v5 — {symbol}")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*52}\n")

    # ── جلب البيانات ──────────────────────────────────────
    df      = fetch_stock_data(symbol)
    price   = float(df["Close"].iloc[-1])
    opts    = fetch_options_data(symbol, price)

    # ── المؤشرات الفنية ───────────────────────────────────
    print("  حساب المؤشرات...", end=" ", flush=True)

    # Kernel Regression
    kernel = gaussian_kernel_regression(df["Close"], bandwidth=20)

    # Stochastic RSI
    k_line, d_line = stochastic_rsi(df["Close"])

    # Yang-Zhang IV Rank
    iv_rank_series = yang_zhang_iv_rank(df)
    iv_rank_val    = float(iv_rank_series[-1]) if len(iv_rank_series) > 0 else 50.0

    print("✓")

    # ── الدرجات ───────────────────────────────────────────
    sc = {
        "kernel":    score_kernel_trend(df["Close"], kernel),
        "stoch_rsi": score_stoch_rsi(k_line, d_line),
        "pcr":       score_put_call_ratio(opts),
        "oi_mom":    score_oi_momentum(opts),
        "iv_skew":   score_iv_skew(opts),
        "iv_rank":   score_iv_rank(iv_rank_val),
    }

    # ── الدرجة الإجمالية ──────────────────────────────────
    tw  = sum(WEIGHTS.values())
    raw = sum(sc[k] * WEIGHTS[k] for k in sc) / tw
    ces = round(raw, 1)

    # ── الإشارة ───────────────────────────────────────────
    if ces >= ENTRY_THRESHOLD:
        signal = "ادخل"
        signal_icon = "✅"
    elif ces >= 55:
        signal = "انتظر"
        signal_icon = "⚠️"
    else:
        signal = "لا تدخل"
        signal_icon = "❌"

    # ── Put/Call معلومات إضافية ───────────────────────────
    pcr_val = round(float(np.median(opts["pcr_list"])), 2) if opts.get("pcr_list") else None
    call_oi = opts.get("total_call_oi", 0)
    put_oi  = opts.get("total_put_oi",  0)
    iv_skew_val = None
    if opts.get("iv_calls_atm") and opts.get("iv_puts_atm"):
        iv_skew_val = round(float(np.mean(opts["iv_calls_atm"])) - float(np.mean(opts["iv_puts_atm"])), 2)

    return {
        "symbol":        symbol,
        "price":         round(price, 2),
        "ces_score":     ces,
        "signal":        signal,
        "threshold":     ENTRY_THRESHOLD,
        "scores": {
            "kernel_trend": sc["kernel"],
            "stoch_rsi":    sc["stoch_rsi"],
            "put_call_ratio": sc["pcr"],
            "oi_momentum":  sc["oi_mom"],
            "iv_skew":      sc["iv_skew"],
            "iv_rank":      sc["iv_rank"],
        },
        "raw_data": {
            "iv_rank_pct":   round(iv_rank_val, 1),
            "pcr":           pcr_val,
            "call_oi":       int(call_oi),
            "put_oi":        int(put_oi),
            "iv_skew_pts":   iv_skew_val,
            "kernel_price":  round(float(kernel.iloc[-1]), 2),
            "stoch_k":       round(float(k_line.iloc[-1]), 1) if pd.notna(k_line.iloc[-1]) else None,
            "stoch_d":       round(float(d_line.iloc[-1]), 1) if pd.notna(d_line.iloc[-1]) else None,
        },
        "weights":    WEIGHTS,
        "timestamp":  datetime.datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════
# 5. طباعة التقرير
# ══════════════════════════════════════════════════════════════

def print_report(r: dict):
    sc  = r["scores"]
    raw = r["raw_data"]
    w   = r["weights"]

    def bar(s):
        n = int(s / 10)
        return "█" * n + "░" * (10 - n)

    def arrow(s):
        if s >= 70: return "↑"
        if s >= 40: return "→"
        return "↓"

    print(f"\n  {'═'*52}")
    print(f"  CES v5 — تقرير {r['symbol']}")
    print(f"  {'═'*52}")
    print(f"  السعر الحالي  : ${r['price']}")
    print(f"  {'─'*52}")
    print(f"  {'المؤشر':<22} {'الدرجة':>6}  {'الشريط':<12} {'القراءة'}")
    print(f"  {'─'*52}")

    indicators = [
        ("Kernel Trend",     sc["kernel_trend"],    f"Kernel=${raw['kernel_price']}",     w["kernel"]),
        ("Stochastic RSI",   sc["stoch_rsi"],        f"K={raw['stoch_k']} D={raw['stoch_d']}", w["stoch_rsi"]),
        ("Put/Call Ratio",   sc["put_call_ratio"],   f"PCR={raw['pcr']}",                  w["pcr"]),
        ("OI Momentum",      sc["oi_momentum"],      f"C={raw['call_oi']:,} P={raw['put_oi']:,}", w["oi_mom"]),
        ("IV Skew",          sc["iv_skew"],          f"Skew={raw['iv_skew_pts']}pts",      w["iv_skew"]),
        ("IV Rank (YZ)",     sc["iv_rank"],          f"{raw['iv_rank_pct']}%",             w["iv_rank"]),
    ]

    for name, score, reading, weight in indicators:
        print(f"  {name:<22} {score:>5.0f}  {bar(score):<12} {reading}  (×{weight}%)")

    print(f"  {'─'*52}")
    print(f"  الدرجة الإجمالية : {r['ces_score']:.1f} / 100")
    print(f"  الإشارة          : {r['signal']}")
    print(f"  {'═'*52}\n")

    # تفسير Options Flow
    print(f"  تفسير ذكاء الأوبشن:")
    if raw["pcr"] is not None:
        pcr = raw["pcr"]
        if pcr < 0.7:
            print(f"  ✓ PCR={pcr} — السوق متفائل، الكول مطلوب")
        elif pcr < 1.0:
            print(f"  ~ PCR={pcr} — محايد مع ميل صاعد")
        else:
            print(f"  ✗ PCR={pcr} — السوق يشتري حماية، احذر")

    if raw["call_oi"] and raw["put_oi"]:
        call_dom = raw["call_oi"] / (raw["call_oi"] + raw["put_oi"]) * 100
        print(f"  {'✓' if call_dom > 55 else '~'} Call OI {call_dom:.0f}% من إجمالي OI")

    if raw["iv_skew_pts"] is not None:
        skew = raw["iv_skew_pts"]
        if skew > 0:
            print(f"  ✓ IV Skew إيجابي ({skew:+.1f}pts) — سوق يدفع أكثر للكول")
        else:
            print(f"  ✗ IV Skew سلبي ({skew:+.1f}pts) — سوق يشتري حماية")

    print()


# ══════════════════════════════════════════════════════════════
# 6. Backtest على البيانات التاريخية
# ══════════════════════════════════════════════════════════════

def compute_ces_series_backtest(df: pd.DataFrame) -> pd.Series:
    """
    يحسب CES على كل الشموع التاريخية.
    ملاحظة: يستخدم المؤشرات الفنية فقط (بدون Options — لا توجد بيانات تاريخية للأوبشن)
    الأوزان تُعاد توزيعها على المؤشرات الفنية الأربعة
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    o = df["Open"]
    v = df["Volume"]

    # Kernel Regression
    kernel = gaussian_kernel_regression(c, bandwidth=20)
    kernel_slope = kernel - kernel.shift(5)

    sc_kernel = np.select(
        [(c > kernel) & (kernel_slope > 0),
         (c > kernel) & (kernel_slope > 0) & (kernel_slope < 0.5),
         c > kernel,
         (c < kernel) & (kernel_slope < 0)],
        [100, 80, 65, 0], 30
    )

    # Stochastic RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    rng_rsi = rsi_max - rsi_min
    stoch_k = np.where(rng_rsi > 0, (rsi - rsi_min) / rng_rsi * 100, 50)
    stoch_k = pd.Series(stoch_k, index=c.index).rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()
    stoch_k_prev = stoch_k.shift(1)

    sc_srsi = np.select(
        [(stoch_k.shift(1) < 20) & (stoch_k > stoch_k_prev),
         (stoch_k > stoch_d) & (stoch_k > stoch_k_prev) & (stoch_k < 80),
         stoch_k > stoch_d,
         stoch_k > 80],
        [100, 80, 50, 10], 25
    )

    # MACD (بديل مؤقت لـ OI Momentum في الباكتيست)
    e12  = c.ewm(span=12, adjust=False).mean()
    e26  = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig  = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    hist_prev = hist.shift(1)

    sc_macd = np.select(
        [(macd > sig) & (hist > hist_prev) & (macd > 0),
         (macd > sig) & (hist > hist_prev),
         macd > sig,
         (hist > hist_prev) & (hist < 0)],
        [100, 80, 55, 40], 10
    )

    # Yang-Zhang IV Rank
    iv_rank_arr = yang_zhang_iv_rank(df)
    iv_rank_s   = pd.Series(iv_rank_arr, index=df.index)

    sc_iv = np.select(
        [iv_rank_s < 20, iv_rank_s < 35, iv_rank_s < 50, iv_rank_s < 70],
        [100, 85, 65, 25], 0
    )

    # أوزان الباكتيست (بدون Options)
    w_k, w_sr, w_m, w_iv = 30, 25, 25, 20
    tw  = w_k + w_sr + w_m + w_iv
    raw = (pd.Series(sc_kernel, index=df.index) * w_k +
           pd.Series(sc_srsi,   index=df.index) * w_sr +
           pd.Series(sc_macd,   index=df.index) * w_m  +
           pd.Series(sc_iv,     index=df.index) * w_iv) / tw

    return raw.ewm(span=3, adjust=False).mean()



# ══════════════════════════════════════════════════════════════
# 7. نظام الإحصاء الاحترافي
# ══════════════════════════════════════════════════════════════

def compute_advanced_stats(trade_log: list, hold_days: int) -> dict:
    """
    يحسب إحصاءات احترافية شاملة على نتائج الباكتيست:
    - Expectancy، Drawdown، Streaks، Monthly، Market Regime
    """
    if not trade_log or len(trade_log) < 5:
        return {}

    rets  = np.array([t["return_pct"] / 100 for t in trade_log])
    wins  = rets[rets > 0]
    loss  = rets[rets <= 0]
    dates = [t["entry_date"] for t in trade_log]

    # ── 1. Expectancy ──────────────────────────────────────
    wr       = len(wins) / len(rets)
    avg_win  = float(wins.mean())  if len(wins)  else 0
    avg_loss = float(abs(loss.mean())) if len(loss) else 0
    expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)

    # ── 2. Equity Curve + Max Drawdown ────────────────────
    equity = np.cumprod(1 + rets)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak * 100
    max_dd = float(dd.min())

    # فترة الـ Drawdown
    in_dd = False
    dd_start = dd_end = dd_duration = 0
    max_dd_duration = 0
    for i, d in enumerate(dd):
        if d < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif d >= 0 and in_dd:
            in_dd = False
            duration = i - dd_start
            if duration > max_dd_duration:
                max_dd_duration = duration

    # ── 3. Recovery Factor ────────────────────────────────
    total_return   = float((equity[-1] - 1) * 100)
    recovery_factor = round(total_return / abs(max_dd), 2) if max_dd != 0 else 99

    # ── 4. Consecutive Wins/Losses ────────────────────────
    max_wins = max_losses = cur_wins = cur_losses = 0
    for r in rets:
        if r > 0:
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_wins   = max(max_wins,   cur_wins)
        max_losses = max(max_losses, cur_losses)

    # ── 5. Monthly Breakdown ──────────────────────────────
    monthly = {}
    for t in trade_log:
        month = t["entry_date"][:7]  # YYYY-MM
        if month not in monthly:
            monthly[month] = {"trades": 0, "wins": 0, "total_ret": 0}
        monthly[month]["trades"]    += 1
        monthly[month]["wins"]      += 1 if t["win"] else 0
        monthly[month]["total_ret"] += t["return_pct"]

    monthly_summary = []
    for month, data in sorted(monthly.items()):
        monthly_summary.append({
            "month":    month,
            "trades":   data["trades"],
            "win_rate": round(data["wins"] / data["trades"] * 100, 1),
            "total_ret": round(data["total_ret"], 1)
        })

    # أفضل وأسوأ شهر
    if monthly_summary:
        best_month  = max(monthly_summary, key=lambda x: x["total_ret"])
        worst_month = min(monthly_summary, key=lambda x: x["total_ret"])
        winning_months = sum(1 for m in monthly_summary if m["total_ret"] > 0)
        monthly_wr = round(winning_months / len(monthly_summary) * 100, 1)
    else:
        best_month = worst_month = {}
        monthly_wr = 0

    # ── 6. Yearly Breakdown ───────────────────────────────
    yearly = {}
    for t in trade_log:
        year = t["entry_date"][:4]
        if year not in yearly:
            yearly[year] = {"trades": 0, "wins": 0, "total_ret": 0}
        yearly[year]["trades"]    += 1
        yearly[year]["wins"]      += 1 if t["win"] else 0
        yearly[year]["total_ret"] += t["return_pct"]

    yearly_summary = []
    for year, data in sorted(yearly.items()):
        yearly_summary.append({
            "year":     year,
            "trades":   data["trades"],
            "win_rate": round(data["wins"] / data["trades"] * 100, 1),
            "total_ret": round(data["total_ret"], 1)
        })

    # ── 7. Consistency Score ──────────────────────────────
    # مقياس مركب: كلما كان أعلى كلما كان النظام أكثر اتساقاً
    consistency = round(
        (wr * 30) +
        (min(recovery_factor, 10) / 10 * 25) +
        (monthly_wr / 100 * 25) +
        (min(max_wins, 5) / 5 * 10) +
        (max(0, 10 - max_losses) / 10 * 10),
        1
    )

    return {
        "expectancy":        round(expectancy * 100, 2),
        "total_return":      round(total_return, 1),
        "max_drawdown":      round(max_dd, 1),
        "max_dd_duration":   max_dd_duration,
        "recovery_factor":   recovery_factor,
        "max_win_streak":    max_wins,
        "max_loss_streak":   max_losses,
        "avg_win":           round(avg_win * 100, 2),
        "avg_loss":          round(avg_loss * 100, 2),
        "win_loss_ratio":    round(avg_win / avg_loss, 2) if avg_loss else 99,
        "monthly_win_rate":  monthly_wr,
        "best_month":        best_month,
        "worst_month":       worst_month,
        "monthly_breakdown": monthly_summary,
        "yearly_breakdown":  yearly_summary,
        "consistency_score": consistency,
    }


def print_advanced_stats(stats: dict, symbol: str, best: dict):
    """يطبع تقرير الإحصاء الاحترافي"""
    if not stats:
        return

    print(f"\n  {'═'*55}")
    print(f"  تقرير الأداء الاحترافي — {symbol}")
    print(f"  {'═'*55}")

    # ── الإحصاء الأساسي ───────────────────────────────────
    print(f"\n  الأداء العام:")
    print(f"  {'─'*55}")
    print(f"  إجمالي العائد     : {stats['total_return']:>+8.1f}%")
    print(f"  Expectancy        : {stats['expectancy']:>+8.2f}%  لكل صفقة")
    print(f"  Win Rate          : {best['win_rate']:>8.1f}%")
    print(f"  Profit Factor     : {best['profit_factor']:>8.2f}")
    print(f"  Sharpe Ratio      : {best['sharpe']:>8.2f}")

    # ── المخاطرة ──────────────────────────────────────────
    print(f"\n  إدارة المخاطرة:")
    print(f"  {'─'*55}")
    print(f"  Max Drawdown      : {stats['max_drawdown']:>8.1f}%")
    print(f"  مدة الـ Drawdown  : {stats['max_dd_duration']:>8} صفقة")
    print(f"  Recovery Factor   : {stats['recovery_factor']:>8.2f}")
    print(f"  متوسط الربح       : {stats['avg_win']:>+8.2f}%")
    print(f"  متوسط الخسارة     : {stats['avg_loss']:>-8.2f}%")
    print(f"  نسبة ربح/خسارة   : {stats['win_loss_ratio']:>8.2f}x")

    # ── الاتساق ───────────────────────────────────────────
    print(f"\n  الاتساق:")
    print(f"  {'─'*55}")
    print(f"  أطول سلسلة ربح   : {stats['max_win_streak']:>8} صفقات متتالية")
    print(f"  أطول سلسلة خسارة : {stats['max_loss_streak']:>8} صفقات متتالية")
    print(f"  الأشهر الرابحة    : {stats['monthly_win_rate']:>8.1f}%")
    print(f"  Consistency Score : {stats['consistency_score']:>8.1f} / 100")

    # ── الأداء السنوي ─────────────────────────────────────
    if stats.get("yearly_breakdown"):
        print(f"\n  الأداء السنوي:")
        print(f"  {'─'*55}")
        print(f"  {'السنة':<8}  {'صفقات':>6}  {'WR%':>6}  {'العائد الكلي':>12}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*12}")
        for y in stats["yearly_breakdown"]:
            trend = "↑" if y["total_ret"] > 0 else "↓"
            print(f"  {y['year']:<8}  {y['trades']:>6}  {y['win_rate']:>6.1f}%  {y['total_ret']:>+11.1f}%  {trend}")

    # ── أفضل وأسوأ شهر ────────────────────────────────────
    if stats.get("best_month") and stats.get("worst_month"):
        bm = stats["best_month"]
        wm = stats["worst_month"]
        print(f"\n  أفضل شهر  : {bm.get('month','?')}  ({bm.get('total_ret',0):+.1f}%  |  {bm.get('trades',0)} صفقات)")
        print(f"  أسوأ شهر  : {wm.get('month','?')}  ({wm.get('total_ret',0):+.1f}%  |  {wm.get('trades',0)} صفقات)")

    # ── الحكم النهائي ─────────────────────────────────────
    score = stats["consistency_score"]
    if score >= 75:
        verdict = "ممتاز — نظام قابل للتطبيق الفعلي"
        icon = "✅"
    elif score >= 60:
        verdict = "جيد — يحتاج تحسيناً طفيفاً"
        icon = "✓"
    elif score >= 45:
        verdict = "مقبول — يحتاج تطويراً"
        icon = "⚠"
    else:
        verdict = "ضعيف — يحتاج مراجعة جذرية"
        icon = "✗"

    print(f"\n  {'═'*55}")
    print(f"  {icon} الحكم النهائي: {verdict}")
    print(f"  {'═'*55}\n")

def run_backtest(symbol: str, threshold: float = 72, hold_days: int = 10,
                 period: str = "5y",
                 profit_target: float = 0.15,
                 stop_loss: float = 0.07,
                 max_days: int = 20,
                 smart_exit: bool = True,
                 use_squeeze_exit: bool = True,
                 use_regime: bool = True) -> dict:
    """
    يشغّل Backtest كامل على CES v5
    smart_exit=True: خروج ذكي (هدف + وقف + مهلة + CES)
    smart_exit=False: خروج ثابت بعد hold_days (النمط القديم)
    """
    print(f"\n{'═'*52}")
    print(f"  Backtest — CES v5 على {symbol}")
    if smart_exit:
        print(f"  الخروج: ذكي | هدف +{profit_target*100:.0f}% | وقف -{stop_loss*100:.0f}% | مهلة: 7/14/21y حسب CES")
    else:
        print(f"  الخروج: ثابت {hold_days} يوم")
    print(f"  الفترة: {period} | Hold: {hold_days} يوم")
    print(f"{'═'*52}\n")

    # جلب البيانات
    print(f"  جارٍ تحميل البيانات...", end=" ", flush=True)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    print(f"✓ {len(df)} يوم ({df.index[0].date()} → {df.index[-1].date()})")

    if len(df) < 300:
        print("  بيانات غير كافية")
        return {}

    # جلب SPY و VIX للفلتر
    print(f"  جارٍ جلب SPY و VIX...", end=" ", flush=True)
    try:
        spy_df  = yf.Ticker("SPY").history(period=period, interval="1d", auto_adjust=True)
        spy_df  = spy_df[["Close"]].rename(columns={"Close": "SPY"})
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        spy_ma200 = spy_df["SPY"].rolling(200).mean()

        vix_df = yf.Ticker("^VIX").history(period=period, interval="1d", auto_adjust=True)
        vix_df = vix_df[["Close"]].rename(columns={"Close": "VIX"})
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        # محاذاة الفهارس
        spy_aligned  = spy_df["SPY"].reindex(df.index, method="ffill")
        spy_ma_aligned = spy_ma200.reindex(df.index, method="ffill")
        vix_aligned  = vix_df["VIX"].reindex(df.index, method="ffill")
        regime_ok = True
        print("✓")
    except Exception as e:
        print(f"تعذّر ({e}) — بدون فلتر")
        spy_aligned = spy_ma_aligned = vix_aligned = None
        regime_ok = False

    # حساب CES
    print(f"  حساب CES على كل الشموع...", end=" ", flush=True)
    ces = compute_ces_series_backtest(df)
    print("✓")

    close = df["Close"]

    # فلتر Market Regime لكل يوم
    # SPY فوق MA200 + VIX تحت 30 = بيئة صاعدة
    if regime_ok and spy_aligned is not None and vix_aligned is not None:
        spy_ok        = spy_aligned > spy_ma_aligned
        vix_ok_series = vix_aligned < 30
        regime_filter = spy_ok & vix_ok_series
        filtered_days = regime_filter.sum()
        total_days    = len(regime_filter)
        print(f"  Regime Filter: {filtered_days}/{total_days} يوم صالح ({filtered_days/total_days*100:.0f}%)")
    else:
        regime_filter = pd.Series(True, index=df.index)
        print("  Regime Filter: غير مفعل — كل الأيام صالحة")

    # اختبار عدة عتبات
    # العتبة المثلى من الـ profile + عتبات مجاورة للمقارنة
    profile_thr = PROFILE.get("threshold", 72)
    thr_range   = sorted(set([
        max(55, profile_thr - 10),
        max(55, profile_thr - 7),
        max(55, profile_thr - 5),
        max(55, profile_thr - 3),
        profile_thr,
        min(90, profile_thr + 3),
        min(90, profile_thr + 5),
        min(90, profile_thr + 7),
        min(90, profile_thr + 10),
    ]))
    thresholds = thr_range
    thr_results = []

    # اختبار مع وبدون فلتر Market Regime
    for use_regime in [False, True]:
        label = "مع فلتر Regime" if use_regime else "بدون فلتر"

        for thr in thresholds:
            # تطبيق فلتر Regime إذا كان مفعلاً
            if regime_ok and use_regime:
                regime_mask = regime_filter.reindex(ces.index, fill_value=False)
                signal_mask = (ces >= thr) & (ces.shift(1) < thr) & regime_mask
            else:
                signal_mask = (ces >= thr) & (ces.shift(1) < thr)

            entries = ces.index[signal_mask]
            # debug: طباعة عدد الإشارات
            # print(f"    thr={thr} regime={use_regime} signals={len(entries)}")
            rets = []
            trade_log = []

            for entry in entries:
                try:
                    i = close.index.get_loc(entry)
                    if i >= len(close) - 2:
                        continue
                    ep = float(close.iloc[i])

                    if smart_exit:
                        # ── حساب Squeeze Momentum ─────────────────────
                        # TTM Squeeze: Bollinger Bands داخل Keltner Channel
                        # Momentum = Linear Regression على الفرق
                        def calc_squeeze_mom(prices, idx_start, length=20):
                            """يحسب Squeeze Momentum لنقطة معينة"""
                            if idx_start < length:
                                return 0.0, 0.0
                            sl = prices.iloc[idx_start-length:idx_start+1]
                            if len(sl) < length:
                                return 0.0, 0.0
                            # Bollinger Bands
                            bb_mid = sl.mean()
                            bb_std = sl.std()
                            bb_upper = bb_mid + 2 * bb_std
                            bb_lower = bb_mid - 2 * bb_std
                            # Keltner Channel
                            kc_range = sl.max() - sl.min()
                            kc_upper = bb_mid + 1.5 * kc_range / length
                            kc_lower = bb_mid - 1.5 * kc_range / length
                            # Squeeze: BB داخل KC
                            in_squeeze = bb_upper <= kc_upper and bb_lower >= kc_lower
                            # Momentum
                            mid_val = (sl.max() + sl.min()) / 2
                            delta   = sl.iloc[-1] - mid_val
                            return float(delta), float(delta - (sl.iloc[-2] - mid_val) if len(sl) > 1 else 0)

                        # ── الخروج الذكي ──────────────────────────────
                        ces_at_entry = float(ces.iloc[i])
                        if ces_at_entry >= 88:
                            dyn_days = 7
                        elif ces_at_entry >= 80:
                            dyn_days = 14
                        else:
                            dyn_days = 21

                        exit_price  = None
                        exit_date   = None
                        exit_reason = "مهلة"
                        limit       = min(i + dyn_days, len(close) - 1)
                        peak_price  = ep   # نتتبع أعلى سعر للخروج الذكي

                        for k in range(i + 1, limit + 1):
                            day_price = float(close.iloc[k])
                            day_ret   = (day_price - ep) / ep
                            peak_price = max(peak_price, day_price)

                            # 1. هدف الربح
                            if day_ret >= profit_target:
                                exit_price  = day_price
                                exit_date   = close.index[k]
                                exit_reason = f"هدف +{profit_target*100:.0f}%"
                                break

                            # 2. وقف الخسارة
                            if day_ret <= -stop_loss:
                                exit_price  = day_price
                                exit_date   = close.index[k]
                                exit_reason = f"وقف -{stop_loss*100:.0f}%"
                                break

                            # 3. Squeeze Momentum — اختياري حسب profile السهم
                            if use_squeeze_exit and k > i + 3 and day_ret > 0.05:
                                mom_now, mom_delta = calc_squeeze_mom(close, k)
                                mom_prev, _ = calc_squeeze_mom(close, k-1)
                                if mom_now < 0 and mom_prev >= 0:
                                    exit_price  = day_price
                                    exit_date   = close.index[k]
                                    exit_reason = "Squeeze انعكس"
                                    break
                                rsi_k = float(ces.iloc[k]) if k < len(ces) else 50
                                if mom_delta < -0.5 and day_ret > 0.08:
                                    exit_price  = day_price
                                    exit_date   = close.index[k]
                                    exit_reason = "Squeeze تباطأ"
                                    break

                            # 4. Trailing Stop بعد ربح جيد — احمِ 60% من المكسب
                            if day_ret > 0.10:
                                trail_stop = (peak_price - ep) / ep * 0.4 + ep
                                if day_price < trail_stop:
                                    exit_price  = day_price
                                    exit_date   = close.index[k]
                                    exit_reason = "Trailing Stop"
                                    break

                        if exit_price is None:
                            j           = min(i + dyn_days, len(close) - 1)
                            exit_price  = float(close.iloc[j])
                            exit_date   = close.index[j]
                            exit_reason = f"مهلة {dyn_days}y"

                        xp  = exit_price
                        ret = (xp - ep) / ep

                    else:
                        # ── الخروج الثابت (القديم) ─────────────────────
                        j   = min(i + hold_days, len(close) - 1)
                        if j <= i:
                            continue
                        xp          = float(close.iloc[j])
                        exit_date   = close.index[j]
                        exit_reason = f"ثابت {hold_days}y"
                        ret         = (xp - ep) / ep

                    rets.append(ret)

                    spy_val    = float(spy_aligned.iloc[i])    if regime_ok else 0
                    vix_val    = float(vix_aligned.iloc[i])    if regime_ok else 0
                    spy_ma_val = float(spy_ma_aligned.iloc[i]) if regime_ok else 0
                    market_state = "صاعد" if (spy_val > spy_ma_val and vix_val < 30) else "هابط/متقلب"

                    trade_log.append({
                        "entry_date":   str(entry.date()),
                        "entry_price":  round(ep, 2),
                        "exit_price":   round(xp, 2),
                        "exit_reason":  exit_reason,
                        "return_pct":   round(ret * 100, 2),
                        "win":          ret > 0,
                        "market_state": market_state,
                        "vix":          round(vix_val, 1),
                        "days_held":    int((exit_date - entry).days),
                    })
                except Exception:
                    continue

            if len(rets) < 5:
                continue

            a    = np.array(rets)
            wins = a[a > 0]
            loss = a[a <= 0]
            wr   = len(wins) / len(a)
            aw   = wins.mean() if len(wins) else 0
            al   = abs(loss.mean()) if len(loss) else 1e-9
            pf   = (aw * len(wins)) / (al * len(loss) + 1e-9) if len(loss) else 99

            thr_results.append({
                "threshold":     thr,
                "trades":        len(a),
                "win_rate":      round(wr * 100, 1),
                "profit_factor": round(min(pf, 99), 2),
                "avg_return":    round(a.mean() * 100, 2),
                "best_trade":    round(a.max() * 100, 2),
                "worst_trade":   round(a.min() * 100, 2),
                "sharpe":        round(a.mean() / (a.std() + 1e-9) * np.sqrt(252 / hold_days), 2),
                "score":         round(wr * 0.5 + min(pf, 5) / 5 * 0.5, 4),
                "trade_log":     trade_log,
                "regime_filter": use_regime,
            })

    if not thr_results:
        print("  لا توجد صفقات كافية")
        return {}

    # أفضل عتبة
    best = max(thr_results, key=lambda x: x["score"])
    best_with_regime    = [r for r in thr_results if r.get("regime_filter")]
    best_without_regime = [r for r in thr_results if not r.get("regime_filter")]

    best_no  = max(best_without_regime, key=lambda x: x["score"]) if best_without_regime else best
    best_yes = max(best_with_regime,    key=lambda x: x["score"]) if best_with_regime    else best

    # ── طباعة المقارنة ────────────────────────────────────
    for label, subset, best_r in [
        ("بدون فلتر Regime", best_without_regime, best_no),
        ("مع فلتر Regime (SPY>MA200 + VIX<30)", best_with_regime, best_yes)
    ]:
        if not subset:
            continue
        print(f"\n  {'─'*55}")
        print(f"  {label}")
        print(f"  {'─'*55}")
        print(f"  {'Thr':>4}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'AvgRet':>7}  {'Sharpe':>7}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
        for r in subset:
            marker = " ◄ أفضل" if r["threshold"] == best_r["threshold"] else ""
            print(f"  {r['threshold']:>4}  {r['trades']:>6}  {r['win_rate']:>6}%  "
                  f"{r['profit_factor']:>6}  {r['avg_return']:>6}%  {r['sharpe']:>7}{marker}")

    # المقارنة المباشرة
    print(f"\n  {'═'*55}")
    print(f"  مقارنة مباشرة — أفضل عتبة في كل حالة")
    print(f"  {'═'*55}")
    print(f"  {'':30}  {'بدون':>8}  {'مع فلتر':>8}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}")
    print(f"  {'العتبة المثلى':<30}  {best_no['threshold']:>8}  {best_yes['threshold']:>8}")
    print(f"  {'عدد الصفقات':<30}  {best_no['trades']:>8}  {best_yes['trades']:>8}")
    print(f"  {'Win Rate':<30}  {best_no['win_rate']:>7}%  {best_yes['win_rate']:>7}%")
    print(f"  {'Profit Factor':<30}  {best_no['profit_factor']:>8}  {best_yes['profit_factor']:>8}")
    print(f"  {'متوسط العائد':<30}  {best_no['avg_return']:>7}%  {best_yes['avg_return']:>7}%")
    print(f"  {'Sharpe':<30}  {best_no['sharpe']:>8}  {best_yes['sharpe']:>8}")
    improvement_wr = best_yes['win_rate'] - best_no['win_rate']
    improvement_pf = best_yes['profit_factor'] - best_no['profit_factor']
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}")
    print(f"  {'التحسن في WR':<30}  {'':>8}  {improvement_wr:>+7.1f}%")
    print(f"  {'التحسن في PF':<30}  {'':>8}  {improvement_pf:>+8.2f}")
    print(f"  {'═'*55}")

    best = best_yes if best_yes["score"] > best_no["score"] else best_no

    # ── تفاصيل أفضل عتبة ──────────────────────────────────
    print(f"\n  {'═'*52}")
    print(f"  أفضل عتبة: {best['threshold']}")
    print(f"  {'═'*52}")
    print(f"  عدد الصفقات  : {best['trades']}")
    print(f"  Win Rate     : {best['win_rate']}%")
    print(f"  Profit Factor: {best['profit_factor']}")
    print(f"  متوسط العائد : {best['avg_return']}%")
    print(f"  أفضل صفقة   : +{best['best_trade']}%")
    print(f"  أسوأ صفقة   : {best['worst_trade']}%")
    print(f"  Sharpe       : {best['sharpe']}")

    # آخر 10 صفقات
    print(f"\n  آخر 10 صفقات:")
    print(f"  {'─'*52}")
    print(f"  {'التاريخ':<12}  {'دخول':>8}  {'خروج':>8}  {'العائد':>8}")
    print(f"  {'#':<4}  {'التاريخ':<12}  {'دخول':>8}  {'خروج':>8}  {'العائد':>8}  {'أيام':>5}  سبب الخروج")
    print(f"  {'─'*65}")
    for idx2, t in enumerate(best["trade_log"][-10:], 1):
        icon   = "✓" if t["win"] else "✗"
        reason = t.get("exit_reason", "-")
        days   = t.get("days_held", "-")
        print(f"  {icon} {t['entry_date']:<12}  ${t['entry_price']:>7}  ${t['exit_price']:>7}  {t['return_pct']:>+7}%  {str(days):>5}  {reason}")

    print(f"  {'═'*52}\n")

    # الإحصاء الاحترافي على أفضل عتبة
    stats = compute_advanced_stats(best["trade_log"], hold_days)
    print_advanced_stats(stats, symbol, best)

    # حفظ
    output = {
        "symbol":       symbol,
        "period":       period,
        "hold_days":    hold_days,
        "all_thresholds": thr_results,
        "best":         best,
        "advanced_stats": stats,
        "timestamp":    datetime.datetime.now().isoformat(),
    }
    fname = f"ces_v5_backtest_{symbol}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"  محفوظ في: {fname}\n")

    return output


# ══════════════════════════════════════════════════════════════
# نقطة الدخول
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # الوضع: backtest أو تحليل حي
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "live"

    if mode == "backtest":
        # python ces_v5.py MU backtest
        # python ces_v5.py MU backtest 10
        hold   = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        smart  = "--fixed" not in sys.argv
        p      = PROFILE

        print(f"\n  السهم    : {p['symbol']}")
        print(f"  النمط    : {p['style']}")
        print(f"  العتبة   : {p['threshold']}")
        print(f"  Squeeze  : {'مفعل' if p['use_squeeze'] else 'معطل'}")
        print(f"  Regime   : {'مفعل' if p['regime_filter'] else 'معطل'}")
        print(f"  الوصف    : {p['description']}")

        result_bt = run_backtest(
            SYMBOL,
            hold_days=hold,
            smart_exit=smart,
            profit_target=p["profit_target"],
            stop_loss=p["stop_loss"],
            use_squeeze_exit=p["use_squeeze"],
            use_regime=p["regime_filter"],
        )
        if "--trades" in sys.argv and result_bt.get("best"):
            trades = result_bt["best"]["trade_log"]
            thr    = result_bt["best"]["threshold"]
            print(f"\n  {'='*55}")
            print(f"  كل الصفقات ({len(trades)}) — عتبة {thr} | Hold {hold} يوم")
            print(f"  {'='*55}")
            print(f"  {'#':<4}  {'تاريخ الدخول':<13}  {'دخول':>8}  {'خروج':>8}  {'العائد':>8}  نتيجة")
            print(f"  {'-'*55}")
            for i, t in enumerate(trades, 1):
                icon = "✓ ربح   " if t["win"] else "✗ خسارة"
                print(f"  {i:<4}  {t['entry_date']:<13}  ${t['entry_price']:>7}  ${t['exit_price']:>7}  {t['return_pct']:>+7}%  {icon}")
            print(f"  {'='*55}\n")
    else:
        # python ces_v5.py MU
        result = compute_ces_v5(SYMBOL)
        print_report(result)
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"  محفوظ في: {OUTPUT}\n")