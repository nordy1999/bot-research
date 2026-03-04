"""
AlmostFinishedBot - Feature Engineering Module
Standalone feature builder. Safe to import -- no training, no downloads, no side effects.
Used by train_models.py and walkforward_backtest.py
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")


# == Indicators (pure numpy, no index issues) ===============================

def ema_np(arr, span):
    out = np.full(len(arr), np.nan); k = 2.0 / (span + 1)
    start = 0
    while start < len(arr) and np.isnan(arr[start]): start += 1
    if start >= len(arr): return out
    out[start] = arr[start]
    for i in range(start + 1, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out

def rsi_np(close, period=14):
    delta = np.diff(close, prepend=np.nan)
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = ema_np(gain, period); avg_l = ema_np(loss, period)
    rs    = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-12))
    return 100.0 - 100.0 / (1.0 + rs)

def atr_np(high, low, close, period=14):
    tr = np.maximum(high[1:]-low[1:],
         np.maximum(np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
    return ema_np(np.concatenate([[np.nan], tr]), period)

def adx_np(high, low, close, period=14):
    h,l,c = high.astype(float), low.astype(float), close.astype(float)
    tr  = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr = ema_np(np.concatenate([[np.nan], tr]), period)
    up  = np.concatenate([[np.nan], h[1:]-h[:-1]])
    dn  = np.concatenate([[np.nan], l[:-1]-l[1:]])
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    ndm = np.where((dn>up)&(dn>0), dn, 0.0)
    safe_atr = np.where(atr==0, 1e-9, atr)
    pdi = 100.0 * ema_np(pdm, period) / safe_atr
    ndi = 100.0 * ema_np(ndm, period) / safe_atr
    denom = np.where(pdi+ndi==0, 1e-9, pdi+ndi)
    return ema_np(100.0 * np.abs(pdi-ndi) / denom, period)

def bbands_np(close, period=20):
    mid = np.full(len(close), np.nan)
    wid = np.full(len(close), np.nan)
    pos = np.full(len(close), np.nan)
    for i in range(period-1, len(close)):
        sl = close[i-period+1:i+1]; m = sl.mean(); s = sl.std()
        mid[i] = m
        wid[i] = (4*s) / (m if m != 0 else 1e-9)
        pos[i] = (close[i]-m) / (2*s+1e-9)
    return mid, wid, pos

def stoch_np(high, low, close, k_period=14, d_period=3):
    n = len(close)
    k = np.full(n, np.nan)
    for i in range(k_period-1, n):
        hh = np.max(high[i-k_period+1:i+1])
        ll = np.min(low[i-k_period+1:i+1])
        k[i] = (close[i] - ll) / (hh - ll + 1e-9)
    d = ema_np(k, d_period)
    return k, d


# == Feature Engineering ====================================================

def make_features(df):
    """Build feature DataFrame from OHLCV DataFrame. Pure function, no side effects."""
    n     = len(df)
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    vol   = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(n)
    cols  = {}

    # Returns at multiple horizons
    for p in [1,2,3,5,10,20]:
        r = np.full(n, np.nan); r[p:] = (close[p:]-close[:-p])/(close[:-p]+1e-9)
        cols[f"ret_{p}"] = r

    # EMA distances
    for span in [9,21,50,100]:
        e = ema_np(close, span); cols[f"ema_{span}_dist"] = (close-e)/(close+1e-9)

    # EMA crossovers
    e9=ema_np(close,9); e21=ema_np(close,21); e50=ema_np(close,50)
    cols["ema_cross_9_21"] = (e9-e21)/(close+1e-9)
    cols["ema_cross_21_50"] = (e21-e50)/(close+1e-9)

    # RSI
    rsi = rsi_np(close,14)
    cols["rsi"] = rsi/100.0
    cols["rsi_extreme"] = np.where(rsi>70, (rsi-70)/30, np.where(rsi<30, (rsi-30)/30, 0.0))

    # MACD
    m12=ema_np(close,12); m26=ema_np(close,26); macd=m12-m26; sig=ema_np(macd,9)
    cols["macd"]      = macd/(close+1e-9)
    cols["macd_hist"] = (macd-sig)/(close+1e-9)

    # ATR
    atr_vals = atr_np(high,low,close,14)
    cols["atr_pct"] = atr_vals/(close+1e-9)

    # Bollinger Bands
    _, bbw, bbpos = bbands_np(close,20)
    cols["bb_width"] = bbw; cols["bb_pos"] = bbpos

    # ADX
    cols["adx"] = adx_np(high,low,close,14)/100.0

    # Stochastic
    stoch_k, stoch_d = stoch_np(high, low, close)
    cols["stoch_k"] = stoch_k
    cols["stoch_kd_cross"] = stoch_k - stoch_d

    # Volume features
    vol_ma = np.full(n, np.nan)
    for i in range(19,n): vol_ma[i] = vol[i-19:i+1].mean()
    cols["vol_ratio"] = vol/(vol_ma+1e-9)

    # Candle anatomy
    hl = high-low+1e-9
    cols["body_pct"]   = np.abs(close-open_)/hl
    cols["upper_wick"] = (high-np.maximum(close,open_))/hl
    cols["lower_wick"] = (np.minimum(close,open_)-low)/hl
    cols["candle_dir"]  = np.sign(close - open_)

    # Momentum
    mom=np.full(n,np.nan); mom[10:]=close[10:]-close[:-10]; cols["mom_10"]=mom/(close+1e-9)
    roc=np.full(n,np.nan); roc[5:]=(close[5:]-close[:-5])/(close[:-5]+1e-9); cols["roc_5"]=roc

    # Volatility regime: rolling std of returns
    ret1 = np.full(n, np.nan); ret1[1:] = (close[1:]-close[:-1])/(close[:-1]+1e-9)
    vol_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_20[i] = np.std(ret1[i-20:i])
    cols["vol_regime"] = vol_20

    # Position in range
    for p in [20]:
        rolling_hi = np.full(n, np.nan)
        rolling_lo = np.full(n, np.nan)
        for i in range(p-1, n):
            rolling_hi[i] = np.max(high[i-p+1:i+1])
            rolling_lo[i] = np.min(low[i-p+1:i+1])
        rng = rolling_hi - rolling_lo + 1e-9
        cols[f"pos_in_range_{p}"] = (close - rolling_lo) / rng

    # Hour of day (gold has strong session effects)
    if hasattr(df.index, 'hour'):
        hours = df.index.hour.values.astype(float)
    else:
        hours = np.zeros(n)
    cols["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    cols["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Day of week (cyclical)
    if hasattr(df.index, 'dayofweek'):
        dow = df.index.dayofweek.values.astype(float)
    else:
        dow = np.zeros(n)
    cols["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    cols["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

    # Session encoding
    session = np.zeros(n)
    for i in range(n):
        h = int(hours[i])
        if 0 <= h < 7: session[i] = 0
        elif 7 <= h < 12: session[i] = 1
        elif 12 <= h < 16: session[i] = 2
        elif 16 <= h < 21: session[i] = 3
        else: session[i] = -1
    cols["session"] = session / 3.0

    # Regime from market_regime.py (graceful if not available)
    try:
        sys.path.insert(0, BASE)
        from market_regime import detect_regime
        regs, _adx, _bbw, _hurst = detect_regime(df)
        rmap = {"TRENDING_STRONG":3,"TRENDING":2,"RANGING":1,"HIGH_VOL":4,"LOW_VOL":0,
                "TREND_UP":4,"TREND_DOWN":3,"HIGH_VOL_BREAKOUT":2,
                "LOW_VOL_COMPRESSION":1,"CHOPPY_RANGE":0,"UNKNOWN":1}
        cols["regime"] = np.array([rmap.get(r,1) for r in regs], float)/4.0
    except Exception:
        cols["regime"] = np.ones(n)*0.5

    # News sentiment score
    try:
        with open(os.path.join(BASE,"news_cache.json")) as f: nd=json.load(f)
        cols["news_score"] = np.full(n, float(nd.get("score",0)))
    except Exception:
        cols["news_score"] = np.zeros(n)

    return pd.DataFrame(cols, index=df.index)


def make_target(close_arr, horizon=5, threshold=0.001):
    """
    Binary target: 1 if price rises > threshold over horizon bars, else 0.
    """
    n = len(close_arr); tgt = np.full(n, np.nan)
    for i in range(n - horizon):
        ret = (close_arr[i+horizon] - close_arr[i]) / (close_arr[i] + 1e-9)
        tgt[i] = 1.0 if ret > threshold else 0.0
    return tgt


def make_regime_labels(df):
    """Get regime string per row. Returns list of strings same length as df."""
    try:
        sys.path.insert(0, BASE)
        from market_regime import detect_regime
        regs, _, _, _ = detect_regime(df)
        return regs
    except Exception:
        return ["UNKNOWN"] * len(df)


def kelly_fraction(win_rate, avg_win, avg_loss):
    if avg_loss <= 0: return 0.01
    b = avg_win / avg_loss
    k = (b * win_rate - (1 - win_rate)) / b
    return max(0.005, min(k * 0.5, 0.05))


def download_gold(period="6mo", interval="1h"):
    """Download gold data and clean columns. Returns clean DataFrame."""
    import yfinance as yf
    df = yf.download("GC=F", period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or len(df) < 200:
        df = yf.download("GC=F", period="60d", interval="30m", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.dropna(subset=["open","high","low","close"])
    return df
