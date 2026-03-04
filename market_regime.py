"""
AlmostFinishedBot - Enhanced Market Regime Detector v2
5 Regimes: TREND_UP / TREND_DOWN / CHOPPY_RANGE / HIGH_VOL_BREAKOUT / LOW_VOL_COMPRESSION
ADX + Bollinger Band Width + Hurst Exponent + Volume Profile
"""
import numpy as np, pandas as pd, os, json, time

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")

def ema_np(arr, span):
    out = np.full(len(arr), np.nan); k = 2.0 / (span + 1)
    start = next((i for i, v in enumerate(arr) if not np.isnan(v)), len(arr))
    if start >= len(arr): return out
    out[start] = arr[start]
    for i in range(start + 1, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out

def adx_np(high, low, close, period=14):
    h,l,c = high.astype(float), low.astype(float), close.astype(float)
    tr  = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr = ema_np(np.concatenate([[np.nan], tr]), period)
    up  = np.concatenate([[np.nan], h[1:]-h[:-1]])
    dn  = np.concatenate([[np.nan], l[:-1]-l[1:]])
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    ndm = np.where((dn>up)&(dn>0), dn, 0.0)
    safe = np.where(atr==0, 1e-9, atr)
    pdi  = 100.0 * ema_np(pdm, period) / safe
    ndi  = 100.0 * ema_np(ndm, period) / safe
    denom= np.where(pdi+ndi==0, 1e-9, pdi+ndi)
    return ema_np(100.0 * np.abs(pdi-ndi)/denom, period), pdi, ndi

def bb_width_np(close, period=20):
    wid = np.full(len(close), np.nan)
    for i in range(period-1, len(close)):
        sl = close[i-period+1:i+1]; m = sl.mean(); s = sl.std()
        wid[i] = (4*s) / (m if m != 0 else 1e-9)
    return wid

def hurst_exponent(series, min_lag=2, max_lag=20):
    if len(series) < max_lag * 2: return 0.5
    lags = range(min_lag, min(max_lag, len(series)//2))
    tau  = []
    for lag in lags:
        diffs = np.subtract(series[lag:], series[:-lag])
        tau.append(np.std(diffs) if len(diffs) > 0 and np.std(diffs) > 0 else 1e-9)
    try:
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(poly[0])
    except: return 0.5

def volume_profile_bias(close, volume, period=50):
    if len(close) < period: return 0
    c=close[-period:]; v=volume[-period:]; mid=(c.max()+c.min())/2
    va=v[c>=mid].sum(); vb=v[c<mid].sum(); tot=va+vb
    if tot==0: return 0
    r=(va-vb)/tot
    return 1 if r>0.15 else (-1 if r<-0.15 else 0)

def detect_regime(df, adx_trend=25, adx_strong=40, bb_high=0.04, bb_low=0.012):
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(df))
    adx_arr, pdi_arr, ndi_arr = adx_np(high, low, close, 14)
    bbw_arr = bb_width_np(close, 20)
    n = len(df)
    hurst_arr = np.full(n, np.nan)
    for i in range(60, n):
        hurst_arr[i] = hurst_exponent(close[i-60:i])
    regimes = []
    for i in range(n):
        a=adx_arr[i]; b=bbw_arr[i]; pdi=pdi_arr[i]; ndi=ndi_arr[i]
        h=hurst_arr[i] if not np.isnan(hurst_arr[i]) else 0.5
        if np.isnan(a) or np.isnan(b): regimes.append("UNKNOWN")
        elif b <= bb_low and a < adx_trend: regimes.append("LOW_VOL_COMPRESSION")
        elif b >= bb_high and a < adx_strong: regimes.append("HIGH_VOL_BREAKOUT")
        elif a >= adx_trend and pdi > ndi and h >= 0.50: regimes.append("TREND_UP")
        elif a >= adx_trend and ndi > pdi and h >= 0.50: regimes.append("TREND_DOWN")
        else: regimes.append("CHOPPY_RANGE")
    return regimes, adx_arr, bbw_arr, hurst_arr

def regime_to_code(r):
    return {"TREND_UP":4,"TREND_DOWN":3,"HIGH_VOL_BREAKOUT":2,"LOW_VOL_COMPRESSION":1,"CHOPPY_RANGE":0,"UNKNOWN":0}.get(r,0)

def regime_allows_long(r): return r in ("TREND_UP","HIGH_VOL_BREAKOUT","LOW_VOL_COMPRESSION")
def regime_allows_short(r): return r in ("TREND_DOWN","HIGH_VOL_BREAKOUT","LOW_VOL_COMPRESSION")
def regime_risk_multiplier(r):
    return {"TREND_UP":1.0,"TREND_DOWN":1.0,"HIGH_VOL_BREAKOUT":0.5,
            "LOW_VOL_COMPRESSION":0.75,"CHOPPY_RANGE":0.25,"UNKNOWN":0.1}.get(r,0.25)

def get_current_regime():
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="5d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns=[c[0].lower() for c in df.columns]
        else: df.columns=[str(c).lower() for c in df.columns]
        if hasattr(df.index,"tz") and df.index.tz: df.index=df.index.tz_convert("UTC").tz_localize(None)
        df = df.dropna(subset=["open","high","low","close"])
        regs,adxv,bbwv,hurstv = detect_regime(df)
        latest = regs[-1]
        vb = volume_profile_bias(df["close"].values,
             df["volume"].values if "volume" in df.columns else np.ones(len(df)))
        cache = {"regime":latest,
            "adx":float(adxv[-1]) if not np.isnan(adxv[-1]) else 0,
            "bbw":float(bbwv[-1]) if not np.isnan(bbwv[-1]) else 0,
            "hurst":float(hurstv[-1]) if not np.isnan(hurstv[-1]) else 0.5,
            "volume_bias":int(vb),"risk_mult":regime_risk_multiplier(latest),
            "allows_long":regime_allows_long(latest),"allows_short":regime_allows_short(latest),
            "timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
        os.makedirs(BASE,exist_ok=True)
        with open(os.path.join(BASE,"regime_cache.json"),"w") as f: json.dump(cache,f,indent=2)
        return cache
    except Exception as e: return {"regime":"UNKNOWN","error":str(e)}

if __name__ == "__main__":
    r=get_current_regime()
    print(f"\n  Regime  : {r.get('regime')}")
    print(f"  ADX     : {r.get('adx',0):.1f}")
    print(f"  BBW     : {r.get('bbw',0):.4f}")
    print(f"  Hurst   : {r.get('hurst',0):.3f}")
    print(f"  Vol bias: {r.get('volume_bias',0):+d}")
    print(f"  Risk x  : {r.get('risk_mult',0):.2f}")
    print(f"  Long OK : {r.get('allows_long')}")
    print(f"  Short OK: {r.get('allows_short')}")
