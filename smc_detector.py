"""
AlmostFinishedBot - SMC Detector v2
Order Blocks, FVGs, Liquidity Sweeps, Breaker Blocks — 1H + 15M confluence
"""
import numpy as np, pandas as pd, os, json, time, sys

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")

def find_swing_points(high, low, lookback=5):
    n=len(high); sh=[]; sl=[]
    for i in range(lookback, n-lookback):
        if all(high[i]>=high[i-j] for j in range(1,lookback+1)) and all(high[i]>=high[i+j] for j in range(1,lookback+1)): sh.append((i,high[i]))
        if all(low[i]<=low[i-j] for j in range(1,lookback+1)) and all(low[i]<=low[i+j] for j in range(1,lookback+1)): sl.append((i,low[i]))
    return sh, sl

def find_order_blocks(open_, high, low, close, lookback=3):
    n=len(close); bull=[]; bear=[]
    atr=[0]*14
    for i in range(14,n): atr.append(np.mean([abs(high[j]-low[j]) for j in range(i-14,i)]))
    for i in range(lookback+1, n-lookback):
        if atr[i]==0: continue
        fh=max(high[i:i+lookback]); fl=min(low[i:i+lookback])
        mu=fh-close[i]; md=close[i]-fl
        if close[i]<open_[i] and mu>atr[i]*1.5: bull.append({"index":i,"top":high[i],"bottom":low[i],"mid":(high[i]+low[i])/2,"type":"BULL_OB"})
        if close[i]>open_[i] and md>atr[i]*1.5: bear.append({"index":i,"top":high[i],"bottom":low[i],"mid":(high[i]+low[i])/2,"type":"BEAR_OB"})
    return bull[-5:], bear[-5:]

def find_fvg(high, low, close, min_gap_pct=0.001):
    n=len(close); bf=[]; bf2=[]
    for i in range(1,n-1):
        gb=low[i+1]-high[i-1]; gbe=low[i-1]-high[i+1]
        if gb>close[i]*min_gap_pct: bf.append({"index":i,"top":low[i+1],"bottom":high[i-1],"mid":(low[i+1]+high[i-1])/2,"type":"BULL_FVG"})
        if gbe>close[i]*min_gap_pct: bf2.append({"index":i,"top":low[i-1],"bottom":high[i+1],"mid":(low[i-1]+high[i+1])/2,"type":"BEAR_FVG"})
    return bf[-5:], bf2[-5:]

def find_liquidity_sweeps(high, low, close, swing_highs, swing_lows, lookback=3):
    n=len(close); bs=[]; bes=[]
    rh=[p for (_,p) in swing_highs[-5:]]; rl=[p for (_,p) in swing_lows[-5:]]
    for i in range(lookback,n):
        for sl in rl:
            if low[i]<sl and close[i]>sl: bs.append({"index":i,"price":close[i],"swept_level":sl,"type":"BULL_SWEEP"}); break
        for sh in rh:
            if high[i]>sh and close[i]<sh: bes.append({"index":i,"price":close[i],"swept_level":sh,"type":"BEAR_SWEEP"}); break
    return bs[-3:], bes[-3:]

def check_smc_signal(df):
    o=df["open"].values.astype(float); h=df["high"].values.astype(float)
    l=df["low"].values.astype(float); c=df["close"].values.astype(float)
    n=len(c)
    if n<30: return 0,0,["Insufficient data"]
    sh,sl=find_swing_points(h,l); bo,beo=find_order_blocks(o,h,l,c)
    bf,bfe=find_fvg(h,l,c); bsw,besw=find_liquidity_sweeps(h,l,c,sh,sl)
    lc=c[-1]; lh=h[-1]; ll=l[-1]
    bs=0; bes2=0; rb=[]; rbe=[]
    for ob in bo[-3:]:
        if ob["bottom"]<=lc<=ob["top"]*1.003: bs+=35; rb.append(f"Bullish OB @ {ob['mid']:.2f}")
    for ob in beo[-3:]:
        if ob["bottom"]*0.997<=lc<=ob["top"]: bes2+=35; rbe.append(f"Bearish OB @ {ob['mid']:.2f}")
    for sw in bsw:
        if sw["index"]>=n-4: bs+=40; rb.append(f"Bull sweep @ {sw['swept_level']:.2f}")
    for sw in besw:
        if sw["index"]>=n-4: bes2+=40; rbe.append(f"Bear sweep @ {sw['swept_level']:.2f}")
    for fvg in bf[-3:]:
        if abs(lc-fvg["mid"])/lc<0.005: bs+=25; rb.append(f"Bull FVG @ {fvg['mid']:.2f}")
    for fvg in bfe[-3:]:
        if abs(lc-fvg["mid"])/lc<0.005: bes2+=25; rbe.append(f"Bear FVG @ {fvg['mid']:.2f}")
    if bs>=35 and bs>bes2: return 1,min(bs,100),rb
    elif bes2>=35 and bes2>bs: return -1,min(bes2,100),rbe
    return 0,0,["No SMC confluence"]

def run_smc_scan():
    try:
        import yfinance as yf
        results={}
        for interval,label in [("1h","1H"),("15m","15M")]:
            df=yf.download("GC=F",period="5d",interval=interval,progress=False,auto_adjust=True)
            if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0].lower() for c in df.columns]
            else: df.columns=[str(c).lower() for c in df.columns]
            if hasattr(df.index,"tz") and df.index.tz: df.index=df.index.tz_convert("UTC").tz_localize(None)
            df=df.dropna(subset=["open","high","low","close"])
            sig,score,reasons=check_smc_signal(df)
            results[label]={"signal":sig,"score":score,"reasons":reasons}
        h1=results.get("1H",{"signal":0,"score":0,"reasons":[]})
        m15=results.get("15M",{"signal":0,"score":0,"reasons":[]})
        if h1["signal"]!=0 and h1["signal"]==m15["signal"]:
            cs=h1["signal"]; csc=(h1["score"]+m15["score"])/2; cr=h1["reasons"]+m15["reasons"]
        elif h1["signal"]!=0 and h1["score"]>=70:
            cs=h1["signal"]; csc=h1["score"]*0.8; cr=h1["reasons"]
        else:
            cs=0; csc=0; cr=["No confluence across timeframes"]
        cache={"signal":cs,"score":round(csc,1),"reasons":cr[:6],"h1":h1,"m15":m15,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
        os.makedirs(BASE,exist_ok=True)
        with open(os.path.join(BASE,"smc_cache.json"),"w") as f: json.dump(cache,f,indent=2)
        sig_str={1:"BUY",-1:"SELL",0:"NONE"}.get(cs,"NONE")
        print(f"  SMC: {sig_str} (score {csc:.0f})")
        for r in cr[:3]: print(f"    - {r}")
        return cache
    except Exception as e: print(f"  SMC error: {e}"); return {"signal":0,"score":0,"reasons":[str(e)]}

if __name__ == "__main__":
    print("Running SMC scan...")
    r = run_smc_scan()
    sig_map = {1:"BUY", -1:"SELL", 0:"NONE"}
    print(f"  Signal: " + sig_map.get(r["signal"],"NONE"))
    print(f"  Score : " + str(r["score"]))
    for reason in r.get("reasons",[]): print("    - " + str(reason))
