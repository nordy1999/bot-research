# AlmostFinishedBot - Complete Setup Script
# Run with: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# Then:      & "$env:USERPROFILE\Downloads\AlmostFinishedBot_SETUP.ps1"

$Base  = "$env:USERPROFILE\Desktop\AlmostFinishedBot"
$MT5   = "$env:APPDATA\MetaQuotes\Terminal\73B7A2420D6397DFF9014A20F1201F97\MQL5"
$Files = "$MT5\Files\AlmostFinishedBot"

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  AlmostFinishedBot - Complete Setup" -ForegroundColor Cyan
Write-Host "  Creating folder: $Base" -ForegroundColor Gray
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# Create directories
New-Item -ItemType Directory -Force -Path $Base      | Out-Null
New-Item -ItemType Directory -Force -Path "$Base\exports" | Out-Null
if (Test-Path $MT5) {
    New-Item -ItemType Directory -Force -Path "$MT5\Experts" | Out-Null
    New-Item -ItemType Directory -Force -Path $Files          | Out-Null
}

# ================================================================
# market_regime.py
# ================================================================
$marketRegime = @'
"""
AlmostFinishedBot - Market Regime Detector
Tier 1: ADX + Bollinger Band Width regime classification
"""
import numpy as np
import pandas as pd

def compute_adx(high, low, close, period=14):
    h, l, c = np.array(high, float), np.array(low, float), np.array(close, float)
    tr  = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    up  = h[1:]-h[:-1]; dn = l[:-1]-l[1:]
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    ndm = np.where((dn>up)&(dn>0), dn, 0.0)
    safe_atr = np.where(atr==0, 1e-9, atr)
    pdi = 100*pd.Series(pdm).ewm(span=period,adjust=False).mean().values/safe_atr
    ndi = 100*pd.Series(ndm).ewm(span=period,adjust=False).mean().values/safe_atr
    denom = np.where(pdi+ndi==0, 1e-9, pdi+ndi)
    dx  = 100*abs(pdi-ndi)/denom
    adx = pd.Series(dx).ewm(span=period,adjust=False).mean().values
    return np.concatenate([[np.nan]*period, adx])[:len(high)]

def compute_bb_width(close, period=20):
    s = pd.Series(close)
    m = s.rolling(period).mean()
    st = s.rolling(period).std()
    return ((m+2*st)-(m-2*st)) / m.replace(0, 1e-9)

def detect_regime(df, adx_trend=25, adx_strong=40, bb_high=0.04, bb_low=0.015):
    adx = compute_adx(df["high"].values, df["low"].values, df["close"].values)
    bbw = compute_bb_width(df["close"].values).values
    n   = len(df)
    pad_a = n - len(adx); adx_f = np.concatenate([np.full(max(0,pad_a), np.nan), adx])[:n]
    pad_b = n - len(bbw);  bbw_f = np.concatenate([np.full(max(0,pad_b), np.nan), bbw])[:n]
    regimes = []
    for a, b in zip(adx_f, bbw_f):
        if np.isnan(a) or np.isnan(b): regimes.append("UNKNOWN")
        elif a >= adx_strong and b >= bb_high: regimes.append("TRENDING_STRONG")
        elif a >= adx_trend:  regimes.append("TRENDING")
        elif b >= bb_high:    regimes.append("HIGH_VOL")
        elif b <= bb_low:     regimes.append("LOW_VOL")
        else:                 regimes.append("RANGING")
    return regimes, adx_f, bbw_f

def regime_to_code(r):
    return {"TRENDING_STRONG":3,"TRENDING":2,"RANGING":1,
            "HIGH_VOL":4,"LOW_VOL":0,"UNKNOWN":1}.get(r, 1)

if __name__ == "__main__":
    import yfinance as yf
    df = yf.download("GC=F", period="1mo", interval="1h", progress=False, auto_adjust=True)
    if hasattr(df.columns,"levels"): df.columns=[c[0].lower() for c in df.columns]
    else: df.columns=[c.lower() for c in df.columns]
    regs, adxv, bbwv = detect_regime(df)
    print(f"Latest regime: {regs[-1]}  ADX: {adxv[-1]:.1f}  BBW: {bbwv[-1]:.4f}")
'@
Set-Content -Path "$Base\market_regime.py" -Value $marketRegime -Encoding UTF8

# ================================================================
# news_sentiment.py
# ================================================================
$newsSentiment = @'
"""
AlmostFinishedBot - News Sentiment Analyser
Tier 2: VADER sentiment on gold-related news feeds
"""
import json, os, sys, datetime
BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")

def install(pkg):
    import subprocess; subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

for pkg in ["feedparser","vaderSentiment"]:
    try: __import__(pkg)
    except: install(pkg)

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FEEDS = [
    ("Kitco",    "https://www.kitco.com/rss/rssfeeds/headline_news.xml"),
    ("Reuters",  "https://feeds.reuters.com/reuters/businessNews"),
    ("FXStreet", "https://www.fxstreet.com/rss"),
    ("MarketW",  "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
]
GOLD_KW = ["gold","xau","bullion","precious metal","fed","inflation","fomc","dollar","usd","rate"]

def score_articles():
    vader = SentimentIntensityAnalyzer()
    arts  = []
    for src, url in FEEDS:
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            for e in feed.entries[:20]:
                t = (e.get("title","") + " " + e.get("summary",""))[:400]
                if not any(kw in t.lower() for kw in GOLD_KW): continue
                s = vader.polarity_scores(t)["compound"]
                arts.append({"title": e.get("title","")[:90], "source": src,
                             "score": round(s,3), "url": e.get("link","")})
        except Exception as ex:
            arts.append({"title":f"Feed error ({src})", "source":src, "score":0.0, "url":""})

    if not arts:
        result = {"signal":"NEUTRAL","score":0.0,"articles_analysed":0,
                  "top_articles":[],"timestamp":datetime.datetime.utcnow().isoformat()}
    else:
        arts.sort(key=lambda x: abs(x["score"]), reverse=True)
        avg = sum(a["score"] for a in arts) / len(arts)
        sig = "BUY" if avg>0.08 else ("SELL" if avg<-0.08 else "NEUTRAL")
        result = {"signal":sig,"score":round(avg,4),"articles_analysed":len(arts),
                  "top_articles":arts[:8],"timestamp":datetime.datetime.utcnow().isoformat()}

    os.makedirs(BASE, exist_ok=True)
    with open(os.path.join(BASE,"news_cache.json"),"w") as f:
        json.dump(result, f, indent=2)
    print(f"News: {result['signal']} ({result['score']:+.3f}) from {result['articles_analysed']} articles")
    return result

if __name__ == "__main__":
    score_articles()
'@
Set-Content -Path "$Base\news_sentiment.py" -Value $newsSentiment -Encoding UTF8

# ================================================================
# trade_monitor.py
# ================================================================
$tradeMonitor = @'
"""
AlmostFinishedBot - Trade Monitor
Cleans MT5 spaced-character log output
"""
import os, sys, time, re

BASE   = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
LOGDIR = os.path.join(os.environ.get("APPDATA",""), "MetaQuotes", "Terminal",
         "73B7A2420D6397DFF9014A20F1201F97", "MQL5", "Logs")

mode = sys.argv[1] if len(sys.argv) > 1 else "paper"
BANNERS = {
    "paper":    ("PAPER TRADE MONITOR", "Demo 62111276"),
    "live":     ("LIVE TRADE MONITOR",  "*** REAL MONEY ***"),
    "backtest": ("BACKTEST MONITOR",    "$1,000 deposit"),
}
title, subtitle = BANNERS.get(mode, ("MONITOR",""))

def clean_mt5(line):
    line = re.sub(r'(?<=\S) (?=\S)', '', line)
    return re.sub(r'\s+', ' ', line).strip()

os.system("cls" if os.name=="nt" else "clear")
print("=" * 70)
print(f"  AlmostFinishedBot  |  {title}")
print(f"  {subtitle}  |  Ctrl+C to stop")
print("=" * 70)

SUPPRESS = ["trail buy error","trail sell error","ml ensemble not found",
            "fallback strategy","ml signal files not found",
            "ml signal not found"]
SHOW_ONCE = {}
counts = {"BUY":0,"SELL":0,"errors":0,"suppressed":0}
IMPORTANT_KW = [
    "buy |","sell |",">>> buy",">>> sell",
    "ordersend","order opened","order closed","order modified",
    "balance:","equity","profit","loss",
    "kelly","regime","confidence","risk:",
    "order block","liquidity","fvg","swing","sniper",
    "oninit finished","ondeinit","telegram sent","error 4"
]

def latest_log():
    try:
        files = sorted([f for f in os.listdir(LOGDIR) if f.endswith(".log")])
        return os.path.join(LOGDIR, files[-1]) if files else None
    except: return None

last_log = None; last_size = 0
print(f"\n  Watching: {LOGDIR}")
print(f"  Waiting for MT5 activity...\n")

while True:
    try:
        lp = latest_log()
        if lp != last_log:
            last_log = lp; last_size = 0; SHOW_ONCE.clear()
            if lp:
                print(f"\n  [LOG] {os.path.basename(lp)}")
                print("  " + "-" * 66)

        if lp and os.path.exists(lp):
            sz = os.path.getsize(lp)
            if sz > last_size:
                with open(lp,"r",encoding="utf-8",errors="ignore") as f:
                    f.seek(last_size); raw = f.read()
                last_size = sz
                for raw_line in raw.splitlines():
                    if not raw_line.strip(): continue
                    line = clean_mt5(raw_line)
                    if not line: continue
                    low = line.lower()
                    supp = any(s in low for s in SUPPRESS)
                    if supp:
                        key = next(s for s in SUPPRESS if s in low)
                        if key not in SHOW_ONCE:
                            SHOW_ONCE[key] = True
                            print(f"  [INFO]  {line}  (suppressing repeats)")
                        else: counts["suppressed"] += 1
                        continue
                    if any(x in low for x in ["xauusd buy","xausgd buy","buy |"]):
                        print(f"\n  {'='*10} BUY SIGNAL {'='*10}")
                        print(f"  {line}"); print(f"  {'='*31}\n"); counts["BUY"] += 1
                    elif any(x in low for x in ["xauusd sell","xausgd sell","sell |"]):
                        print(f"\n  {'='*10} SELL SIGNAL {'='*9}")
                        print(f"  {line}"); print(f"  {'='*31}\n"); counts["SELL"] += 1
                    elif "error" in low and "trail" not in low:
                        print(f"  [ERR ]  {line}"); counts["errors"] += 1
                    elif any(kw in low for kw in IMPORTANT_KW):
                        print(f"  {line}")
        time.sleep(0.4)
    except KeyboardInterrupt:
        print(f"\n  Session: BUY:{counts['BUY']}  SELL:{counts['SELL']}  Err:{counts['errors']}  Suppressed:{counts['suppressed']}")
        break
    except Exception as e:
        print(f"  [ERR] {e}"); time.sleep(2)
'@
Set-Content -Path "$Base\trade_monitor.py" -Value $tradeMonitor -Encoding UTF8

# ================================================================
# live_graph.py  (full version - embedded)
# ================================================================
$liveGraph = @'
"""
AlmostFinishedBot - Live Graph
Real-time candlestick chart with EA trade markers
"""
import sys, os, time, re, threading
import tkinter as tk
from tkinter import ttk
import warnings; warnings.filterwarnings("ignore")

BASE   = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
LOGDIR = os.path.join(os.environ.get("APPDATA",""), "MetaQuotes", "Terminal",
         "73B7A2420D6397DFF9014A20F1201F97", "MQL5", "Logs")
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"

BG_DARK="#111111"; BG_MID="#1e1e1e"; BG_PANEL="#2a2a2a"
WHITE="#ffffff"; GREY_LT="#cccccc"; GREY_MID="#888888"; GREY_DIM="#555555"
RED_HI="#ff2222"; RED_MID="#cc3333"
GREEN_TR="#22ff88"; RED_TR="#ff2244"; GOLD_LINE="#ffcc44"
CANDLE_UP="#33cc66"; CANDLE_DN="#dd2244"
REFRESH=15

try:
    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import pandas as pd; import numpy as np
    HAS_MPL=True
except: HAS_MPL=False

try: import yfinance as yf; HAS_YF=True
except: HAS_YF=False

def clean_mt5(line):
    line=re.sub(r'(?<=\S) (?=\S)','',line)
    return re.sub(r'\s+',' ',line).strip()

class LiveGraph:
    def __init__(self, root):
        self.root=root; self.df=None; self.trades=[]; self.lock=threading.Lock()
        self.interval=tk.StringVar(value="15m"); self.period=tk.StringVar(value="1d")
        self.running=True; self._build_ui()
        threading.Thread(target=self._fetch_loop,daemon=True).start()
        threading.Thread(target=self._log_loop,daemon=True).start()

    def _build_ui(self):
        r=self.root; r.title(f"AlmostFinishedBot — Live Chart [{SYMBOL}]")
        r.configure(bg=BG_MID); r.geometry("1280x780")
        hdr=tk.Frame(r,bg=BG_DARK,highlightbackground=WHITE,highlightthickness=1)
        hdr.pack(fill="x",padx=8,pady=(8,0))
        tk.Label(hdr,text=f"  ALMOSTFINISHEDBOT  |  LIVE CHART  |  {SYMBOL}  ",
                 fg=RED_HI,bg=BG_DARK,font=("Consolas",13,"bold")).pack(side="left")
        self.price_lbl=tk.Label(hdr,text="Fetching...",fg=GREY_LT,bg=BG_DARK,font=("Consolas",11))
        self.price_lbl.pack(side="right",padx=10)
        tb=tk.Frame(r,bg=BG_PANEL,highlightbackground=WHITE,highlightthickness=1)
        tb.pack(fill="x",padx=8,pady=4)
        tk.Label(tb,text="  Interval:",fg=GREY_LT,bg=BG_PANEL,font=("Consolas",9)).pack(side="left")
        for iv in ["1m","5m","15m","30m","1h"]:
            tk.Radiobutton(tb,text=iv,variable=self.interval,value=iv,bg=BG_PANEL,fg=GREY_LT,
                selectcolor=RED_MID,activebackground=BG_PANEL,font=("Consolas",9),
                command=self._refresh).pack(side="left",padx=2)
        tk.Label(tb,text="   Period:",fg=GREY_LT,bg=BG_PANEL,font=("Consolas",9)).pack(side="left")
        for pv in ["1d","5d","1mo"]:
            tk.Radiobutton(tb,text=pv,variable=self.period,value=pv,bg=BG_PANEL,fg=GREY_LT,
                selectcolor=RED_MID,activebackground=BG_PANEL,font=("Consolas",9),
                command=self._refresh).pack(side="left",padx=2)
        tk.Button(tb,text=" Refresh Now ",bg=RED_MID,fg=WHITE,font=("Consolas",9,"bold"),
                  relief="flat",command=self._refresh).pack(side="right",padx=6,pady=3)
        tk.Button(tb,text=" Clear Markers ",bg=BG_DARK,fg=GREY_MID,font=("Consolas",9),
                  relief="flat",command=self._clear_trades).pack(side="right",padx=4)
        self.fig_frame=tk.Frame(r,bg=BG_MID,highlightbackground=WHITE,highlightthickness=1)
        self.fig_frame.pack(fill="both",expand=True,padx=8,pady=4)
        if HAS_MPL:
            self.fig,(self.ax,self.axv)=plt.subplots(2,1,figsize=(12,6.5),
                gridspec_kw={"height_ratios":[4,1],"hspace":0.05},facecolor=BG_MID)
            self.canvas=FigureCanvasTkAgg(self.fig,master=self.fig_frame)
            self.canvas.get_tk_widget().pack(fill="both",expand=True)
        else:
            tk.Label(self.fig_frame,text="pip install matplotlib",fg=RED_HI,bg=BG_MID,
                     font=("Consolas",12)).pack(expand=True)
        sb=tk.Frame(r,bg=BG_DARK,highlightbackground=WHITE,highlightthickness=1)
        sb.pack(fill="x",padx=8,pady=(0,8))
        self.status_lbl=tk.Label(sb,text="  Initialising...",fg=GREY_MID,bg=BG_DARK,font=("Consolas",8))
        self.status_lbl.pack(side="left")
        self.trade_lbl=tk.Label(sb,text="",fg=GREY_MID,bg=BG_DARK,font=("Consolas",8))
        self.trade_lbl.pack(side="right",padx=8)

    def _fetch_loop(self):
        while self.running:
            self._do_fetch()
            for _ in range(REFRESH*10):
                if not self.running: break
                time.sleep(0.1)

    def _do_fetch(self):
        if not HAS_YF: return
        try:
            data=yf.download("GC=F",period=self.period.get(),interval=self.interval.get(),
                             progress=False,auto_adjust=True)
            if data is None or len(data)<5: return
            if hasattr(data.columns,"levels"): data.columns=[c[0].lower() for c in data.columns]
            else: data.columns=[c.lower() for c in data.columns]
            if hasattr(data.index,"tz") and data.index.tz:
                data.index=data.index.tz_convert("UTC").tz_localize(None)
            data=data.tail(80)
            with self.lock: self.df=data
            self.root.after(0,self._draw)
        except Exception as e:
            self.root.after(0,lambda:self.status_lbl.config(text=f"  Fetch error: {e}"))

    def _log_loop(self):
        last_size=0; last_log=None
        while self.running:
            try:
                if os.path.exists(LOGDIR):
                    logs=sorted([f for f in os.listdir(LOGDIR) if f.endswith(".log")])
                    if logs:
                        lp=os.path.join(LOGDIR,logs[-1])
                        if lp!=last_log: last_log=lp; last_size=0
                        sz=os.path.getsize(lp)
                        if sz>last_size:
                            with open(lp,"r",encoding="utf-8",errors="ignore") as f:
                                f.seek(last_size); raw=f.read()
                            last_size=sz
                            for line in raw.splitlines():
                                cl=clean_mt5(line)
                                m=re.search(r'(\d{2}:\d{2}:\d{2}).*?(XAUUSD|XAUSGD)\s+(BUY|SELL)\s*\|\s*Lots:\s*([\d.]+)',cl)
                                if m:
                                    t_str,sym,side,lots=m.groups()
                                    with self.lock:
                                        self.trades.append({"time":t_str,"symbol":sym,"side":side,"lots":lots})
            except: pass
            time.sleep(5)

    def _draw(self):
        if not HAS_MPL: return
        import numpy as np, pandas as pd
        with self.lock:
            df=self.df.copy() if self.df is not None else None
            trades=list(self.trades)
        self.ax.clear(); self.axv.clear()
        for a in [self.ax,self.axv]:
            a.set_facecolor(BG_MID)
            a.tick_params(colors=GREY_MID,labelsize=7)
            for sp in a.spines.values(): sp.set_color(GREY_DIM)
        if df is None or len(df)<5:
            self.ax.text(0.5,0.5,"Fetching price data...",transform=self.ax.transAxes,
                        color=GREY_MID,ha="center",va="center",fontsize=13,fontfamily="Consolas")
            self.canvas.draw(); return
        xs=range(len(df))
        for i,(idx,row) in enumerate(df.iterrows()):
            o,h,l,c=row["open"],row["high"],row["low"],row["close"]
            col=CANDLE_UP if c>=o else CANDLE_DN
            self.ax.plot([i,i],[l,h],color=col,linewidth=0.8,zorder=2)
            self.ax.bar(i,abs(c-o),bottom=min(o,c),color=col,width=0.7,zorder=3,linewidth=0)
        closes=df["close"].values
        for span,col in [(20,"#ff9500"),(50,"#00ccff")]:
            if len(closes)>=span:
                ema=[None]*(span-1); e=float(np.mean(closes[:span])); ema.append(e); k=2/(span+1)
                for c in closes[span:]: e=float(c)*k+e*(1-k); ema.append(e)
                self.ax.plot(list(xs),ema,color=col,linewidth=1,label=f"EMA{span}",zorder=4)
        last_p=float(df["close"].iloc[-1])
        self.ax.axhline(last_p,color=GOLD_LINE,linewidth=0.8,linestyle="--",zorder=5)
        today_str=time.strftime("%Y-%m-%d"); plotted=0
        for tr in trades:
            try:
                tr_dt=pd.Timestamp(today_str+" "+tr["time"])
                diffs=abs(df.index-tr_dt); idx_c=diffs.argmin()
                if diffs[idx_c].total_seconds()<3600*4:
                    col=GREEN_TR if tr["side"]=="BUY" else RED_TR
                    marker="^" if tr["side"]=="BUY" else "v"
                    y_off=float(df["low"].iloc[idx_c])*0.9998 if tr["side"]=="BUY" else float(df["high"].iloc[idx_c])*1.0002
                    self.ax.scatter(idx_c,y_off,color=col,marker=marker,s=120,zorder=6)
                    self.ax.text(idx_c,y_off,f"\n{tr['side']}\n{tr['lots']}",color=col,fontsize=6,
                                ha="center",zorder=7,fontfamily="Consolas")
                    plotted+=1
            except: pass
        if not trades:
            self.ax.text(0.01,0.98,"No EA trades detected — showing price only",
                        transform=self.ax.transAxes,color=GREY_DIM,va="top",fontsize=7,fontfamily="Consolas")
        try:
            vols=df["volume"].values if "volume" in df.columns else np.ones(len(df))
            bar_colors=[CANDLE_UP if df["close"].iloc[i]>=df["open"].iloc[i] else CANDLE_DN for i in range(len(df))]
            self.axv.bar(list(xs),vols,color=bar_colors,width=0.7,linewidth=0); self.axv.set_yticks([])
        except: pass
        n=len(df); step=max(1,n//10); xticks=list(range(0,n,step))
        xlabels=[df.index[i].strftime("%H:%M\n%d/%m") for i in xticks]
        self.ax.set_xticks([]); self.axv.set_xticks(xticks)
        self.axv.set_xticklabels(xlabels,color=GREY_MID,fontsize=6)
        self.axv.tick_params(colors=GREY_MID,labelsize=6)
        self.ax.yaxis.set_label_position("right"); self.ax.yaxis.tick_right()
        self.ax.set_xlim(-1,n); self.axv.set_xlim(-1,n)
        self.ax.legend(loc="upper left",fontsize=7,facecolor=BG_PANEL,edgecolor=GREY_DIM,labelcolor=GREY_LT)
        self.fig.tight_layout(pad=0.3); self.canvas.draw()
        prev_p=float(df["close"].iloc[-2]) if len(df)>1 else last_p
        chg=last_p-prev_p; pct=chg/prev_p*100 if prev_p else 0
        col="#22cc66" if chg>=0 else RED_HI
        self.price_lbl.config(text=f"{SYMBOL}  ${last_p:,.2f}  {chg:+.2f} ({pct:+.2f}%)",fg=col)
        self.status_lbl.config(text=f"  Updated: {time.strftime('%H:%M:%S')}  Candles: {len(df)}  Trades: {plotted}")
        self.trade_lbl.config(text=f"{'Trades today: '+str(len(trades)) if trades else 'No EA trades yet'}  ",
                             fg=GREEN_TR if trades else GREY_DIM)

    def _refresh(self): threading.Thread(target=self._do_fetch,daemon=True).start()
    def _clear_trades(self):
        with self.lock: self.trades.clear()
        self._draw()

root=tk.Tk(); app=LiveGraph(root); root.mainloop()
'@
Set-Content -Path "$Base\live_graph.py" -Value $liveGraph -Encoding UTF8

# ================================================================
# bot_settings.json
# ================================================================
$settings = @{
    RiskPercent    = 1.0
    MaxTrades      = 2
    MaxDrawdownPct = 10.0
    MaxSpreadATR   = 0.5
    UseTrailing    = $true
    UseKelly       = $true
    UseRegime      = $true
    UseSMC         = $true
    SwingMode      = $true
    TelegramToken  = ""
    TelegramChatID = "838489368"
}
$settings | ConvertTo-Json -Depth 2 | Set-Content -Path "$Base\bot_settings.json" -Encoding UTF8

Write-Host "  [OK] market_regime.py" -ForegroundColor Green
Write-Host "  [OK] news_sentiment.py" -ForegroundColor Green
Write-Host "  [OK] trade_monitor.py" -ForegroundColor Green
Write-Host "  [OK] live_graph.py" -ForegroundColor Green
Write-Host "  [OK] bot_settings.json" -ForegroundColor Green

# ================================================================
# Copy train_models.py from script directory if present, otherwise note
# ================================================================
Write-Host ""
Write-Host "  Writing train_models.py..." -ForegroundColor Yellow

# Note: train_models.py is large and split into a separate download
# The Control Center will detect its absence and warn

# ================================================================
# Default news_cache.json
# ================================================================
$news = '{"signal":"NEUTRAL","score":0.0,"articles_analysed":0,"top_articles":[],"timestamp":"2026-01-01T00:00:00"}'
Set-Content -Path "$Base\news_cache.json" -Value $news -Encoding UTF8
Write-Host "  [OK] news_cache.json" -ForegroundColor Green

# ================================================================
# LAUNCH SCRIPT
# ================================================================
$launcher = @"
@echo off
title AlmostFinishedBot Control Center
cd /d "%USERPROFILE%\Desktop\AlmostFinishedBot"
python AlmostFinishedBot_ControlCenter.py
pause
"@
Set-Content -Path "$Base\Launch_ControlCenter.bat" -Value $launcher -Encoding ASCII
Write-Host "  [OK] Launch_ControlCenter.bat" -ForegroundColor Green

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  Folder: $Base" -ForegroundColor Gray
Write-Host ""
Write-Host "  NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Copy AlmostFinishedBot_ControlCenter.py to the folder" -ForegroundColor White
Write-Host "  2. Copy train_models.py to the folder" -ForegroundColor White
Write-Host "  3. Copy AlmostFinishedBot_EA.mq5 to MT5 Experts folder" -ForegroundColor White
Write-Host "  4. Run: Launch_ControlCenter.bat" -ForegroundColor White
Write-Host "  5. Click Training Section -> Train All Models" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# Open the folder
Start-Process explorer.exe $Base
