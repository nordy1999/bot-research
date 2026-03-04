"""
AlmostFinishedBot - Control Center v2.0
Sections: Trading / News & Sentiment / Analysis / Training / Testing / Tools / Risk & Prop Firm
Grey/white/red theme | All new features integrated
"""
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os, sys, subprocess, json, time, threading

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
os.makedirs(BASE, exist_ok=True)

BG0="#0f0f0f"; BG1="#1a1a1a"; BG2="#242424"; BG3="#303030"
BORDER="#ffffff"; BORD_D="#555555"
RED_HI="#ff2222"; RED_MID="#cc3333"; RED_DIM="#882222"
WHITE="#ffffff"; GREY_LT="#dddddd"; GREY_MID="#999999"; GREY_DIM="#555555"
GREEN="#22cc66"; GOLD="#ffcc44"; BLUE="#4499ff"

FH=("Consolas",10,"bold"); FB=("Consolas",9); FS=("Consolas",8)
FT=("Consolas",14,"bold"); FBB=("Consolas",9,"bold")

def bordered(parent, **kw):
    outer=tk.Frame(parent,bg=BORDER,padx=1,pady=1,**kw)
    inner=tk.Frame(outer,bg=BG2); inner.pack(fill="both",expand=True)
    return outer, inner

def section_header(parent, text, color=None):
    c = color or RED_HI
    frm=tk.Frame(parent,bg=BG0,highlightbackground=BORDER,highlightthickness=1)
    frm.pack(fill="x",padx=6,pady=(12,2))
    tk.Label(frm,text=f"  {text}  ",fg=c,bg=BG0,font=FH).pack(side="left",pady=4)
    return frm

def info_btn(parent, title, body):
    def show():
        win=tk.Toplevel(); win.title(title); win.configure(bg=BG1); win.geometry("480x300"); win.resizable(False,False)
        fo,fi=bordered(win); fo.pack(fill="x",padx=8,pady=(8,0))
        tk.Label(fi,text=f" {title}",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=6)
        fo2,fi2=bordered(win); fo2.pack(fill="both",expand=True,padx=8,pady=8)
        tk.Label(fi2,text=body,fg=GREY_LT,bg=BG2,font=FB,wraplength=440,justify="left").pack(anchor="nw",padx=12,pady=12)
        tk.Button(win,text=" Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=win.destroy).pack(pady=8)
    b=tk.Button(parent,text=" ? ",bg=BG2,fg=GREY_DIM,font=FS,relief="flat",cursor="hand2",command=show,width=3)
    b.bind("<Enter>",lambda e:b.config(fg=WHITE,bg=BG3))
    b.bind("<Leave>",lambda e:b.config(fg=GREY_DIM,bg=BG2))
    return b

class App:
    def __init__(self, root):
        self.root=root; self.root.title("AlmostFinishedBot — Control Center v2")
        self.root.configure(bg=BG1); self.root.geometry("920x920"); self.root.minsize(820,700)
        self.active_procs={}; self._build(); self._start_bg()
        self._auto_refresh(); self._retrain_check()

    def _build(self):
        r=self.root
        oh,ih=bordered(r); oh.pack(fill="x",padx=8,pady=(8,4))
        tk.Label(ih,text="  ALMOSTFINISHEDBOT  |  CONTROL CENTER  v2  ",
                 fg=RED_HI,bg=BG2,font=("Consolas",15,"bold")).pack(side="left",pady=8)
        self.acc_lbl=tk.Label(ih,text="",fg=GREY_MID,bg=BG2,font=FS); self.acc_lbl.pack(side="right",padx=6)
        self.mode_lbl=tk.Label(ih,text="",fg=GOLD,bg=BG2,font=FBB); self.mode_lbl.pack(side="right",padx=6)

        canvas=tk.Canvas(r,bg=BG1,highlightthickness=0)
        vsb=tk.Scrollbar(r,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y"); canvas.pack(side="left",fill="both",expand=True,padx=(8,0),pady=4)
        self.sf=tk.Frame(canvas,bg=BG1)
        sfid=canvas.create_window((0,0),window=self.sf,anchor="nw")
        self.sf.bind("<Configure>",lambda e:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",lambda e:canvas.itemconfig(sfid,width=e.width))
        canvas.bind_all("<MouseWheel>",lambda e:canvas.yview_scroll(int(-1*(e.delta/120)),"units"))
        sf=self.sf

        # ═══════════════════════════════════════════════════════
        # TRADING SECTION
        # ═══════════════════════════════════════════════════════
        section_header(sf,"▶  TRADING SECTION",GREEN)
        tg=tk.Frame(sf,bg=BG1); tg.pack(fill="x",padx=6)
        self._card(tg,0,0,"▶  CLICK HERE TO TRADE","Choose Paper / Backtest / Live trading mode",self._trade_prompt,bold=True,mc=GREEN)
        self._card(tg,0,1,"⬛  KILL SWITCH — STOP ALL","Immediately kills all processes and closes MT5",self._kill_all,bold=True,mc=RED_HI)
        self._card(tg,1,0,"📊  Live Graph — XAUUSD","Real-time candlestick chart with EA trade markers",lambda:self._open_graph("XAUUSD"))
        self._card(tg,1,1,"📊  Live Graph — XAUSGD","Real-time candlestick chart with EA trade markers",lambda:self._open_graph("XAUSGD"))
        self._card(tg,2,0,"📋  Trade Monitor — Paper","Watch MT5 logs live (paper mode, cleans spaced chars)",lambda:self._run_console("trade_monitor.py","paper","Paper Monitor"))
        self._card(tg,2,1,"📋  Trade Monitor — Live","Watch MT5 logs live (LIVE money mode)",lambda:self._run_console("trade_monitor.py","live","Live Monitor"))

        # ═══════════════════════════════════════════════════════
        # RISK & PROP FIRM SECTION
        # ═══════════════════════════════════════════════════════
        section_header(sf,"⚡  RISK & PROP FIRM SECTION",GOLD)
        rg=tk.Frame(sf,bg=BG1); rg.pack(fill="x",padx=6)
        self._card(rg,0,0,"⚡  God Mode Status","Shows if God Mode (8% risk) is currently active — requires 88%+ confidence + SMC + regime alignment",self._show_risk_status)
        self._card(rg,0,1,"🏆  Prop Firm Mode ON/OFF","Toggle FTMO/Funded Trader mode: 0.5% risk, 3% daily DD limit, 6% total DD limit, 8% profit target",self._toggle_prop_firm)
        self._card(rg,1,0,"📈  Prop Firm Progress","Shows challenge progress bar, DD remaining, profit vs target",self._show_prop_progress)
        self._card(rg,1,1,"💰  Compounding Calculator","Projects your £100 at 1%, 1.5%, 2% daily return over 7/14/30/60/90 days",self._show_compounding)
        self._card(rg,2,0,"🎯  Kelly Criterion Sizing","Current Kelly fraction, half-Kelly used, confidence scaling tiers",self._show_kelly)
        self._card(rg,2,1,"📊  Session & Vol Targeting","Current trading session, ATR-based position sizing, volatility target",self._show_session)

        # ═══════════════════════════════════════════════════════
        # NEWS & SENTIMENT SECTION
        # ═══════════════════════════════════════════════════════
        section_header(sf,"📰  NEWS & SENTIMENT SECTION",BLUE)
        no,ni=bordered(sf); no.pack(fill="x",padx=6,pady=2)
        nh=tk.Frame(ni,bg=BG2); nh.pack(fill="x",padx=4,pady=4)
        tk.Label(nh,text="  Gold News Sentiment + Event Guard",fg=GREY_DIM,bg=BG2,font=FS).pack(side="left")
        tk.Button(nh,text=" Refresh ",bg=BG3,fg=GREY_LT,font=FS,relief="flat",command=self._fetch_news).pack(side="right",padx=4)
        tk.Button(nh,text=" Check Guard ",bg=RED_DIM,fg=WHITE,font=FS,relief="flat",command=self._check_news_guard).pack(side="right",padx=4)
        self.news_lbl=tk.Label(ni,text="  —  not loaded  —",fg=GREY_MID,bg=BG2,font=FBB); self.news_lbl.pack(anchor="w",padx=8,pady=2)
        self.guard_lbl=tk.Label(ni,text="",fg=GREY_MID,bg=BG2,font=FS); self.guard_lbl.pack(anchor="w",padx=8,pady=1)
        no2,ni2=bordered(ni); no2.pack(fill="x",padx=4,pady=(0,4))
        self.news_box=tk.Text(ni2,height=4,bg=BG0,fg=GREY_MID,font=FS,state="disabled",wrap="word",relief="flat")
        self.news_box.pack(fill="x",padx=4,pady=4)

        ng=tk.Frame(sf,bg=BG1); ng.pack(fill="x",padx=6)
        self._card(ng,0,0,"📰  Fetch Gold News Sentiment","VADER analysis of Kitco, Reuters, FXStreet, MarketWatch headlines",lambda:self._run_popup("news_sentiment.py","","News Sentiment",500,360))
        self._card(ng,0,1,"🛡️  News Guard — Event Blackout","Checks 30min blackout before/after FOMC, NFP, CPI. Sentiment flood override.",lambda:self._run_popup("news_guard.py","","News Guard",520,400))
        self._card(ng,1,0,"🔗  Correlation Guard — DXY/SPX/BTC","Checks DXY decoupling, VIX spike, US10Y vs gold. Kill switch if needed.",lambda:self._run_popup("correlation_guard.py","","Correlation Guard",520,400))
        self._card(ng,1,1,"🌍  Macro Overview","Opens all guard statuses: news + correlation + regime in one window",self._show_macro_overview)

        # ═══════════════════════════════════════════════════════
        # ANALYSIS SECTION
        # ═══════════════════════════════════════════════════════
        section_header(sf,"📈  ANALYSIS SECTION")
        ag=tk.Frame(sf,bg=BG1); ag.pack(fill="x",padx=6)
        self._card(ag,0,0,"📈  Market Regime Detector","ADX + BBW + Hurst Exponent + Volume Profile — classifies into 5 regimes",self._run_regime)
        self._card(ag,0,1,"🤖  Current AI Signal","Latest ensemble signal: BUY/SELL/NEUTRAL, confidence, sized risk %",self._show_signal)
        self._card(ag,1,0,"🕵️  SMC Detector","Order Blocks, FVGs, Liquidity Sweeps, Breaker Blocks — 1H + 15M confluence",lambda:self._run_popup("smc_detector.py","","SMC Detector",520,420))
        self._card(ag,1,1,"📊  Performance Charts","Win rate, profit factor, Sharpe ratio, equity curve, drawdown chart",self._run_analytics)
        self._card(ag,2,0,"💡  Signal Explanation","Full 'why this trade' breakdown: SMC reason + regime + confidence + Kelly",self._show_signal_explanation)
        self._card(ag,2,1,"📁  Open Exports Folder","Opens exports folder containing CSV reports and analytics",self._open_exports)

        # ═══════════════════════════════════════════════════════
        # TRAINING SECTION
        # ═══════════════════════════════════════════════════════
        section_header(sf,"🧠  TRAINING SECTION")
        trg=tk.Frame(sf,bg=BG1); trg.pack(fill="x",padx=6)
        self._card(trg,0,0,"🧠  Train All Models","XGBoost + LightGBM + GradBoost + LSTM | Regime features | Kelly calc | ~5-8 min",lambda:self._run_console("train_models.py","","Train Models"))
        self._card(trg,0,1,"🎯  Show Model Accuracy","Each model's accuracy, weight, training samples, model age",self._show_accuracy)
        self._card(trg,1,0,"📊  Regime Analysis (Train)","Re-runs regime detector and prints ADX, BBW, Hurst, regime per candle",lambda:self._run_popup("market_regime.py","","Regime",500,360))
        self._card(trg,1,1,"🔄  Install / Update Packages","Installs xgboost, lightgbm, tensorflow, yfinance, feedparser, etc.",self._install_packages)

        # ═══════════════════════════════════════════════════════
        # TESTING SECTION
        # ═══════════════════════════════════════════════════════
        section_header(sf,"🧪  TESTING SECTION")
        tsg=tk.Frame(sf,bg=BG1); tsg.pack(fill="x",padx=6)
        self._card(tsg,0,0,"🧪  Strategy Tester / Backtest","Opens MT5 Strategy Tester + backtest monitor",self._backtest)
        self._card(tsg,0,1,"📟  Test Telegram","Sends test message to your Telegram bot",self._test_telegram)
        self._card(tsg,1,0,"🏥  System Health Check","Checks all files, packages, model ages, missing components",self._health_check)
        self._card(tsg,1,1,"🔑  Get Telegram Chat ID","Auto-detects your Telegram Chat ID via getUpdates API",self._get_chat_id)
        self._card(tsg,2,0,"☀️  Sunday Gap Check","Detects Sunday open gap in gold price — gap fill / continuation strategy",self._check_sunday_gap)
        self._card(tsg,2,1,"🌏  Asian Session Check","Shows if Asian session is active (best for small accounts / sniping)",self._check_session)

        # ═══════════════════════════════════════════════════════
        # TOOLS & FILES SECTION
        # ═══════════════════════════════════════════════════════
        section_header(sf,"🔧  TOOLS & FILES SECTION")
        flg=tk.Frame(sf,bg=BG1); flg.pack(fill="x",padx=6)
        self._card(flg,0,0,"📁  Open Bot Folder",f"Opens {BASE}",lambda:os.startfile(BASE))
        self._card(flg,0,1,"📁  Open MT5 Experts","Opens MetaTrader 5 MQL5 Experts folder",self._open_mt5_experts)
        self._card(flg,1,0,"📋  Open MT5 Logs","Opens MT5 logs folder for manual inspection",self._open_mt5_logs)
        self._card(flg,1,1,"⚙️  Quick Settings","Edit risk %, prop firm settings, Telegram token, mode",self._open_settings)
        self._card(flg,2,0,"📋  Copy Folder Path","Copies folder path to clipboard",lambda:(self.root.clipboard_clear(),self.root.clipboard_append(BASE),self._status("Path copied")))
        self._card(flg,2,1,"🔄  Reload Control Center","Restarts this Control Center",self._reload)

        # ═══════════════════════════════════════════════════════
        # PROCESS TRACKER
        # ═══════════════════════════════════════════════════════
        section_header(sf,"⚙️  RUNNING PROCESSES")
        po,pi=bordered(sf); po.pack(fill="x",padx=6,pady=2)
        tk.Label(pi,text="  Processes launched from this Control Center",fg=GREY_DIM,bg=BG2,font=FS).pack(anchor="w",padx=4,pady=2)
        self.proc_box=tk.Text(pi,height=3,bg=BG0,fg=GREY_MID,font=FS,state="disabled",relief="flat")
        self.proc_box.pack(fill="x",padx=4,pady=(0,4))

        # STATUS BAR
        so,si=bordered(r); so.pack(fill="x",padx=8,pady=(0,8))
        self.status_lbl=tk.Label(si,text="  Ready.",fg=GREY_MID,bg=BG2,font=FS); self.status_lbl.pack(side="left",pady=2)
        self.time_lbl=tk.Label(si,text="",fg=GREY_DIM,bg=BG2,font=FS); self.time_lbl.pack(side="right",padx=8)

        self._load_accuracy(); self._refresh_news_display(); self._refresh_guard_display()

    # ── Card builder ──────────────────────────────────────────────
    def _card(self,parent,row,col,label,info_text,cmd,bold=False,mc=None):
        mc=mc or RED_MID
        outer=tk.Frame(parent,bg=BORDER,padx=1,pady=1)
        outer.grid(row=row,column=col,padx=3,pady=3,sticky="nsew")
        inner=tk.Frame(outer,bg=BG2); inner.pack(fill="both",expand=True)
        parent.columnconfigure(0,weight=1,uniform="col"); parent.columnconfigure(1,weight=1,uniform="col")
        font=FBB if bold else FB
        btn=tk.Button(inner,text=f"  {label}",fg=mc,bg=BG2,font=font,relief="flat",anchor="w",cursor="hand2",command=cmd)
        btn.pack(side="left",fill="both",expand=True,padx=2,pady=6)
        info_btn(inner,label,info_text).pack(side="right",padx=4)
        def on_e(e): btn.config(bg=BG3); inner.config(bg=BG3); outer.config(bg=WHITE)
        def on_l(e): btn.config(bg=BG2); inner.config(bg=BG2); outer.config(bg=BORDER)
        btn.bind("<Enter>",on_e); btn.bind("<Leave>",on_l)

    def _status(self,msg):
        try: self.status_lbl.config(text=f"  {msg}")
        except: pass
        self.root.update_idletasks()

    # ── Background tasks ──────────────────────────────────────────
    def _start_bg(self):
        def tick():
            while True:
                try: self.time_lbl.config(text=time.strftime("  %H:%M:%S  "))
                except: pass
                self._refresh_procs(); time.sleep(3)
        threading.Thread(target=tick,daemon=True).start()

    def _auto_refresh(self):
        def run():
            time.sleep(20); self._fetch_news()
            while True: time.sleep(300); self._fetch_news(); self._check_news_guard()
        threading.Thread(target=run,daemon=True).start()

    def _retrain_check(self):
        def run():
            time.sleep(5)
            ep=os.path.join(BASE,"ensemble.pkl")
            if os.path.exists(ep):
                age=(time.time()-os.path.getmtime(ep))/86400
                if age>7: self._status(f"⚠ Models {age:.0f} days old — retrain recommended")
        threading.Thread(target=run,daemon=True).start()

    def _refresh_procs(self):
        lines=[]; dead=[]
        for name,proc in self.active_procs.items():
            alive=proc.poll() is None
            lines.append(f"  {'●' if alive else '○'} {name}  PID:{proc.pid}")
            if not alive: dead.append(name)
        try:
            self.proc_box.config(state="normal"); self.proc_box.delete("1.0","end")
            self.proc_box.insert("end","\n".join(lines) if lines else "  No processes running")
            self.proc_box.config(state="disabled")
        except: pass

    # ── Run helpers ───────────────────────────────────────────────
    def _run_console(self,script,args,title):
        sp=os.path.join(BASE,script)
        if not os.path.exists(sp): messagebox.showerror("Not found",f"{sp} not found"); return
        cmd=f'start "{title}" cmd /k python "{sp}" {args}'
        proc=subprocess.Popen(cmd,shell=True,cwd=BASE)
        self.active_procs[title]=proc; self._status(f"Launched: {title}")

    def _run_popup(self,script,args,title,w=560,h=420):
        sp=os.path.join(BASE,script)
        if not os.path.exists(sp): messagebox.showerror("Not found",f"{sp} not found — download it first"); return
        win=tk.Toplevel(self.root); win.title(title); win.configure(bg=BG1); win.geometry(f"{w}x{h}")
        oh,ih=bordered(win); oh.pack(fill="x",padx=6,pady=(6,0))
        tk.Label(ih,text=f" {title}",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FS,wrap="word",relief="flat")
        out.pack(fill="both",expand=True,padx=6,pady=4)
        bf=tk.Frame(win,bg=BG1); bf.pack(fill="x",padx=6,pady=(0,6))
        stopped=[False]; proc_r=[None]
        def stop(): stopped[0]=True; (proc_r[0].kill() if proc_r[0] else None); win.destroy()
        tk.Button(bf,text=" Stop & Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=stop).pack(side="right",padx=4)
        def run():
            import subprocess as sp2
            cmd2=[sys.executable,os.path.join(BASE,script)]
            if args: cmd2.append(args)
            p=sp2.Popen(cmd2,stdout=sp2.PIPE,stderr=sp2.STDOUT,text=True,cwd=BASE,encoding="utf-8",errors="ignore")
            proc_r[0]=p
            for ln in p.stdout:
                if stopped[0]: break
                try: out.insert("end",ln); out.see("end"); win.update_idletasks()
                except: break
        threading.Thread(target=run,daemon=True).start()

    # ── Trading ───────────────────────────────────────────────────
    def _trade_prompt(self):
        win=tk.Toplevel(self.root); win.title("Choose Trading Mode"); win.configure(bg=BG1); win.geometry("520x400")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=(8,4))
        tk.Label(ih,text=" SELECT TRADING MODE",fg=RED_HI,bg=BG2,font=FT).pack(anchor="w",padx=10,pady=8)
        def mbtn(label,sub,color,cmd):
            f=tk.Frame(win,bg=color,highlightbackground=WHITE,highlightthickness=1); f.pack(fill="x",padx=10,pady=5)
            tk.Button(f,text=f"  {label}",fg=WHITE,bg=color,font=FH,relief="flat",cursor="hand2",
                command=lambda:(win.destroy(),cmd())).pack(fill="x",padx=4,pady=8)
            tk.Label(f,text=f"  {sub}",fg=WHITE,bg=color,font=FS).pack(anchor="w",padx=4,pady=(0,6))
        mbtn("📋  Paper Trading","Demo account 62111276 — no real money","#0a2a14",
             lambda:self._run_console("trade_monitor.py","paper","Paper Monitor"))
        mbtn("🧪  Strategy Tester","Backtesting — $1,000 simulated deposit","#0a1830",
             lambda:self._run_console("trade_monitor.py","backtest","Backtest Monitor"))
        mbtn("🔴  LIVE Trading — REAL MONEY","Requires double confirmation","#2a0808",
             self._live_confirm)

    def _live_confirm(self):
        if messagebox.askyesno("CONFIRM","This is REAL MONEY. Are you sure?"):
            if messagebox.askyesno("FINAL CHECK","FINAL: Start LIVE trading with real money?"):
                self._run_console("trade_monitor.py","live","LIVE Monitor"); self._status("LIVE TRADING STARTED")

    def _kill_all(self):
        if not messagebox.askyesno("Kill All","Stop ALL processes?"): return
        for _,p in self.active_procs.items():
            try: p.kill()
            except: pass
        subprocess.run(["taskkill","/F","/IM","terminal64.exe"],capture_output=True)
        self.active_procs.clear(); self._status("All killed")

    def _open_graph(self,sym):
        sp=os.path.join(BASE,"live_graph.py")
        if not os.path.exists(sp): messagebox.showerror("Not found","live_graph.py not found"); return
        proc=subprocess.Popen([sys.executable,sp,sym],cwd=BASE)
        self.active_procs[f"Graph-{sym}"]=proc; self._status(f"Graph: {sym}")

    def _backtest(self):
        self._run_console("trade_monitor.py","backtest","Backtest Monitor")
        subprocess.Popen(["start","","mt5.exe"],shell=True)

    # ── Risk & Prop Firm ──────────────────────────────────────────
    def _show_risk_status(self):
        win=tk.Toplevel(self.root); win.title("Risk Status"); win.configure(bg=BG1); win.geometry("500x380")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Current Risk Status",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,relief="flat",wrap="word")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        def run():
            sys.path.insert(0,BASE)
            try:
                from risk_manager import calculate_risk, get_session
                ec_path=os.path.join(BASE,"ensemble.pkl")
                conf=0.6; kelly=0.01
                if os.path.exists(ec_path):
                    import joblib; ec=joblib.load(ec_path)
                    conf=ec.get("ensemble_accuracy",0.6); kelly=ec.get("kelly_fraction",0.01)
                r=calculate_risk(current_balance=100,ml_confidence=conf,
                                 smc_score=50,regime="TREND_UP",news_ok=True)
                out.insert("end",f"\n  Risk %      : {r['risk_pct']:.3f}%\n")
                out.insert("end",f"  Mode        : {r['mode']}\n")
                out.insert("end",f"  God Mode    : {'YES ⚡' if r['god_mode'] else 'No'}\n")
                out.insert("end",f"  Session     : {r['session']}\n")
                out.insert("end",f"  Trade OK    : {r['trade_allowed']}\n\n")
                out.insert("end",f"  Multipliers:\n")
                out.insert("end",f"    Confidence  : {r['conf_mult']:.2f}x\n")
                out.insert("end",f"    Regime      : {r['regime_mult']:.2f}x\n")
                out.insert("end",f"    Equity Curve: {r['ec_mult']:.2f}x\n")
                out.insert("end",f"    Volatility  : {r['vol_mult']:.2f}x\n")
                out.insert("end",f"    Session     : {r['session_mult']:.2f}x\n")
                out.insert("end",f"    News Guard  : {r['news_mult']:.2f}x\n\n")
                out.insert("end",f"  ATR %       : {r['atr_pct']:.3f}%\n")
                out.insert("end",f"  Prop Firm   : {'ON' if r['prop_firm_mode'] else 'OFF'}\n")
            except Exception as e:
                out.insert("end",f"  Error: {e}\n  Install packages and train models first.\n")
        threading.Thread(target=run,daemon=True).start()

    def _toggle_prop_firm(self):
        settings_path=os.path.join(BASE,"bot_settings.json")
        try:
            with open(settings_path) as f: s=json.load(f)
        except: s={}
        current=s.get("prop_firm_mode",False)
        s["prop_firm_mode"]=not current
        with open(settings_path,"w") as f: json.dump(s,f,indent=2)
        state="ON" if s["prop_firm_mode"] else "OFF"
        self.mode_lbl.config(text=f"  PROP FIRM MODE: {state}  " if s["prop_firm_mode"] else "")
        messagebox.showinfo("Prop Firm Mode",
            f"Prop Firm Mode: {state}\n\n"
            f"{'ENABLED: 0.5% risk, 3% daily DD limit, 6% total DD, 8% profit target' if s['prop_firm_mode'] else 'DISABLED: back to normal risk settings'}")
        self._status(f"Prop firm mode: {state}")

    def _show_prop_progress(self):
        win=tk.Toplevel(self.root); win.title("Prop Firm Progress"); win.configure(bg=BG1); win.geometry("500x360")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Prop Firm Challenge Progress",fg=GOLD,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,relief="flat",wrap="word")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        def run():
            sys.path.insert(0,BASE)
            try:
                from risk_manager import load_settings
                s=load_settings(); bal=s.get("current_balance",100); start=s.get("starting_balance",100)
                target=s.get("prop_profit_target",8); dd_lim=s.get("prop_total_dd_limit",6)
                profit_pct=(bal-start)/start*100
                dd_pct=max(0,(start-bal)/start*100)
                progress=min(profit_pct/target*100,100) if target>0 else 0
                filled=int(30*progress/100); bar="█"*filled+"░"*(30-filled)
                out.insert("end",f"\n  Challenge Target : +{target:.1f}%\n")
                out.insert("end",f"  DD Limit         : -{dd_lim:.1f}%\n\n")
                out.insert("end",f"  Progress: [{bar}] {progress:.1f}%\n")
                out.insert("end",f"  Profit  : {profit_pct:+.2f}%  (£{bal-start:+.2f})\n")
                out.insert("end",f"  DD now  : {dd_pct:.2f}%\n")
                out.insert("end",f"  Balance : £{bal:.2f}\n\n")
                if profit_pct>=target:
                    out.insert("end","  ★ TARGET REACHED — STOP AND SUBMIT CHALLENGE ★\n")
                elif dd_pct>=dd_lim:
                    out.insert("end","  ✗ DD LIMIT BREACHED — CHALLENGE FAILED\n")
                elif (dd_lim-dd_pct)<1.0:
                    out.insert("end",f"  ⚠ Only {dd_lim-dd_pct:.2f}% DD buffer left — REDUCE RISK\n")
                out.insert("end",f"\n  Mode: {'PROP FIRM' if s.get('prop_firm_mode') else 'Normal (Prop Firm OFF)'}\n")
            except Exception as e:
                out.insert("end",f"  Error: {e}\n")
        threading.Thread(target=run,daemon=True).start()

    def _show_compounding(self):
        win=tk.Toplevel(self.root); win.title("Compounding Calculator"); win.configure(bg=BG1); win.geometry("480x360")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Compounding Projections",fg=GOLD,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,relief="flat",wrap="word")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        try:
            s=json.load(open(os.path.join(BASE,"bot_settings.json"))) if os.path.exists(os.path.join(BASE,"bot_settings.json")) else {}
            bal=s.get("current_balance",100)
        except: bal=100
        out.insert("end",f"\n  Starting balance: £{bal:.2f}\n\n")
        out.insert("end",f"  {'Days':>6}  {'1.0%/day':>10}  {'1.5%/day':>10}  {'2.0%/day':>10}  {'2.5%/day':>10}\n")
        out.insert("end","  " + "─"*52 + "\n")
        for d in [7,14,30,60,90,180]:
            v=[bal*(r**d) for r in [1.01,1.015,1.02,1.025]]
            out.insert("end",f"  {d:>6}d  £{v[0]:>8.2f}  £{v[1]:>8.2f}  £{v[2]:>8.2f}  £{v[3]:>8.2f}\n")
        out.insert("end","\n  Goal: pass funded challenge (£25k-£100k account)\n")
        out.insert("end","  Then trade THEIR money and keep 70-90% of profits.\n")
        out.config(state="disabled")

    def _show_session(self):
        win=tk.Toplevel(self.root); win.title("Session & Volatility"); win.configure(bg=BG1); win.geometry("460x340")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Session & Volatility Targeting",fg=GOLD,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,relief="flat",wrap="word")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        def run():
            sys.path.insert(0,BASE)
            try:
                from risk_manager import get_session, get_current_atr_pct, is_sunday_open
                import datetime
                session,mode=get_session(); atr=get_current_atr_pct()
                out.insert("end",f"\n  Current Session : {session}\n")
                out.insert("end",f"  Recommended Mode: {mode}\n")
                out.insert("end",f"  UTC Time        : {datetime.datetime.utcnow().strftime('%H:%M:%S')}\n\n")
                out.insert("end",f"  Gold ATR (1h)   : {atr:.3f}%\n")
                out.insert("end",f"  Vol Target      : 1% gold move = 1% portfolio\n\n")
                out.insert("end","  Session Schedule (UTC):\n")
                out.insert("end","    Asian Session    : 00:00 - 07:00  (sniper mode)\n")
                out.insert("end","    London Open      : 07:00 - 12:00  (swing mode)\n")
                out.insert("end","    London + NY      : 12:00 - 16:00  (swing mode)\n")
                out.insert("end","    NY Close         : 16:00 - 21:00  (sniper mode)\n")
                out.insert("end","    Off Hours        : 21:00 - 00:00  (avoid)\n\n")
                out.insert("end",f"  Sunday Open     : {'YES — check gap!' if is_sunday_open() else 'No'}\n")
            except Exception as e: out.insert("end",f"  Error: {e}\n")
        threading.Thread(target=run,daemon=True).start()

    # ── News ──────────────────────────────────────────────────────
    def _fetch_news(self):
        def run():
            try:
                sys.path.insert(0,BASE)
                import news_sentiment as ns; result=ns.score_articles()
                self._refresh_news_display(result)
            except Exception as e:
                try: self.news_lbl.config(text=f"  News error: {e}")
                except: pass
        threading.Thread(target=run,daemon=True).start()

    def _check_news_guard(self):
        def run():
            try:
                sys.path.insert(0,BASE)
                import news_guard as ng; result=ng.get_news_guard_status()
                self._refresh_guard_display(result)
            except Exception as e:
                try: self.guard_lbl.config(text=f"  Guard error: {e}")
                except: pass
        threading.Thread(target=run,daemon=True).start()

    def _refresh_news_display(self, result=None):
        if result is None:
            try:
                with open(os.path.join(BASE,"news_cache.json")) as f: result=json.load(f)
            except: return
        try:
            sig=result.get("signal","?"); score=result.get("score",0); n=result.get("articles_analysed",0)
            col=GREEN if sig=="BUY" else (RED_HI if sig=="SELL" else GOLD)
            ts=result.get("timestamp","")[:16]
            self.news_lbl.config(text=f"  Sentiment: {sig}  Score: {score:+.3f}  |  {n} articles  |  {ts} UTC",fg=col)
            arts=result.get("top_articles",[])[:4]
            self.news_box.config(state="normal"); self.news_box.delete("1.0","end")
            for a in arts: self.news_box.insert("end",f"  [{a['source']:8s}] {a['score']:+.2f}  {a['title'][:62]}\n")
            self.news_box.config(state="disabled")
        except: pass

    def _refresh_guard_display(self, result=None):
        if result is None:
            try:
                with open(os.path.join(BASE,"news_guard_status.json")) as f: result=json.load(f)
            except: return
        try:
            ok=result.get("trading_ok",True); rm=result.get("risk_multiplier",1.0)
            bl=result.get("blackout_reason",""); dc=result.get("dxy_decoupling",False)
            col=GREEN if ok else RED_HI
            txt=f"  Guard: {'OK' if ok else 'BLOCKED'}  Risk: {rm:.1f}x"
            if bl and bl!="Clear": txt+=f"  |  {bl}"
            if dc: txt+=f"  |  DXY DECOUPLE"
            self.guard_lbl.config(text=txt,fg=col)
        except: pass

    def _show_macro_overview(self):
        win=tk.Toplevel(self.root); win.title("Macro Overview"); win.configure(bg=BG1); win.geometry("560x480")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Macro Overview — All Guards",fg=BLUE,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FS,relief="flat",wrap="word")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        def run():
            for fname,label in [("regime_cache.json","REGIME"),("news_guard_status.json","NEWS GUARD"),
                                 ("correlation_status.json","CORRELATION"),("current_signal.json","AI SIGNAL"),
                                 ("smc_cache.json","SMC")]:
                try:
                    with open(os.path.join(BASE,fname)) as f: d=json.load(f)
                    out.insert("end",f"\n  ── {label} ──\n")
                    for k,v in list(d.items())[:6]:
                        out.insert("end",f"    {k}: {v}\n")
                except: out.insert("end",f"\n  ── {label}: not loaded ──\n")
            out.see("1.0")
        threading.Thread(target=run,daemon=True).start()

    # ── Analysis ──────────────────────────────────────────────────
    def _run_regime(self):
        win=tk.Toplevel(self.root); win.title("Regime"); win.configure(bg=BG1); win.geometry("520x360")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Market Regime",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,relief="flat",wrap="word"); out.pack(fill="both",expand=True,padx=8,pady=4)
        def run():
            try:
                sys.path.insert(0,BASE)
                from market_regime import get_current_regime
                r=get_current_regime()
                out.insert("end",f"\n  Regime  : {r.get('regime')}\n")
                out.insert("end",f"  ADX     : {r.get('adx',0):.1f}\n")
                out.insert("end",f"  BBW     : {r.get('bbw',0):.4f}\n")
                out.insert("end",f"  Hurst   : {r.get('hurst',0):.3f}\n")
                out.insert("end",f"  Vol bias: {r.get('volume_bias',0):+d}\n")
                out.insert("end",f"  Risk x  : {r.get('risk_mult',0):.2f}\n")
                out.insert("end",f"  Long OK : {r.get('allows_long')}\n")
                out.insert("end",f"  Short OK: {r.get('allows_short')}\n")
            except Exception as e: out.insert("end",f"  Error: {e}\n")
        threading.Thread(target=run,daemon=True).start()

    def _show_signal(self):
        sp=os.path.join(BASE,"current_signal.json")
        win=tk.Toplevel(self.root); win.title("AI Signal"); win.configure(bg=BG1); win.geometry("460x320")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Current AI Signal",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,relief="flat",wrap="word"); out.pack(fill="both",expand=True,padx=8,pady=4)
        try:
            with open(sp) as f: d=json.load(f)
            out.insert("end",f"\n  Signal      : {d.get('signal','?')}\n")
            out.insert("end",f"  Confidence  : {d.get('confidence',0)*100:.1f}%\n")
            out.insert("end",f"  Kelly Risk  : {d.get('kelly_fraction',0)*100:.3f}%\n")
            out.insert("end",f"  Sized Risk  : {d.get('sized_risk_pct',0):.3f}%\n\n")
            out.insert("end",f"  XGB : {d.get('xgb_p',0)*100:.1f}%\n")
            out.insert("end",f"  LGB : {d.get('lgb_p',0)*100:.1f}%\n")
            out.insert("end",f"  GB  : {d.get('gb_p',0)*100:.1f}%\n\n")
            out.insert("end",f"  Generated: {d.get('timestamp','?')}\n")
        except: out.insert("end","  No signal file — train models first.\n")
        out.config(state="disabled")

    def _show_signal_explanation(self):
        win=tk.Toplevel(self.root); win.title("Signal Explanation"); win.configure(bg=BG1); win.geometry("560,440")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Why This Signal?",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FS,relief="flat",wrap="word"); out.pack(fill="both",expand=True,padx=8,pady=4)
        def run():
            reasons=[]
            for fname,key,label in [
                ("current_signal.json","signal","AI Signal"),
                ("regime_cache.json","regime","Regime"),
                ("smc_cache.json","reasons","SMC"),
                ("news_guard_status.json","blackout_reason","News Guard"),
                ("correlation_status.json","warnings","Correlation"),
            ]:
                try:
                    with open(os.path.join(BASE,fname)) as f: d=json.load(f)
                    val=d.get(key,"?")
                    if isinstance(val,list): reasons.extend([f"  • {v}" for v in val[:3]])
                    else: reasons.append(f"  • {label}: {val}")
                except: reasons.append(f"  • {label}: not available")
            out.insert("end","\n  Full reasoning for current signal:\n\n")
            for r in reasons: out.insert("end",r+"\n")
        threading.Thread(target=run,daemon=True).start()

    def _run_analytics(self):
        sp=os.path.join(BASE,"analytics.py")
        if os.path.exists(sp): self._run_popup("analytics.py","","Analytics",560,420)
        else: messagebox.showinfo("Analytics","analytics.py not found.")

    def _open_exports(self):
        ep=os.path.join(BASE,"exports"); os.makedirs(ep,exist_ok=True); os.startfile(ep)

    # ── Training ──────────────────────────────────────────────────
    def _load_accuracy(self):
        try:
            import joblib
            ec=joblib.load(os.path.join(BASE,"ensemble.pkl"))
            acc=ec.get("ensemble_accuracy",0)*100
            self.acc_lbl.config(text=f"Ensemble: {acc:.1f}%  ",fg=GREY_MID)
        except: pass

    def _show_accuracy(self):
        win=tk.Toplevel(self.root); win.title("Model Accuracy"); win.configure(bg=BG1); win.geometry("500x380")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Model Accuracy",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,relief="flat",wrap="word"); out.pack(fill="both",expand=True,padx=8,pady=4)
        try:
            import joblib; ec=joblib.load(os.path.join(BASE,"ensemble.pkl"))
            for nm,ak,wk in [("XGBoost","xgb_accuracy","xgb_weight"),("LightGBM","lgb_accuracy","lgb_weight"),
                             ("GradBoost","gb_accuracy","gb_weight"),("LSTM","lstm_accuracy",None)]:
                ac=ec.get(ak,0); wt=ec.get(wk,0) if wk else 0
                bar="▓"*int(ac*20)
                out.insert("end",f"  {nm:12s}  {ac*100:5.1f}%  wt:{wt*100:4.1f}%  {bar}\n")
            out.insert("end",f"\n  Ensemble  : {ec.get('ensemble_accuracy',0)*100:.1f}%\n")
            out.insert("end",f"  Kelly     : {ec.get('kelly_fraction',0)*100:.3f}%\n")
            out.insert("end",f"  Win rate  : {ec.get('win_rate',0)*100:.1f}%\n")
            age=(time.time()-os.path.getmtime(os.path.join(BASE,"ensemble.pkl")))/86400
            out.insert("end",f"  Model age : {age:.1f} days\n")
            out.insert("end",f"  Trained at: {ec.get('trained_at','?')}\n")
        except: out.insert("end","  No model data — train first.\n")
        out.config(state="disabled")

    def _install_packages(self):
        pkgs=["pandas","numpy","scikit-learn","xgboost","lightgbm",
              "tensorflow","yfinance","feedparser","vaderSentiment",
              "joblib","requests","matplotlib"]
        win=tk.Toplevel(self.root); win.title("Install"); win.configure(bg=BG1); win.geometry("500x360")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Installing Packages",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FS,relief="flat",wrap="word"); out.pack(fill="both",expand=True,padx=8,pady=4)
        def run():
            for p in pkgs:
                out.insert("end",f"  {p}..."); out.see("end"); win.update()
                r=subprocess.run([sys.executable,"-m","pip","install",p,"-q"],capture_output=True,text=True)
                out.insert("end"," OK\n" if r.returncode==0 else f" FAIL\n")
                out.see("end")
            out.insert("end","\n  Done.\n")
        threading.Thread(target=run,daemon=True).start()

    # ── Kelly ──────────────────────────────────────────────────────
    def _show_kelly(self):
        win=tk.Toplevel(self.root); win.title("Kelly Criterion"); win.configure(bg=BG1); win.geometry("460x320")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Kelly Criterion",fg=GOLD,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,relief="flat",wrap="word"); out.pack(fill="both",expand=True,padx=8,pady=4)
        try:
            import joblib; ec=joblib.load(os.path.join(BASE,"ensemble.pkl"))
            k=ec.get("kelly_fraction",0)*100; wr=ec.get("win_rate",0)*100
            aw=ec.get("avg_win",0)*100; al=ec.get("avg_loss",0)*100
            out.insert("end",f"\n  Kelly Fraction   : {k:.3f}%\n")
            out.insert("end",f"  Half-Kelly used  : {k/2:.3f}%\n\n")
            out.insert("end",f"  Win Rate         : {wr:.1f}%\n")
            out.insert("end",f"  Avg Win          : {aw:.4f}%\n")
            out.insert("end",f"  Avg Loss         : {al:.4f}%\n\n")
            out.insert("end","  Confidence scaling:\n")
            out.insert("end",f"    >88% + SMC + regime  →  GOD MODE ({8.0:.1f}% risk)\n")
            out.insert("end",f"    >80% confidence      →  {k*2:.3f}% (2x Kelly)\n")
            out.insert("end",f"    >70% confidence      →  {k*1.5:.3f}% (1.5x Kelly)\n")
            out.insert("end",f"    >60% confidence      →  {k:.3f}% (1x Kelly)\n")
            out.insert("end",f"    <60% confidence      →  {k*0.5:.3f}% (0.5x Kelly)\n")
        except: out.insert("end","  No model data — train first.\n")
        out.config(state="disabled")

    # ── Testing ───────────────────────────────────────────────────
    def _test_telegram(self):
        try:
            import requests
            s=json.load(open(os.path.join(BASE,"bot_settings.json")))
            tok=s.get("TelegramToken",""); cid=s.get("TelegramChatID","")
            if not tok or not cid: messagebox.showerror("Error","Set token and chat ID in Settings"); return
            r=requests.get(f"https://api.telegram.org/bot{tok}/sendMessage",
                params={"chat_id":cid,"text":"AlmostFinishedBot v2 — test message ✓"},timeout=10)
            messagebox.showinfo("Telegram","Sent!" if r.status_code==200 else f"Failed: {r.text[:100]}")
        except Exception as e: messagebox.showerror("Telegram",str(e))

    def _get_chat_id(self):
        win=tk.Toplevel(self.root); win.title("Chat ID"); win.configure(bg=BG1); win.geometry("460x340")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Get Telegram Chat ID",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        tk.Label(win,text="  1. Send your bot any message in Telegram\n  2. Click Auto-Detect",
                 fg=GREY_LT,bg=BG1,font=FB,justify="left").pack(anchor="w",padx=12,pady=8)
        found=tk.StringVar(value=""); id_lbl=tk.Label(win,textvariable=found,fg=WHITE,bg=BG0,font=("Consolas",16,"bold"))
        id_lbl.pack(fill="x",padx=12,pady=8)
        def detect():
            try:
                import requests
                s=json.load(open(os.path.join(BASE,"bot_settings.json")))
                r=requests.get(f"https://api.telegram.org/bot{s.get('TelegramToken','')}/getUpdates",timeout=10).json()
                results=r.get("result",[])
                if results:
                    cid=str(results[-1]["message"]["from"]["id"]); found.set(f"  Chat ID: {cid}"); win._cid=cid; save_btn.config(state="normal",bg=GREEN)
                else: found.set("  No messages — send bot a message first")
            except Exception as e: found.set(f"  Error: {e}")
        def save():
            try:
                s=json.load(open(os.path.join(BASE,"bot_settings.json")))
                s["TelegramChatID"]=win._cid
                json.dump(s,open(os.path.join(BASE,"bot_settings.json"),"w"),indent=2)
                messagebox.showinfo("Saved",f"Chat ID {win._cid} saved!"); win.destroy()
            except Exception as e: messagebox.showerror("Error",str(e))
        tk.Button(win,text=" Auto-Detect ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=detect).pack(pady=6)
        save_btn=tk.Button(win,text=" Save ",bg=GREY_DIM,fg=WHITE,font=FBB,relief="flat",state="disabled",command=save); save_btn.pack(pady=4)

    def _health_check(self):
        win=tk.Toplevel(self.root); win.title("Health Check"); win.configure(bg=BG1); win.geometry("540x520")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" System Health Check",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FS,relief="flat",wrap="word"); out.pack(fill="both",expand=True,padx=8,pady=4)
        out.tag_config("ok",foreground=GREEN); out.tag_config("miss",foreground=RED_HI); out.tag_config("warn",foreground=GOLD)
        def run():
            files=[("xgb_model.pkl","Model"),("lgb_model.pkl","Model"),("gb_model.pkl","Model"),
                   ("scaler.pkl","Scaler"),("ensemble.pkl","Ensemble"),
                   ("market_regime.py","Script"),("smc_detector.py","Script"),
                   ("news_guard.py","Script"),("correlation_guard.py","Script"),
                   ("risk_manager.py","Script"),("train_models.py","Script"),
                   ("live_graph.py","Script"),("trade_monitor.py","Script"),
                   ("news_sentiment.py","Script"),("AlmostFinishedBot_ControlCenter.py","Script")]
            out.insert("end","\n  Files:\n")
            for fn,ft in files:
                fp=os.path.join(BASE,fn)
                if os.path.exists(fp):
                    age=(time.time()-os.path.getmtime(fp))/86400
                    tag="warn" if (age>7 and ft=="Model") else "ok"
                    out.insert("end",f"  {'WARN' if tag=='warn' else 'OK  '}  {fn} ({age:.0f}d)\n",tag)
                else: out.insert("end",f"  MISS  {fn}\n","miss")
            out.insert("end","\n  Packages:\n")
            for p in ["pandas","numpy","sklearn","xgboost","lightgbm","tensorflow","yfinance","feedparser","vaderSentiment","joblib","requests","matplotlib"]:
                try: __import__(p); out.insert("end",f"  OK    {p}\n","ok")
                except: out.insert("end",f"  MISS  {p}\n","miss")
        threading.Thread(target=run,daemon=True).start()

    def _check_sunday_gap(self):
        def run():
            sys.path.insert(0,BASE)
            try:
                from risk_manager import check_sunday_gap
                detected,gap_pct,direction=check_sunday_gap()
                if detected: messagebox.showinfo("Sunday Gap",f"Gap detected!\nDirection: {direction}\nSize: {gap_pct:+.3f}%\n\nConsider gap-fill or continuation trade on first 15M candle.")
                else: messagebox.showinfo("Sunday Gap","No significant Sunday gap detected at this time.")
            except Exception as e: messagebox.showerror("Error",str(e))
        threading.Thread(target=run,daemon=True).start()

    def _check_session(self):
        sys.path.insert(0,BASE)
        try:
            from risk_manager import get_session
            import datetime
            session,mode=get_session()
            utc=datetime.datetime.utcnow().strftime("%H:%M UTC")
            messagebox.showinfo("Session",f"Current Time  : {utc}\nSession       : {session}\nRecommended   : {mode} mode\n\nAsian (00-07 UTC): Best for small accounts\nLondon+NY (07-16 UTC): Biggest moves")
        except Exception as e: messagebox.showerror("Error",str(e))

    # ── Tools ─────────────────────────────────────────────────────
    def _open_mt5_experts(self):
        p=os.path.join(os.environ.get("APPDATA",""),"MetaQuotes","Terminal","73B7A2420D6397DFF9014A20F1201F97","MQL5","Experts")
        os.startfile(p) if os.path.exists(p) else messagebox.showerror("Not found","MT5 Experts not found")

    def _open_mt5_logs(self):
        p=os.path.join(os.environ.get("APPDATA",""),"MetaQuotes","Terminal","73B7A2420D6397DFF9014A20F1201F97","MQL5","Logs")
        os.startfile(p) if os.path.exists(p) else messagebox.showerror("Not found","MT5 Logs not found")

    def _open_settings(self):
        cfg=os.path.join(BASE,"bot_settings.json")
        DEFAULT_SETTINGS = {
            "mode":"normal","base_risk_pct":1.0,"god_mode_max_risk":8.0,
            "god_mode_threshold":0.88,"prop_firm_mode":False,
            "prop_daily_dd_limit":3.0,"prop_total_dd_limit":6.0,
            "prop_profit_target":8.0,"equity_halve_trigger":5.0,
            "equity_restore_pct":20.0,"vol_target_pct":1.0,
            "TelegramToken":"","TelegramChatID":"838489368",
            "RiskPercent":1.0,"MaxTrades":2,"MaxDrawdownPct":10.0,
        }
        data={**DEFAULT_SETTINGS}
        if os.path.exists(cfg):
            try:
                with open(cfg) as f: loaded=json.load(f)
                data.update(loaded)
            except: pass
        win=tk.Toplevel(self.root); win.title("Settings"); win.configure(bg=BG1); win.geometry("500x500")
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Bot Settings",fg=RED_HI,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        canvas2=tk.Canvas(win,bg=BG1,highlightthickness=0); canvas2.pack(fill="both",expand=True,padx=8,pady=4)
        vsb2=tk.Scrollbar(win,orient="vertical",command=canvas2.yview); vsb2.pack(side="right",fill="y")
        canvas2.configure(yscrollcommand=vsb2.set)
        sf2=tk.Frame(canvas2,bg=BG1); canvas2.create_window((0,0),window=sf2,anchor="nw")
        sf2.bind("<Configure>",lambda e:canvas2.configure(scrollregion=canvas2.bbox("all")))
        fields={}
        skip={"compound_auto","current_balance","peak_balance","starting_balance"}
        bool_fields={"UseTrailing","UseKelly","UseRegime","UseSMC","SwingMode","prop_firm_mode","compound_auto"}
        bool_vars={}
        for key,val in data.items():
            if key in skip: continue
            rf=tk.Frame(sf2,bg=BG1); rf.pack(fill="x",padx=4,pady=2)
            tk.Label(rf,text=f"  {key}:",fg=GREY_LT,bg=BG1,font=FB,width=24,anchor="w").pack(side="left")
            if key in bool_fields or isinstance(val,bool):
                v=tk.BooleanVar(value=bool(val)); bool_vars[key]=v
                tk.Checkbutton(rf,variable=v,bg=BG1,fg=GREY_LT,selectcolor=BG0,activebackground=BG1).pack(side="left")
            else:
                oe,ie=bordered(rf,pady=0); oe.pack(side="left",fill="x",expand=True)
                e=tk.Entry(ie,bg=BG0,fg=WHITE,font=FB,relief="flat",insertbackground=WHITE)
                e.insert(0,str(val)); e.pack(fill="x",padx=4,pady=2); fields[key]=e
        def save():
            new=dict(data)
            for k,e in fields.items():
                try: new[k]=float(e.get()) if k not in {"TelegramToken","TelegramChatID","mode"} else e.get()
                except: new[k]=e.get()
            for k,v in bool_vars.items(): new[k]=v.get()
            with open(cfg,"w") as f: json.dump(new,f,indent=2)
            self._status("Settings saved"); win.destroy()
        tk.Button(win,text=" Save Settings ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=save).pack(pady=10)

    def _reload(self): os.execv(sys.executable,[sys.executable]+sys.argv)

def main():
    root=tk.Tk(); App(root); root.mainloop()

if __name__ == "__main__":
    main()
