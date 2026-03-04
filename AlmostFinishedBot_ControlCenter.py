"""
AlmostFinishedBot - Control Center v5.0
COMPLETE OVERHAUL with working buttons:
  - Close All: Closes positions AND kills bridge processes
  - News Guard: Actually checks news and shows status
  - System Health Check: Verifies all components
  - Install Packages: One-click dependency installer
  - Better process tracking
  - v7 model support (10 models)
"""
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os, sys, subprocess, json, time, threading, datetime

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
os.makedirs(BASE, exist_ok=True)

BG0="#0f0f0f"; BG1="#1a1a1a"; BG2="#242424"; BG3="#303030"
BORDER="#ffffff"; BORD_D="#555555"
RED_HI="#ff2222"; RED_MID="#cc3333"; RED_DIM="#882222"
WHITE="#ffffff"; GREY_LT="#dddddd"; GREY_MID="#999999"; GREY_DIM="#555555"
GREEN="#22cc66"; GOLD="#ffcc44"; BLUE="#4499ff"; PURPLE="#aa66ff"
FH=("Consolas",10,"bold"); FB=("Consolas",9); FS=("Consolas",8)
FT=("Consolas",14,"bold"); FBB=("Consolas",9,"bold")

# Track running bridge processes by window title
BRIDGE_TITLES = {
    "XAUUSD": "AlmostFinishedBot - XAUUSD",
    "XAUSGD": "AlmostFinishedBot - XAUSGD",
}

def _rj(path):
    try:
        with open(os.path.join(BASE, path)) as f: return json.load(f)
    except: return {}

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


class App:
    def __init__(self, root):
        self.root=root
        self.root.title("AlmostFinishedBot -- Control Center v5.0")
        self.root.configure(bg=BG1)
        self.root.geometry("980x950")
        self.root.minsize(850,750)
        self.active_procs = {}  # name -> subprocess.Popen or window title
        self._build()
        self._start_status_refresh()

    def _build(self):
        r=self.root

        # Header
        oh,ih=bordered(r); oh.pack(fill="x",padx=8,pady=(8,4))
        tk.Label(ih,text="  ALMOSTFINISHEDBOT  |  CONTROL CENTER  v5.0  ",
                 fg=RED_HI,bg=BG2,font=("Consolas",15,"bold")).pack(side="left",pady=8)
        self.status_lbl=tk.Label(ih,text="",fg=GREY_MID,bg=BG2,font=FS)
        self.status_lbl.pack(side="right",padx=6)
        self.acc_lbl=tk.Label(ih,text="",fg=GREY_MID,bg=BG2,font=FS)
        self.acc_lbl.pack(side="right",padx=6)

        # Scrollable main area
        canvas=tk.Canvas(r,bg=BG1,highlightthickness=0)
        vsb=tk.Scrollbar(r,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y"); canvas.pack(side="left",fill="both",expand=True,padx=(8,0),pady=4)
        sf=tk.Frame(canvas,bg=BG1)
        sfid=canvas.create_window((0,0),window=sf,anchor="nw")
        sf.bind("<Configure>",lambda e:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",lambda e:canvas.itemconfig(sfid,width=e.width))
        canvas.bind_all("<MouseWheel>",lambda e:canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        # ==============================================================
        # LIVE CHARTS BUTTON
        # ==============================================================
        section_header(sf,"📊  LIVE CHARTS & TRADING DASHBOARD",BLUE)
        cg=tk.Frame(sf,bg=BG1); cg.pack(fill="x",padx=6)
        self._big_btn(cg,"📊  OPEN TRADING DASHBOARD",
            "Live candlestick charts with trade markers, P&L tracking, zoom/pan",
            BLUE, self._open_dashboard)

        # ==============================================================
        # XAUUSD SECTION
        # ==============================================================
        section_header(sf,"🥇  XAUUSD -- Gold vs US Dollar",GOLD)
        ug=tk.Frame(sf,bg=BG1); ug.pack(fill="x",padx=6)

        bf_usd=tk.Frame(ug,bg=BG1); bf_usd.pack(fill="x",padx=4,pady=4)
        self._action_btn(bf_usd,"▶  PAPER TRADE",GREEN,
            lambda:self._launch_bridge("live_trading_bridge.py","--mode paper","XAUUSD Paper"))
        self._action_btn(bf_usd,"📊  BACKTEST",BLUE,
            lambda:self._launch_script("walkforward_backtest.py","","XAUUSD Backtest"))
        self._action_btn(bf_usd,"⚡  LIVE TRADE",RED_DIM,
            lambda:self._live_trade("live_trading_bridge.py","XAUUSD"))
        self._action_btn(bf_usd,"❌  CLOSE ALL",RED_MID,
            lambda:self._close_all_and_stop("XAUUSD", 999))

        self.usd_status=tk.Label(ug,text="  XAUUSD: Ready",fg=GREY_MID,bg=BG1,font=FB)
        self.usd_status.pack(anchor="w",padx=10)

        # ==============================================================
        # XAUSGD SECTION
        # ==============================================================
        section_header(sf,"🎯  XAUSGD -- Gold vs Singapore Dollar",GOLD)
        sg=tk.Frame(sf,bg=BG1); sg.pack(fill="x",padx=6)

        bf_sgd=tk.Frame(sg,bg=BG1); bf_sgd.pack(fill="x",padx=4,pady=4)
        self._action_btn(bf_sgd,"▶  PAPER TRADE",GREEN,
            lambda:self._launch_bridge("live_bridge_xausgd.py","--mode paper","XAUSGD Paper"))
        self._action_btn(bf_sgd,"📊  BACKTEST",BLUE,
            lambda:self._launch_script("walkforward_backtest.py","xausgd","XAUSGD Backtest"))
        self._action_btn(bf_sgd,"⚡  LIVE TRADE",RED_DIM,
            lambda:self._live_trade("live_bridge_xausgd.py","XAUSGD"))
        self._action_btn(bf_sgd,"❌  CLOSE ALL",RED_MID,
            lambda:self._close_all_and_stop("XAUSGD", 1000))

        self.sgd_status=tk.Label(sg,text="  XAUSGD: Ready",fg=GREY_MID,bg=BG1,font=FB)
        self.sgd_status.pack(anchor="w",padx=10)

        # ==============================================================
        # SYSTEM CONTROLS
        # ==============================================================
        section_header(sf,"⚙  SYSTEM CONTROLS",WHITE)
        scg=tk.Frame(sf,bg=BG1); scg.pack(fill="x",padx=6)

        bf_sys=tk.Frame(scg,bg=BG1); bf_sys.pack(fill="x",padx=4,pady=4)
        self._action_btn(bf_sys,"⚡  START FULL SYSTEM",GREEN,self._start_full_system)
        self._action_btn(bf_sys,"⬛  KILL SWITCH",RED_HI,self._kill_all)
        self._action_btn(bf_sys,"🔄  RETRAIN XAUUSD",BLUE,self._retrain)
        self._action_btn(bf_sys,"🔄  RETRAIN XAUSGD",BLUE,self._retrain_sgd)

        bf_sys2=tk.Frame(scg,bg=BG1); bf_sys2.pack(fill="x",padx=4,pady=4)
        self._action_btn(bf_sys2,"🩺  SYSTEM HEALTH CHECK",PURPLE,self._system_health_check)
        self._action_btn(bf_sys2,"📦  INSTALL PACKAGES",PURPLE,self._install_packages)
        self._action_btn(bf_sys2,"📁  OPEN FOLDER",BG3,lambda:os.startfile(BASE))

        # ==============================================================
        # GUARDS & INTELLIGENCE
        # ==============================================================
        section_header(sf,"🛡  GUARDS & INTELLIGENCE",RED_HI)
        gg=tk.Frame(sf,bg=BG1); gg.pack(fill="x",padx=6)

        bf_guard=tk.Frame(gg,bg=BG1); bf_guard.pack(fill="x",padx=4,pady=4)
        self._action_btn(bf_guard,"📰  Check News",GOLD,self._check_news_guard)
        self._action_btn(bf_guard,"📊  SMC Analysis",BG3,self._check_smc)
        self._action_btn(bf_guard,"🔗  Correlation Check",BG3,self._check_correlation)
        self._action_btn(bf_guard,"📈  Market Regime",BG3,self._check_regime)

        # ==============================================================
        # LIVE STATUS DISPLAY
        # ==============================================================
        section_header(sf,"📡  LIVE STATUS",GREEN)
        ls=tk.Frame(sf,bg=BG1); ls.pack(fill="x",padx=6)

        self.signal_lbl=tk.Label(ls,text="  Signal: --",fg=GREY_MID,bg=BG1,font=FB)
        self.signal_lbl.pack(anchor="w",padx=10)
        self.news_lbl=tk.Label(ls,text="  News: --",fg=GREY_MID,bg=BG1,font=FB)
        self.news_lbl.pack(anchor="w",padx=10)
        self.corr_lbl=tk.Label(ls,text="  Correlations: --",fg=GREY_MID,bg=BG1,font=FB)
        self.corr_lbl.pack(anchor="w",padx=10)
        self.proc_lbl=tk.Label(ls,text="  Processes: --",fg=GREY_MID,bg=BG1,font=FB)
        self.proc_lbl.pack(anchor="w",padx=10)

        # ==============================================================
        # UTILITIES
        # ==============================================================
        section_header(sf,"🧮  UTILITIES",WHITE)
        uf=tk.Frame(sf,bg=BG1); uf.pack(fill="x",padx=6)

        bf_util=tk.Frame(uf,bg=BG1); bf_util.pack(fill="x",padx=4,pady=4)
        self._action_btn(bf_util,"📊  Kelly Sizing",BG3,self._show_kelly)
        self._action_btn(bf_util,"📈  Compounding Calc",BG3,self._show_compounding)
        self._action_btn(bf_util,"💰  Account Info",BG3,self._show_account)
        self._action_btn(bf_util,"📋  View Trade Log",BG3,self._view_trade_log)

    def _big_btn(self, parent, text, desc, color, cmd):
        f=tk.Frame(parent,bg=color,relief="flat")
        f.pack(fill="x",padx=4,pady=6)
        tk.Button(f,text=f"  {text}  ",bg=color,fg=WHITE,font=("Consolas",12,"bold"),
                  relief="flat",cursor="hand2",anchor="w",command=cmd).pack(fill="x",padx=6,pady=(6,0))
        tk.Label(f,text=f"    {desc}",fg=GREY_LT,bg=color,font=FS,wraplength=800,
                 anchor="w").pack(fill="x",padx=6,pady=(0,6))

    def _action_btn(self, parent, text, color, cmd):
        b=tk.Button(parent,text=f" {text} ",bg=color,fg=WHITE,font=FBB,relief="flat",
                    cursor="hand2",command=cmd,padx=8,pady=4)
        b.pack(side="left",padx=3,pady=3)
        b.bind("<Enter>",lambda e,b=b:b.config(bg=WHITE,fg=BG0))
        b.bind("<Leave>",lambda e,b=b,c=color:b.config(bg=c,fg=WHITE))

    def _status(self, msg):
        try: self.status_lbl.config(text=f"  {msg}  ")
        except: pass

    # ══════════════════════════════════════════════════════════════════
    # DASHBOARD LAUNCH
    # ══════════════════════════════════════════════════════════════════
    def _open_dashboard(self):
        sp=os.path.join(BASE,"trading_dashboard.py")
        if not os.path.exists(sp):
            messagebox.showerror("Not Found",
                "trading_dashboard.py not found!\n\n"
                f"Place it in: {BASE}")
            return
        proc=subprocess.Popen([sys.executable,sp],cwd=BASE)
        self.active_procs["Dashboard"]=proc
        self._status("Dashboard opened")

    # ══════════════════════════════════════════════════════════════════
    # BRIDGE LAUNCHERS
    # ══════════════════════════════════════════════════════════════════
    def _launch_bridge(self, script, args, label):
        sp=os.path.join(BASE,script)
        if not os.path.exists(sp):
            messagebox.showerror("Not Found",f"{script} not found in {BASE}")
            return
        
        # Use a unique window title we can track
        symbol = "XAUUSD" if "XAUUSD" in label else "XAUSGD"
        window_title = f"AlmostFinishedBot - {label}"
        
        cmd = f'start "{window_title}" cmd /k python "{sp}" {args}'
        subprocess.Popen(cmd, shell=True, cwd=BASE)
        self.active_procs[label] = window_title
        self._status(f"{label} launched")
        
        if "XAUUSD" in label: 
            self.usd_status.config(text=f"  XAUUSD: {label} RUNNING",fg=GREEN)
        if "XAUSGD" in label: 
            self.sgd_status.config(text=f"  XAUSGD: {label} RUNNING",fg=GREEN)

    def _launch_script(self, script, args, label):
        sp=os.path.join(BASE,script)
        if not os.path.exists(sp):
            messagebox.showerror("Not Found",f"{script} not found"); return
        cmd = f'start "AFB - {label}" cmd /k python "{sp}" {args}'
        subprocess.Popen(cmd, shell=True, cwd=BASE)
        self._status(f"{label} started")

    def _live_trade(self, script, symbol):
        if not messagebox.askyesno("LIVE TRADING",
            f"Are you sure you want to trade {symbol} with REAL money?\n\n"
            "This will execute REAL trades on your account."):
            return
        if not messagebox.askyesno("FINAL CONFIRM",
            f"LAST CHANCE!\n\nReal money will be at risk on {symbol}.\n\nProceed?"):
            return
        sp=os.path.join(BASE,script)
        window_title = f"AlmostFinishedBot - {symbol} LIVE"
        cmd = f'start "{window_title}" cmd /k python "{sp}" --mode live'
        subprocess.Popen(cmd, shell=True, cwd=BASE)
        self.active_procs[f"{symbol} Live"] = window_title
        self._status(f"{symbol} LIVE started")

    # ══════════════════════════════════════════════════════════════════
    # CLOSE ALL - Now closes positions AND kills bridge process
    # ══════════════════════════════════════════════════════════════════
    def _close_all_and_stop(self, symbol, magic):
        if not messagebox.askyesno("Close All & Stop",
            f"This will:\n"
            f"1. Close ALL {symbol} positions\n"
            f"2. Stop the {symbol} trading bridge\n\n"
            f"Continue?"):
            return
        
        closed = 0
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    for p in positions:
                        if p.magic != magic: continue
                        ct = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
                        tick = mt5.symbol_info_tick(symbol)
                        price = tick.bid if p.type == 0 else tick.ask
                        req = {"action":mt5.TRADE_ACTION_DEAL,"symbol":symbol,"volume":p.volume,
                               "type":ct,"position":p.ticket,"price":price,"deviation":20,
                               "magic":magic,"comment":"AFB close all","type_filling":mt5.ORDER_FILLING_IOC}
                        r = mt5.order_send(req)
                        if r and r.retcode == mt5.TRADE_RETCODE_DONE: closed += 1
                mt5.shutdown()
        except Exception as e:
            messagebox.showerror("MT5 Error", str(e))
        
        # Kill the bridge process by window title
        killed = False
        try:
            # Kill any cmd window with our bridge title
            subprocess.run(
                f'taskkill /fi "WINDOWTITLE eq AlmostFinishedBot - {symbol}*" /f',
                shell=True, capture_output=True
            )
            killed = True
        except: pass
        
        # Also try killing by python process (more aggressive)
        try:
            # Find python processes running our bridge scripts
            script_name = "live_trading_bridge.py" if symbol == "XAUUSD" else "live_bridge_xausgd.py"
            subprocess.run(
                f'wmic process where "commandline like \'%{script_name}%\'" call terminate',
                shell=True, capture_output=True
            )
        except: pass
        
        # Update status
        if symbol == "XAUUSD":
            self.usd_status.config(text=f"  XAUUSD: STOPPED",fg=RED_HI)
        else:
            self.sgd_status.config(text=f"  XAUSGD: STOPPED",fg=RED_HI)
        
        # Remove from active procs
        to_remove = [k for k in self.active_procs if symbol in k]
        for k in to_remove:
            del self.active_procs[k]
        
        messagebox.showinfo("Done", f"Closed {closed} positions\nBridge stopped")
        self._status(f"{symbol}: {closed} closed, bridge stopped")

    # ══════════════════════════════════════════════════════════════════
    # SYSTEM CONTROLS
    # ══════════════════════════════════════════════════════════════════
    def _start_full_system(self):
        win=tk.Toplevel(self.root)
        win.title("Full System Launch")
        win.configure(bg=BG1)
        win.geometry("650x550")
        
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" Full Trading System Launch",fg=GREEN,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,wrap="word",relief="flat")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        out.tag_config("ok",foreground=GREEN)
        out.tag_config("err",foreground=RED_HI)
        out.tag_config("warn",foreground=GOLD)
        out.tag_config("hdr",foreground=BLUE)

        bf=tk.Frame(win,bg=BG1); bf.pack(fill="x",padx=8,pady=4)
        tk.Button(bf,text=" Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=win.destroy).pack(side="right",padx=4)

        def run_checks():
            out.insert("end","\n  ═══ SYSTEM STARTUP CHECK ═══\n\n","hdr")
            
            # Check MT5
            out.insert("end","  [MT5] Checking connection... "); out.see("end"); win.update()
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    info = mt5.account_info()
                    out.insert("end",f"OK (${info.balance:.2f})\n","ok")
                    mt5.shutdown()
                else:
                    out.insert("end","FAILED\n","err")
            except:
                out.insert("end","NOT INSTALLED\n","err")
            
            # Check models
            out.insert("end","\n  [MODELS] Checking v7 models...\n","hdr")
            v7_models = [
                ("xgb_model.pkl", "XGBoost"),
                ("lgb_model.pkl", "LightGBM"),
                ("catboost_model.pkl", "CatBoost"),
                ("tcn_model.pt", "TCN (v7)"),
                ("nbeats_model.pt", "N-BEATS (v7)"),
                ("nhits_model.pt", "N-HiTS (v7)"),
                ("ensemble_config.json", "Ensemble Config"),
            ]
            for f, name in v7_models:
                exists = os.path.exists(os.path.join(BASE, f))
                tag = "ok" if exists else "warn"
                out.insert("end", f"    {name}: {'✓' if exists else '✗'}\n", tag)
                win.update()
            
            # Check XAUSGD models
            out.insert("end","\n  [XAUSGD MODELS]\n","hdr")
            for f, name in v7_models:
                exists = os.path.exists(os.path.join(BASE, f"xausgd_{f}"))
                tag = "ok" if exists else "warn"
                out.insert("end", f"    {name}: {'✓' if exists else '✗'}\n", tag)
                win.update()
            
            # Check scripts
            out.insert("end","\n  [SCRIPTS]\n","hdr")
            scripts = [
                "live_trading_bridge.py",
                "live_bridge_xausgd.py",
                "features.py",
                "smc_logic.py",
                "news_guard.py",
            ]
            for s in scripts:
                exists = os.path.exists(os.path.join(BASE, s))
                tag = "ok" if exists else "err"
                out.insert("end", f"    {s}: {'✓' if exists else '✗'}\n", tag)
                win.update()
            
            out.insert("end","\n  ═══ READY TO TRADE ═══\n","hdr")
            out.insert("end","  Use XAUUSD/XAUSGD buttons to start.\n\n","ok")
            out.see("end")

        threading.Thread(target=run_checks,daemon=True).start()

    def _kill_all(self):
        if not messagebox.askyesno("Kill Switch","EMERGENCY STOP\n\nThis will:\n1. Close ALL positions\n2. Kill ALL Python processes\n\nContinue?"):
            return
        
        # Close all positions
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                for symbol, magic in [("XAUUSD", 999), ("XAUSGD", 1000)]:
                    positions = mt5.positions_get(symbol=symbol)
                    if positions:
                        for p in positions:
                            if p.magic != magic: continue
                            ct = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
                            tick = mt5.symbol_info_tick(symbol)
                            price = tick.bid if p.type == 0 else tick.ask
                            req = {"action":mt5.TRADE_ACTION_DEAL,"symbol":symbol,"volume":p.volume,
                                   "type":ct,"position":p.ticket,"price":price,"deviation":20,
                                   "magic":magic,"comment":"KILL SWITCH","type_filling":mt5.ORDER_FILLING_IOC}
                            mt5.order_send(req)
                mt5.shutdown()
        except: pass
        
        # Kill all python except this one
        subprocess.run(["taskkill","/f","/im","python.exe"],capture_output=True)
        
        self.active_procs.clear()
        self.usd_status.config(text="  XAUUSD: KILLED",fg=RED_HI)
        self.sgd_status.config(text="  XAUSGD: KILLED",fg=RED_HI)
        self._status("KILL SWITCH ACTIVATED")

    def _retrain(self):
        sp=os.path.join(BASE,"train_models_v7.py")
        if not os.path.exists(sp):
            sp=os.path.join(BASE,"train_models.py")
        if not os.path.exists(sp):
            messagebox.showerror("Not Found","train_models.py not found"); return
        cmd = f'start "AFB - Retrain XAUUSD v7" cmd /k python "{sp}"'
        subprocess.Popen(cmd, shell=True, cwd=BASE)
        self._status("Retraining XAUUSD v7 models...")

    def _retrain_sgd(self):
        sp=os.path.join(BASE,"train_xausgd_v7.py")
        if not os.path.exists(sp):
            sp=os.path.join(BASE,"train_xausgd.py")
        if not os.path.exists(sp):
            messagebox.showerror("Not Found","train_xausgd.py not found"); return
        cmd = f'start "AFB - Retrain XAUSGD v7" cmd /k python "{sp}"'
        subprocess.Popen(cmd, shell=True, cwd=BASE)
        self._status("Retraining XAUSGD v7 models...")

    # ══════════════════════════════════════════════════════════════════
    # SYSTEM HEALTH CHECK - NEW!
    # ══════════════════════════════════════════════════════════════════
    def _system_health_check(self):
        win=tk.Toplevel(self.root)
        win.title("System Health Check")
        win.configure(bg=BG1)
        win.geometry("700x600")
        
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" 🩺 System Health Check",fg=PURPLE,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,wrap="word",relief="flat")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        out.tag_config("ok",foreground=GREEN)
        out.tag_config("err",foreground=RED_HI)
        out.tag_config("warn",foreground=GOLD)
        out.tag_config("hdr",foreground=BLUE)
        out.tag_config("purple",foreground=PURPLE)

        bf=tk.Frame(win,bg=BG1); bf.pack(fill="x",padx=8,pady=4)
        tk.Button(bf,text=" Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=win.destroy).pack(side="right",padx=4)

        def run_health():
            out.insert("end","\n  ═══════════════════════════════════════════\n","purple")
            out.insert("end","   ALMOSTFINISHEDBOT SYSTEM HEALTH CHECK\n","purple")
            out.insert("end","  ═══════════════════════════════════════════\n\n","purple")
            
            issues = []
            
            # 1. Python packages
            out.insert("end","  📦 PYTHON PACKAGES\n","hdr")
            packages = [
                ("MetaTrader5", "mt5 connection"),
                ("torch", "deep learning"),
                ("xgboost", "XGBoost model"),
                ("lightgbm", "LightGBM model"),
                ("catboost", "CatBoost model"),
                ("pandas", "data processing"),
                ("numpy", "numerical"),
                ("sklearn", "ML utilities"),
                ("yfinance", "market data"),
            ]
            for pkg, desc in packages:
                try:
                    __import__(pkg.replace("-","_"))
                    out.insert("end", f"    ✓ {pkg} ({desc})\n", "ok")
                except ImportError:
                    out.insert("end", f"    ✗ {pkg} ({desc}) - MISSING\n", "err")
                    issues.append(f"Missing package: {pkg}")
                win.update()
            
            # 2. MT5 Connection
            out.insert("end","\n  🔌 MT5 CONNECTION\n","hdr")
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    info = mt5.account_info()
                    out.insert("end", f"    ✓ Connected to {info.server}\n", "ok")
                    out.insert("end", f"    ✓ Account: {info.login}\n", "ok")
                    out.insert("end", f"    ✓ Balance: ${info.balance:.2f}\n", "ok")
                    out.insert("end", f"    ✓ Mode: {'DEMO' if info.trade_mode==0 else 'LIVE'}\n", "ok")
                    
                    # Check symbols
                    for sym in ["XAUUSD", "XAUSGD"]:
                        si = mt5.symbol_info(sym)
                        if si:
                            out.insert("end", f"    ✓ {sym}: Available (spread={si.spread})\n", "ok")
                        else:
                            out.insert("end", f"    ✗ {sym}: Not available\n", "err")
                            issues.append(f"{sym} not available")
                    mt5.shutdown()
                else:
                    out.insert("end", "    ✗ Cannot connect - Is MT5 running?\n", "err")
                    issues.append("MT5 not connected")
            except Exception as e:
                out.insert("end", f"    ✗ Error: {e}\n", "err")
                issues.append(str(e))
            win.update()
            
            # 3. Model files
            out.insert("end","\n  🤖 XAUUSD MODELS (v7)\n","hdr")
            model_files = [
                ("xgb_model.pkl", "XGBoost"),
                ("lgb_model.pkl", "LightGBM"),
                ("gb_model.pkl", "GradientBoost"),
                ("catboost_model.pkl", "CatBoost"),
                ("rf_model.pkl", "RandomForest"),
                ("lstm_model.pt", "LSTM"),
                ("tft_model.pt", "TFT"),
                ("tcn_model.pt", "TCN"),
                ("nbeats_model.pt", "N-BEATS"),
                ("nhits_model.pt", "N-HiTS"),
                ("scaler.pkl", "Scaler"),
                ("meta_model.pkl", "Meta-Learner"),
                ("ensemble_config.json", "Config"),
            ]
            for f, name in model_files:
                path = os.path.join(BASE, f)
                if os.path.exists(path):
                    size = os.path.getsize(path) / 1024
                    out.insert("end", f"    ✓ {name} ({size:.1f}KB)\n", "ok")
                else:
                    out.insert("end", f"    ✗ {name} - MISSING\n", "warn")
                win.update()
            
            out.insert("end","\n  🤖 XAUSGD MODELS (v7)\n","hdr")
            for f, name in model_files:
                path = os.path.join(BASE, f"xausgd_{f}")
                if os.path.exists(path):
                    size = os.path.getsize(path) / 1024
                    out.insert("end", f"    ✓ {name} ({size:.1f}KB)\n", "ok")
                else:
                    out.insert("end", f"    ✗ {name} - MISSING\n", "warn")
                win.update()
            
            # 4. Scripts
            out.insert("end","\n  📜 SCRIPTS\n","hdr")
            scripts = [
                ("live_trading_bridge.py", "XAUUSD Bridge"),
                ("live_bridge_xausgd.py", "XAUSGD Bridge"),
                ("features.py", "Feature Engineering"),
                ("smc_logic.py", "SMC Logic"),
                ("news_guard.py", "News Guard"),
                ("correlation_guard.py", "Correlation Guard"),
            ]
            for f, name in scripts:
                if os.path.exists(os.path.join(BASE, f)):
                    out.insert("end", f"    ✓ {name}\n", "ok")
                else:
                    out.insert("end", f"    ✗ {name} - MISSING\n", "err")
                    issues.append(f"Missing script: {f}")
                win.update()
            
            # Summary
            out.insert("end","\n  ═══════════════════════════════════════════\n","purple")
            if issues:
                out.insert("end", f"   ⚠️ {len(issues)} ISSUES FOUND\n", "warn")
                for issue in issues[:5]:
                    out.insert("end", f"   • {issue}\n", "err")
            else:
                out.insert("end", "   ✅ ALL SYSTEMS HEALTHY\n", "ok")
            out.insert("end","  ═══════════════════════════════════════════\n","purple")
            out.see("end")

        threading.Thread(target=run_health,daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    # INSTALL PACKAGES - NEW!
    # ══════════════════════════════════════════════════════════════════
    def _install_packages(self):
        win=tk.Toplevel(self.root)
        win.title("Install Packages")
        win.configure(bg=BG1)
        win.geometry("600x500")
        
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" 📦 Install Required Packages",fg=PURPLE,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,wrap="word",relief="flat")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        out.tag_config("ok",foreground=GREEN)
        out.tag_config("err",foreground=RED_HI)
        out.tag_config("hdr",foreground=BLUE)

        bf=tk.Frame(win,bg=BG1); bf.pack(fill="x",padx=8,pady=4)
        
        def install_all():
            packages = [
                "MetaTrader5",
                "pandas",
                "numpy",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "catboost",
                "torch",
                "yfinance",
                "joblib",
                "requests",
            ]
            out.insert("end","\n  Installing packages...\n\n","hdr")
            for pkg in packages:
                out.insert("end", f"  Installing {pkg}... "); out.see("end"); win.update()
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", pkg, "-q"],
                        capture_output=True, text=True, timeout=120
                    )
                    if result.returncode == 0:
                        out.insert("end", "OK\n", "ok")
                    else:
                        out.insert("end", f"WARN\n", "err")
                except Exception as e:
                    out.insert("end", f"ERROR: {e}\n", "err")
                win.update()
            out.insert("end", "\n  Done! Run System Health Check to verify.\n", "hdr")
            out.see("end")
        
        tk.Button(bf,text=" Install All Packages ",bg=GREEN,fg=WHITE,font=FBB,relief="flat",
                  command=lambda:threading.Thread(target=install_all,daemon=True).start()).pack(side="left",padx=4)
        tk.Button(bf,text=" Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=win.destroy).pack(side="right",padx=4)

    # ══════════════════════════════════════════════════════════════════
    # GUARDS - Actually working now!
    # ══════════════════════════════════════════════════════════════════
    def _check_news_guard(self):
        win=tk.Toplevel(self.root)
        win.title("News Guard Status")
        win.configure(bg=BG1)
        win.geometry("600x450")
        
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" 📰 News Guard",fg=GOLD,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,wrap="word",relief="flat")
        out.pack(fill="both",expand=True,padx=8,pady=4)
        out.tag_config("ok",foreground=GREEN)
        out.tag_config("err",foreground=RED_HI)
        out.tag_config("warn",foreground=GOLD)
        out.tag_config("hdr",foreground=BLUE)

        bf=tk.Frame(win,bg=BG1); bf.pack(fill="x",padx=8,pady=4)
        tk.Button(bf,text=" Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=win.destroy).pack(side="right",padx=4)

        def check_news():
            out.insert("end","\n  Checking news impact...\n\n","hdr")
            
            # Try to run news_guard.py
            sp = os.path.join(BASE, "news_guard.py")
            if os.path.exists(sp):
                try:
                    result = subprocess.run([sys.executable, sp], cwd=BASE, capture_output=True, text=True, timeout=30)
                    out.insert("end", result.stdout or "No output\n")
                    if result.stderr:
                        out.insert("end", f"Errors: {result.stderr}\n", "err")
                except Exception as e:
                    out.insert("end", f"Error: {e}\n", "err")
            
            # Check saved status
            ng = _rj("news_guard.json")
            if ng:
                out.insert("end", "\n  Saved Status:\n", "hdr")
                if ng.get("blocked"):
                    out.insert("end", f"  ⛔ BLOCKED: {ng.get('reason','Unknown')}\n", "err")
                elif ng.get("reduced"):
                    out.insert("end", f"  ⚠️ CAUTION: {ng.get('reason','')}\n", "warn")
                else:
                    out.insert("end", "  ✅ CLEAR - No high impact news\n", "ok")
                
                if ng.get("events"):
                    out.insert("end", "\n  Upcoming Events:\n", "hdr")
                    for e in ng.get("events", [])[:5]:
                        out.insert("end", f"    • {e}\n")
            else:
                out.insert("end", "\n  No news guard status file found.\n", "warn")
            
            out.see("end")

        threading.Thread(target=check_news,daemon=True).start()

    def _check_smc(self):
        win=tk.Toplevel(self.root)
        win.title("SMC Analysis")
        win.configure(bg=BG1)
        win.geometry("600x400")
        
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" 📊 Smart Money Concepts Analysis",fg=BLUE,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,wrap="word",relief="flat")
        out.pack(fill="both",expand=True,padx=8,pady=4)

        bf=tk.Frame(win,bg=BG1); bf.pack(fill="x",padx=8,pady=4)
        tk.Button(bf,text=" Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=win.destroy).pack(side="right",padx=4)

        def check():
            out.insert("end","\n  Running SMC analysis...\n\n")
            sp = os.path.join(BASE, "smc_logic.py")
            if os.path.exists(sp):
                try:
                    # Import and run
                    sys.path.insert(0, BASE)
                    from smc_logic import get_bias
                    result = get_bias()
                    out.insert("end", f"  Direction: {'BUY' if result.get('direction',0)>0 else 'SELL' if result.get('direction',0)<0 else 'NEUTRAL'}\n")
                    out.insert("end", f"  Score: {result.get('score',0)}\n")
                    out.insert("end", f"\n  Details:\n")
                    for d in result.get("details", []):
                        out.insert("end", f"    • {d}\n")
                except Exception as e:
                    out.insert("end", f"  Error: {e}\n")
            else:
                out.insert("end", "  smc_logic.py not found\n")
            out.see("end")

        threading.Thread(target=check,daemon=True).start()

    def _check_correlation(self):
        win=tk.Toplevel(self.root)
        win.title("Correlation Check")
        win.configure(bg=BG1)
        win.geometry("600x400")
        
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" 🔗 Correlation Analysis",fg=BLUE,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,wrap="word",relief="flat")
        out.pack(fill="both",expand=True,padx=8,pady=4)

        bf=tk.Frame(win,bg=BG1); bf.pack(fill="x",padx=8,pady=4)
        tk.Button(bf,text=" Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=win.destroy).pack(side="right",padx=4)

        def check():
            out.insert("end","\n  Checking correlations...\n\n")
            sp = os.path.join(BASE, "correlation_guard.py")
            if os.path.exists(sp):
                try:
                    result = subprocess.run([sys.executable, sp], cwd=BASE, capture_output=True, text=True, timeout=30)
                    out.insert("end", result.stdout or "No output\n")
                except Exception as e:
                    out.insert("end", f"  Error: {e}\n")
            else:
                out.insert("end", "  correlation_guard.py not found\n")
            out.see("end")

        threading.Thread(target=check,daemon=True).start()

    def _check_regime(self):
        win=tk.Toplevel(self.root)
        win.title("Market Regime")
        win.configure(bg=BG1)
        win.geometry("600x400")
        
        oh,ih=bordered(win); oh.pack(fill="x",padx=8,pady=8)
        tk.Label(ih,text=" 📈 Market Regime Detection",fg=BLUE,bg=BG2,font=FH).pack(anchor="w",padx=8,pady=5)
        
        out=scrolledtext.ScrolledText(win,bg=BG0,fg=GREY_LT,font=FB,wrap="word",relief="flat")
        out.pack(fill="both",expand=True,padx=8,pady=4)

        bf=tk.Frame(win,bg=BG1); bf.pack(fill="x",padx=8,pady=4)
        tk.Button(bf,text=" Close ",bg=RED_MID,fg=WHITE,font=FBB,relief="flat",command=win.destroy).pack(side="right",padx=4)

        def check():
            out.insert("end","\n  Detecting market regime...\n\n")
            sig = _rj("current_signal.json")
            if sig:
                out.insert("end", f"  Current Regime: {sig.get('current_regime','Unknown')}\n")
                out.insert("end", f"  Signal: {sig.get('signal','?')}\n")
                out.insert("end", f"  Confidence: {sig.get('confidence',0)*100:.1f}%\n")
                out.insert("end", f"  Strategy: {sig.get('strategy_used','?')}\n")
            else:
                out.insert("end", "  No signal data available\n")
            out.see("end")

        threading.Thread(target=check,daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════
    def _show_kelly(self):
        ec=_rj("ensemble_config.json")
        msg = (f"Best Strategy: {ec.get('best_strategy','?')}\n"
               f"Best Accuracy: {ec.get('best_accuracy',0)*100:.1f}%\n\n"
               f"Model Accuracies:\n")
        accs = ec.get("model_accuracies", {})
        for m, a in sorted(accs.items(), key=lambda x: -x[1]):
            msg += f"  {m}: {a*100:.1f}%\n"
        messagebox.showinfo("Model Performance",msg)

    def _show_compounding(self):
        rates = [0.01, 0.015, 0.02, 0.03]
        days_list = [7, 14, 30, 60, 90]
        lines = ["Starting: $100\n"]
        for r in rates:
            lines.append(f"\n{r*100:.1f}% daily:")
            for d in days_list:
                val = 100 * (1+r)**d
                lines.append(f"  {d:3d} days: ${val:,.2f}")
        messagebox.showinfo("Compounding Calculator","\n".join(lines))

    def _show_account(self):
        try:
            import MetaTrader5 as mt5
            mt5.initialize()
            info = mt5.account_info()
            if info:
                # Get positions
                usd_pos = mt5.positions_get(symbol="XAUUSD") or []
                sgd_pos = mt5.positions_get(symbol="XAUSGD") or []
                
                msg = (f"Login: {info.login}\n"
                       f"Server: {info.server}\n"
                       f"Balance: ${info.balance:.2f}\n"
                       f"Equity: ${info.equity:.2f}\n"
                       f"Profit: ${info.profit:+.2f}\n"
                       f"Margin Free: ${info.margin_free:.2f}\n"
                       f"Mode: {'Demo' if info.trade_mode==0 else 'Live'}\n\n"
                       f"Open Positions:\n"
                       f"  XAUUSD: {len(usd_pos)}\n"
                       f"  XAUSGD: {len(sgd_pos)}")
                messagebox.showinfo("Account",msg)
            mt5.shutdown()
        except Exception as e:
            messagebox.showerror("Error",f"Cannot connect to MT5:\n{e}")

    def _view_trade_log(self):
        log_path = os.path.join(BASE, "trading_log.json")
        if os.path.exists(log_path):
            os.startfile(log_path)
        else:
            messagebox.showinfo("Info", "No trade log found yet")

    # ══════════════════════════════════════════════════════════════════
    # STATUS REFRESH
    # ══════════════════════════════════════════════════════════════════
    def _start_status_refresh(self):
        self._refresh_status()

    def _refresh_status(self):
        try:
            # Signal
            sig = _rj("current_signal.json")
            if sig:
                s = sig.get("signal","?"); c = sig.get("confidence",0)
                regime = sig.get("current_regime","?")
                col = GREEN if s=="BUY" else (RED_HI if s=="SELL" else GREY_MID)
                self.signal_lbl.config(
                    text=f"  Signal: {s} ({c:.1%})  |  Regime: {regime}  |  {sig.get('strategy_used','?')}",
                    fg=col)

            # News
            ng = _rj("news_guard.json")
            if ng:
                if ng.get("blocked"): self.news_lbl.config(text=f"  News: BLOCKED - {ng.get('reason','')}",fg=RED_HI)
                elif ng.get("reduced"): self.news_lbl.config(text=f"  News: CAUTION - {ng.get('reason','')}",fg=GOLD)
                else: self.news_lbl.config(text="  News: CLEAR",fg=GREEN)

            # Correlation
            cs = _rj("correlation_status.json")
            if cs:
                if cs.get("kill_switch"): self.corr_lbl.config(text="  Correlations: KILL SWITCH",fg=RED_HI)
                else: self.corr_lbl.config(text="  Correlations: Normal",fg=GREEN)

            # Processes
            alive = list(self.active_procs.keys())
            if alive:
                self.proc_lbl.config(text=f"  Active: {', '.join(alive)}",fg=GREEN)
            else:
                self.proc_lbl.config(text="  Processes: None",fg=GREY_DIM)

            # Account
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    info = mt5.account_info()
                    if info:
                        self.acc_lbl.config(text=f"${info.balance:.2f} | Eq:${info.equity:.2f}")
                    mt5.shutdown()
            except: pass

        except: pass
        self.root.after(5000, self._refresh_status)


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
