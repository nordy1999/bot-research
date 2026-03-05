"""
AlmostFinishedBot - Control Center v6.0
========================================
FULL REWRITE — every button works correctly:

KILL SWITCH:
  1. Closes ALL open positions (XAUUSD + XAUSGD) via MT5
  2. Kills ALL Python bridge processes by name + window title
  3. Kills any orphan cmd windows
  4. Updates status labels to KILLED

CLOSE ALL (per symbol):
  1. Closes positions for that symbol only
  2. Terminates only that symbol's bridge process

PAPER TRADE / LIVE TRADE:
  - Launches correct bridge script with tracked process handle

SYSTEM HEALTH CHECK:
  - Checks packages, MT5, models, scripts

All other buttons: News, SMC, Correlation, Regime, Kelly, Compounding, Account
"""
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os, sys, subprocess, json, time, threading, datetime

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
os.makedirs(BASE, exist_ok=True)

# ── MT5 credentials ───────────────────────────────────────────────
MT5_LOGIN    = 62111880
MT5_PASSWORD = "h3%ejzpaUy"
MT5_SERVER   = "PepperstoneUK-Demo"

# ── Colours ───────────────────────────────────────────────────────
BG0 = "#0f0f0f"; BG1 = "#1a1a1a"; BG2 = "#242424"; BG3 = "#303030"
BORDER = "#ffffff"; BORD_D = "#555555"
RED_HI = "#ff2222"; RED_MID = "#cc3333"; RED_DIM = "#882222"
WHITE = "#ffffff"; GREY_LT = "#dddddd"; GREY_MID = "#999999"; GREY_DIM = "#555555"
GREEN = "#22cc66"; GOLD = "#ffcc44"; BLUE = "#4499ff"; PURPLE = "#aa66ff"
ORANGE = "#ff9900"

FH  = ("Consolas", 10, "bold")
FB  = ("Consolas", 9)
FS  = ("Consolas", 8)
FT  = ("Consolas", 14, "bold")
FBB = ("Consolas", 9, "bold")

MAGIC_XAUUSD = 999
MAGIC_XAUSGD = 1000


def _rj(path):
    try:
        with open(os.path.join(BASE, path)) as f:
            return json.load(f)
    except Exception:
        return {}


def bordered(parent, **kw):
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1, **kw)
    inner = tk.Frame(outer, bg=BG2)
    inner.pack(fill="both", expand=True)
    return outer, inner


def section_header(parent, text, color=None):
    c = color or RED_HI
    frm = tk.Frame(parent, bg=BG0, highlightbackground=BORDER, highlightthickness=1)
    frm.pack(fill="x", padx=6, pady=(12, 2))
    tk.Label(frm, text=f"  {text}  ", fg=c, bg=BG0, font=FH).pack(side="left", pady=4)
    return frm


# ═══════════════════════════════════════════════════════════════════
# MT5 helpers (standalone so buttons work without the full bridge)
# ═══════════════════════════════════════════════════════════════════
def mt5_init():
    try:
        import MetaTrader5 as mt5
        ok = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not ok:
            ok = mt5.initialize()
        return ok, mt5
    except ImportError:
        return False, None


def mt5_close_all_positions(symbols_magics):
    """
    Close all positions for given [(symbol, magic), ...] pairs.
    Returns (closed_count, error_message)
    """
    ok, mt5 = mt5_init()
    if not ok or mt5 is None:
        return 0, "MT5 not available"
    closed = 0
    err    = ""
    try:
        for symbol, magic in symbols_magics:
            positions = mt5.positions_get(symbol=symbol) or []
            for p in positions:
                try:
                    ct    = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
                    tick  = mt5.symbol_info_tick(symbol)
                    if not tick:
                        continue
                    price = tick.bid if p.type == 0 else tick.ask
                    req   = {
                        "action":       mt5.TRADE_ACTION_DEAL,
                        "symbol":       symbol,
                        "volume":       p.volume,
                        "type":         ct,
                        "position":     p.ticket,
                        "price":        price,
                        "deviation":    50,
                        "magic":        magic,
                        "comment":      "KILL SWITCH",
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    r = mt5.order_send(req)
                    if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                        closed += 1
                    else:
                        err += f"{symbol}:{p.ticket} retcode={r.retcode if r else 'None'} "
                except Exception as ex:
                    err += str(ex) + " "
    except Exception as ex:
        err = str(ex)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass
    return closed, err


def kill_bridge_processes(symbol=None):
    """
    Kill bridge Python processes.
    If symbol is None, kill ALL bridges.
    """
    scripts = []
    if symbol is None:
        scripts = ["live_trading_bridge.py", "live_bridge_xausgd.py"]
    elif "USD" in symbol:
        scripts = ["live_trading_bridge.py"]
    else:
        scripts = ["live_bridge_xausgd.py"]

    # Kill by window title
    titles = []
    if symbol is None:
        titles = [
            "AFB_XAUUSD_Paper", "AFB_XAUSGD_Paper",
            "AlmostFinishedBot - XAUUSD", "AlmostFinishedBot - XAUSGD",
            "AFB_XAUUSD_LIVE", "AFB_XAUSGD_LIVE",
            "AFB - XAUUSD Backtest", "AFB - XAUSGD Backtest",
        ]
    else:
        sym_short = "XAUUSD" if "USD" in symbol else "XAUSGD"
        titles = [
            f"AFB_{sym_short}_Paper", f"AFB_{sym_short}_LIVE",
            f"AlmostFinishedBot - {sym_short}",
        ]

    for title in titles:
        try:
            subprocess.run(
                f'taskkill /fi "WINDOWTITLE eq {title}*" /f',
                shell=True, capture_output=True)
        except Exception:
            pass

    # Kill by script name in commandline
    for script in scripts:
        try:
            subprocess.run(
                f'wmic process where "commandline like \'%{script}%\'" call terminate',
                shell=True, capture_output=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AlmostFinishedBot — Control Center v6.0")
        self.root.configure(bg=BG1)
        self.root.geometry("1000x980")
        self.root.minsize(860, 760)
        # Track live processes: name -> Popen or window title string
        self.active_procs = {}
        self._build()
        self._start_status_refresh()

    # ── Build UI ─────────────────────────────────────────────────
    def _build(self):
        r = self.root

        # Header
        oh, ih = bordered(r)
        oh.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(ih,
                 text="  ALMOSTFINISHEDBOT  |  CONTROL CENTER  v6.0  ",
                 fg=RED_HI, bg=BG2, font=("Consolas", 15, "bold")
                 ).pack(side="left", pady=8)
        self.status_lbl = tk.Label(ih, text="", fg=GREY_MID, bg=BG2, font=FS)
        self.status_lbl.pack(side="right", padx=6)
        self.acc_lbl = tk.Label(ih, text="", fg=GREY_MID, bg=BG2, font=FS)
        self.acc_lbl.pack(side="right", padx=6)

        # Scrollable area
        canvas = tk.Canvas(r, bg=BG1, highlightthickness=0)
        vsb    = tk.Scrollbar(r, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=4)
        sf = tk.Frame(canvas, bg=BG1)
        sfid = canvas.create_window((0, 0), window=sf, anchor="nw")
        sf.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(sfid, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # ── Trading Dashboard ─────────────────────────────────────
        section_header(sf, "📊  TRADING DASHBOARD  (Charts + Trade Markers)", BLUE)
        cg = tk.Frame(sf, bg=BG1); cg.pack(fill="x", padx=6)
        self._big_btn(cg, "📊  OPEN TRADING DASHBOARD",
                      "Live candles — Green▲BUY / Red▼SELL markers — Zoom/Pan",
                      BLUE, self._open_dashboard)

        # ── XAUUSD ────────────────────────────────────────────────
        section_header(sf, "🥇  XAUUSD — Gold vs US Dollar", GOLD)
        ug = tk.Frame(sf, bg=BG1); ug.pack(fill="x", padx=6)
        bf = tk.Frame(ug, bg=BG1); bf.pack(fill="x", padx=4, pady=4)
        self._action_btn(bf, "▶  PAPER TRADE", GREEN,
            lambda: self._launch_bridge("live_trading_bridge.py", "--mode paper", "XAUUSD"))
        self._action_btn(bf, "📊  BACKTEST", BLUE,
            lambda: self._launch_script("walkforward_backtest.py", "", "XAUUSD Backtest"))
        self._action_btn(bf, "⚡  LIVE TRADE", RED_DIM,
            lambda: self._live_trade("live_trading_bridge.py", "XAUUSD"))
        self._action_btn(bf, "❌  CLOSE ALL + STOP", RED_MID,
            lambda: self._close_symbol_and_stop("XAUUSD", MAGIC_XAUUSD))
        self.usd_status = tk.Label(ug, text="  XAUUSD: Ready",
                                    fg=GREY_MID, bg=BG1, font=FB)
        self.usd_status.pack(anchor="w", padx=10)

        # ── XAUSGD ────────────────────────────────────────────────
        section_header(sf, "🎯  XAUSGD — Gold vs Singapore Dollar", GOLD)
        sg = tk.Frame(sf, bg=BG1); sg.pack(fill="x", padx=6)
        bfg = tk.Frame(sg, bg=BG1); bfg.pack(fill="x", padx=4, pady=4)
        self._action_btn(bfg, "▶  PAPER TRADE", GREEN,
            lambda: self._launch_bridge("live_bridge_xausgd.py", "--mode paper", "XAUSGD"))
        self._action_btn(bfg, "📊  BACKTEST", BLUE,
            lambda: self._launch_script("walkforward_backtest.py", "xausgd", "XAUSGD Backtest"))
        self._action_btn(bfg, "⚡  LIVE TRADE", RED_DIM,
            lambda: self._live_trade("live_bridge_xausgd.py", "XAUSGD"))
        self._action_btn(bfg, "❌  CLOSE ALL + STOP", RED_MID,
            lambda: self._close_symbol_and_stop("XAUSGD", MAGIC_XAUSGD))
        self.sgd_status = tk.Label(sg, text="  XAUSGD: Ready",
                                    fg=GREY_MID, bg=BG1, font=FB)
        self.sgd_status.pack(anchor="w", padx=10)

        # ── System Controls ───────────────────────────────────────
        section_header(sf, "⚙  SYSTEM CONTROLS", WHITE)
        scg = tk.Frame(sf, bg=BG1); scg.pack(fill="x", padx=6)
        bf2 = tk.Frame(scg, bg=BG1); bf2.pack(fill="x", padx=4, pady=4)
        self._action_btn(bf2, "⚡  START FULL SYSTEM", GREEN,  self._start_full_system)
        self._action_btn(bf2, "⬛  KILL SWITCH",       RED_HI, self._kill_all)
        self._action_btn(bf2, "🔄  RETRAIN XAUUSD",   BLUE,   self._retrain)
        self._action_btn(bf2, "🔄  RETRAIN XAUSGD",   BLUE,   self._retrain_sgd)
        bf3 = tk.Frame(scg, bg=BG1); bf3.pack(fill="x", padx=4, pady=4)
        self._action_btn(bf3, "🩺  HEALTH CHECK",     PURPLE, self._system_health_check)
        self._action_btn(bf3, "📦  INSTALL PACKAGES", PURPLE, self._install_packages)
        self._action_btn(bf3, "📁  OPEN FOLDER",      BG3,    lambda: os.startfile(BASE))

        # ── Guards ────────────────────────────────────────────────
        section_header(sf, "🛡  GUARDS & INTELLIGENCE", RED_HI)
        gg = tk.Frame(sf, bg=BG1); gg.pack(fill="x", padx=6)
        bfx = tk.Frame(gg, bg=BG1); bfx.pack(fill="x", padx=4, pady=4)
        self._action_btn(bfx, "📰  News Guard",       GOLD,  self._check_news)
        self._action_btn(bfx, "📊  SMC Analysis",     BG3,   self._check_smc)
        self._action_btn(bfx, "🔗  Correlation",      BG3,   self._check_correlation)
        self._action_btn(bfx, "📈  Market Regime",    BG3,   self._check_regime)

        # ── Live Status ───────────────────────────────────────────
        section_header(sf, "📡  LIVE STATUS", GREEN)
        ls = tk.Frame(sf, bg=BG1); ls.pack(fill="x", padx=6)
        self.signal_lbl = tk.Label(ls, text="  Signal: --", fg=GREY_MID, bg=BG1, font=FB)
        self.signal_lbl.pack(anchor="w", padx=10)
        self.news_lbl   = tk.Label(ls, text="  News: --",   fg=GREY_MID, bg=BG1, font=FB)
        self.news_lbl.pack(anchor="w", padx=10)
        self.corr_lbl   = tk.Label(ls, text="  Correlations: --", fg=GREY_MID, bg=BG1, font=FB)
        self.corr_lbl.pack(anchor="w", padx=10)
        self.proc_lbl   = tk.Label(ls, text="  Processes: none", fg=GREY_MID, bg=BG1, font=FB)
        self.proc_lbl.pack(anchor="w", padx=10)

        # ── Utilities ─────────────────────────────────────────────
        section_header(sf, "🧮  UTILITIES", WHITE)
        uf = tk.Frame(sf, bg=BG1); uf.pack(fill="x", padx=6)
        bfu = tk.Frame(uf, bg=BG1); bfu.pack(fill="x", padx=4, pady=4)
        self._action_btn(bfu, "📊  Kelly Sizing",     BG3,  self._show_kelly)
        self._action_btn(bfu, "📈  Compounding",      BG3,  self._show_compounding)
        self._action_btn(bfu, "💰  Account Info",     BG3,  self._show_account)
        self._action_btn(bfu, "📋  Trade Log",        BG3,  self._view_log)

    # ── Widget helpers ────────────────────────────────────────────
    def _big_btn(self, parent, text, desc, color, cmd):
        f = tk.Frame(parent, bg=color, relief="flat")
        f.pack(fill="x", padx=4, pady=6)
        tk.Button(f, text=f"  {text}  ", bg=color, fg=WHITE,
                  font=("Consolas", 12, "bold"), relief="flat",
                  cursor="hand2", anchor="w", command=cmd
                  ).pack(fill="x", padx=6, pady=(6, 0))
        tk.Label(f, text=f"    {desc}", fg=GREY_LT, bg=color,
                 font=FS, wraplength=850, anchor="w"
                 ).pack(fill="x", padx=6, pady=(0, 6))

    def _action_btn(self, parent, text, color, cmd):
        b = tk.Button(parent, text=f" {text} ", bg=color, fg=WHITE,
                      font=FBB, relief="flat", cursor="hand2",
                      command=cmd, padx=8, pady=4)
        b.pack(side="left", padx=3, pady=3)
        b.bind("<Enter>", lambda e, b=b: b.config(bg=WHITE, fg=BG0))
        b.bind("<Leave>", lambda e, b=b, c=color: b.config(bg=c, fg=WHITE))

    def _status(self, msg):
        try:
            self.status_lbl.config(text=f"  {msg}  ")
        except Exception:
            pass

    # ── Dashboard ─────────────────────────────────────────────────
    def _open_dashboard(self):
        sp = os.path.join(BASE, "trading_dashboard.py")
        if not os.path.exists(sp):
            messagebox.showerror("Not Found", f"trading_dashboard.py not found in:\n{BASE}")
            return
        proc = subprocess.Popen([sys.executable, sp], cwd=BASE)
        self.active_procs["Dashboard"] = proc
        self._status("Dashboard opened")

    # ── Bridge launchers ──────────────────────────────────────────
    def _launch_bridge(self, script, args, symbol):
        sp = os.path.join(BASE, script)
        if not os.path.exists(sp):
            messagebox.showerror("Not Found", f"{script} not found in:\n{BASE}")
            return
        title = f"AFB_{symbol}_Paper"
        cmd   = f'start "{title}" cmd /k python "{sp}" {args}'
        subprocess.Popen(cmd, shell=True, cwd=BASE)
        self.active_procs[symbol] = title
        self._status(f"{symbol} paper trade started")
        label = self.usd_status if "USD" in symbol else self.sgd_status
        label.config(text=f"  {symbol}: PAPER TRADING", fg=GREEN)

    def _launch_script(self, script, args, label):
        sp = os.path.join(BASE, script)
        if not os.path.exists(sp):
            messagebox.showerror("Not Found", f"{script} not found")
            return
        cmd = f'start "AFB - {label}" cmd /k python "{sp}" {args}'
        subprocess.Popen(cmd, shell=True, cwd=BASE)
        self._status(f"{label} started")

    def _live_trade(self, script, symbol):
        if not messagebox.askyesno("LIVE TRADING",
            f"Real money trading on {symbol}.\n\nAre you sure?"):
            return
        if not messagebox.askyesno("FINAL CONFIRM",
            f"LAST CHANCE — real money at risk on {symbol}.\n\nProceed?"):
            return
        sp    = os.path.join(BASE, script)
        title = f"AFB_{symbol}_LIVE"
        cmd   = f'start "{title}" cmd /k python "{sp}" --mode live'
        subprocess.Popen(cmd, shell=True, cwd=BASE)
        self.active_procs[f"{symbol}_Live"] = title
        label = self.usd_status if "USD" in symbol else self.sgd_status
        label.config(text=f"  {symbol}: LIVE TRADING ⚡", fg=RED_HI)
        self._status(f"{symbol} LIVE started")

    # ── Close symbol + stop bridge ────────────────────────────────
    def _close_symbol_and_stop(self, symbol, magic):
        if not messagebox.askyesno("Close & Stop",
            f"This will:\n"
            f"1. Close ALL {symbol} positions\n"
            f"2. Stop the {symbol} bridge\n\n"
            f"Continue?"):
            return

        def do():
            closed, err = mt5_close_all_positions([(symbol, magic)])
            kill_bridge_processes(symbol)
            # Clean active_procs
            to_rm = [k for k in self.active_procs if symbol in k]
            for k in to_rm:
                del self.active_procs[k]
            label = self.usd_status if "USD" in symbol else self.sgd_status
            label.config(text=f"  {symbol}: STOPPED", fg=RED_HI)
            msg = f"Closed {closed} position(s), bridge stopped."
            if err:
                msg += f"\nErrors: {err}"
            messagebox.showinfo("Done", msg)
            self._status(f"{symbol}: {closed} closed, bridge stopped")

        threading.Thread(target=do, daemon=True).start()

    # ── Kill Switch ───────────────────────────────────────────────
    def _kill_all(self):
        if not messagebox.askyesno("⚠ KILL SWITCH",
            "EMERGENCY STOP\n\n"
            "This will:\n"
            "1. Close ALL positions (XAUUSD + XAUSGD)\n"
            "2. Kill ALL Python bridge processes\n\n"
            "Proceed?", icon="warning"):
            return

        def do():
            report = []
            # Close all positions
            closed, err = mt5_close_all_positions([
                ("XAUUSD", MAGIC_XAUUSD),
                ("XAUSGD", MAGIC_XAUSGD),
            ])
            report.append(f"✓ Closed {closed} position(s)")
            if err:
                report.append(f"  Errors: {err}")

            # Kill all bridges
            kill_bridge_processes(symbol=None)
            report.append("✓ All bridges terminated")

            # Kill all python subprocesses except this one
            my_pid = os.getpid()
            try:
                subprocess.run(
                    ["wmic", "process", "where",
                     f"name='python.exe' and ProcessId!={my_pid}",
                     "call", "terminate"],
                    capture_output=True)
                report.append("✓ Python subprocesses killed")
            except Exception as e:
                report.append(f"✗ Kill error: {e}")

            self.active_procs.clear()
            self.usd_status.config(text="  XAUUSD: KILLED", fg=RED_HI)
            self.sgd_status.config(text="  XAUSGD: KILLED", fg=RED_HI)
            self._status("KILL SWITCH ACTIVATED")
            messagebox.showinfo("Kill Switch", "\n".join(report))

        threading.Thread(target=do, daemon=True).start()

    # ── Retrain ───────────────────────────────────────────────────
    def _retrain(self):
        sp = os.path.join(BASE, "train_models_v7.py")
        if not os.path.exists(sp):
            sp = os.path.join(BASE, "train_models.py")
        if not os.path.exists(sp):
            messagebox.showerror("Not Found", "train_models.py not found"); return
        subprocess.Popen(
            f'start "AFB - Retrain XAUUSD" cmd /k python "{sp}"',
            shell=True, cwd=BASE)
        self._status("Retraining XAUUSD...")

    def _retrain_sgd(self):
        sp = os.path.join(BASE, "train_xausgd_v7.py")
        if not os.path.exists(sp):
            sp = os.path.join(BASE, "train_xausgd.py")
        if not os.path.exists(sp):
            messagebox.showerror("Not Found", "train_xausgd.py not found"); return
        subprocess.Popen(
            f'start "AFB - Retrain XAUSGD" cmd /k python "{sp}"',
            shell=True, cwd=BASE)
        self._status("Retraining XAUSGD...")

    # ── System Health Check ───────────────────────────────────────
    def _system_health_check(self):
        win = tk.Toplevel(self.root)
        win.title("System Health Check")
        win.configure(bg=BG1)
        win.geometry("720x640")
        oh, ih = bordered(win)
        oh.pack(fill="x", padx=8, pady=8)
        tk.Label(ih, text=" 🩺 System Health Check", fg=PURPLE, bg=BG2,
                 font=FH).pack(anchor="w", padx=8, pady=5)
        out = scrolledtext.ScrolledText(win, bg=BG0, fg=GREY_LT, font=FB,
                                         wrap="word", relief="flat")
        out.pack(fill="both", expand=True, padx=8, pady=4)
        for tag, col in [("ok", GREEN), ("err", RED_HI), ("warn", GOLD),
                          ("hdr", BLUE), ("purple", PURPLE)]:
            out.tag_config(tag, foreground=col)
        bf = tk.Frame(win, bg=BG1); bf.pack(fill="x", padx=8, pady=4)
        tk.Button(bf, text=" Close ", bg=RED_MID, fg=WHITE, font=FBB,
                  relief="flat", command=win.destroy).pack(side="right", padx=4)

        def run():
            issues = []
            out.insert("end", "\n  ═══ HEALTH CHECK ═══\n\n", "purple")
            out.insert("end", "  📦 PACKAGES\n", "hdr")
            for pkg, desc in [
                ("MetaTrader5", "MT5 bridge"), ("torch", "PyTorch"),
                ("xgboost", "XGB"), ("lightgbm", "LGB"),
                ("catboost", "CatBoost"), ("pandas", "data"),
                ("numpy", "math"), ("sklearn", "ML"), ("yfinance", "data"),
            ]:
                try:
                    __import__(pkg.replace("-", "_"))
                    out.insert("end", f"    ✓ {pkg} ({desc})\n", "ok")
                except ImportError:
                    out.insert("end", f"    ✗ {pkg} — MISSING\n", "err")
                    issues.append(f"Missing: {pkg}")
                win.update()

            out.insert("end", "\n  🔌 MT5\n", "hdr")
            ok, mt5 = mt5_init()
            if ok and mt5:
                info = mt5.account_info()
                if info:
                    mode = "DEMO" if info.trade_mode == 0 else "LIVE"
                    out.insert("end", f"    ✓ #{info.login} {info.server} [{mode}]\n", "ok")
                    out.insert("end", f"    ✓ Balance: £{info.balance:.2f}\n", "ok")
                    for sym in ["XAUUSD", "XAUSGD"]:
                        si = mt5.symbol_info(sym)
                        tag = "ok" if si else "err"
                        out.insert("end", f"    {'✓' if si else '✗'} {sym}\n", tag)
                        if not si:
                            issues.append(f"{sym} not available")
                try:
                    mt5.shutdown()
                except Exception:
                    pass
            else:
                out.insert("end", "    ✗ Cannot connect — is MT5 running?\n", "err")
                issues.append("MT5 not connected")
            win.update()

            out.insert("end", "\n  🤖 MODELS\n", "hdr")
            for f, name in [
                ("xgb_model.pkl", "XGBoost"), ("lgb_model.pkl", "LightGBM"),
                ("catboost_model.pkl", "CatBoost"), ("rf_model.pkl", "RF"),
                ("lstm_model.pt", "LSTM"), ("tcn_model.pt", "TCN"),
                ("nbeats_model.pt", "N-BEATS"), ("nhits_model.pt", "N-HiTS"),
                ("meta_model.pkl", "Meta"), ("scaler.pkl", "Scaler"),
                ("ensemble_config.json", "Config"),
            ]:
                exists = os.path.exists(os.path.join(BASE, f))
                tag = "ok" if exists else "warn"
                out.insert("end", f"    {'✓' if exists else '✗'} {name}\n", tag)
                win.update()

            out.insert("end", "\n  📜 SCRIPTS\n", "hdr")
            for f, name in [
                ("live_trading_bridge.py", "XAUUSD Bridge"),
                ("live_bridge_xausgd.py",  "XAUSGD Bridge"),
                ("features.py",            "Features"),
                ("smc_logic.py",           "SMC"),
                ("news_guard.py",          "News Guard"),
                ("correlation_guard.py",   "Correlation"),
                ("risk_manager.py",        "Risk Manager"),
                ("walkforward_backtest.py","Backtest"),
                ("trading_dashboard.py",   "Dashboard"),
            ]:
                exists = os.path.exists(os.path.join(BASE, f))
                tag = "ok" if exists else "err"
                out.insert("end", f"    {'✓' if exists else '✗'} {name}\n", tag)
                if not exists:
                    issues.append(f"Missing script: {f}")
                win.update()

            out.insert("end", "\n  ═══════════════════\n", "purple")
            if issues:
                out.insert("end", f"  ⚠ {len(issues)} ISSUES\n", "warn")
                for i in issues[:8]:
                    out.insert("end", f"  • {i}\n", "err")
            else:
                out.insert("end", "  ✅ ALL SYSTEMS HEALTHY\n", "ok")
            out.see("end")

        threading.Thread(target=run, daemon=True).start()

    # ── Install packages ──────────────────────────────────────────
    def _install_packages(self):
        win = tk.Toplevel(self.root)
        win.title("Install Packages")
        win.configure(bg=BG1)
        win.geometry("600x500")
        oh, ih = bordered(win)
        oh.pack(fill="x", padx=8, pady=8)
        tk.Label(ih, text=" 📦 Install Packages", fg=PURPLE, bg=BG2,
                 font=FH).pack(anchor="w", padx=8, pady=5)
        out = scrolledtext.ScrolledText(win, bg=BG0, fg=GREY_LT, font=FB,
                                         wrap="word", relief="flat")
        out.pack(fill="both", expand=True, padx=8, pady=4)
        out.tag_config("ok", foreground=GREEN)
        out.tag_config("err", foreground=RED_HI)
        out.tag_config("hdr", foreground=BLUE)
        bf = tk.Frame(win, bg=BG1); bf.pack(fill="x", padx=8, pady=4)

        def install():
            pkgs = ["MetaTrader5", "pandas", "numpy", "scikit-learn",
                    "xgboost", "lightgbm", "catboost", "torch",
                    "yfinance", "joblib", "requests", "matplotlib"]
            out.insert("end", "\n  Installing...\n\n", "hdr")
            for pkg in pkgs:
                out.insert("end", f"  {pkg}... "); out.see("end"); win.update()
                try:
                    r = subprocess.run(
                        [sys.executable, "-m", "pip", "install", pkg, "-q"],
                        capture_output=True, text=True, timeout=120)
                    tag = "ok" if r.returncode == 0 else "err"
                    out.insert("end", "OK\n" if r.returncode == 0 else "FAILED\n", tag)
                except Exception as e:
                    out.insert("end", f"ERROR: {e}\n", "err")
                win.update()
            out.insert("end", "\n  Done! Run Health Check to verify.\n", "hdr")
            out.see("end")

        tk.Button(bf, text=" Install All ", bg=GREEN, fg=WHITE, font=FBB,
                  relief="flat",
                  command=lambda: threading.Thread(target=install, daemon=True).start()
                  ).pack(side="left", padx=4)
        tk.Button(bf, text=" Close ", bg=RED_MID, fg=WHITE, font=FBB,
                  relief="flat", command=win.destroy).pack(side="right", padx=4)

    # ── Guards ────────────────────────────────────────────────────
    def _popup_run(self, title, color, func):
        win = tk.Toplevel(self.root)
        win.title(title); win.configure(bg=BG1); win.geometry("640x420")
        oh, ih = bordered(win); oh.pack(fill="x", padx=8, pady=8)
        tk.Label(ih, text=f" {title}", fg=color, bg=BG2,
                 font=FH).pack(anchor="w", padx=8, pady=5)
        out = scrolledtext.ScrolledText(win, bg=BG0, fg=GREY_LT, font=FB,
                                         wrap="word", relief="flat")
        out.pack(fill="both", expand=True, padx=8, pady=4)
        bf = tk.Frame(win, bg=BG1); bf.pack(fill="x", padx=8, pady=4)
        tk.Button(bf, text=" Close ", bg=RED_MID, fg=WHITE, font=FBB,
                  relief="flat", command=win.destroy).pack(side="right", padx=4)
        threading.Thread(
            target=func, args=(out, win), daemon=True).start()

    def _check_news(self):
        def run(out, win):
            out.insert("end", "\n  Checking news guard...\n\n")
            sp = os.path.join(BASE, "news_guard.py")
            if os.path.exists(sp):
                try:
                    r = subprocess.run(
                        [sys.executable, sp], cwd=BASE,
                        capture_output=True, text=True, timeout=30)
                    out.insert("end", r.stdout or "No output\n")
                    if r.stderr:
                        out.insert("end", f"Errors:\n{r.stderr}\n")
                except Exception as e:
                    out.insert("end", f"Error: {e}\n")
            ng = _rj("news_guard.json")
            if ng:
                if ng.get("blocked"):
                    out.insert("end", f"\n  ⛔ BLOCKED: {ng.get('reason', '')}\n")
                elif ng.get("reduced"):
                    out.insert("end", f"\n  ⚠ CAUTION: {ng.get('reason', '')}\n")
                else:
                    out.insert("end", "\n  ✅ CLEAR — No high impact news\n")
            else:
                out.insert("end", "\n  No news_guard.json found\n")
            out.see("end")
        self._popup_run("📰 News Guard", GOLD, run)

    def _check_smc(self):
        def run(out, win):
            out.insert("end", "\n  Running SMC analysis...\n\n")
            try:
                sys.path.insert(0, BASE)
                from smc_logic import get_bias
                result = get_bias()
                d = result.get("direction", 0)
                label = "BUY" if d > 0 else ("SELL" if d < 0 else "NEUTRAL")
                out.insert("end", f"  Direction: {label}\n")
                out.insert("end", f"  Score: {result.get('score', 0)}\n\n")
                out.insert("end", "  Details:\n")
                for det in result.get("details", []):
                    out.insert("end", f"    • {det}\n")
            except Exception as e:
                out.insert("end", f"  Error: {e}\n")
            out.see("end")
        self._popup_run("📊 SMC Analysis", BLUE, run)

    def _check_correlation(self):
        def run(out, win):
            out.insert("end", "\n  Running correlation check...\n\n")
            sp = os.path.join(BASE, "correlation_guard.py")
            if os.path.exists(sp):
                try:
                    r = subprocess.run(
                        [sys.executable, sp], cwd=BASE,
                        capture_output=True, text=True, timeout=30)
                    out.insert("end", r.stdout or "No output\n")
                except Exception as e:
                    out.insert("end", f"  Error: {e}\n")
            else:
                out.insert("end", "  correlation_guard.py not found\n")
            out.see("end")
        self._popup_run("🔗 Correlation", BLUE, run)

    def _check_regime(self):
        def run(out, win):
            out.insert("end", "\n  Reading market regime...\n\n")
            sig = _rj("current_signal.json")
            if sig:
                out.insert("end", f"  Regime:     {sig.get('current_regime', '?')}\n")
                out.insert("end", f"  Signal:     {sig.get('signal', '?')}\n")
                out.insert("end", f"  Confidence: {sig.get('confidence', 0)*100:.1f}%\n")
                out.insert("end", f"  Strategy:   {sig.get('strategy_used', '?')}\n")
                out.insert("end", f"  Version:    {sig.get('version', '?')}\n")
                out.insert("end", f"  Time:       {sig.get('timestamp', '?')}\n")
            else:
                out.insert("end", "  No current_signal.json found\n")
            rc = _rj("regime_cache.json")
            if rc:
                out.insert("end", f"\n  Cached Regime: {rc.get('regime', '?')}\n")
                out.insert("end", f"  ADX:           {rc.get('adx', 0):.1f}\n")
                out.insert("end", f"  BBW:           {rc.get('bbw', 0):.4f}\n")
                out.insert("end", f"  Hurst:         {rc.get('hurst', 0):.3f}\n")
                out.insert("end", f"  Risk mult:     {rc.get('risk_mult', 0):.2f}\n")
            out.see("end")
        self._popup_run("📈 Market Regime", BLUE, run)

    # ── Utilities ─────────────────────────────────────────────────
    def _show_kelly(self):
        ec   = _rj("ensemble_config.json")
        accs = ec.get("model_accuracies", {})
        if not accs:
            messagebox.showinfo("Kelly / Model Performance",
                                "No ensemble_config.json found.\nRun training first.")
            return
        lines = [f"Best Strategy: {ec.get('best_strategy', '?')}",
                 f"Best Accuracy: {ec.get('best_accuracy', 0)*100:.1f}%",
                 "", "Model Accuracies (sorted):"]
        for m, a in sorted(accs.items(), key=lambda x: -x[1]):
            lines.append(f"  {m:20s}: {a*100:.1f}%")
        lines += [
            "",
            "Kelly Fraction guide:",
            "  Win rate × avg_win / avg_loss = edge",
            "  Half-Kelly = edge × 0.5",
            "  Recommended: use risk_manager.py calculate_risk()",
        ]
        messagebox.showinfo("Model Performance & Kelly", "\n".join(lines))

    def _show_compounding(self):
        lines = ["Compounding from £500:", ""]
        for rate in [0.005, 0.01, 0.015, 0.02]:
            lines.append(f"{rate*100:.1f}%/day:")
            for days in [7, 14, 30, 60, 90]:
                val = 500 * (1 + rate) ** days
                lines.append(f"  {days:3d}d → £{val:,.2f}")
            lines.append("")
        messagebox.showinfo("Compounding Calculator (from £500)", "\n".join(lines))

    def _show_account(self):
        ok, mt5 = mt5_init()
        if not ok or mt5 is None:
            messagebox.showerror("Error", "Cannot connect to MT5")
            return
        try:
            info = mt5.account_info()
            if not info:
                messagebox.showerror("Error", "No account info"); return
            usd_pos = mt5.positions_get(symbol="XAUUSD") or []
            sgd_pos = mt5.positions_get(symbol="XAUSGD") or []
            mode    = "DEMO" if info.trade_mode == 0 else "LIVE"
            msg = (
                f"Login:       {info.login}\n"
                f"Server:      {info.server}\n"
                f"Mode:        {mode}\n"
                f"Balance:     £{info.balance:.2f}\n"
                f"Equity:      £{info.equity:.2f}\n"
                f"Profit:      £{info.profit:+.2f}\n"
                f"Margin Free: £{info.margin_free:.2f}\n"
                f"Leverage:    1:{info.leverage}\n\n"
                f"Open Positions:\n"
                f"  XAUUSD: {len(usd_pos)}\n"
                f"  XAUSGD: {len(sgd_pos)}"
            )
            messagebox.showinfo("Account Info", msg)
        finally:
            try:
                mt5.shutdown()
            except Exception:
                pass

    def _view_log(self):
        log_path = os.path.join(BASE, "trading_log.json")
        if os.path.exists(log_path):
            os.startfile(log_path)
        else:
            messagebox.showinfo("Trade Log", "No trading_log.json found yet")

    def _start_full_system(self):
        win = tk.Toplevel(self.root)
        win.title("Full System Launch")
        win.configure(bg=BG1)
        win.geometry("660x580")
        oh, ih = bordered(win); oh.pack(fill="x", padx=8, pady=8)
        tk.Label(ih, text=" ⚡ Full System Startup", fg=GREEN, bg=BG2,
                 font=FH).pack(anchor="w", padx=8, pady=5)
        out = scrolledtext.ScrolledText(win, bg=BG0, fg=GREY_LT, font=FB,
                                         wrap="word", relief="flat")
        out.pack(fill="both", expand=True, padx=8, pady=4)
        for tag, col in [("ok", GREEN), ("err", RED_HI),
                          ("warn", GOLD), ("hdr", BLUE)]:
            out.tag_config(tag, foreground=col)
        bf = tk.Frame(win, bg=BG1); bf.pack(fill="x", padx=8, pady=4)
        tk.Button(bf, text=" Close ", bg=RED_MID, fg=WHITE, font=FBB,
                  relief="flat", command=win.destroy).pack(side="right", padx=4)

        def run():
            out.insert("end", "\n  ═══ STARTUP CHECK ═══\n\n", "hdr")
            # MT5
            out.insert("end", "  [MT5] ... ")
            ok, mt5 = mt5_init()
            if ok and mt5:
                info = mt5.account_info()
                out.insert("end", f"OK (£{info.balance:.2f})\n", "ok")
                try: mt5.shutdown()
                except: pass
            else:
                out.insert("end", "FAILED\n", "err")
            win.update()

            # Scripts
            out.insert("end", "\n  [SCRIPTS]\n", "hdr")
            for s in ["live_trading_bridge.py", "live_bridge_xausgd.py",
                       "features.py", "smc_logic.py", "news_guard.py",
                       "risk_manager.py"]:
                exists = os.path.exists(os.path.join(BASE, s))
                tag = "ok" if exists else "err"
                out.insert("end", f"    {'✓' if exists else '✗'} {s}\n", tag)
                win.update()

            out.insert("end", "\n  ═══ READY ═══\n", "hdr")
            out.insert("end", "  Use XAUUSD/XAUSGD buttons to trade.\n", "ok")
            out.see("end")

        threading.Thread(target=run, daemon=True).start()

    # ── Status refresh ────────────────────────────────────────────
    def _start_status_refresh(self):
        self._refresh_status()

    def _refresh_status(self):
        try:
            sig = _rj("current_signal.json")
            if sig:
                s  = sig.get("signal", "?")
                c  = sig.get("confidence", 0)
                rg = sig.get("current_regime", "?")
                col = GREEN if s == "BUY" else (RED_HI if s == "SELL" else GREY_MID)
                self.signal_lbl.config(
                    text=f"  Signal: {s} ({c:.1%}) | Regime: {rg} | {sig.get('strategy_used', '?')}",
                    fg=col)
            ng = _rj("news_guard.json")
            if ng:
                if ng.get("blocked"):
                    self.news_lbl.config(text=f"  News: ⛔ BLOCKED", fg=RED_HI)
                elif ng.get("reduced"):
                    self.news_lbl.config(text=f"  News: ⚠ CAUTION", fg=GOLD)
                else:
                    self.news_lbl.config(text="  News: ✅ CLEAR", fg=GREEN)
            cs = _rj("correlation_status.json")
            if cs:
                if cs.get("kill_switch"):
                    self.corr_lbl.config(text="  Correlations: KILL SWITCH", fg=RED_HI)
                else:
                    self.corr_lbl.config(text="  Correlations: Normal", fg=GREEN)
            alive = [k for k, v in self.active_procs.items()
                     if isinstance(v, str) or (hasattr(v, "poll") and v.poll() is None)]
            if alive:
                self.proc_lbl.config(text=f"  Active: {', '.join(alive)}", fg=GREEN)
            else:
                self.proc_lbl.config(text="  Processes: none", fg=GREY_DIM)
            # Account info
            ok, mt5 = mt5_init()
            if ok and mt5:
                info = mt5.account_info()
                if info:
                    self.acc_lbl.config(
                        text=f"£{info.balance:.2f} | Eq:£{info.equity:.2f}")
                try: mt5.shutdown()
                except: pass
        except Exception:
            pass
        self.root.after(5000, self._refresh_status)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
