"""
AlmostFinishedBot - Interactive Chart Dashboard v1
Two tabs: LIVE CHART + BACKTEST REPLAY

LIVE CHART:
  - Period buttons: 1H, 2H, 3H, 6H, 12H, 24H, 1W, 2W, 3W, 4W
  - Auto-updates chart every 1 second (redraws from cached data)
  - Background data download every 15 seconds
  - Candlestick display with price overlay

BACKTEST REPLAY:
  - Must click 'Start Replay' before any P&L is shown
  - Speed slider (1x to 100x) to control replay speed
  - Running P&L counter with live balance
  - BUY/SELL markers appear as replay progresses
  - Entry-to-exit lines for completed trades
  - Equity curve builds in real-time
  - Progress bar with bar counter
"""
import os, sys, json, time, threading, datetime
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
sys.path.insert(0, BASE)

# ── Theme (matches Control Center dark theme) ─────────────────────
BG0    = "#0f0f0f"
BG1    = "#141414"
BG2    = "#1a1a1a"
BG3    = "#252525"
BORDER = "#2a2a2a"
WHITE  = "#e0e0e0"
GREY   = "#888888"
GREEN  = "#22cc66"
RED    = "#ee3344"
GOLD   = "#ffcc44"
BLUE   = "#4488ff"
ORANGE = "#ff8800"
CYAN   = "#00ccff"


def load_backtest():
    p = os.path.join(BASE, "backtest_results.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def download_live_data(period="5d", interval="1h"):
    try:
        import yfinance as yf
        df = yf.download("GC=F", period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df = df.dropna(subset=["open","high","low","close"])
        return df
    except Exception as e:
        print(f"  Download error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD CLASS
# ══════════════════════════════════════════════════════════════════
class ChartDashboard:
    def __init__(self, start_tab=None):
        self.root = tk.Tk()
        self.root.title("AlmostFinishedBot  |  Chart Dashboard")
        self.root.configure(bg=BG0)
        self.root.geometry("1320x850")
        self.root.minsize(1000, 650)
        self.start_tab = start_tab or "live"

        # State -- live chart
        self.live_period = "24H"
        self.live_df = None
        self.live_download_lock = threading.Lock()
        self.downloading = False
        self.last_download = 0

        # State -- backtest replay
        self.bt_data = load_backtest()
        self.replay_running = False
        self.replay_paused = False
        self.replay_idx = 0
        self.replay_speed = 5
        self.replay_balance = 100.0
        self.replay_trades_shown = []
        self.replay_equity = []
        self.replay_trade_ptr = 0
        self.replay_started_once = False

        # Matplotlib setup
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        self.plt = plt
        self.mdates = mdates
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.Figure = Figure
        self.Line2D = Line2D

        self._build_ui()
        self._initial_download()
        self._tick_1sec()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.replay_running = False
        self.root.destroy()

    # ══════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG2, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  ALMOSTFINISHEDBOT  |  CHART DASHBOARD",
                 fg=RED, bg=BG2, font=("Consolas", 13, "bold")).pack(side="left", padx=8, pady=8)
        self.clock_lbl = tk.Label(hdr, text="", fg=GREY, bg=BG2, font=("Consolas", 10))
        self.clock_lbl.pack(side="right", padx=10)

        # ── Tab buttons ───────────────────────────────────────────
        tab_bar = tk.Frame(self.root, bg=BG1)
        tab_bar.pack(fill="x")
        self.tab_var = tk.StringVar(value=self.start_tab)

        for label, val in [("  LIVE CHART  ", "live"), ("  BACKTEST REPLAY  ", "backtest")]:
            b = tk.Radiobutton(tab_bar, text=label, variable=self.tab_var, value=val,
                               indicatoron=0, bg=BG2, fg=WHITE, selectcolor=BG3,
                               activebackground=BG3, activeforeground=WHITE,
                               font=("Consolas", 11, "bold"), relief="flat", bd=0,
                               padx=20, pady=8, command=self._switch_tab)
            b.pack(side="left", padx=1)

        # ── Main container ────────────────────────────────────────
        self.main = tk.Frame(self.root, bg=BG0)
        self.main.pack(fill="both", expand=True)

        self._build_live_panel()
        self._build_backtest_panel()
        self._switch_tab()

    def _switch_tab(self):
        tab = self.tab_var.get()
        if tab == "live":
            self.bt_frame.pack_forget()
            self.live_frame.pack(fill="both", expand=True)
        else:
            self.live_frame.pack_forget()
            self.bt_frame.pack(fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════
    #  1-SECOND MASTER TICK
    # ══════════════════════════════════════════════════════════════
    def _tick_1sec(self):
        # Update clock
        now = datetime.datetime.now(datetime.timezone.utc)
        self.clock_lbl.config(text=now.strftime("%H:%M:%S UTC"))

        # Live chart: redraw every second from cached data
        if self.tab_var.get() == "live" and self.live_df is not None:
            self._draw_live_chart()

        # Background download every 15 seconds
        elapsed = time.time() - self.last_download
        if elapsed >= 15 and not self.downloading:
            threading.Thread(target=self._bg_download, daemon=True).start()

        self.root.after(1000, self._tick_1sec)

    def _initial_download(self):
        self.live_status.config(text="  Downloading...")
        threading.Thread(target=self._bg_download, daemon=True).start()

    def _bg_download(self):
        if self.downloading:
            return
        self.downloading = True
        try:
            yf_period, yf_interval = self._period_to_yf(self.live_period)
            df = download_live_data(yf_period, yf_interval)
            if df is not None and len(df) > 0:
                self.live_df = self._trim_to_period(df, self.live_period)
                self.last_download = time.time()
                self.root.after(0, lambda: self.live_status.config(
                    text=f"  {len(self.live_df)} candles  |  {datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}"))
            else:
                self.root.after(0, lambda: self.live_status.config(text="  Download failed"))
        except Exception:
            pass
        finally:
            self.downloading = False

    # ══════════════════════════════════════════════════════════════
    #  LIVE CHART PANEL
    # ══════════════════════════════════════════════════════════════
    def _build_live_panel(self):
        self.live_frame = tk.Frame(self.main, bg=BG0)

        # ── Time period buttons ───────────────────────────────────
        btn_bar = tk.Frame(self.live_frame, bg=BG1)
        btn_bar.pack(fill="x", padx=4, pady=(4,0))
        tk.Label(btn_bar, text="  Period:", fg=GREY, bg=BG1,
                 font=("Consolas", 10)).pack(side="left", padx=(6,2))

        self.time_btns = {}
        for p in ["1H","2H","3H","6H","12H","24H","1W","2W","3W","4W"]:
            b = tk.Button(btn_bar, text=f" {p} ", bg=BG2, fg=WHITE,
                         font=("Consolas", 10, "bold"), relief="flat",
                         activebackground=BG3, activeforeground=GOLD, cursor="hand2",
                         command=lambda pp=p: self._set_period(pp))
            b.pack(side="left", padx=2, pady=4)
            self.time_btns[p] = b

        self.live_status = tk.Label(btn_bar, text="  Loading...", fg=GOLD, bg=BG1,
                                    font=("Consolas", 9))
        self.live_status.pack(side="right", padx=10)
        self._highlight_btn("24H")

        # ── Price display ─────────────────────────────────────────
        price_bar = tk.Frame(self.live_frame, bg=BG1)
        price_bar.pack(fill="x", padx=4, pady=(2,0))
        self.price_lbl = tk.Label(price_bar, text="  XAUUSD: ---", fg=GREEN, bg=BG1,
                                  font=("Consolas", 20, "bold"))
        self.price_lbl.pack(side="left", padx=10, pady=4)
        self.change_lbl = tk.Label(price_bar, text="", fg=GREY, bg=BG1,
                                   font=("Consolas", 12))
        self.change_lbl.pack(side="left", padx=10)

        # Pulse indicator (flashes to show updates)
        self.pulse_lbl = tk.Label(price_bar, text="  \u25cf", fg=GREEN, bg=BG1,
                                  font=("Consolas", 10))
        self.pulse_lbl.pack(side="right", padx=10)
        self._pulse_state = True

        # ── Chart ─────────────────────────────────────────────────
        chart_box = tk.Frame(self.live_frame, bg=BG0)
        chart_box.pack(fill="both", expand=True, padx=4, pady=4)

        self.live_fig = self.Figure(figsize=(12, 5), facecolor=BG0)
        self.live_ax = self.live_fig.add_subplot(111)
        self.live_canvas = self.FigureCanvasTkAgg(self.live_fig, master=chart_box)
        self.live_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _highlight_btn(self, selected):
        for p, b in self.time_btns.items():
            if p == selected:
                b.config(bg=BLUE, fg=WHITE)
            else:
                b.config(bg=BG2, fg=WHITE)

    def _set_period(self, period):
        self.live_period = period
        self._highlight_btn(period)
        self.live_status.config(text=f"  Loading {period}...")
        self.last_download = 0  # Force immediate re-download
        threading.Thread(target=self._bg_download, daemon=True).start()

    def _period_to_yf(self, period):
        return {
            "1H":  ("1d",  "5m"),   "2H":  ("1d",  "5m"),
            "3H":  ("1d",  "5m"),   "6H":  ("5d",  "15m"),
            "12H": ("5d",  "15m"),  "24H": ("5d",  "15m"),
            "1W":  ("7d",  "1h"),   "2W":  ("14d", "1h"),
            "3W":  ("21d", "1h"),   "4W":  ("1mo", "1h"),
        }.get(period, ("5d", "1h"))

    def _trim_to_period(self, df, period):
        if df is None or len(df) == 0:
            return df
        now = df.index[-1]
        h_map = {"1H":1,"2H":2,"3H":3,"6H":6,"12H":12,"24H":24,
                 "1W":168,"2W":336,"3W":504,"4W":672}
        hrs = h_map.get(period, 24)
        cutoff = now - pd.Timedelta(hours=hrs)
        return df[df.index >= cutoff]

    def _draw_live_chart(self):
        df = self.live_df
        if df is None or len(df) == 0:
            return

        ax = self.live_ax
        ax.clear()
        ax.set_facecolor(BG2)

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        open_ = df["open"].values.astype(float)
        times = df.index

        # Candlestick bars
        for i in range(len(close)):
            c = GREEN if close[i] >= open_[i] else RED
            ax.plot([times[i], times[i]], [low[i], high[i]], color=c, linewidth=0.5, alpha=0.6)
            ax.plot([times[i], times[i]],
                   [min(open_[i], close[i]), max(open_[i], close[i])],
                   color=c, linewidth=2.5, alpha=0.9)

        # Close price line
        ax.plot(times, close, color=WHITE, linewidth=0.6, alpha=0.35)

        # Current price horizontal line
        last_p = close[-1]
        ax.axhline(y=last_p, color=GOLD, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.text(times[-1], last_p, f"  ${last_p:,.2f}", color=GOLD,
               fontsize=8, va="center", fontfamily="Consolas")

        # EMA 20 overlay
        if len(close) >= 20:
            ema = np.full(len(close), np.nan)
            ema[19] = np.mean(close[:20])
            k = 2.0/21
            for i in range(20, len(close)):
                ema[i] = close[i]*k + ema[i-1]*(1-k)
            ax.plot(times, ema, color=CYAN, linewidth=0.8, alpha=0.6, label="EMA 20")

        # Style
        ax.tick_params(colors=GREY, labelsize=8)
        ax.grid(True, alpha=0.06, color=GREY)
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['bottom','left']: ax.spines[s].set_color(BORDER)
        ax.set_ylabel("Price (USD)", color=GREY, fontsize=9)

        # X-axis format based on period
        if self.live_period in ("1H","2H","3H","6H"):
            ax.xaxis.set_major_formatter(self.mdates.DateFormatter("%H:%M"))
        elif self.live_period in ("12H","24H"):
            ax.xaxis.set_major_formatter(self.mdates.DateFormatter("%d %H:%M"))
        else:
            ax.xaxis.set_major_formatter(self.mdates.DateFormatter("%b %d"))

        self.live_fig.tight_layout()
        self.live_canvas.draw_idle()

        # Update price labels
        change_pct = (close[-1] - close[0]) / close[0] * 100
        ch_col = GREEN if change_pct >= 0 else RED
        self.price_lbl.config(text=f"  XAUUSD: ${last_p:,.2f}")
        self.change_lbl.config(text=f"{change_pct:+.2f}% ({self.live_period})", fg=ch_col)

        # Pulse indicator toggle
        self._pulse_state = not self._pulse_state
        self.pulse_lbl.config(fg=GREEN if self._pulse_state else BG1)

    # ══════════════════════════════════════════════════════════════
    #  BACKTEST REPLAY PANEL
    # ══════════════════════════════════════════════════════════════
    def _build_backtest_panel(self):
        self.bt_frame = tk.Frame(self.main, bg=BG0)

        # ── Controls bar ──────────────────────────────────────────
        ctrl = tk.Frame(self.bt_frame, bg=BG1)
        ctrl.pack(fill="x", padx=4, pady=(4,0))

        self.replay_btn = tk.Button(ctrl, text="  \u25b6  Start Replay  ", bg=GREEN, fg=BG0,
                                    font=("Consolas", 11, "bold"), relief="flat", cursor="hand2",
                                    command=self._toggle_replay)
        self.replay_btn.pack(side="left", padx=6, pady=6)

        self.reset_btn = tk.Button(ctrl, text="  \u21bb Reset  ", bg=BG3, fg=WHITE,
                                   font=("Consolas", 10), relief="flat", cursor="hand2",
                                   command=self._reset_replay)
        self.reset_btn.pack(side="left", padx=4, pady=6)

        # Speed slider
        speed_frame = tk.Frame(ctrl, bg=BG1)
        speed_frame.pack(side="left", padx=(20,0))
        tk.Label(speed_frame, text="Speed:", fg=GREY, bg=BG1,
                 font=("Consolas", 10)).pack(side="left", padx=4)

        self.speed_var = tk.IntVar(value=5)
        self.speed_slider = tk.Scale(speed_frame, from_=1, to=100, orient="horizontal",
                                     variable=self.speed_var, bg=BG1, fg=WHITE,
                                     highlightbackground=BG1, troughcolor=BG3,
                                     font=("Consolas", 8), length=180, sliderlength=18,
                                     showvalue=0, command=self._on_speed)
        self.speed_slider.pack(side="left", padx=2)

        self.speed_lbl = tk.Label(speed_frame, text="5x", fg=GOLD, bg=BG1,
                                  font=("Consolas", 11, "bold"), width=5)
        self.speed_lbl.pack(side="left")

        # P&L display (right side -- hidden content until replay starts)
        pnl_box = tk.Frame(ctrl, bg=BG1)
        pnl_box.pack(side="right", padx=10)
        self.pnl_lbl = tk.Label(pnl_box, text="  \u25c0 Click Start to begin  ",
                                fg=GREY, bg=BG1, font=("Consolas", 15, "bold"))
        self.pnl_lbl.pack(side="right", padx=4, pady=4)
        self.trades_count_lbl = tk.Label(pnl_box, text="", fg=GREY, bg=BG1,
                                         font=("Consolas", 9))
        self.trades_count_lbl.pack(side="right", padx=4)

        # ── Progress bar ──────────────────────────────────────────
        prog_frame = tk.Frame(self.bt_frame, bg=BG1)
        prog_frame.pack(fill="x", padx=4, pady=(2,0))

        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("BT.Horizontal.TProgressbar", troughcolor=BG3,
                        background=GREEN, darkcolor=GREEN, lightcolor=GREEN, bordercolor=BG1)
        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var,
                                             maximum=100, style="BT.Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", padx=6, pady=3)

        self.progress_lbl = tk.Label(prog_frame, text="", fg=GREY, bg=BG1,
                                     font=("Consolas", 9))
        self.progress_lbl.pack(side="right", padx=6)

        # ── Chart area (2 subplots: price + equity) ───────────────
        chart_box = tk.Frame(self.bt_frame, bg=BG0)
        chart_box.pack(fill="both", expand=True, padx=4, pady=4)

        self.bt_fig = self.Figure(figsize=(12, 6), facecolor=BG0)
        gs = self.bt_fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        self.bt_ax1 = self.bt_fig.add_subplot(gs[0])
        self.bt_ax2 = self.bt_fig.add_subplot(gs[1], sharex=self.bt_ax1)
        self.bt_canvas = self.FigureCanvasTkAgg(self.bt_fig, master=chart_box)
        self.bt_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Draw base chart (price only, no trades)
        self._draw_bt_base()

    def _draw_bt_base(self):
        """Draw price chart ONLY -- no P&L, no trades, no equity. User must click Start."""
        ax1 = self.bt_ax1; ax1.clear(); ax1.set_facecolor(BG2)
        ax2 = self.bt_ax2; ax2.clear(); ax2.set_facecolor(BG2)

        if self.bt_data is None:
            ax1.text(0.5, 0.5, "No backtest data found\n\nRun walkforward_backtest.py first",
                    transform=ax1.transAxes, ha="center", va="center",
                    color=GREY, fontsize=13, fontfamily="Consolas")
            ax2.text(0.5, 0.5, "", transform=ax2.transAxes)
            self.bt_canvas.draw()
            return

        eq = self.bt_data.get("equity_curve", [])
        if not eq:
            return

        # Store for replay
        self.bt_times  = [pd.Timestamp(e["time"]) for e in eq]
        self.bt_prices = [e["price"] for e in eq]
        self.bt_bals   = [e["balance"] for e in eq]
        self.bt_bars   = [e["bar"] for e in eq]
        self.bt_trades = self.bt_data.get("trades", [])

        # Dim price line (full history visible but no annotations)
        ax1.plot(self.bt_times, self.bt_prices, color=GREY, linewidth=0.7, alpha=0.5)
        ax1.set_ylabel("Gold (USD)", color=GREY, fontsize=9)
        ax1.tick_params(colors=GREY, labelsize=8)
        ax1.grid(True, alpha=0.06)
        for s in ['top','right']: ax1.spines[s].set_visible(False)
        for s in ['bottom','left']: ax1.spines[s].set_color(BORDER)

        # Empty equity panel
        ax2.axhline(y=100, color=GREY, linestyle="--", linewidth=0.5, alpha=0.3)
        ax2.set_ylabel("Balance (GBP)", color=GREY, fontsize=9)
        ax2.set_ylim(95, 105)
        ax2.tick_params(colors=GREY, labelsize=8)
        ax2.grid(True, alpha=0.06)
        for s in ['top','right']: ax2.spines[s].set_visible(False)
        for s in ['bottom','left']: ax2.spines[s].set_color(BORDER)

        ax2.xaxis.set_major_formatter(self.mdates.DateFormatter("%b %d"))
        self.bt_fig.autofmt_xdate()
        self.bt_fig.tight_layout()
        self.bt_canvas.draw()

    def _on_speed(self, val):
        s = int(val)
        self.replay_speed = s
        self.speed_lbl.config(text=f"{s}x")

    def _toggle_replay(self):
        if self.bt_data is None:
            return
        if self.replay_running:
            # Pause
            self.replay_running = False
            self.replay_paused = True
            self.replay_btn.config(text="  \u25b6  Resume  ", bg=GOLD, fg=BG0)
        elif self.replay_paused:
            # Resume
            self.replay_running = True
            self.replay_paused = False
            self.replay_btn.config(text="  \u23f8  Pause  ", bg=ORANGE, fg=BG0)
            self._replay_tick()
        else:
            # Fresh start
            self._reset_state()
            self.replay_started_once = True
            self.replay_running = True
            self.replay_btn.config(text="  \u23f8  Pause  ", bg=ORANGE, fg=BG0)
            self._replay_tick()

    def _reset_replay(self):
        self.replay_running = False
        self.replay_paused = False
        self.replay_started_once = False
        self._reset_state()
        self.replay_btn.config(text="  \u25b6  Start Replay  ", bg=GREEN, fg=BG0)
        self.pnl_lbl.config(text="  \u25c0 Click Start to begin  ", fg=GREY)
        self.trades_count_lbl.config(text="")
        self.progress_var.set(0)
        self.progress_lbl.config(text="")
        self._draw_bt_base()

    def _reset_state(self):
        self.replay_idx = 0
        self.replay_balance = 100.0
        self.replay_trades_shown = []
        self.replay_equity = []
        self.replay_trade_ptr = 0

    def _replay_tick(self):
        if not self.replay_running:
            return

        eq = self.bt_data.get("equity_curve", [])
        trades = self.bt_data.get("trades", [])
        n_eq = len(eq)

        if self.replay_idx >= n_eq:
            # Finished
            self.replay_running = False
            self.replay_btn.config(text="  \u2713  Finished  ", bg=BLUE, fg=WHITE)
            stats = self.bt_data.get("stats", {})
            self.progress_lbl.config(
                text=f"DONE | Win: {stats.get('win_rate',0):.0f}% | "
                     f"PF: {stats.get('profit_factor',0):.2f} | "
                     f"Sharpe: {stats.get('sharpe',0):.1f}")
            return

        # Advance by speed factor
        steps = min(self.replay_speed, n_eq - self.replay_idx)
        end_idx = self.replay_idx + steps

        for i in range(self.replay_idx, end_idx):
            t = pd.Timestamp(eq[i]["time"])
            bal = eq[i]["balance"]
            bar = eq[i]["bar"]
            self.replay_equity.append((t, bal))
            self.replay_balance = bal

            # Check for new trades that start at or before this bar
            while self.replay_trade_ptr < len(trades):
                tr = trades[self.replay_trade_ptr]
                if tr["entry_bar"] <= bar:
                    self.replay_trades_shown.append(tr)
                    self.replay_trade_ptr += 1
                else:
                    break

        self.replay_idx = end_idx

        # Update chart
        self._draw_bt_replay()

        # Update P&L
        pnl = self.replay_balance - 100.0
        pnl_col = GREEN if pnl >= 0 else RED
        self.pnl_lbl.config(
            text=f"  P&L: \u00a3{pnl:+.2f}  ({pnl:+.1f}%)  ",
            fg=pnl_col)

        # Count completed trades
        cur_bar = eq[min(self.replay_idx - 1, n_eq - 1)]["bar"]
        completed = [t for t in self.replay_trades_shown if t["exit_bar"] <= cur_bar]
        wins = sum(1 for t in completed if t.get("pnl_pct", 0) > 0)
        wr = (wins / len(completed) * 100) if completed else 0
        self.trades_count_lbl.config(
            text=f"Trades: {len(completed)} | Wins: {wins} ({wr:.0f}%) | "
                 f"Bal: \u00a3{self.replay_balance:.2f}")

        # Progress
        pct = self.replay_idx / n_eq * 100
        self.progress_var.set(pct)
        self.progress_lbl.config(text=f"{pct:.0f}% | Bar {self.replay_idx}/{n_eq}")

        # Next tick (faster = shorter delay)
        delay = max(10, 150 // max(self.replay_speed, 1))
        self.root.after(delay, self._replay_tick)

    def _draw_bt_replay(self):
        """Redraw backtest chart with current replay progress."""
        ax1 = self.bt_ax1; ax1.clear(); ax1.set_facecolor(BG2)
        ax2 = self.bt_ax2; ax2.clear(); ax2.set_facecolor(BG2)

        # Full price (dimmed)
        ax1.plot(self.bt_times, self.bt_prices, color=GREY, linewidth=0.4, alpha=0.2)

        # Revealed price (bright white)
        vis_t = self.bt_times[:self.replay_idx]
        vis_p = self.bt_prices[:self.replay_idx]
        if vis_t:
            ax1.plot(vis_t, vis_p, color=WHITE, linewidth=0.9, alpha=0.85)
            # Playhead line
            ax1.axvline(x=vis_t[-1], color=GOLD, linewidth=0.5, alpha=0.25)

        # Current bar number
        cur_bar = self.bt_bars[min(self.replay_idx - 1, len(self.bt_bars) - 1)] if self.replay_idx > 0 else 0

        # Trade markers
        for tr in self.replay_trades_shown:
            entry_t = pd.Timestamp(tr["entry_time"])
            exit_t  = pd.Timestamp(tr["exit_time"])
            is_done = tr["exit_bar"] <= cur_bar
            pnl     = tr.get("pnl_pct", 0)

            # Entry marker
            if tr["direction"] == "BUY":
                mk = "^"
                col = GREEN if (is_done and pnl > 0) else (ORANGE if is_done else BLUE)
            else:
                mk = "v"
                col = GREEN if (is_done and pnl > 0) else (RED if is_done else BLUE)

            ax1.scatter(entry_t, tr["entry_price"], marker=mk, c=col, s=65,
                       zorder=5, edgecolors=WHITE, linewidths=0.4)

            # Exit + connecting line (only when trade is complete)
            if is_done:
                ax1.scatter(exit_t, tr["exit_price"], marker="x", c=col, s=28,
                           zorder=4, alpha=0.7)
                lc = GREEN if pnl > 0 else RED
                ax1.plot([entry_t, exit_t], [tr["entry_price"], tr["exit_price"]],
                        color=lc, linewidth=1.0, alpha=0.35)

        # Axes styling
        ax1.set_ylabel("Gold (USD)", color=GREY, fontsize=9)
        ax1.tick_params(colors=GREY, labelsize=8)
        ax1.grid(True, alpha=0.06)
        for s in ['top','right']: ax1.spines[s].set_visible(False)
        for s in ['bottom','left']: ax1.spines[s].set_color(BORDER)

        # Legend
        legend_items = [
            self.Line2D([0],[0], marker='^', color='w', markerfacecolor=GREEN,
                       markersize=8, label='BUY win', linestyle='None'),
            self.Line2D([0],[0], marker='^', color='w', markerfacecolor=ORANGE,
                       markersize=8, label='BUY loss', linestyle='None'),
            self.Line2D([0],[0], marker='v', color='w', markerfacecolor=GREEN,
                       markersize=8, label='SELL win', linestyle='None'),
            self.Line2D([0],[0], marker='v', color='w', markerfacecolor=RED,
                       markersize=8, label='SELL loss', linestyle='None'),
            self.Line2D([0],[0], marker='o', color='w', markerfacecolor=BLUE,
                       markersize=6, label='Open trade', linestyle='None'),
        ]
        ax1.legend(handles=legend_items, loc="upper left", fontsize=7,
                  facecolor=BG2, edgecolor=BORDER, labelcolor=WHITE)

        # ── Equity curve ──────────────────────────────────────────
        ax2.axhline(y=100, color=GREY, linestyle="--", linewidth=0.5, alpha=0.3)

        if self.replay_equity:
            eq_t = [e[0] for e in self.replay_equity]
            eq_v = [e[1] for e in self.replay_equity]
            ax2.plot(eq_t, eq_v, color=GREEN, linewidth=1.2)
            ax2.fill_between(eq_t, 100, eq_v, alpha=0.1,
                            where=[v >= 100 for v in eq_v], color=GREEN)
            ax2.fill_between(eq_t, 100, eq_v, alpha=0.1,
                            where=[v < 100 for v in eq_v], color=RED)

        ax2.set_ylabel("Balance (GBP)", color=GREY, fontsize=9)
        ax2.tick_params(colors=GREY, labelsize=8)
        ax2.grid(True, alpha=0.06)
        for s in ['top','right']: ax2.spines[s].set_visible(False)
        for s in ['bottom','left']: ax2.spines[s].set_color(BORDER)

        ax2.xaxis.set_major_formatter(self.mdates.DateFormatter("%b %d"))

        self.bt_fig.tight_layout()
        self.bt_canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("  Starting AlmostFinishedBot Chart Dashboard...")
    print("  Close the window to exit.")
    # Support command-line arg to open specific tab
    start_tab = None
    if len(sys.argv) > 1 and sys.argv[1] in ("live", "backtest"):
        start_tab = sys.argv[1]
    app = ChartDashboard(start_tab=start_tab)
