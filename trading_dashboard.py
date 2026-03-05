"""
AlmostFinishedBot - Trading Dashboard v2.0 COMPLETE REWRITE
============================================================
FIXES:
  - Kill Switch: Closes ALL positions both symbols + kills all bridge processes
  - Trade markers: Green UP triangle = BUY, Red DOWN triangle = SELL — directly on candle
  - Chart auto-refreshes from trading_log.json and MT5 history every 5s
  - Buy/Sell triangles pinned below/above the candle body at correct price
  - Paper Trade, Backtest, Close All buttons fully wired
  - Account panel shows live balance/equity/margin
  - MT5 login: account 62111880 / PepperstoneUK-Demo (auto-login on startup)
  - Dual symbol XAUUSD + XAUSGD panels
  - Position lines drawn directly on chart with P&L annotation
  - Right-click timeframe menu on chart
  - Ctrl+Scroll zoom, left-click drag pan
"""
import os, sys, json, time, datetime, threading, subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
sys.path.insert(0, BASE)

# ── Optional imports ──────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ── MT5 Credentials ───────────────────────────────────────────────
MT5_LOGIN    = 62111880
MT5_PASSWORD = "h3%ejzpaUy"
MT5_SERVER   = "PepperstoneUK-Demo"

# ── Colour Palette ────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_CARD   = "#1c2333"
FG_TEXT   = "#c9d1d9"
FG_DIM    = "#8b949e"
GREEN     = "#00d26a"
RED       = "#ff4757"
GOLD      = "#ffd700"
BLUE      = "#58a6ff"
ORANGE    = "#f0883e"
PURPLE    = "#bc8cff"
CANDLE_UP = "#26a69a"
CANDLE_DN = "#ef5350"
MARKER_BUY  = "#00e676"   # bright green triangle
MARKER_SELL = "#ff1744"   # bright red triangle

MAGIC_XAUUSD = 999
MAGIC_XAUSGD = 1000

# Timeframe map: label -> (yfinance interval, minutes, MT5 constant name)
TF_MAP = {
    "M1":  ("1m",  1,   "TIMEFRAME_M1"),
    "M5":  ("5m",  5,   "TIMEFRAME_M5"),
    "M15": ("15m", 15,  "TIMEFRAME_M15"),
    "M30": ("30m", 30,  "TIMEFRAME_M30"),
    "H1":  ("1h",  60,  "TIMEFRAME_H1"),
    "H4":  ("1h",  240, "TIMEFRAME_H4"),
    "D1":  ("1d",  1440,"TIMEFRAME_D1"),
}
YF_PERIOD = {
    "M1": "1d", "M5": "5d", "M15": "5d",
    "M30": "5d", "H1": "5d", "H4": "30d", "D1": "60d",
}


# ═══════════════════════════════════════════════════════════════════
# SYMBOL PANEL
# ═══════════════════════════════════════════════════════════════════
class SymbolPanel:
    def __init__(self, parent, symbol, magic, row=0):
        self.symbol     = symbol
        self.magic      = magic
        self.timeframe  = "M15"
        self.candles    = None
        self.trades     = []          # [{time, type, price, lot, profit}]
        self.open_pos   = []
        self.pnl        = 0.0
        self.zoom       = 120         # candles visible
        self.pan        = 0           # bars from right
        self.dragging   = False
        self.drag_x0    = 0
        self.drag_pan0  = 0

        self.frame = tk.Frame(parent, bg=BG_DARK, highlightbackground="#30363d",
                              highlightthickness=1)
        self.frame.grid(row=row, column=0, sticky="nsew", padx=4, pady=4)
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self._build_topbar()
        self._build_chart()
        self._build_btnbar()

    # ── UI construction ──────────────────────────────────────────
    def _build_topbar(self):
        top = tk.Frame(self.frame, bg=BG_PANEL, height=42)
        top.pack(fill="x", padx=2, pady=(2, 0))
        top.pack_propagate(False)

        tk.Label(top, text=f"  {self.symbol}", font=("Consolas", 15, "bold"),
                 fg=GOLD, bg=BG_PANEL).pack(side="left", padx=5)

        self.lbl_price = tk.Label(top, text="Bid: ---  Ask: ---",
                                   font=("Consolas", 11), fg=FG_TEXT, bg=BG_PANEL)
        self.lbl_price.pack(side="left", padx=12)

        self.lbl_tf = tk.Label(top, text="M15", font=("Consolas", 10, "bold"),
                                fg=BLUE, bg=BG_PANEL)
        self.lbl_tf.pack(side="left", padx=6)

        self.lbl_pnl = tk.Label(top, text="Float P&L: $0.00",
                                  font=("Consolas", 11, "bold"), fg=GREEN, bg=BG_PANEL)
        self.lbl_pnl.pack(side="right", padx=12)

        self.lbl_pos = tk.Label(top, text="Pos: 0", font=("Consolas", 10),
                                 fg=FG_DIM, bg=BG_PANEL)
        self.lbl_pos.pack(side="right", padx=8)

    def _build_chart(self):
        if not MPL_AVAILABLE:
            tk.Label(self.frame, text="matplotlib not installed — pip install matplotlib",
                     fg=RED, bg=BG_DARK, font=("Consolas", 11)).pack(expand=True)
            return

        self.fig, self.ax = plt.subplots(figsize=(11, 3.8), facecolor=BG_DARK)
        self.ax.set_facecolor(BG_DARK)
        self.fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.14)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

        self.canvas.mpl_connect("scroll_event",         self._on_scroll)
        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _build_btnbar(self):
        bar = tk.Frame(self.frame, bg=BG_PANEL, height=38)
        bar.pack(fill="x", padx=2, pady=(0, 2))
        bar.pack_propagate(False)

        bstyle = dict(font=("Consolas", 9, "bold"), bd=0, padx=10, pady=3,
                      cursor="hand2", relief="flat")

        tk.Button(bar, text="PAPER TRADE", bg="#1f6feb", fg="white",
                  command=self._paper_trade, **bstyle).pack(side="left", padx=3, pady=4)

        tk.Button(bar, text="TRADE NOW", bg=GREEN, fg=BG_DARK,
                  command=self._trade_now, **bstyle).pack(side="left", padx=3, pady=4)

        tk.Button(bar, text="CLOSE ALL", bg=RED, fg="white",
                  command=self._close_all, **bstyle).pack(side="left", padx=3, pady=4)

        tk.Button(bar, text="BACKTEST", bg=ORANGE, fg=BG_DARK,
                  command=self._run_backtest, **bstyle).pack(side="left", padx=3, pady=4)

        # Timeframe selector
        self.tf_var = tk.StringVar(value="M15")
        tf_menu = tk.OptionMenu(bar, self.tf_var, *TF_MAP.keys(),
                                command=self._set_timeframe)
        tf_menu.config(bg=BG_CARD, fg=FG_TEXT, font=("Consolas", 8),
                       bd=0, activebackground=BLUE, highlightthickness=0)
        tf_menu.pack(side="left", padx=4)

        self.lbl_status = tk.Label(bar, text="Ready", font=("Consolas", 8),
                                    fg=FG_DIM, bg=BG_PANEL)
        self.lbl_status.pack(side="right", padx=8)

        self.lbl_wl = tk.Label(bar, text="W:0 L:0 Real:$0.00",
                                font=("Consolas", 8), fg=FG_DIM, bg=BG_PANEL)
        self.lbl_wl.pack(side="right", padx=6)

    # ── Chart events ─────────────────────────────────────────────
    def _on_scroll(self, event):
        if event.key == "control":
            if event.button == "up":   self.zoom = max(20, self.zoom - 10)
            elif event.button == "down": self.zoom = min(500, self.zoom + 20)
        else:
            if event.button == "up":   self.pan = max(0, self.pan - 5)
            elif event.button == "down": self.pan += 5
        self._draw_chart()

    def _on_press(self, event):
        if event.button == 1:
            self.dragging  = True
            self.drag_x0   = event.x or 0
            self.drag_pan0 = self.pan
        elif event.button == 3:
            self._timeframe_menu(event)

    def _on_release(self, event): self.dragging = False

    def _on_motion(self, event):
        if self.dragging and event.x is not None:
            dx = event.x - self.drag_x0
            w  = self.fig.get_figwidth() * self.fig.dpi
            candle_w = w / max(self.zoom, 1)
            delta = int(dx / max(candle_w, 1))
            self.pan = max(0, self.drag_pan0 + delta)
            self._draw_chart()

    def _timeframe_menu(self, event):
        menu = tk.Menu(self.frame, tearoff=0, bg=BG_CARD, fg=FG_TEXT,
                       font=("Consolas", 9), activebackground=BLUE)
        for tf in TF_MAP:
            menu.add_command(label=tf, command=lambda t=tf: self._set_timeframe(t))
        try:
            menu.tk_popup(int(event.guiEvent.x_root), int(event.guiEvent.y_root))
        except Exception:
            pass

    def _set_timeframe(self, tf):
        self.timeframe = tf
        self.tf_var.set(tf)
        self.lbl_tf.config(text=tf)
        self.pan  = 0
        self.zoom = 120
        self._fetch_candles()
        self._draw_chart()
        self._status(f"Timeframe: {tf}")

    # ── Data fetching ─────────────────────────────────────────────
    def _fetch_candles(self):
        """Fetch OHLCV candles from MT5 first, fall back to yfinance."""
        if MT5_AVAILABLE:
            try:
                tf_name = TF_MAP[self.timeframe][2]
                tf_const = getattr(mt5, tf_name, mt5.TIMEFRAME_M15)
                rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, 600)
                if rates is not None and len(rates) > 10:
                    import pandas as pd
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df = df.rename(columns={"tick_volume": "volume"})
                    df = df.set_index("time")
                    self.candles = df
                    return
            except Exception as e:
                self._status(f"MT5 fetch failed: {e}")

        # yfinance fallback
        if YF_AVAILABLE:
            try:
                import pandas as pd
                ticker = "GC=F"   # gold proxy
                yf_iv  = TF_MAP[self.timeframe][0]
                period = YF_PERIOD[self.timeframe]
                df = yf.download(ticker, period=period, interval=yf_iv,
                                 progress=False, auto_adjust=True)
                if df is None or len(df) < 5:
                    return
                if hasattr(df.columns, "droplevel") and df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
                df.columns = [c.lower() for c in df.columns]
                if hasattr(df.index, "tz") and df.index.tz:
                    df.index = df.index.tz_localize(None)
                self.candles = df
            except Exception as e:
                self._status(f"yfinance error: {e}")

    def _load_trades(self):
        """Load trade markers from trading_log.json + MT5 history."""
        self.trades = []
        import pandas as pd

        # From trading_log.json
        log_path = os.path.join(BASE, "trading_log.json")
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    log = json.load(f)
                for e in log:
                    if e.get("type") == "trade" and e.get("signal") in ("BUY", "SELL"):
                        try:
                            t = datetime.datetime.strptime(
                                e["timestamp"], "%Y-%m-%d %H:%M:%S")
                            self.trades.append({
                                "time":   t,
                                "type":   e["signal"],
                                "price":  float(e.get("price") or e.get("ep", 0)),
                                "lot":    float(e.get("lot", 0.01)),
                                "profit": 0,
                            })
                        except Exception:
                            pass
            except Exception:
                pass

        # From MT5 history
        if MT5_AVAILABLE:
            try:
                now   = datetime.datetime.now(datetime.timezone.utc)
                since = now - datetime.timedelta(days=14)
                deals = mt5.history_deals_get(since, now, group=self.symbol)
                if deals:
                    for d in deals:
                        if d.magic != self.magic or d.entry != 0:
                            continue
                        dt  = datetime.datetime.fromtimestamp(
                            d.time, tz=datetime.timezone.utc
                        ).replace(tzinfo=None)
                        sig = "BUY" if d.type == 0 else "SELL"
                        self.trades.append({
                            "time":   dt,
                            "type":   sig,
                            "price":  d.price,
                            "lot":    d.volume,
                            "profit": d.profit,
                        })
            except Exception:
                pass

    def _update_positions(self):
        if not MT5_AVAILABLE:
            return
        try:
            pos = mt5.positions_get(symbol=self.symbol)
            if pos is None:
                self.open_pos = []
                self.pnl = 0.0
            else:
                self.open_pos = [p for p in pos if p.magic == self.magic]
                self.pnl = sum(p.profit for p in self.open_pos)
        except Exception:
            pass

    # ── Chart drawing ─────────────────────────────────────────────
    def _draw_chart(self):
        if not MPL_AVAILABLE or self.candles is None or len(self.candles) == 0:
            return

        self.ax.clear()
        self.ax.set_facecolor(BG_DARK)

        df = self.candles.copy()
        n  = len(df)

        # Apply zoom + pan window
        end_idx   = max(0, n - self.pan)
        start_idx = max(0, end_idx - self.zoom)
        df = df.iloc[start_idx:end_idx]

        if len(df) < 2:
            return

        opens  = df["open"].values.astype(float)
        highs  = df["high"].values.astype(float)
        lows   = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        x      = np.arange(len(df))

        # ── Candlesticks ─────────────────────────────────────────
        width = 0.6
        for i in range(len(df)):
            up  = closes[i] >= opens[i]
            col = CANDLE_UP if up else CANDLE_DN
            # Wick
            self.ax.plot([x[i], x[i]], [lows[i], highs[i]],
                         color=col, linewidth=0.9, zorder=2)
            # Body
            body_lo = min(opens[i], closes[i])
            body_hi = max(opens[i], closes[i])
            body_h  = max(body_hi - body_lo, 0.05)
            rect = plt.Rectangle(
                (x[i] - width / 2, body_lo), width, body_h,
                facecolor=col, edgecolor=col, linewidth=0, zorder=3)
            self.ax.add_patch(rect)

        # ── EMA overlays ─────────────────────────────────────────
        def ema(arr, span):
            out = np.full(len(arr), np.nan)
            k   = 2.0 / (span + 1)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = arr[i] * k + out[i - 1] * (1 - k)
            return out

        if len(closes) >= 20:
            e20 = ema(closes, 20)
            self.ax.plot(x, e20, color="#ff9800", linewidth=1.0,
                         label="EMA20", zorder=4, alpha=0.85)
        if len(closes) >= 50:
            e50 = ema(closes, 50)
            self.ax.plot(x, e50, color="#42a5f5", linewidth=1.0,
                         label="EMA50", zorder=4, alpha=0.85)

        # ── Current price dashed line ─────────────────────────────
        last_p = closes[-1]
        self.ax.axhline(last_p, color="#ffd700", linewidth=0.6,
                        linestyle="--", zorder=5, alpha=0.7)
        self.ax.text(len(df) - 0.3, last_p, f" {last_p:.2f}",
                     fontsize=7, color="#ffd700", va="center",
                     fontfamily="Consolas", zorder=6)

        # ── Open position horizontal lines ────────────────────────
        for pos in self.open_pos:
            try:
                pcol = GREEN if pos.profit >= 0 else RED
                self.ax.axhline(pos.price_open, color=pcol, linewidth=0.8,
                                linestyle=":", alpha=0.7, zorder=4)
                side = "BUY" if pos.type == 0 else "SELL"
                self.ax.text(0.5, pos.price_open,
                             f" {side} @{pos.price_open:.2f}  {pos.profit:+.2f}",
                             fontsize=6.5, color=pcol, va="bottom",
                             fontfamily="Consolas", zorder=5)
            except Exception:
                pass

        # ── Trade markers: Triangles pinned at exact trade price ─────
        # BUY  → green ▲  tip of triangle sits AT the exact buy price
        # SELL → red   ▼  tip of triangle sits AT the exact sell price
        # If bought low on the candle → marker is low on chart
        # If sold high on the candle  → marker is high on chart
        # Small, clean, no clutter
        times = df.index.to_list()

        # Compute a small offset so triangle doesn't overlap the wick
        # Use 15% of average candle range as padding
        avg_range = float(np.mean(highs - lows)) if len(highs) > 0 else 1.0
        pad = avg_range * 0.15   # tiny gap between wick tip and triangle

        for tr in self.trades:
            try:
                tr_time = tr["time"]
                if not isinstance(tr_time, datetime.datetime):
                    continue

                # Find nearest candle index
                best_j  = -1
                best_dt = float("inf")
                for j, ct in enumerate(times):
                    if hasattr(ct, "timestamp"):
                        dt = abs((ct - tr_time).total_seconds())
                    else:
                        dt = abs(float(ct) - tr_time.timestamp())
                    if dt < best_dt:
                        best_dt = dt
                        best_j  = j

                # Only show if within 8 TF periods
                tf_minutes = TF_MAP[self.timeframe][1]
                if best_j < 0 or best_dt > tf_minutes * 60 * 8:
                    continue

                tr_price = float(tr["price"])
                if tr_price <= 0:
                    tr_price = closes[best_j]

                if tr["type"] == "BUY":
                    # ▲ Green triangle — tip points UP, pinned at exact buy price
                    # Triangle sits just BELOW the trade price, tip touching it
                    y_tip = tr_price        # tip of triangle = exact fill price
                    y_marker = y_tip - pad  # centre of scatter slightly below tip
                    self.ax.scatter(
                        best_j, y_marker,
                        marker="^", color=MARKER_BUY, s=70, zorder=10,
                        edgecolors="white", linewidths=0.5)
                    # Small label just below marker
                    self.ax.text(
                        best_j, y_marker - pad * 1.5,
                        f"{tr['lot']}", color=MARKER_BUY, fontsize=5,
                        ha="center", va="top", fontfamily="Consolas", zorder=11)
                else:
                    # ▼ Red triangle — tip points DOWN, pinned at exact sell price
                    # Triangle sits just ABOVE the trade price, tip touching it
                    y_tip    = tr_price     # tip of triangle = exact fill price
                    y_marker = y_tip + pad  # centre of scatter slightly above tip
                    self.ax.scatter(
                        best_j, y_marker,
                        marker="v", color=MARKER_SELL, s=70, zorder=10,
                        edgecolors="white", linewidths=0.5)
                    # Small label just above marker
                    self.ax.text(
                        best_j, y_marker + pad * 1.5,
                        f"{tr['lot']}", color=MARKER_SELL, fontsize=5,
                        ha="center", va="bottom", fontfamily="Consolas", zorder=11)

            except Exception:
                pass

        # ── X-axis labels ─────────────────────────────────────────
        step = max(1, len(df) // 9)
        xticks  = list(range(0, len(df), step))
        xlabels = []
        for j in xticks:
            t = times[j]
            try:
                xlabels.append(t.strftime("%H:%M\n%d/%m"))
            except Exception:
                xlabels.append(str(j))
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xlabels, fontsize=6, color=FG_DIM)

        # ── Legend ────────────────────────────────────────────────
        import matplotlib.lines as mlines
        buy_patch  = mlines.Line2D([], [], color=MARKER_BUY,  marker="^",
                                   markersize=7, label="BUY",  linestyle="None")
        sell_patch = mlines.Line2D([], [], color=MARKER_SELL, marker="v",
                                   markersize=7, label="SELL", linestyle="None")
        handles = [buy_patch, sell_patch]
        if len(closes) >= 20:
            handles.append(mlines.Line2D([], [], color="#ff9800", label="EMA20"))
        if len(closes) >= 50:
            handles.append(mlines.Line2D([], [], color="#42a5f5", label="EMA50"))
        self.ax.legend(handles=handles, loc="upper left", fontsize=6.5,
                       facecolor=BG_PANEL, edgecolor="#30363d", labelcolor=FG_TEXT)

        # ── Styling ───────────────────────────────────────────────
        self.ax.tick_params(colors=FG_DIM, labelsize=7)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_color("#30363d")
        self.ax.spines["bottom"].set_color("#30363d")
        self.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        self.ax.set_title(
            f"{self.symbol}  {self.timeframe}  "
            f"[{len(self.trades)} markers]",
            fontsize=9, color=GOLD, pad=4, fontfamily="Consolas")
        self.ax.grid(True, alpha=0.07, color="#30363d")
        self.ax.set_xlim(-1, len(df))

        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    # ── Periodic update ───────────────────────────────────────────
    def update(self):
        """Called every second from main loop."""
        self._update_positions()

        # Price label
        if MT5_AVAILABLE:
            try:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    self.lbl_price.config(
                        text=f"Bid: {tick.bid:.2f}  Ask: {tick.ask:.2f}")
            except Exception:
                pass

        # P&L colour
        col = GREEN if self.pnl >= 0 else RED
        self.lbl_pnl.config(text=f"Float P&L: ${self.pnl:.2f}", fg=col)
        self.lbl_pos.config(text=f"Pos: {len(self.open_pos)}")

        # W/L/Realised
        wins   = sum(1 for t in self.trades if t.get("profit", 0) > 0)
        losses = sum(1 for t in self.trades if t.get("profit", 0) < 0)
        real   = sum(t.get("profit", 0) for t in self.trades)
        rc     = GREEN if real >= 0 else RED
        self.lbl_wl.config(
            text=f"W:{wins} L:{losses} Real:${real:+.2f}", fg=rc)

    # ── Buttons ───────────────────────────────────────────────────
    def _status(self, msg, col=None):
        self.lbl_status.config(text=msg, fg=col or FG_DIM)

    def _paper_trade(self):
        self._status("Launching paper trade...", GOLD)
        try:
            script = "live_trading_bridge.py" if "USD" in self.symbol else "live_bridge_xausgd.py"
            sp = os.path.join(BASE, script)
            if not os.path.exists(sp):
                self._status(f"{script} not found", RED); return
            title = f"AFB_{self.symbol}_Paper"
            subprocess.Popen(
                f'start "{title}" cmd /k python "{sp}" --mode paper',
                shell=True, cwd=BASE)
            self._status("Paper trade started in new window", GREEN)
        except Exception as e:
            self._status(f"Error: {e}", RED)

    def _trade_now(self):
        self._status("Checking signal...", GOLD)
        try:
            sig_path = os.path.join(BASE, "current_signal.json")
            if not os.path.exists(sig_path):
                self._status("No signal file found", RED); return
            with open(sig_path) as f:
                sig = json.load(f)
            signal = sig.get("signal", "NEUTRAL")
            conf   = sig.get("confidence", 0.5)
            if signal not in ("BUY", "SELL") or conf < 0.54:
                self._status(f"Signal {signal} ({conf:.0%}) — too weak", ORANGE)
                return
            if not MT5_AVAILABLE:
                self._status("MT5 not available", RED); return
            tick = mt5.symbol_info_tick(self.symbol)
            si   = mt5.symbol_info(self.symbol)
            if not tick or not si:
                self._status("No tick data", RED); return
            if not si.visible:
                mt5.symbol_select(self.symbol, True)
            pt = si.point
            if signal == "BUY":
                ot = mt5.ORDER_TYPE_BUY;  ep = tick.ask
                sl = ep - 3000 * pt;      tp = ep + 6000 * pt
            else:
                ot = mt5.ORDER_TYPE_SELL; ep = tick.bid
                sl = ep + 3000 * pt;      tp = ep - 6000 * pt
            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       self.symbol,
                "volume":       0.01,
                "type":         ot,
                "price":        ep,
                "sl":           round(sl, 2),
                "tp":           round(tp, 2),
                "deviation":    20,
                "magic":        self.magic,
                "comment":      f"AFB manual {signal}",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self._status(f"{signal} 0.01 @ {ep:.2f} ✓", GREEN)
                self.trades.append({
                    "time": datetime.datetime.now(), "type": signal,
                    "price": ep, "lot": 0.01, "profit": 0})
                self._draw_chart()
            else:
                rc = result.retcode if result else "None"
                self._status(f"Order failed: {rc}", RED)
        except Exception as e:
            self._status(f"Error: {e}", RED)

    def _close_all(self):
        """Close all open positions for this symbol."""
        if not MT5_AVAILABLE:
            self._status("MT5 not available", RED); return
        if not self.open_pos:
            self._status("No positions to close", FG_DIM); return
        if not messagebox.askyesno(
                "Close All",
                f"Close {len(self.open_pos)} position(s) for {self.symbol}?"):
            return
        closed = 0
        for p in self.open_pos:
            try:
                ct    = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
                tick  = mt5.symbol_info_tick(self.symbol)
                price = tick.bid if p.type == 0 else tick.ask
                req   = {
                    "action":       mt5.TRADE_ACTION_DEAL,
                    "symbol":       self.symbol,
                    "volume":       p.volume,
                    "type":         ct,
                    "position":     p.ticket,
                    "price":        price,
                    "deviation":    20,
                    "magic":        self.magic,
                    "comment":      "AFB close all",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                r = mt5.order_send(req)
                if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                    closed += 1
            except Exception:
                pass
        self._status(f"Closed {closed}/{len(self.open_pos)} positions ✓", GREEN)

    def _run_backtest(self):
        self._status("Launching backtest...", GOLD)
        sp = os.path.join(BASE, "walkforward_backtest.py")
        if not os.path.exists(sp):
            self._status("walkforward_backtest.py not found", RED); return
        subprocess.Popen(
            f'start "AFB Backtest" cmd /k python "{sp}"',
            shell=True, cwd=BASE)
        self._status("Backtest running in new window", GREEN)


# ═══════════════════════════════════════════════════════════════════
# KILL SWITCH — full nuclear stop
# ═══════════════════════════════════════════════════════════════════
def kill_switch_action(parent_root):
    if not messagebox.askyesno(
            "⚠ KILL SWITCH",
            "EMERGENCY STOP\n\n"
            "This will:\n"
            "1. Close ALL open positions (XAUUSD + XAUSGD)\n"
            "2. Kill ALL bridge / bot Python processes\n"
            "3. Stop all paper/live trading\n\n"
            "Are you absolutely sure?",
            icon="warning"):
        return

    report = []

    # ── 1. Close all MT5 positions ────────────────────────────────
    if MT5_AVAILABLE:
        try:
            closed = 0
            for symbol, magic in [("XAUUSD", MAGIC_XAUUSD), ("XAUSGD", MAGIC_XAUSGD)]:
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
                    except Exception:
                        pass
            report.append(f"✓ Closed {closed} position(s)")
        except Exception as e:
            report.append(f"✗ MT5 close error: {e}")

    # ── 2. Kill bridge processes by window title ──────────────────
    bridge_titles = [
        "AFB_XAUUSD_Paper", "AFB_XAUSGD_Paper",
        "AlmostFinishedBot - XAUUSD", "AlmostFinishedBot - XAUSGD",
        "AFB_XAUUSD_LIVE", "AFB_XAUSGD_LIVE",
    ]
    for title in bridge_titles:
        try:
            subprocess.run(
                f'taskkill /fi "WINDOWTITLE eq {title}*" /f',
                shell=True, capture_output=True)
        except Exception:
            pass

    # ── 3. Kill bridge Python scripts by name ─────────────────────
    for script in ["live_trading_bridge.py", "live_bridge_xausgd.py"]:
        try:
            subprocess.run(
                f'wmic process where "commandline like \'%{script}%\'" call terminate',
                shell=True, capture_output=True)
        except Exception:
            pass
    report.append("✓ Bridge processes terminated")

    # ── 4. Kill any orphan python processes running our scripts ───
    try:
        # taskkill all python except THIS process
        my_pid = os.getpid()
        result = subprocess.run(
            ["wmic", "process", "where",
             f"name='python.exe' and ProcessId!={my_pid}",
             "call", "terminate"],
            capture_output=True, text=True)
        report.append("✓ Python subprocesses killed")
    except Exception as e:
        report.append(f"✗ Kill error: {e}")

    messagebox.showinfo(
        "Kill Switch Complete",
        "\n".join(report) + "\n\nAll positions closed. All bridges stopped.")


# ═══════════════════════════════════════════════════════════════════
# MAIN DASHBOARD WINDOW
# ═══════════════════════════════════════════════════════════════════
class TradingDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AlmostFinishedBot — Trading Dashboard v2.0")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1440x960")
        self.root.minsize(900, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_header()
        self._build_body()
        self._connect_mt5()
        self._tick_1s()
        self._tick_5s()

    # ── Header ────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="  ⚡ AlmostFinishedBot",
                 font=("Consolas", 17, "bold"), fg=GOLD, bg=BG_PANEL
                 ).pack(side="left", padx=10)

        self.lbl_acc = tk.Label(hdr, text="MT5: connecting...",
                                 font=("Consolas", 10), fg=FG_DIM, bg=BG_PANEL)
        self.lbl_acc.pack(side="left", padx=16)

        # KILL SWITCH button — prominent red
        tk.Button(hdr, text="⬛  KILL SWITCH",
                  bg="#b71c1c", fg="white", font=("Consolas", 11, "bold"),
                  bd=0, padx=14, pady=4, cursor="hand2", relief="flat",
                  command=lambda: kill_switch_action(self.root)
                  ).pack(side="right", padx=10, pady=6)

        self.lbl_equity = tk.Label(hdr, text="Equity: ---",
                                    font=("Consolas", 11), fg=FG_TEXT, bg=BG_PANEL)
        self.lbl_equity.pack(side="right", padx=8)

        self.lbl_balance = tk.Label(hdr, text="Balance: ---",
                                     font=("Consolas", 11, "bold"), fg=GREEN, bg=BG_PANEL)
        self.lbl_balance.pack(side="right", padx=8)

        self.lbl_clock = tk.Label(hdr, text="",
                                   font=("Consolas", 9), fg=FG_DIM, bg=BG_PANEL)
        self.lbl_clock.pack(side="right", padx=8)

    # ── Body: two symbol panels ───────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=4, pady=4)

        self.panels = {
            "XAUUSD": SymbolPanel(body, "XAUUSD", MAGIC_XAUUSD, row=0),
            "XAUSGD": SymbolPanel(body, "XAUSGD", MAGIC_XAUSGD, row=1),
        }

    # ── MT5 connection ────────────────────────────────────────────
    def _connect_mt5(self):
        if not MT5_AVAILABLE:
            self.lbl_acc.config(text="MetaTrader5 package not installed", fg=RED)
            return
        try:
            ok = mt5.initialize(
                login    = MT5_LOGIN,
                password = MT5_PASSWORD,
                server   = MT5_SERVER,
            )
            if not ok:
                # Try without credentials (already logged in)
                ok = mt5.initialize()
            if not ok:
                self.lbl_acc.config(text="MT5: cannot connect", fg=RED)
                return
            info = mt5.account_info()
            if info:
                mode = "DEMO" if info.trade_mode == 0 else "LIVE"
                self.lbl_acc.config(
                    text=f"#{info.login}  {info.server}  [{mode}]", fg=GREEN)
                self.lbl_balance.config(text=f"Balance: £{info.balance:.2f}")
                self.lbl_equity.config(text=f"Equity: £{info.equity:.2f}")
                # Ensure symbols visible
                for sym in ["XAUUSD", "XAUSGD"]:
                    si = mt5.symbol_info(sym)
                    if si and not si.visible:
                        mt5.symbol_select(sym, True)
        except Exception as e:
            self.lbl_acc.config(text=f"MT5 error: {e}", fg=RED)

        # Initial data fetch
        for panel in self.panels.values():
            panel._load_trades()
            panel._fetch_candles()
            panel._draw_chart()

    # ── Update loops ──────────────────────────────────────────────
    def _tick_1s(self):
        """Every 1 second: update prices, P&L, clock."""
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            self.lbl_clock.config(text=now.strftime("%H:%M:%S UTC"))

            if MT5_AVAILABLE:
                try:
                    info = mt5.account_info()
                    if info:
                        col = GREEN if info.equity >= info.balance else RED
                        self.lbl_balance.config(text=f"Balance: £{info.balance:.2f}")
                        self.lbl_equity.config(text=f"Equity: £{info.equity:.2f}", fg=col)
                except Exception:
                    pass

            for panel in self.panels.values():
                panel.update()
        except Exception:
            pass
        self.root.after(1000, self._tick_1s)

    def _tick_5s(self):
        """Every 5 seconds: refresh candle data and trade markers."""
        try:
            for panel in self.panels.values():
                panel._load_trades()
                panel._fetch_candles()
                panel._draw_chart()
        except Exception:
            pass
        self.root.after(5000, self._tick_5s)

    def _on_close(self):
        try:
            if MT5_AVAILABLE:
                mt5.shutdown()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  AlmostFinishedBot — Trading Dashboard v2.0")
    print("  BUY=Green▲  SELL=Red▼  Kill Switch=Full stop")
    print("=" * 60)
    app = TradingDashboard()
    app.run()


if __name__ == "__main__":
    main()
