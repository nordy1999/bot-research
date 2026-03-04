"""
AlmostFinishedBot - Trading Dashboard v1
Full GUI with dual-symbol support (XAUUSD + XAUSGD)

Features per symbol:
  - Live candlestick chart (auto-updating every second)
  - Trade markers (green triangle BUY, red triangle SELL)
  - Zoom (Ctrl+Scroll), Pan (click+drag), right-click timeframe menu
  - P&L display, open positions, account info
  - Buttons: Trade Now, Paper Trade, Backtest
  - Sell open positions button
  - Multiple timeframes: M1, M2, M3, M5, M15, M30, H1, H4, D1
"""
import os, sys, json, time, datetime, threading, queue
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
sys.path.insert(0, BASE)

# ── Try importing optional deps ──────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ── Colors & Styling ─────────────────────────────────────────────
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
CANDLE_UP = "#00d26a"
CANDLE_DN = "#ff4757"

MAGIC_NUMBER = 999

# Timeframe mappings
TF_MAP = {
    "M1":  ("1m",  1),
    "M2":  ("2m",  2),
    "M3":  ("5m",  3),   # yfinance doesn't have 3m, use 5m
    "M5":  ("5m",  5),
    "M15": ("15m", 15),
    "M30": ("30m", 30),
    "H1":  ("1h",  60),
    "H4":  ("1h",  240),  # aggregate from 1h
    "D1":  ("1d",  1440),
}

# MT5 timeframe constants
MT5_TF = {}
if MT5_AVAILABLE:
    MT5_TF = {
        "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3,
        "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
    }


class SymbolPanel:
    """One panel per symbol with chart, controls, and P&L."""

    def __init__(self, parent, symbol, row=0):
        self.symbol = symbol
        self.timeframe = "M5"
        self.candles = None
        self.trades = []  # list of {"time": dt, "type": "BUY"/"SELL", "price": float, "lot": float}
        self.open_positions = []
        self.pnl = 0.0
        self.updating = True
        self.zoom_level = 100  # number of candles to show
        self.pan_offset = 0
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_offset = 0

        # Main frame for this symbol
        self.frame = tk.Frame(parent, bg=BG_DARK, bd=1, relief="solid", highlightbackground="#30363d", highlightthickness=1)
        self.frame.grid(row=row, column=0, sticky="nsew", padx=5, pady=5)
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # ── Top bar: symbol name + info ──
        top = tk.Frame(self.frame, bg=BG_PANEL, height=45)
        top.pack(fill="x", padx=2, pady=(2,0))
        top.pack_propagate(False)

        self.lbl_symbol = tk.Label(top, text=f"  {symbol}", font=("Consolas", 16, "bold"),
                                    fg=GOLD, bg=BG_PANEL, anchor="w")
        self.lbl_symbol.pack(side="left", padx=5)

        self.lbl_price = tk.Label(top, text="Price: ---", font=("Consolas", 12),
                                   fg=FG_TEXT, bg=BG_PANEL)
        self.lbl_price.pack(side="left", padx=15)

        self.lbl_pnl = tk.Label(top, text="P&L: $0.00", font=("Consolas", 12, "bold"),
                                 fg=GREEN, bg=BG_PANEL)
        self.lbl_pnl.pack(side="right", padx=15)

        self.lbl_positions = tk.Label(top, text="Positions: 0", font=("Consolas", 10),
                                       fg=FG_DIM, bg=BG_PANEL)
        self.lbl_positions.pack(side="right", padx=10)

        self.lbl_tf = tk.Label(top, text=f"TF: {self.timeframe}", font=("Consolas", 10),
                                fg=BLUE, bg=BG_PANEL)
        self.lbl_tf.pack(side="right", padx=10)

        # ── Chart area ──
        if MPL_AVAILABLE:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 4), facecolor=BG_DARK)
            self.ax.set_facecolor(BG_DARK)
            self.fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.12)

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

            # Bind events
            self.canvas.mpl_connect("scroll_event", self._on_scroll)
            self.canvas.mpl_connect("button_press_event", self._on_press)
            self.canvas.mpl_connect("button_release_event", self._on_release)
            self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        else:
            tk.Label(self.frame, text="matplotlib not available", fg=RED, bg=BG_DARK,
                     font=("Consolas", 14)).pack(expand=True)

        # ── Button bar ──
        btn_bar = tk.Frame(self.frame, bg=BG_PANEL, height=40)
        btn_bar.pack(fill="x", padx=2, pady=(0,2))

        btn_style = {"font": ("Consolas", 9, "bold"), "bd": 0, "padx": 10, "pady": 4, "cursor": "hand2"}

        self.btn_paper = tk.Button(btn_bar, text="PAPER TRADE", bg="#1f6feb", fg="white",
                                    command=self._start_paper, **btn_style)
        self.btn_paper.pack(side="left", padx=3, pady=4)

        self.btn_trade = tk.Button(btn_bar, text="TRADE NOW", bg=GREEN, fg=BG_DARK,
                                    command=self._trade_now, **btn_style)
        self.btn_trade.pack(side="left", padx=3, pady=4)

        self.btn_sell = tk.Button(btn_bar, text="CLOSE ALL", bg=RED, fg="white",
                                   command=self._close_all, **btn_style)
        self.btn_sell.pack(side="left", padx=3, pady=4)

        self.btn_backtest = tk.Button(btn_bar, text="BACKTEST", bg=ORANGE, fg=BG_DARK,
                                       command=self._backtest, **btn_style)
        self.btn_backtest.pack(side="left", padx=3, pady=4)

        self.lbl_realised = tk.Label(btn_bar, text="Realised: $0.00", font=("Consolas", 9, "bold"),
                                      fg=FG_DIM, bg=BG_PANEL)
        self.lbl_realised.pack(side="right", padx=5)
        self.lbl_wl = tk.Label(btn_bar, text="W:0 L:0", font=("Consolas", 9), fg=FG_DIM, bg=BG_PANEL)
        self.lbl_wl.pack(side="right", padx=5)
        # Status label
        self.lbl_status = tk.Label(btn_bar, text="Ready", font=("Consolas", 9),
                                    fg=FG_DIM, bg=BG_PANEL)
        self.lbl_status.pack(side="right", padx=10)

    def _on_scroll(self, event):
        """Ctrl+Scroll to zoom in/out."""
        if event.key == "control":
            if event.button == "up":
                self.zoom_level = max(20, self.zoom_level - 10)
            elif event.button == "down":
                self.zoom_level = min(500, self.zoom_level + 10)
        else:
            # Regular scroll = pan
            if event.button == "up":
                self.pan_offset = max(0, self.pan_offset - 5)
            elif event.button == "down":
                self.pan_offset += 5
        self._draw_chart()

    def _on_press(self, event):
        if event.button == 1:  # left click
            self.dragging = True
            self.drag_start_x = event.x
            self.drag_start_offset = self.pan_offset
        elif event.button == 3:  # right click - timeframe menu
            self._show_tf_menu(event)

    def _on_release(self, event):
        self.dragging = False

    def _on_motion(self, event):
        if self.dragging and event.x is not None:
            dx = event.x - self.drag_start_x
            candle_width_px = self.fig.get_figwidth() * self.fig.dpi / max(self.zoom_level, 1)
            pan_candles = int(dx / max(candle_width_px, 1))
            self.pan_offset = max(0, self.drag_start_offset + pan_candles)
            self._draw_chart()

    def _show_tf_menu(self, event):
        menu = tk.Menu(self.frame, tearoff=0, bg=BG_CARD, fg=FG_TEXT,
                       font=("Consolas", 10), activebackground=BLUE)
        for tf in ["M1", "M2", "M3", "M5", "M15", "M30", "H1", "H4", "D1"]:
            menu.add_command(label=tf, command=lambda t=tf: self._set_timeframe(t))
        try:
            menu.tk_popup(int(event.guiEvent.x_root), int(event.guiEvent.y_root))
        except Exception:
            pass

    def _set_timeframe(self, tf):
        self.timeframe = tf
        self.lbl_tf.config(text=f"TF: {tf}")
        self.pan_offset = 0
        self.zoom_level = 100
        self._fetch_data()
        self._draw_chart()
        self.lbl_status.config(text=f"Timeframe: {tf}")

    def _fetch_data(self):
        """Fetch candle data from MT5 or yfinance."""
        try:
            if MT5_AVAILABLE and mt5.terminal_info() is not None:
                tf_const = MT5_TF.get(self.timeframe, mt5.TIMEFRAME_M5)
                rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, 500)
                if rates is not None and len(rates) > 0:
                    import pandas as pd
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    self.candles = df
                    return
        except Exception:
            pass

        # Fallback to yfinance for XAUUSD
        if YF_AVAILABLE and "USD" in self.symbol:
            try:
                yf_tf = TF_MAP.get(self.timeframe, ("5m", 5))[0]
                period = "5d" if self.timeframe in ("M1","M2","M3","M5","M15","M30") else "3mo"
                df = yf.download("GC=F", period=period, interval=yf_tf, progress=False)
                if df is not None and len(df) > 0:
                    if hasattr(df.columns, "droplevel") and df.columns.nlevels > 1:
                        df.columns = df.columns.droplevel(1)
                    df.columns = [c.lower() for c in df.columns]
                    df = df.reset_index()
                    if "Datetime" in df.columns:
                        df = df.rename(columns={"Datetime": "time"})
                    elif "Date" in df.columns:
                        df = df.rename(columns={"Date": "time"})
                    elif "datetime" in df.columns:
                        df = df.rename(columns={"datetime": "time"})
                    elif "date" in df.columns:
                        df = df.rename(columns={"date": "time"})
                    self.candles = df
            except Exception:
                pass

    def _load_historical_trades(self):
        self.trades = []
        try:
            outcome_path = os.path.join(BASE, "trade_outcomes.json")
            if os.path.exists(outcome_path):
                with open(outcome_path) as f:
                    outcomes = json.load(f)
                for o in outcomes:
                    if o.get("signal") in ("BUY", "SELL"):
                        try:
                            t = datetime.datetime.strptime(o["entry_time"], "%Y-%m-%d %H:%M:%S")
                            self.trades.append({"time": t, "type": o["signal"],
                                "price": o.get("entry_price", 0), "lot": o.get("lot", 0.01),
                                "profit": o.get("profit_usd", 0), "result": o.get("result", "?")})
                        except Exception: pass
        except Exception: pass
        if MT5_AVAILABLE:
            try:
                now = datetime.datetime.now(datetime.timezone.utc)
                from_time = now - datetime.timedelta(days=7)
                deals = mt5.history_deals_get(from_time, now, group=self.symbol)
                if deals:
                    for d in deals:
                        if d.magic == MAGIC_NUMBER and d.entry == 0:
                            dt = datetime.datetime.fromtimestamp(d.time, tz=datetime.timezone.utc)
                            sig = "BUY" if d.type == 0 else "SELL"
                            self.trades.append({"time": dt.replace(tzinfo=None), "type": sig,
                                "price": d.price, "lot": d.volume, "profit": 0, "result": "OPEN"})
            except Exception: pass

    def _draw_chart(self):
        """Draw candlestick chart with trade markers."""
        if not MPL_AVAILABLE or self.candles is None or len(self.candles) == 0:
            return

        self.ax.clear()
        self.ax.set_facecolor(BG_DARK)

        df = self.candles.copy()
        n = len(df)

        # Apply zoom and pan
        end_idx = max(0, n - self.pan_offset)
        start_idx = max(0, end_idx - self.zoom_level)
        df = df.iloc[start_idx:end_idx]

        if len(df) == 0:
            return

        # Get OHLC columns
        ocol = next((c for c in df.columns if c.lower() == "open"), None)
        hcol = next((c for c in df.columns if c.lower() == "high"), None)
        lcol = next((c for c in df.columns if c.lower() == "low"), None)
        ccol = next((c for c in df.columns if c.lower() == "close"), None)
        tcol = next((c for c in df.columns if c.lower() in ("time", "datetime", "date")), None)

        if not all([ocol, hcol, lcol, ccol]):
            return

        opens = df[ocol].values.astype(float)
        highs = df[hcol].values.astype(float)
        lows = df[lcol].values.astype(float)
        closes = df[ccol].values.astype(float)
        x = np.arange(len(df))

        # Draw candles
        width = 0.6
        for i in range(len(df)):
            color = CANDLE_UP if closes[i] >= opens[i] else CANDLE_DN
            # Wick
            self.ax.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=0.8)
            # Body
            body_low = min(opens[i], closes[i])
            body_high = max(opens[i], closes[i])
            body_height = max(body_high - body_low, 0.01)
            rect = plt.Rectangle((x[i] - width/2, body_low), width, body_height,
                                  facecolor=color, edgecolor=color, linewidth=0.5)
            self.ax.add_patch(rect)

        # Draw trade markers
        if tcol and len(self.trades) > 0:
            times = df[tcol].values
            for trade in self.trades:
                try:
                    # Find closest candle
                    trade_time = trade["time"]
                    if hasattr(trade_time, "timestamp"):
                        trade_ts = trade_time.timestamp()
                    else:
                        trade_ts = float(trade_time)

                    min_diff = float("inf")
                    best_idx = -1
                    for j, t in enumerate(times):
                        if hasattr(t, "timestamp"):
                            diff = abs(t.timestamp() - trade_ts)
                        else:
                            diff = abs(float(t)/1e9 - trade_ts)
                        if diff < min_diff:
                            min_diff = diff
                            best_idx = j

                    if best_idx >= 0:
                        price = trade["price"]
                        if trade["type"] == "BUY":
                            self.ax.scatter(best_idx, price, marker="^", color=GREEN,
                                          s=120, zorder=5, edgecolors="white", linewidths=0.5)
                        else:
                            self.ax.scatter(best_idx, price, marker="v", color=RED,
                                          s=120, zorder=5, edgecolors="white", linewidths=0.5)
                except Exception:
                    pass

        # Trade P&L annotations
        # (Already handled via scatter markers above)

        # X-axis labels
        if tcol:
            times = df[tcol].values
            step = max(1, len(df) // 8)
            tick_positions = list(range(0, len(df), step))
            tick_labels = []
            for idx in tick_positions:
                t = times[idx]
                try:
                    if hasattr(t, "strftime"):
                        tick_labels.append(t.strftime("%H:%M"))
                    else:
                        dt = datetime.datetime.utcfromtimestamp(int(t) / 1e9 if int(t) > 1e12 else int(t))
                        tick_labels.append(dt.strftime("%H:%M"))
                except Exception:
                    tick_labels.append(str(idx))
            self.ax.set_xticks(tick_positions)
            self.ax.set_xticklabels(tick_labels, fontsize=7, color=FG_DIM)

        # Open position entry price lines
        for pos in self.open_positions:
            try:
                pc = GREEN if pos.profit >= 0 else RED
                self.ax.axhline(y=pos.price_open, color=pc, linewidth=0.8, linestyle=":", alpha=0.6)
                side = "BUY" if pos.type == 0 else "SELL"
                self.ax.text(0, pos.price_open, f" {side}@{pos.price_open:.2f} ({pos.profit:+.2f})",
                            fontsize=7, color=pc, va="bottom")
            except Exception: pass

        # Current price line
        if len(closes) > 0:
            last_price = closes[-1]
            self.ax.axhline(y=last_price, color=BLUE, linewidth=0.5, linestyle="--", alpha=0.7)
            self.ax.text(len(df) - 1, last_price, f" {last_price:.2f}",
                        fontsize=8, color=BLUE, va="center")

        # Styling
        self.ax.tick_params(colors=FG_DIM, labelsize=8)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_color("#30363d")
        self.ax.spines["left"].set_color("#30363d")
        self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        self.ax.set_title(f"{self.symbol} {self.timeframe}", fontsize=10, color=GOLD, pad=5)
        self.ax.grid(True, alpha=0.1, color="#30363d")

        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _update_positions(self):
        """Fetch open positions from MT5."""
        if not MT5_AVAILABLE:
            return
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                self.open_positions = []
                self.pnl = 0.0
            else:
                self.open_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
                self.pnl = sum(p.profit for p in self.open_positions)
        except Exception:
            pass

    def update_display(self):
        """Called every second to update everything."""
        self._update_positions()

        # Update price
        try:
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    self.lbl_price.config(text=f"Bid: {tick.bid:.2f}  Ask: {tick.ask:.2f}")
        except Exception:
            pass

        # Update P&L
        pnl_color = GREEN if self.pnl >= 0 else RED
        self.lbl_pnl.config(text=f"P&L: ${self.pnl:.2f}", fg=pnl_color)

        # Update positions count
        self.lbl_positions.config(text=f"Positions: {len(self.open_positions)}")
        # Update realised P&L
        try:
            wins = sum(1 for t in self.trades if t.get("result") == "WIN")
            losses = sum(1 for t in self.trades if t.get("result") == "LOSS")
            realised = sum(t.get("profit", 0) for t in self.trades if t.get("result") in ("WIN","LOSS"))
            rc = GREEN if realised >= 0 else RED
            self.lbl_realised.config(text=f"Realised: ${realised:+.2f}", fg=rc)
            self.lbl_wl.config(text=f"W:{wins} L:{losses}")
        except Exception: pass

    def _start_paper(self):
        """Launch paper trading bridge for this symbol."""
        self.lbl_status.config(text=f"Starting paper trade for {self.symbol}...", fg=GOLD)
        try:
            import subprocess
            bridge_path = os.path.join(BASE, "live_trading_bridge.py")
            subprocess.Popen(
                ["python", bridge_path, "--mode", "paper"],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            )
            self.lbl_status.config(text="Paper trading started in new window", fg=GREEN)
        except Exception as e:
            self.lbl_status.config(text=f"Error: {e}", fg=RED)

    def _trade_now(self):
        """Place an immediate trade based on current ML signal."""
        self.lbl_status.config(text="Analyzing...", fg=GOLD)
        try:
            sig_path = os.path.join(BASE, "current_signal.json")
            if os.path.exists(sig_path):
                with open(sig_path) as f:
                    sig = json.load(f)
                signal = sig.get("signal", "NEUTRAL")
                conf = sig.get("confidence", 0.5)
                if signal in ("BUY", "SELL") and conf > 0.55:
                    if MT5_AVAILABLE:
                        tick = mt5.symbol_info_tick(self.symbol)
                        si = mt5.symbol_info(self.symbol)
                        if tick and si:
                            if not si.visible:
                                mt5.symbol_select(self.symbol, True)
                            pt = si.point
                            if signal == "BUY":
                                ot = mt5.ORDER_TYPE_BUY
                                ep = tick.ask
                                sl = ep - 2500 * pt
                                tp = ep + 5000 * pt
                            else:
                                ot = mt5.ORDER_TYPE_SELL
                                ep = tick.bid
                                sl = ep + 2500 * pt
                                tp = ep - 5000 * pt

                            req = {
                                "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
                                "volume": 0.01, "type": ot, "price": ep,
                                "sl": sl, "tp": tp, "deviation": 20,
                                "magic": MAGIC_NUMBER, "comment": f"AFB manual {signal}",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            result = mt5.order_send(req)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                self.lbl_status.config(text=f"{signal} 0.01 @ {ep:.2f} OK!", fg=GREEN)
                                self.trades.append({"time": datetime.datetime.now(), "type": signal,
                                                    "price": ep, "lot": 0.01})
                            else:
                                rc = result.retcode if result else "None"
                                self.lbl_status.config(text=f"Order failed: {rc}", fg=RED)
                else:
                    self.lbl_status.config(text=f"Signal: {signal} ({conf:.0%}) - too weak", fg=ORANGE)
            else:
                self.lbl_status.config(text="No signal file found", fg=RED)
        except Exception as e:
            self.lbl_status.config(text=f"Error: {e}", fg=RED)

    def _close_all(self):
        """Close all open positions for this symbol."""
        if not MT5_AVAILABLE:
            self.lbl_status.config(text="MT5 not available", fg=RED)
            return
        if not self.open_positions:
            self.lbl_status.config(text="No positions to close", fg=FG_DIM)
            return

        confirm = messagebox.askyesno("Close All", f"Close {len(self.open_positions)} position(s) for {self.symbol}?")
        if not confirm:
            return

        closed = 0
        for p in self.open_positions:
            try:
                close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(self.symbol)
                price = tick.bid if p.type == 0 else tick.ask
                req = {
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
                    "volume": p.volume, "type": close_type, "position": p.ticket,
                    "price": price, "deviation": 20, "magic": MAGIC_NUMBER,
                    "comment": "AFB close all", "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(req)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    closed += 1
            except Exception:
                pass

        self.lbl_status.config(text=f"Closed {closed}/{len(self.open_positions)} positions", fg=GREEN)

    def _backtest(self):
        """Launch backtest."""
        self.lbl_status.config(text="Starting backtest...", fg=GOLD)
        try:
            import subprocess
            bt_path = os.path.join(BASE, "walkforward_backtest.py")
            if os.path.exists(bt_path):
                subprocess.Popen(
                    ["python", bt_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
                )
                self.lbl_status.config(text="Backtest running in new window", fg=GREEN)
            else:
                self.lbl_status.config(text="walkforward_backtest.py not found", fg=RED)
        except Exception as e:
            self.lbl_status.config(text=f"Error: {e}", fg=RED)


class TradingDashboard:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AlmostFinishedBot - Trading Dashboard")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1400x900")
        self.root.minsize(900, 600)

        # Try to set icon
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        # ── Header ──
        header = tk.Frame(self.root, bg=BG_PANEL, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="  AlmostFinishedBot", font=("Consolas", 18, "bold"),
                 fg=GOLD, bg=BG_PANEL).pack(side="left", padx=10)

        self.lbl_account = tk.Label(header, text="Account: ---", font=("Consolas", 11),
                                     fg=FG_TEXT, bg=BG_PANEL)
        self.lbl_account.pack(side="left", padx=20)

        self.lbl_balance = tk.Label(header, text="Balance: ---", font=("Consolas", 11, "bold"),
                                     fg=GREEN, bg=BG_PANEL)
        self.lbl_balance.pack(side="right", padx=20)

        self.lbl_equity = tk.Label(header, text="Equity: ---", font=("Consolas", 11),
                                    fg=FG_TEXT, bg=BG_PANEL)
        self.lbl_equity.pack(side="right", padx=10)

        self.lbl_clock = tk.Label(header, text="", font=("Consolas", 10),
                                   fg=FG_DIM, bg=BG_PANEL)
        self.lbl_clock.pack(side="right", padx=10)

        # ── Symbol panels container ──
        container = tk.Frame(self.root, bg=BG_DARK)
        container.pack(fill="both", expand=True, padx=5, pady=5)

        # Create panels for each symbol
        self.panels = {}
        self.panels["XAUUSD"] = SymbolPanel(container, "XAUUSD", row=0)
        self.panels["XAUSGD"] = SymbolPanel(container, "XAUSGD", row=1)

        # ── Connect to MT5 ──
        self._connect_mt5()

        # ── Start update loops ──
        self._update_loop()
        self._chart_update_loop()

    def _connect_mt5(self):
        if not MT5_AVAILABLE:
            self.lbl_account.config(text="MT5 not installed", fg=RED)
            return

        if not mt5.initialize():
            self.lbl_account.config(text="MT5 not connected", fg=RED)
            return

        info = mt5.account_info()
        if info:
            self.lbl_account.config(text=f"Account: {info.login} ({info.server})")
            self.lbl_balance.config(text=f"Balance: ${info.balance:.2f}")
            self.lbl_equity.config(text=f"Equity: ${info.equity:.2f}")

            # Ensure symbols are visible
            for sym in ["XAUUSD", "XAUSGD"]:
                try:
                    si = mt5.symbol_info(sym)
                    if si and not si.visible:
                        mt5.symbol_select(sym, True)
                except Exception:
                    pass

        # Initial data fetch + historical trades
        for panel in self.panels.values():
            panel._load_historical_trades()
            panel._fetch_data()
            panel._draw_chart()

    def _update_loop(self):
        """Update prices, positions, P&L every second."""
        try:
            # Update account info
            if MT5_AVAILABLE:
                info = mt5.account_info()
                if info:
                    bal_color = GREEN if info.equity >= info.balance else RED
                    self.lbl_balance.config(text=f"Balance: ${info.balance:.2f}")
                    self.lbl_equity.config(text=f"Equity: ${info.equity:.2f}", fg=bal_color)

            # Update clock
            now = datetime.datetime.now(datetime.timezone.utc)
            self.lbl_clock.config(text=now.strftime("%H:%M:%S UTC"))

            # Update each panel
            for panel in self.panels.values():
                panel.update_display()

        except Exception:
            pass

        self.root.after(1000, self._update_loop)

    def _chart_update_loop(self):
        """Refresh chart data every 5 seconds."""
        try:
            for panel in self.panels.values():
                panel._load_historical_trades()
                panel._fetch_data()
                panel._draw_chart()
        except Exception:
            pass

        self.root.after(5000, self._chart_update_loop)

    def run(self):
        self.root.mainloop()
        # Cleanup
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
            except Exception:
                pass


def main():
    print("=" * 55)
    print("  AlmostFinishedBot - Trading Dashboard")
    print("  Loading...")
    print("=" * 55)

    app = TradingDashboard()
    app.run()


if __name__ == "__main__":
    main()
