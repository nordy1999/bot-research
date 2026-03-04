"""
AlmostFinishedBot - Live Graph
Always shows price (GC=F proxy for XAUUSD), overlays EA trade markers from MT5 logs
Grey/white/red theme
"""
import sys, os, time, re, threading
import tkinter as tk
from tkinter import ttk
import warnings; warnings.filterwarnings("ignore")

BASE   = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
LOGDIR = os.path.join(os.environ.get("APPDATA",""), "MetaQuotes", "Terminal",
         "73B7A2420D6397DFF9014A20F1201F97", "MQL5", "Logs")
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"

# ── Colours ──────────────────────────────────────────────────────
BG_DARK  = "#111111"; BG_MID   = "#1e1e1e"; BG_PANEL = "#2a2a2a"
WHITE    = "#ffffff";  GREY_LT  = "#cccccc"; GREY_MID = "#888888"
GREY_DIM = "#555555";  RED_HI   = "#ff2222"; RED_MID  = "#cc3333"
GREEN_TR = "#22ff88";  RED_TR   = "#ff2244"; GOLD_LINE= "#ffcc44"
CANDLE_UP= "#33cc66";  CANDLE_DN= "#dd2244"
REFRESH  = 15  # seconds

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import pandas as pd
    import numpy as np
    HAS_MPL = True
except:
    HAS_MPL = False

try:
    import yfinance as yf; HAS_YF = True
except:
    HAS_YF = False

def clean_mt5(line):
    line = re.sub(r'(?<=\S) (?=\S)', '', line)
    return re.sub(r'\s+', ' ', line).strip()

class LiveGraph:
    def __init__(self, root):
        self.root   = root
        self.df     = None
        self.trades = []
        self.lock   = threading.Lock()
        self.interval = tk.StringVar(value="15m")
        self.period   = tk.StringVar(value="1d")
        self.running  = True
        self._build_ui()
        threading.Thread(target=self._fetch_loop, daemon=True).start()
        threading.Thread(target=self._log_loop,   daemon=True).start()

    def _build_ui(self):
        r = self.root
        r.title(f"AlmostFinishedBot — Live Chart  [{SYMBOL}]")
        r.configure(bg=BG_MID)
        r.geometry("1280x780")

        # Header
        hdr = tk.Frame(r, bg=BG_DARK, highlightbackground=WHITE, highlightthickness=1)
        hdr.pack(fill="x", padx=8, pady=(8,0))
        tk.Label(hdr, text=f"  ALMOSTFINISHEDBOT  |  LIVE CHART  |  {SYMBOL}  ",
                 fg=RED_HI, bg=BG_DARK, font=("Consolas",13,"bold")).pack(side="left")
        self.price_lbl = tk.Label(hdr, text="Fetching price...",
                                  fg=GREY_LT, bg=BG_DARK, font=("Consolas",11))
        self.price_lbl.pack(side="right", padx=10)

        # Toolbar
        tb = tk.Frame(r, bg=BG_PANEL, highlightbackground=WHITE, highlightthickness=1)
        tb.pack(fill="x", padx=8, pady=4)
        tk.Label(tb, text="  Interval:", fg=GREY_LT, bg=BG_PANEL,
                 font=("Consolas",9)).pack(side="left")
        for iv in ["1m","5m","15m","30m","1h"]:
            tk.Radiobutton(tb, text=iv, variable=self.interval, value=iv,
                bg=BG_PANEL, fg=GREY_LT, selectcolor=RED_MID,
                activebackground=BG_PANEL, font=("Consolas",9),
                command=self._refresh).pack(side="left", padx=2)
        tk.Label(tb, text="   Period:", fg=GREY_LT, bg=BG_PANEL,
                 font=("Consolas",9)).pack(side="left")
        for pv in ["1d","5d","1mo"]:
            tk.Radiobutton(tb, text=pv, variable=self.period, value=pv,
                bg=BG_PANEL, fg=GREY_LT, selectcolor=RED_MID,
                activebackground=BG_PANEL, font=("Consolas",9),
                command=self._refresh).pack(side="left", padx=2)
        tk.Button(tb, text=" Refresh Now ", bg=RED_MID, fg=WHITE,
                  font=("Consolas",9,"bold"), relief="flat",
                  command=self._refresh).pack(side="right", padx=6, pady=3)
        tk.Button(tb, text=" Clear Markers ", bg=BG_DARK, fg=GREY_MID,
                  font=("Consolas",9), relief="flat",
                  command=self._clear_trades).pack(side="right", padx=4)

        # Canvas
        self.fig_frame = tk.Frame(r, bg=BG_MID,
                                  highlightbackground=WHITE, highlightthickness=1)
        self.fig_frame.pack(fill="both", expand=True, padx=8, pady=4)

        if HAS_MPL:
            self.fig, (self.ax, self.axv) = plt.subplots(
                2, 1, figsize=(12,6.5),
                gridspec_kw={"height_ratios":[4,1],"hspace":0.05},
                facecolor=BG_MID)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            tk.Label(self.fig_frame, text="matplotlib not installed\npip install matplotlib",
                     fg=RED_HI, bg=BG_MID, font=("Consolas",12)).pack(expand=True)

        # Status bar
        sb = tk.Frame(r, bg=BG_DARK, highlightbackground=WHITE, highlightthickness=1)
        sb.pack(fill="x", padx=8, pady=(0,8))
        self.status_lbl = tk.Label(sb, text="  Initialising...",
                                   fg=GREY_MID, bg=BG_DARK, font=("Consolas",8))
        self.status_lbl.pack(side="left")
        self.trade_lbl = tk.Label(sb, text="",
                                  fg=GREY_MID, bg=BG_DARK, font=("Consolas",8))
        self.trade_lbl.pack(side="right", padx=8)

    def _fetch_loop(self):
        while self.running:
            self._do_fetch()
            for _ in range(REFRESH*10):
                if not self.running: break
                time.sleep(0.1)

    def _do_fetch(self):
        if not HAS_YF: return
        ticker = "GC=F"
        try:
            data = yf.download(ticker, period=self.period.get(),
                               interval=self.interval.get(),
                               progress=False, auto_adjust=True)
            if data is None or len(data) < 5: return
            if hasattr(data.columns,"levels"):
                data.columns = [c[0].lower() for c in data.columns]
            else:
                data.columns = [c.lower() for c in data.columns]
            if hasattr(data.index,"tz") and data.index.tz:
                data.index = data.index.tz_convert("UTC").tz_localize(None)
            data = data.tail(80)
            with self.lock: self.df = data
            self.root.after(0, self._draw)
        except Exception as e:
            self.root.after(0, lambda: self.status_lbl.config(text=f"  Fetch error: {e}"))

    def _log_loop(self):
        last_size = 0; last_log = None
        while self.running:
            try:
                if os.path.exists(LOGDIR):
                    logs = sorted([f for f in os.listdir(LOGDIR) if f.endswith(".log")])
                    if logs:
                        lp = os.path.join(LOGDIR, logs[-1])
                        if lp != last_log: last_log=lp; last_size=0
                        sz = os.path.getsize(lp)
                        if sz > last_size:
                            with open(lp,"r",encoding="utf-8",errors="ignore") as f:
                                f.seek(last_size); raw=f.read()
                            last_size = sz
                            for line in raw.splitlines():
                                cl = clean_mt5(line)
                                m = re.search(r'(\d{2}:\d{2}:\d{2}).*?(XAUUSD|XAUSGD)\s+(BUY|SELL)\s*\|\s*Lots:\s*([\d.]+)', cl)
                                if m:
                                    t_str, sym, side, lots = m.groups()
                                    with self.lock:
                                        self.trades.append({"time":t_str,"symbol":sym,
                                                            "side":side,"lots":lots})
            except: pass
            time.sleep(5)

    def _draw(self):
        if not HAS_MPL: return
        with self.lock:
            df = self.df.copy() if self.df is not None else None
            trades = list(self.trades)

        self.ax.clear(); self.axv.clear()
        for a in [self.ax, self.axv]:
            a.set_facecolor(BG_MID)
            a.tick_params(colors=GREY_MID, labelsize=7)
            for sp in a.spines.values(): sp.set_color(GREY_DIM)
            a.yaxis.label.set_color(GREY_MID)

        if df is None or len(df) < 5:
            self.ax.text(0.5, 0.5, "Fetching price data...",
                         transform=self.ax.transAxes, color=GREY_MID,
                         ha="center", va="center", fontsize=13,
                         fontfamily="Consolas")
            self.canvas.draw()
            return

        xs = range(len(df))
        # Candlesticks
        for i, (idx, row) in enumerate(df.iterrows()):
            o,h,l,c = row["open"],row["high"],row["low"],row["close"]
            col = CANDLE_UP if c >= o else CANDLE_DN
            self.ax.plot([i,i],[l,h], color=col, linewidth=0.8, zorder=2)
            self.ax.bar(i, abs(c-o), bottom=min(o,c), color=col,
                        width=0.7, zorder=3, linewidth=0)

        # EMAs
        import numpy as np
        closes = df["close"].values
        if len(closes) >= 20:
            ema20 = [None]*19
            e = float(np.mean(closes[:20]))
            ema20.append(e)
            k = 2/21
            for c in closes[20:]:
                e = float(c)*k + e*(1-k); ema20.append(e)
            self.ax.plot(list(xs), ema20, color="#ff9500", linewidth=1, label="EMA20", zorder=4)
        if len(closes) >= 50:
            ema50 = [None]*49
            e = float(np.mean(closes[:50]))
            ema50.append(e)
            k = 2/51
            for c in closes[50:]:
                e = float(c)*k + e*(1-k); ema50.append(e)
            self.ax.plot(list(xs), ema50, color="#00ccff", linewidth=1, label="EMA50", zorder=4)

        # Current price line
        last_p = float(df["close"].iloc[-1])
        self.ax.axhline(last_p, color=GOLD_LINE, linewidth=0.8, linestyle="--", zorder=5)

        # Trade markers
        today_str = time.strftime("%Y-%m-%d")
        plotted = 0
        for tr in trades:
            try:
                tr_dt_str = today_str + " " + tr["time"]
                import pandas as pd
                tr_dt = pd.Timestamp(tr_dt_str)
                diffs = abs(df.index - tr_dt)
                idx_c = diffs.argmin()
                if diffs[idx_c].total_seconds() < 3600*4:
                    col = GREEN_TR if tr["side"]=="BUY" else RED_TR
                    marker = "^" if tr["side"]=="BUY" else "v"
                    y_off = float(df["low"].iloc[idx_c]) * 0.9998 if tr["side"]=="BUY" else float(df["high"].iloc[idx_c]) * 1.0002
                    self.ax.scatter(idx_c, y_off, color=col, marker=marker, s=120, zorder=6)
                    self.ax.text(idx_c, y_off, f"\n{tr['side']}\n{tr['lots']}",
                                color=col, fontsize=6, ha="center", zorder=7,
                                fontfamily="Consolas")
                    self.ax.axvline(idx_c, color=col, linewidth=0.5,
                                    linestyle=":", alpha=0.5, zorder=3)
                    plotted += 1
            except: pass

        if not trades:
            self.ax.text(0.01, 0.98, "No EA trades detected — showing price only",
                         transform=self.ax.transAxes, color=GREY_DIM,
                         va="top", fontsize=7, fontfamily="Consolas")

        # Volume
        try:
            vols = df["volume"].values if "volume" in df.columns else np.ones(len(df))
            bar_colors = [CANDLE_UP if df["close"].iloc[i]>=df["open"].iloc[i]
                         else CANDLE_DN for i in range(len(df))]
            self.axv.bar(list(xs), vols, color=bar_colors, width=0.7, linewidth=0)
            self.axv.set_yticks([])
        except: pass

        # X labels
        import pandas as pd
        n = len(df)
        step = max(1, n//10)
        xticks = list(range(0,n,step))
        xlabels = [df.index[i].strftime("%H:%M\n%d/%m") for i in xticks]
        self.ax.set_xticks([]); self.axv.set_xticks(xticks)
        self.axv.set_xticklabels(xlabels, color=GREY_MID, fontsize=6)
        self.axv.tick_params(colors=GREY_MID, labelsize=6)

        self.ax.yaxis.set_label_position("right"); self.ax.yaxis.tick_right()
        self.ax.set_xlim(-1, n)
        self.ax.legend(loc="upper left", fontsize=7,
                       facecolor=BG_PANEL, edgecolor=GREY_DIM, labelcolor=GREY_LT)
        self.axv.set_xlim(-1, n)
        self.fig.tight_layout(pad=0.3)
        self.canvas.draw()

        # Update labels
        prev_p = float(df["close"].iloc[-2]) if len(df)>1 else last_p
        chg = last_p - prev_p; pct = chg/prev_p*100 if prev_p else 0
        col = "#22cc66" if chg>=0 else RED_HI
        self.price_lbl.config(text=f"{SYMBOL}  ${last_p:,.2f}  {chg:+.2f} ({pct:+.2f}%)", fg=col)
        self.status_lbl.config(
            text=f"  Last update: {time.strftime('%H:%M:%S')}  "
                 f"Candles: {len(df)}  Trades: {plotted}  Refresh: {REFRESH}s")
        tcol = GREEN_TR if trades else GREY_DIM
        tmsg = f"Trades today: {len(trades)}" if trades else "No EA trades yet"
        self.trade_lbl.config(text=f"{tmsg}  ", fg=tcol)

    def _refresh(self):
        threading.Thread(target=self._do_fetch, daemon=True).start()

    def _clear_trades(self):
        with self.lock: self.trades.clear()
        self._draw()

def main():
    root = tk.Tk()
    if not HAS_MPL:
        root.title("Error"); root.geometry("400x200")
        tk.Label(root, text="Install matplotlib:\npip install matplotlib",
                 font=("Consolas",12)).pack(expand=True)
    else:
        app = LiveGraph(root)
    root.mainloop()

if __name__ == "__main__":
    main()
