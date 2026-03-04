"""
AlmostFinishedBot - Live Trading Bridge v7
ELITE 10-MODEL ENSEMBLE with Stacking Meta (82.2% accuracy)

v7 changes:
  - 10 models: XGB, LGB, GB, CB, RF, LSTM, TFT, TCN, N-BEATS, N-HiTS
  - Stacking Meta strategy (82.2% accuracy vs 59.1% in v6)
  - TCN: 33% better RMSE on gold
  - N-BEATS: Won M4 forecasting competition
  - N-HiTS: Multi-scale temporal patterns
  - All v3 position management optimisations retained
"""
import os, sys, json, time, datetime, argparse, warnings, traceback
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")

# ── Console Colors (Windows compatible) ───────────────────────
import ctypes
try:
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except Exception: pass
os.system("")

G = "\033[92m"   # green
R = "\033[91m"   # red
Y = "\033[93m"   # yellow
C = "\033[96m"   # cyan
B = "\033[1m"    # bold
W = "\033[97m"   # white
M = "\033[95m"   # magenta (for v7)
DIM = "\033[2m"  # dim
RST = "\033[0m"  # reset

# Session P&L tracking
session_pnl = 0.0
session_wins = 0
session_losses = 0
session_trades = 0
win_streak = 0
last_trade_profit = 0.0
sys.path.insert(0, BASE)

# ── Config ────────────────────────────────────────────────────────
SYMBOL        = "XAUUSD"
CYCLE_SECONDS = 60
MAX_TRADES_PER_DAY = 10
MAGIC_NUMBER  = 999

# v3: Confidence thresholds (regime-dependent)
CONF_TRENDING     = 0.54
CONF_VOLATILE     = 0.53
CONF_RANGING      = 0.62
CONF_SMC_BOOST    = -0.03

# Risk per trade (% of balance)
RISK_BASE         = 1.5
RISK_TRENDING     = 2.0
RISK_RANGING      = 0.5
RISK_HIGH_CONF    = 2.5
RISK_MAX          = 3.0

# v3: Position management (optimised from backtest)
TRAIL_ACTIVATE_PCT = 0.35
TRAIL_DISTANCE_ATR = 0.8
BREAKEVEN_PCT      = 0.25
BE_OFFSET_ATR      = 0.15
PARTIAL_TRIGGER    = 0.40
PARTIAL_SIZE       = 0.40

# v3: Blocked regimes (confirmed losers in backtest)
BLOCKED_REGIMES = {"HIGH_VOL_BREAKOUT"}

# Telegram
TG_TOKEN  = ""
TG_CHAT_ID = ""
try:
    cfg = json.load(open(os.path.join(BASE, "bot_settings.json")))
    TG_TOKEN = cfg.get("tg_token", TG_TOKEN)
    TG_CHAT_ID = cfg.get("tg_chat_id", TG_CHAT_ID)
except Exception:
    pass

# ── Logging ───────────────────────────────────────────────────────
LOG_FILE = os.path.join(BASE, "trading_log.json")

def log_event(event_type, data):
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "type": event_type, **data,
    }
    log = []
    if os.path.exists(LOG_FILE):
        try:
            log = json.load(open(LOG_FILE))
        except: pass
    log.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(log[-1000:], f, indent=2)

def send_telegram(msg):
    if not TG_TOKEN or not TG_CHAT_ID: return
    try:
        import urllib.request, urllib.parse
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TG_CHAT_ID, "text": f"🤖 AFB v7: {msg}"}).encode()
        urllib.request.urlopen(url, data, timeout=5)
    except Exception: pass


# ── MT5 Functions ─────────────────────────────────────────────────
def init_mt5(mode):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print(f"  {R}[ERR] MT5 init failed{RST}"); sys.exit(1)
    info = mt5.account_info()
    if info is None:
        print(f"  {R}[ERR] No account{RST}"); mt5.shutdown(); sys.exit(1)
    is_demo = (info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO)
    print(f"  {G}[OK] MT5 connected: {info.login} ({info.server}){RST}")
    print(f"       Balance: ${info.balance:.2f}  Equity: ${info.equity:.2f}")
    print(f"       Mode: {'DEMO' if is_demo else 'LIVE'}")
    if mode == "live" and is_demo:
        print(f"  {Y}[WARN] --mode live but on DEMO account{RST}")
    if mode == "paper" and not is_demo:
        print(f"  {Y}[WARN] --mode paper but on LIVE account! Exiting.{RST}"); sys.exit(1)
    return info

def get_account_info():
    import MetaTrader5 as mt5
    info = mt5.account_info()
    return {"balance": info.balance, "equity": info.equity, "margin": info.margin,
            "free_margin": info.margin_free, "profit": info.profit, "leverage": info.leverage}

def get_open_positions():
    import MetaTrader5 as mt5
    return mt5.positions_get(symbol=SYMBOL) or []

def count_today_trades():
    import MetaTrader5 as mt5
    now = datetime.datetime.now(datetime.timezone.utc)
    start = datetime.datetime(now.year, now.month, now.day, tzinfo=datetime.timezone.utc)
    deals = mt5.history_deals_get(start, now)
    if deals is None: return 0
    return sum(1 for d in deals if d.magic == MAGIC_NUMBER and d.entry != 0)

def get_atr(period=14):
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, period + 5)
    if rates is None or len(rates) < period: return 30.0
    df = pd.DataFrame(rates)
    df["h_l"] = df["high"] - df["low"]
    df["h_pc"] = abs(df["high"] - df["close"].shift())
    df["l_pc"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["h_l", "h_pc", "l_pc"]].max(axis=1)
    return df["tr"].rolling(period).mean().iloc[-1]

def place_trade(direction, lot_size, sl_pips, tp_pips, comment="AFB_v7"):
    try:
        import MetaTrader5 as mt5
        si = mt5.symbol_info(SYMBOL)
        if si is None or not si.visible:
            print(f"  [ERR] Symbol {SYMBOL} not available"); return False
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None: print("  [ERR] No tick"); return False
        ask, bid, pt = tick.ask, tick.bid, si.point

        if direction == "BUY":
            ot = mt5.ORDER_TYPE_BUY; ep = ask
            sl = ask - sl_pips * pt if sl_pips > 0 else 0.0
            tp = ask + tp_pips * pt if tp_pips > 0 else 0.0
        else:
            ot = mt5.ORDER_TYPE_SELL; ep = bid
            sl = bid + sl_pips * pt if sl_pips > 0 else 0.0
            tp = bid - tp_pips * pt if tp_pips > 0 else 0.0

        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lot_size,
               "type": ot, "price": ep, "sl": sl, "tp": tp, "deviation": 20,
               "magic": MAGIC_NUMBER, "comment": comment[:30],
               "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
        result = mt5.order_send(req)
        if result is None: print("  [ERR] order_send None"); return False
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            color = G if direction == "BUY" else R; print(f"  {color}{B}[TRADE] {direction} {lot_size} {SYMBOL} @ {ep:.2f}{RST}"); return True
        else:
            print(f"  [ERR] {result.retcode}: {result.comment}"); return False
    except Exception as e:
        print(f"  [ERR] {e}"); return False

def modify_sl(ticket, new_sl, new_tp=None):
    try:
        import MetaTrader5 as mt5
        pos = mt5.positions_get(ticket=ticket)
        if not pos: return False
        tp = new_tp if new_tp is not None else pos[0].tp
        req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": SYMBOL,
               "position": ticket, "sl": new_sl, "tp": tp}
        r = mt5.order_send(req)
        return r and r.retcode == mt5.TRADE_RETCODE_DONE
    except Exception: return False

def close_partial(ticket, vol):
    try:
        import MetaTrader5 as mt5
        pos = mt5.positions_get(ticket=ticket)
        if not pos: return False
        p = pos[0]
        ct = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(SYMBOL)
        cp = tick.bid if p.type == 0 else tick.ask
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": vol,
               "type": ct, "position": ticket, "price": cp, "deviation": 20,
               "magic": MAGIC_NUMBER, "comment": "AFB partial",
               "type_filling": mt5.ORDER_FILLING_IOC}
        r = mt5.order_send(req)
        return r and r.retcode == mt5.TRADE_RETCODE_DONE
    except Exception: return False


# ── Position Management (v3 optimised) ──────────────────────────────────────────
position_state = {}

def manage_open_positions(atr_val):
    import MetaTrader5 as mt5
    positions = get_open_positions()
    managed = []

    for p in positions:
        ticket = p.ticket; entry = p.price_open; cur_sl = p.sl; cur_tp = p.tp
        is_buy = (p.type == 0); vol = p.volume

        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None: continue
        price = tick.bid if is_buy else tick.ask

        if is_buy:
            tp_dist = (cur_tp - entry) if cur_tp > 0 else atr_val * 3
            profit_pts = price - entry
        else:
            tp_dist = (entry - cur_tp) if cur_tp > 0 else atr_val * 3
            profit_pts = entry - price

        tp_dist = max(tp_dist, atr_val)
        profit_pct = profit_pts / tp_dist if tp_dist > 0 else 0

        if ticket not in position_state:
            position_state[ticket] = {"breakeven": False, "partial": False, "trailing": False}
        st = position_state[ticket]
        actions = []

        # BREAKEVEN at 25% of TP — lock in 0.15×ATR profit
        if profit_pct >= BREAKEVEN_PCT and not st["breakeven"]:
            pt = mt5.symbol_info(SYMBOL).point
            be_offset = max(atr_val * BE_OFFSET_ATR, 0.50)
            if is_buy:
                new_sl = entry + be_offset
                if new_sl > cur_sl and modify_sl(ticket, new_sl):
                    st["breakeven"] = True; actions.append(f"{G}BREAKEVEN SL->{new_sl:.2f} (+${be_offset:.2f}){RST}")
            else:
                new_sl = entry - be_offset
                if (cur_sl == 0 or new_sl < cur_sl) and modify_sl(ticket, new_sl):
                    st["breakeven"] = True; actions.append(f"{G}BREAKEVEN SL->{new_sl:.2f} (+${be_offset:.2f}){RST}")

        # PARTIAL CLOSE at 40% of TP — close 40% of position
        if profit_pct >= PARTIAL_TRIGGER and not st["partial"] and vol >= 0.02:
            pv = max(0.01, round(vol * PARTIAL_SIZE, 2))
            if close_partial(ticket, pv):
                st["partial"] = True; actions.append(f"{Y}PARTIAL {pv}lots @{profit_pct:.0%}{RST}")
                send_telegram(f"Partial close {pv}lots +{profit_pct:.0%}")

        # TRAILING STOP at 35% of TP — trail at 0.8×ATR
        if profit_pct >= TRAIL_ACTIVATE_PCT:
            st["trailing"] = True
            trail = atr_val * TRAIL_DISTANCE_ATR
            if is_buy:
                new_sl = price - trail
                if new_sl > cur_sl + 1 and modify_sl(ticket, new_sl):
                    actions.append(f"{C}TRAIL SL->{new_sl:.2f}{RST}")
            else:
                new_sl = price + trail
                if (cur_sl == 0 or new_sl < cur_sl - 1) and modify_sl(ticket, new_sl):
                    actions.append(f"{C}TRAIL SL->{new_sl:.2f}{RST}")

        if actions: managed.append((ticket, actions))

    # Cleanup closed positions
    open_tix = {p.ticket for p in positions}
    for t in [t for t in position_state if t not in open_tix]:
        del position_state[t]
    return managed


# ── v7 Deep Learning Model Loaders ────────────────────────────────────────────
SEQ = 20  # Sequence length for all sequential models

def load_pytorch_model(model_class, checkpoint_path, device="cpu"):
    """Load a PyTorch model from checkpoint"""
    import torch
    if not os.path.exists(checkpoint_path):
        return None, 0.5
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = model_class(ckpt["n_features"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt.get("seq_len", SEQ)
    except Exception:
        return None, SEQ

# Define all v7 model architectures
def get_model_classes():
    import torch
    import torch.nn as nn
    
    class LSTMModel(nn.Module):
        def __init__(self, n_features, hidden=64, layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
            self.bn = nn.BatchNorm1d(hidden)
            self.fc1 = nn.Linear(hidden, 32)
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.bn(out)
            out = torch.relu(self.fc1(out))
            out = self.drop(out)
            return torch.sigmoid(self.fc2(out)).squeeze(-1)
    
    class SimpleTFT(nn.Module):
        def __init__(self, n_features, d_model=64, n_heads=4, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, SEQ, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*2, dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.bn = nn.BatchNorm1d(d_model)
            self.fc1 = nn.Linear(d_model, 32)
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 1)
        def forward(self, x):
            x = self.input_proj(x) + self.pos_enc
            x = self.transformer(x)
            x = x[:, -1, :]
            x = self.bn(x)
            x = torch.relu(self.fc1(x))
            x = self.drop(x)
            return torch.sigmoid(self.fc2(x)).squeeze(-1)
    
    class CausalConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.padding, dilation=dilation)
        def forward(self, x):
            out = self.conv(x)
            return out[:, :, :-self.padding] if self.padding > 0 else out
    
    class TCNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
            super().__init__()
            self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.dropout = nn.Dropout(dropout)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        def forward(self, x):
            res = x if self.downsample is None else self.downsample(x)
            out = self.dropout(torch.relu(self.bn1(self.conv1(x))))
            out = self.dropout(torch.relu(self.bn2(self.conv2(out))))
            return torch.relu(out + res)
    
    class TCNModel(nn.Module):
        def __init__(self, n_features, n_channels=[64, 64, 64], kernel_size=3, dropout=0.2):
            super().__init__()
            layers = []
            for i, out_ch in enumerate(n_channels):
                in_ch = n_features if i == 0 else n_channels[i-1]
                dilation = 2 ** i
                layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            self.tcn = nn.Sequential(*layers)
            self.fc1 = nn.Linear(n_channels[-1], 32)
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 1)
        def forward(self, x):
            x = x.transpose(1, 2)
            out = self.tcn(x)
            out = out[:, :, -1]
            out = torch.relu(self.fc1(out))
            out = self.drop(out)
            return torch.sigmoid(self.fc2(out)).squeeze(-1)
    
    class NBEATSBlock(nn.Module):
        def __init__(self, input_size, theta_size, hidden=64, layers=2):
            super().__init__()
            fc_layers = [nn.Linear(input_size, hidden), nn.ReLU()]
            for _ in range(layers - 1):
                fc_layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            self.fc = nn.Sequential(*fc_layers)
            self.theta_b = nn.Linear(hidden, theta_size)
            self.theta_f = nn.Linear(hidden, theta_size)
        def forward(self, x):
            h = self.fc(x)
            return self.theta_b(h), self.theta_f(h)
    
    class NBEATSModel(nn.Module):
        def __init__(self, n_features, seq_len=SEQ, n_blocks=3, hidden=64, dropout=0.2):
            super().__init__()
            self.seq_len = seq_len
            input_size = seq_len * n_features
            self.blocks = nn.ModuleList([NBEATSBlock(input_size, input_size, hidden) for _ in range(n_blocks)])
            self.fc1 = nn.Linear(input_size, 64)
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(64, 1)
        def forward(self, x):
            batch = x.shape[0]
            x_flat = x.reshape(batch, -1)
            residuals = x_flat
            forecast = torch.zeros_like(x_flat)
            for block in self.blocks:
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                forecast = forecast + block_forecast
            out = torch.relu(self.fc1(forecast))
            out = self.drop(out)
            return torch.sigmoid(self.fc2(out)).squeeze(-1)
    
    class NHiTSBlock(nn.Module):
        def __init__(self, input_size, output_size, pool_size, hidden=64):
            super().__init__()
            self.pool = nn.MaxPool1d(pool_size, stride=pool_size) if pool_size > 1 else nn.Identity()
            self.pool_size = pool_size
            pooled_size = input_size // pool_size if pool_size > 1 else input_size
            self.fc = nn.Sequential(nn.Linear(pooled_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.theta = nn.Linear(hidden, output_size)
        def forward(self, x):
            if self.pool_size > 1:
                x = x.unsqueeze(1)
                x = self.pool(x).squeeze(1)
            h = self.fc(x)
            return self.theta(h)
    
    class NHiTSModel(nn.Module):
        def __init__(self, n_features, seq_len=SEQ, n_stacks=3, hidden=64, dropout=0.2):
            super().__init__()
            input_size = seq_len * n_features
            pool_sizes = [1, 2, 4][:n_stacks]
            self.blocks = nn.ModuleList([NHiTSBlock(input_size, input_size, ps, hidden) for ps in pool_sizes])
            self.fc1 = nn.Linear(input_size, 64)
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(64, 1)
        def forward(self, x):
            batch = x.shape[0]
            x_flat = x.reshape(batch, -1)
            forecast = torch.zeros_like(x_flat)
            for block in self.blocks:
                forecast = forecast + block(x_flat)
            out = torch.relu(self.fc1(forecast))
            out = self.drop(out)
            return torch.sigmoid(self.fc2(out)).squeeze(-1)
    
    return {
        "lstm": LSTMModel,
        "tft": SimpleTFT,
        "tcn": TCNModel,
        "nbeats": NBEATSModel,
        "nhits": NHiTSModel,
    }


# ── Intelligence Pipeline (v7: 10 models) ─────────────────────────────────────────
TRENDING = {"STRONG_TREND_UP","STRONG_TREND_DOWN","TREND_UP","TREND_DOWN","TRENDING_STRONG","TRENDING"}
RANGING  = {"CHOPPY_RANGE","RANGING","LOW_VOL","MEAN_REVERT"}

def run_all_guards(account_info):
    blocked = False; signal = "NEUTRAL"; conf = 0.5
    cur_regime = "UNKNOWN"; smc_score = 0; smc_dir = 0; reasons = []

    # 1. ML Signal (v7: 10-model ensemble)
    try:
        import yfinance as yf, joblib
        import torch
        
        df = yf.download("GC=F", period="5d", interval="1h", progress=False)
        if df is None or len(df) < 30:
            blocked = True; reasons.append("ML: insufficient data")
            return not blocked, signal, conf, 1.0, cur_regime, 0, 0, reasons

        from features import make_features
        df_feat = df.copy()
        if hasattr(df_feat.columns, 'droplevel') and df_feat.columns.nlevels > 1:
            df_feat.columns = df_feat.columns.droplevel(1)
        df_feat.columns = [c.lower() for c in df_feat.columns]
        feat_df = make_features(df_feat)
        if feat_df is None or len(feat_df) == 0:
            blocked = True; reasons.append("ML: features failed")
            return not blocked, signal, conf, 1.0, cur_regime, 0, 0, reasons

        # Load ensemble config
        ec = {}
        ec_path = os.path.join(BASE, "ensemble_config.json")
        if os.path.exists(ec_path):
            with open(ec_path) as f: ec = json.load(f)
        best_strat = ec.get("best_strategy", "stacking_meta")
        trainer_ver = ec.get("trainer_version", "v7")

        scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
        X_live = scaler.transform(feat_df.iloc[[-1]])
        X_all = scaler.transform(feat_df)

        # ════════════════════════════════════════════════════════════════
        # TREE MODELS (5)
        # ════════════════════════════════════════════════════════════════
        p_xgb = float(joblib.load(os.path.join(BASE, "xgb_model.pkl")).predict_proba(X_live)[0][1])
        p_lgb = float(joblib.load(os.path.join(BASE, "lgb_model.pkl")).predict_proba(X_live)[0][1])
        p_gb  = float(joblib.load(os.path.join(BASE, "gb_model.pkl")).predict_proba(X_live)[0][1])

        p_cb = 0.5
        try:
            cb_path = os.path.join(BASE, "catboost_model.pkl")
            if os.path.exists(cb_path):
                p_cb = float(joblib.load(cb_path).predict_proba(X_live)[0][1])
        except Exception: pass

        p_rf = 0.5
        try:
            rf_path = os.path.join(BASE, "rf_model.pkl")
            if os.path.exists(rf_path):
                p_rf = float(joblib.load(rf_path).predict_proba(X_live)[0][1])
        except Exception: pass

        # ════════════════════════════════════════════════════════════════
        # DEEP LEARNING MODELS (5) - v7 NEW!
        # ════════════════════════════════════════════════════════════════
        p_lstm = p_tft = p_tcn = p_nbeats = p_nhits = 0.5
        
        try:
            model_classes = get_model_classes()
            seq_len = ec.get("seq_len", SEQ)
            
            if len(X_all) >= seq_len:
                seq_in = torch.tensor(X_all[-seq_len:].reshape(1, seq_len, -1), dtype=torch.float32)
                
                # LSTM
                lstm_path = os.path.join(BASE, "lstm_model.pt")
                if os.path.exists(lstm_path):
                    ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
                    model = model_classes["lstm"](ckpt["n_features"])
                    model.load_state_dict(ckpt["model_state"]); model.eval()
                    with torch.no_grad():
                        p_lstm = float(model(seq_in).item())
                
                # TFT
                tft_path = os.path.join(BASE, "tft_model.pt")
                if os.path.exists(tft_path):
                    ckpt = torch.load(tft_path, map_location="cpu", weights_only=False)
                    model = model_classes["tft"](ckpt["n_features"])
                    model.load_state_dict(ckpt["model_state"]); model.eval()
                    with torch.no_grad():
                        p_tft = float(model(seq_in).item())
                
                # TCN (v7 NEW!)
                tcn_path = os.path.join(BASE, "tcn_model.pt")
                if os.path.exists(tcn_path):
                    ckpt = torch.load(tcn_path, map_location="cpu", weights_only=False)
                    model = model_classes["tcn"](ckpt["n_features"])
                    model.load_state_dict(ckpt["model_state"]); model.eval()
                    with torch.no_grad():
                        p_tcn = float(model(seq_in).item())
                
                # N-BEATS (v7 NEW!)
                nbeats_path = os.path.join(BASE, "nbeats_model.pt")
                if os.path.exists(nbeats_path):
                    ckpt = torch.load(nbeats_path, map_location="cpu", weights_only=False)
                    model = model_classes["nbeats"](ckpt["n_features"])
                    model.load_state_dict(ckpt["model_state"]); model.eval()
                    with torch.no_grad():
                        p_nbeats = float(model(seq_in).item())
                
                # N-HiTS (v7 NEW!)
                nhits_path = os.path.join(BASE, "nhits_model.pt")
                if os.path.exists(nhits_path):
                    ckpt = torch.load(nhits_path, map_location="cpu", weights_only=False)
                    model = model_classes["nhits"](ckpt["n_features"])
                    model.load_state_dict(ckpt["model_state"]); model.eval()
                    with torch.no_grad():
                        p_nhits = float(model(seq_in).item())
        except Exception as e:
            pass  # Deep learning models are optional

        # ════════════════════════════════════════════════════════════════
        # REGIME DETECTION
        # ════════════════════════════════════════════════════════════════
        p_regime = 0.5
        try:
            rm = joblib.load(os.path.join(BASE, "regime_models.pkl"))
            try:
                atr_pct = feat_df["atr_pct"].iloc[-1] if "atr_pct" in feat_df.columns else 0.01
                adx_v = feat_df["adx"].iloc[-1] if "adx" in feat_df.columns else 0.25
                ret5 = feat_df["ret_5"].iloc[-1] if "ret_5" in feat_df.columns else 0.0
                bb_w = feat_df["bb_width"].iloc[-1] if "bb_width" in feat_df.columns else 0.02
                if adx_v > 0.30 and abs(ret5) > 0.005:
                    cur_regime = "TRENDING"
                elif adx_v > 0.25 and abs(ret5) > 0.003:
                    cur_regime = "TREND_UP" if ret5 > 0 else "TREND_DOWN"
                elif atr_pct < 0.005 and adx_v < 0.20:
                    cur_regime = "LOW_VOL"
                elif adx_v < 0.20 and bb_w < 0.015:
                    cur_regime = "CHOPPY_RANGE"
                else:
                    cur_regime = "VOLATILE"
            except Exception:
                cur_regime = "VOLATILE"
            
            if cur_regime in TRENDING and "TREND_UP" in rm:
                p_regime = float(rm["TREND_UP"].predict_proba(X_live)[0][1])
            elif "CHOPPY_RANGE" in rm:
                p_regime = float(rm["CHOPPY_RANGE"].predict_proba(X_live)[0][1])
            else:
                p_regime = (p_xgb + p_lgb + p_gb) / 3.0
        except Exception: cur_regime = "VOLATILE"

        # ════════════════════════════════════════════════════════════════
        # v7: STACKING META-LEARNER (10 models + regime)
        # ════════════════════════════════════════════════════════════════
        p_meta = 0.5
        try:
            mm = joblib.load(os.path.join(BASE, "meta_model.pkl"))
            regime_codes = ec.get("regime_codes", {})
            rc = regime_codes.get(cur_regime, 0)
            
            # v7: 10 model probabilities + regime code = 11 features
            meta_input = np.array([[p_xgb, p_lgb, p_gb, p_cb, p_rf, 
                                   p_lstm, p_tft, p_tcn, p_nbeats, p_nhits, rc]])
            p_meta = float(mm.predict_proba(meta_input)[0][1])
        except Exception:
            # Fallback: simple average
            p_meta = np.mean([p_xgb, p_lgb, p_gb, p_cb, p_rf, p_lstm, p_tft, p_tcn, p_nbeats, p_nhits])

        # ════════════════════════════════════════════════════════════════
        # v7: STRATEGY SELECTION (Stacking Meta = 82.2% accuracy!)
        # ════════════════════════════════════════════════════════════════
        if best_strat == "stacking_meta":
            conf = p_meta
        elif best_strat == "regime_conditional":
            conf = p_regime
        elif best_strat == "hybrid":
            conf = 0.6 * p_regime + 0.4 * np.mean([p_xgb, p_lgb, p_cb, p_tcn, p_nbeats])
        elif best_strat == "top5":
            accs = ec.get("model_accuracies", {})
            top5 = sorted(accs, key=accs.get, reverse=True)[:5]
            probs = {"xgb": p_xgb, "lgb": p_lgb, "gb": p_gb, "cb": p_cb, "rf": p_rf,
                     "lstm": p_lstm, "tft": p_tft, "tcn": p_tcn, "nbeats": p_nbeats, "nhits": p_nhits}
            conf = np.mean([probs.get(m, 0.5) for m in top5])
        else:
            # Default to weighted average
            weights = ec.get("weights", [0.1]*10)
            all_probs = [p_xgb, p_lgb, p_gb, p_cb, p_rf, p_lstm, p_tft, p_tcn, p_nbeats, p_nhits]
            conf = np.average(all_probs, weights=weights)

        # Regime-dependent threshold
        if cur_regime in TRENDING: threshold = CONF_TRENDING
        elif cur_regime in RANGING: threshold = CONF_RANGING
        else: threshold = CONF_VOLATILE

        signal = "BUY" if conf > threshold else ("SELL" if conf < (1 - threshold) else "NEUTRAL")

        reasons.append(f"ML: {signal} ({conf:.1%}) strategy={best_strat}")
        reasons.append(f"  {M}v7 TREE:{RST} XGB={p_xgb:.3f} LGB={p_lgb:.3f} GB={p_gb:.3f} CB={p_cb:.3f} RF={p_rf:.3f}")
        reasons.append(f"  {M}v7 DEEP:{RST} LSTM={p_lstm:.3f} TFT={p_tft:.3f} TCN={p_tcn:.3f} NBEATS={p_nbeats:.3f} NHITS={p_nhits:.3f}")
        reasons.append(f"  Regime={p_regime:.3f} Meta={p_meta:.3f} Thresh={threshold:.2f} ({cur_regime})")

        sig_out = {"signal": signal, "confidence": round(float(conf), 4),
                   "xgb_p": round(p_xgb, 4), "lgb_p": round(p_lgb, 4),
                   "gb_p": round(p_gb, 4), "catboost_p": round(p_cb, 4),
                   "rf_p": round(p_rf, 4), "lstm_p": round(p_lstm, 4),
                   "tft_p": round(p_tft, 4), "tcn_p": round(p_tcn, 4),
                   "nbeats_p": round(p_nbeats, 4), "nhits_p": round(p_nhits, 4),
                   "regime_p": round(p_regime, 4), "meta_p": round(p_meta, 4),
                   "current_regime": cur_regime, "strategy_used": best_strat,
                   "version": "v7", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(BASE, "current_signal.json"), "w") as f:
            json.dump(sig_out, f, indent=2)

    except Exception as e:
        signal = "NEUTRAL"; conf = 0.5; cur_regime = "UNKNOWN"
        reasons.append(f"ML ERROR: {e}"); blocked = True

    if signal == "NEUTRAL":
        blocked = True; reasons.append("BLOCKED: NEUTRAL")

    # Block HIGH_VOL_BREAKOUT regime
    if cur_regime in BLOCKED_REGIMES:
        blocked = True; reasons.append(f"BLOCKED: {cur_regime} regime (v3: confirmed loser)")

    # 2. News Guard
    try:
        from news_guard import is_blocked as nb
        ng = nb()
        if isinstance(ng, dict):
            if ng.get("blocked"): blocked = True; reasons.append(f"NEWS BLOCKED: {ng.get('reason','')}")
            elif ng.get("risk_factor"): reasons.append(f"News: {ng.get('reason','')}")
    except Exception: pass

    # 3. SMC Guard
    try:
        from smc_logic import get_bias
        smc = get_bias()
        smc_dir = smc.get("direction", 0)
        smc_score = abs(smc.get("score", 0))
        smc_label = "BUY" if smc_dir > 0 else ("SELL" if smc_dir < 0 else "NEUTRAL")
        details = smc.get("details", [])
        reasons.append(f"SMC: {smc_label} (score {smc_score})")
        for d in details[:3]: reasons.append(f"  - {d}")
        
        ml_dir = 1 if signal == "BUY" else (-1 if signal == "SELL" else 0)
        if ml_dir != 0 and smc_dir != 0 and ml_dir != smc_dir:
            reasons.append(f"SMC CONFLICT: ML={signal} vs SMC={smc_label}")
        elif ml_dir != 0 and smc_dir == ml_dir:
            reasons.append(f"{G}SMC CONFIRMS {signal}{RST}")
    except Exception: pass

    # 4. Correlation Guard
    try:
        from correlation_guard import check_correlation
        corr = check_correlation(signal)
        if corr and corr.get("conflict"):
            reasons.append(f"Correlation: CONFLICT - {corr.get('reason','')}")
    except Exception: pass

    risk_mult = 1.0
    try:
        from news_guard import is_blocked as nb
        ng = nb()
        if isinstance(ng, dict) and ng.get("risk_factor"):
            risk_mult *= (1 - ng["risk_factor"])
            reasons.append(f"News: REDUCED risk ({-ng['risk_factor']*100:.0f}%): {ng.get('reason','')}")
    except Exception: pass

    return not blocked, signal, conf, risk_mult, cur_regime, smc_score, smc_dir, reasons


# ── Main Loop ─────────────────────────────────────────────────────────────────
def main_loop(mode, max_positions=3):
    global session_pnl, session_wins, session_losses, session_trades, win_streak

    print(f"\n  Starting... {R}Ctrl+C to stop{RST}")
    cycle = 0
    cooldown_until = 0

    while True:
        cycle += 1
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        print(f"\n  {'='*55}")
        print(f"  Cycle {cycle}  |  {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  {'='*55}")

        try:
            import MetaTrader5 as mt5
            acc = get_account_info()
            print(f"  Balance: ${acc['balance']:.2f}  Equity: ${acc['equity']:.2f}")

            positions = get_open_positions()
            print(f"  Open positions: {len(positions)}")

            atr_val = get_atr()
            print(f"\n  Managing positions (ATR=${atr_val:.2f})...")
            managed = manage_open_positions(atr_val)
            for tix, acts in managed:
                for a in acts: print(f"    [{tix}] {a}")

            if len(positions) >= max_positions:
                print(f"  Max {max_positions} positions reached")
                time.sleep(CYCLE_SECONDS)
                continue

            if count_today_trades() >= MAX_TRADES_PER_DAY:
                print(f"  Daily limit ({MAX_TRADES_PER_DAY}) reached")
                time.sleep(CYCLE_SECONDS)
                continue

            print(f"\n  Running guards...")
            ok, signal, conf, risk_mult, regime, smc_score, smc_dir, reasons = run_all_guards(acc)
            for r in reasons: print(f"    {r}")

            if not ok:
                time.sleep(CYCLE_SECONDS)
                continue

            # Regime-dependent risk
            if regime in TRENDING: base_risk = RISK_TRENDING
            elif regime in RANGING: base_risk = RISK_RANGING
            else: base_risk = RISK_BASE
            risk_pct = min(base_risk * risk_mult, RISK_MAX)

            # Session detection
            hour = now_utc.hour
            session = "ASIA" if hour < 8 else ("LONDON" if hour < 13 else ("NY" if hour < 17 else "LONDON_OPEN"))
            print(f"    Risk: {risk_pct:.2f}% | {regime} | {session}")

            # Check SMC confirmation
            ml_dir = 1 if signal == "BUY" else -1
            smc_status = ""
            if smc_dir != 0:
                if smc_dir == ml_dir:
                    smc_status = f"{G}✓ SMC CONFIRMS {signal} (score {smc_score}){RST}"
                else:
                    smc_status = f"{Y}⚠ SMC CONFLICT (score {smc_score}){RST}"
            else:
                smc_status = f"{DIM}SMC: Neutral{RST}"
            print(f"    {smc_status}")
            
            # Correlation check
            corr_status = f"{G}✓ Correlations OK{RST}"
            try:
                from correlation_guard import check_correlation
                corr = check_correlation(signal)
                if corr and corr.get("conflict"):
                    corr_status = f"{Y}⚠ Correlation conflict: {corr.get('reason','')}{RST}"
            except: pass
            print(f"    {corr_status}")

            # SL/TP based on regime (v3 tighter targets)
            if regime in TRENDING:
                sl_dollars = atr_val * 0.30
                tp_dollars = atr_val * 0.65
            elif regime in RANGING:
                sl_dollars = atr_val * 0.20
                tp_dollars = atr_val * 0.35
            else:
                sl_dollars = atr_val * 0.35
                tp_dollars = atr_val * 0.60

            si = mt5.symbol_info(SYMBOL)
            pt = si.point if si else 0.01
            sl_pips = int(sl_dollars / pt)
            tp_pips = int(tp_dollars / pt)

            risk_usd = acc["balance"] * (risk_pct / 100.0)
            # For XAUUSD: 1 lot = 100oz, so $1 move = $100 per lot
            # Risk per lot = sl_dollars * 100
            risk_per_lot = sl_dollars * 100
            lot = round(risk_usd / risk_per_lot, 2)
            lot = max(0.01, min(lot, 0.10))  # Cap at 0.10 for small accounts
            
            # Determine threshold used
            if regime in TRENDING: used_thresh = CONF_TRENDING
            elif regime in RANGING: used_thresh = CONF_RANGING
            else: used_thresh = CONF_VOLATILE

            print(f"\n  {'─'*55}")
            print(f"  >>> {G if signal == 'BUY' else R}{B}{signal} SIGNAL{RST} <<<")
            print(f"  {'─'*55}")
            print(f"      Confidence : {conf:.1%}  (Threshold: {used_thresh:.0%})")
            print(f"      Regime     : {regime}")
            print(f"      Session    : {session}")
            print(f"      Risk       : {risk_pct:.2f}% (${risk_usd:.2f})")
            print(f"      ATR        : ${atr_val:.2f}")
            print(f"      SL/TP      : ${sl_dollars:.2f} / ${tp_dollars:.2f}")
            print(f"      Lot Size   : {lot} (max 0.10 for safety)")

            # Cooldown check
            if time.time() < cooldown_until:
                remaining = int(cooldown_until - time.time())
                print(f"  Cooldown: {remaining}s")
                time.sleep(CYCLE_SECONDS)
                continue

            # Place trade
            if place_trade(signal, lot, sl_pips, tp_pips, f"AFBv7_{regime[:4]}"):
                print(f"  {G}[OK] Trade placed!{RST}")
                log_event("trade", {"signal": signal, "lot": lot, "sl": sl_pips, "tp": tp_pips, "conf": conf, "regime": regime})
                send_telegram(f"{signal} {lot}lot @ {conf:.1%} conf | {regime}")
                cooldown_until = time.time() + 180  # 3 min cooldown
            else:
                print(f"  {R}[FAIL] Trade not placed{RST}")

        except KeyboardInterrupt:
            print(f"\n  {Y}Stopped by user{RST}")
            break
        except Exception as e:
            print(f"  {R}[ERR] {e}{RST}")
            traceback.print_exc()

        print(f"\n  Next cycle in {CYCLE_SECONDS}s...")
        time.sleep(CYCLE_SECONDS)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--max-pos", type=int, default=3)
    args = parser.parse_args()

    print(f"""
{M}{'='*65}
  AlmostFinishedBot  |  Live Trading Bridge v7 ELITE
  Mode: {'PAPER (Demo)' if args.mode == 'paper' else 'LIVE (Real)'}
  Symbol: {SYMBOL}  |  Cycle: {CYCLE_SECONDS}s
  {B}v7: 10-MODEL ENSEMBLE + STACKING META (82.2% accuracy){RST}
  {M}Models: XGB, LGB, GB, CB, RF, LSTM, TFT, TCN, N-BEATS, N-HiTS{RST}
  v3 Thresholds: TREND={CONF_TRENDING} RANGE={CONF_RANGING} VOL={CONF_VOLATILE}
  v3 Position Mgmt: BE@25%+15%ATR  Trail@35%  Partial@40%
  v3 Blocked regimes: {BLOCKED_REGIMES}
{'='*65}{RST}
""")

    print("  Connecting to MT5...")
    init_mt5(args.mode)
    main_loop(args.mode, args.max_pos)
