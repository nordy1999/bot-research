"""
AlmostFinishedBot - XAUSGD Live Trading Bridge v7
ELITE 10-MODEL ENSEMBLE with Stacking Meta

v7 changes:
  - 10 models: XGB, LGB, GB, CB, RF, LSTM, TFT, TCN, N-BEATS, N-HiTS
  - Uses xausgd_ prefixed models
  - All v3 position management optimisations retained
"""
import os, sys, json, time, datetime, warnings, argparse
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
sys.path.insert(0, BASE)
import ctypes
try:
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except Exception: pass
os.system("")

G="\033[92m";R="\033[91m";Y="\033[93m";C="\033[96m";B="\033[1m";M="\033[95m";DIM="\033[2m";RST="\033[0m"
SYMBOL="XAUSGD";MODEL_PREFIX="xausgd_";MAGIC_NUMBER=1000;CYCLE_SECONDS=60
MAX_POSITIONS=3;MAX_TRADES_PER_DAY=10;VIRTUAL_BALANCE=100.0
SEQ = 20

# v3: Confidence thresholds (optimised)
CONF_TRENDING=0.54
CONF_RANGING=0.62
CONF_VOLATILE=0.53

TRENDING={"TRENDING","TREND_UP","TREND_DOWN","STRONG_TREND"}
RANGING={"CHOPPY_RANGE","RANGING","LOW_VOL","MEAN_REVERT"}
RISK_BASE=1.5;RISK_TRENDING=2.0;RISK_RANGING=0.5

# v3: Position management
BREAKEVEN_PCT=0.25
BE_OFFSET_ATR=0.15
PARTIAL_TRIGGER=0.40
PARTIAL_SIZE=0.40
TRAIL_ACTIVATE_PCT=0.35
TRAIL_ATR_MULT=0.8

BLOCKED_REGIMES = {"HIGH_VOL_BREAKOUT"}

session_pnl=0.0;session_wins=0;session_losses=0;session_trades=0
active_trades={};position_state={}

def get_session(hour):
    if 0<=hour<2: return "ASIA_EARLY",0.9
    if 2<=hour<5: return "ASIA_PEAK",1.3
    if 5<=hour<8: return "ASIA_LONDON",1.5
    if 8<=hour<12: return "LONDON",1.1
    if 12<=hour<16: return "LONDON_NY",1.0
    if 16<=hour<20: return "NY",0.7
    return "OFF_HOURS",0.5

def init_mt5():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print(f"  {R}[ERR] MT5 init failed{RST}"); sys.exit(1)
    info = mt5.account_info()
    if info is None:
        print(f"  {R}[ERR] No account{RST}"); mt5.shutdown(); sys.exit(1)
    print(f"  {G}[OK] MT5: {info.login} ({info.server}){RST}")
    print(f"       Balance: ${info.balance:.2f}  Equity: ${info.equity:.2f}")
    print(f"       Sizing from virtual ${VIRTUAL_BALANCE}")
    si = mt5.symbol_info(SYMBOL)
    if si:
        print(f"       {SYMBOL}: point={si.point} min_lot={si.volume_min} digits={si.digits}")
    return info

def get_account_info():
    import MetaTrader5 as mt5
    info = mt5.account_info()
    return {"balance": info.balance, "equity": info.equity, "margin": info.margin,
            "free_margin": info.margin_free, "profit": info.profit}

def get_open_positions():
    import MetaTrader5 as mt5
    return [p for p in (mt5.positions_get(symbol=SYMBOL) or []) if p.magic == MAGIC_NUMBER]

def get_atr(period=14):
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, period + 5)
    if rates is None or len(rates) < period: return 40.0
    import pandas as pd
    df = pd.DataFrame(rates)
    df["h_l"] = df["high"] - df["low"]
    df["h_pc"] = abs(df["high"] - df["close"].shift())
    df["l_pc"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["h_l", "h_pc", "l_pc"]].max(axis=1)
    return df["tr"].rolling(period).mean().iloc[-1]

def place_trade(direction, lot_size, sl_pips, tp_pips, comment="AFBv7_SGD"):
    try:
        import MetaTrader5 as mt5
        si = mt5.symbol_info(SYMBOL)
        if si is None or not si.visible:
            if not mt5.symbol_select(SYMBOL, True):
                print(f"  [ERR] {SYMBOL} not available"); return False
            si = mt5.symbol_info(SYMBOL)
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
            color = G if direction == "BUY" else R
            print(f"  {color}{B}[TRADE] {direction} {lot_size} {SYMBOL} @ {ep:.3f}{RST}")
            return True
        else:
            print(f"  [ERR] {result.retcode}: {result.comment}")
            if result.retcode == 10019:
                print(f"  Retrying minimum lot {si.volume_min}...")
                req["volume"] = si.volume_min
                r2 = mt5.order_send(req)
                if r2 and r2.retcode == mt5.TRADE_RETCODE_DONE:
                    c = G if direction == "BUY" else R
                    print(f"  {c}{B}[TRADE] {direction} {si.volume_min} {SYMBOL} @ {ep:.3f}{RST}")
                    return True
                print(f"  Still failed: {r2.retcode if r2 else 'None'}")
            return False
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

def manage_positions(atr_val):
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

        # Breakeven
        if profit_pct >= BREAKEVEN_PCT and not st["breakeven"]:
            be_offset = max(atr_val * BE_OFFSET_ATR, 0.50)
            if is_buy:
                new_sl = entry + be_offset
                if new_sl > cur_sl and modify_sl(ticket, new_sl):
                    st["breakeven"] = True; actions.append(f"{G}BE SL->{new_sl:.3f}{RST}")
            else:
                new_sl = entry - be_offset
                if (cur_sl == 0 or new_sl < cur_sl) and modify_sl(ticket, new_sl):
                    st["breakeven"] = True; actions.append(f"{G}BE SL->{new_sl:.3f}{RST}")

        # Partial close
        if profit_pct >= PARTIAL_TRIGGER and not st["partial"] and vol >= 0.02:
            pv = max(0.01, round(vol * PARTIAL_SIZE, 2))
            if close_partial(ticket, pv):
                st["partial"] = True; actions.append(f"{Y}PARTIAL {pv}lots{RST}")

        # Trailing stop
        if profit_pct >= TRAIL_ACTIVATE_PCT:
            st["trailing"] = True
            trail = atr_val * TRAIL_ATR_MULT
            if is_buy:
                new_sl = price - trail
                if new_sl > cur_sl + 0.5 and modify_sl(ticket, new_sl):
                    actions.append(f"{C}TRAIL SL->{new_sl:.3f}{RST}")
            else:
                new_sl = price + trail
                if (cur_sl == 0 or new_sl < cur_sl - 0.5) and modify_sl(ticket, new_sl):
                    actions.append(f"{C}TRAIL SL->{new_sl:.3f}{RST}")

        if actions: managed.append((ticket, actions))

    # Cleanup
    open_tix = {p.ticket for p in positions}
    for t in [t for t in position_state if t not in open_tix]:
        del position_state[t]
    return managed


# ── v7 Deep Learning Model Classes ────────────────────────────────────────────
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


# ── v7 ML Pipeline (10 models) ────────────────────────────────────────────
def run_ml():
    import MetaTrader5 as mt5
    import joblib
    import torch
    
    try:
        # Get XAUSGD data from MT5
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 200)
        if rates is None or len(rates) < 50:
            return "NEUTRAL", 0.5, "VOLATILE", []
        
        import pandas as pd
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time": "datetime"})
        df.set_index("datetime", inplace=True)
        
        from features import make_features
        feat_df = make_features(df)
        if feat_df is None or len(feat_df) < 20:
            return "NEUTRAL", 0.5, "VOLATILE", []
        
        # Load ensemble config
        ec = {}
        ec_path = os.path.join(BASE, f"{MODEL_PREFIX}ensemble_config.json")
        if os.path.exists(ec_path):
            with open(ec_path) as f: ec = json.load(f)
        best_strat = ec.get("best_strategy", "top3")
        
        scaler_path = os.path.join(BASE, f"{MODEL_PREFIX}scaler.pkl")
        if not os.path.exists(scaler_path):
            return "NEUTRAL", 0.5, "VOLATILE", []
        scaler = joblib.load(scaler_path)
        X_live = scaler.transform(feat_df.iloc[[-1]])
        X_all = scaler.transform(feat_df)
        
        # ════════════════════════════════════════════════════════════════
        # TREE MODELS (5)
        # ════════════════════════════════════════════════════════════════
        def load_model(name):
            path = os.path.join(BASE, f"{MODEL_PREFIX}{name}.pkl")
            if os.path.exists(path):
                return float(joblib.load(path).predict_proba(X_live)[0][1])
            return 0.5
        
        p_xgb = load_model("xgb_model")
        p_lgb = load_model("lgb_model")
        p_gb = load_model("gb_model")
        p_cb = load_model("catboost_model")
        p_rf = load_model("rf_model")
        
        # ════════════════════════════════════════════════════════════════
        # DEEP LEARNING MODELS (5) - v7
        # ════════════════════════════════════════════════════════════════
        p_lstm = p_tft = p_tcn = p_nbeats = p_nhits = 0.5
        
        try:
            model_classes = get_model_classes()
            seq_len = ec.get("seq_len", SEQ)
            
            if len(X_all) >= seq_len:
                seq_in = torch.tensor(X_all[-seq_len:].reshape(1, seq_len, -1), dtype=torch.float32)
                
                def load_pt_model(name, model_class):
                    path = os.path.join(BASE, f"{MODEL_PREFIX}{name}_model.pt")
                    if os.path.exists(path):
                        ckpt = torch.load(path, map_location="cpu", weights_only=False)
                        model = model_class(ckpt["n_features"])
                        model.load_state_dict(ckpt["model_state"]); model.eval()
                        with torch.no_grad():
                            return float(model(seq_in).item())
                    return 0.5
                
                p_lstm = load_pt_model("lstm", model_classes["lstm"])
                p_tft = load_pt_model("tft", model_classes["tft"])
                p_tcn = load_pt_model("tcn", model_classes["tcn"])
                p_nbeats = load_pt_model("nbeats", model_classes["nbeats"])
                p_nhits = load_pt_model("nhits", model_classes["nhits"])
        except Exception:
            pass
        
        # ════════════════════════════════════════════════════════════════
        # REGIME DETECTION
        # ════════════════════════════════════════════════════════════════
        cur_regime = "VOLATILE"
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
        except: pass

        # ════════════════════════════════════════════════════════════════
        # META-LEARNER
        # ════════════════════════════════════════════════════════════════
        p_meta = 0.5
        try:
            mm_path = os.path.join(BASE, f"{MODEL_PREFIX}meta_model.pkl")
            if os.path.exists(mm_path):
                mm = joblib.load(mm_path)
                regime_codes = ec.get("regime_codes", {})
                rc = regime_codes.get(cur_regime, 0)
                # v7: 10 model probs + regime
                meta_input = np.array([[p_xgb, p_lgb, p_gb, p_cb, p_rf, 
                                       p_lstm, p_tft, p_tcn, p_nbeats, p_nhits, rc]])
                p_meta = float(mm.predict_proba(meta_input)[0][1])
        except:
            p_meta = np.mean([p_xgb, p_lgb, p_gb, p_cb, p_rf, p_lstm, p_tft, p_tcn, p_nbeats, p_nhits])

        # ════════════════════════════════════════════════════════════════
        # STRATEGY SELECTION
        # ════════════════════════════════════════════════════════════════
        if best_strat == "stacking_meta":
            conf = p_meta
        elif best_strat == "top3":
            accs = ec.get("model_accuracies", {})
            top3 = sorted(accs, key=accs.get, reverse=True)[:3]
            probs = {"xgb": p_xgb, "lgb": p_lgb, "gb": p_gb, "cb": p_cb, "rf": p_rf,
                     "lstm": p_lstm, "tft": p_tft, "tcn": p_tcn, "nbeats": p_nbeats, "nhits": p_nhits}
            conf = np.mean([probs.get(m, 0.5) for m in top3])
        elif best_strat == "top5":
            accs = ec.get("model_accuracies", {})
            top5 = sorted(accs, key=accs.get, reverse=True)[:5]
            probs = {"xgb": p_xgb, "lgb": p_lgb, "gb": p_gb, "cb": p_cb, "rf": p_rf,
                     "lstm": p_lstm, "tft": p_tft, "tcn": p_tcn, "nbeats": p_nbeats, "nhits": p_nhits}
            conf = np.mean([probs.get(m, 0.5) for m in top5])
        else:
            # Default weighted
            all_probs = [p_xgb, p_lgb, p_gb, p_cb, p_rf, p_lstm, p_tft, p_tcn, p_nbeats, p_nhits]
            conf = np.mean(all_probs)

        # Threshold
        if cur_regime in TRENDING: threshold = CONF_TRENDING
        elif cur_regime in RANGING: threshold = CONF_RANGING
        else: threshold = CONF_VOLATILE

        signal = "BUY" if conf > threshold else ("SELL" if conf < (1 - threshold) else "NEUTRAL")
        
        reasons = [
            f"ML: {signal} ({conf:.1%}) regime={cur_regime} thresh={threshold:.2f} strat={best_strat}",
            f"xgb={p_xgb:.3f} | lgb={p_lgb:.3f} | gb={p_gb:.3f} | cb={p_cb:.3f} | rf={p_rf:.3f}",
            f"lstm={p_lstm:.3f} | tft={p_tft:.3f} | tcn={p_tcn:.3f} | nbeats={p_nbeats:.3f} | nhits={p_nhits:.3f}",
            f"regime={cur_regime} | meta={p_meta:.3f}"
        ]
        return signal, conf, cur_regime, reasons
        
    except Exception as e:
        return "NEUTRAL", 0.5, "VOLATILE", [f"ML ERROR: {e}"]


# ── Main Loop ─────────────────────────────────────────────────────────────────
def main_loop(mode):
    print(f"\n  Starting... {R}Ctrl+C to stop{RST}")
    cycle = 0
    cooldown_until = 0
    
    while True:
        cycle += 1
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        session_name, session_mult = get_session(now_utc.hour)
        
        print(f"\n  {'='*55}")
        print(f"  Cycle {cycle}  |  {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC  |  {session_name} (x{session_mult})")
        print(f"  {'='*55}")
        
        try:
            import MetaTrader5 as mt5
            acc = get_account_info()
            positions = get_open_positions()
            unrealised = sum(p.profit for p in positions)
            print(f"  Balance: ${acc['balance']:.2f}  Equity: ${acc['equity']:.2f}  (sizing: ${VIRTUAL_BALANCE})")
            print(f"  Open {SYMBOL}: {len(positions)}  Unrealised: ${unrealised:+.2f}")
            
            atr_val = get_atr()
            print(f"\n  Managing positions (ATR=${atr_val:.2f})...")
            managed = manage_positions(atr_val)
            for tix, acts in managed:
                for a in acts: print(f"    [{tix}] {a}")
            
            if len(positions) >= MAX_POSITIONS:
                print(f"  Max {MAX_POSITIONS} positions reached")
                time.sleep(CYCLE_SECONDS)
                continue
            
            print(f"\n  Running ML...")
            signal, conf, regime, reasons = run_ml()
            for r in reasons: print(f"    {r}")
            
            if signal == "NEUTRAL":
                print(f"  Signal: NEUTRAL, no trade")
                time.sleep(CYCLE_SECONDS)
                continue
            
            if regime in BLOCKED_REGIMES:
                print(f"  {R}BLOCKED: {regime} regime{RST}")
                time.sleep(CYCLE_SECONDS)
                continue
            
            # Risk calculation from virtual balance
            if regime in TRENDING: base_risk = RISK_TRENDING
            elif regime in RANGING: base_risk = RISK_RANGING
            else: base_risk = RISK_BASE
            risk_pct = base_risk * session_mult
            
            # SL/TP
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
            pt = si.point if si else 0.001
            sl_pips = int(sl_dollars / pt)
            tp_pips = int(tp_dollars / pt)
            
            risk_usd = VIRTUAL_BALANCE * (risk_pct / 100.0)
            # For XAUSGD: 1 lot = 100oz, similar to XAUUSD
            risk_per_lot = sl_dollars * 100
            lot = round(risk_usd / risk_per_lot, 2)
            lot = max(0.01, min(lot, 0.10))  # Cap at 0.10 for safety
            
            # Determine threshold used
            if regime in TRENDING: used_thresh = CONF_TRENDING
            elif regime in RANGING: used_thresh = CONF_RANGING
            else: used_thresh = CONF_VOLATILE
            
            print(f"\n  {'─'*50}")
            print(f"  >>> {G if signal == 'BUY' else R}{B}{signal} SIGNAL{RST} <<<")
            print(f"  {'─'*50}")
            print(f"      Confidence : {conf:.1%} (Threshold: {used_thresh:.0%})")
            print(f"      Regime     : {regime}")
            print(f"      Session    : {session_name} (x{session_mult})")
            print(f"      Risk       : {risk_pct:.2f}% (${risk_usd:.2f})")
            print(f"      ATR        : ${atr_val:.2f}")
            print(f"      SL/TP      : ${sl_dollars:.2f} / ${tp_dollars:.2f}")
            print(f"      Lot Size   : {lot}")
            
            # Cooldown
            if time.time() < cooldown_until:
                remaining = int(cooldown_until - time.time())
                print(f"  Cooldown: {remaining}s")
                time.sleep(CYCLE_SECONDS)
                continue
            
            # Place trade
            if place_trade(signal, lot, sl_pips, tp_pips, f"AFBv7_{regime[:4]}"):
                print(f"  {G}[OK] Trade placed!{RST}")
                cooldown_until = time.time() + 180
            else:
                print(f"  {R}[FAIL] Trade not placed{RST}")
            
        except KeyboardInterrupt:
            print(f"\n  {Y}Stopped by user{RST}")
            break
        except Exception as e:
            print(f"  {R}[ERR] {e}{RST}")
            import traceback
            traceback.print_exc()
        
        print(f"\n  Next cycle in {CYCLE_SECONDS}s...")
        time.sleep(CYCLE_SECONDS)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = parser.parse_args()
    
    print(f"""
{M}{'='*65}
  AlmostFinishedBot  |  XAUSGD Bridge v7 ELITE
  Mode: {'PAPER (Demo)' if args.mode == 'paper' else 'LIVE (Real)'}
  Symbol: {SYMBOL}  |  Cycle: {CYCLE_SECONDS}s  |  Magic: {MAGIC_NUMBER}
  Virtual Balance: ${VIRTUAL_BALANCE} (sized for ${VIRTUAL_BALANCE})
  {B}v7: 10-MODEL ENSEMBLE{RST}
  {M}Models: XGB, LGB, GB, CB, RF, LSTM, TFT, TCN, N-BEATS, N-HiTS{RST}
  v3 Thresholds: TREND={CONF_TRENDING} RANGE={CONF_RANGING} VOL={CONF_VOLATILE}
  Peak: ASIA_PEAK(2-5UTC x1.3) ASIA_LONDON(5-8UTC x1.5)
  v3 Position Mgmt: BE@25%+15%ATR  Trail@35%  Partial@40%
  v3 Blocked regimes: {BLOCKED_REGIMES}
{'='*65}{RST}
""")
    
    print("  Connecting to MT5...")
    init_mt5()
    main_loop(args.mode)
