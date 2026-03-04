"""
AlmostFinishedBot - Walk-Forward Backtest v3
2 YEARS | LSTM + TFT | ALL GUARDS | GBP 200

v3 changes:
  - Added PyTorch LSTM + TFT per walk-forward window
  - 2 year backtest period (fetches max MT5 data)
  - 7 model ensemble: XGB, LGB, GB, CB, RF, LSTM, TFT
  - All live bridge guards replicated
"""
import os, sys, warnings, json, time, datetime, logging

# ---- NUCLEAR WARNING SUPPRESSION ----
# Python's warnings.filterwarnings doesn't propagate to joblib child processes.
# The only reliable fix is to monkey-patch warnings.warn itself.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["JOBLIB_VERBOSITY"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

# Monkey-patch: completely disable warnings.warn
_original_warn = warnings.warn
def _noop_warn(*args, **kwargs): pass
warnings.warn = _noop_warn

# Also patch at C level for sklearn's internal usage
import warnings as _w
_w.warn = _noop_warn
_w.filterwarnings("ignore")
# Suppress sklearn parallel warnings
# (already monkey-patched above, these are belt-and-suspenders)

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
sys.path.insert(0, BASE)

def P(msg=""):
    """Print with flush so output is visible immediately."""
    print(msg, flush=True)

P("=" * 65)
P("  AlmostFinishedBot  |  Walk-Forward Backtest v3")
P("  2 YEARS | 7 Models (LSTM+TFT) | ALL GUARDS | GBP 200")
P("=" * 65)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier; HAS_CB = True
except: HAS_CB = False

HAS_TORCH = False; DEVICE = "cpu"
try:
    import torch; import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.backends.cudnn.benchmark = True  # optimize for fixed input sizes
        P(f"  PyTorch: GPU ({torch.cuda.get_device_name(0)})")
        P(f"  CUDA version: {torch.version.cuda}")
        P(f"  GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        P("  PyTorch: CPU (CUDA not available!)")
        P("  Check: pip install torch --index-url https://download.pytorch.org/whl/cu121")
except Exception as e:
    P(f"  PyTorch: not available ({e})")

# Re-apply monkey-patch after all imports (some imports restore warnings.warn)
warnings.warn = _noop_warn

from features import make_features, make_target
from market_regime import detect_regime, regime_risk_multiplier

# ===================================================================
# CONFIG
# ===================================================================
BALANCE_START   = 200.0
CONF_TRENDING   = 0.54; CONF_VOLATILE = 0.57; CONF_RANGING = 0.62
RISK_BASE=1.5; RISK_TRENDING=2.0; RISK_RANGING=0.5; RISK_HIGH_CONF=2.5; RISK_MAX=3.0
MAX_POS=3; MAX_TRADES_DAY=10; COOLDOWN_BARS=3
BE_PCT=0.40; TRAIL_PCT=0.55; PARTIAL_PCT=0.50; TRAIL_ATR_MULT=1.0
HORIZON=5; THRESHOLD=0.001; PURGE=7; MIN_ACC=0.52
TRAIN_BARS = 60 * 24    # 60 days training window for 2yr backtest
TEST_BARS  = 10 * 24    # 10 days test window
BACKTEST_DAYS = 730      # 2 years
SEQ_LEN = 20             # LSTM/TFT sequence length

SPREAD={"XAUUSD":0.30,"XAUSGD":0.50}
PIP_VAL={"XAUUSD":100,"XAUSGD":75}
SESSION_MULT={"ASIAN":0.8,"LONDON_OPEN":1.0,"LONDON_NY":1.0,"NY_CLOSE":0.8,"OFF_HOURS":0.3}
TRENDING_SET={"STRONG_TREND_UP","STRONG_TREND_DOWN","TREND_UP","TREND_DOWN","TRENDING_STRONG","TRENDING"}
RANGING_SET={"CHOPPY_RANGE","RANGING","LOW_VOL","MEAN_REVERT","LOW_VOL_COMPRESSION"}

# ===================================================================
# PyTorch Models
# ===================================================================
class LSTMModel(nn.Module):
    def __init__(s, nf, h=64, ly=2, d=0.3):
        super().__init__()
        s.lstm=nn.LSTM(nf,h,ly,batch_first=True,dropout=d)
        s.bn=nn.BatchNorm1d(h); s.fc1=nn.Linear(h,32); s.drop=nn.Dropout(d); s.fc2=nn.Linear(32,1)
    def forward(s, x):
        o,_=s.lstm(x); o=o[:,-1,:]; o=s.bn(o); o=torch.relu(s.fc1(o)); o=s.drop(o)
        return torch.sigmoid(s.fc2(o)).squeeze(-1)

class SimpleTFT(nn.Module):
    def __init__(s, nf, dm=64, nh=4, sl=20, d=0.3):
        super().__init__()
        s.ip=nn.Linear(nf,dm); s.pe=nn.Parameter(torch.randn(1,sl,dm)*0.02)
        el=nn.TransformerEncoderLayer(dm,nh,dm*2,d,batch_first=True)
        s.tr=nn.TransformerEncoder(el,num_layers=2); s.bn=nn.BatchNorm1d(dm)
        s.fc1=nn.Linear(dm,32); s.drop=nn.Dropout(d); s.fc2=nn.Linear(32,1)
    def forward(s, x):
        x=s.ip(x)+s.pe; x=s.tr(x); x=x[:,-1,:]; x=s.bn(x); x=torch.relu(s.fc1(x))
        x=s.drop(x); return torch.sigmoid(s.fc2(x)).squeeze(-1)

def train_pytorch_model(ModelClass, X_tr, y_tr, X_val, y_val, n_features, epochs=30, **kwargs):
    """Train a PyTorch model, return (model, val_accuracy, val_probs_aligned)."""
    def make_seqs(X, y, s):
        xs, ys = [], []
        for i in range(s, len(X)):
            xs.append(X[i-s:i]); ys.append(y[i])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    Xs, ys = make_seqs(X_tr, y_tr, SEQ_LEN)
    Xvs, yvs = make_seqs(X_val, y_val, SEQ_LEN)
    if len(Xs) < 32 or len(Xvs) < 16:
        return None, 0.5, np.full(len(y_val), 0.5)

    train_dl = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)), batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(Xvs), torch.tensor(yvs)), batch_size=128)

    model = ModelClass(n_features, **kwargs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001 if "LSTM" in ModelClass.__name__ else 0.0005)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.BCELoss()
    bvl = float("inf"); pat = 6; w = 0; bs = None

    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            p = model(xb); lo = loss_fn(p, yb)
            opt.zero_grad(); lo.backward(); opt.step()
        model.eval(); vls = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vls.append(loss_fn(model(xb), yb).item())
        vl = np.mean(vls); sch.step(vl)
        if vl < bvl: bvl = vl; w = 0; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            w += 1
            if w >= pat: break

    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
    acc = accuracy_score(yvs, (raw > 0.5).astype(int))
    # Align to full val length
    aligned = np.full(len(y_val), 0.5)
    aligned[-len(raw):] = raw
    return model, acc, aligned

# ===================================================================
# DATA
# ===================================================================
def fetch_mt5(symbol, bars=20000):
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize(): return None
        si = mt5.symbol_info(symbol)
        if si and not si.visible: mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
        mt5.shutdown()
        if rates is not None and len(rates) > 200:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.set_index("time")
            if "tick_volume" in df.columns: df["volume"] = df["tick_volume"]
            return df
    except Exception as e: P(f"  MT5 {symbol}: {e}")
    return None

P("\n  Fetching max data (2+ years)...")
df_usd = fetch_mt5("XAUUSD", 20000)
df_sgd = fetch_mt5("XAUSGD", 20000)
if df_usd is None:
    try:
        from features import download_gold
        df_usd = download_gold("2y", "1h"); P("  XAUUSD: yfinance fallback")
    except: pass

for sym, df in [("XAUUSD", df_usd), ("XAUSGD", df_sgd)]:
    if df is not None:
        days = (df.index[-1] - df.index[0]).days
        P(f"  {sym}: {len(df)} candles ({days} days / {days/365:.1f} years)")
    else: P(f"  {sym}: NO DATA")

# ===================================================================
# FEATURES + REGIMES
# ===================================================================
def get_session_from_hour(hour):
    if 0 <= hour < 7: return "ASIAN"
    elif 7 <= hour < 12: return "LONDON_OPEN"
    elif 12 <= hour < 16: return "LONDON_NY"
    elif 16 <= hour < 21: return "NY_CLOSE"
    else: return "OFF_HOURS"

def is_weekend_bar(dt):
    if hasattr(dt, 'weekday'):
        wd = dt.weekday()
        if wd == 5: return True
        if wd == 6 and (not hasattr(dt, 'hour') or dt.hour < 22): return True
    return False

def build_all(df, symbol):
    feat = make_features(df)
    feat["target"] = make_target(df["close"].values, HORIZON, THRESHOLD)
    regimes, adx_arr, bbw_arr = detect_regime(df)[:3]
    feat["_regime_str"] = regimes[:len(feat)]
    if hasattr(feat.index, 'hour'): feat["_hour"] = feat.index.hour
    else: feat["_hour"] = 12
    feat["_session"] = feat["_hour"].apply(get_session_from_hour)
    feat["_is_weekend"] = [is_weekend_bar(dt) for dt in feat.index]
    close = df["close"].values[:len(feat)]
    high = df["high"].values[:len(feat)]
    low = df["low"].values[:len(feat)]
    tr = np.maximum(high[1:]-low[1:], np.maximum(np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
    atr_arr = np.full(len(feat), np.nan)
    for i in range(14, len(tr)): atr_arr[i+1] = np.mean(tr[max(0,i-13):i+1])
    feat["_atr_dollar"] = atr_arr; feat["_close"] = close
    feat["_high"] = high[:len(feat)]; feat["_low"] = low[:len(feat)]
    atr_pct_col = feat.get("atr_pct", pd.Series(np.zeros(len(feat))))
    if isinstance(atr_pct_col, pd.Series):
        feat["_vol_spike"] = (atr_pct_col > atr_pct_col.rolling(50).mean() * 2.0).astype(int)
    else: feat["_vol_spike"] = 0
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat

P("\n  Building features + regimes...")
feat_usd = build_all(df_usd, "XAUUSD") if df_usd is not None else None
feat_sgd = build_all(df_sgd, "XAUSGD") if df_sgd is not None else None
for sym, f in [("XAUUSD", feat_usd), ("XAUSGD", feat_sgd)]:
    if f is not None:
        rc = f["_regime_str"].value_counts()
        P(f"  {sym}: {len(f)} rows | " + " ".join(f"{r}:{c}" for r,c in rc.head(5).items()))

# ===================================================================
# TRAIN 7-MODEL ENSEMBLE
# ===================================================================
META_COLS = ["target","_regime_str","_adx_raw","_bbw_raw","_hour",
             "_session","_is_weekend","_atr_dollar","_close","_high","_low","_vol_spike"]

def get_fcols(feat):
    return [c for c in feat.columns if c not in META_COLS]

def train_window(feat, tr_s, tr_e, fcols):
    X = feat.iloc[tr_s:tr_e][fcols].values.astype(float)
    y = feat.iloc[tr_s:tr_e]["target"].values.astype(int)
    vs = int(len(X)*0.8)
    if vs < 100 or len(X)-vs < 20: return None
    Xtr, ytr = X[:vs], y[:vs]; Xvl, yvl = X[vs:], y[vs:]

    sc = StandardScaler(); Xts = sc.fit_transform(Xtr); Xvs = sc.transform(Xvl)
    mods = {}; probs = {}; accs = {}

    # Tree models
    m = xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,subsample=0.7,
        colsample_bytree=0.7,eval_metric="logloss",verbosity=0,random_state=42)
    m.fit(Xts,ytr,eval_set=[(Xvs,yvl)],verbose=False)
    mods["xgb"]=m; probs["xgb"]=m.predict_proba(Xvs)[:,1]

    m = lgb.LGBMClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,subsample=0.7,
        colsample_bytree=0.7,verbose=-1,random_state=42)
    m.fit(Xts,ytr,eval_set=[(Xvs,yvl)],callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(-1)])
    mods["lgb"]=m; probs["lgb"]=m.predict_proba(Xvs)[:,1]

    m = GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.7,random_state=42)
    m.fit(Xts,ytr); mods["gb"]=m; probs["gb"]=m.predict_proba(Xvs)[:,1]

    if HAS_CB:
        m = CatBoostClassifier(iterations=300,depth=4,learning_rate=0.03,verbose=0,random_seed=42)
        m.fit(Xts,ytr,eval_set=(Xvs,yvl),verbose=0,early_stopping_rounds=20)
        mods["cb"]=m; probs["cb"]=m.predict_proba(Xvs)[:,1]

    m = RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=15,
        max_features="sqrt",random_state=42,n_jobs=4)
    m.fit(Xts,ytr); mods["rf"]=m; probs["rf"]=m.predict_proba(Xvs)[:,1]

    # LSTM
    if HAS_TORCH:
        try:
            lstm_m, lstm_acc, lstm_p = train_pytorch_model(
                LSTMModel, Xts, ytr, Xvs, yvl, Xts.shape[1], epochs=25)
            if lstm_m is not None:
                mods["lstm"] = lstm_m; probs["lstm"] = lstm_p
                mods["_lstm_scaler"] = sc
        except Exception as e:
            P(f"      LSTM failed: {e}")

        # TFT
        try:
            tft_m, tft_acc, tft_p = train_pytorch_model(
                SimpleTFT, Xts, ytr, Xvs, yvl, Xts.shape[1], epochs=25, sl=SEQ_LEN)
            if tft_m is not None:
                mods["tft"] = tft_m; probs["tft"] = tft_p
        except Exception as e:
            P(f"      TFT failed: {e}")

    accs = {k: accuracy_score(yvl, (p>0.5).astype(int)) for k,p in probs.items()
            if k not in ("_lstm_scaler",)}
    edges = {k: max((v-0.5)**2,0) for k,v in accs.items() if v >= MIN_ACC}
    te = sum(edges.values())
    wts = {k: v/te for k,v in edges.items()} if te > 0 else {k: 1/len(accs) for k in accs}
    return {"models": mods, "scaler": sc, "weights": wts, "accs": accs}

def predict_ens(ens, X_row, X_seq=None):
    """Predict with ensemble. X_row for trees, X_seq for LSTM/TFT."""
    Xs = ens["scaler"].transform(X_row.reshape(1,-1))
    ps = {}
    for k, m in ens["models"].items():
        if k.startswith("_"): continue
        if k in ("lstm", "tft"):
            if X_seq is not None and HAS_TORCH:
                try:
                    m.eval()
                    with torch.no_grad():
                        t = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        ps[k] = float(m(t).cpu().numpy().item())
                except: ps[k] = 0.5
            else: ps[k] = 0.5
        else:
            try: ps[k] = float(m.predict_proba(Xs)[0][1])
            except: ps[k] = 0.5

    conf = sum(ens["weights"].get(k,0)*ps.get(k,0.5) for k in ens["weights"])
    tw = sum(ens["weights"].values())
    return conf/tw if tw > 0 else 0.5, ps

# ===================================================================
# POSITION SIMULATOR
# ===================================================================
class Pos:
    def __init__(s, sym, dir, entry, lot, sl, tp, bar, atr):
        s.sym=sym; s.dir=dir; s.entry=entry; s.lot=lot
        s.sl=sl; s.tp=tp; s.bar=bar; s.atr=atr
        s.closed=False; s.exit_p=0; s.profit=0; s.reason=""
        s.be=False; s.partial=False; s.trailing=False
    def tick(s, h, l, c, bar):
        if s.closed: return
        tp_dist = (s.tp-s.entry) if s.dir=="BUY" else (s.entry-s.tp)
        profit_pts = (c-s.entry) if s.dir=="BUY" else (s.entry-c)
        tp_dist = max(tp_dist, s.atr)
        progress = profit_pts / tp_dist if tp_dist > 0 else 0
        if progress >= BE_PCT and not s.be:
            s.be = True; s.sl = s.entry + (0.05 if s.dir=="BUY" else -0.05)
        if progress >= PARTIAL_PCT and not s.partial and s.lot >= 0.02:
            s.partial = True; half = round(s.lot/2, 2); pv = PIP_VAL[s.sym]
            s.profit += profit_pts * half * pv; s.lot = max(0.01, round(s.lot-half, 2))
        if progress >= TRAIL_PCT:
            s.trailing = True; td = s.atr * TRAIL_ATR_MULT
            if s.dir=="BUY":
                ns = c - td
                if ns > s.sl: s.sl = ns
            else:
                ns = c + td
                if ns < s.sl: s.sl = ns
        hit_tp = (h >= s.tp) if s.dir=="BUY" else (l <= s.tp)
        hit_sl = (l <= s.sl) if s.dir=="BUY" else (h >= s.sl)
        if hit_tp: s.closed=True; s.exit_p=s.tp; s.reason="TP"
        elif hit_sl: s.closed=True; s.exit_p=s.sl; s.reason="BE" if s.be else "SL"
        if s.closed:
            pv = PIP_VAL[s.sym]
            s.profit += ((s.exit_p-s.entry) if s.dir=="BUY" else (s.entry-s.exit_p)) * s.lot * pv

# ===================================================================
# WALK-FORWARD WITH ALL GUARDS + LSTM/TFT
# ===================================================================
def walkforward(df_price, feat, symbol):
    fcols = get_fcols(feat)
    total = len(feat)
    bt_bars = min(BACKTEST_DAYS*24, total - TRAIN_BARS - PURGE)
    if bt_bars < TEST_BARS: P(f"  {symbol}: not enough data"); return None
    start = total - bt_bars
    nw = bt_bars // TEST_BARS
    P(f"\n  {symbol}: {nw} windows, {bt_bars} bars ({bt_bars//24}d)")

    entries = []; waccs = []; w_times = []
    for w in range(nw):
        w_t0 = time.time()
        ts = start + w*TEST_BARS; te = ts + TEST_BARS
        tr_e = ts - PURGE; tr_s = max(0, tr_e - TRAIN_BARS)
        if tr_e <= tr_s + 50 or te > total: continue

        ens = train_window(feat, tr_s, tr_e, fcols)
        if ens is None:
            P(f"    W{w+1}/{nw} SKIPPED (train failed)")
            continue
        ba = max(ens["accs"].values()); waccs.append(ba)
        n_models = len([k for k in ens["accs"]])
        acc_s = " ".join(f"{k}:{v*100:.0f}" for k,v in ens["accs"].items())
        w_elapsed = time.time() - w_t0; w_times.append(w_elapsed)
        avg_w = np.mean(w_times); eta = avg_w * (nw - w - 1)
        P(f"    W{w+1}/{nw} [{acc_s}] best={ba*100:.1f}% ({n_models}m) {w_elapsed:.0f}s [ETA {eta/60:.0f}m]")

        # Pre-scale all test features for LSTM/TFT sequence building
        X_all_scaled = ens["scaler"].transform(feat.iloc[max(0,ts-SEQ_LEN):te][fcols].values.astype(float))

        for i in range(ts, min(te, total)):
            row = feat.iloc[i]
            if row.get("_is_weekend", False): continue
            session = row.get("_session", "LONDON_OPEN")
            if session == "OFF_HOURS": continue
            sess_mult = SESSION_MULT.get(session, 0.5)

            X = row[fcols].values.astype(float)

            # Build sequence for LSTM/TFT
            X_seq = None
            if HAS_TORCH:
                seq_idx = i - max(0, ts - SEQ_LEN)  # Index into X_all_scaled
                if seq_idx >= SEQ_LEN:
                    X_seq = X_all_scaled[seq_idx-SEQ_LEN:seq_idx].astype(np.float32)

            conf, ps = predict_ens(ens, X, X_seq)

            regime_str = row.get("_regime_str", "UNKNOWN")
            if regime_str in TRENDING_SET: threshold = CONF_TRENDING
            elif regime_str in RANGING_SET: threshold = CONF_RANGING
            else: threshold = CONF_VOLATILE

            if conf > threshold: sig = "BUY"
            elif conf < (1 - threshold): sig = "SELL"
            else: continue

            vol_spike = row.get("_vol_spike", 0)
            news_mult = 0.7 if vol_spike else 1.0
            regime_rmult = regime_risk_multiplier(regime_str)
            if regime_rmult <= 0.1: continue

            if conf > 0.80: conf_mult = 1.5
            elif conf > 0.70: conf_mult = 1.2
            elif conf > 0.60: conf_mult = 1.0
            else: conf_mult = 0.5

            if regime_str in TRENDING_SET: base_risk = RISK_TRENDING
            elif regime_str in RANGING_SET: base_risk = RISK_RANGING
            else: base_risk = RISK_BASE

            risk_pct = min(base_risk * conf_mult * regime_rmult * news_mult * sess_mult, RISK_MAX)
            if risk_pct <= 0: continue

            atr = row.get("_atr_dollar", 25.0)
            if np.isnan(atr) or atr <= 0: atr = 25.0
            sp = SPREAD[symbol]

            if regime_str in TRENDING_SET: sl_d=atr*0.35; tp_d=atr*0.90
            elif regime_str in RANGING_SET: sl_d=atr*0.25; tp_d=atr*0.50
            else: sl_d=atr*0.40; tp_d=atr*0.80
            sl_d=max(sl_d,15.0); tp_d=max(tp_d,20.0)

            close_p = row.get("_close", 0)
            if close_p <= 0: continue
            if sig == "BUY": entry=close_p+sp/2; sl=entry-sl_d; tp=entry+tp_d
            else: entry=close_p-sp/2; sl=entry+sl_d; tp=entry-tp_d

            sl_pips = sl_d/0.01 if symbol=="XAUUSD" else sl_d/0.001
            weak_conf = (conf < 0.57 and conf >= 0.50)

            entries.append({"bar":i,"sym":symbol,"dir":sig,"entry":entry,
                "sl":sl,"tp":tp,"sl_d":sl_d,"tp_d":tp_d,"atr":atr,
                "conf":round(conf,4),"regime":regime_str,"risk_pct":round(risk_pct,3),
                "sl_pips":sl_pips,"weak_conf":weak_conf,"news_reduced":bool(vol_spike),
                "session":session,"time":str(feat.index[i]) if i<len(feat) else ""})

    return {"entries": entries, "waccs": waccs, "symbol": symbol}

# ===================================================================
# SIMULATE WITH ALL GUARDS
# ===================================================================
def simulate(results_list, dfs):
    bal = BALANCE_START; eq_curve = [bal]; open_p = []; closed = []
    peak = bal; mdd = 0
    all_e = []
    for r in results_list:
        if r is None: continue
        for e in r["entries"]: all_e.append(e)
    all_e.sort(key=lambda x: x["bar"])
    if not all_e: return {"eq":eq_curve,"trades":[],"bal":bal,"mdd":0,"peak":bal}

    mn = all_e[0]["bar"]
    mx = min(max(e["bar"] for e in all_e) + 10*24, max(len(df) for df in dfs.values()))
    ei = 0; recent_trades = []; last_trade_bar = {}; trades_today = 0; current_day = -1

    for bar in range(mn, mx):
        for p in open_p:
            if p.closed: continue
            df = dfs.get(p.sym)
            if df is None or bar >= len(df): continue
            p.tick(float(df.iloc[bar]["high"]), float(df.iloc[bar]["low"]),
                   float(df.iloc[bar]["close"]), bar)
            if p.closed:
                bal += p.profit
                closed.append({"sym":p.sym,"dir":p.dir,"entry":round(p.entry,2),
                    "exit":round(p.exit_p,2),"profit":round(p.profit,2),
                    "reason":p.reason,"bars":bar-p.bar,"lot":p.lot,
                    "regime":getattr(p,'_regime',''),"time":getattr(p,'_time','')})
                recent_trades.append((p.dir, p.profit > 0))
                recent_trades = recent_trades[-10:]
        open_p = [p for p in open_p if not p.closed]

        for sym, df in dfs.items():
            if bar < len(df) and hasattr(df.index[bar], 'day'):
                d = df.index[bar].day
                if d != current_day: current_day = d; trades_today = 0
                break

        while ei < len(all_e) and all_e[ei]["bar"] == bar:
            e = all_e[ei]; ei += 1; sym = e["sym"]
            sym_open = sum(1 for p in open_p if p.sym == sym)
            if sym_open >= MAX_POS: continue
            if trades_today >= MAX_TRADES_DAY: continue
            if bar - last_trade_bar.get(sym, -999) < COOLDOWN_BARS: continue
            dir_losses = 0
            for rd, rw in reversed(recent_trades):
                if rd == e["dir"] and not rw: dir_losses += 1
                else: break
            if dir_losses >= 2: continue
            if bal <= 10: continue

            risk_amount = bal * (e["risk_pct"]/100); sl_pips = e["sl_pips"]
            if sl_pips <= 0: continue
            lot = risk_amount / (sl_pips * 1.0)
            if e["weak_conf"]: lot *= 0.5
            if e["news_reduced"]: lot *= 0.7
            lot = max(0.01, min(round(lot, 2), 1.0))
            actual_risk = lot * e["sl_d"] * PIP_VAL[sym] / 100
            if actual_risk > bal * 0.05:
                lot = max(0.01, round(bal * 0.05 / (e["sl_d"] * PIP_VAL[sym] / 100), 2))

            pos = Pos(sym, e["dir"], e["entry"], lot, e["sl"], e["tp"], bar, e["atr"])
            pos._regime = e["regime"]; pos._time = e["time"]
            open_p.append(pos); last_trade_bar[sym] = bar; trades_today += 1

        unr = 0
        for p in open_p:
            if p.closed: continue
            df = dfs.get(p.sym)
            if df is None or bar >= len(df): continue
            c = float(df.iloc[bar]["close"]); pv = PIP_VAL[p.sym]
            unr += ((c-p.entry) if p.dir=="BUY" else (p.entry-c)) * p.lot * pv
        equity = bal + unr; eq_curve.append(equity)
        if equity > peak: peak = equity
        dd = (peak-equity)/peak if peak > 0 else 0
        if dd > mdd: mdd = dd

    for p in open_p:
        if p.closed: continue
        df = dfs.get(p.sym)
        if df is None: continue
        c = float(df.iloc[-1]["close"]); pv = PIP_VAL[p.sym]
        p.profit += ((c-p.entry) if p.dir=="BUY" else (p.entry-c)) * p.lot * pv
        bal += p.profit
        closed.append({"sym":p.sym,"dir":p.dir,"entry":round(p.entry,2),
            "exit":round(c,2),"profit":round(p.profit,2),"reason":"FORCE",
            "bars":0,"lot":p.lot,"regime":"","time":""})

    return {"eq":eq_curve,"trades":closed,"bal":round(bal,2),"mdd":round(mdd*100,2),"peak":round(peak,2)}

# ===================================================================
# EXECUTE
# ===================================================================
P(f"\n  Running walk-forward (2yr, 7 models, all guards)...")
P(f"  This will take a while with LSTM+TFT...\n")
t0 = time.time()
r_usd = walkforward(df_usd, feat_usd, "XAUUSD") if feat_usd is not None else None
r_sgd = walkforward(df_sgd, feat_sgd, "XAUSGD") if feat_sgd is not None else None
elapsed = time.time() - t0
P(f"\n  Walk-forward computed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

for r, sym in [(r_usd,"XAUUSD"),(r_sgd,"XAUSGD")]:
    if r: P(f"  {sym}: {len(r['entries'])} signals")

dfs = {}
if df_usd is not None: dfs["XAUUSD"] = df_usd
if df_sgd is not None: dfs["XAUSGD"] = df_sgd

P("\n  Simulating trades...")
sim = simulate([r_usd, r_sgd], dfs)

# ===================================================================
# REPORT
# ===================================================================
pnl = sim["bal"] - BALANCE_START; pct = pnl/BALANCE_START*100
actual_days = 0
if sim["trades"]:
    first_t = [t for t in sim["trades"] if t.get("time")]
    if first_t: actual_days = BACKTEST_DAYS

P(f"\n  {'='*65}")
P(f"  WALK-FORWARD RESULTS  |  ~{BACKTEST_DAYS//365}yr  |  7 Models  |  All Guards")
P(f"  {'='*65}")
P(f"  Starting Balance:  GBP {BALANCE_START:.2f}")
P(f"  Final Balance:     GBP {sim['bal']:.2f}")
P(f"  P&L:               GBP {pnl:+.2f} ({pct:+.1f}%)")
P(f"  Peak Equity:       GBP {sim['peak']:.2f}")
P(f"  Max Drawdown:      {sim['mdd']:.1f}%")
P(f"  Total Trades:      {len(sim['trades'])}")

if sim["trades"]:
    wins=[t for t in sim["trades"] if t["profit"]>0]
    losses=[t for t in sim["trades"] if t["profit"]<0]
    bes=[t for t in sim["trades"] if abs(t["profit"])<0.10]
    wr=len(wins)/len(sim["trades"])*100
    aw=np.mean([t["profit"] for t in wins]) if wins else 0
    al=np.mean([t["profit"] for t in losses]) if losses else 0
    tp_sum=sum(t["profit"] for t in wins); sl_sum=abs(sum(t["profit"] for t in losses))
    pf=tp_sum/sl_sum if sl_sum>0 else float("inf")
    sharpe = 0
    if len(sim["trades"]) > 1:
        rets = [t["profit"] for t in sim["trades"]]
        if np.std(rets) > 0: sharpe = np.mean(rets)/np.std(rets)*np.sqrt(252/max(1,len(rets)))

    P(f"  Win Rate:          {wr:.1f}%  ({len(wins)}W / {len(losses)}L / {len(bes)}BE)")
    P(f"  Avg Win:           GBP {aw:.2f}")
    P(f"  Avg Loss:          GBP {al:.2f}")
    P(f"  Profit Factor:     {pf:.2f}")
    P(f"  Sharpe (approx):   {sharpe:.2f}")
    if any(t["bars"]>0 for t in sim["trades"]):
        avg_bars = np.mean([t["bars"] for t in sim["trades"] if t["bars"]>0])
        P(f"  Avg Hold:          {avg_bars:.0f} bars ({avg_bars/24:.1f} days)")

    # Monthly breakdown
    monthly = {}
    for t in sim["trades"]:
        tm = t.get("time","")[:7]
        if tm:
            if tm not in monthly: monthly[tm] = {"pnl":0,"n":0,"w":0}
            monthly[tm]["pnl"] += t["profit"]; monthly[tm]["n"] += 1
            if t["profit"] > 0: monthly[tm]["w"] += 1
    if monthly:
        P(f"\n  Monthly Breakdown:")
        P(f"  {'Month':>8s} {'Trades':>7s} {'WR':>6s} {'P&L':>10s}")
        for m in sorted(monthly.keys()):
            d = monthly[m]
            mwr = d["w"]/d["n"]*100 if d["n"]>0 else 0
            P(f"  {m:>8s} {d['n']:>7d} {mwr:>5.0f}% {d['pnl']:>+10.2f}")

    # Per-symbol
    for sym in ["XAUUSD","XAUSGD"]:
        st=[t for t in sim["trades"] if t["sym"]==sym]
        if st:
            sp=sum(t["profit"] for t in st); sw=sum(1 for t in st if t["profit"]>0)
            sl_c=sum(1 for t in st if t["profit"]<0)
            swr=sw/len(st)*100 if st else 0
            P(f"\n  --- {sym}: {len(st)} trades | {sw}W/{sl_c}L ({swr:.0f}%) | GBP {sp:+.2f} ---")
            regimes = {}
            for t in st:
                r = t.get("regime","?")
                if r not in regimes: regimes[r] = {"n":0,"pnl":0}
                regimes[r]["n"] += 1; regimes[r]["pnl"] += t["profit"]
            for r, d in sorted(regimes.items(), key=lambda x: x[1]["pnl"], reverse=True):
                P(f"      {r:25s}: {d['n']:3d} trades  GBP {d['pnl']:+.2f}")

    P(f"\n  Close Reasons:")
    rs={}
    for t in sim["trades"]: r=t["reason"]; rs[r]=rs.get(r,0)+1
    for r,c in sorted(rs.items()): P(f"    {r:10s}: {c}")

    # Window accuracies
    for r, sym in [(r_usd,"XAUUSD"),(r_sgd,"XAUSGD")]:
        if r and r["waccs"]:
            P(f"\n  {sym} avg OOS accuracy: {np.mean(r['waccs'])*100:.1f}% over {len(r['waccs'])} windows")

    # Last 20 trades
    P(f"\n  Last 20 trades:")
    P(f"  {'Sym':7s} {'Dir':5s} {'Entry':>10s} {'Exit':>10s} {'P&L':>8s} {'Rsn':>5s} {'Bars':>4s} {'Regime':>12s}")
    for t in sim["trades"][-20:]:
        P(f"  {t['sym']:7s} {t['dir']:5s} {t['entry']:10.2f} {t['exit']:10.2f} {t['profit']:+8.2f} {t['reason']:>5s} {t['bars']:4d} {t.get('regime',''):>12s}")

# Save
out = {"start_bal":BALANCE_START,"final_bal":sim["bal"],"pnl":round(pnl,2),
       "pnl_pct":round(pct,2),"max_dd":sim["mdd"],"n_trades":len(sim["trades"]),
       "trades":sim["trades"],"equity":[round(e,2) for e in sim["eq"]],
       "models":"XGB+LGB+GB+CB+RF+LSTM+TFT",
       "guards":"ALL (regime,session,antichop,cooldown,news_proxy,risk_scaling)",
       "backtest_days":BACKTEST_DAYS,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
rp = os.path.join(BASE,"walkforward_results.json")
with open(rp,"w") as f: json.dump(out,f,indent=2)
P(f"\n  Saved: {rp}")
P(f"  Backtest complete! ({elapsed/60:.1f} minutes)")
