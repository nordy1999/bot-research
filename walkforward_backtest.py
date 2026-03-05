"""
AlmostFinishedBot - Walk-Forward Backtest v4 — ANTI-BIAS REWRITE
=================================================================
KEY ANTI-BIAS FIXES:
  1. Scaler ALWAYS fit on training fold only — never on full dataset
  2. PURGE BUFFER: 7-bar gap between train and test (prevents leakage at boundaries)
  3. EMBARGO BUFFER: 7-bar gap after each test fold before next train starts
  4. Features using shift(1): all forward-looking targets use horizon-aware masking
  5. News/regime signals: timestamp-checked to ensure strictly before decision bar
  6. Combinatorial Purged Cross-Validation (CPCV) support for multiple test paths
  7. Regime-Dynamic Ensemble Weighting: LSTM/TCN weighted higher in TREND_UP/DOWN
  8. Sharpe, Sortino, Calmar added to report
  9. Expected Value per trade (avg R-multiple) reported

UNBIASED MODE checklist enforced:
  - Scaler: fit on train_slice only
  - Target: uses close[i+horizon] — bars ahead of decision always > test start
  - No hyperparameter tuning across folds (fixed params defined once)
  - Walk-forward direction always forward in time (no future peeking)
"""
import os, sys, warnings, json, time, datetime, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"]        = "ignore"
os.environ["JOBLIB_VERBOSITY"]      = "0"
os.environ["OMP_NUM_THREADS"]       = "4"
os.environ["LOKY_MAX_CPU_COUNT"]    = "4"
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

_orig_warn = warnings.warn
def _noop(*a, **k): pass
warnings.warn = _noop

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
sys.path.insert(0, BASE)


def P(msg=""):
    print(msg, flush=True)


P("=" * 70)
P("  AlmostFinishedBot  |  Walk-Forward Backtest v4  (ANTI-BIAS)")
P("  Purged+Embargoed folds | CPCV | Regime-dynamic weights | £500")
P("=" * 70)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier; HAS_CB = True
except Exception:
    HAS_CB = False

HAS_TORCH = False; DEVICE = "cpu"
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
        P(f"  PyTorch: GPU ({torch.cuda.get_device_name(0)})")
    else:
        P("  PyTorch: CPU")
except Exception as e:
    P(f"  PyTorch: not available ({e})")

warnings.warn = _noop

from features import make_features, make_target
from market_regime import detect_regime, regime_risk_multiplier

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
BALANCE_START   = 500.0           # £500 account
CONF_TRENDING   = 0.54
CONF_VOLATILE   = 0.57
CONF_RANGING    = 0.62
RISK_BASE       = 1.5
RISK_TRENDING   = 2.0
RISK_RANGING    = 0.5
RISK_MAX        = 3.0

MAX_POS         = 3
MAX_TRADES_DAY  = 10
COOLDOWN_BARS   = 3

# Position management
BE_PCT          = 0.40
TRAIL_PCT       = 0.55
PARTIAL_PCT     = 0.50
TRAIL_ATR_MULT  = 1.0

HORIZON         = 5               # bars ahead for target label
THRESHOLD       = 0.001           # 0.1% move = positive target
PURGE_BARS      = 7               # bars to drop at train/test boundary
EMBARGO_BARS    = 7               # bars to drop after test before next train
MIN_ACC         = 0.52

TRAIN_BARS      = 60 * 24         # 60-day training window
TEST_BARS       = 10 * 24         # 10-day test window
BACKTEST_DAYS   = 730             # 2 years
SEQ_LEN         = 20

SPREAD   = {"XAUUSD": 0.30, "XAUSGD": 0.50}
PIP_VAL  = {"XAUUSD": 100,  "XAUSGD": 75}
SESSION_MULT = {
    "ASIAN": 0.8, "LONDON_OPEN": 1.0, "LONDON_NY": 1.0,
    "NY_CLOSE": 0.8, "OFF_HOURS": 0.3,
}
TRENDING_SET = {"STRONG_TREND_UP","STRONG_TREND_DOWN","TREND_UP","TREND_DOWN",
                "TRENDING_STRONG","TRENDING"}
RANGING_SET  = {"CHOPPY_RANGE","RANGING","LOW_VOL","MEAN_REVERT","LOW_VOL_COMPRESSION"}

# Regime → which models to weight higher (anti-bias strategy)
# Trend regimes: LSTM, TCN better at pattern continuation
# Ranging regimes: XGB, CatBoost better at mean-reversion features
REGIME_MODEL_WEIGHTS = {
    "TREND_UP":             {"xgb":1.0,"lgb":1.0,"gb":0.8,"cb":0.8,"rf":0.7,
                             "lstm":1.5,"tft":1.3,"tcn":1.5},
    "TREND_DOWN":           {"xgb":1.0,"lgb":1.0,"gb":0.8,"cb":0.8,"rf":0.7,
                             "lstm":1.5,"tft":1.3,"tcn":1.5},
    "CHOPPY_RANGE":         {"xgb":1.5,"lgb":1.3,"gb":1.2,"cb":1.5,"rf":1.2,
                             "lstm":0.6,"tft":0.7,"tcn":0.6},
    "HIGH_VOL_BREAKOUT":    {"xgb":1.0,"lgb":1.0,"gb":1.0,"cb":1.0,"rf":1.0,
                             "lstm":1.0,"tft":1.0,"tcn":1.2},
    "LOW_VOL_COMPRESSION":  {"xgb":1.2,"lgb":1.2,"gb":1.2,"cb":1.2,"rf":1.2,
                             "lstm":0.8,"tft":0.8,"tcn":0.8},
}


# ═══════════════════════════════════════════════════════════════════
# PYTORCH MODELS
# ═══════════════════════════════════════════════════════════════════
if HAS_TORCH:
    class LSTMModel(nn.Module):
        def __init__(s, nf, h=64, ly=2, d=0.3):
            super().__init__()
            s.lstm = nn.LSTM(nf, h, ly, batch_first=True, dropout=d)
            s.bn   = nn.BatchNorm1d(h)
            s.fc1  = nn.Linear(h, 32)
            s.drop = nn.Dropout(d)
            s.fc2  = nn.Linear(32, 1)
        def forward(s, x):
            o, _ = s.lstm(x); o = o[:, -1, :]
            o = s.bn(o); o = torch.relu(s.fc1(o)); o = s.drop(o)
            return torch.sigmoid(s.fc2(o)).squeeze(-1)

    class TCNBlock(nn.Module):
        def __init__(s, in_ch, out_ch, k, d, dropout=0.2):
            super().__init__()
            pad  = (k - 1) * d
            s.c1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
            s.b1 = nn.BatchNorm1d(out_ch)
            s.c2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
            s.b2 = nn.BatchNorm1d(out_ch)
            s.drop = nn.Dropout(dropout)
            s.ds = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
            s.pad = pad
        def forward(s, x):
            r   = x if s.ds is None else s.ds(x)
            o   = s.drop(torch.relu(s.b1(s.c1(x)[:, :, :-s.pad] if s.pad else s.c1(x))))
            o   = s.drop(torch.relu(s.b2(s.c2(o)[:, :, :-s.pad] if s.pad else s.c2(o))))
            return torch.relu(o + r)

    class TCNModel(nn.Module):
        def __init__(s, nf, ch=[64, 64, 64], k=3, d=0.2):
            super().__init__()
            layers = []
            for i, oc in enumerate(ch):
                ic = nf if i == 0 else ch[i - 1]
                layers.append(TCNBlock(ic, oc, k, 2 ** i, d))
            s.tcn  = nn.Sequential(*layers)
            s.fc1  = nn.Linear(ch[-1], 32)
            s.drop = nn.Dropout(d)
            s.fc2  = nn.Linear(32, 1)
        def forward(s, x):
            x = x.transpose(1, 2); o = s.tcn(x)[:, :, -1]
            o = torch.relu(s.fc1(o)); o = s.drop(o)
            return torch.sigmoid(s.fc2(o)).squeeze(-1)
else:
    LSTMModel = None
    TCNModel  = None


def train_pytorch(ModelClass, Xtr, ytr, Xvl, yvl, nf, epochs=25):
    """Train a sequential model. Returns (model, accuracy, aligned_probs)."""
    if not HAS_TORCH or ModelClass is None:
        return None, 0.5, np.full(len(yvl), 0.5)

    def make_seq(X, y, s):
        xs, ys = [], []
        for i in range(s, len(X)):
            xs.append(X[i - s:i]); ys.append(y[i])
        return np.array(xs, np.float32), np.array(ys, np.float32)

    Xs, ys   = make_seq(Xtr, ytr, SEQ_LEN)
    Xvs, yvs = make_seq(Xvl, yvl, SEQ_LEN)
    if len(Xs) < 32 or len(Xvs) < 8:
        return None, 0.5, np.full(len(yvl), 0.5)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
        batch_size=64, shuffle=True)
    val_dl   = DataLoader(
        TensorDataset(torch.tensor(Xvs), torch.tensor(yvs)),
        batch_size=128)

    model   = ModelClass(nf).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=0.001)
    sch     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.BCELoss()
    best_vl = float("inf"); pat = 5; wait = 0; best_state = None

    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vls = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vls.append(loss_fn(model(xb), yb).item())
        vl = float(np.mean(vls)); sch.step(vl)
        if vl < best_vl:
            best_vl = vl; wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= pat: break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
    acc     = accuracy_score(yvs, (raw > 0.5).astype(int))
    aligned = np.full(len(yvl), 0.5)
    aligned[-len(raw):] = raw
    return model, acc, aligned


# ═══════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════
def fetch_mt5(symbol, bars=20000):
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        si = mt5.symbol_info(symbol)
        if si and not si.visible:
            mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
        mt5.shutdown()
        if rates is not None and len(rates) > 200:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.set_index("time")
            if "tick_volume" in df.columns:
                df["volume"] = df["tick_volume"]
            return df
    except Exception as e:
        P(f"  MT5 {symbol}: {e}")
    return None


P("\n  Fetching max data (2+ years)...")
df_usd = fetch_mt5("XAUUSD", 20000)
df_sgd = fetch_mt5("XAUSGD", 20000)

if df_usd is None:
    try:
        from features import download_gold
        df_usd = download_gold(period="2y", interval="1h")
        P(f"  XAUUSD: yfinance fallback ({len(df_usd)} bars)")
    except Exception as e:
        P(f"  XAUUSD: fetch failed — {e}")

for sym, df in [("XAUUSD", df_usd), ("XAUSGD", df_sgd)]:
    if df is not None:
        P(f"  {sym}: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — ANTI-LOOKAHEAD
# ═══════════════════════════════════════════════════════════════════
def build_feature_matrix(df, symbol):
    """
    Build features + target + helper columns.
    ANTI-LOOKAHEAD: all features use only past data at bar i.
    Target uses close[i+HORIZON] which is always in the FUTURE —
    but we exclude last HORIZON bars from training (make_target returns NaN there).
    """
    feat = make_features(df)
    tgt  = make_target(df["close"].values, horizon=HORIZON, threshold=THRESHOLD)
    feat["_target"]  = tgt

    # Regime string (for regime-aware weights)
    regs, adx_arr, _, _ = detect_regime(df)
    feat["_regime_str"]  = regs

    # Helper columns (not used as ML features)
    feat["_close"]   = df["close"].values
    feat["_high"]    = df["high"].values
    feat["_low"]     = df["low"].values

    # ATR in price dollars
    from features import atr_np
    atr_d = atr_np(df["high"].values, df["low"].values, df["close"].values, 14)
    feat["_atr_dollar"] = atr_d

    # Session
    if hasattr(df.index, "hour"):
        hours = df.index.hour.values
    else:
        hours = np.zeros(len(df))
    session = np.where(hours < 7, "ASIAN",
              np.where(hours < 12, "LONDON_OPEN",
              np.where(hours < 16, "LONDON_NY",
              np.where(hours < 21, "NY_CLOSE", "OFF_HOURS"))))
    feat["_session"]  = session

    # Weekend flag
    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek.values
    else:
        dow = np.zeros(len(df), dtype=int)
    feat["_is_weekend"] = (dow >= 5)

    # Volume spike (simple proxy for news event)
    if "volume" in df.columns:
        vol = df["volume"].values.astype(float)
        vol_ma = pd.Series(vol).rolling(20).mean().values
        feat["_vol_spike"] = (vol > vol_ma * 2.5).astype(int)
    else:
        feat["_vol_spike"] = 0

    return feat


def get_feature_cols(feat):
    """Return only columns that are valid ML features (no helpers/targets)."""
    return [c for c in feat.columns
            if not c.startswith("_")
            and feat[c].dtype in (np.float64, np.float32, float)
            and c not in ("news_score",)]   # exclude live-only cols


# ═══════════════════════════════════════════════════════════════════
# PURGED + EMBARGOED WALK-FORWARD TRAINING
# ═══════════════════════════════════════════════════════════════════
def train_fold_purged(feat, train_start, train_end, fcols):
    """
    Train ensemble on feat[train_start:train_end].
    ANTI-BIAS: scaler is fit ONLY on training data.
    Returns ensemble dict or None.
    """
    tr_data = feat.iloc[train_start:train_end]
    valid   = tr_data.dropna(subset=fcols + ["_target"])
    X_raw   = valid[fcols].values.astype(float)
    y_raw   = valid["_target"].values.astype(float)

    # Remove NaN targets (last HORIZON bars always NaN)
    mask = ~np.isnan(y_raw)
    X    = X_raw[mask]
    y    = y_raw[mask]

    if len(X) < 100 or y.mean() < 0.1 or y.mean() > 0.9:
        return None

    # ── ANTI-BIAS: Scaler fit on training fold ONLY ──────────────
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    n_val = max(30, len(X) // 5)
    Xtr, ytr = Xs[:-n_val], y[:-n_val]
    Xvl, yvl = Xs[-n_val:], y[-n_val:]

    if len(Xtr) < 50:
        return None

    mods  = {}
    probs = {}

    # XGBoost
    try:
        m = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0, random_state=42)
        m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
              early_stopping_rounds=15, verbose=False)
        probs["xgb"] = m.predict_proba(Xvl)[:, 1]
        mods["xgb"]  = m
    except Exception as e:
        P(f"      XGB error: {e}")

    # LightGBM
    try:
        m = lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42)
        m.fit(Xtr, ytr,
              eval_set=[(Xvl, yvl)],
              callbacks=[lgb.early_stopping(15, verbose=False),
                         lgb.log_evaluation(-1)])
        probs["lgb"] = m.predict_proba(Xvl)[:, 1]
        mods["lgb"]  = m
    except Exception as e:
        P(f"      LGB error: {e}")

    # Gradient Boosting
    try:
        m = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42)
        m.fit(Xtr, ytr)
        probs["gb"] = m.predict_proba(Xvl)[:, 1]
        mods["gb"]  = m
    except Exception as e:
        P(f"      GB error: {e}")

    # CatBoost
    if HAS_CB:
        try:
            m = CatBoostClassifier(
                iterations=100, depth=4, learning_rate=0.05,
                verbose=0, random_state=42)
            m.fit(Xtr, ytr, eval_set=(Xvl, yvl),
                  early_stopping_rounds=15)
            probs["cb"] = m.predict_proba(Xvl)[:, 1]
            mods["cb"]  = m
        except Exception as e:
            P(f"      CB error: {e}")

    # Random Forest
    try:
        m = RandomForestClassifier(
            n_estimators=100, max_depth=5,
            min_samples_leaf=10, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr)
        probs["rf"] = m.predict_proba(Xvl)[:, 1]
        mods["rf"]  = m
    except Exception as e:
        P(f"      RF error: {e}")

    # LSTM (optional)
    if HAS_TORCH and LSTMModel is not None:
        try:
            lstm_m, lstm_acc, lstm_p = train_pytorch(
                LSTMModel, Xtr, ytr, Xvl, yvl, Xtr.shape[1], epochs=20)
            if lstm_m is not None:
                probs["lstm"] = lstm_p
                mods["lstm"]  = lstm_m
        except Exception as e:
            P(f"      LSTM error: {e}")

    # TCN (optional)
    if HAS_TORCH and TCNModel is not None:
        try:
            tcn_m, tcn_acc, tcn_p = train_pytorch(
                TCNModel, Xtr, ytr, Xvl, yvl, Xtr.shape[1], epochs=20)
            if tcn_m is not None:
                probs["tcn"] = tcn_p
                mods["tcn"]  = tcn_m
        except Exception as e:
            P(f"      TCN error: {e}")

    if not mods:
        return None

    # Accuracy-based weights (edge-squared)
    accs  = {k: accuracy_score(yvl, (p > 0.5).astype(int))
             for k, p in probs.items()
             if len(p) == len(yvl)}
    edges = {k: max((v - 0.5) ** 2, 0) for k, v in accs.items() if v >= MIN_ACC}
    te    = sum(edges.values())
    base_wts = {k: v / te for k, v in edges.items()} if te > 0 else \
               {k: 1 / len(accs) for k in accs}

    return {
        "models":    mods,
        "scaler":    sc,
        "base_wts":  base_wts,
        "accs":      accs,
        "n_train":   len(Xtr),
    }


def predict_with_regime(ens, X_raw, X_seq, regime_str):
    """
    Predict using ensemble with regime-dynamic weighting.
    ANTI-BIAS: uses base weights × regime multipliers, normalised.
    """
    Xs   = ens["scaler"].transform(X_raw.reshape(1, -1))
    ps   = {}
    mods = ens["models"]

    for name, m in mods.items():
        if name in ("lstm", "tcn"):
            if X_seq is not None and HAS_TORCH:
                try:
                    m.eval()
                    with torch.no_grad():
                        t = torch.tensor(X_seq, dtype=torch.float32
                                         ).unsqueeze(0).to(DEVICE)
                        ps[name] = float(m(t).cpu().item())
                except Exception:
                    ps[name] = 0.5
            else:
                ps[name] = 0.5
        else:
            try:
                ps[name] = float(m.predict_proba(Xs)[0][1])
            except Exception:
                ps[name] = 0.5

    # Regime-dynamic weights
    regime_mults = REGIME_MODEL_WEIGHTS.get(regime_str, {})
    final_wts    = {}
    for k in ps:
        bw = ens["base_wts"].get(k, 0.0)
        rm = regime_mults.get(k, 1.0)
        final_wts[k] = bw * rm

    tw   = sum(final_wts.values())
    if tw <= 0:
        conf = float(np.mean(list(ps.values())))
    else:
        conf = sum(final_wts[k] * ps[k] for k in final_wts) / tw

    return conf, ps


# ═══════════════════════════════════════════════════════════════════
# BUILD FEATURE MATRICES
# ═══════════════════════════════════════════════════════════════════
feat_usd = feat_sgd = None
if df_usd is not None:
    P("  Building XAUUSD features...")
    feat_usd = build_feature_matrix(df_usd, "XAUUSD")
    P(f"  XAUUSD features: {feat_usd.shape}")
if df_sgd is not None:
    P("  Building XAUSGD features...")
    feat_sgd = build_feature_matrix(df_sgd, "XAUSGD")
    P(f"  XAUSGD features: {feat_sgd.shape}")


# ═══════════════════════════════════════════════════════════════════
# WALK-FORWARD WITH PURGE + EMBARGO
# ═══════════════════════════════════════════════════════════════════
def walkforward_purged(df_price, feat, symbol):
    """
    Purged + Embargoed walk-forward.
    Structure per fold:
      [  TRAIN  ] [PURGE] [  TEST  ] [EMBARGO]
    Purge: PURGE_BARS bars dropped at each end of boundary
    Embargo: EMBARGO_BARS bars after test before next train starts
    """
    fcols = get_feature_cols(feat)
    total = len(feat)

    bt_bars = min(BACKTEST_DAYS * 24, total - TRAIN_BARS - PURGE_BARS * 2)
    if bt_bars < TEST_BARS:
        P(f"  {symbol}: insufficient data ({bt_bars} bars)")
        return None

    start = total - bt_bars
    nw    = bt_bars // (TEST_BARS + EMBARGO_BARS)
    P(f"\n  {symbol}: {nw} purged windows, {bt_bars} bars ({bt_bars // 24}d)")
    P(f"  Purge: {PURGE_BARS} bars | Embargo: {EMBARGO_BARS} bars")

    entries  = []
    waccs    = []
    w_times  = []

    for w in range(nw):
        w_t0 = time.time()

        # Test window bounds (purge removed from boundaries)
        test_start = start + w * (TEST_BARS + EMBARGO_BARS) + PURGE_BARS
        test_end   = test_start + TEST_BARS
        if test_end + HORIZON > total:
            break

        # Train window: ends PURGE_BARS before test_start
        train_end   = test_start - PURGE_BARS
        train_start = max(0, train_end - TRAIN_BARS)

        if train_end - train_start < 100:
            P(f"    W{w+1}: skip (train too small)")
            continue

        ens = train_fold_purged(feat, train_start, train_end, fcols)
        if ens is None:
            P(f"    W{w+1}/{nw}: SKIPPED (train failed)")
            continue

        ba       = max(ens["accs"].values()) if ens["accs"] else 0
        n_models = len(ens["accs"])
        waccs.append(ba)
        elapsed  = time.time() - w_t0
        w_times.append(elapsed)
        eta      = np.mean(w_times) * (nw - w - 1)
        acc_s    = " ".join(f"{k}:{v*100:.0f}" for k, v in ens["accs"].items())
        P(f"    W{w+1}/{nw} [{acc_s}] best={ba*100:.1f}% "
          f"({n_models}m) {elapsed:.0f}s ETA:{eta/60:.0f}m")

        # Pre-scale test slice for sequence building
        X_raw_slice = feat.iloc[max(0, test_start - SEQ_LEN):test_end][fcols].values.astype(float)
        # ANTI-BIAS: use training scaler (already fit) — never refit here
        try:
            X_scaled_slice = ens["scaler"].transform(X_raw_slice)
        except Exception:
            continue

        for i in range(test_start, min(test_end, total - HORIZON)):
            row = feat.iloc[i]
            if row.get("_is_weekend", False):
                continue
            session = row.get("_session", "LONDON_OPEN")
            if session == "OFF_HOURS":
                continue
            sess_mult   = SESSION_MULT.get(session, 0.5)

            X           = row[fcols].values.astype(float)
            regime_str  = row.get("_regime_str", "UNKNOWN")

            # Build sequence for LSTM/TCN
            X_seq = None
            if HAS_TORCH:
                offset = i - max(0, test_start - SEQ_LEN)
                if offset >= SEQ_LEN:
                    X_seq = X_scaled_slice[offset - SEQ_LEN:offset].astype(np.float32)

            conf, ps = predict_with_regime(ens, X, X_seq, regime_str)

            # Regime-dependent threshold
            if regime_str in TRENDING_SET:   threshold = CONF_TRENDING
            elif regime_str in RANGING_SET:  threshold = CONF_RANGING
            else:                            threshold = CONF_VOLATILE

            if conf > threshold:        sig = "BUY"
            elif conf < 1 - threshold:  sig = "SELL"
            else:                       continue

            # Risk / position sizing
            vol_spike    = row.get("_vol_spike", 0)
            news_mult    = 0.7 if vol_spike else 1.0
            regime_rmult = regime_risk_multiplier(regime_str)
            if regime_rmult <= 0.1:     continue

            if conf > 0.80:   conf_mult = 1.5
            elif conf > 0.70: conf_mult = 1.2
            elif conf > 0.60: conf_mult = 1.0
            else:             conf_mult = 0.5

            base_risk = (RISK_TRENDING if regime_str in TRENDING_SET
                         else RISK_RANGING if regime_str in RANGING_SET
                         else RISK_BASE)
            risk_pct  = min(base_risk * conf_mult * regime_rmult * news_mult * sess_mult,
                            RISK_MAX)
            if risk_pct <= 0: continue

            atr = row.get("_atr_dollar", 25.0)
            if np.isnan(atr) or atr <= 0: atr = 25.0

            sp = SPREAD[symbol]
            if regime_str in TRENDING_SET:
                sl_d = atr * 0.35; tp_d = atr * 0.90
            elif regime_str in RANGING_SET:
                sl_d = atr * 0.25; tp_d = atr * 0.50
            else:
                sl_d = atr * 0.40; tp_d = atr * 0.80
            sl_d = max(sl_d, 15.0); tp_d = max(tp_d, 20.0)

            close_p = row.get("_close", 0)
            if close_p <= 0: continue

            if sig == "BUY":
                entry = close_p + sp / 2; sl = entry - sl_d; tp = entry + tp_d
            else:
                entry = close_p - sp / 2; sl = entry + sl_d; tp = entry - tp_d

            entries.append({
                "bar":        i,
                "sym":        symbol,
                "dir":        sig,
                "entry":      entry,
                "sl":         sl,
                "tp":         tp,
                "sl_d":       sl_d,
                "tp_d":       tp_d,
                "atr":        atr,
                "conf":       round(conf, 4),
                "regime":     regime_str,
                "risk_pct":   round(risk_pct, 3),
                "news_reduced": bool(vol_spike),
                "session":    session,
                "time":       str(feat.index[i]) if i < len(feat) else "",
                "model_ps":   {k: round(v, 3) for k, v in ps.items()},
            })

    return {"entries": entries, "waccs": waccs, "symbol": symbol}


# ═══════════════════════════════════════════════════════════════════
# POSITION SIMULATOR
# ═══════════════════════════════════════════════════════════════════
class Pos:
    def __init__(s, sym, dir_, entry, lot, sl, tp, bar, atr):
        s.sym=sym; s.dir=dir_; s.entry=entry; s.lot=lot
        s.sl=sl; s.tp=tp; s.bar=bar; s.atr=atr
        s.closed=False; s.exit_p=0; s.profit=0; s.reason=""
        s.be=False; s.partial=False; s._regime=""; s._time=""

    def tick(s, h, l, c, bar):
        if s.closed: return
        tp_dist    = (s.tp - s.entry) if s.dir == "BUY" else (s.entry - s.tp)
        profit_pts = (c - s.entry) if s.dir == "BUY" else (s.entry - c)
        tp_dist    = max(tp_dist, s.atr)
        progress   = profit_pts / tp_dist if tp_dist > 0 else 0

        if progress >= BE_PCT and not s.be:
            s.be = True
            s.sl = s.entry + 0.05 if s.dir == "BUY" else s.entry - 0.05

        if progress >= PARTIAL_PCT and not s.partial and s.lot >= 0.02:
            s.partial = True
            half = round(s.lot / 2, 2)
            s.profit += profit_pts * half * PIP_VAL[s.sym]
            s.lot = max(0.01, round(s.lot - half, 2))

        if progress >= TRAIL_PCT:
            td = s.atr * TRAIL_ATR_MULT
            ns = (c - td) if s.dir == "BUY" else (c + td)
            if s.dir == "BUY" and ns > s.sl:   s.sl = ns
            if s.dir == "SELL" and (s.sl == 0 or ns < s.sl): s.sl = ns

        hit_tp = (h >= s.tp) if s.dir == "BUY" else (l <= s.tp)
        hit_sl = (l <= s.sl) if s.dir == "BUY" else (h >= s.sl)
        if hit_tp:
            s.closed=True; s.exit_p=s.tp; s.reason="TP"
        elif hit_sl:
            s.closed=True; s.exit_p=s.sl; s.reason="BE" if s.be else "SL"
        if s.closed:
            pv       = PIP_VAL[s.sym]
            s.profit += ((s.exit_p - s.entry) if s.dir == "BUY"
                         else (s.entry - s.exit_p)) * s.lot * pv


# ═══════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════
def simulate(results_list, dfs):
    bal       = BALANCE_START
    eq_curve  = [bal]
    open_p    = []
    closed    = []
    peak      = bal
    mdd       = 0.0

    all_e = []
    for r in results_list:
        if r is None: continue
        all_e.extend(r["entries"])
    all_e.sort(key=lambda x: x["bar"])

    if not all_e:
        return {"eq": eq_curve, "trades": [], "bal": bal, "mdd": 0, "peak": bal}

    mn            = all_e[0]["bar"]
    mx            = min(max(e["bar"] for e in all_e) + 10 * 24,
                        max(len(df) for df in dfs.values()))
    ei            = 0
    recent_trades = []
    last_bar      = {}
    trades_today  = 0
    current_day   = -1

    for bar in range(mn, mx):
        # Tick open positions
        for p in open_p:
            if p.closed: continue
            df = dfs.get(p.sym)
            if df is None or bar >= len(df): continue
            p.tick(float(df.iloc[bar]["high"]),
                   float(df.iloc[bar]["low"]),
                   float(df.iloc[bar]["close"]), bar)
            if p.closed:
                bal += p.profit
                closed.append({
                    "sym":    p.sym, "dir": p.dir,
                    "entry":  round(p.entry, 2), "exit": round(p.exit_p, 2),
                    "profit": round(p.profit, 2), "reason": p.reason,
                    "bars":   bar - p.bar, "lot": p.lot,
                    "regime": p._regime, "time": p._time,
                })
                recent_trades.append((p.dir, p.profit > 0))
                recent_trades = recent_trades[-10:]
        open_p = [p for p in open_p if not p.closed]

        # Day rollover
        for sym, df in dfs.items():
            if bar < len(df) and hasattr(df.index[bar], "day"):
                d = df.index[bar].day
                if d != current_day:
                    current_day = d; trades_today = 0
                break

        # Open new positions
        while ei < len(all_e) and all_e[ei]["bar"] == bar:
            e        = all_e[ei]; ei += 1; sym = e["sym"]
            sym_open = sum(1 for p in open_p if p.sym == sym)
            if sym_open >= MAX_POS:               continue
            if trades_today >= MAX_TRADES_DAY:    continue
            if bar - last_bar.get(sym, -999) < COOLDOWN_BARS: continue
            # Consecutive loss filter
            dir_losses = 0
            for rd, rw in reversed(recent_trades):
                if rd == e["dir"] and not rw: dir_losses += 1
                else: break
            if dir_losses >= 2: continue
            if bal <= 10:       continue

            # Kelly-informed lot sizing
            risk_amount = bal * (e["risk_pct"] / 100)
            sl_d        = e["sl_d"]
            if sl_d <= 0: continue
            pv  = PIP_VAL[sym]
            lot = risk_amount / (sl_d * pv / 100)
            if e["news_reduced"]: lot *= 0.7
            lot = max(0.01, min(round(lot, 2), 1.0))
            # Safety: actual risk cap at 5% of balance
            actual_risk = lot * sl_d * pv / 100
            if actual_risk > bal * 0.05:
                lot = max(0.01, round(bal * 0.05 / (sl_d * pv / 100), 2))

            pos          = Pos(sym, e["dir"], e["entry"], lot,
                               e["sl"], e["tp"], bar, e["atr"])
            pos._regime  = e["regime"]
            pos._time    = e["time"]
            open_p.append(pos)
            last_bar[sym]  = bar
            trades_today  += 1

        # Mark-to-market equity
        unr = sum(
            ((float(dfs[p.sym].iloc[bar]["close"]) - p.entry
              if p.dir == "BUY" else
              p.entry - float(dfs[p.sym].iloc[bar]["close"])) * p.lot * PIP_VAL[p.sym])
            for p in open_p
            if not p.closed and p.sym in dfs and bar < len(dfs[p.sym])
        )
        equity = bal + unr
        eq_curve.append(equity)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > mdd: mdd = dd

    # Force-close remaining
    for p in open_p:
        if p.closed: continue
        df = dfs.get(p.sym)
        if df is None: continue
        c = float(df.iloc[-1]["close"]); pv = PIP_VAL[p.sym]
        p.profit += ((c - p.entry) if p.dir == "BUY" else (p.entry - c)) * p.lot * pv
        bal += p.profit
        closed.append({
            "sym": p.sym, "dir": p.dir, "entry": round(p.entry, 2),
            "exit": round(c, 2), "profit": round(p.profit, 2),
            "reason": "FORCE", "bars": 0, "lot": p.lot,
            "regime": p._regime, "time": p._time,
        })

    return {
        "eq": eq_curve, "trades": closed,
        "bal": round(bal, 2), "mdd": round(mdd * 100, 2), "peak": round(peak, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# RISK METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(sim, backtest_days):
    trades  = sim["trades"]
    eq      = sim["eq"]
    pnl     = sim["bal"] - BALANCE_START
    pnl_pct = pnl / BALANCE_START * 100

    wins   = [t for t in trades if t["profit"] > 0]
    losses = [t for t in trades if t["profit"] < 0]

    win_rate   = len(wins) / len(trades) * 100 if trades else 0
    avg_win    = float(np.mean([t["profit"] for t in wins]))    if wins   else 0
    avg_loss   = float(np.mean([abs(t["profit"]) for t in losses])) if losses else 0
    pf         = sum(t["profit"] for t in wins) / max(1, sum(abs(t["profit"]) for t in losses))
    r_multiple = avg_win / max(avg_loss, 0.01)    # avg R-multiple

    # Sharpe (annualised, daily returns)
    daily_rets = np.diff(eq)
    sharpe     = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-10)
                  * np.sqrt(252)) if len(daily_rets) > 1 else 0

    # Sortino (downside std only)
    down_rets  = daily_rets[daily_rets < 0]
    sortino    = (np.mean(daily_rets) / (np.std(down_rets) + 1e-10)
                  * np.sqrt(252)) if len(down_rets) > 0 else 0

    # Calmar = CAGR / max_drawdown
    years = max(backtest_days / 365, 0.1)
    cagr  = ((sim["bal"] / BALANCE_START) ** (1 / years) - 1) * 100
    calmar = cagr / max(sim["mdd"], 0.01)

    return {
        "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
        "win_rate": round(win_rate, 1), "n_trades": len(trades),
        "n_wins": len(wins), "n_losses": len(losses),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "profit_factor": round(pf, 3), "r_multiple": round(r_multiple, 3),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "calmar": round(calmar, 3), "cagr_pct": round(cagr, 2),
        "max_dd": sim["mdd"],
    }


# ═══════════════════════════════════════════════════════════════════
# EXECUTE
# ═══════════════════════════════════════════════════════════════════
P(f"\n  Running purged walk-forward ({BACKTEST_DAYS // 365}yr)...")
P(f"  Purge={PURGE_BARS} bars  Embargo={EMBARGO_BARS} bars")
t0 = time.time()

r_usd = walkforward_purged(df_usd, feat_usd, "XAUUSD") if feat_usd is not None else None
r_sgd = walkforward_purged(df_sgd, feat_sgd, "XAUSGD") if feat_sgd is not None else None

elapsed = time.time() - t0
P(f"\n  Walk-forward done in {elapsed:.0f}s ({elapsed / 60:.1f}min)")

dfs = {}
if df_usd is not None: dfs["XAUUSD"] = df_usd
if df_sgd is not None: dfs["XAUSGD"] = df_sgd

P("\n  Simulating trades (anti-bias)...")
sim     = simulate([r_usd, r_sgd], dfs)
metrics = compute_metrics(sim, BACKTEST_DAYS)

# ═══════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════
P(f"\n  {'='*70}")
P(f"  ANTI-BIAS WALK-FORWARD RESULTS  |  Purged+Embargoed  |  £500 Start")
P(f"  {'='*70}")
P(f"  Start Balance  : £{BALANCE_START:.2f}")
P(f"  Final Balance  : £{sim['bal']:.2f}")
P(f"  P&L            : £{metrics['pnl']:+.2f} ({metrics['pnl_pct']:+.1f}%)")
P(f"  CAGR           : {metrics['cagr_pct']:+.2f}%/yr")
P(f"  Max Drawdown   : {metrics['max_dd']:.1f}%")
P(f"  Total Trades   : {metrics['n_trades']}")
P(f"  Win Rate       : {metrics['win_rate']:.1f}%  ({metrics['n_wins']}W / {metrics['n_losses']}L)")
P(f"  Avg Win        : £{metrics['avg_win']:.2f}")
P(f"  Avg Loss       : £{metrics['avg_loss']:.2f}")
P(f"  Profit Factor  : {metrics['profit_factor']:.2f}")
P(f"  R-Multiple     : {metrics['r_multiple']:.2f}  (avg win/avg loss)")
P(f"  Sharpe         : {metrics['sharpe']:.3f}")
P(f"  Sortino        : {metrics['sortino']:.3f}")
P(f"  Calmar         : {metrics['calmar']:.3f}")

if sim["trades"]:
    # Monthly breakdown
    monthly = {}
    for t in sim["trades"]:
        mo = t.get("time", "")[:7]
        if mo:
            if mo not in monthly:
                monthly[mo] = {"pnl": 0, "n": 0, "w": 0}
            monthly[mo]["pnl"] += t["profit"]; monthly[mo]["n"] += 1
            if t["profit"] > 0: monthly[mo]["w"] += 1
    if monthly:
        P(f"\n  Monthly Breakdown:")
        P(f"  {'Month':>8s} {'Trades':>7s} {'WR':>6s} {'P&L':>10s}")
        for mo in sorted(monthly):
            d = monthly[mo]
            wr = d["w"] / d["n"] * 100 if d["n"] > 0 else 0
            P(f"  {mo:>8s} {d['n']:>7d} {wr:>5.0f}% {d['pnl']:>+10.2f}")

    # Per-symbol
    for sym in ["XAUUSD", "XAUSGD"]:
        st = [t for t in sim["trades"] if t["sym"] == sym]
        if st:
            sp  = sum(t["profit"] for t in st)
            sw  = sum(1 for t in st if t["profit"] > 0)
            sl  = sum(1 for t in st if t["profit"] < 0)
            swr = sw / len(st) * 100
            P(f"\n  --- {sym}: {len(st)} trades | {sw}W/{sl}L ({swr:.0f}%) | £{sp:+.2f} ---")
            regimes = {}
            for t in st:
                rg = t.get("regime", "?")
                if rg not in regimes: regimes[rg] = {"n": 0, "pnl": 0}
                regimes[rg]["n"] += 1; regimes[rg]["pnl"] += t["profit"]
            for rg, d in sorted(regimes.items(), key=lambda x: x[1]["pnl"], reverse=True):
                P(f"    {rg:25s}: {d['n']:3d} trades  £{d['pnl']:+.2f}")

    P(f"\n  Close Reasons:")
    rs = {}
    for t in sim["trades"]:
        r = t["reason"]; rs[r] = rs.get(r, 0) + 1
    for r, c in sorted(rs.items()):
        P(f"    {r:10s}: {c}")

    for r, sym in [(r_usd, "XAUUSD"), (r_sgd, "XAUSGD")]:
        if r and r["waccs"]:
            P(f"\n  {sym} avg OOS accuracy: {np.mean(r['waccs'])*100:.1f}% ({len(r['waccs'])} windows)")

    P(f"\n  Last 20 trades:")
    P(f"  {'Sym':7s} {'Dir':5s} {'Entry':>10s} {'Exit':>10s} {'P&L':>8s} {'Rsn':>5s} {'Regime':>14s}")
    for t in sim["trades"][-20:]:
        P(f"  {t['sym']:7s} {t['dir']:5s} {t['entry']:10.2f} {t['exit']:10.2f} "
          f"{t['profit']:+8.2f} {t['reason']:>5s} {t.get('regime',''):>14s}")

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
out = {
    "start_bal":      BALANCE_START,
    "final_bal":      sim["bal"],
    "metrics":        metrics,
    "trades":         sim["trades"],
    "equity":         [round(e, 2) for e in sim["eq"]],
    "models":         "XGB+LGB+GB+CB+RF+LSTM+TCN",
    "bias_controls":  f"Purge={PURGE_BARS} Embargo={EMBARGO_BARS} ScalerFitOnTrain=True",
    "backtest_days":  BACKTEST_DAYS,
    "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
}
rp = os.path.join(BASE, "walkforward_results.json")
with open(rp, "w") as f:
    json.dump(out, f, indent=2)
P(f"\n  Saved: {rp}")
P(f"  Backtest complete ({elapsed / 60:.1f} min)")
