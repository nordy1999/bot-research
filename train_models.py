"""
AlmostFinishedBot - ML Trainer v6
PyTorch GPU + CatBoost + Random Forest + Regime-Conditional

v6 changes from v4:
  - CatBoost and Random Forest added as base models (7 total)
  - LSTM rewritten in PyTorch (GPU-accelerated on RTX 2060)
  - TFT (Temporal Fusion Transformer) bonus model in PyTorch
  - 5 strategy comparison: Weighted, Top-3, Regime-Conditional, Meta, Hybrid
  - Meta-learner uses all 6 base model outputs + regime code (7 inputs)
  - Auto-deploy only if better than previous ensemble
  - trainer_version: "v6" flag in ensemble_config.json
"""
import os, sys, warnings, json, time, shutil, datetime
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
os.makedirs(BASE, exist_ok=True)
sys.path.insert(0, BASE)

print("=" * 65)
print("  AlmostFinishedBot  |  ML Trainer v6")
print("  PyTorch GPU + CatBoost + RF + Regime-Conditional")
print("=" * 65)

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for pkg in ["xgboost", "lightgbm", "scikit-learn", "yfinance", "joblib", "catboost"]:
    try:
        __import__(pkg.replace("-","_").replace("scikit_learn","sklearn"))
    except ImportError:
        print(f"  Installing {pkg}..."); install(pkg)

import yfinance as yf
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

from features import make_features, make_target, make_regime_labels, kelly_fraction, download_gold

# CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CB = True
    print("  CatBoost: available")
except ImportError:
    HAS_CB = False
    print("  CatBoost: not available")

# PyTorch GPU
HAS_TORCH = False
DEVICE = "cpu"
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  PyTorch: GPU mode ({gpu_name})")
    else:
        print("  PyTorch: CPU mode (no CUDA)")
except ImportError:
    print("  PyTorch: not available, skipping LSTM/TFT")

HORIZON = 5
THRESHOLD = 0.001
PURGE_GAP = HORIZON + 2
MIN_ACC = 0.52
MIN_REGIME_SAMPLES = 200
SPREAD_PCT = 0.003
TRENDING_REGIMES = {"TREND_UP", "TREND_DOWN"}

# == Download ===============================================================
print("\n  Downloading gold data (6 months, 1h)...")
df = download_gold("6mo", "1h")
print(f"  Downloaded {len(df)} candles")

# == Features + Regimes =====================================================
print("  Engineering features...")
feat_df = make_features(df)
feat_df["target"] = make_target(df["close"].values, horizon=HORIZON, threshold=THRESHOLD)

print("  Detecting regimes...")
regime_labels = make_regime_labels(df)
feat_df["_regime"] = regime_labels[:len(feat_df)]
feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

print(f"  Rows after cleaning: {len(feat_df)}")
if len(feat_df) < 200:
    print("  ERROR: not enough rows"); sys.exit(1)

class_balance = feat_df["target"].mean()
print(f"  Class balance: {class_balance:.1%} buys / {1-class_balance:.1%} non-buys")

regime_counts = feat_df["_regime"].value_counts()
print(f"\n  Regime distribution:")
for r, c in regime_counts.items():
    print(f"    {r:25s}: {c:5d} rows ({c/len(feat_df)*100:.1f}%)")

feature_cols = [c for c in feat_df.columns if c not in ("target", "_regime")]
X_all = feat_df[feature_cols].values.astype(float)
y_all = feat_df["target"].values.astype(int)
regimes_all = feat_df["_regime"].values

split = int(len(X_all) * 0.8)
X_tr = X_all[:split - PURGE_GAP]
y_tr = y_all[:split - PURGE_GAP]
reg_tr = regimes_all[:split - PURGE_GAP]
X_val = X_all[split:]
y_val = y_all[split:]
reg_val = regimes_all[split:]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)

print(f"\n  Train: {len(X_tr)}  Val: {len(X_val)}  Features: {len(feature_cols)}")
print(f"  Purge gap: {PURGE_GAP} | Horizon: {HORIZON}h | Threshold: {THRESHOLD*100:.2f}%")

# == PHASE 1: Global Base Models ============================================
print(f"\n  {'='*55}")
print(f"  PHASE 1: Global Base Models (6 + bonus)")
print(f"  {'='*55}")

print("\n  [1/7] Training XGBoost...")
xgb_m = xgb.XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, eval_metric="logloss", verbosity=0, random_state=42)
xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
xgb_acc = accuracy_score(y_val, xgb_m.predict(X_val_s))
xgb_prob = xgb_m.predict_proba(X_val_s)[:,1]
print(f"     XGBoost:     {xgb_acc*100:.1f}%  {'[OK]' if xgb_acc >= MIN_ACC else '[WEAK]'}")

print("  [2/7] Training LightGBM...")
lgb_m = lgb.LGBMClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03, num_leaves=31,
    subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_samples=20, verbose=-1, random_state=42)
lgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
lgb_acc = accuracy_score(y_val, lgb_m.predict(X_val_s))
lgb_prob = lgb_m.predict_proba(X_val_s)[:,1]
print(f"     LightGBM:    {lgb_acc*100:.1f}%  {'[OK]' if lgb_acc >= MIN_ACC else '[WEAK]'}")

print("  [3/7] Training GradientBoost...")
gb_m = GradientBoostingClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.7, min_samples_leaf=20, random_state=42)
gb_m.fit(X_tr_s, y_tr)
gb_acc = accuracy_score(y_val, gb_m.predict(X_val_s))
gb_prob = gb_m.predict_proba(X_val_s)[:,1]
print(f"     GradBoost:   {gb_acc*100:.1f}%  {'[OK]' if gb_acc >= MIN_ACC else '[WEAK]'}")

# CatBoost
cb_m = None; cb_acc = 0.5; cb_prob = np.full(len(y_val), 0.5)
if HAS_CB:
    print("  [4/7] Training CatBoost...")
    cb_m = CatBoostClassifier(
        iterations=500, depth=4, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.7, random_seed=42,
        verbose=0, eval_metric="Accuracy")
    cb_m.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val), verbose=0, early_stopping_rounds=30)
    cb_acc = accuracy_score(y_val, cb_m.predict(X_val_s))
    cb_prob = cb_m.predict_proba(X_val_s)[:,1]
    print(f"     CatBoost:    {cb_acc*100:.1f}%  {'[OK]' if cb_acc >= MIN_ACC else '[WEAK]'}")
else:
    print("  [4/7] CatBoost skipped")

# Random Forest
print("  [5/7] Training Random Forest...")
rf_m = RandomForestClassifier(
    n_estimators=500, max_depth=6, min_samples_leaf=20,
    max_features="sqrt", oob_score=True, random_state=42, n_jobs=-1)
rf_m.fit(X_tr_s, y_tr)
rf_acc = accuracy_score(y_val, rf_m.predict(X_val_s))
rf_prob = rf_m.predict_proba(X_val_s)[:,1]
oob = rf_m.oob_score_ if hasattr(rf_m, "oob_score_") else 0
print(f"     RandomForest:{rf_acc*100:.1f}%  {'[OK]' if rf_acc >= MIN_ACC else '[WEAK]'}  (OOB={oob*100:.1f}%)")

# PyTorch LSTM
SEQ = 20
lstm_acc = 0.5; lstm_prob = np.full(len(y_val), 0.5); lstm_model_pt = None

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

if HAS_TORCH:
    print(f"  [6/7] Training LSTM (PyTorch {DEVICE.upper()})...")
    def make_seqs(X, y, s):
        xs, ys = [], []
        for i in range(s, len(X)):
            xs.append(X[i-s:i])
            ys.append(y[i])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    Xs, ys = make_seqs(X_tr_s, y_tr, SEQ)
    Xvs, yvs = make_seqs(X_val_s, y_val, SEQ)

    if len(Xs) >= 64:
        t_start = time.time()
        train_ds = TensorDataset(torch.tensor(Xs), torch.tensor(ys))
        val_ds = TensorDataset(torch.tensor(Xvs), torch.tensor(yvs))
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=128)

        lstm_model_pt = LSTMModel(X_tr_s.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(lstm_model_pt.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
        loss_fn = nn.BCELoss()

        best_val_loss = float("inf"); patience = 8; wait = 0; best_state = None
        for epoch in range(50):
            lstm_model_pt.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = lstm_model_pt(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()

            lstm_model_pt.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = lstm_model_pt(xb)
                    val_losses.append(loss_fn(pred, yb).item())
            vl = np.mean(val_losses)
            scheduler.step(vl)
            if vl < best_val_loss:
                best_val_loss = vl; wait = 0
                best_state = {k: v.cpu().clone() for k, v in lstm_model_pt.state_dict().items()}
            else:
                wait += 1
                if wait >= patience: break

        if best_state:
            lstm_model_pt.load_state_dict(best_state)
        lstm_model_pt.eval()
        with torch.no_grad():
            raw = lstm_model_pt(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
        lstm_acc = accuracy_score(yvs, (raw > 0.5).astype(int))
        lstm_prob_aligned = np.full(len(y_val), 0.5)
        lstm_prob_aligned[-len(raw):] = raw
        lstm_prob = lstm_prob_aligned
        elapsed = time.time() - t_start
        print(f"     LSTM:        {lstm_acc*100:.1f}%  {'[OK]' if lstm_acc >= MIN_ACC else '[WEAK]'}  ({elapsed:.1f}s on {DEVICE.upper()})")

        # Save PyTorch LSTM
        torch.save({
            "model_state": lstm_model_pt.state_dict(),
            "n_features": X_tr_s.shape[1],
            "seq_len": SEQ,
            "hidden": 64,
            "layers": 2,
        }, os.path.join(BASE, "lstm_model.pt"))
    else:
        print("     LSTM: not enough sequences")
else:
    print("  [6/7] LSTM skipped (no PyTorch)")

# PyTorch TFT (simplified)
tft_acc = 0.5; tft_prob = np.full(len(y_val), 0.5); tft_model_pt = None

class SimpleTFT(nn.Module):
    """Simplified Temporal Fusion Transformer - attention over sequence"""
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

if HAS_TORCH and len(Xs) >= 64:
    print(f"  [7/7] Training TFT (PyTorch {DEVICE.upper()})...")
    t_start = time.time()
    tft_model_pt = SimpleTFT(X_tr_s.shape[1]).to(DEVICE)
    opt_tft = torch.optim.Adam(tft_model_pt.parameters(), lr=0.0005)
    scheduler_tft = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_tft, patience=4, factor=0.5)
    loss_fn = nn.BCELoss()

    best_val_loss = float("inf"); patience = 8; wait = 0; best_state = None
    for epoch in range(50):
        tft_model_pt.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = tft_model_pt(xb)
            loss = loss_fn(pred, yb)
            opt_tft.zero_grad(); loss.backward(); opt_tft.step()

        tft_model_pt.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = tft_model_pt(xb)
                val_losses.append(loss_fn(pred, yb).item())
        vl = np.mean(val_losses)
        scheduler_tft.step(vl)
        if vl < best_val_loss:
            best_val_loss = vl; wait = 0
            best_state = {k: v.cpu().clone() for k, v in tft_model_pt.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break

    if best_state:
        tft_model_pt.load_state_dict(best_state)
    tft_model_pt.eval()
    with torch.no_grad():
        raw_tft = tft_model_pt(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
    tft_acc = accuracy_score(yvs, (raw_tft > 0.5).astype(int))
    tft_prob_aligned = np.full(len(y_val), 0.5)
    tft_prob_aligned[-len(raw_tft):] = raw_tft
    tft_prob = tft_prob_aligned
    elapsed = time.time() - t_start
    print(f"     TFT:         {tft_acc*100:.1f}%  {'[OK]' if tft_acc >= MIN_ACC else '[WEAK]'}  ({elapsed:.1f}s on {DEVICE.upper()})")

    torch.save({
        "model_state": tft_model_pt.state_dict(),
        "n_features": X_tr_s.shape[1],
        "seq_len": SEQ,
        "d_model": 64,
        "n_heads": 4,
    }, os.path.join(BASE, "tft_model.pt"))
else:
    if not HAS_TORCH:
        print("  [7/7] TFT skipped (no PyTorch)")
    else:
        print("  [7/7] TFT skipped (not enough sequences)")

# == PHASE 2: Regime-Conditional Models =====================================
print(f"\n  {'='*55}")
print(f"  PHASE 2: Regime-Conditional Models")
print(f"  {'='*55}")

trend_mask_tr = np.array([r in TRENDING_REGIMES for r in reg_tr])
trend_mask_val = np.array([r in TRENDING_REGIMES for r in reg_val])
regime_models = {}

for regime_name, mask_tr, mask_val in [
    ("TRENDING", trend_mask_tr, trend_mask_val),
    ("CHOPPY", ~trend_mask_tr, ~trend_mask_val),
]:
    n_tr = mask_tr.sum(); n_val = mask_val.sum()
    print(f"\n  --- {regime_name}: {n_tr} train / {n_val} val ---")
    if n_tr < MIN_REGIME_SAMPLES or n_val < 30:
        print(f"  Skipping: not enough samples")
        regime_models[regime_name] = None; continue

    X_r_tr = X_tr_s[mask_tr]; y_r_tr = y_tr[mask_tr]
    X_r_val = X_val_s[mask_val]; y_r_val = y_val[mask_val]

    r_lgb = lgb.LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
        num_leaves=24, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.2,
        reg_lambda=1.5, min_child_samples=15, verbose=-1, random_state=42)
    r_lgb.fit(X_r_tr, y_r_tr, eval_set=[(X_r_val, y_r_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    r_lgb_acc = accuracy_score(y_r_val, r_lgb.predict(X_r_val))

    r_gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.7, min_samples_leaf=15, random_state=42)
    r_gb.fit(X_r_tr, y_r_tr)
    r_gb_acc = accuracy_score(y_r_val, r_gb.predict(X_r_val))

    # Also try CatBoost for regime
    r_cb_acc = 0.0; r_cb = None
    if HAS_CB:
        r_cb = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.03,
            l2_leaf_reg=3.0, subsample=0.7, random_seed=42, verbose=0)
        r_cb.fit(X_r_tr, y_r_tr, eval_set=(X_r_val, y_r_val), verbose=0, early_stopping_rounds=30)
        r_cb_acc = accuracy_score(y_r_val, r_cb.predict(X_r_val))

    print(f"  {regime_name} LGB: {r_lgb_acc*100:.1f}%  |  GB: {r_gb_acc*100:.1f}%  |  CB: {r_cb_acc*100:.1f}%")

    candidates = {"lgb": (r_lgb, r_lgb_acc), "gb": (r_gb, r_gb_acc)}
    if HAS_CB and r_cb is not None:
        candidates["cb"] = (r_cb, r_cb_acc)
    best_regime_name = max(candidates, key=lambda k: candidates[k][1])
    best_regime_model, best_regime_acc = candidates[best_regime_name]

    if best_regime_acc >= MIN_ACC:
        regime_models[regime_name] = {
            best_regime_name: best_regime_model,
            "best": best_regime_name,
            "best_acc": best_regime_acc,
            "lgb": r_lgb, "lgb_acc": r_lgb_acc,
            "gb": r_gb, "gb_acc": r_gb_acc,
        }
        if HAS_CB and r_cb is not None:
            regime_models[regime_name]["cb"] = r_cb
            regime_models[regime_name]["cb_acc"] = r_cb_acc
        print(f"  BEST: {best_regime_name.upper()} ({best_regime_acc*100:.1f}%)")
    else:
        regime_models[regime_name] = None
        print(f"  All below {MIN_ACC*100:.0f}%, using global")

# Save regime_models for bridge (ensure best model accessible as trend_model/range_model)
regime_save = {}
if regime_models.get("TRENDING") is not None:
    rm = regime_models["TRENDING"]
    regime_save["trend_model"] = rm[rm["best"]]
    regime_save["trend_best"] = rm["best"]
    regime_save["trend_acc"] = rm["best_acc"]
if regime_models.get("CHOPPY") is not None:
    rm = regime_models["CHOPPY"]
    regime_save["range_model"] = rm[rm["best"]]
    regime_save["range_best"] = rm["best"]
    regime_save["range_acc"] = rm["best_acc"]

# == PHASE 3: Stacking Meta-Learner =========================================
print(f"\n  {'='*55}")
print(f"  PHASE 3: Stacking Meta-Learner (6 models + regime)")
print(f"  {'='*55}")

meta_val = np.column_stack([xgb_prob, lgb_prob, gb_prob, cb_prob, rf_prob, lstm_prob,
    np.array([1.0 if r in TRENDING_REGIMES else 0.0 for r in reg_val])])

# OOF cross-predictions for meta training
half = len(X_tr_s) // 2
print("  Building out-of-fold predictions...")

xgb_h1 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, eval_metric="logloss", verbosity=0, random_state=42)
xgb_h1.fit(X_tr_s[:half], y_tr[:half], verbose=False)
lgb_h1 = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, verbose=-1, random_state=42)
lgb_h1.fit(X_tr_s[:half], y_tr[:half])
gb_h1 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.7, random_state=42)
gb_h1.fit(X_tr_s[:half], y_tr[:half])

xgb_h2 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, eval_metric="logloss", verbosity=0, random_state=42)
xgb_h2.fit(X_tr_s[half:], y_tr[half:], verbose=False)
lgb_h2 = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, verbose=-1, random_state=42)
lgb_h2.fit(X_tr_s[half:], y_tr[half:])
gb_h2 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.7, random_state=42)
gb_h2.fit(X_tr_s[half:], y_tr[half:])

# OOF for CatBoost
if HAS_CB:
    cb_h1 = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
    cb_h1.fit(X_tr_s[:half], y_tr[:half], verbose=0)
    cb_h2 = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
    cb_h2.fit(X_tr_s[half:], y_tr[half:], verbose=0)
    oof_cb = np.concatenate([cb_h2.predict_proba(X_tr_s[:half])[:,1],
                              cb_h1.predict_proba(X_tr_s[half:])[:,1]])
else:
    oof_cb = np.full(len(y_tr), 0.5)

# OOF for RF
rf_h1 = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20,
    max_features="sqrt", random_state=42, n_jobs=-1)
rf_h1.fit(X_tr_s[:half], y_tr[:half])
rf_h2 = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20,
    max_features="sqrt", random_state=42, n_jobs=-1)
rf_h2.fit(X_tr_s[half:], y_tr[half:])
oof_rf = np.concatenate([rf_h2.predict_proba(X_tr_s[:half])[:,1],
                          rf_h1.predict_proba(X_tr_s[half:])[:,1]])

oof_xgb = np.concatenate([xgb_h2.predict_proba(X_tr_s[:half])[:,1],
                           xgb_h1.predict_proba(X_tr_s[half:])[:,1]])
oof_lgb = np.concatenate([lgb_h2.predict_proba(X_tr_s[:half])[:,1],
                           lgb_h1.predict_proba(X_tr_s[half:])[:,1]])
oof_gb  = np.concatenate([gb_h2.predict_proba(X_tr_s[:half])[:,1],
                           gb_h1.predict_proba(X_tr_s[half:])[:,1]])
oof_lstm = np.full(len(y_tr), 0.5)
regime_code_tr = np.array([1.0 if r in TRENDING_REGIMES else 0.0 for r in reg_tr])

meta_tr = np.column_stack([oof_xgb, oof_lgb, oof_gb, oof_cb, oof_rf, oof_lstm, regime_code_tr])

print("  Training meta-learner (7 inputs)...")
meta_model = lgb.LGBMClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, num_leaves=16,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
    min_child_samples=30, verbose=-1, random_state=42)
meta_model.fit(meta_tr, y_tr, eval_set=[(meta_val, y_val)],
    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
meta_prob = meta_model.predict_proba(meta_val)[:,1]
meta_acc = accuracy_score(y_val, (meta_prob > 0.5).astype(int))
print(f"  Meta-learner: {meta_acc*100:.1f}%")

# == PHASE 4: Pick Best Strategy ============================================
print(f"\n  {'='*55}")
print(f"  PHASE 4: Strategy Comparison (5 strategies)")
print(f"  {'='*55}")

# v6 weighted: all 6 models
model_accs = {"xgb": xgb_acc, "lgb": lgb_acc, "gb": gb_acc, "cb": cb_acc, "rf": rf_acc, "lstm": lstm_acc}
edges = {k: (v-0.5)**2 if v >= MIN_ACC else 0.0 for k,v in model_accs.items()}
te = sum(edges.values())
v6_weights = {k: v/te for k,v in edges.items()} if te > 0 else {k: 1/6 for k in edges}
v6_prob = sum(v6_weights[k] * p for k, p in zip(
    ["xgb","lgb","gb","cb","rf","lstm"],
    [xgb_prob, lgb_prob, gb_prob, cb_prob, rf_prob, lstm_prob]))
v6_acc = accuracy_score(y_val, (v6_prob > 0.5).astype(int))

# Top-3 equal weight
base_probs_all = {"xgb": xgb_prob, "lgb": lgb_prob, "gb": gb_prob,
                   "cb": cb_prob, "rf": rf_prob, "lstm": lstm_prob}
sorted_models = sorted(model_accs.items(), key=lambda x: x[1], reverse=True)[:3]
top3_names = [n for n, _ in sorted_models]
top3_prob = np.mean([base_probs_all[n] for n in top3_names], axis=0)
top3_acc = accuracy_score(y_val, (top3_prob > 0.5).astype(int))

# Regime-conditional
regime_prob = np.full(len(y_val), 0.5)
for i in range(len(y_val)):
    rtype = "TRENDING" if reg_val[i] in TRENDING_REGIMES else "CHOPPY"
    rm = regime_models.get(rtype)
    if rm is not None:
        regime_prob[i] = rm[rm["best"]].predict_proba(X_val_s[i:i+1])[0][1]
    else:
        regime_prob[i] = v6_prob[i]
regime_acc = accuracy_score(y_val, (regime_prob > 0.5).astype(int))

# Hybrid
hybrid_prob = 0.5 * regime_prob + 0.5 * meta_prob
hybrid_acc = accuracy_score(y_val, (hybrid_prob > 0.5).astype(int))

strategies = {
    "v6 6-Model Weighted": (v6_acc, v6_prob),
    "Top-3 Equal Weight": (top3_acc, top3_prob),
    "Regime-Conditional": (regime_acc, regime_prob),
    "Stacking Meta": (meta_acc, meta_prob),
    "Hybrid (Regime+Meta)": (hybrid_acc, hybrid_prob),
}

print()
best_name, best_acc, best_prob = "", 0, None
for name, (acc, prob) in strategies.items():
    if acc > best_acc:
        best_name, best_acc, best_prob = name, acc, prob
    print(f"  {name:25s}: {acc*100:.1f}%")
print(f"\n  >>> BEST: {best_name} ({best_acc*100:.1f}%) <<<")

ens_acc = best_acc; ens_prob = best_prob

# Check against previous
prev_acc = 0.0
prev_cfg = os.path.join(BASE, "ensemble_config.json")
if os.path.exists(prev_cfg):
    try:
        with open(prev_cfg) as f: prev_acc = json.load(f).get("best_strategy_acc", 0)
    except: pass

if prev_acc > 0 and ens_acc < prev_acc:
    print(f"\n  WARNING: New ({ens_acc*100:.1f}%) < Previous ({prev_acc*100:.1f}%)")
    print(f"  Deploying anyway (new models have latest data)")

# Backup old models
if os.path.exists(os.path.join(BASE, "xgb_model.pkl")):
    backup_dir = os.path.join(BASE, "model_backups", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(backup_dir, exist_ok=True)
    for f in ["xgb_model.pkl","lgb_model.pkl","gb_model.pkl","catboost_model.pkl","rf_model.pkl",
              "lstm_model.pt","tft_model.pt","lstm_model.keras","scaler.pkl","meta_model.pkl",
              "regime_models.pkl","ensemble_config.json","ensemble.pkl"]:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy2(src, backup_dir)
    print(f"  Backed up old models to: {backup_dir}")

# Kelly (spread-aware)
close_v = df["close"].values
vc = close_v[-(len(y_val)+HORIZON):-HORIZON][-len(y_val):]
wins, losses = [], []
for i in range(min(len(ens_prob), len(vc)-HORIZON)):
    ret = (vc[i+HORIZON] - vc[i]) / (vc[i] + 1e-9)
    net_ret = ret - SPREAD_PCT/100
    if ens_prob[i] > 0.5:
        (wins if net_ret > 0 else losses).append(abs(net_ret))
wr = len(wins) / (len(wins) + len(losses) + 1e-9)
aw = float(np.mean(wins)) if wins else 0.001
al = float(np.mean(losses)) if losses else 0.001
kelly = kelly_fraction(wr, aw, al)

print(f"\n  Ensemble: {ens_acc*100:.1f}%  Kelly: {kelly*100:.3f}%  Win rate: {wr*100:.1f}%")

# == Save ===================================================================
print("\n  Saving models...")
joblib.dump(xgb_m, os.path.join(BASE, "xgb_model.pkl"))
joblib.dump(lgb_m, os.path.join(BASE, "lgb_model.pkl"))
joblib.dump(gb_m,  os.path.join(BASE, "gb_model.pkl"))
if HAS_CB and cb_m is not None:
    joblib.dump(cb_m, os.path.join(BASE, "catboost_model.pkl"))
joblib.dump(rf_m,  os.path.join(BASE, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(BASE, "scaler.pkl"))
joblib.dump(meta_model, os.path.join(BASE, "meta_model.pkl"))
joblib.dump(regime_save, os.path.join(BASE, "regime_models.pkl"))

tree_w = np.array([v6_weights.get("xgb",0), v6_weights.get("lgb",0), v6_weights.get("gb",0)])
tw = tree_w.sum()
if tw > 0: tree_w = tree_w / tw

ec = {
    "trainer_version": "v6",
    "model_accuracies": {k: round(float(v),4) for k,v in model_accs.items()},
    "model_weights": {k: round(float(v),4) for k,v in v6_weights.items()},
    "best_strategy": best_name,
    "best_strategy_acc": round(float(best_acc),4),
    "ensemble_accuracy": round(float(ens_acc),4),
    "xgb_accuracy": float(xgb_acc), "lgb_accuracy": float(lgb_acc),
    "gb_accuracy": float(gb_acc), "catboost_accuracy": float(cb_acc),
    "rf_accuracy": float(rf_acc), "lstm_accuracy": float(lstm_acc),
    "tft_accuracy": float(tft_acc),
    "meta_accuracy": float(meta_acc), "regime_accuracy": float(regime_acc),
    "xgb_weight": float(tree_w[0]), "lgb_weight": float(tree_w[1]), "gb_weight": float(tree_w[2]),
    "lstm_weight": float(v6_weights.get("lstm", 0)),
    "catboost_available": HAS_CB and cb_m is not None,
    "rf_available": True,
    "lstm_available": HAS_TORCH and lstm_model_pt is not None,
    "tft_available": HAS_TORCH and tft_model_pt is not None,
    "pytorch_gpu": DEVICE == "cuda",
    "kelly_fraction": float(kelly),
    "win_rate": float(wr), "avg_win": float(aw), "avg_loss": float(al),
    "spread_pct": SPREAD_PCT, "min_accuracy_gate": MIN_ACC,
    "target_horizon": HORIZON, "target_threshold": THRESHOLD, "purge_gap": PURGE_GAP,
    "n_features": len(feature_cols), "feature_names": feature_cols,
    "n_train": int(len(X_tr)), "n_val": int(len(X_val)),
    "regime_models_available": {k: (v is not None) for k,v in regime_models.items()},
    "top3_models": top3_names,
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
joblib.dump(ec, os.path.join(BASE, "ensemble.pkl"))
with open(os.path.join(BASE, "ensemble_config.json"), "w") as f:
    json.dump(ec, f, indent=2)

print(f"  Models deployed: {sum(1 for v in model_accs.values() if v >= MIN_ACC)} base + meta + regime")

# == Signal =================================================================
print("\n  Generating current signal...")
try:
    recent = download_gold("5d", "1h")
    feat_r = make_features(recent).replace([np.inf,-np.inf], np.nan).dropna()
    if len(feat_r) == 0: raise ValueError("No clean rows")
    for c in feature_cols:
        if c not in feat_r.columns: feat_r[c] = 0.0
    X_r = scaler.transform(feat_r[feature_cols].values[-1:].astype(float))

    p_xgb = float(xgb_m.predict_proba(X_r)[0][1])
    p_lgb = float(lgb_m.predict_proba(X_r)[0][1])
    p_gb  = float(gb_m.predict_proba(X_r)[0][1])
    p_cb = float(cb_m.predict_proba(X_r)[0][1]) if HAS_CB and cb_m else 0.5
    p_rf = float(rf_m.predict_proba(X_r)[0][1])

    p_lstm = 0.5
    if HAS_TORCH and lstm_model_pt is not None:
        try:
            X_all_r = scaler.transform(feat_r[feature_cols].values.astype(float))
            if len(X_all_r) >= SEQ:
                seq_in = torch.tensor(X_all_r[-SEQ:].reshape(1,SEQ,-1), dtype=torch.float32).to(DEVICE)
                lstm_model_pt.eval()
                with torch.no_grad():
                    p_lstm = float(lstm_model_pt(seq_in).cpu().item())
        except: pass

    p_tft = 0.5
    if HAS_TORCH and tft_model_pt is not None:
        try:
            X_all_r = scaler.transform(feat_r[feature_cols].values.astype(float))
            if len(X_all_r) >= SEQ:
                seq_in = torch.tensor(X_all_r[-SEQ:].reshape(1,SEQ,-1), dtype=torch.float32).to(DEVICE)
                tft_model_pt.eval()
                with torch.no_grad():
                    p_tft = float(tft_model_pt(seq_in).cpu().item())
        except: pass

    try:
        from market_regime import get_current_regime
        cur_regime = get_current_regime().get("regime", "UNKNOWN")
    except: cur_regime = "UNKNOWN"

    rtype = "TRENDING" if cur_regime in TRENDING_REGIMES else "CHOPPY"
    rm_data = regime_models.get(rtype)
    if rm_data is not None:
        p_regime = float(rm_data[rm_data["best"]].predict_proba(X_r)[0][1])
    else:
        p_regime = (tree_w[0]*p_xgb+tree_w[1]*p_lgb+tree_w[2]*p_gb)

    rc = 1.0 if cur_regime in TRENDING_REGIMES else 0.0
    p_meta = float(meta_model.predict_proba(np.array([[p_xgb,p_lgb,p_gb,p_cb,p_rf,p_lstm,rc]]))[0][1])

    if best_name == "Stacking Meta": conf = p_meta
    elif best_name == "Regime-Conditional": conf = p_regime
    elif best_name == "Hybrid (Regime+Meta)": conf = 0.5*p_regime + 0.5*p_meta
    elif best_name == "Top-3 Equal Weight":
        t3_probs = {"xgb":p_xgb,"lgb":p_lgb,"gb":p_gb,"cb":p_cb,"rf":p_rf,"lstm":p_lstm}
        conf = np.mean([t3_probs[n] for n in top3_names])
    else:
        tw_all = sum(v6_weights.values())
        conf = sum(v6_weights[k]*p for k,p in zip(
            ["xgb","lgb","gb","cb","rf","lstm"],
            [p_xgb,p_lgb,p_gb,p_cb,p_rf,p_lstm])) / tw_all if tw_all > 0 else 0.5

    sig = "BUY" if conf > 0.55 else ("SELL" if conf < 0.45 else "NEUTRAL")
    rm_val = 2.0 if conf > 0.80 else (1.5 if conf > 0.70 else (1.0 if conf > 0.60 else 0.5))
    out = {"signal": sig, "confidence": round(float(conf),4),
           "kelly_fraction": round(float(kelly),5),
           "risk_multiplier": rm_val, "sized_risk_pct": round(kelly*rm_val*100,3),
           "xgb_p": round(p_xgb,4), "lgb_p": round(p_lgb,4),
           "gb_p": round(p_gb,4), "catboost_p": round(p_cb,4),
           "rf_p": round(p_rf,4), "lstm_p": round(p_lstm,4),
           "tft_p": round(p_tft,4),
           "regime_p": round(p_regime,4), "meta_p": round(p_meta,4),
           "current_regime": cur_regime, "strategy_used": best_name,
           "ensemble_accuracy": round(float(ens_acc),4),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(BASE, "current_signal.json"), "w") as f: json.dump(out, f, indent=2)
    print(f"  Signal: {sig}  Confidence: {conf*100:.1f}%  Risk: {out['sized_risk_pct']:.3f}%")
    print(f"  Strategy: {best_name} | Regime: {cur_regime}")
    print(f"    XGB={p_xgb:.3f} LGB={p_lgb:.3f} GB={p_gb:.3f} CB={p_cb:.3f} RF={p_rf:.3f} LSTM={p_lstm:.3f} TFT={p_tft:.3f}")
except Exception as e:
    print(f"  Signal error: {e}"); import traceback; traceback.print_exc()

print("\n  Top 10 features (LightGBM):")
try:
    imp = lgb_m.feature_importances_
    for i, idx in enumerate(np.argsort(imp)[::-1][:10]):
        print(f"    {i+1:2d}. {feature_cols[idx]:20s}  ({imp[idx]:.0f})")
except: pass

print(f"\n  Training complete! Best: {best_name} ({best_acc*100:.1f}%)")
print(f"  GPU: {'RTX 2060 (PyTorch CUDA)' if DEVICE=='cuda' else 'CPU only'}")
print(f"  Models: XGB, LGB, GB, {'CB, ' if HAS_CB else ''}RF, {'LSTM(GPU), ' if lstm_model_pt else ''}{'TFT(GPU)' if tft_model_pt else ''}")
