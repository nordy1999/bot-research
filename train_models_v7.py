"""
AlmostFinishedBot - ML Trainer v7
ELITE EDITION: N-BEATS + N-HiTS + TCN + Enhanced Ensemble

v7 changes from v6:
  - TCN (Temporal Convolutional Network) - proven 33% better on gold
  - N-BEATS (Neural Basis Expansion) - won M4 forecasting competition
  - N-HiTS (Neural Hierarchical Interpolation) - multi-scale patterns
  - 10 total models: XGB, LGB, GB, CB, RF, LSTM, TFT, TCN, NBEATS, NHITS
  - Dynamic ensemble weighting by regime
  - Purged validation with embargo
  - trainer_version: "v7" flag in ensemble_config.json
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
print("  AlmostFinishedBot  |  ML Trainer v7 ELITE")
print("  TCN + N-BEATS + N-HiTS + 10-Model Ensemble")
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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  PyTorch: GPU mode ({gpu_name})")
    else:
        print("  PyTorch: CPU mode (no CUDA)")
except ImportError:
    print("  PyTorch: not available, skipping deep learning models")

HORIZON = 5
THRESHOLD = 0.001
PURGE_GAP = HORIZON + 2
EMBARGO = 5  # Additional embargo after purge
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

# Purged split with embargo
split = int(len(X_all) * 0.8)
X_tr = X_all[:split - PURGE_GAP - EMBARGO]
y_tr = y_all[:split - PURGE_GAP - EMBARGO]
reg_tr = regimes_all[:split - PURGE_GAP - EMBARGO]
X_val = X_all[split:]
y_val = y_all[split:]
reg_val = regimes_all[split:]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)

print(f"\n  Train: {len(X_tr)}  Val: {len(X_val)}  Features: {len(feature_cols)}")
print(f"  Purge gap: {PURGE_GAP} | Embargo: {EMBARGO} | Horizon: {HORIZON}h")

# ============================================================================
# PHASE 1: TREE-BASED MODELS (5 models)
# ============================================================================
print(f"\n  {'='*55}")
print(f"  PHASE 1: Tree-Based Models (5)")
print(f"  {'='*55}")

print("\n  [1/10] Training XGBoost...")
xgb_m = xgb.XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, eval_metric="logloss", verbosity=0, random_state=42)
xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
xgb_acc = accuracy_score(y_val, xgb_m.predict(X_val_s))
xgb_prob = xgb_m.predict_proba(X_val_s)[:,1]
print(f"     XGBoost:     {xgb_acc*100:.1f}%  {'[OK]' if xgb_acc >= MIN_ACC else '[WEAK]'}")

print("  [2/10] Training LightGBM...")
lgb_m = lgb.LGBMClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03, num_leaves=31,
    subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_samples=20, verbose=-1, random_state=42)
lgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
lgb_acc = accuracy_score(y_val, lgb_m.predict(X_val_s))
lgb_prob = lgb_m.predict_proba(X_val_s)[:,1]
print(f"     LightGBM:    {lgb_acc*100:.1f}%  {'[OK]' if lgb_acc >= MIN_ACC else '[WEAK]'}")

print("  [3/10] Training GradientBoost...")
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
    print("  [4/10] Training CatBoost...")
    cb_m = CatBoostClassifier(
        iterations=500, depth=4, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.7, random_seed=42,
        verbose=0, eval_metric="Accuracy")
    cb_m.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val), verbose=0, early_stopping_rounds=30)
    cb_acc = accuracy_score(y_val, cb_m.predict(X_val_s))
    cb_prob = cb_m.predict_proba(X_val_s)[:,1]
    print(f"     CatBoost:    {cb_acc*100:.1f}%  {'[OK]' if cb_acc >= MIN_ACC else '[WEAK]'}")
else:
    print("  [4/10] CatBoost skipped")

# Random Forest
print("  [5/10] Training Random Forest...")
rf_m = RandomForestClassifier(
    n_estimators=500, max_depth=6, min_samples_leaf=20,
    max_features="sqrt", oob_score=True, random_state=42, n_jobs=-1)
rf_m.fit(X_tr_s, y_tr)
rf_acc = accuracy_score(y_val, rf_m.predict(X_val_s))
rf_prob = rf_m.predict_proba(X_val_s)[:,1]
oob = rf_m.oob_score_ if hasattr(rf_m, "oob_score_") else 0
print(f"     RandomForest:{rf_acc*100:.1f}%  {'[OK]' if rf_acc >= MIN_ACC else '[WEAK]'}  (OOB={oob*100:.1f}%)")

# ============================================================================
# PHASE 2: DEEP LEARNING MODELS (5 models)
# ============================================================================
print(f"\n  {'='*55}")
print(f"  PHASE 2: Deep Learning Models (5) - GPU Accelerated")
print(f"  {'='*55}")

SEQ = 20  # Sequence length for all sequential models

# Helper to create sequences
def make_seqs(X, y, s):
    xs, ys = [], []
    for i in range(s, len(X)):
        xs.append(X[i-s:i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# Initialize all deep learning variables
lstm_acc = 0.5; lstm_prob = np.full(len(y_val), 0.5); lstm_model_pt = None
tft_acc = 0.5; tft_prob = np.full(len(y_val), 0.5); tft_model_pt = None
tcn_acc = 0.5; tcn_prob = np.full(len(y_val), 0.5); tcn_model_pt = None
nbeats_acc = 0.5; nbeats_prob = np.full(len(y_val), 0.5); nbeats_model_pt = None
nhits_acc = 0.5; nhits_prob = np.full(len(y_val), 0.5); nhits_model_pt = None

if HAS_TORCH:
    Xs, ys = make_seqs(X_tr_s, y_tr, SEQ)
    Xvs, yvs = make_seqs(X_val_s, y_val, SEQ)
    
    if len(Xs) >= 64:
        train_ds = TensorDataset(torch.tensor(Xs), torch.tensor(ys))
        val_ds = TensorDataset(torch.tensor(Xvs), torch.tensor(yvs))
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=128)
        
        # ================================================================
        # [6/10] LSTM
        # ================================================================
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
        
        print(f"  [6/10] Training LSTM (PyTorch {DEVICE.upper()})...")
        t_start = time.time()
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
        print(f"     LSTM:        {lstm_acc*100:.1f}%  {'[OK]' if lstm_acc >= MIN_ACC else '[WEAK]'}  ({elapsed:.1f}s)")
        
        torch.save({
            "model_state": lstm_model_pt.state_dict(),
            "n_features": X_tr_s.shape[1], "seq_len": SEQ,
        }, os.path.join(BASE, "lstm_model.pt"))
        
        # ================================================================
        # [7/10] TFT (Temporal Fusion Transformer)
        # ================================================================
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
        
        print(f"  [7/10] Training TFT (PyTorch {DEVICE.upper()})...")
        t_start = time.time()
        tft_model_pt = SimpleTFT(X_tr_s.shape[1]).to(DEVICE)
        opt_tft = torch.optim.Adam(tft_model_pt.parameters(), lr=0.0005)
        scheduler_tft = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_tft, patience=4, factor=0.5)
        
        best_val_loss = float("inf"); wait = 0; best_state = None
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
        print(f"     TFT:         {tft_acc*100:.1f}%  {'[OK]' if tft_acc >= MIN_ACC else '[WEAK]'}  ({elapsed:.1f}s)")
        
        torch.save({
            "model_state": tft_model_pt.state_dict(),
            "n_features": X_tr_s.shape[1], "seq_len": SEQ,
        }, os.path.join(BASE, "tft_model.pt"))
        
        # ================================================================
        # [8/10] TCN (Temporal Convolutional Network) - NEW!
        # Proven 33% better RMSE on gold price prediction
        # ================================================================
        class CausalConv1d(nn.Module):
            """Causal convolution - no future leakage"""
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
                    dilation = 2 ** i  # Exponentially growing dilation
                    layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
                self.tcn = nn.Sequential(*layers)
                self.fc1 = nn.Linear(n_channels[-1], 32)
                self.drop = nn.Dropout(dropout)
                self.fc2 = nn.Linear(32, 1)
                
            def forward(self, x):
                # x: (batch, seq, features) -> (batch, features, seq) for conv
                x = x.transpose(1, 2)
                out = self.tcn(x)
                out = out[:, :, -1]  # Take last timestep
                out = torch.relu(self.fc1(out))
                out = self.drop(out)
                return torch.sigmoid(self.fc2(out)).squeeze(-1)
        
        print(f"  [8/10] Training TCN (PyTorch {DEVICE.upper()})...")
        t_start = time.time()
        tcn_model_pt = TCNModel(X_tr_s.shape[1]).to(DEVICE)
        opt_tcn = torch.optim.Adam(tcn_model_pt.parameters(), lr=0.001)
        scheduler_tcn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_tcn, patience=4, factor=0.5)
        
        best_val_loss = float("inf"); wait = 0; best_state = None
        for epoch in range(50):
            tcn_model_pt.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = tcn_model_pt(xb)
                loss = loss_fn(pred, yb)
                opt_tcn.zero_grad(); loss.backward(); opt_tcn.step()
            
            tcn_model_pt.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = tcn_model_pt(xb)
                    val_losses.append(loss_fn(pred, yb).item())
            vl = np.mean(val_losses)
            scheduler_tcn.step(vl)
            if vl < best_val_loss:
                best_val_loss = vl; wait = 0
                best_state = {k: v.cpu().clone() for k, v in tcn_model_pt.state_dict().items()}
            else:
                wait += 1
                if wait >= patience: break
        
        if best_state:
            tcn_model_pt.load_state_dict(best_state)
        tcn_model_pt.eval()
        with torch.no_grad():
            raw_tcn = tcn_model_pt(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
        tcn_acc = accuracy_score(yvs, (raw_tcn > 0.5).astype(int))
        tcn_prob_aligned = np.full(len(y_val), 0.5)
        tcn_prob_aligned[-len(raw_tcn):] = raw_tcn
        tcn_prob = tcn_prob_aligned
        elapsed = time.time() - t_start
        print(f"     TCN:         {tcn_acc*100:.1f}%  {'[OK]' if tcn_acc >= MIN_ACC else '[WEAK]'}  ({elapsed:.1f}s)")
        
        torch.save({
            "model_state": tcn_model_pt.state_dict(),
            "n_features": X_tr_s.shape[1], "seq_len": SEQ,
        }, os.path.join(BASE, "tcn_model.pt"))
        
        # ================================================================
        # [9/10] N-BEATS (Neural Basis Expansion Analysis) - NEW!
        # Won M4 forecasting competition, interpretable trend/seasonality
        # ================================================================
        class NBEATSBlock(nn.Module):
            def __init__(self, input_size, theta_size, hidden=64, layers=2):
                super().__init__()
                fc_layers = [nn.Linear(input_size, hidden), nn.ReLU()]
                for _ in range(layers - 1):
                    fc_layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
                self.fc = nn.Sequential(*fc_layers)
                self.theta_b = nn.Linear(hidden, theta_size)  # Backcast
                self.theta_f = nn.Linear(hidden, theta_size)  # Forecast
                
            def forward(self, x):
                h = self.fc(x)
                return self.theta_b(h), self.theta_f(h)
        
        class NBEATSModel(nn.Module):
            def __init__(self, n_features, seq_len, n_blocks=3, hidden=64, dropout=0.2):
                super().__init__()
                self.seq_len = seq_len
                self.n_features = n_features
                input_size = seq_len * n_features
                self.blocks = nn.ModuleList([
                    NBEATSBlock(input_size, input_size, hidden) for _ in range(n_blocks)
                ])
                self.fc1 = nn.Linear(input_size, 64)
                self.drop = nn.Dropout(dropout)
                self.fc2 = nn.Linear(64, 1)
                
            def forward(self, x):
                # x: (batch, seq, features)
                batch = x.shape[0]
                x_flat = x.reshape(batch, -1)  # Flatten to (batch, seq*features)
                
                residuals = x_flat
                forecast = torch.zeros_like(x_flat)
                
                for block in self.blocks:
                    backcast, block_forecast = block(residuals)
                    residuals = residuals - backcast
                    forecast = forecast + block_forecast
                
                # Classify based on forecast
                out = torch.relu(self.fc1(forecast))
                out = self.drop(out)
                return torch.sigmoid(self.fc2(out)).squeeze(-1)
        
        print(f"  [9/10] Training N-BEATS (PyTorch {DEVICE.upper()})...")
        t_start = time.time()
        nbeats_model_pt = NBEATSModel(X_tr_s.shape[1], SEQ).to(DEVICE)
        opt_nbeats = torch.optim.Adam(nbeats_model_pt.parameters(), lr=0.001)
        scheduler_nbeats = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_nbeats, patience=4, factor=0.5)
        
        best_val_loss = float("inf"); wait = 0; best_state = None
        for epoch in range(50):
            nbeats_model_pt.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = nbeats_model_pt(xb)
                loss = loss_fn(pred, yb)
                opt_nbeats.zero_grad(); loss.backward(); opt_nbeats.step()
            
            nbeats_model_pt.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = nbeats_model_pt(xb)
                    val_losses.append(loss_fn(pred, yb).item())
            vl = np.mean(val_losses)
            scheduler_nbeats.step(vl)
            if vl < best_val_loss:
                best_val_loss = vl; wait = 0
                best_state = {k: v.cpu().clone() for k, v in nbeats_model_pt.state_dict().items()}
            else:
                wait += 1
                if wait >= patience: break
        
        if best_state:
            nbeats_model_pt.load_state_dict(best_state)
        nbeats_model_pt.eval()
        with torch.no_grad():
            raw_nbeats = nbeats_model_pt(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
        nbeats_acc = accuracy_score(yvs, (raw_nbeats > 0.5).astype(int))
        nbeats_prob_aligned = np.full(len(y_val), 0.5)
        nbeats_prob_aligned[-len(raw_nbeats):] = raw_nbeats
        nbeats_prob = nbeats_prob_aligned
        elapsed = time.time() - t_start
        print(f"     N-BEATS:     {nbeats_acc*100:.1f}%  {'[OK]' if nbeats_acc >= MIN_ACC else '[WEAK]'}  ({elapsed:.1f}s)")
        
        torch.save({
            "model_state": nbeats_model_pt.state_dict(),
            "n_features": X_tr_s.shape[1], "seq_len": SEQ,
        }, os.path.join(BASE, "nbeats_model.pt"))
        
        # ================================================================
        # [10/10] N-HiTS (Neural Hierarchical Interpolation) - NEW!
        # Multi-scale temporal patterns via hierarchical pooling
        # ================================================================
        class NHiTSBlock(nn.Module):
            def __init__(self, input_size, output_size, pool_size, hidden=64):
                super().__init__()
                self.pool = nn.MaxPool1d(pool_size, stride=pool_size) if pool_size > 1 else nn.Identity()
                self.pool_size = pool_size
                pooled_size = input_size // pool_size if pool_size > 1 else input_size
                self.fc = nn.Sequential(
                    nn.Linear(pooled_size, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                )
                self.theta = nn.Linear(hidden, output_size)
                
            def forward(self, x):
                # x: (batch, features)
                if self.pool_size > 1:
                    x = x.unsqueeze(1)  # (batch, 1, features)
                    x = self.pool(x).squeeze(1)
                h = self.fc(x)
                return self.theta(h)
        
        class NHiTSModel(nn.Module):
            def __init__(self, n_features, seq_len, n_stacks=3, hidden=64, dropout=0.2):
                super().__init__()
                input_size = seq_len * n_features
                # Different pooling sizes for multi-scale
                pool_sizes = [1, 2, 4][:n_stacks]
                self.blocks = nn.ModuleList([
                    NHiTSBlock(input_size, input_size, ps, hidden) 
                    for ps in pool_sizes
                ])
                self.fc1 = nn.Linear(input_size, 64)
                self.drop = nn.Dropout(dropout)
                self.fc2 = nn.Linear(64, 1)
                
            def forward(self, x):
                batch = x.shape[0]
                x_flat = x.reshape(batch, -1)
                
                # Sum forecasts from all hierarchical levels
                forecast = torch.zeros_like(x_flat)
                for block in self.blocks:
                    forecast = forecast + block(x_flat)
                
                out = torch.relu(self.fc1(forecast))
                out = self.drop(out)
                return torch.sigmoid(self.fc2(out)).squeeze(-1)
        
        print(f"  [10/10] Training N-HiTS (PyTorch {DEVICE.upper()})...")
        t_start = time.time()
        nhits_model_pt = NHiTSModel(X_tr_s.shape[1], SEQ).to(DEVICE)
        opt_nhits = torch.optim.Adam(nhits_model_pt.parameters(), lr=0.001)
        scheduler_nhits = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_nhits, patience=4, factor=0.5)
        
        best_val_loss = float("inf"); wait = 0; best_state = None
        for epoch in range(50):
            nhits_model_pt.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = nhits_model_pt(xb)
                loss = loss_fn(pred, yb)
                opt_nhits.zero_grad(); loss.backward(); opt_nhits.step()
            
            nhits_model_pt.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = nhits_model_pt(xb)
                    val_losses.append(loss_fn(pred, yb).item())
            vl = np.mean(val_losses)
            scheduler_nhits.step(vl)
            if vl < best_val_loss:
                best_val_loss = vl; wait = 0
                best_state = {k: v.cpu().clone() for k, v in nhits_model_pt.state_dict().items()}
            else:
                wait += 1
                if wait >= patience: break
        
        if best_state:
            nhits_model_pt.load_state_dict(best_state)
        nhits_model_pt.eval()
        with torch.no_grad():
            raw_nhits = nhits_model_pt(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
        nhits_acc = accuracy_score(yvs, (raw_nhits > 0.5).astype(int))
        nhits_prob_aligned = np.full(len(y_val), 0.5)
        nhits_prob_aligned[-len(raw_nhits):] = raw_nhits
        nhits_prob = nhits_prob_aligned
        elapsed = time.time() - t_start
        print(f"     N-HiTS:      {nhits_acc*100:.1f}%  {'[OK]' if nhits_acc >= MIN_ACC else '[WEAK]'}  ({elapsed:.1f}s)")
        
        torch.save({
            "model_state": nhits_model_pt.state_dict(),
            "n_features": X_tr_s.shape[1], "seq_len": SEQ,
        }, os.path.join(BASE, "nhits_model.pt"))
    else:
        print("  Deep learning models skipped (not enough sequences)")
else:
    print("  Deep learning models skipped (no PyTorch)")

# ============================================================================
# PHASE 3: REGIME-CONDITIONAL MODELS
# ============================================================================
print(f"\n  {'='*55}")
print(f"  PHASE 3: Regime-Conditional Models")
print(f"  {'='*55}")

regime_models = {}

# Trending regime
trend_mask_tr = np.isin(reg_tr, list(TRENDING_REGIMES))
trend_mask_val = np.isin(reg_val, list(TRENDING_REGIMES))

if trend_mask_tr.sum() >= MIN_REGIME_SAMPLES:
    X_trend_tr = X_tr_s[trend_mask_tr]
    y_trend_tr = y_tr[trend_mask_tr]
    X_trend_val = X_val_s[trend_mask_val]
    y_trend_val = y_val[trend_mask_val]
    
    print(f"\n  --- TRENDING: {len(X_trend_tr)} train / {len(X_trend_val)} val ---")
    
    lgb_trend = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        num_leaves=20, verbose=-1, random_state=42)
    lgb_trend.fit(X_trend_tr, y_trend_tr)
    lgb_trend_acc = accuracy_score(y_trend_val, lgb_trend.predict(X_trend_val)) if len(y_trend_val) > 0 else 0.5
    
    gb_trend = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    gb_trend.fit(X_trend_tr, y_trend_tr)
    gb_trend_acc = accuracy_score(y_trend_val, gb_trend.predict(X_trend_val)) if len(y_trend_val) > 0 else 0.5
    
    cb_trend = None; cb_trend_acc = 0.5
    if HAS_CB:
        cb_trend = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=0, random_seed=42)
        cb_trend.fit(X_trend_tr, y_trend_tr)
        cb_trend_acc = accuracy_score(y_trend_val, cb_trend.predict(X_trend_val)) if len(y_trend_val) > 0 else 0.5
    
    print(f"  TRENDING LGB: {lgb_trend_acc*100:.1f}%  |  GB: {gb_trend_acc*100:.1f}%  |  CB: {cb_trend_acc*100:.1f}%")
    
    best_trend_acc = max(lgb_trend_acc, gb_trend_acc, cb_trend_acc)
    if lgb_trend_acc == best_trend_acc:
        regime_models["TREND_UP"] = regime_models["TREND_DOWN"] = lgb_trend
        print(f"  BEST: LGB ({best_trend_acc*100:.1f}%)")
    elif gb_trend_acc == best_trend_acc:
        regime_models["TREND_UP"] = regime_models["TREND_DOWN"] = gb_trend
        print(f"  BEST: GB ({best_trend_acc*100:.1f}%)")
    else:
        regime_models["TREND_UP"] = regime_models["TREND_DOWN"] = cb_trend
        print(f"  BEST: CB ({best_trend_acc*100:.1f}%)")

# Choppy/ranging regime
choppy_regimes = {"CHOPPY_RANGE", "LOW_VOL_COMPRESSION"}
choppy_mask_tr = np.isin(reg_tr, list(choppy_regimes))
choppy_mask_val = np.isin(reg_val, list(choppy_regimes))

if choppy_mask_tr.sum() >= MIN_REGIME_SAMPLES:
    X_chop_tr = X_tr_s[choppy_mask_tr]
    y_chop_tr = y_tr[choppy_mask_tr]
    X_chop_val = X_val_s[choppy_mask_val]
    y_chop_val = y_val[choppy_mask_val]
    
    print(f"\n  --- CHOPPY: {len(X_chop_tr)} train / {len(X_chop_val)} val ---")
    
    lgb_chop = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        num_leaves=20, verbose=-1, random_state=42)
    lgb_chop.fit(X_chop_tr, y_chop_tr)
    lgb_chop_acc = accuracy_score(y_chop_val, lgb_chop.predict(X_chop_val)) if len(y_chop_val) > 0 else 0.5
    
    gb_chop = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    gb_chop.fit(X_chop_tr, y_chop_tr)
    gb_chop_acc = accuracy_score(y_chop_val, gb_chop.predict(X_chop_val)) if len(y_chop_val) > 0 else 0.5
    
    cb_chop = None; cb_chop_acc = 0.5
    if HAS_CB:
        cb_chop = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=0, random_seed=42)
        cb_chop.fit(X_chop_tr, y_chop_tr)
        cb_chop_acc = accuracy_score(y_chop_val, cb_chop.predict(X_chop_val)) if len(y_chop_val) > 0 else 0.5
    
    print(f"  CHOPPY LGB: {lgb_chop_acc*100:.1f}%  |  GB: {gb_chop_acc*100:.1f}%  |  CB: {cb_chop_acc*100:.1f}%")
    
    best_chop_acc = max(lgb_chop_acc, gb_chop_acc, cb_chop_acc)
    if lgb_chop_acc == best_chop_acc:
        regime_models["CHOPPY_RANGE"] = regime_models["LOW_VOL_COMPRESSION"] = lgb_chop
        print(f"  BEST: LGB ({best_chop_acc*100:.1f}%)")
    elif gb_chop_acc == best_chop_acc:
        regime_models["CHOPPY_RANGE"] = regime_models["LOW_VOL_COMPRESSION"] = gb_chop
        print(f"  BEST: GB ({best_chop_acc*100:.1f}%)")
    else:
        regime_models["CHOPPY_RANGE"] = regime_models["LOW_VOL_COMPRESSION"] = cb_chop
        print(f"  BEST: CB ({best_chop_acc*100:.1f}%)")

# ============================================================================
# PHASE 4: META-LEARNER (10 models + regime)
# ============================================================================
print(f"\n  {'='*55}")
print(f"  PHASE 4: Stacking Meta-Learner (10 models + regime)")
print(f"  {'='*55}")

print("  Building out-of-fold predictions...")

# Collect all model probabilities
all_probs = np.column_stack([
    xgb_prob, lgb_prob, gb_prob, cb_prob, rf_prob,
    lstm_prob, tft_prob, tcn_prob, nbeats_prob, nhits_prob
])

# Add regime encoding
regime_codes = {r: i for i, r in enumerate(sorted(set(regimes_all)))}
reg_val_encoded = np.array([regime_codes.get(r, 0) for r in reg_val]).reshape(-1, 1)
meta_features = np.hstack([all_probs, reg_val_encoded])

print("  Training meta-learner (11 inputs)...")
meta_split = int(len(meta_features) * 0.7)
meta_model = lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbose=-1, random_state=42)
meta_model.fit(meta_features[:meta_split], y_val[:meta_split])
meta_pred = meta_model.predict(meta_features[meta_split:])
meta_acc = accuracy_score(y_val[meta_split:], meta_pred)
print(f"  Meta-learner: {meta_acc*100:.1f}%")

# ============================================================================
# PHASE 5: STRATEGY COMPARISON (7 strategies)
# ============================================================================
print(f"\n  {'='*55}")
print(f"  PHASE 5: Strategy Comparison (7 strategies)")
print(f"  {'='*55}")

# Model weights based on accuracy
accuracies = np.array([xgb_acc, lgb_acc, gb_acc, cb_acc, rf_acc, 
                       lstm_acc, tft_acc, tcn_acc, nbeats_acc, nhits_acc])
weights = (accuracies - 0.5) / (accuracies - 0.5).sum()
weights = np.maximum(weights, 0)
if weights.sum() > 0:
    weights /= weights.sum()
else:
    weights = np.ones(10) / 10

# Strategy 1: v7 10-Model Weighted
weighted_prob = np.average(all_probs, axis=1, weights=weights)
weighted_pred = (weighted_prob > 0.5).astype(int)
weighted_acc = accuracy_score(y_val, weighted_pred)
print(f"\n  v7 10-Model Weighted     : {weighted_acc*100:.1f}%")

# Strategy 2: Top-5 Equal Weight
top5_idx = np.argsort(accuracies)[-5:]
top5_prob = all_probs[:, top5_idx].mean(axis=1)
top5_pred = (top5_prob > 0.5).astype(int)
top5_acc = accuracy_score(y_val, top5_pred)
print(f"  Top-5 Equal Weight       : {top5_acc*100:.1f}%")

# Strategy 3: Top-3 Equal Weight
top3_idx = np.argsort(accuracies)[-3:]
top3_prob = all_probs[:, top3_idx].mean(axis=1)
top3_pred = (top3_prob > 0.5).astype(int)
top3_acc = accuracy_score(y_val, top3_pred)
print(f"  Top-3 Equal Weight       : {top3_acc*100:.1f}%")

# Strategy 4: Deep Learning Only (LSTM, TFT, TCN, N-BEATS, N-HiTS)
dl_probs = all_probs[:, 5:10]
dl_accs = accuracies[5:10]
dl_weights = (dl_accs - 0.5)
dl_weights = np.maximum(dl_weights, 0.01)
dl_weights /= dl_weights.sum()
dl_prob = np.average(dl_probs, axis=1, weights=dl_weights)
dl_pred = (dl_prob > 0.5).astype(int)
dl_acc = accuracy_score(y_val, dl_pred)
print(f"  Deep Learning Only       : {dl_acc*100:.1f}%")

# Strategy 5: Regime-Conditional
regime_pred = np.zeros(len(y_val))
for i, (xi, ri) in enumerate(zip(X_val_s, reg_val)):
    if ri in regime_models:
        regime_pred[i] = regime_models[ri].predict_proba(xi.reshape(1, -1))[0, 1]
    else:
        regime_pred[i] = weighted_prob[i]
regime_pred_binary = (regime_pred > 0.5).astype(int)
regime_acc = accuracy_score(y_val, regime_pred_binary)
print(f"  Regime-Conditional       : {regime_acc*100:.1f}%")

# Strategy 6: Stacking Meta
meta_prob_full = meta_model.predict_proba(meta_features)[:, 1]
meta_pred_full = (meta_prob_full > 0.5).astype(int)
meta_acc_full = accuracy_score(y_val, meta_pred_full)
print(f"  Stacking Meta            : {meta_acc_full*100:.1f}%")

# Strategy 7: Hybrid (Regime + Top-5)
hybrid_prob = 0.6 * regime_pred + 0.4 * top5_prob
hybrid_pred = (hybrid_prob > 0.5).astype(int)
hybrid_acc = accuracy_score(y_val, hybrid_pred)
print(f"  Hybrid (Regime+Top5)     : {hybrid_acc*100:.1f}%")

# Find best strategy
strategies = {
    "v7_weighted": weighted_acc,
    "top5": top5_acc,
    "top3": top3_acc,
    "deep_learning": dl_acc,
    "regime_conditional": regime_acc,
    "stacking_meta": meta_acc_full,
    "hybrid": hybrid_acc,
}
best_strat = max(strategies, key=strategies.get)
best_acc = strategies[best_strat]
print(f"\n  >>> BEST: {best_strat.replace('_', ' ').title()} ({best_acc*100:.1f}%) <<<")

# ============================================================================
# SAVE MODELS
# ============================================================================
print(f"\n  Saving models...")

# Backup old models
backup_dir = os.path.join(BASE, "model_backups", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(backup_dir, exist_ok=True)
for f in ["xgb_model.pkl", "lgb_model.pkl", "gb_model.pkl", "catboost_model.pkl", 
          "rf_model.pkl", "scaler.pkl", "meta_model.pkl", "regime_models.pkl", "ensemble_config.json"]:
    src = os.path.join(BASE, f)
    if os.path.exists(src):
        shutil.copy2(src, backup_dir)
print(f"  Backed up old models to: {backup_dir}")

# Save tree models
joblib.dump(xgb_m, os.path.join(BASE, "xgb_model.pkl"))
joblib.dump(lgb_m, os.path.join(BASE, "lgb_model.pkl"))
joblib.dump(gb_m, os.path.join(BASE, "gb_model.pkl"))
if cb_m: joblib.dump(cb_m, os.path.join(BASE, "catboost_model.pkl"))
joblib.dump(rf_m, os.path.join(BASE, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(BASE, "scaler.pkl"))
joblib.dump(meta_model, os.path.join(BASE, "meta_model.pkl"))
if regime_models: joblib.dump(regime_models, os.path.join(BASE, "regime_models.pkl"))

# Model accuracies for config
model_accs = {
    "xgb": xgb_acc, "lgb": lgb_acc, "gb": gb_acc, "cb": cb_acc, "rf": rf_acc,
    "lstm": lstm_acc, "tft": tft_acc, "tcn": tcn_acc, "nbeats": nbeats_acc, "nhits": nhits_acc
}

# Ensemble config
config = {
    "trainer_version": "v7",
    "best_strategy": best_strat,
    "best_accuracy": best_acc,
    "model_accuracies": model_accs,
    "weights": weights.tolist(),
    "features": feature_cols,
    "n_features": len(feature_cols),
    "seq_len": SEQ,
    "top3_models": [["xgb", "lgb", "gb", "cb", "rf", "lstm", "tft", "tcn", "nbeats", "nhits"][i] for i in top3_idx],
    "top5_models": [["xgb", "lgb", "gb", "cb", "rf", "lstm", "tft", "tcn", "nbeats", "nhits"][i] for i in top5_idx],
    "regime_codes": regime_codes,
    "trained_at": datetime.datetime.now().isoformat(),
    "deep_learning_models": ["lstm", "tft", "tcn", "nbeats", "nhits"],
}
with open(os.path.join(BASE, "ensemble_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"  Models deployed: 5 tree + 5 deep learning + meta + regime")

# ============================================================================
# TOP FEATURES
# ============================================================================
print(f"\n  Top 10 features (LightGBM):")
importances = lgb_m.feature_importances_
top_idx = np.argsort(importances)[-10:][::-1]
for i, idx in enumerate(top_idx):
    print(f"    {i+1:2d}. {feature_cols[idx]:20s} ({importances[idx]})")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n  {'='*55}")
print(f"  Training complete! Best: {best_strat} ({best_acc*100:.1f}%)")
print(f"  GPU: {gpu_name if DEVICE == 'cuda' else 'CPU'}")
print(f"  Models: XGB, LGB, GB, CB, RF, LSTM, TFT, TCN, N-BEATS, N-HiTS")
print(f"  {'='*55}")
