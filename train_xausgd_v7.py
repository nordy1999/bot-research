"""
AlmostFinishedBot - XAUSGD ML Trainer v7
ELITE EDITION: N-BEATS + N-HiTS + TCN + 10-Model Ensemble

Same v7 stack as XAUUSD but for XAUSGD:
  - Fetches data from MT5 (not yfinance)
  - Saves all models with xausgd_ prefix
  - 10 models: XGB, LGB, GB, CB, RF, LSTM, TFT, TCN, NBEATS, NHITS
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

SYMBOL = "XAUSGD"
MODEL_PREFIX = "xausgd_"

print("=" * 65)
print(f"  AlmostFinishedBot  |  {SYMBOL} Trainer v7 ELITE")
print("  TCN + N-BEATS + N-HiTS + 10-Model Ensemble")
print("=" * 65)

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for pkg in ["xgboost", "lightgbm", "scikit-learn", "joblib", "catboost"]:
    try:
        __import__(pkg.replace("-","_").replace("scikit_learn","sklearn"))
    except ImportError:
        print(f"  Installing {pkg}..."); install(pkg)

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    HAS_CB = True; print("  CatBoost: available")
except ImportError:
    HAS_CB = False; print("  CatBoost: not available")

HAS_TORCH = False; DEVICE = "cpu"
try:
    import torch; import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"  PyTorch: GPU mode ({torch.cuda.get_device_name(0)})")
    else:
        print("  PyTorch: CPU mode")
except ImportError:
    print("  PyTorch: not available")

from features import make_features, make_target

HORIZON = 5
THRESHOLD = 0.001
PURGE_GAP = HORIZON + 2
EMBARGO = 5
MIN_ACC = 0.52
SEQ = 20

# ── Get XAUSGD data from MT5 ─────────────────────────────────────────────────
print(f"\n  Fetching {SYMBOL} data...")
try:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("  [ERR] MT5 init failed")
        sys.exit(1)
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 10000)
    mt5.shutdown()
    if rates is None or len(rates) < 500:
        print(f"  [ERR] Not enough data: {len(rates) if rates else 0}")
        sys.exit(1)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "datetime"})
    df.set_index("datetime", inplace=True)
    print(f"  [OK] MT5: {len(df)} hourly candles")
except Exception as e:
    print(f"  [ERR] {e}")
    sys.exit(1)

print(f"  Got {len(df)} candles")

# ── Features ─────────────────────────────────────────────────────────────────
print("  Engineering features...")
feat_df = make_features(df)
feat_df["target"] = make_target(df["close"].values, horizon=HORIZON, threshold=THRESHOLD)
feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

print(f"  Rows: {len(feat_df)}  Features: {len([c for c in feat_df.columns if c != 'target'])}")
print(f"  Balance: {feat_df['target'].mean():.1%} buys")

feature_cols = [c for c in feat_df.columns if c != "target"]
X_all = feat_df[feature_cols].values.astype(float)
y_all = feat_df["target"].values.astype(int)

split = int(len(X_all) * 0.8)
X_tr = X_all[:split - PURGE_GAP - EMBARGO]
y_tr = y_all[:split - PURGE_GAP - EMBARGO]
X_val = X_all[split:]
y_val = y_all[split:]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)

print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1: BASE MODELS
# ════════════════════════════════════════════════════════════════════════════
print(f"\n  {'='*55}")
print(f"  PHASE 1: Base Models")
print(f"  {'='*55}")

print("\n  [1/10] XGBoost...")
xgb_m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, verbosity=0, random_state=42)
xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
xgb_acc = accuracy_score(y_val, xgb_m.predict(X_val_s))
print(f"     XGBoost:     {xgb_acc*100:.1f}%")

print("  [2/10] LightGBM...")
lgb_m = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
    num_leaves=31, verbose=-1, random_state=42)
lgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
lgb_acc = accuracy_score(y_val, lgb_m.predict(X_val_s))
print(f"     LightGBM:    {lgb_acc*100:.1f}%")

print("  [3/10] GradientBoost...")
gb_m = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
gb_m.fit(X_tr_s, y_tr)
gb_acc = accuracy_score(y_val, gb_m.predict(X_val_s))
print(f"     GradBoost:   {gb_acc*100:.1f}%")

cb_m = None; cb_acc = 0.5
if HAS_CB:
    print("  [4/10] CatBoost...")
    cb_m = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
    cb_m.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val), verbose=0, early_stopping_rounds=30)
    cb_acc = accuracy_score(y_val, cb_m.predict(X_val_s))
    print(f"     CatBoost:    {cb_acc*100:.1f}%")

print("  [5/10] Random Forest...")
rf_m = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=20,
    oob_score=True, random_state=42, n_jobs=-1)
rf_m.fit(X_tr_s, y_tr)
rf_acc = accuracy_score(y_val, rf_m.predict(X_val_s))
oob = rf_m.oob_score_ if hasattr(rf_m, "oob_score_") else 0
print(f"     RandomForest:{rf_acc*100:.1f}%  (OOB={oob*100:.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING MODELS
# ════════════════════════════════════════════════════════════════════════════
lstm_acc = tft_acc = tcn_acc = nbeats_acc = nhits_acc = 0.5
lstm_prob = tft_prob = tcn_prob = nbeats_prob = nhits_prob = np.full(len(y_val), 0.5)

if HAS_TORCH:
    def make_seqs(X, y, s):
        xs, ys = [], []
        for i in range(s, len(X)):
            xs.append(X[i-s:i])
            ys.append(y[i])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)
    
    Xs, ys = make_seqs(X_tr_s, y_tr, SEQ)
    Xvs, yvs = make_seqs(X_val_s, y_val, SEQ)
    
    if len(Xs) >= 64:
        train_ds = TensorDataset(torch.tensor(Xs), torch.tensor(ys))
        val_ds = TensorDataset(torch.tensor(Xvs), torch.tensor(yvs))
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=128)
        loss_fn = nn.BCELoss()
        
        def train_model(model, name, epochs=50, lr=0.001):
            model = model.to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
            best_val_loss = float("inf"); best_state = None; wait = 0; patience = 8
            t_start = time.time()
            
            for epoch in range(epochs):
                model.train()
                for xb, yb in train_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_dl:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        pred = model(xb)
                        val_losses.append(loss_fn(pred, yb).item())
                vl = np.mean(val_losses)
                scheduler.step(vl)
                if vl < best_val_loss:
                    best_val_loss = vl; wait = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience: break
            
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                raw = model(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
            acc = accuracy_score(yvs, (raw > 0.5).astype(int))
            elapsed = time.time() - t_start
            print(f"     {name}:{' '*(10-len(name))}{acc*100:.1f}%  ({elapsed:.1f}s {DEVICE.upper()})")
            
            prob_aligned = np.full(len(y_val), 0.5)
            prob_aligned[-len(raw):] = raw
            return model, acc, prob_aligned
        
        # LSTM
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
        
        print(f"  [6/10] LSTM (PyTorch {DEVICE.upper()})...")
        lstm_m, lstm_acc, lstm_prob = train_model(LSTMModel(X_tr_s.shape[1]), "LSTM")
        torch.save({"model_state": lstm_m.state_dict(), "n_features": X_tr_s.shape[1], "seq_len": SEQ},
                   os.path.join(BASE, f"{MODEL_PREFIX}lstm_model.pt"))
        
        # TFT
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
        
        print(f"  [7/10] TFT (PyTorch {DEVICE.upper()})...")
        tft_m, tft_acc, tft_prob = train_model(SimpleTFT(X_tr_s.shape[1]), "TFT", lr=0.0005)
        torch.save({"model_state": tft_m.state_dict(), "n_features": X_tr_s.shape[1], "seq_len": SEQ},
                   os.path.join(BASE, f"{MODEL_PREFIX}tft_model.pt"))
        
        # TCN
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
        
        print(f"  [8/10] TCN (PyTorch {DEVICE.upper()})...")
        tcn_m, tcn_acc, tcn_prob = train_model(TCNModel(X_tr_s.shape[1]), "TCN")
        torch.save({"model_state": tcn_m.state_dict(), "n_features": X_tr_s.shape[1], "seq_len": SEQ},
                   os.path.join(BASE, f"{MODEL_PREFIX}tcn_model.pt"))
        
        # N-BEATS
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
        
        print(f"  [9/10] N-BEATS (PyTorch {DEVICE.upper()})...")
        nbeats_m, nbeats_acc, nbeats_prob = train_model(NBEATSModel(X_tr_s.shape[1]), "N-BEATS")
        torch.save({"model_state": nbeats_m.state_dict(), "n_features": X_tr_s.shape[1], "seq_len": SEQ},
                   os.path.join(BASE, f"{MODEL_PREFIX}nbeats_model.pt"))
        
        # N-HiTS
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
        
        print(f"  [10/10] N-HiTS (PyTorch {DEVICE.upper()})...")
        nhits_m, nhits_acc, nhits_prob = train_model(NHiTSModel(X_tr_s.shape[1]), "N-HiTS")
        torch.save({"model_state": nhits_m.state_dict(), "n_features": X_tr_s.shape[1], "seq_len": SEQ},
                   os.path.join(BASE, f"{MODEL_PREFIX}nhits_model.pt"))

# ════════════════════════════════════════════════════════════════════════════
# META-LEARNER
# ════════════════════════════════════════════════════════════════════════════
print(f"\n  {'='*55}")
print(f"  Meta-Learner")
print(f"  {'='*55}")

xgb_prob = xgb_m.predict_proba(X_val_s)[:,1]
lgb_prob = lgb_m.predict_proba(X_val_s)[:,1]
gb_prob = gb_m.predict_proba(X_val_s)[:,1]
cb_prob = cb_m.predict_proba(X_val_s)[:,1] if cb_m else np.full(len(y_val), 0.5)
rf_prob = rf_m.predict_proba(X_val_s)[:,1]

all_probs = np.column_stack([xgb_prob, lgb_prob, gb_prob, cb_prob, rf_prob,
                              lstm_prob, tft_prob, tcn_prob, nbeats_prob, nhits_prob])

# Add dummy regime code
reg_code = np.zeros((len(y_val), 1))
meta_features = np.hstack([all_probs, reg_code])

print("  Training meta-learner...")
meta_split = int(len(meta_features) * 0.7)
meta_model = lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbose=-1)
meta_model.fit(meta_features[:meta_split], y_val[:meta_split])
meta_pred = meta_model.predict(meta_features[meta_split:])
meta_acc = accuracy_score(y_val[meta_split:], meta_pred)
print(f"  Meta: {meta_acc*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# STRATEGY COMPARISON
# ════════════════════════════════════════════════════════════════════════════
print(f"\n  {'='*55}")
print(f"  Strategy Comparison")
print(f"  {'='*55}")

accuracies = np.array([xgb_acc, lgb_acc, gb_acc, cb_acc, rf_acc,
                        lstm_acc, tft_acc, tcn_acc, nbeats_acc, nhits_acc])
weights = (accuracies - 0.5)
weights = np.maximum(weights, 0)
if weights.sum() > 0:
    weights /= weights.sum()
else:
    weights = np.ones(10) / 10

weighted_prob = np.average(all_probs, axis=1, weights=weights)
weighted_acc = accuracy_score(y_val, (weighted_prob > 0.5).astype(int))
print(f"\n  v7 Weighted              : {weighted_acc*100:.1f}%")

top3_idx = np.argsort(accuracies)[-3:]
top3_prob = all_probs[:, top3_idx].mean(axis=1)
top3_acc = accuracy_score(y_val, (top3_prob > 0.5).astype(int))
print(f"  Top-3                    : {top3_acc*100:.1f}%")

top5_idx = np.argsort(accuracies)[-5:]
top5_prob = all_probs[:, top5_idx].mean(axis=1)
top5_acc = accuracy_score(y_val, (top5_prob > 0.5).astype(int))
print(f"  Top-5                    : {top5_acc*100:.1f}%")

meta_prob_full = meta_model.predict_proba(meta_features)[:, 1]
meta_acc_full = accuracy_score(y_val, (meta_prob_full > 0.5).astype(int))
print(f"  Meta                     : {meta_acc_full*100:.1f}%")

strategies = {"v7_weighted": weighted_acc, "top3": top3_acc, "top5": top5_acc, "stacking_meta": meta_acc_full}
best_strat = max(strategies, key=strategies.get)
best_acc = strategies[best_strat]
print(f"\n  >>> BEST: {best_strat} ({best_acc*100:.1f}%) <<<")

# ════════════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ════════════════════════════════════════════════════════════════════════════
print(f"\n  Saving models...")
joblib.dump(xgb_m, os.path.join(BASE, f"{MODEL_PREFIX}xgb_model.pkl"))
joblib.dump(lgb_m, os.path.join(BASE, f"{MODEL_PREFIX}lgb_model.pkl"))
joblib.dump(gb_m, os.path.join(BASE, f"{MODEL_PREFIX}gb_model.pkl"))
if cb_m: joblib.dump(cb_m, os.path.join(BASE, f"{MODEL_PREFIX}catboost_model.pkl"))
joblib.dump(rf_m, os.path.join(BASE, f"{MODEL_PREFIX}rf_model.pkl"))
joblib.dump(scaler, os.path.join(BASE, f"{MODEL_PREFIX}scaler.pkl"))
joblib.dump(meta_model, os.path.join(BASE, f"{MODEL_PREFIX}meta_model.pkl"))

model_accs = {"xgb": xgb_acc, "lgb": lgb_acc, "gb": gb_acc, "cb": cb_acc, "rf": rf_acc,
              "lstm": lstm_acc, "tft": tft_acc, "tcn": tcn_acc, "nbeats": nbeats_acc, "nhits": nhits_acc}

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
    "regime_codes": {},
    "trained_at": datetime.datetime.now().isoformat(),
}
with open(os.path.join(BASE, f"{MODEL_PREFIX}ensemble_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"  All saved with prefix: {MODEL_PREFIX}")

print(f"\n  {'='*55}")
print(f"  {SYMBOL} Training complete! Best: {best_strat} ({best_acc*100:.1f}%)")
print(f"  GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
print(f"  {'='*55}")
