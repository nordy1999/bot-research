"""
AlmostFinishedBot - ML Trainer v2 (fixed)
Tier 1: Market regime detection (ADX + BBW)
Tier 2: XGBoost + LightGBM + GradBoost + LSTM ensemble
Tier 3: Kelly Criterion + confidence-based sizing
"""
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
os.makedirs(BASE, exist_ok=True)

print("=" * 65)
print("  AlmostFinishedBot  |  ML Trainer v2")
print("  XGBoost + LightGBM + GradBoost + LSTM  |  Regime-Aware")
print("=" * 65)

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for pkg in ["xgboost", "lightgbm", "scikit-learn", "yfinance", "joblib"]:
    try:
        __import__(pkg.replace("-","_").replace("scikit_learn","sklearn"))
    except ImportError:
        print(f"  Installing {pkg}..."); install(pkg)

import yfinance as yf
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

HAS_TF = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
    print("  TensorFlow/LSTM: available")
except Exception:
    print("  TensorFlow: not available, skipping LSTM")

# ── All indicators return plain numpy arrays (no index issues) ────
def ema_np(arr, span):
    out = np.full(len(arr), np.nan); k = 2.0 / (span + 1)
    start = 0
    while start < len(arr) and np.isnan(arr[start]): start += 1
    if start >= len(arr): return out
    out[start] = arr[start]
    for i in range(start + 1, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out

def rsi_np(close, period=14):
    delta = np.diff(close, prepend=np.nan)
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = ema_np(gain, period); avg_l = ema_np(loss, period)
    rs    = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-12))
    return 100.0 - 100.0 / (1.0 + rs)

def atr_np(high, low, close, period=14):
    tr = np.maximum(high[1:]-low[1:],
         np.maximum(np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
    return ema_np(np.concatenate([[np.nan], tr]), period)

def adx_np(high, low, close, period=14):
    h,l,c = high.astype(float), low.astype(float), close.astype(float)
    tr  = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr = ema_np(np.concatenate([[np.nan], tr]), period)
    up  = np.concatenate([[np.nan], h[1:]-h[:-1]])
    dn  = np.concatenate([[np.nan], l[:-1]-l[1:]])
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    ndm = np.where((dn>up)&(dn>0), dn, 0.0)
    safe_atr = np.where(atr==0, 1e-9, atr)
    pdi = 100.0 * ema_np(pdm, period) / safe_atr
    ndi = 100.0 * ema_np(ndm, period) / safe_atr
    denom = np.where(pdi+ndi==0, 1e-9, pdi+ndi)
    return ema_np(100.0 * np.abs(pdi-ndi) / denom, period)

def bbands_np(close, period=20):
    mid = np.full(len(close), np.nan)
    wid = np.full(len(close), np.nan)
    pos = np.full(len(close), np.nan)
    for i in range(period-1, len(close)):
        sl = close[i-period+1:i+1]; m = sl.mean(); s = sl.std()
        mid[i] = m
        wid[i] = (4*s) / (m if m != 0 else 1e-9)
        pos[i] = (close[i]-m) / (2*s+1e-9)
    return mid, wid, pos

def make_features(df):
    n     = len(df)
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    vol   = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(n)
    cols  = {}

    for p in [1,3,5,10,20]:
        r = np.full(n, np.nan); r[p:] = (close[p:]-close[:-p])/(close[:-p]+1e-9)
        cols[f"ret_{p}"] = r

    for span in [9,21,50,100]:
        e = ema_np(close, span); cols[f"ema_{span}_dist"] = (close-e)/(close+1e-9)

    e9=ema_np(close,9); e21=ema_np(close,21); e50=ema_np(close,50)
    cols["ema_cross_9_21"] = (e9-e21)/(close+1e-9)
    cols["rsi"] = rsi_np(close,14)/100.0

    m12=ema_np(close,12); m26=ema_np(close,26); macd=m12-m26; sig=ema_np(macd,9)
    cols["macd"]      = macd/(close+1e-9)
    cols["macd_hist"] = (macd-sig)/(close+1e-9)

    cols["atr_pct"] = atr_np(high,low,close,14)/(close+1e-9)

    _, bbw, bbpos = bbands_np(close,20)
    cols["bb_width"] = bbw; cols["bb_pos"] = bbpos

    cols["adx"] = adx_np(high,low,close,14)/100.0

    vol_ma = np.full(n, np.nan)
    for i in range(19,n): vol_ma[i] = vol[i-19:i+1].mean()
    cols["vol_ratio"] = vol/(vol_ma+1e-9)

    hl = high-low+1e-9
    cols["body_pct"]   = np.abs(close-open_)/hl
    cols["upper_wick"] = (high-np.maximum(close,open_))/hl
    cols["lower_wick"] = (np.minimum(close,open_)-low)/hl

    mom=np.full(n,np.nan); mom[10:]=close[10:]-close[:-10]; cols["mom_10"]=mom/(close+1e-9)
    roc=np.full(n,np.nan); roc[5:]=(close[5:]-close[:-5])/(close[:-5]+1e-9); cols["roc_5"]=roc

    try:
        sys.path.insert(0,BASE)
        from market_regime import detect_regime
        regs,_,_ = detect_regime(df)
        rmap={"TRENDING_STRONG":3,"TRENDING":2,"RANGING":1,"HIGH_VOL":4,"LOW_VOL":0,"UNKNOWN":1}
        cols["regime"] = np.array([rmap.get(r,1) for r in regs],float)/4.0
    except Exception:
        cols["regime"] = np.ones(n)*0.5

    try:
        with open(os.path.join(BASE,"news_cache.json")) as f: nd=json.load(f)
        cols["news_score"] = np.full(n, float(nd.get("score",0)))
    except Exception:
        cols["news_score"] = np.zeros(n)

    return pd.DataFrame(cols, index=df.index)

def make_target(close_arr, horizon=3, threshold=0.0003):
    n=len(close_arr); tgt=np.full(n,np.nan)
    for i in range(n-horizon):
        ret=(close_arr[i+horizon]-close_arr[i])/(close_arr[i]+1e-9)
        tgt[i]=1.0 if ret>threshold else 0.0
    return tgt

def kelly_fraction(win_rate, avg_win, avg_loss):
    if avg_loss<=0: return 0.01
    b=avg_win/avg_loss; k=(b*win_rate-(1-win_rate))/b
    return max(0.005, min(k*0.5, 0.05))

# ── Download ──────────────────────────────────────────────────────
print("\n  Downloading gold data (6 months, 1h)...")
df = yf.download("GC=F", period="6mo", interval="1h", progress=False, auto_adjust=True)
if df is None or len(df) < 200:
    print("  Retrying 60d/30m...")
    df = yf.download("GC=F", period="60d", interval="30m", progress=False, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0].lower() for c in df.columns]
else:
    df.columns = [str(c).lower() for c in df.columns]
if hasattr(df.index,"tz") and df.index.tz is not None:
    df.index = df.index.tz_convert("UTC").tz_localize(None)
df = df.dropna(subset=["open","high","low","close"])
print(f"  Downloaded {len(df)} candles")

# ── Features ──────────────────────────────────────────────────────
print("  Engineering features...")
feat_df = make_features(df)
feat_df["target"] = make_target(df["close"].values, horizon=3, threshold=0.0003)
feat_df = feat_df.replace([np.inf,-np.inf], np.nan).dropna()

print(f"  Rows after cleaning: {len(feat_df)}")
if len(feat_df) < 100:
    print("  ERROR: not enough rows after cleaning"); sys.exit(1)

feature_cols = [c for c in feat_df.columns if c != "target"]
X_all = feat_df[feature_cols].values.astype(float)
y_all = feat_df["target"].values.astype(int)

split = int(len(X_all)*0.8)
X_tr, X_val = X_all[:split], X_all[split:]
y_tr, y_val = y_all[:split], y_all[split:]

scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)
print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Features: {len(feature_cols)}")

# ── Models ────────────────────────────────────────────────────────
print()
print("  [1/4] Training XGBoost...")
xgb_m = xgb.XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",verbosity=0,random_state=42)
xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
xgb_acc=accuracy_score(y_val,xgb_m.predict(X_val_s))
xgb_prob=xgb_m.predict_proba(X_val_s)[:,1]
print(f"     XGBoost: {xgb_acc*100:.1f}%")

print("  [2/4] Training LightGBM...")
lgb_m = lgb.LGBMClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,verbose=-1,random_state=42)
lgb_m.fit(X_tr_s,y_tr,eval_set=[(X_val_s,y_val)],
    callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(-1)])
lgb_acc=accuracy_score(y_val,lgb_m.predict(X_val_s))
lgb_prob=lgb_m.predict_proba(X_val_s)[:,1]
print(f"     LightGBM: {lgb_acc*100:.1f}%")

print("  [3/4] Training GradientBoost...")
gb_m = GradientBoostingClassifier(n_estimators=200,max_depth=4,
    learning_rate=0.08,subsample=0.8,random_state=42)
gb_m.fit(X_tr_s,y_tr)
gb_acc=accuracy_score(y_val,gb_m.predict(X_val_s))
gb_prob=gb_m.predict_proba(X_val_s)[:,1]
print(f"     GradBoost: {gb_acc*100:.1f}%")

accs=np.array([xgb_acc,lgb_acc,gb_acc]); weights=accs/accs.sum()
print(f"\n  Dynamic weights: XGB={weights[0]:.2f} LGB={weights[1]:.2f} GB={weights[2]:.2f}")

lstm_acc=0.5; lstm_prob=np.full(len(X_val_s),0.5)
if HAS_TF:
    print("  [4/4] Training LSTM...")
    SEQ=20
    def make_seqs(X,y,s):
        xs,ys=[],[]
        for i in range(s,len(X)): xs.append(X[i-s:i]); ys.append(y[i])
        return np.array(xs),np.array(ys)
    Xs,ys=make_seqs(X_tr_s,y_tr,SEQ); Xvs,yvs=make_seqs(X_val_s,y_val,SEQ)
    if len(Xs)>=64:
        m=Sequential([LSTM(128,input_shape=(SEQ,X_tr_s.shape[1]),return_sequences=True),
            Dropout(0.2),BatchNormalization(),LSTM(64),Dropout(0.2),BatchNormalization(),
            Dense(32,activation="relu"),Dense(1,activation="sigmoid")])
        m.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        m.fit(Xs,ys,validation_data=(Xvs,yvs),epochs=50,batch_size=64,verbose=0,
            callbacks=[EarlyStopping(patience=10,restore_best_weights=True),
                       ReduceLROnPlateau(patience=5,factor=0.5,verbose=0)])
        raw=m.predict(Xvs,verbose=0).flatten()
        lstm_acc=accuracy_score(yvs,(raw>0.5).astype(int))
        lstm_prob[-len(raw):]=raw
        print(f"     LSTM: {lstm_acc*100:.1f}%")
        m.save(os.path.join(BASE,"lstm_model.keras"))
    else: print("     LSTM: not enough sequences"); HAS_TF=False
else:
    print("  [4/4] LSTM skipped")

ens_prob=weights[0]*xgb_prob+weights[1]*lgb_prob+weights[2]*gb_prob
ens_acc=accuracy_score(y_val,(ens_prob>0.5).astype(int))

# Kelly
close_v=df["close"].values
vc=close_v[-(len(y_val)+3):-3][-len(y_val):]
wins,losses=[],[]
for i in range(min(len(ens_prob),len(vc)-3)):
    ret=(vc[i+3]-vc[i])/(vc[i]+1e-9)
    if (ens_prob[i]>0.5): (wins if ret>0 else losses).append(abs(ret))
wr=len(wins)/(len(wins)+len(losses)+1e-9)
aw=float(np.mean(wins)) if wins else 0.001
al=float(np.mean(losses)) if losses else 0.001
kelly=kelly_fraction(wr,aw,al)

print(f"\n  Ensemble: {ens_acc*100:.1f}%  Kelly: {kelly*100:.3f}%  Win rate: {wr*100:.1f}%")

print("\n  Saving models...")
joblib.dump(xgb_m, os.path.join(BASE,"xgb_model.pkl"))
joblib.dump(lgb_m, os.path.join(BASE,"lgb_model.pkl"))
joblib.dump(gb_m,  os.path.join(BASE,"gb_model.pkl"))
joblib.dump(scaler,os.path.join(BASE,"scaler.pkl"))

ec={"xgb_accuracy":float(xgb_acc),"lgb_accuracy":float(lgb_acc),
    "gb_accuracy":float(gb_acc),"lstm_accuracy":float(lstm_acc),
    "ensemble_accuracy":float(ens_acc),
    "xgb_weight":float(weights[0]),"lgb_weight":float(weights[1]),"gb_weight":float(weights[2]),
    "lstm_available":HAS_TF,"kelly_fraction":float(kelly),
    "win_rate":float(wr),"avg_win":float(aw),"avg_loss":float(al),
    "n_features":len(feature_cols),"feature_names":feature_cols,
    "n_train":int(len(X_tr)),"n_val":int(len(X_val)),
    "trained_at":time.strftime("%Y-%m-%d %H:%M:%S")}
joblib.dump(ec,os.path.join(BASE,"ensemble.pkl"))
with open(os.path.join(BASE,"ensemble_config.json"),"w") as f: json.dump(ec,f,indent=2)

print("\n  Generating current signal...")
try:
    recent=yf.download("GC=F",period="5d",interval="1h",progress=False,auto_adjust=True)
    if isinstance(recent.columns,pd.MultiIndex): recent.columns=[c[0].lower() for c in recent.columns]
    else: recent.columns=[str(c).lower() for c in recent.columns]
    if hasattr(recent.index,"tz") and recent.index.tz is not None:
        recent.index=recent.index.tz_convert("UTC").tz_localize(None)
    recent=recent.dropna(subset=["open","high","low","close"])
    feat_r=make_features(recent).replace([np.inf,-np.inf],np.nan).dropna()
    if len(feat_r)==0: raise ValueError("No clean rows")
    X_r=scaler.transform(feat_r[feature_cols].values[-1:].astype(float))
    p_xgb=float(xgb_m.predict_proba(X_r)[0][1])
    p_lgb=float(lgb_m.predict_proba(X_r)[0][1])
    p_gb =float(gb_m.predict_proba(X_r)[0][1])
    conf =weights[0]*p_xgb+weights[1]*p_lgb+weights[2]*p_gb
    sig  ="BUY" if conf>0.55 else ("SELL" if conf<0.45 else "NEUTRAL")
    rm   =2.0 if conf>0.80 else (1.5 if conf>0.70 else (1.0 if conf>0.60 else 0.5))
    out  ={"signal":sig,"confidence":round(float(conf),4),"kelly_fraction":round(float(kelly),5),
           "risk_multiplier":rm,"sized_risk_pct":round(kelly*rm*100,3),
           "xgb_p":round(p_xgb,4),"lgb_p":round(p_lgb,4),"gb_p":round(p_gb,4),
           "timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(BASE,"current_signal.json"),"w") as f: json.dump(out,f,indent=2)
    print(f"  Signal: {sig}  Confidence: {conf*100:.1f}%  Risk: {out['sized_risk_pct']:.3f}%")
except Exception as e:
    print(f"  Signal generation skipped: {e}")

print("\n  Training complete!")
