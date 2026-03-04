"""
AlmostFinishedBot - XAUSGD ML Trainer v6
PyTorch GPU + CatBoost + RF + Regime-Conditional

Same v6 stack as XAUUSD but for XAUSGD:
  - Fetches data from MT5 (or constructs XAUUSD * USDSGD)
  - Saves all models with xausgd_ prefix
  - 7 base models: XGB, LGB, GB, CatBoost, RF, LSTM(GPU), TFT(GPU)
  - Regime-conditional + meta-learner + 5 strategy comparison
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
print(f"  AlmostFinishedBot  |  {SYMBOL} Trainer v6")
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
        DEVICE = "cuda"; print(f"  PyTorch: GPU mode ({torch.cuda.get_device_name(0)})")
    else: print("  PyTorch: CPU mode")
except ImportError: print("  PyTorch: not available")

HORIZON = 5; THRESHOLD = 0.001; PURGE_GAP = HORIZON + 2
MIN_ACC = 0.52; MIN_REGIME_SAMPLES = 200; SPREAD_PCT = 0.003
TRENDING_REGIMES = {"TREND_UP", "TREND_DOWN"}

def fetch_xausgd_data():
    print("\n  Fetching XAUSGD data...")
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            si = mt5.symbol_info(SYMBOL)
            if si and not si.visible: mt5.symbol_select(SYMBOL, True)
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 10000)
            if rates is not None and len(rates) > 500:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                print(f"  [OK] MT5: {len(df)} hourly candles"); mt5.shutdown(); return df
            mt5.shutdown()
    except Exception as e: print(f"  MT5 failed: {e}")
    print("  Constructing from XAUUSD * USDSGD...")
    try:
        gold = yf.download("GC=F", period="6mo", interval="1h", progress=False)
        sgd = yf.download("SGD=X", period="6mo", interval="1h", progress=False)
        for d in [gold, sgd]:
            if hasattr(d.columns, "droplevel") and d.columns.nlevels > 1: d.columns = d.columns.droplevel(1)
        gold.columns = [c.lower() for c in gold.columns]; sgd.columns = [c.lower() for c in sgd.columns]
        gold = gold.reset_index(); sgd = sgd.reset_index()
        for d in [gold, sgd]:
            for col in d.columns:
                if "date" in col.lower() or "time" in col.lower(): d.rename(columns={col: "time"}, inplace=True); break
        gold["time"] = pd.to_datetime(gold["time"]).dt.tz_localize(None)
        sgd["time"] = pd.to_datetime(sgd["time"]).dt.tz_localize(None)
        merged = pd.merge_asof(gold.sort_values("time"), sgd.sort_values("time"),
                               on="time", suffixes=("_gold","_sgd"), tolerance=pd.Timedelta("2h")).dropna()
        merged["open"]=merged["open_gold"]*merged["open_sgd"]; merged["high"]=merged["high_gold"]*merged["high_sgd"]
        merged["low"]=merged["low_gold"]*merged["low_sgd"]; merged["close"]=merged["close_gold"]*merged["close_sgd"]
        merged["tick_volume"]=merged.get("volume_gold",0)
        df = merged[["time","open","high","low","close","tick_volume"]].copy()
        print(f"  [OK] Constructed: {len(df)} candles"); return df
    except Exception as e: print(f"  [FAIL] {e}"); return None

def make_features_sgd(df):
    df = df.copy().sort_values("time").reset_index(drop=True)
    c=df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float); o=df["open"].astype(float)
    for w in [1,2,3,5,10,20]: df[f"ret_{w}"]=c.pct_change(w)
    for w in [9,21,50,100]:
        ema=c.ewm(span=w).mean(); df[f"ema_{w}_dist"]=(c-ema)/(ema+1e-10)
    df["ema_cross_9_21"]=c.ewm(span=9).mean()-c.ewm(span=21).mean()
    df["ema_cross_21_50"]=c.ewm(span=21).mean()-c.ewm(span=50).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); lv=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/(lv+1e-10); df["rsi"]=100-(100/(1+rs)); df["rsi_extreme"]=((df["rsi"]>70)|(df["rsi"]<30)).astype(float)
    e12=c.ewm(span=12).mean(); e26=c.ewm(span=26).mean(); df["macd"]=e12-e26; df["macd_hist"]=df["macd"]-df["macd"].ewm(span=9).mean()
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    df["atr"]=tr.rolling(14).mean(); df["atr_pct"]=df["atr"]/(c+1e-10)
    s20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    df["bb_width"]=(4*sd20)/(s20+1e-10); df["bb_pos"]=(c-(s20-2*sd20))/(4*sd20+1e-10)
    pdm=(h-h.shift(1)).clip(lower=0); mdm=(l.shift(1)-l).clip(lower=0); tr14=tr.rolling(14).mean()
    pdi=100*pdm.rolling(14).mean()/(tr14+1e-10); mdi=100*mdm.rolling(14).mean()/(tr14+1e-10)
    dx=100*(pdi-mdi).abs()/(pdi+mdi+1e-10); df["adx"]=dx.rolling(14).mean()/100.0
    low14=l.rolling(14).min(); high14=h.rolling(14).max()
    df["stoch_k"]=(c-low14)/(high14-low14+1e-10); df["stoch_kd_cross"]=df["stoch_k"]-df["stoch_k"].rolling(3).mean()
    if "tick_volume" in df.columns: v=df["tick_volume"].astype(float); df["vol_ratio"]=v/(v.rolling(20).mean()+1e-10)
    else: df["vol_ratio"]=1.0
    df["body_pct"]=(c-o).abs()/(h-l+1e-10)
    df["upper_wick"]=(h-pd.concat([c,o],axis=1).max(axis=1))/(h-l+1e-10)
    df["lower_wick"]=(pd.concat([c,o],axis=1).min(axis=1)-l)/(h-l+1e-10)
    df["candle_dir"]=(c>o).astype(float); df["mom_10"]=c-c.shift(10); df["roc_5"]=c.pct_change(5)
    atr_z=(df["atr_pct"]-df["atr_pct"].rolling(50).mean())/(df["atr_pct"].rolling(50).std()+1e-10)
    df["vol_regime"]=atr_z.clip(-3,3)
    h20=h.rolling(20).max(); l20=l.rolling(20).min(); df["pos_in_range_20"]=(c-l20)/(h20-l20+1e-10)
    if "time" in df.columns:
        dt=pd.to_datetime(df["time"])
        df["hour_sin"]=np.sin(2*np.pi*dt.dt.hour/24); df["hour_cos"]=np.cos(2*np.pi*dt.dt.hour/24)
        df["dow_sin"]=np.sin(2*np.pi*dt.dt.dayofweek/7); df["dow_cos"]=np.cos(2*np.pi*dt.dt.dayofweek/7)
        hour=dt.dt.hour; df["session"]=0
        df.loc[(hour>=0)&(hour<8),"session"]=1; df.loc[(hour>=8)&(hour<16),"session"]=2; df.loc[(hour>=16),"session"]=3
    df["regime"]=0; df.loc[df["adx"]>0.25,"regime"]=1; df["news_score"]=0.0
    return df

def make_target_sgd(close_arr, horizon=5, threshold=0.001):
    n=len(close_arr); target=np.zeros(n,dtype=int)
    for i in range(n-horizon):
        fm=np.max(close_arr[i+1:i+horizon+1]); ret=(fm-close_arr[i])/(close_arr[i]+1e-10)
        if ret>threshold: target[i]=1
    return target

def make_regime_labels_sgd(df):
    c=df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float)
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean(); atr_pct=atr/(c+1e-10); ret5=c.pct_change(5)
    pdm=(h-h.shift(1)).clip(lower=0); mdm=(l.shift(1)-l).clip(lower=0); tr14=tr.rolling(14).mean()
    pdi=100*pdm.rolling(14).mean()/(tr14+1e-10); mdi=100*mdm.rolling(14).mean()/(tr14+1e-10)
    dx=100*(pdi-mdi).abs()/(pdi+mdi+1e-10); adx=dx.rolling(14).mean()/100.0
    s20=c.rolling(20).mean(); sd20=c.rolling(20).std(); bb_w=(4*sd20)/(s20+1e-10)
    labels=[]
    for i in range(len(df)):
        a=adx.iloc[i] if not np.isnan(adx.iloc[i]) else 0.25
        r=ret5.iloc[i] if not np.isnan(ret5.iloc[i]) else 0
        ap=atr_pct.iloc[i] if not np.isnan(atr_pct.iloc[i]) else 0.01
        bw=bb_w.iloc[i] if not np.isnan(bb_w.iloc[i]) else 0.02
        if a>0.25 and abs(r)>0.003: labels.append("TREND_UP" if r>0 else "TREND_DOWN")
        elif ap>0.02: labels.append("HIGH_VOL_BREAKOUT")
        elif ap<0.005 and a<0.20: labels.append("LOW_VOL_COMPRESSION")
        else: labels.append("CHOPPY_RANGE")
    return labels

# == Fetch ==================================================================
df_raw = fetch_xausgd_data()
if df_raw is None or len(df_raw)<500: print("  [FAIL] Not enough data"); sys.exit(1)
print(f"  Got {len(df_raw)} candles")

print("  Engineering features...")
feat_df = make_features_sgd(df_raw)
feat_df["target"] = make_target_sgd(df_raw["close"].values, HORIZON, THRESHOLD)
regime_labels = make_regime_labels_sgd(df_raw)
feat_df["_regime"] = regime_labels[:len(feat_df)]

drop_cols=["time","open","high","low","close","tick_volume","target","_regime","spread","real_volume","atr"]
feature_cols=[c for c in feat_df.columns if c not in drop_cols]
feat_df=feat_df.replace([np.inf,-np.inf],np.nan).dropna()
print(f"  Rows: {len(feat_df)}  Features: {len(feature_cols)}")
print(f"  Balance: {feat_df['target'].mean():.1%} buys")

X_all=feat_df[feature_cols].values.astype(float); y_all=feat_df["target"].values.astype(int); regimes_all=feat_df["_regime"].values
split=int(len(X_all)*0.8)
X_tr=X_all[:split-PURGE_GAP]; y_tr=y_all[:split-PURGE_GAP]; reg_tr=regimes_all[:split-PURGE_GAP]
X_val=X_all[split:]; y_val=y_all[split:]; reg_val=regimes_all[split:]
scaler=StandardScaler(); X_tr_s=scaler.fit_transform(X_tr); X_val_s=scaler.transform(X_val)
print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

# PHASE 1
print(f"\n  {'='*55}\n  PHASE 1: Base Models\n  {'='*55}")
print("\n  [1/7] XGBoost...")
xgb_m=xgb.XGBClassifier(n_estimators=500,max_depth=4,learning_rate=0.03,subsample=0.7,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,min_child_weight=5,eval_metric="logloss",verbosity=0,random_state=42)
xgb_m.fit(X_tr_s,y_tr,eval_set=[(X_val_s,y_val)],verbose=False)
xgb_acc=accuracy_score(y_val,xgb_m.predict(X_val_s)); xgb_prob=xgb_m.predict_proba(X_val_s)[:,1]
print(f"     XGBoost:     {xgb_acc*100:.1f}%")

print("  [2/7] LightGBM...")
lgb_m=lgb.LGBMClassifier(n_estimators=500,max_depth=4,learning_rate=0.03,num_leaves=31,subsample=0.7,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,min_child_samples=20,verbose=-1,random_state=42)
lgb_m.fit(X_tr_s,y_tr,eval_set=[(X_val_s,y_val)],callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(-1)])
lgb_acc=accuracy_score(y_val,lgb_m.predict(X_val_s)); lgb_prob=lgb_m.predict_proba(X_val_s)[:,1]
print(f"     LightGBM:    {lgb_acc*100:.1f}%")

print("  [3/7] GradientBoost...")
gb_m=GradientBoostingClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.7,min_samples_leaf=20,random_state=42)
gb_m.fit(X_tr_s,y_tr); gb_acc=accuracy_score(y_val,gb_m.predict(X_val_s)); gb_prob=gb_m.predict_proba(X_val_s)[:,1]
print(f"     GradBoost:   {gb_acc*100:.1f}%")

cb_m=None; cb_acc=0.5; cb_prob=np.full(len(y_val),0.5)
if HAS_CB:
    print("  [4/7] CatBoost...")
    cb_m=CatBoostClassifier(iterations=500,depth=4,learning_rate=0.03,l2_leaf_reg=3.0,subsample=0.7,random_seed=42,verbose=0,eval_metric="Accuracy")
    cb_m.fit(X_tr_s,y_tr,eval_set=(X_val_s,y_val),verbose=0,early_stopping_rounds=30)
    cb_acc=accuracy_score(y_val,cb_m.predict(X_val_s)); cb_prob=cb_m.predict_proba(X_val_s)[:,1]
    print(f"     CatBoost:    {cb_acc*100:.1f}%")

print("  [5/7] Random Forest...")
rf_m=RandomForestClassifier(n_estimators=500,max_depth=6,min_samples_leaf=20,max_features="sqrt",oob_score=True,random_state=42,n_jobs=-1)
rf_m.fit(X_tr_s,y_tr); rf_acc=accuracy_score(y_val,rf_m.predict(X_val_s)); rf_prob=rf_m.predict_proba(X_val_s)[:,1]
print(f"     RandomForest:{rf_acc*100:.1f}%  (OOB={rf_m.oob_score_*100:.1f}%)")

SEQ=20; lstm_acc=0.5; lstm_prob=np.full(len(y_val),0.5); lstm_model_pt=None
Xs=None; yvs=None; train_dl=None; val_dl=None; Xvs=None

class LSTMModel(nn.Module):
    def __init__(s,nf,h=64,ly=2,d=0.3):
        super().__init__(); s.lstm=nn.LSTM(nf,h,ly,batch_first=True,dropout=d)
        s.bn=nn.BatchNorm1d(h); s.fc1=nn.Linear(h,32); s.drop=nn.Dropout(d); s.fc2=nn.Linear(32,1)
    def forward(s,x):
        o,_=s.lstm(x); o=o[:,-1,:]; o=s.bn(o); o=torch.relu(s.fc1(o)); o=s.drop(o)
        return torch.sigmoid(s.fc2(o)).squeeze(-1)

if HAS_TORCH:
    print(f"  [6/7] LSTM (PyTorch {DEVICE.upper()})...")
    def make_seqs(X,y,s):
        xs,ys=[],[]
        for i in range(s,len(X)): xs.append(X[i-s:i]); ys.append(y[i])
        return np.array(xs,dtype=np.float32),np.array(ys,dtype=np.float32)
    Xs,ys_seq=make_seqs(X_tr_s,y_tr,SEQ); Xvs,yvs=make_seqs(X_val_s,y_val,SEQ)
    if len(Xs)>=64:
        t0=time.time()
        train_dl=DataLoader(TensorDataset(torch.tensor(Xs),torch.tensor(ys_seq)),batch_size=64,shuffle=True)
        val_dl=DataLoader(TensorDataset(torch.tensor(Xvs),torch.tensor(yvs)),batch_size=128)
        lstm_model_pt=LSTMModel(X_tr_s.shape[1]).to(DEVICE)
        opt=torch.optim.Adam(lstm_model_pt.parameters(),lr=0.001)
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=4,factor=0.5); loss_fn=nn.BCELoss()
        bvl=float("inf"); pat=8; w=0; bs=None
        for ep in range(50):
            lstm_model_pt.train()
            for xb,yb in train_dl:
                xb,yb=xb.to(DEVICE),yb.to(DEVICE); p=lstm_model_pt(xb); lo=loss_fn(p,yb)
                opt.zero_grad(); lo.backward(); opt.step()
            lstm_model_pt.eval(); vls=[]
            with torch.no_grad():
                for xb,yb in val_dl: xb,yb=xb.to(DEVICE),yb.to(DEVICE); vls.append(loss_fn(lstm_model_pt(xb),yb).item())
            vl=np.mean(vls); sch.step(vl)
            if vl<bvl: bvl=vl; w=0; bs={k:v.cpu().clone() for k,v in lstm_model_pt.state_dict().items()}
            else:
                w+=1
                if w>=pat: break
        if bs: lstm_model_pt.load_state_dict(bs)
        lstm_model_pt.eval()
        with torch.no_grad(): raw=lstm_model_pt(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
        lstm_acc=accuracy_score(yvs,(raw>0.5).astype(int))
        la=np.full(len(y_val),0.5); la[-len(raw):]=raw; lstm_prob=la
        print(f"     LSTM:        {lstm_acc*100:.1f}%  ({time.time()-t0:.1f}s {DEVICE.upper()})")
        torch.save({"model_state":lstm_model_pt.state_dict(),"n_features":X_tr_s.shape[1],"seq_len":SEQ,"hidden":64,"layers":2},
                    os.path.join(BASE,f"{MODEL_PREFIX}lstm_model.pt"))

tft_acc=0.5; tft_prob=np.full(len(y_val),0.5); tft_model_pt=None
class SimpleTFT(nn.Module):
    def __init__(s,nf,dm=64,nh=4,sl=20,d=0.3):
        super().__init__(); s.ip=nn.Linear(nf,dm); s.pe=nn.Parameter(torch.randn(1,sl,dm)*0.02)
        el=nn.TransformerEncoderLayer(dm,nh,dm*2,d,batch_first=True)
        s.tr=nn.TransformerEncoder(el,num_layers=2); s.bn=nn.BatchNorm1d(dm)
        s.fc1=nn.Linear(dm,32); s.drop=nn.Dropout(d); s.fc2=nn.Linear(32,1)
    def forward(s,x):
        x=s.ip(x)+s.pe; x=s.tr(x); x=x[:,-1,:]; x=s.bn(x); x=torch.relu(s.fc1(x))
        x=s.drop(x); return torch.sigmoid(s.fc2(x)).squeeze(-1)

if HAS_TORCH and Xs is not None and len(Xs)>=64:
    print(f"  [7/7] TFT (PyTorch {DEVICE.upper()})...")
    t0=time.time()
    tft_model_pt=SimpleTFT(X_tr_s.shape[1],sl=SEQ).to(DEVICE)
    opt2=torch.optim.Adam(tft_model_pt.parameters(),lr=0.0005)
    sch2=torch.optim.lr_scheduler.ReduceLROnPlateau(opt2,patience=4,factor=0.5); loss_fn=nn.BCELoss()
    bvl=float("inf"); w=0; bs=None
    for ep in range(50):
        tft_model_pt.train()
        for xb,yb in train_dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); p=tft_model_pt(xb); lo=loss_fn(p,yb)
            opt2.zero_grad(); lo.backward(); opt2.step()
        tft_model_pt.eval(); vls=[]
        with torch.no_grad():
            for xb,yb in val_dl: xb,yb=xb.to(DEVICE),yb.to(DEVICE); vls.append(loss_fn(tft_model_pt(xb),yb).item())
        vl=np.mean(vls); sch2.step(vl)
        if vl<bvl: bvl=vl; w=0; bs={k:v.cpu().clone() for k,v in tft_model_pt.state_dict().items()}
        else:
            w+=1
            if w>=8: break
    if bs: tft_model_pt.load_state_dict(bs)
    tft_model_pt.eval()
    with torch.no_grad(): raw_t=tft_model_pt(torch.tensor(Xvs).to(DEVICE)).cpu().numpy()
    tft_acc=accuracy_score(yvs,(raw_t>0.5).astype(int))
    ta=np.full(len(y_val),0.5); ta[-len(raw_t):]=raw_t; tft_prob=ta
    print(f"     TFT:         {tft_acc*100:.1f}%  ({time.time()-t0:.1f}s {DEVICE.upper()})")
    torch.save({"model_state":tft_model_pt.state_dict(),"n_features":X_tr_s.shape[1],"seq_len":SEQ,"d_model":64,"n_heads":4},
                os.path.join(BASE,f"{MODEL_PREFIX}tft_model.pt"))

# PHASE 2: Regime
print(f"\n  {'='*55}\n  PHASE 2: Regime-Conditional\n  {'='*55}")
tmtr=np.array([r in TRENDING_REGIMES for r in reg_tr]); tmval=np.array([r in TRENDING_REGIMES for r in reg_val])
regime_models={}
for rn,mtr,mval in [("TRENDING",tmtr,tmval),("CHOPPY",~tmtr,~tmval)]:
    nt=mtr.sum(); nv=mval.sum()
    print(f"\n  --- {rn}: {nt} train / {nv} val ---")
    if nt<MIN_REGIME_SAMPLES or nv<30: regime_models[rn]=None; print("  Skipping"); continue
    Xrt=X_tr_s[mtr]; yrt=y_tr[mtr]; Xrv=X_val_s[mval]; yrv=y_val[mval]
    rl=lgb.LGBMClassifier(n_estimators=400,max_depth=4,learning_rate=0.03,num_leaves=24,subsample=0.7,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=1.5,min_child_samples=15,verbose=-1,random_state=42)
    rl.fit(Xrt,yrt,eval_set=[(Xrv,yrv)],callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(-1)])
    rla=accuracy_score(yrv,rl.predict(Xrv))
    rg=GradientBoostingClassifier(n_estimators=250,max_depth=3,learning_rate=0.05,subsample=0.7,min_samples_leaf=15,random_state=42)
    rg.fit(Xrt,yrt); rga=accuracy_score(yrv,rg.predict(Xrv))
    rca=0; rc=None
    if HAS_CB:
        rc=CatBoostClassifier(iterations=400,depth=4,learning_rate=0.03,l2_leaf_reg=3.0,subsample=0.7,random_seed=42,verbose=0)
        rc.fit(Xrt,yrt,eval_set=(Xrv,yrv),verbose=0,early_stopping_rounds=30); rca=accuracy_score(yrv,rc.predict(Xrv))
    print(f"  LGB:{rla*100:.1f}% GB:{rga*100:.1f}% CB:{rca*100:.1f}%")
    cands={"lgb":(rl,rla),"gb":(rg,rga)}
    if rc: cands["cb"]=(rc,rca)
    bn=max(cands,key=lambda k:cands[k][1]); bm,ba=cands[bn]
    if ba>=MIN_ACC:
        regime_models[rn]={bn:bm,"best":bn,"best_acc":ba,"lgb":rl,"gb":rg}
        if rc: regime_models[rn]["cb"]=rc
        print(f"  BEST: {bn.upper()} ({ba*100:.1f}%)")
    else: regime_models[rn]=None

regime_save={}
if regime_models.get("TRENDING"): rm=regime_models["TRENDING"]; regime_save["trend_model"]=rm[rm["best"]]
if regime_models.get("CHOPPY"): rm=regime_models["CHOPPY"]; regime_save["range_model"]=rm[rm["best"]]

# PHASE 3: Meta
print(f"\n  {'='*55}\n  PHASE 3: Meta-Learner\n  {'='*55}")
meta_val=np.column_stack([xgb_prob,lgb_prob,gb_prob,cb_prob,rf_prob,lstm_prob,
    np.array([1.0 if r in TRENDING_REGIMES else 0.0 for r in reg_val])])
half=len(X_tr_s)//2
print("  OOF predictions...")
xh1=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,subsample=0.7,colsample_bytree=0.7,eval_metric="logloss",verbosity=0,random_state=42); xh1.fit(X_tr_s[:half],y_tr[:half],verbose=False)
xh2=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,subsample=0.7,colsample_bytree=0.7,eval_metric="logloss",verbosity=0,random_state=42); xh2.fit(X_tr_s[half:],y_tr[half:],verbose=False)
lh1=lgb.LGBMClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,subsample=0.7,colsample_bytree=0.7,verbose=-1,random_state=42); lh1.fit(X_tr_s[:half],y_tr[:half])
lh2=lgb.LGBMClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,subsample=0.7,colsample_bytree=0.7,verbose=-1,random_state=42); lh2.fit(X_tr_s[half:],y_tr[half:])
gh1=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.7,random_state=42); gh1.fit(X_tr_s[:half],y_tr[:half])
gh2=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.7,random_state=42); gh2.fit(X_tr_s[half:],y_tr[half:])
ox=np.concatenate([xh2.predict_proba(X_tr_s[:half])[:,1],xh1.predict_proba(X_tr_s[half:])[:,1]])
ol=np.concatenate([lh2.predict_proba(X_tr_s[:half])[:,1],lh1.predict_proba(X_tr_s[half:])[:,1]])
og=np.concatenate([gh2.predict_proba(X_tr_s[:half])[:,1],gh1.predict_proba(X_tr_s[half:])[:,1]])
if HAS_CB:
    ch1=CatBoostClassifier(iterations=300,depth=4,learning_rate=0.03,verbose=0,random_seed=42); ch1.fit(X_tr_s[:half],y_tr[:half],verbose=0)
    ch2=CatBoostClassifier(iterations=300,depth=4,learning_rate=0.03,verbose=0,random_seed=42); ch2.fit(X_tr_s[half:],y_tr[half:],verbose=0)
    oc=np.concatenate([ch2.predict_proba(X_tr_s[:half])[:,1],ch1.predict_proba(X_tr_s[half:])[:,1]])
else: oc=np.full(len(y_tr),0.5)
rh1=RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=20,max_features="sqrt",random_state=42,n_jobs=-1); rh1.fit(X_tr_s[:half],y_tr[:half])
rh2=RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=20,max_features="sqrt",random_state=42,n_jobs=-1); rh2.fit(X_tr_s[half:],y_tr[half:])
orf=np.concatenate([rh2.predict_proba(X_tr_s[:half])[:,1],rh1.predict_proba(X_tr_s[half:])[:,1]])
rc_tr=np.array([1.0 if r in TRENDING_REGIMES else 0.0 for r in reg_tr])
meta_tr=np.column_stack([ox,ol,og,oc,orf,np.full(len(y_tr),0.5),rc_tr])
print("  Training meta-learner...")
meta_model=lgb.LGBMClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,num_leaves=16,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.5,reg_lambda=2.0,min_child_samples=30,verbose=-1,random_state=42)
meta_model.fit(meta_tr,y_tr,eval_set=[(meta_val,y_val)],callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(-1)])
meta_prob=meta_model.predict_proba(meta_val)[:,1]; meta_acc=accuracy_score(y_val,(meta_prob>0.5).astype(int))
print(f"  Meta: {meta_acc*100:.1f}%")

# PHASE 4: Strategies
print(f"\n  {'='*55}\n  PHASE 4: Strategy Comparison\n  {'='*55}")
ma={"xgb":xgb_acc,"lgb":lgb_acc,"gb":gb_acc,"cb":cb_acc,"rf":rf_acc,"lstm":lstm_acc}
edges={k:(v-0.5)**2 if v>=MIN_ACC else 0 for k,v in ma.items()}; te=sum(edges.values())
w6={k:v/te for k,v in edges.items()} if te>0 else {k:1/6 for k in edges}
v6p=sum(w6[k]*p for k,p in zip(["xgb","lgb","gb","cb","rf","lstm"],[xgb_prob,lgb_prob,gb_prob,cb_prob,rf_prob,lstm_prob]))
v6a=accuracy_score(y_val,(v6p>0.5).astype(int))
sm=sorted(ma.items(),key=lambda x:x[1],reverse=True)[:3]; t3n=[n for n,_ in sm]
bp={"xgb":xgb_prob,"lgb":lgb_prob,"gb":gb_prob,"cb":cb_prob,"rf":rf_prob,"lstm":lstm_prob}
t3p=np.mean([bp[n] for n in t3n],axis=0); t3a=accuracy_score(y_val,(t3p>0.5).astype(int))
rp=np.full(len(y_val),0.5)
for i in range(len(y_val)):
    rt="TRENDING" if reg_val[i] in TRENDING_REGIMES else "CHOPPY"
    rm=regime_models.get(rt)
    if rm: rp[i]=rm[rm["best"]].predict_proba(X_val_s[i:i+1])[0][1]
    else: rp[i]=v6p[i]
ra=accuracy_score(y_val,(rp>0.5).astype(int))
hp=0.5*rp+0.5*meta_prob; ha=accuracy_score(y_val,(hp>0.5).astype(int))

strats={"v6 Weighted":(v6a,v6p),"Top-3":(t3a,t3p),"Regime-Conditional":(ra,rp),"Meta":(meta_acc,meta_prob),"Hybrid":(ha,hp)}
print()
bn2,ba2,bp2="",0,None
for n,(a,p) in strats.items():
    if a>ba2: bn2,ba2,bp2=n,a,p
    print(f"  {n:25s}: {a*100:.1f}%")
print(f"\n  >>> BEST: {bn2} ({ba2*100:.1f}%) <<<")

# Kelly
cv=df_raw["close"].values; vc=cv[-(len(y_val)+HORIZON):-HORIZON][-len(y_val):]
wins,losses=[],[]
for i in range(min(len(bp2),len(vc)-HORIZON)):
    ret=(vc[i+HORIZON]-vc[i])/(vc[i]+1e-9); nr=ret-SPREAD_PCT/100
    if bp2[i]>0.5: (wins if nr>0 else losses).append(abs(nr))
wr=len(wins)/(len(wins)+len(losses)+1e-9)
aw=float(np.mean(wins)) if wins else 0.001; al=float(np.mean(losses)) if losses else 0.001
b=aw/al; kelly=max(0.01,min((wr*b-(1-wr))/(b+1e-10),0.05))
print(f"\n  Accuracy: {ba2*100:.1f}%  Kelly: {kelly*100:.3f}%  WR: {wr*100:.1f}%")

# Save
print("\n  Saving models...")
joblib.dump(xgb_m,os.path.join(BASE,f"{MODEL_PREFIX}xgb_model.pkl"))
joblib.dump(lgb_m,os.path.join(BASE,f"{MODEL_PREFIX}lgb_model.pkl"))
joblib.dump(gb_m,os.path.join(BASE,f"{MODEL_PREFIX}gb_model.pkl"))
if HAS_CB and cb_m: joblib.dump(cb_m,os.path.join(BASE,f"{MODEL_PREFIX}catboost_model.pkl"))
joblib.dump(rf_m,os.path.join(BASE,f"{MODEL_PREFIX}rf_model.pkl"))
joblib.dump(scaler,os.path.join(BASE,f"{MODEL_PREFIX}scaler.pkl"))
joblib.dump(meta_model,os.path.join(BASE,f"{MODEL_PREFIX}meta_model.pkl"))
joblib.dump(regime_save,os.path.join(BASE,f"{MODEL_PREFIX}regime_models.pkl"))

ec={"trainer_version":"v6","symbol":SYMBOL,
    "model_accuracies":{k:round(float(v),4) for k,v in ma.items()},
    "model_weights":{k:round(float(v),4) for k,v in w6.items()},
    "best_strategy":bn2,"best_strategy_acc":round(float(ba2),4),
    "catboost_available":HAS_CB and cb_m is not None,"rf_available":True,
    "lstm_available":HAS_TORCH and lstm_model_pt is not None,
    "tft_available":HAS_TORCH and tft_model_pt is not None,
    "pytorch_gpu":DEVICE=="cuda","kelly_fraction":float(kelly),"win_rate":float(wr),
    "n_features":len(feature_cols),"feature_names":feature_cols,
    "trained_at":time.strftime("%Y-%m-%d %H:%M:%S")}
with open(os.path.join(BASE,f"{MODEL_PREFIX}ensemble_config.json"),"w") as f: json.dump(ec,f,indent=2)

print(f"  All saved with prefix: {MODEL_PREFIX}")
print(f"\n  XAUSGD Training complete! Best: {bn2} ({ba2*100:.1f}%)")
print(f"  GPU: {'RTX 2060' if DEVICE=='cuda' else 'CPU'}")
