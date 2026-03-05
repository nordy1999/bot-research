"""
AlmostFinishedBot - Correlation Guard
Monitors DXY, US10Y yield, SPX, BTC in real-time.
Detects gold decoupling, VIX spikes, correlation breakdowns.
Saves correlation_status.json for EA and Control Center.

v2.1 FIX: Added check_correlation() bridge-compatible wrapper
"""
import os, sys, json, time
import numpy as np

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
os.makedirs(BASE, exist_ok=True)

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try: import yfinance as yf
except ImportError: install("yfinance"); import yfinance as yf

import pandas as pd

# Tickers: GC=F=Gold, DX-Y.NYB=DXY, ^TNX=US10Y yield, ^GSPC=SPX, BTC-USD=Bitcoin, ^VIX=VIX
TICKERS = {
    "gold": "GC=F",
    "dxy":  "DX-Y.NYB",
    "us10y":"^TNX",
    "spx":  "^GSPC",
    "btc":  "BTC-USD",
    "vix":  "^VIX",
}

def fetch_changes(period="2d", interval="1h"):
    """Fetch recent % changes for all assets."""
    results = {}
    try:
        tickers = list(TICKERS.values())
        data    = yf.download(tickers, period=period, interval=interval,
                              progress=False, auto_adjust=True)

        for name, ticker in TICKERS.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    series = data["Close"][ticker].dropna()
                else:
                    series = data["Close"].dropna()

                if len(series) < 2:
                    results[name] = {"change_1h":0,"change_4h":0,"current":0}
                    continue

                curr    = float(series.iloc[-1])
                prev_1h = float(series.iloc[-2]) if len(series) >= 2 else curr
                prev_4h = float(series.iloc[-5]) if len(series) >= 5 else curr

                results[name] = {
                    "current":   round(curr, 4),
                    "change_1h": round((curr - prev_1h) / (prev_1h + 1e-9) * 100, 3),
                    "change_4h": round((curr - prev_4h) / (prev_4h + 1e-9) * 100, 3),
                }
            except Exception:
                results[name] = {"change_1h":0,"change_4h":0,"current":0}

    except Exception as e:
        print(f"  Data fetch error: {e}")
        for name in TICKERS:
            results[name] = {"change_1h":0,"change_4h":0,"current":0}

    return results

def check_correlations(changes):
    """
    Run all correlation checks. Returns list of warnings + kill_switch bool.
    Normal gold relationships:
      Gold vs DXY   : INVERSE  (gold up = DXY down)
      Gold vs SPX   : weak positive
      Gold vs US10Y : INVERSE  (gold up = yields down)
      Gold vs VIX   : POSITIVE (gold up on fear)
    """
    warnings = []
    kill_switch = False
    reduce_risk = False

    gold_1h = changes.get("gold",{}).get("change_1h", 0)
    dxy_1h  = changes.get("dxy", {}).get("change_1h", 0)
    y10_1h  = changes.get("us10y",{}).get("change_1h",0)
    spx_1h  = changes.get("spx", {}).get("change_1h", 0)
    vix_1h  = changes.get("vix", {}).get("change_1h", 0)
    btc_1h  = changes.get("btc", {}).get("change_1h", 0)

    # ── Gold/DXY decoupling ──────────────────────────────────────
    if gold_1h > 0.4 and dxy_1h > 0.4:
        warnings.append(f"DECOUPLE: Gold +{gold_1h:.2f}% AND DXY +{dxy_1h:.2f}% simultaneously (unusual)")
        kill_switch = True
    elif gold_1h < -0.4 and dxy_1h < -0.4:
        warnings.append(f"DECOUPLE: Gold {gold_1h:.2f}% AND DXY {dxy_1h:.2f}% both falling (unusual)")
        kill_switch = True

    # ── VIX spike (panic/crash incoming) ─────────────────────────
    vix_curr = changes.get("vix",{}).get("current", 0)
    if vix_1h > 15:
        warnings.append(f"VIX SPIKE: +{vix_1h:.1f}% in 1h (panic — reduce size)")
        reduce_risk = True
    if vix_curr > 30:
        warnings.append(f"VIX ELEVATED: {vix_curr:.1f} > 30 (high fear environment)")
        reduce_risk = True

    # ── Gold/US10Y decoupling ─────────────────────────────────────
    if gold_1h > 0.5 and y10_1h > 2.0:
        warnings.append(f"GEOPOLITICAL BID: Gold +{gold_1h:.2f}% despite yields +{y10_1h:.2f}% — geopolitical/CB buying")

    # ── SPX crash (risk-off, gold likely to spike) ────────────────
    if spx_1h < -1.5:
        warnings.append(f"SPX CRASH: {spx_1h:.2f}% in 1h — potential gold spike, be careful of reversals")
        reduce_risk = True

    # ── BTC/Gold correlation break ────────────────────────────────
    if btc_1h < -5 and gold_1h < -0.5:
        warnings.append(f"RISK ASSETS DUMP: BTC {btc_1h:.1f}% + Gold {gold_1h:.2f}% — broad risk-off")
        reduce_risk = True

    # ── Positive signal detectors ─────────────────────────────────
    boosts = []
    if gold_1h > 0.3 and dxy_1h < -0.2 and y10_1h < -1.0:
        boosts.append("PERFECT BULL: Gold rising + DXY falling + Yields falling")
    if vix_1h > 5 and gold_1h > 0.3:
        boosts.append(f"FLIGHT TO SAFETY: VIX +{vix_1h:.1f}% + Gold +{gold_1h:.2f}%")
    if gold_1h > 0.6 and dxy_1h > 0.3:
        boosts.append(f"CB/INSTITUTIONAL BUY: Gold +{gold_1h:.2f}% vs DXY +{dxy_1h:.2f}%")

    # ── Risk multiplier ───────────────────────────────────────────
    if kill_switch:
        risk_mult = 0.0
    elif reduce_risk:
        risk_mult = 0.4
    elif boosts:
        risk_mult = 1.3
    else:
        risk_mult = 1.0

    return warnings, boosts, kill_switch, reduce_risk, risk_mult

def run_correlation_check():
    """Full check — saves status and returns result dict."""
    print("  Running correlation checks...")
    changes = fetch_changes("2d", "1h")
    warnings, boosts, kill, reduce, risk_mult = check_correlations(changes)

    status = {
        "kill_switch":     kill,
        "reduce_risk":     reduce,
        "risk_multiplier": round(risk_mult, 2),
        "trading_ok":      not kill,
        "warnings":        warnings,
        "boosts":          boosts,
        "assets":          changes,
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(BASE,"correlation_status.json"),"w") as f:
        json.dump(status, f, indent=2)

    if kill:
        print(f"  KILL SWITCH ACTIVE — no trading")
    elif reduce:
        print(f"  Risk reduced to {risk_mult:.1f}x")
    else:
        print(f"  Correlations normal — risk {risk_mult:.1f}x")

    for w in warnings: print(f"  ⚠ {w}")
    for b in boosts:   print(f"  ✓ {b}")

    # Print asset summary
    print(f"\n  Asset changes (1h):")
    for name, d in changes.items():
        chg = d.get("change_1h", 0)
        bar = "▲" if chg > 0 else "▼"
        print(f"    {name.upper():8s} {bar} {chg:+.3f}%  (current: {d.get('current',0):.4f})")

    return status


# ── Bridge-compatible wrapper ─────────────────────────────────────────────────
# FIX: Bridge imports `from correlation_guard import check_correlation` — this was missing
def check_correlation(signal=None):
    """
    Returns dict with 'conflict', 'kill_switch', 'reduce_risk', 'risk_mult', 'reason', 'boosts'.
    Reads correlation_status.json if fresh (<5 min), otherwise runs live check.
    """
    import datetime as _dt
    cache_path = os.path.join(BASE, "correlation_status.json")
    status = None

    # Try cache first (avoid 6-ticker yfinance download every 60s cycle)
    try:
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                status = json.load(f)
            ts = status.get("timestamp", "")
            if ts:
                cache_time = _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                age = (_dt.datetime.now() - cache_time).total_seconds()
                if age > 300:  # stale after 5 min
                    status = None
    except Exception:
        status = None

    if status is None:
        status = run_correlation_check()

    conflict = status.get("kill_switch", False) or status.get("reduce_risk", False)
    warnings_list = status.get("warnings", [])
    reason = "; ".join(warnings_list[:2]) if warnings_list else "OK"

    return {
        "conflict":    conflict,
        "kill_switch": status.get("kill_switch", False),
        "reduce_risk": status.get("reduce_risk", False),
        "risk_mult":   status.get("risk_multiplier", 1.0),
        "reason":      reason,
        "boosts":      status.get("boosts", []),
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  AlmostFinishedBot — Correlation Guard")
    print("=" * 55)
    run_correlation_check()
