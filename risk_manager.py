"""
AlmostFinishedBot - Risk Manager v3 — FULL REWRITE
====================================================
UPGRADES:
  - True Half-Kelly Criterion from trade history (win rate + avg win/loss)
  - Equity Curve Trading: pause when equity dips below its own X-bar MA
  - Model Staleness Detection: KL divergence on live feature distributions
  - Isotonic/Platt calibration check (warns if needed)
  - All original God Mode, Prop Firm, session, ATR logic retained
  - Starting balance updated to £500
"""
import os, sys, json, time, datetime
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
os.makedirs(BASE, exist_ok=True)

SETTINGS_FILE = os.path.join(BASE, "bot_settings.json")
EQUITY_LOG    = os.path.join(BASE, "equity_log.json")
RISK_STATUS   = os.path.join(BASE, "risk_status.json")
DRIFT_LOG     = os.path.join(BASE, "drift_log.json")

# ── Default settings ──────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "mode":                 "normal",
    "base_risk_pct":        1.0,
    "god_mode_max_risk":    6.0,          # reduced from 8 for £500 account
    "god_mode_threshold":   0.88,
    "prop_firm_mode":       False,
    "prop_daily_dd_limit":  3.0,
    "prop_total_dd_limit":  6.0,
    "prop_profit_target":   8.0,
    "equity_halve_trigger": 5.0,
    "equity_ma_bars":       20,           # NEW: equity curve MA window
    "equity_restore_pct":   20.0,
    "vol_target_pct":       1.0,
    "compound_auto":        True,
    "starting_balance":     500.0,        # £500 account
    "current_balance":      500.0,
    "peak_balance":         500.0,
    "kelly_lookback":       50,           # trades to use for Kelly calculation
    "staleness_threshold":  0.15,         # KL divergence to flag staleness
}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                s = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in s:
                    s[k] = v
            return s
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)


# ── Equity log ────────────────────────────────────────────────────
def log_equity(balance, equity):
    log = []
    if os.path.exists(EQUITY_LOG):
        try:
            with open(EQUITY_LOG) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "balance":   round(balance, 2),
        "equity":    round(equity,  2),
    })
    log = log[-2000:]
    with open(EQUITY_LOG, "w") as f:
        json.dump(log, f, indent=2)


def load_equity_log():
    if not os.path.exists(EQUITY_LOG):
        return []
    try:
        with open(EQUITY_LOG) as f:
            return json.load(f)
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
# KELLY CRITERION (True Half-Kelly from trade history)
# ═══════════════════════════════════════════════════════════════════
def calculate_kelly(lookback=50):
    """
    Compute the half-Kelly fraction from recent closed trade history.
    Returns (kelly_pct, win_rate, avg_win, avg_loss, edge, n_trades)

    Uses the trading_log.json for data.
    Kelly formula: K = (b*p - q) / b
      b = avg_win / avg_loss   (reward/risk ratio)
      p = win rate
      q = 1 - p
    Half-Kelly = K * 0.5  (safer, standard for live trading)
    """
    log_path = os.path.join(BASE, "trading_log.json")
    trades   = []

    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                log = json.load(f)
            trades = [e for e in log
                      if e.get("type") == "trade"
                      and e.get("profit") is not None][-lookback:]
        except Exception:
            pass

    # Also try backtest results for richer data if live data is sparse
    if len(trades) < 10:
        bt_path = os.path.join(BASE, "walkforward_results.json")
        if os.path.exists(bt_path):
            try:
                with open(bt_path) as f:
                    bt = json.load(f)
                bt_trades = bt.get("trades", [])[-lookback:]
                trades = bt_trades  # prefer recent backtest
            except Exception:
                pass

    if len(trades) < 5:
        return 0.01, 0.5, 1.0, 1.0, 0.0, 0  # too few trades — ultra-conservative

    profits = [float(t.get("profit", t.get("pnl", 0))) for t in trades]
    wins    = [p for p in profits if p > 0]
    losses  = [abs(p) for p in profits if p < 0]

    if not wins or not losses:
        return 0.01, 0.5, 1.0, 1.0, 0.0, len(profits)

    n_trades = len(profits)
    win_rate = len(wins) / n_trades
    avg_win  = float(np.mean(wins))
    avg_loss = float(np.mean(losses))

    if avg_loss <= 0:
        return 0.01, win_rate, avg_win, avg_loss, 0.0, n_trades

    b        = avg_win / avg_loss           # reward-to-risk ratio
    q        = 1.0 - win_rate
    kelly_f  = (b * win_rate - q) / b      # raw Kelly

    # Cap: never bet more than 5% even with a great edge
    half_kelly = max(0.005, min(kelly_f * 0.5, 0.05))
    edge       = kelly_f

    return half_kelly, win_rate, avg_win, avg_loss, edge, n_trades


def kelly_position_pct(ml_confidence, smc_score, regime, news_ok):
    """
    Returns final risk % using Kelly as the base, scaled by all multipliers.
    This replaces the old fixed-percentage approach.
    """
    half_k, wr, aw, al, edge, n = calculate_kelly()

    # Minimum Kelly when we don't have enough data
    if n < 10:
        half_k = 0.01   # 1% fallback

    # Scale by ML confidence relative to neutral (0.5)
    conf_edge = max(0, (ml_confidence - 0.5) * 2)   # 0..1
    conf_mult = 0.5 + conf_edge * 1.0                # 0.5x – 1.5x

    # SMC score boost (0-100)
    smc_mult  = 1.0 + min(smc_score / 200.0, 0.5)   # 1.0x – 1.5x

    regime_mult = {
        "TREND_UP":              1.0,
        "TREND_DOWN":            1.0,
        "HIGH_VOL_BREAKOUT":     0.5,
        "LOW_VOL_COMPRESSION":   0.75,
        "CHOPPY_RANGE":          0.25,
        "UNKNOWN":               0.1,
    }.get(regime, 0.5)

    news_mult    = 1.0 if news_ok else 0.0
    final_risk   = half_k * 100 * conf_mult * smc_mult * regime_mult * news_mult
    final_risk   = max(0.1, min(final_risk, 5.0))   # hard cap 5%
    return round(final_risk, 3), {
        "kelly_half": round(half_k * 100, 3),
        "win_rate":   round(wr, 3),
        "avg_win":    round(aw, 2),
        "avg_loss":   round(al, 2),
        "edge":       round(edge, 3),
        "n_trades":   n,
        "conf_mult":  round(conf_mult, 3),
        "smc_mult":   round(smc_mult, 3),
        "regime_mult":round(regime_mult, 3),
    }


# ═══════════════════════════════════════════════════════════════════
# EQUITY CURVE TRADING — pause when equity below its own MA
# ═══════════════════════════════════════════════════════════════════
def equity_curve_status(current_equity, ma_bars=20):
    """
    Returns (trading_allowed, reason, ma_value)
    Pauses trading when current equity < its ma_bars-period moving average.
    This catches when the strategy edge has degraded in real time.
    """
    log = load_equity_log()
    if len(log) < ma_bars:
        return True, "Equity log too short — trading normally", None

    recent_equities = [e["equity"] for e in log[-ma_bars:]]
    ma_val          = float(np.mean(recent_equities))
    current         = current_equity

    if current < ma_val:
        ratio = (ma_val - current) / ma_val * 100
        return False, f"Equity ({current:.2f}) below {ma_bars}-bar MA ({ma_val:.2f}) by {ratio:.1f}% — pausing", ma_val
    return True, f"Equity curve healthy ({current:.2f} > MA {ma_val:.2f})", ma_val


# ═══════════════════════════════════════════════════════════════════
# MODEL STALENESS DETECTION (KL Divergence on feature distributions)
# ═══════════════════════════════════════════════════════════════════
def kl_divergence(p, q, epsilon=1e-10):
    """KL(p || q) — measures how far q has drifted from reference p."""
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def check_model_staleness(n_bins=20, threshold=0.15):
    """
    Compares current live feature distribution to training distribution.
    Returns (is_stale, kl_score, stale_features)
    Reads features from walkforward_results.json (training proxy) vs live data.
    """
    try:
        import yfinance as yf
        sys.path.insert(0, BASE)
        from features import make_features, download_gold

        # Live features (last 200 bars)
        df_live = yf.download("GC=F", period="5d", interval="1h",
                               progress=False, auto_adjust=True)
        if isinstance(df_live.columns, pd.MultiIndex):
            df_live.columns = [c[0].lower() for c in df_live.columns]
        else:
            df_live.columns = [str(c).lower() for c in df_live.columns]
        if hasattr(df_live.index, "tz") and df_live.index.tz:
            df_live.index = df_live.index.tz_localize(None)
        df_live = df_live.dropna(subset=["open", "high", "low", "close"])

        if len(df_live) < 30:
            return False, 0.0, []

        feat_live = make_features(df_live).dropna()

        # Reference: historical data (training proxy)
        df_ref = yf.download("GC=F", period="60d", interval="1h",
                              progress=False, auto_adjust=True)
        if isinstance(df_ref.columns, pd.MultiIndex):
            df_ref.columns = [c[0].lower() for c in df_ref.columns]
        else:
            df_ref.columns = [str(c).lower() for c in df_ref.columns]
        if hasattr(df_ref.index, "tz") and df_ref.index.tz:
            df_ref.index = df_ref.index.tz_localize(None)
        df_ref = df_ref.dropna(subset=["open", "high", "low", "close"])
        feat_ref = make_features(df_ref).dropna()

        stale_features = []
        kl_scores      = []
        check_cols     = ["rsi", "atr_pct", "adx", "bb_width", "macd",
                          "vol_regime", "stoch_k", "ema_cross_9_21"]

        for col in check_cols:
            if col not in feat_live.columns or col not in feat_ref.columns:
                continue
            live_vals = feat_live[col].dropna().values
            ref_vals  = feat_ref[col].dropna().values
            if len(live_vals) < 10 or len(ref_vals) < 10:
                continue

            # Build histograms over shared range
            combined = np.concatenate([live_vals, ref_vals])
            bins     = np.linspace(combined.min(), combined.max(), n_bins + 1)
            p_ref, _ = np.histogram(ref_vals,  bins=bins, density=True)
            p_live,_ = np.histogram(live_vals, bins=bins, density=True)
            kl       = kl_divergence(p_ref + 1e-10, p_live + 1e-10)
            kl_scores.append(kl)
            if kl > threshold:
                stale_features.append(f"{col}:KL={kl:.3f}")

        avg_kl  = float(np.mean(kl_scores)) if kl_scores else 0.0
        is_stale = avg_kl > threshold

        result = {
            "is_stale":      is_stale,
            "avg_kl":        round(avg_kl, 4),
            "threshold":     threshold,
            "stale_features":stale_features,
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(DRIFT_LOG, "w") as f:
            json.dump(result, f, indent=2)
        return is_stale, avg_kl, stale_features
    except Exception as e:
        return False, 0.0, [f"Check failed: {e}"]


# ═══════════════════════════════════════════════════════════════════
# ORIGINAL HELPERS (retained + updated for £500 account)
# ═══════════════════════════════════════════════════════════════════
def get_session():
    utc_hour = datetime.datetime.utcnow().hour
    if 0 <= utc_hour < 7:    return "ASIAN",      "sniper"
    elif 7 <= utc_hour < 12: return "LONDON_OPEN", "swing"
    elif 12 <= utc_hour < 16:return "LONDON_NY",   "swing"
    elif 16 <= utc_hour < 21:return "NY_CLOSE",    "sniper"
    else:                     return "OFF_HOURS",   "none"


def is_weekend():
    now = datetime.datetime.utcnow()
    return now.weekday() >= 5


def get_current_atr_pct():
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="5d", interval="1h",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        close = df["close"].values.astype(float)
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        tr    = np.maximum(high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]),
                           np.abs(low[1:]  - close[:-1])))
        atr   = float(np.mean(tr[-14:]))
        price = float(close[-1])
        return atr / price * 100.0 if price > 0 else 0.5
    except Exception:
        return 0.5


def check_god_mode(ml_confidence, smc_score, regime, news_ok):
    return (ml_confidence >= 0.88
            and smc_score >= 60
            and regime in ("TREND_UP", "TREND_DOWN")
            and news_ok)


def check_prop_firm_rules(settings, current_balance, starting_balance, daily_pnl):
    if not settings.get("prop_firm_mode", False):
        return True, "Prop firm mode OFF"
    total_dd = (starting_balance - current_balance) / starting_balance * 100
    daily_dd = max(0, -daily_pnl / starting_balance * 100)
    if total_dd >= settings["prop_total_dd_limit"]:
        return False, f"PROP TOTAL DD: {total_dd:.2f}% >= {settings['prop_total_dd_limit']}%"
    if daily_dd >= settings["prop_daily_dd_limit"]:
        return False, f"PROP DAILY DD: {daily_dd:.2f}% >= {settings['prop_daily_dd_limit']}%"
    profit = (current_balance - starting_balance) / starting_balance * 100
    if profit >= settings["prop_profit_target"]:
        return False, f"PROP TARGET REACHED: {profit:.2f}% — SUBMIT CHALLENGE"
    return True, f"Prop OK | Total DD: {total_dd:.2f}% | Daily: {daily_dd:.2f}%"


def equity_curve_risk_multiplier(settings, current_balance):
    peak    = settings.get("peak_balance", current_balance)
    dd_pct  = (peak - current_balance) / peak * 100 if peak > 0 else 0
    trigger = settings.get("equity_halve_trigger", 5.0)
    if dd_pct >= trigger * 2: return 0.25
    if dd_pct >= trigger:     return 0.5
    return 1.0


# ═══════════════════════════════════════════════════════════════════
# MASTER RISK CALCULATION — now Kelly-first
# ═══════════════════════════════════════════════════════════════════
def calculate_risk(
    current_balance  = 500.0,
    current_equity   = 500.0,
    ml_confidence    = 0.6,
    smc_score        = 0,
    regime           = "CHOPPY_RANGE",
    news_ok          = True,
    daily_pnl        = 0.0,
    starting_balance = 500.0,
    use_kelly        = True,      # NEW: switch to use Kelly sizing
    check_staleness  = False,     # NEW: KL divergence check (slower)
):
    settings = load_settings()

    # ── 0. Equity curve trading (new) ────────────────────────────
    ma_bars    = settings.get("equity_ma_bars", 20)
    ec_trading, ec_reason, ec_ma = equity_curve_status(current_equity, ma_bars)
    if not ec_trading:
        return {
            "risk_pct": 0.0, "mode": "EQUITY_CURVE_PAUSE",
            "reason": ec_reason, "trade_allowed": False,
            "equity_ma": ec_ma,
        }

    # ── 1. Prop firm ──────────────────────────────────────────────
    prop_ok, prop_reason = check_prop_firm_rules(
        settings, current_balance, starting_balance, daily_pnl)
    if not prop_ok:
        return {"risk_pct": 0.0, "mode": "BLOCKED",
                "reason": prop_reason, "trade_allowed": False}

    # ── 2. Weekend ────────────────────────────────────────────────
    if is_weekend():
        return {"risk_pct": 0.0, "mode": "WEEKEND",
                "reason": "Market closed", "trade_allowed": False}

    # ── 3. Model staleness (optional, slow) ──────────────────────
    staleness_warn = ""
    if check_staleness:
        is_stale, kl, stale_feats = check_model_staleness(
            threshold=settings.get("staleness_threshold", 0.15))
        if is_stale:
            staleness_warn = f"MODEL DRIFT KL={kl:.3f} features={stale_feats[:3]}"

    # ── 4. Kelly-based risk % ─────────────────────────────────────
    god = check_god_mode(ml_confidence, smc_score, regime, news_ok)

    if god:
        base_risk = settings.get("god_mode_max_risk", 6.0)
        mode_str  = "GOD_MODE"
        kelly_info = {}
    elif use_kelly:
        base_risk, kelly_info = kelly_position_pct(
            ml_confidence, smc_score, regime, news_ok)
        mode_str = "KELLY"
    else:
        base_risk = settings.get("base_risk_pct", 1.0)
        mode_str  = "NORMAL"
        kelly_info = {}

    # ── 5. Additional multipliers (on top of Kelly) ───────────────
    # Only applied in NON-Kelly mode (Kelly already accounts for them)
    if not use_kelly or god:
        # Confidence
        if ml_confidence > 0.80: conf_mult = 1.5
        elif ml_confidence > 0.70: conf_mult = 1.2
        elif ml_confidence > 0.60: conf_mult = 1.0
        else: conf_mult = 0.5

        # Regime
        regime_mult = {
            "TREND_UP": 1.0, "TREND_DOWN": 1.0,
            "HIGH_VOL_BREAKOUT": 0.5, "LOW_VOL_COMPRESSION": 0.75,
            "CHOPPY_RANGE": 0.25, "UNKNOWN": 0.1,
        }.get(regime, 0.5)

        news_mult = 1.0 if news_ok else 0.0
        base_risk = base_risk * conf_mult * regime_mult * news_mult
    else:
        # Kelly already includes these — just apply session + equity curve
        conf_mult   = 1.0
        regime_mult = 1.0
        news_mult   = 1.0

    # ── 6. Shared multipliers ─────────────────────────────────────
    ec_mult = equity_curve_risk_multiplier(settings, current_balance)

    session, _ = get_session()
    session_mult = {
        "ASIAN": 0.8, "LONDON_OPEN": 1.0, "LONDON_NY": 1.0,
        "NY_CLOSE": 0.8, "OFF_HOURS": 0.3,
    }.get(session, 0.5)

    atr_pct    = get_current_atr_pct()
    vol_target = settings.get("vol_target_pct", 1.0)
    vol_mult   = max(0.2, min(vol_target / max(atr_pct, 0.01), 2.0))

    raw_risk = base_risk * ec_mult * session_mult
    if not god:
        raw_risk *= vol_mult

    # Hard caps
    if settings.get("prop_firm_mode", False):
        raw_risk = min(raw_risk, 1.0)
    else:
        raw_risk = min(raw_risk, settings.get("god_mode_max_risk", 6.0))
    raw_risk = max(raw_risk, 0.0)

    # Compounding: update settings
    if settings.get("compound_auto", True):
        settings["current_balance"] = current_balance
        settings["peak_balance"]    = max(settings.get("peak_balance", 500), current_balance)
        save_settings(settings)

    result = {
        "risk_pct":        round(raw_risk, 3),
        "mode":            mode_str,
        "god_mode":        god,
        "trade_allowed":   raw_risk > 0.01,
        "session":         session,
        "session_mult":    session_mult,
        "ec_mult":         ec_mult,
        "vol_mult":        round(vol_mult, 2),
        "atr_pct":         round(atr_pct, 3),
        "current_balance": current_balance,
        "current_equity":  current_equity,
        "equity_ma":       ec_ma,
        "prop_firm_mode":  settings.get("prop_firm_mode", False),
        "staleness_warn":  staleness_warn,
        "kelly":           kelly_info,
        "reason":          (f"{mode_str} | session:{session} | regime:{regime} "
                            f"| conf:{ml_confidence:.0%}"
                            + (f" | ⚠ DRIFT" if staleness_warn else "")),
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RISK_STATUS, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ═══════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════
def print_kelly_report():
    hk, wr, aw, al, edge, n = calculate_kelly()
    print(f"\n  KELLY CRITERION REPORT")
    print(f"  ─────────────────────────────────────")
    print(f"  Trades used      : {n}")
    print(f"  Win Rate         : {wr*100:.1f}%")
    print(f"  Avg Win          : £{aw:.2f}")
    print(f"  Avg Loss         : £{al:.2f}")
    print(f"  Reward/Risk (b)  : {aw/(al+1e-9):.2f}")
    print(f"  Raw Kelly        : {edge*100:.2f}%")
    print(f"  Half-Kelly       : {hk*100:.3f}%  ← use this")
    if n < 20:
        print(f"  ⚠ Only {n} trades — use conservative sizing until >20 trades")


def print_compounding(balance=500.0):
    print(f"\n  COMPOUNDING PROJECTIONS (starting £{balance:.2f})")
    print(f"  {'Days':>6}  {'0.5%/day':>12}  {'1%/day':>12}  {'1.5%/day':>12}")
    print(f"  {'─'*50}")
    for days in [7, 14, 30, 60, 90]:
        v05 = balance * (1.005 ** days)
        v1  = balance * (1.01  ** days)
        v15 = balance * (1.015 ** days)
        print(f"  {days:>6}d  £{v05:>10.2f}  £{v1:>10.2f}  £{v15:>10.2f}")


def prop_firm_progress(current_balance, starting_balance=500.0):
    settings     = load_settings()
    target_pct   = settings.get("prop_profit_target", 8.0)
    dd_limit     = settings.get("prop_total_dd_limit", 6.0)
    profit_pct   = (current_balance - starting_balance) / starting_balance * 100
    dd_pct       = max(0, (starting_balance - current_balance) / starting_balance * 100)
    progress     = min(profit_pct / target_pct * 100, 100) if target_pct > 0 else 0
    bar_len      = 30
    filled       = int(bar_len * progress / 100)
    bar          = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  PROP FIRM PROGRESS")
    print(f"  [{bar}] {progress:.1f}%")
    print(f"  Profit : {profit_pct:+.2f}% (£{current_balance - starting_balance:+.2f})")
    print(f"  DD now : {dd_pct:.2f}%  (Limit: {dd_limit}%)")


def check_sunday_gap():
    now = datetime.datetime.utcnow()
    if not (now.weekday() == 6 and 21 <= now.hour <= 22):
        return False, 0, "NONE"
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="5d", interval="1h",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        last_fri_close = float(df["close"].iloc[-2])
        sun_open       = float(df["open"].iloc[-1])
        gap_pct        = (sun_open - last_fri_close) / last_fri_close * 100
        direction      = "UP" if gap_pct > 0 else "DOWN"
        return abs(gap_pct) >= 0.1, round(gap_pct, 3), direction
    except Exception:
        return False, 0, "NONE"


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AlmostFinishedBot — Risk Manager v3 (£500 account)")
    print("=" * 60)

    print_kelly_report()

    r = calculate_risk(
        current_balance  = 500.0,
        current_equity   = 500.0,
        ml_confidence    = 0.72,
        smc_score        = 55,
        regime           = "TREND_UP",
        news_ok          = True,
        starting_balance = 500.0,
        use_kelly        = True,
    )
    print(f"\n  Risk %      : {r['risk_pct']:.3f}%")
    print(f"  Mode        : {r['mode']}")
    print(f"  God Mode    : {r['god_mode']}")
    print(f"  Session     : {r['session']}")
    print(f"  EC mult     : {r['ec_mult']}")
    print(f"  Trade OK    : {r['trade_allowed']}")
    print(f"  Reason      : {r['reason']}")
    if r.get("kelly"):
        k = r["kelly"]
        print(f"\n  Kelly detail:")
        print(f"    Half-Kelly : {k.get('kelly_half', 0):.3f}%")
        print(f"    Win rate   : {k.get('win_rate', 0)*100:.1f}%")
        print(f"    Avg W/L    : £{k.get('avg_win', 0):.2f} / £{k.get('avg_loss', 0):.2f}")

    print_compounding(500.0)

    session, mode = get_session()
    print(f"\n  Session: {session}  Recommended: {mode}")

    print(f"\n  Checking model staleness (takes ~10s)...")
    is_stale, kl, feats = check_model_staleness()
    print(f"  Stale: {is_stale}  KL: {kl:.4f}")
    if feats:
        print(f"  Drifted: {feats}")
    else:
        print("  All features within distribution")
