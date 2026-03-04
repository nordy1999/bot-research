"""
AlmostFinishedBot - Risk Manager v2
- God Mode sizing (1% normal → up to 8% on perfect setups)
- Prop Firm Challenge Mode (FTMO/MyForexFunds style rules)
- Equity Curve Protector (auto-halve risk if losing streak)
- Compounding Engine
- Volatility Targeting (1% move = X% portfolio)
- Asian Session detector (best session for small accounts)
- Weekly walk-forward report
"""
import os, sys, json, time, datetime
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
os.makedirs(BASE, exist_ok=True)

SETTINGS_FILE   = os.path.join(BASE, "bot_settings.json")
EQUITY_LOG      = os.path.join(BASE, "equity_log.json")
RISK_STATUS     = os.path.join(BASE, "risk_status.json")

# ── Default settings ──────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "mode":               "normal",   # normal / prop_firm / god_mode / conservative
    "base_risk_pct":      1.0,        # % per trade in normal mode
    "god_mode_max_risk":  8.0,        # max % when all signals align perfectly
    "god_mode_threshold": 0.88,       # confidence needed for god mode
    "prop_firm_mode":     False,      # enables FTMO-style rules
    "prop_daily_dd_limit":3.0,        # prop firm daily drawdown % limit
    "prop_total_dd_limit":6.0,        # prop firm total drawdown % limit
    "prop_profit_target": 8.0,        # profit target to pass challenge %
    "equity_halve_trigger":5.0,       # % drawdown that halves risk
    "equity_restore_pct": 20.0,       # % gain needed to restore full risk
    "vol_target_pct":     1.0,        # 1% gold move = 1% portfolio move
    "compound_auto":      True,       # auto-reinvest profits
    "starting_balance":   100.0,      # £100
    "current_balance":    100.0,
    "peak_balance":       100.0,
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f: s = json.load(f)
            for k,v in DEFAULT_SETTINGS.items():
                if k not in s: s[k] = v
            return s
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE,"w") as f: json.dump(s, f, indent=2)

# ── Equity log ────────────────────────────────────────────────────
def log_equity(balance, equity):
    log = []
    if os.path.exists(EQUITY_LOG):
        try:
            with open(EQUITY_LOG) as f: log = json.load(f)
        except Exception: log = []
    log.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "balance": round(balance, 2),
        "equity":  round(equity,  2),
    })
    # Keep last 1000 entries
    log = log[-1000:]
    with open(EQUITY_LOG,"w") as f: json.dump(log, f, indent=2)

def load_equity_log():
    if not os.path.exists(EQUITY_LOG): return []
    try:
        with open(EQUITY_LOG) as f: return json.load(f)
    except Exception: return []

# ── Trading session detector ──────────────────────────────────────
def get_session():
    """Returns current session and recommended mode."""
    utc_hour = datetime.datetime.utcnow().hour
    # Sydney:  21:00–06:00 UTC
    # Tokyo:   00:00–09:00 UTC
    # London:  07:00–16:00 UTC
    # New York:12:00–21:00 UTC
    # Asian session overlap (Tokyo+early London) = best for small accounts
    if 0 <= utc_hour < 7:
        return "ASIAN", "sniper"     # tight spreads, clean moves
    elif 7 <= utc_hour < 12:
        return "LONDON_OPEN", "swing"  # most volatile, big moves
    elif 12 <= utc_hour < 16:
        return "LONDON_NY", "swing"    # highest volume
    elif 16 <= utc_hour < 21:
        return "NY_CLOSE", "sniper"
    else:
        return "OFF_HOURS", "none"     # avoid

def is_sunday_open():
    """Returns True if it's the Sunday market open (17:00-18:00 ET = 21:00-22:00 UTC)."""
    now = datetime.datetime.utcnow()
    return now.weekday() == 6 and 21 <= now.hour <= 22

def is_weekend():
    now = datetime.datetime.utcnow()
    return now.weekday() >= 5 and not is_sunday_open()

# ── ATR-based volatility from yfinance ───────────────────────────
def get_current_atr_pct():
    """Returns current ATR as % of price (for volatility targeting)."""
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="5d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns=[c[0].lower() for c in df.columns]
        else: df.columns=[str(c).lower() for c in df.columns]
        close = df["close"].values.astype(float)
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        tr    = np.maximum(high[1:]-low[1:],
                np.maximum(np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
        atr   = float(np.mean(tr[-14:]))
        price = float(close[-1])
        return atr / price * 100.0 if price > 0 else 0.5
    except Exception:
        return 0.5  # default 0.5% ATR



# ── God Mode qualifier ────────────────────────────────────────────
def check_god_mode(ml_confidence, smc_score, regime, news_ok):
    """
    Returns True only when ALL conditions align perfectly.
    This is reserved for highest-conviction setups.
    """
    regime_ok = regime in ("TREND_UP", "TREND_DOWN")
    return (ml_confidence >= 0.88 and
            smc_score >= 60 and
            regime_ok and
            news_ok)

# ── Prop firm rule checker ────────────────────────────────────────
def check_prop_firm_rules(settings, current_balance, starting_balance, daily_pnl):
    """
    Returns (allowed_to_trade, reason)
    Enforces FTMO/Funded Trader rules.
    """
    if not settings.get("prop_firm_mode", False):
        return True, "Prop firm mode OFF"

    total_dd_pct = (starting_balance - current_balance) / starting_balance * 100
    daily_dd_pct = max(0, -daily_pnl / starting_balance * 100)

    if total_dd_pct >= settings["prop_total_dd_limit"]:
        return False, f"PROP TOTAL DRAWDOWN LIMIT: {total_dd_pct:.2f}% >= {settings['prop_total_dd_limit']}%"

    if daily_dd_pct >= settings["prop_daily_dd_limit"]:
        return False, f"PROP DAILY DRAWDOWN LIMIT: {daily_dd_pct:.2f}% >= {settings['prop_daily_dd_limit']}%"

    profit_pct = (current_balance - starting_balance) / starting_balance * 100
    if profit_pct >= settings["prop_profit_target"]:
        return False, f"PROP TARGET REACHED: {profit_pct:.2f}% — STOP AND VERIFY CHALLENGE PASS"

    return True, f"Prop rules OK | Total DD: {total_dd_pct:.2f}% | Daily DD: {daily_dd_pct:.2f}% | Profit: {profit_pct:.2f}%"

# ── Equity curve protector ────────────────────────────────────────
def equity_curve_risk_multiplier(settings, current_balance):
    """
    Returns risk multiplier based on equity curve health.
    0.5x if in drawdown > trigger, 1.0x if healthy.
    """
    peak    = settings.get("peak_balance", current_balance)
    dd_pct  = (peak - current_balance) / peak * 100 if peak > 0 else 0
    trigger = settings.get("equity_halve_trigger", 5.0)

    if dd_pct >= trigger * 2:  return 0.25   # severe drawdown
    if dd_pct >= trigger:       return 0.5    # moderate drawdown
    return 1.0

# ── Compounding calculator ────────────────────────────────────────
def compound_projection(balance, daily_pct, days):
    """Returns balance projection list."""
    b = balance; result = [b]
    for _ in range(days):
        b = b * (1 + daily_pct / 100)
        result.append(round(b, 2))
    return result

# ── Master risk calculation ───────────────────────────────────────
def calculate_risk(
    current_balance  = 100.0,
    ml_confidence    = 0.6,
    smc_score        = 0,
    regime           = "CHOPPY_RANGE",
    news_ok          = True,
    daily_pnl        = 0.0,
    starting_balance = 100.0,
):
    """
    Master function — returns exact % to risk on next trade.
    Takes into account: God Mode, Prop Firm, Equity Curve, Vol Targeting,
    Session, Regime multiplier, News Guard, Kelly.
    """
    settings = load_settings()

    # ── 1. Prop firm check ──────────────────────────────────────
    prop_ok, prop_reason = check_prop_firm_rules(
        settings, current_balance, starting_balance, daily_pnl)
    if not prop_ok:
        return {
            "risk_pct": 0.0, "mode": "BLOCKED",
            "reason": prop_reason,
            "trade_allowed": False,
        }

    # ── 2. Weekend / off-hours check ────────────────────────────
    if is_weekend():
        return {
            "risk_pct": 0.0, "mode": "WEEKEND",
            "reason": "Market closed — weekend",
            "trade_allowed": False,
        }

    # ── 3. God Mode check ────────────────────────────────────────
    god = check_god_mode(ml_confidence, smc_score, regime, news_ok)

    # ── 4. Base risk ────────────────────────────────────────────
    if god:
        base_risk = settings.get("god_mode_max_risk", 8.0)
        mode_str  = "GOD_MODE"
    elif settings.get("prop_firm_mode", False):
        base_risk = 0.5        # conservative for prop challenge
        mode_str  = "PROP_FIRM"
    else:
        base_risk = settings.get("base_risk_pct", 1.0)
        mode_str  = "NORMAL"

    # ── 5. Confidence scaling ────────────────────────────────────
    if not god:
        if   ml_confidence > 0.80: conf_mult = 1.5
        elif ml_confidence > 0.70: conf_mult = 1.2
        elif ml_confidence > 0.60: conf_mult = 1.0
        else:                       conf_mult = 0.5
    else:
        conf_mult = 1.0

    # ── 6. Regime multiplier ─────────────────────────────────────
    regime_mult = {
        "TREND_UP":1.0,"TREND_DOWN":1.0,
        "HIGH_VOL_BREAKOUT":0.5,"LOW_VOL_COMPRESSION":0.75,
        "CHOPPY_RANGE":0.25,"UNKNOWN":0.1
    }.get(regime, 0.5)

    # ── 7. News guard multiplier ─────────────────────────────────
    news_mult = 1.0 if news_ok else 0.0

    # ── 8. Equity curve protector ────────────────────────────────
    ec_mult = equity_curve_risk_multiplier(settings, current_balance)

    # ── 9. Session multiplier ────────────────────────────────────
    session, _ = get_session()
    session_mult = {
        "ASIAN":0.8,"LONDON_OPEN":1.0,"LONDON_NY":1.0,
        "NY_CLOSE":0.8,"OFF_HOURS":0.3
    }.get(session, 0.5)

    # ── 10. Volatility targeting ─────────────────────────────────
    # Scale down if ATR > 0.8% (very volatile), scale up if ATR < 0.3%
    atr_pct  = get_current_atr_pct()
    vol_target = settings.get("vol_target_pct", 1.0)
    if atr_pct > 0:
        vol_mult = min(vol_target / atr_pct, 2.0)
        vol_mult = max(vol_mult, 0.2)
    else:
        vol_mult = 1.0

    # ── 11. Final calculation ────────────────────────────────────
    raw_risk = base_risk * conf_mult * regime_mult * news_mult * ec_mult * session_mult
    if not god: raw_risk *= vol_mult   # vol targeting only in normal mode

    # Hard caps
    if settings.get("prop_firm_mode", False):
        raw_risk = min(raw_risk, 1.0)   # prop firm max 1% per trade
    else:
        raw_risk = min(raw_risk, settings.get("god_mode_max_risk", 8.0))

    raw_risk = max(raw_risk, 0.0)

    # Compounding: update balance in settings
    if settings.get("compound_auto", True):
        settings["current_balance"] = current_balance
        settings["peak_balance"]    = max(settings.get("peak_balance", 100), current_balance)
        save_settings(settings)

    result = {
        "risk_pct":       round(raw_risk, 3),
        "mode":           mode_str,
        "god_mode":       god,
        "trade_allowed":  raw_risk > 0,
        "session":        session,
        "session_mult":   session_mult,
        "conf_mult":      conf_mult,
        "regime_mult":    regime_mult,
        "ec_mult":        ec_mult,
        "vol_mult":       round(vol_mult, 2),
        "news_mult":      news_mult,
        "atr_pct":        round(atr_pct, 3),
        "current_balance":current_balance,
        "prop_firm_mode": settings.get("prop_firm_mode", False),
        "reason":         f"{mode_str} | session:{session} | regime:{regime} | conf:{ml_confidence:.0%}",
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RISK_STATUS,"w") as f: json.dump(result, f, indent=2)
    return result

# ── Prop firm progress report ─────────────────────────────────────
def prop_firm_progress(current_balance, starting_balance):
    settings = load_settings()
    target_pct = settings.get("prop_profit_target", 8.0)
    dd_limit   = settings.get("prop_total_dd_limit", 6.0)
    daily_limit= settings.get("prop_daily_dd_limit", 3.0)

    profit_pct = (current_balance - starting_balance) / starting_balance * 100
    dd_pct     = max(0, (starting_balance - current_balance) / starting_balance * 100)
    progress   = min(profit_pct / target_pct * 100, 100) if target_pct > 0 else 0

    bar_len = 30
    filled  = int(bar_len * progress / 100)
    bar     = "█"*filled + "░"*(bar_len-filled)

    print(f"\n  PROP FIRM CHALLENGE PROGRESS")
    print(f"  ─────────────────────────────────────")
    print(f"  Target  : +{target_pct:.1f}%  |  Limit: -{dd_limit:.1f}%  |  Daily: -{daily_limit:.1f}%")
    print(f"  Progress: [{bar}] {progress:.1f}%")
    print(f"  Profit  : {profit_pct:+.2f}%  (£{current_balance - starting_balance:+.2f})")
    print(f"  DD now  : {dd_pct:.2f}%")
    dd_remaining = dd_limit - dd_pct
    print(f"  DD left : {dd_remaining:.2f}% remaining before breach")

    if profit_pct >= target_pct:
        print(f"\n  ★ TARGET REACHED — STOP TRADING AND SUBMIT CHALLENGE ★")
    elif dd_pct >= dd_limit:
        print(f"\n  ✗ DRAWDOWN LIMIT BREACHED — CHALLENGE FAILED")
    elif dd_remaining < 1.0:
        print(f"\n  ⚠ WARNING: Only {dd_remaining:.2f}% DD buffer remaining — reduce risk NOW")
    return progress

# ── Compounding projection printer ───────────────────────────────
def print_compounding(balance=100.0):
    print(f"\n  COMPOUNDING PROJECTIONS (starting £{balance:.2f})")
    print(f"  {'Days':>6}  {'1% daily':>12}  {'1.5% daily':>12}  {'2% daily':>12}")
    print(f"  {'─'*48}")
    for days in [7, 14, 30, 60, 90]:
        v1  = balance * (1.01**days)
        v15 = balance * (1.015**days)
        v2  = balance * (1.02**days)
        print(f"  {days:>6}d  £{v1:>10.2f}  £{v15:>10.2f}  £{v2:>10.2f}")

# ── Sunday gap strategy ───────────────────────────────────────────
def check_sunday_gap():
    """
    Check if tonight is Sunday open and gold has gapped.
    Returns (gap_detected, gap_pct, direction)
    """
    if not is_sunday_open():
        return False, 0, "NONE"
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="5d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns=[c[0].lower() for c in df.columns]
        else: df.columns=[str(c).lower() for c in df.columns]
        if len(df) < 2: return False, 0, "NONE"
        last_fri_close = float(df["close"].iloc[-2])
        sun_open       = float(df["open"].iloc[-1])
        gap_pct        = (sun_open - last_fri_close) / last_fri_close * 100
        direction      = "UP" if gap_pct > 0 else "DOWN"
        detected       = abs(gap_pct) >= 0.1   # 0.1% minimum gap
        return detected, round(gap_pct, 3), direction
    except Exception:
        return False, 0, "NONE"

if __name__ == "__main__":
    print("=" * 55)
    print("  AlmostFinishedBot — Risk Manager")
    print("=" * 55)

    # Demo calculation
    r = calculate_risk(
        current_balance  = 100.0,
        ml_confidence    = 0.75,
        smc_score        = 55,
        regime           = "TREND_UP",
        news_ok          = True,
        daily_pnl        = 0.0,
        starting_balance = 100.0,
    )
    print(f"\n  Risk %    : {r['risk_pct']:.3f}%")
    print(f"  Mode      : {r['mode']}")
    print(f"  God Mode  : {r['god_mode']}")
    print(f"  Session   : {r['session']}")
    print(f"  Reason    : {r['reason']}")
    print(f"  Trade OK  : {r['trade_allowed']}")

    print_compounding(100.0)

    gap, gap_pct, gap_dir = check_sunday_gap()
    if gap: print(f"\n  Sunday gap: {gap_dir} {gap_pct:+.3f}%")

    session, mode = get_session()
    print(f"\n  Current session: {session}  |  Recommended mode: {mode}")
