"""
AlmostFinishedBot - Trade Monitor
Cleans MT5 spaced-character log output, shows only meaningful events
"""
import os, sys, time, re

BASE   = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
LOGDIR = os.path.join(os.environ.get("APPDATA",""), "MetaQuotes", "Terminal",
         "73B7A2420D6397DFF9014A20F1201F97", "MQL5", "Logs")

mode = sys.argv[1] if len(sys.argv) > 1 else "paper"
BANNERS = {
    "paper":    ("PAPER TRADE MONITOR", "Demo 62111276"),
    "live":     ("LIVE TRADE MONITOR",  "*** REAL MONEY ***"),
    "backtest": ("BACKTEST MONITOR",    "$1,000 deposit"),
}
title, subtitle = BANNERS.get(mode, ("MONITOR",""))

def clean_mt5(line):
    """Remove MT5's UTF-16 spaced-character format: 'U l t r a' -> 'Ultra'"""
    line = re.sub(r'(?<=\S) (?=\S)', '', line)
    return re.sub(r'\s+', ' ', line).strip()

os.system("cls" if os.name=="nt" else "clear")
print("=" * 70)
print(f"  AlmostFinishedBot  |  {title}")
print(f"  {subtitle}  |  Ctrl+C to stop")
print("=" * 70)

SUPPRESS = ["trail buy error","trail sell error","ml ensemble not found",
            "fallback strategy","ml signal files not found"]
SHOW_ONCE = {}
counts = {"BUY":0,"SELL":0,"errors":0,"suppressed":0}

IMPORTANT_KW = [
    "buy |","sell |",">>> buy",">>> sell",
    "ordersend","order opened","order closed","order modified",
    "balance:","equity","profit","loss",
    "kelly","regime","confidence","risk:",
    "order block","liquidity sweep","fvg","fair value",
    "swing mode","sniper mode","day mode",
    "oninit finished","ondeinit",
    "telegram sent","error 4"
]

def latest_log():
    try:
        files = sorted([f for f in os.listdir(LOGDIR) if f.endswith(".log")])
        return os.path.join(LOGDIR, files[-1]) if files else None
    except: return None

last_log = None; last_size = 0

print(f"\n  Watching: {LOGDIR}")
print(f"  Waiting for MT5 activity...\n")

while True:
    try:
        lp = latest_log()
        if lp != last_log:
            last_log = lp; last_size = 0; SHOW_ONCE.clear()
            if lp:
                print(f"\n  [LOG] {os.path.basename(lp)}")
                print("  " + "-" * 66)

        if lp and os.path.exists(lp):
            sz = os.path.getsize(lp)
            if sz > last_size:
                with open(lp,"r",encoding="utf-8",errors="ignore") as f:
                    f.seek(last_size); raw = f.read()
                last_size = sz
                for raw_line in raw.splitlines():
                    if not raw_line.strip(): continue
                    line = clean_mt5(raw_line)
                    if not line: continue
                    low  = line.lower()

                    # Check suppress
                    supp = any(s in low for s in SUPPRESS)
                    if supp:
                        key = next(s for s in SUPPRESS if s in low)
                        if key not in SHOW_ONCE:
                            SHOW_ONCE[key] = True
                            print(f"  [INFO]  {line}  (suppressing repeats)")
                        else:
                            counts["suppressed"] += 1
                        continue

                    # BUY signal
                    if any(x in low for x in ["xauusd buy","xausgd buy","buy |",">>> buy"]):
                        print(f"\n  {'='*10} BUY SIGNAL {'='*10}")
                        print(f"  {line}")
                        print(f"  {'='*31}\n")
                        counts["BUY"] += 1

                    # SELL signal
                    elif any(x in low for x in ["xauusd sell","xausgd sell","sell |",">>> sell"]):
                        print(f"\n  {'='*10} SELL SIGNAL {'='*9}")
                        print(f"  {line}")
                        print(f"  {'='*31}\n")
                        counts["SELL"] += 1

                    # Errors (but not trail errors - those are suppressed)
                    elif "error" in low and "trail" not in low:
                        print(f"  [ERR ]  {line}")
                        counts["errors"] += 1

                    # Important events
                    elif any(kw in low for kw in IMPORTANT_KW):
                        print(f"  {line}")

        time.sleep(0.4)

    except KeyboardInterrupt:
        sup = counts["suppressed"]
        print(f"\n  {'='*40}")
        print(f"  Session Summary:")
        print(f"    BUY signals : {counts['BUY']}")
        print(f"    SELL signals: {counts['SELL']}")
        print(f"    Errors      : {counts['errors']}")
        print(f"    Suppressed  : {sup} repetitive lines hidden")
        print(f"  {'='*40}")
        break
    except Exception as e:
        print(f"  [ERR] Monitor error: {e}")
        time.sleep(2)
