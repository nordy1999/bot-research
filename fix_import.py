"""Run once to fix features import in live_trading_bridge.py"""
import os

path = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot", "live_trading_bridge.py")
code = open(path).read()

old = "from features import make_features as build_features"
lines = [
    "from features import make_features as _mf",
    "def build_features(df):",
    "    df = df.copy()",
    "    if hasattr(df.columns, 'droplevel') and df.columns.nlevels > 1:",
    "        df.columns = df.columns.droplevel(1)",
    "    df.columns = [c.lower() for c in df.columns]",
    "    return _mf(df)",
]
new = "\n".join(lines)

code = code.replace(old, new)
open(path, "w").write(code)
print("PATCHED OK")
