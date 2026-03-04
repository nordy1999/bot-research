"""
AlmostFinishedBot - Signal Sync
Copies signal files from bot directory to MT5 Files directory.
Run this as a background process to keep MT5 in sync.
"""
import os, shutil, time

BOT_DIR  = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")
MT5_DIR  = os.path.join(os.environ.get("APPDATA",""), "MetaQuotes", "Terminal",
           "73B7A2420D6397DFF9014A20F1201F97", "MQL5", "Files", "AlmostFinishedBot")

SYNC_FILES = ["current_signal.json", "regime_cache.json", "news_cache.json",
              "smc_cache.json", "news_guard.json", "correlation_status.json",
              "bot_settings.json", "risk_status.json"]

os.makedirs(MT5_DIR, exist_ok=True)
print("Syncing: " + BOT_DIR + " -> " + MT5_DIR)
print("Files: " + ", ".join(SYNC_FILES))
print("Press Ctrl+C to stop")
print("")

while True:
    for f in SYNC_FILES:
        src = os.path.join(BOT_DIR, f)
        dst = os.path.join(MT5_DIR, f)
        try:
            if os.path.exists(src):
                src_mtime = os.path.getmtime(src)
                dst_mtime = os.path.getmtime(dst) if os.path.exists(dst) else 0
                if src_mtime > dst_mtime:
                    shutil.copy2(src, dst)
        except Exception:
            pass
    time.sleep(2)
