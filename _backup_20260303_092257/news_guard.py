"""
AlmostFinishedBot - News Event Guard
Blocks trading 30 min before/after FOMC, NFP, CPI, PPI, GDP, Fed speeches
Writes news_guard.json that the EA reads every tick
"""
import json, os, sys, datetime, time
BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try: import requests
except: install("requests"); import requests

GUARD_FILE   = os.path.join(BASE, "news_guard.json")
BUFFER_MINS  = 30
HIGH_IMPACT  = [
    "non-farm","nonfarm","nfp","fomc","federal reserve","fed rate",
    "cpi","consumer price","ppi","producer price","gdp","retail sales",
    "jolts","initial claims","unemployment","powell","fed chair",
    "interest rate decision","core pce","ism manufacturing",
    "geopolit","war","sanctions"
]
# Known FOMC dates 2026 (UTC 19:00 on decision days)
FOMC_2026 = ["2026-01-28","2026-03-18","2026-05-06","2026-06-17",
             "2026-07-29","2026-09-16","2026-10-28","2026-12-16"]

def fetch_events():
    events = []
    now    = datetime.datetime.utcnow()
    today  = now.strftime("%Y-%m-%d")

    # Hardcoded recurring high-impact times (UTC)
    weekday = now.weekday()
    if today in FOMC_2026:
        events.append({"title":"FOMC Rate Decision",  "time":"19:00","country":"USD"})
        events.append({"title":"FOMC Press Conference","time":"19:30","country":"USD"})
    if weekday == 4:  # Friday — potential NFP (first Friday of month)
        if 1 <= now.day <= 7:
            events.append({"title":"Non-Farm Payrolls","time":"13:30","country":"USD"})

    # Try live ForexFactory feed
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=6, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200:
            for ev in r.json():
                ev_date = ev.get("date","")[:10]
                if ev_date != today: continue
                if ev.get("impact","").lower() != "high": continue
                title = ev.get("title","").lower()
                if not any(kw in title for kw in HIGH_IMPACT): continue
                events.append({"title":ev.get("title",""),
                               "time": ev.get("time",""),
                               "country":ev.get("country","USD")})
    except Exception:
        pass

    return events

def is_blocked():
    now    = datetime.datetime.utcnow()
    events = fetch_events()
    result = {"blocked":False,"reduced":False,"reason":"Clear",
              "minutes_to_event":999,"next_event":"None",
              "events_today":len(events),"timestamp":now.isoformat()}

    for ev in events:
        t_str = ev.get("time","")
        if not t_str or t_str.lower() in ("tentative","all day",""):
            result["reduced"] = True
            result["reason"]  = f"Tentative event: {ev['title']}"
            continue
        try:
            ev_dt   = datetime.datetime.strptime(now.strftime("%Y-%m-%d")+" "+t_str, "%Y-%m-%d %H:%M")
            diff    = (ev_dt - now).total_seconds() / 60.0

            if -BUFFER_MINS <= diff <= BUFFER_MINS:
                result["blocked"] = True
                result["reason"]  = f"NEWS GUARD: {ev['title']} ({t_str} UTC)"
                result["minutes_to_event"] = int(diff)
                result["next_event"] = ev["title"]
                break
            elif 0 < diff < BUFFER_MINS * 3:
                if diff < result["minutes_to_event"]:
                    result["minutes_to_event"] = int(diff)
                    result["next_event"]        = ev["title"]
                if diff < BUFFER_MINS * 1.5:
                    result["reduced"] = True
                    result["reason"]  = f"Pre-event caution: {ev['title']} in {int(diff)} min"
        except Exception:
            continue

    os.makedirs(BASE, exist_ok=True)
    with open(GUARD_FILE,"w") as f: json.dump(result,f,indent=2)
    return result

def run_loop():
    print("="*60)
    print("  AlmostFinishedBot  |  News Event Guard")
    print("  Updates every 2 minutes  |  Ctrl+C to stop")
    print("="*60)
    while True:
        try:
            r = is_blocked()
            s = "BLOCKED" if r["blocked"] else ("REDUCED" if r["reduced"] else "CLEAR ")
            nxt = f"  Next: {r['next_event']} in {r['minutes_to_event']} min" if r["next_event"]!="None" else ""
            print(f"  [{time.strftime('%H:%M:%S')} UTC]  {s}  —  {r['reason']}{nxt}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(120)

if __name__ == "__main__":
    if "check" in sys.argv: print(json.dumps(is_blocked(),indent=2))
    else: run_loop()
