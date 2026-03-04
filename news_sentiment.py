"""
AlmostFinishedBot - News Sentiment Analyser
Tier 2: VADER sentiment on gold-related news feeds
"""
import json, os, sys, datetime
BASE = os.path.join(os.path.expanduser("~"), "Desktop", "AlmostFinishedBot")

def install(pkg):
    import subprocess; subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

for pkg in ["feedparser","vaderSentiment"]:
    try: __import__(pkg)
    except: install(pkg)

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FEEDS = [
    ("Kitco",    "https://www.kitco.com/rss/rssfeeds/headline_news.xml"),
    ("Reuters",  "https://feeds.reuters.com/reuters/businessNews"),
    ("FXStreet", "https://www.fxstreet.com/rss"),
    ("MarketW",  "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
]
GOLD_KW = ["gold","xau","bullion","precious metal","fed","inflation","fomc","dollar","usd","rate"]

def score_articles():
    vader = SentimentIntensityAnalyzer()
    arts  = []
    for src, url in FEEDS:
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            for e in feed.entries[:20]:
                t = (e.get("title","") + " " + e.get("summary",""))[:400]
                if not any(kw in t.lower() for kw in GOLD_KW): continue
                s = vader.polarity_scores(t)["compound"]
                arts.append({"title": e.get("title","")[:90], "source": src,
                             "score": round(s,3), "url": e.get("link","")})
        except Exception as ex:
            arts.append({"title":f"Feed error ({src})", "source":src, "score":0.0, "url":""})

    if not arts:
        result = {"signal":"NEUTRAL","score":0.0,"articles_analysed":0,
                  "top_articles":[],"timestamp":datetime.datetime.utcnow().isoformat()}
    else:
        arts.sort(key=lambda x: abs(x["score"]), reverse=True)
        avg = sum(a["score"] for a in arts) / len(arts)
        sig = "BUY" if avg>0.08 else ("SELL" if avg<-0.08 else "NEUTRAL")
        result = {"signal":sig,"score":round(avg,4),"articles_analysed":len(arts),
                  "top_articles":arts[:8],"timestamp":datetime.datetime.utcnow().isoformat()}

    os.makedirs(BASE, exist_ok=True)
    with open(os.path.join(BASE,"news_cache.json"),"w") as f:
        json.dump(result, f, indent=2)
    print(f"News: {result['signal']} ({result['score']:+.3f}) from {result['articles_analysed']} articles")
    return result

if __name__ == "__main__":
    score_articles()
