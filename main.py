"""Bitcoin Events Bot — daily news scanner that extracts significant BTC events.

Fetches news from free-crypto-news (cryptocurrency.cv) API, filters by keywords,
and serves a JSON endpoint for the iOS ticker app.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Bitcoin Events Bot")
scheduler = AsyncIOScheduler()

EVENTS_FILE = Path("events.json")
NEWS_BASE_URL = "https://cryptocurrency.cv/api"

# Trusted sources — mainstream + quality crypto media
TRUSTED_SOURCES = {
    # Tier 1 — mainstream/institutional
    "bloomberg", "reuters", "wsj", "wall_street_journal", "cnbc", "financial_times",
    "ft", "nytimes", "new_york_times", "bbc", "ap_news", "associated_press",
    "federal_reserve", "federal_reserve_notes", "sec_gov",
    # Tier 2 — premium crypto media
    "coindesk", "the_block", "blockworks", "decrypt", "cointelegraph",
    "bitcoin_magazine", "bitcoinist",
    "cryptoslate", "cryptonews", "cryptopotato",
}

# Keywords that indicate a significant BTC event (not just noise)
EVENT_KEYWORDS = [
    # Protocol & network
    "halving", "halvening", "block reward",
    "taproot", "segwit", "lightning network", "soft fork", "hard fork",
    # Regulation & legal
    "etf approved", "etf rejected", "etf approval", "etf decision", "spot etf",
    "sec ", "cftc", "ban crypto", "crypto ban", "legal tender",
    "strategic reserve", "bitcoin reserve",
    # Market events
    "all-time high", "ath", "record high",
    "crash", "flash crash", "black swan", "liquidat",
    "bankrupt", "collapse", "hack", "exploit", "stolen",
    # Institutional
    "microstrategy", "strategy ", "saylor",
    "tesla bitcoin", "el salvador",
    # Macro
    "fed rate", "fomc", "interest rate", "cpi ", "inflation data",
    "tariff", "trade war",
]

# Words that indicate noise rather than events
NOISE_KEYWORDS = [
    "price prediction", "price analysis", "technical analysis",
    "might ", "could ", "may reach", "potential",
    "top coins", "best crypto", "altcoin season",
    "sponsored", "advertisement", "affiliate",
]


def is_significant(title: str, description: str = "") -> bool:
    """Check if a news item represents a significant event."""
    text = f"{title} {description}".lower()

    for noise in NOISE_KEYWORDS:
        if noise in text:
            return False

    for keyword in EVENT_KEYWORDS:
        if keyword in text:
            return True

    return False


def categorize_event(title: str) -> str:
    """Assign a short label based on event type."""
    t = title.lower()
    if any(w in t for w in ("etf", "sec ")):
        return "ETF"
    if any(w in t for w in ("halving", "halvening", "block reward")):
        return "Halving"
    if any(w in t for w in ("hack", "exploit", "stolen")):
        return "Hack"
    if any(w in t for w in ("crash", "liquidat", "collapse", "bankrupt")):
        return "Crash"
    if any(w in t for w in ("ban", "legal tender", "regulation")):
        return "Reg"
    if any(w in t for w in ("fed ", "fomc", "cpi ", "inflation", "interest rate")):
        return "Macro"
    if any(w in t for w in ("ath", "all-time high", "record")):
        return "ATH"
    if any(w in t for w in ("reserve", "microstrategy", "strategy ", "saylor", "tesla")):
        return "Inst"
    if any(w in t for w in ("tariff", "trade war")):
        return "Macro"
    return "Event"


def deduplicate_events(events: list[dict]) -> list[dict]:
    """Remove duplicate events by day + normalized title prefix."""
    seen: set[str] = set()
    unique = []
    for event in events:
        day = event["date"][:10]
        title_key = event["title"][:50].lower().strip()
        key = f"{day}|{title_key}"
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return unique


async def fetch_and_filter_news() -> list[dict]:
    """Fetch BTC news from cryptocurrency.cv and filter significant events."""
    events = []

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{NEWS_BASE_URL}/search",
                params={
                    "q": "bitcoin",
                    "limit": 100,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return events

    articles = data.get("articles", data) if isinstance(data, dict) else data

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        pub_date = article.get("pubDate", "")
        source_key = article.get("sourceKey", "")
        source_name = article.get("source", "")

        # Filter: only trusted sources
        if source_key.lower() not in TRUSTED_SOURCES:
            continue

        if not is_significant(title, description):
            continue

        event_date = pub_date[:19] if pub_date else datetime.now(timezone.utc).isoformat()

        events.append({
            "id": f"news-{hash(title) & 0xFFFFFFFF:08x}",
            "date": event_date,
            "label": categorize_event(title),
            "title": title.strip(),
            "kind": "event",
            "source": source_name,
        })

    events = deduplicate_events(events)
    events.sort(key=lambda e: e["date"], reverse=True)
    return events


async def refresh_events():
    """Scheduled job: fetch news, merge with existing events, save to disk."""
    logger.info("Refreshing events...")
    new_events = await fetch_and_filter_news()

    existing = []
    if EVENTS_FILE.exists():
        try:
            existing = json.loads(EVENTS_FILE.read_text())
        except Exception:
            existing = []

    all_events = new_events + existing
    all_events = deduplicate_events(all_events)
    all_events.sort(key=lambda e: e["date"], reverse=True)
    all_events = all_events[:200]

    EVENTS_FILE.write_text(json.dumps(all_events, indent=2, ensure_ascii=False))
    logger.info(f"Saved {len(all_events)} events ({len(new_events)} new)")


# --- HTTP Endpoints ---

@app.get("/")
async def health():
    return {"status": "ok", "service": "bitcoin-events-bot"}


@app.get("/events")
async def get_events():
    """Return cached events as JSON. The iOS app polls this."""
    if not EVENTS_FILE.exists():
        return JSONResponse(content=[], headers={"Cache-Control": "public, max-age=300"})

    events = json.loads(EVENTS_FILE.read_text())
    return JSONResponse(
        content=events,
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.post("/refresh")
async def manual_refresh():
    """Trigger a manual refresh (for testing)."""
    await refresh_events()
    count = 0
    if EVENTS_FILE.exists():
        count = len(json.loads(EVENTS_FILE.read_text()))
    return {"status": "refreshed", "event_count": count}


# --- Startup ---

@app.on_event("startup")
async def startup():
    await refresh_events()

    # Daily at 06:00 and 18:00 UTC
    scheduler.add_job(refresh_events, "cron", hour="6,18", minute=0)
    scheduler.start()
    logger.info("Scheduler started: refresh at 06:00 and 18:00 UTC")
