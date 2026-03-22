"""Bitcoin Events Bot — hourly news scanner that extracts significant BTC events.

Fetches news from free-crypto-news (cryptocurrency.cv) API from trusted sources,
sends all headlines to Gemini Flash Lite for importance ranking.

Importance levels (mapped to chart terms in the iOS app):
- high: 1W, 1M, 1Y charts (max 1-2 per day)
- medium: 1D, 12H charts (max 5 per day)
- low: 1H chart (any notable news)
- skip: noise, not shown on any chart
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import httpx
from google import genai
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Bitcoin Events Bot")
scheduler = AsyncIOScheduler()

DB_PATH = Path(os.environ.get("DATA_DIR", ".")) / "events.db"
NEWS_BASE_URL = "https://cryptocurrency.cv/api"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash"]

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


def get_db() -> sqlite3.Connection:
    """Get SQLite connection, creating tables if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        date TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT DEFAULT '',
        link TEXT DEFAULT '',
        kind TEXT DEFAULT 'event',
        source TEXT DEFAULT '',
        importance TEXT DEFAULT 'low',
        label TEXT DEFAULT 'Event',
        score INTEGER DEFAULT 5
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS seen (
        id TEXT PRIMARY KEY
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_importance ON events(importance)")
    conn.commit()
    return conn


GEMINI_PROMPT = """You are a Bitcoin market event filter for a price chart app.

Your task: from the list of news headlines below, select ONLY real events that affected or could affect BTC price. Assign importance level and a score to each.

Importance levels:
- "high" — Historic, market-moving event. Examples: ETF approval/rejection, halving, major exchange hack/collapse (Mt.Gox, FTX), Fed rate decision with surprise, new ATH, country banning or adopting BTC, major crash (>15%). MAX 2 per day.
- "medium" — Notable event worth marking on a daily chart. Examples: large institutional purchase (MicroStrategy), regulatory action, significant price move (>8%), major protocol upgrade, exchange outage, whale alert (>$500M move).
- "low" — Minor but real event for hourly chart. Examples: notable price dip/pump (>3%), minor regulatory news, exchange listing/delisting, on-chain anomaly.
- "skip" — Not an event. Opinions, analysis, predictions, price commentary, educational content, sponsored content, vague speculation.

Score: 1-10 within the importance level. 10 = most significant for that level. This helps the app decide which events to show when space is limited (e.g. weekly chart can only fit 3-5 markers).

CRITICAL RULES:
1. Be VERY selective. Most headlines are noise — mark them "skip".
2. Multiple articles about the SAME event = keep only the best headline, skip duplicates.
3. "high" is rare — maybe 1-2 per WEEK, not per day.
4. Focus on FACTS and ACTIONS, not opinions or predictions.
5. ALWAYS skip: educational content, explainers, "what is X", guides, historical retrospectives, price predictions, "what's next" articles.
6. Extreme Fear/Greed readings are valid "low" events — they mark sentiment extremes on the chart.
7. For each non-skip event, provide a short label (max 8 chars): ETF, Crash, Hack, Fed, ATH, Reg, Inst, Macro, Fork, Mine, etc.

Headlines:
{titles}

Respond with a JSON array. Each element: {{"index": N, "importance": "high|medium|low|skip", "score": 1-10, "label": "short label"}}
Only include non-skip items. Example:
[{{"index": 1, "importance": "high", "score": 9, "label": "Crash"}}, {{"index": 5, "importance": "low", "score": 4, "label": "Macro"}}]

If ALL headlines are noise, respond with: []"""


DESCRIPTION_PROMPT = """Read the article below and write a 1-2 sentence description for a Bitcoin price chart marker.

Rules:
1. Explain WHAT happened and WHY (cause → effect).
2. Use ONLY facts from the article. Do NOT add your own knowledge or speculation.
3. If the article explains why the price moved, include the reasons.
4. If cause-effect cannot be extracted, summarize the key fact from the headline as clearly as possible.
5. Keep it under 200 characters. No quotes, no source attribution.
6. Write in English.

Headline: {title}

Article text:
{article_text}

Description:"""


def extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML, stripping tags and scripts."""
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html).strip()
    return html[:4000]  # Gemini context limit — first ~4k chars is enough


async def fetch_article_text(url: str) -> str | None:
    """Fetch article text. Tries direct, then Google Cache as fallback."""
    if not url:
        return None

    attempts = [
        ("direct", url),
        ("cache", f"https://webcache.googleusercontent.com/search?q=cache:{url}"),
    ]

    for label, target in attempts:
        try:
            from curl_cffi.requests import AsyncSession
            async with AsyncSession() as s:
                resp = await s.get(target, impersonate="chrome", timeout=15, allow_redirects=True)
                resp.raise_for_status()
                text = extract_text_from_html(resp.text)
                if text and len(text) > 200:
                    logger.info(f"Fetched article ({label}): {url[:60]}")
                    return text
        except Exception as e:
            logger.warning(f"Fetch failed ({label}): {url[:60]}: {e}")
            continue

    return None


async def enrich_descriptions(events: list[dict]) -> list[dict]:
    """For each event, fetch the article and ask Gemini for a cause-effect description."""
    if not GEMINI_API_KEY or not events:
        return events

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = genai.types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=256,
    )

    async def enrich_one(event: dict) -> dict:
        article_text = await fetch_article_text(event.get("link", ""))
        if article_text:
            logger.info(f"Fetched article: {event['title'][:60]}")
        else:
            # Fallback: use API description snippet
            article_text = event.get("description", "")
            if not article_text:
                logger.warning(f"No text available: {event['title'][:60]}")
                return event
            logger.info(f"Using API description (fallback): {event['title'][:60]}")

        prompt = DESCRIPTION_PROMPT.replace("{title}", event["title"]).replace("{article_text}", article_text)

        for model in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config,
                )
                desc = response.text.strip().strip('"')
                if desc and len(desc) > 10:
                    event["description"] = desc[:250]
                    logger.info(f"Enriched: {event['title'][:50]} → {desc[:80]}")
                break
            except Exception as e:
                logger.warning(f"Description enrichment failed ({model}): {e}")
                continue

        return event

    # Fetch articles in parallel (max 5 concurrent)
    semaphore = asyncio.Semaphore(5)

    async def limited_enrich(event: dict) -> dict:
        async with semaphore:
            return await enrich_one(event)

    enriched = await asyncio.gather(*(limited_enrich(e) for e in events))
    return list(enriched)



async def rank_with_gemini(candidates: list[dict]) -> list[dict]:
    """Send all candidates to Gemini, return only significant ones with importance."""
    if not GEMINI_API_KEY or not candidates:
        return []

    titles = "\n".join(
        f"{i+1}. [{c['date'][:10]}] {c['title']}" for i, c in enumerate(candidates)
    )
    prompt = GEMINI_PROMPT.replace("{titles}", titles)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = genai.types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
        )

        response = None
        for model in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config,
                )
                logger.info(f"Gemini responded via {model}")
                break
            except Exception as model_err:
                logger.warning(f"{model} failed: {model_err}")
                continue

        if not response:
            logger.error("All Gemini models unavailable")
            return []

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        rankings = json.loads(text)

        events = []
        for item in rankings:
            idx = item.get("index", 0) - 1  # 1-based to 0-based
            importance = item.get("importance", "skip")
            label = item.get("label", "Event")

            if importance == "skip" or idx < 0 or idx >= len(candidates):
                continue

            c = candidates[idx]
            c["importance"] = importance
            c["label"] = label
            c["score"] = item.get("score", 5)
            events.append(c)

        return events

    except Exception as e:
        logger.error(f"Gemini ranking failed: {e}")
        return []


async def fetch_news() -> list[dict]:
    """Fetch BTC news from cryptocurrency.cv, return trusted-source articles."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{NEWS_BASE_URL}/search",
                params={"q": "bitcoin", "limit": 100},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []

    articles = data.get("articles", data) if isinstance(data, dict) else data

    candidates = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        pub_date = article.get("pubDate", "")
        source_key = article.get("sourceKey", "")
        source_name = article.get("source", "")
        link = article.get("link", "")

        if source_key.lower() not in TRUSTED_SOURCES:
            continue

        event_date = pub_date[:19] if pub_date else datetime.now(timezone.utc).isoformat()

        candidates.append({
            "id": f"news-{hashlib.md5(title.encode()).hexdigest()[:8]}",
            "date": event_date,
            "title": title.strip(),
            "description": description.strip(),
            "link": link,
            "kind": "event",
            "source": source_name,
        })

    logger.info(f"Fetched {len(candidates)} articles from trusted sources")
    return candidates


async def refresh_events():
    """Scheduled job: fetch news, rank with Gemini, insert new events."""
    logger.info("Refreshing events...")
    candidates = await fetch_news()

    if not candidates:
        logger.info("No candidates to rank")
        return

    # Skip already-processed articles
    db = get_db()
    seen_rows = db.execute("SELECT id FROM seen").fetchall()
    seen = {r["id"] for r in seen_rows}
    new_candidates = [c for c in candidates if c["id"] not in seen]
    logger.info(f"{len(new_candidates)} new candidates ({len(candidates) - len(new_candidates)} already seen)")

    if not new_candidates:
        db.close()
        return

    new_events = await rank_with_gemini(new_candidates)
    logger.info(f"Gemini selected {len(new_events)} events from {len(new_candidates)} candidates")

    if new_events:
        new_events = await enrich_descriptions(new_events)
        logger.info("Description enrichment complete")

    # Mark all candidates as seen
    db.executemany("INSERT OR IGNORE INTO seen (id) VALUES (?)",
                   [(c["id"],) for c in new_candidates])

    # Insert new events
    for e in new_events:
        db.execute("""INSERT OR IGNORE INTO events
            (id, date, title, description, link, kind, source, importance, label, score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (e["id"], e["date"], e["title"], e.get("description", ""),
             e.get("link", ""), e.get("kind", "event"), e.get("source", ""),
             e["importance"], e["label"], e.get("score", 5)))

    # Trim old events — keep last 500
    db.execute("""DELETE FROM events WHERE id NOT IN
        (SELECT id FROM events ORDER BY date DESC LIMIT 500)""")
    # Trim seen — keep last 5000
    db.execute("""DELETE FROM seen WHERE rowid NOT IN
        (SELECT rowid FROM seen ORDER BY rowid DESC LIMIT 5000)""")

    db.commit()
    db.close()
    logger.info(f"Inserted {len(new_events)} new events")


# --- HTTP Endpoints ---

@app.get("/")
async def health():
    return {"status": "ok", "service": "bitcoin-events-bot"}


@app.get("/events")
async def get_events(importance: str | None = None, min_score: int = 0, limit: int = 100):
    """Return events. Filters: ?importance=high,medium&min_score=5&limit=50"""
    db = get_db()

    if importance:
        levels = [s.strip() for s in importance.split(",")]
        placeholders = ",".join("?" * len(levels))
        rows = db.execute(
            f"""SELECT * FROM events
                WHERE importance IN ({placeholders}) AND score >= ?
                ORDER BY date DESC LIMIT ?""",
            (*levels, min_score, limit),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM events WHERE score >= ? ORDER BY date DESC LIMIT ?",
            (min_score, limit),
        ).fetchall()

    db.close()
    events = [dict(r) for r in rows]
    return JSONResponse(
        content=events,
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.post("/refresh")
async def manual_refresh():
    """Trigger a manual refresh (for testing)."""
    await refresh_events()
    db = get_db()
    count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    db.close()
    return {"status": "refreshed", "event_count": count}


@app.post("/reset")
async def reset_db():
    """Wipe all data and re-fetch from scratch."""
    db = get_db()
    db.execute("DELETE FROM events")
    db.execute("DELETE FROM seen")
    db.commit()
    db.close()
    logger.info("Database reset")
    await refresh_events()
    db = get_db()
    count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    db.close()
    return {"status": "reset", "event_count": count}


# --- Startup ---

@app.on_event("startup")
async def startup():
    await refresh_events()

    scheduler.add_job(refresh_events, "interval", hours=1)
    scheduler.start()
    logger.info("Scheduler started: hourly refresh")
