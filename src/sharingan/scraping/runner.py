"""Programmatic entry point: run the jutsu spider and write JSON Lines."""

from __future__ import annotations

from pathlib import Path

from scrapy.crawler import CrawlerProcess

from .spiders.jutsu_api import JutsuApiSpider

SETTINGS = {
    "BOT_NAME": "sharingan",
    "USER_AGENT": "sharingan/1.0 (educational NLP project; MediaWiki API client)",
    "ROBOTSTXT_OBEY": True,
    # Politeness: the API serves 50 pages per call, so a slow crawl is still fast.
    "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
    "DOWNLOAD_DELAY": 0.5,
    "AUTOTHROTTLE_ENABLED": True,
    "AUTOTHROTTLE_START_DELAY": 0.5,
    "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
    "RETRY_TIMES": 4,
    "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
    "HTTPCACHE_ENABLED": True,
    "HTTPCACHE_EXPIRATION_SECS": 7 * 24 * 3600,
    "ITEM_PIPELINES": {
        "sharingan.scraping.pipelines.ValidatePipeline": 100,
        "sharingan.scraping.pipelines.NormalisePipeline": 200,
        "sharingan.scraping.pipelines.DeduplicatePipeline": 300,
    },
    "LOG_LEVEL": "INFO",
    "FEED_EXPORT_ENCODING": "utf-8",
}


def crawl_jutsus(output: Path, *, limit: int | None = None, cache_dir: Path | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        **SETTINGS,
        "FEEDS": {str(output): {"format": "jsonlines", "overwrite": True}},
        # Absolute: Scrapy resolves relative cache dirs against its own .scrapy/ folder.
        "HTTPCACHE_DIR": str((cache_dir or output.parent / ".scrapy-cache").resolve()),
    }
    process = CrawlerProcess(settings)
    process.crawl(JutsuApiSpider, limit=limit)
    process.start()
    return output
