"""Jutsu spider built on the MediaWiki Action API.

Why the API rather than HTML pages:
  * the HTML "BrowseData" listing is behind a bot challenge (HTTP 403), the API is not;
  * one request returns wikitext for 50 articles -> ~60 requests instead of ~3,000;
  * wikitext exposes the infobox as structured key/values (no brittle CSS selectors).
"""

from __future__ import annotations

from urllib.parse import quote, urlencode

import scrapy

from ..items import JutsuItem
from ..wikitext import parse_description, parse_infobox

BATCH = 50  # MediaWiki cap for revision content per request


class JutsuApiSpider(scrapy.Spider):
    name = "jutsu_api"
    wiki = "https://naruto.fandom.com"
    category = "Category:Jutsu"

    def __init__(self, limit: int | str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = int(limit) if limit else None
        self.queued = 0

    def _api(self, **params) -> str:
        base = {"format": "json", "formatversion": 2, "maxlag": 5}
        return f"{self.wiki}/api.php?{urlencode({**base, **params})}"

    async def start(self):
        yield self._listing_request()

    def _listing_request(self, cont: dict | None = None) -> scrapy.Request:
        params = dict(action="query", list="categorymembers", cmtitle=self.category,
                      cmnamespace=0, cmlimit=500, **(cont or {}))
        return scrapy.Request(self._api(**params), callback=self.parse_listing)

    def parse_listing(self, response):
        data = response.json()
        titles = [m["title"] for m in data["query"]["categorymembers"]]
        if self.limit is not None:
            titles = titles[: max(self.limit - self.queued, 0)]
        self.queued += len(titles)
        for i in range(0, len(titles), BATCH):
            chunk = titles[i : i + BATCH]
            yield scrapy.Request(
                self._api(action="query", prop="revisions", rvprop="content", rvslots="main",
                          titles="|".join(chunk), redirects=1),
                callback=self.parse_pages,
            )
        if "continue" in data and (self.limit is None or self.queued < self.limit):
            yield self._listing_request({"cmcontinue": data["continue"]["cmcontinue"]})

    def parse_pages(self, response):
        for page in response.json()["query"].get("pages", []):
            revisions = page.get("revisions")
            if not revisions:
                continue
            wikitext = revisions[0]["slots"]["main"]["content"]
            box = parse_infobox(wikitext)
            if not box:
                continue
            yield JutsuItem(
                name=page["title"],
                page_id=page["pageid"],
                url=f"{self.wiki}/wiki/{quote(page['title'].replace(' ', '_'))}",
                description=parse_description(wikitext),
                **box,
            )
