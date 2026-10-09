"""Item pipelines: validate -> normalise -> de-duplicate."""

from __future__ import annotations

import re
import unicodedata

from scrapy.exceptions import DropItem

_WS_RE = re.compile(r"\s+")


class ValidatePipeline:
    def process_item(self, item):
        if not item.get("classification"):
            raise DropItem(f"{item.get('name')!r}: no classification in infobox")
        if len(item.get("description") or "") < 20:
            raise DropItem(f"{item.get('name')!r}: description too short")
        return item


class NormalisePipeline:
    def process_item(self, item):
        for key, value in item.items():
            if isinstance(value, str):
                item[key] = _WS_RE.sub(" ", unicodedata.normalize("NFC", value)).strip()
        return item


class DeduplicatePipeline:
    def __init__(self):
        self.seen: set[int] = set()

    def process_item(self, item):
        if item["page_id"] in self.seen:
            raise DropItem(f"duplicate page {item['page_id']}")
        self.seen.add(item["page_id"])
        return item
