"""Scrapy item schema for a single jutsu article."""

import scrapy


class JutsuItem(scrapy.Item):
    name = scrapy.Field()
    url = scrapy.Field()
    page_id = scrapy.Field()
    classification = scrapy.Field()   # list[str], e.g. ["Ninjutsu", "Shape Transformation"]
    nature = scrapy.Field()           # list[str], e.g. ["Wind Release"]
    rank = scrapy.Field()
    class_type = scrapy.Field()       # Offensive / Defensive / Supplementary
    range = scrapy.Field()
    hand_signs = scrapy.Field()       # list[str]
    users = scrapy.Field()            # list[str]
    debut_anime = scrapy.Field()
    description = scrapy.Field()
