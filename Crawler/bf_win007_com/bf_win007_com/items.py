# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class BfWin007ComItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    match_time = scrapy.Field()
    home_team = scrapy.Field()
    host_team = scrapy.Field()
    live_texts = scrapy.Field()
    live_abs_times = scrapy.Field()
    live_rel_times = scrapy.Field()
    live_score = scrapy.Field()

