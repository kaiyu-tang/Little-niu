# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrapyCrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

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
    pass

class zhibo360Item(scrapy.Item):
    name = scrapy.Field()
    title = scrapy.Field()
    simple_content = scrapy.Field()
    content = scrapy.Field()
    text = scrapy.Field()
    time = scrapy.Field()
    pass

class zhibo7mItem(scrapy.Item):
    # time = scrapy.Field()
    # place = scrapy.Field()
    # judger = scrapy.Field()
    # t_home = scrapy.Field()
    # t_away = scrapy.Field()
    # bifen = scrapy.Field()
    # live_texts = scrapy.Field()
    url = scrapy.Field()
    content = scrapy.Field()

