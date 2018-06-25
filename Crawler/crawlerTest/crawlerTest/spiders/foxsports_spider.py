#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/20 上午11:26  
# @Author  : Kaiyu  
# @Site    :   
# @File    : foxsports_spider.py

import scrapy

class FoxsportsSpider(scrapy.spiders.Spider):
    name = "foxsports"
    allowed_domains = ['foxsports.com']
    base_url = "https://www.foxsports.com/soccer/stats"
    querystring = {"competition": "10", "season": "20172", "category": "STANDARD", "pos": "0", "team": "0", "sort": "1",
                   "sortOrder": "0", "page": "1"}


