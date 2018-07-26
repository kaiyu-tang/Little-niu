#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/7/16 下午3:27  
# @Author  : Kaiyu  
# @Site    :   
# @File    : bf_win007_com_spider.py
import re

import scrapy
from scrapy_crawler.items import BfWin007ComItem
from scrapy.http import Request


class Bfwin007Spider(scrapy.Spider):
    name = 'bf_win007'
    allowed_domains = ['bf.win007.com']
    root_url = 'http://bf.win007.com/TextLive/{}cn.htm'
    base_url = 'http://61.143.225.111/{}.htm'

    # start_urls = ['http://bf.win007.com/TextLive/1395449cn.htm']

    def start_requests(self):
        for i in range(475, 1569999):
            url = self.base_url.format(str(i))
            yield Request(url, self.parse)

    def parse(self, response):
        # filename = response.url.split('/')[-2]
        item = BfWin007ComItem()
        # match_time = response.xpath('//*[@id="matchItems"]/div[2]/span/text()')
        # home_team = response.xpath('//*[@id="home"]/a/span/text()')
        # host_team = response.xpath('//*[@id="guest"]/a/span/text()')

        #live_times = response.xpath('//*[@id="Table6"]/tr/td/font/text()')
        # print(response.text)
        # item['match_time'] = match_time.extract()
        # item['home_team'] = home_team.extract()
        # item['host_team'] = host_team.extract()
        live_texts = response.xpath('//*[@id="Table6"]/tr/td/text()').extract()
        live_texts = [live_text_.replace('\n','').replace('\r','').replace('\\','') for live_text_ in live_texts if len(live_text_)>4]
        item['live_texts'] = live_texts
        # item['live_abs_times'] = re.findall('(\d+):(\d+):(\d+)', live_texts)
        # item['live_rel_times'] = re.findall(']\s+(.*):', live_texts)
        # item['live_score'] = re.findall('(\d+)-(\d+)',live_texts)

        yield item
