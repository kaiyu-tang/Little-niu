#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/7/24 上午11:50  
# @Author  : Kaiyu  
# @Site    :   
# @File    : 360zhibo.py
import re
from scrapy.http import Request
import scrapy
from scrapy_crawler.items import zhibo360Item


class zhibo360(scrapy.Spider):
    name = "zhibo360"
    base_url = "http://www.360zhibo.com/foot/{}.html"
    allowed_domains = ['www.360zhibo.com']

    def start_requests(self):
        for i in range(60164, 61590):
            url = self.base_url.format(i)
            yield Request(url, self.parse)

    def parse(self, response):
        item = zhibo360Item()
        text = ' '.join(response.xpath('//*[@id="pdintro"]/div[2]/text()').extract())
        #print(text)
        title = response.xpath('//*[@id="pdintro"]/div[2]/font[1]/text()')
        simple_content = response.xpath('//*[@id="pdintro"]/div[2]/text()[4]')
        content = response.xpath('//*[@id="pdintro"]/div[2]/text()[position()>3]')
        item['title'] = title.extract()[0]
        #item['simple_content'] = simple_content.extract()[0]
        #item['content'] = content.extract()
        text_ = re.findall(u'内容：(.*)添加时间：(.*)',text,re.S)[0]
        item['text'] = ''.join(text_[0]).replace('\n','').replace('\r','')
        item['time'] = ''.join(text_[1]).replace('\r','').replace('\n','')
        #print(item)
        yield item
