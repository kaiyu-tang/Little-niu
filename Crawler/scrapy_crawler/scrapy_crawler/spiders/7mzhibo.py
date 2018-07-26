import random
import re
from scrapy.http import Request
import scrapy
from scrapy_crawler.items import zhibo7mItem
import json


class zhibo7m(scrapy.Spider):
    name = '7m'
    base_url = 'http://js.wlive.7m.com.cn/livedata.aspx?l=big&d={}&f={}'
    allowed_domains = ['7m.com']

    def start_requests(self):
        for index_ in range(627, 58966):
            rand_f_ = random.randint(153250544022, 1632505440228)
            url_ = self.base_url.format(index_, rand_f_)
            yield Request(url_, self.parse)

    def parse(self, response):
        item = zhibo7mItem()
        item['url'] = response.url
        item['content'] = json.loads(response.text[12:-1].replace("'", '"'))
        yield item
        pass
