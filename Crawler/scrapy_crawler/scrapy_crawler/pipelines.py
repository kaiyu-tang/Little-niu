# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import json

from scrapy.exceptions import DropItem
from scrapy_crawler.items import zhibo360Item
from scrapy_crawler.items import BfWin007ComItem
from scrapy_crawler.items import zhibo7mItem


class ScrapyCrawlerPipeline(object):
    def __init__(self):
        self.file_360 = open('360_items.json', 'a+', encoding='utf-8')
        self.file_bfwin007 = open('bfwin007_item.json', 'a+', encoding='utf-8')
        self.file_7m = open('7m.json', 'a+', encoding='utf-8')

    def process_item(self, item, spider):
        if isinstance(item, zhibo360Item):
            self._process_item_360zhibo(item, spider)
        elif isinstance(item, BfWin007ComItem):
            self._process_item_bfwin007(item, spider)
        elif isinstance(item, zhibo7mItem):
            self._process_item_zhibo7m(item, spider)
        return item

    def _process_item_360zhibo(self, item, spider):
        print(spider.name)
        if len(item['title']) == 0:
            raise DropItem('No live texts')
        line = json.dumps(dict(item), ensure_ascii=False, indent=2, separators=(',', ': ')) + '\n'
        self.file_360.write(line)
        pass

    def _process_item_bfwin007(self, item, spider):
        if len(item['live_texts']) == 0:
            raise DropItem('No live texts')
        line = json.dumps(dict(item), ensure_ascii=False, indent=2, separators=(',', ': ')) + '\n'
        self.file_bfwin007.write(line)
        return item
        pass

    def _process_item_zhibo7m(self, item, spider):
        line = json.dumps(dict(item), ensure_ascii=False, indent=2, separators=(',', ': ')) + "\n"
        self.file_7m.write(line)
        return item
        pass

    def close_spider(self, spider):
        self.file.close()
