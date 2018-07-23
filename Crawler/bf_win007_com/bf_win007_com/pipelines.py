# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import json

from scrapy.exceptions import DropItem


class BfWin007ComPipeline(object):
    def __init__(self):
        self.file = open('items.json', 'w', encoding='utf-8')

    def process_item(self, item, spider):
        if len(item['live_texts']) == 0:
            raise DropItem('No live texts')
        line = json.dumps(dict(item), ensure_ascii=False, indent=2, separators=(',', ': ')) + '\n'
        self.file.write(line)
        return item

    def close_spider(self,spider):
        self.file.close()