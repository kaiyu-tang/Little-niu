#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/7/26 上午11:51  
# @Author  : Kaiyu  
# @Site    :   
# @File    : data-process.py
"""
process the crawled data
"""
import json


def item360(file_name, output_name):
    """
    process the crawled 360zhibo data, split each text in to live text
    :param file_name:
    :param output_name:
    :return:
    """
    with open(file_name, encoding='utf-8') as f:
        new_data = []
        data = dict(f.read())
        for item in data:
            title = item['title']
            text = item['text']
            time = item['time']
            if len(text) > 40:
                tmp_ = text.split('。')
                new_text = [item_ for item_ in tmp_ if '分钟' in item_]
                new_data.append({"title":title,"time":time,"text":new_text})
    with open(out_name,"w",encoding="utf-8") as f:
        json.dump(new_data,f,indent=2,ensure_ascii=False,separators=(',', ": "))


if __name__ == '__main__':
    file_name = "360_items.json"
    out_name = "clean_360_items.json"
    item360(file_name, out_name)
