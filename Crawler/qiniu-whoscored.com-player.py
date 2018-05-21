#!/usr/bin/env python  
# -*- coding: utf-8 -*-
# @Time    : 2018/5/17 下午6:06  
# @Author  : Kaiyu  
# @Site    :   
# @File    : qiniu-whoscored.com-player.py.py
import json
import time

import requests
from threading import Thread
import re
from requests.exceptions import ConnectionError, ConnectTimeout,ReadTimeout
from lxml import etree

tags = {"url", "Summary", "Offensive", "Defensive", "Passing"}


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


url = "https://www.whoscored.com/StatisticsFeed/1/GetPlayerStatistics"

querystring = {"category": "summary", "subcategory": "{}", "statsAccumulationType": "0", "isCurrent": "true",
               "playerId": "{}", "teamIds": "", "matchId": "", "stageId": "", "tournamentOptions": "",
               "sortBy": "Rating", "sortAscending": "", "age": "", "ageComparisonType": "", "appearances": "",
               "appearancesComparisonType": "", "field": "Overall", "nationality": "", "positionOptions": "",
               "timeOfTheGameEnd": "", "timeOfTheGameStart": "", "isMinApp": "false", "page": "",
               "includeZeroValues": "true", "numberOfPlayersToPick": ""}  # playerId and sybcategray is useful there

headers = headers = {
    'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    'Accept-Encoding': "gzip, deflate",
    'Accept-Language': "zh-CN,zh;q=0.9",
    'Cache-Control': "no-cache",
    'Connection': "keep-alive",
    #'Cookie': "announceId=20180103001; JSESSIONID=831C24BC2C1F2A3A6B31E37B63B65823; Hm_lvt_b83b828716a7230e966a4555be5f6151=1526278774,1526374337,1526552611; Hm_lpvt_b83b828716a7230e966a4555be5f6151=1526552621",
    'DNT': "1",
    'Host': "www.tzuqiu.cc",
    'Referer': "http://www.tzuqiu.cc/stats.do",
    'Upgrade-Insecure-Requests': "1",
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36",
    #'Postman-Token': "82796269-b881-47bb-a673-963c70be9c33"
    }


def get_match_summary(url, connect_times=0):
    res = {}
    try:
        response = requests.get(url, headers=headers, timeout=0.8)  # .decode('gb2312', 'ignore')
    except (ConnectionError, ConnectTimeout, ReadTimeout):
        time.sleep(2)
        if connect_times < 2:
            return get_match_summary(url, connect_times + 1)
        else:
            return res
    #print(response.text)
    html = etree.HTML(response.content.decode('utf-8'))
    print(html.text)
    for tr in html.xpath(u'//*[@id="summaryTable"]/tbody'):
        print(tr.text)
    #summary = re_match_summary.findall(response.text)
    #print()
    return res


if __name__ == "__main__":
    get_match_summary("http://www.tzuqiu.cc/players/797/show.do")
