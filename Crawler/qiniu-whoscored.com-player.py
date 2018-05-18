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

headers = {
    'accept': "application/json, text/javascript, */*; q=0.01",
    'accept-encoding': "gzip, deflate, br",
    'accept-language': "zh-CN,zh;q=0.9",
    'dnt': "1",
    'model-last-mode': "X5wigIWFrhjVrViOFCg/GiyJwa5gZzPkmiECc3VfKO4=",
    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36",
    'x-requested-with': "XMLHttpRequest",
    'Cache-Control': "no-cache",
    'Postman-Token': "1496edb2-7ebf-470f-ab35-4bffbff5f703"
}


def get_match_summary(url, re_match_summary, re_match_offensive=None, re_match_defensive=None, re_match_passing=None,
                      connect_times=0):
    res = {}
    try:
        page = requests.get(url).text  # .decode('gb2312', 'ignore')
    except ConnectionError as connection:
        time.sleep(2)
        if connect_times < 100:
            return get_match_summary(url, re_match_summary, re_match_offensive, re_match_defensive, re_match_passing,
                                     connect_times + 1)
        else:
            return res
    summary = re_match_summary.findall(page)
    print()
    return res


if __name__ == "__main__":
    re_match_summary = re.compile(
        r'<table class="table table-hover dt-responsive nowrap stripe dataTable" id=".+?Table" cellspacing="0" width="100%">\n\s+<thead>.+?</thead>\n\s+?<tbody>.*?(<tr>.+?</tr>).*?</tbody>', re.DOTALL)
    get_match_summary("http://www.tzuqiu.cc/players/797/show.do", re_match_summary)
