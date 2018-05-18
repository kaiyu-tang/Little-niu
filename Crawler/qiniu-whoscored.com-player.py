#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/17 下午6:06  
# @Author  : Kaiyu  
# @Site    :   
# @File    : qiniu-whoscored.com-player.py.py
import json

import requests
from threading import Thread
import re
import pandas as pd
tags = {"url", "Id" "summary" , "Offensive", "Defensive", "Passing"}
summary = {""}
df_player = pd.DataFrame(columns=tags)


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

def get_match_summary(url, header, param):
    page = requests.request('GET', url, headers=header, params=param).text
    playerTableStats = json.loads(page)["playerTableStats"]
    for match in playerTableStats:
        match_data = []
        for key in tags:
            match_data.append(match[key])



response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)
