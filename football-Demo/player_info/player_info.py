#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 下午12:02  
# @Author  : Kaiyu  
# @Site    :   
# @File    : player_info.py
import json

import requests


class player_info(object):

    def __init__(self):
        self.url = "http://lego-soccer.ava.ke-xs.cloudappl.com/v1/performance"
        self.headers = {
            'Content-Type': "application/json",
            'Cache-Control': "no-cache"
        }
        pass

    def get_year(self, data, year):
        res = [[] for _ in range(year)]
        ress = []
        years = [0 for _ in range(year)]
        for item in data:
            performance_ = item['performance']
            year_ = int(performance_['season'][:-1])
            min_ = min(years)
            if year_ in years:
                index_ = years.index(year_)
                res[index_].append(item)
            elif year_ > min_:
                index_ = years.index(min_)
                years[index_] = year_
                res[index_] = [item]
        for s in res:
            for item in s:
                ress.append(item)
        return ress
    def get_info(self, name, net=True, year=3):
        if net:
            payload = {}
            payload['player_name'] = name
            response = requests.request("POST", self.url, data=json.dumps(payload), headers=self.headers)
            try:
                response = json.loads(response.text)
                if len(response) != 0:
                    return self.get_year(response, year)
                else:
                    return []
            except:
                return []

        else:
            with open("/home/atlab/Workspace/kaiyu/Demo/toys/football-Demo/player_info/foxsport.json") as f:
                data = json.load(f)
                if name in data:
                    return data[name]
                else:
                    return {}


if __name__ == "__main__":
    player_ = player_info()

    re = player_.get_info("Harry Kane")
    print(re)
