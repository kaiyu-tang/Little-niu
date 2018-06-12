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

    def get_info(self, name):
        payload = {}
        payload['player_name'] = name
        response = requests.request("POST", self.url, data=json.dumps(payload), headers=self.headers)
        response = json.loads(response.text)
        if len(response) != 0:
            return response[0]
        else:
            return {}


if __name__ == "__main__":
    player_ = player_info()
    re = player_.get_info("Castro, G")
    print()
