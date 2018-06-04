#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/22 上午10:41  
# @Author  : Kaiyu  
# @Site    :   
# @File    : qiniu-foxsports.com-matches.py

import json
import os
import random
import time

import requests
from threading import Thread
import re
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout
from lxml import etree, html
from bs4 import BeautifulSoup


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

def get_foxsport_match(url):
    res = {}
    return res

if __name__ == "__main__":

    base = 'https://www.foxsports.com/soccer/stats?competition=&{}season={}&' \
           'category=DISCIPLINE&pos=0&team=0&isOpp=0&sort=11&sortOrder=0&page={}'
    all_matches = {}
    threads = []
    player_start_id = 1
    player_end_id = 50000
    batch_size = 200
    count = 0
    base_dir = "whoscored-matches/{}.json"
    if not os.path.exists("whoscored-matches"):
        os.mkdir("whoscored-matches")

    for loop in range((player_end_id - player_start_id) // batch_size):
        time.sleep(random.randrange(0, 10))
        player_cur_start_id = player_start_id + loop * batch_size
        for player_id in range(player_cur_start_id, min(player_cur_start_id + batch_size, player_end_id)):
            url = base.format(player_id)
            start_time = time.clock()
            thread = ThreadWithReturnValue(target=get_foxsport_match(),
                                           args=(url, 0))
            threads.append(thread)
            thread.start()
            end_time = time.clock()

            print("id: {} runtime: {}".format(player_id, end_time - start_time))
        for index, thread in enumerate(threads):
            cur_res = thread.join()
            try:
                if len(cur_res) != 0:
                    # print(cur_res)
                    with open(base_dir.format(cur_res["url"].split("/")[-2]), 'w') as f_w:
                        print(cur_res["url"])
                        json.dump(cur_res, f_w, ensure_ascii=False, indent=4, separators=(',', ': '))
                    count += 1
            except TypeError as e:
                print(player_end_id + index)