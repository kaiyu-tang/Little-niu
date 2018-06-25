#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/22 下午1:42  
# @Author  : Kaiyu  
# @Site    :   
# @File    : youku.py
import os
import random
import sys
import time

import requests
from lxml import etree
import re
def get_video_url(page):
    videos = re.findall('href="(.+?)"',page)
    print()
    return videos


def download_videos(videos):
    hash_id = str(hash(videos[0]))
    path = os.path.join(os.getcwd(), hash_id)
    video_num = len(videos)
    os.mkdir(path)
    os.chdir(path)
    for index,url in enumerate(videos):
        if index%2:
            print("downloading index: {}/{}".format(index, video_num))
            #time.sleep(random.randint(30, 60))
            os.system('youtube-dl {}'.format(url))



if __name__=='__main__':
    page = ''''''
    videos = get_video_url(page)
    download_videos(videos)
