#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/7/25 上午11:44  
# @Author  : Kaiyu  
# @Site    :   
# @File    : entry.py
from scrapy.cmdline import execute

execute(['scrapy', 'crawl', '7m', '-s', 'LOG_FILE=7m.log'])