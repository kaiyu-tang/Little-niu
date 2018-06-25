#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/20 上午11:40  
# @Author  : Kaiyu  
# @Site    :   
# @File    : entrypoint.py.py

from scrapy.cmdline import execute
execute(['scrapy', 'crawl', 'foxsports'])