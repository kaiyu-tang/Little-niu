#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/29 下午3:05  
# @Author  : Kaiyu  
# @Site    :   
# @File    : main.py
import os

import requests
from gensim.models import Word2Vec
from lxml import etree





if __name__ == "__main__":
    #get_text("https://www.zhibo8.cc/zhibo/zuqiu/2018/0614114286.htm?redirect=zhibo")
    word2vec_model = Word2Vec.load(os.path.join(Config.dir_model, Config.word2vec_model_name))
