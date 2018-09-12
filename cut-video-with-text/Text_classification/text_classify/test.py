#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/8/20 下午3:28  
# @Author  : Kaiyu  
# @Site    :   
# @File    : test.py

from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text
import gluonnlp as nlp

if __name__ == "__main__":
    text_data = "你好"
    #tmp = nlp.embedding.list_sources("fasttext")
    #print(tmp)
    embedding = nlp.embedding.create("fasttext", source="wiki.zh")
    print(embedding["罗纳尔多"])