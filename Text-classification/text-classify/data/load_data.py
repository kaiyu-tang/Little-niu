#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 下午8:15  
# @Author  : Kaiyu  
# @Site    :   
# @File    : load_data.py

import os

import numpy as np
from gensim.models.doc2vec import LabeledSentence
import json
import jieba
from itertools import chain
from json.decoder import JSONDecodeError


def readfile(dir_path):
    res = []
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        with open(os.path.join(dir_path, file_name)) as f:
            data = json.load(f)
            res.extend(data['narrate'])
    return res


def load_data(data_path):
    with open(data_path) as f_:
        js_data = json.load(f_)['all']
        for item in js_data:
            item['text'] = jieba.cut(item['text'])
    return js_data


if __name__ == '__main__':
    path = ''
    js_data = readfile(path)
    with open('okoo-label.json', 'w') as f:
        json.dump({'all': js_data}, f)
    sentence = load_data(js_data)
