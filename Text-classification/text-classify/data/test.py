#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/7/3 下午4:00  
# @Author  : Kaiyu  
# @Site    :   
# @File    : test.py

from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
import json

if __name__ == '__main__':
    with open('okoo-merged-labels.json', encoding='utf-8') as f:
        data = json.load(f)['all']
        data = [(item['text'], str(item['merged_label'])) for item in data]
        train_data = data[:-1000]
        test_data = data[-1000:-1]
        model = NBC(train_data)
        for test_item in test_data:
            label_ = model.classify(test_item[0])
            print('True: {} predict: {}'.format(str(test_item[1]), label_))
        print(model.accuracy(test_data))
