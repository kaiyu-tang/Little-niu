#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/4 上午10:40  
# @Author  : Kaiyu  
# @Site    :   
# @File    : test.py
import json
import matplotlib.pyplot as plt

file = '/Users/harry/PycharmProjects/toys/Text-classification/text-cluster/data/match-data-dbow.json'

if __name__ == '__main__':
    all = [0 for i in range(8)]
    change = [0 for i in range(8)]
    with open(file) as f:
        data = json.load(f)
        for key0 in data:
            pre = 0
            for item in data[key0]:
                left = int(item['text'][-3])
                right = int(item['text'][-1])
                label = int(item['label'])
                all[label] += 1
                if left + right - pre > 0:
                    pre = left + right
                    change[label] += 1
    bili = [i/j for i,j in zip(change,all)]
    print(all)
    print(change)
    print(bili)

    plt.figure(0)
    plt.plot(all)
    plt.title('all')
    plt.figure(1)
    plt.plot(change)
    plt.title('changed')
    plt.figure(2)
    plt.plot(bili)
    plt.title('bili')
    plt.show()
