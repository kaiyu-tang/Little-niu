#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/8/2 下午2:26  
# @Author  : Kaiyu  
# @Site    :   
# @File    : util.py
import numpy as np
from sklearn import preprocessing

def calculate_weight(sentences,labels):
    num_label = max(labels)
    label_dict = [0 for _ in range(num_label)]
    length = len(labels)
    for label_ in labels:
        label_dict[label_] += 1
    label_dict = np.array(label_dict).astype(float)
    return preprocessing.scale(label_dict, axis=0)