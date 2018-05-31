#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/30 下午2:46  
# @Author  : Kaiyu  
# @Site    :   
# @File    : load_data.py
import os

import numpy as np
from gensim.models.doc2vec import LabeledSentence
import json


def read_from_dir(root):
    file_list = os.listdir(root)
    res = {}
    with open('match-data.json', 'w') as res_f:
        for file_ in file_list:
            id = file_[:-5]
            with open(os.path.join(root, file_)) as f:
                narrate = json.load(f)['narrate']
                res_na = []
                narrate[-1]['time'] = "90'"
                if narrate[0]['time'] == '&nbsp;':
                    narrate[0]['time'] = "0'"
                for nar_ in narrate:
                    res_na.append('{} {}-{}'.format(nar_['text'], nar_['vs'][0], nar_['vs'][1]))
            res[id] = res_na
        json.dump(res, res_f, ensure_ascii=False, indent=4, separators=(',', ': '))
    return res


def data_clean(corpus):
    punctuation = ''',?;:(){}[]，？；：（）【】 '''
    for key in corpus.keys():
        corpus[key] = [z.replace('。', '.') for z in corpus[key]]
    for c in punctuation:
        for key in corpus.keys():
            corpus[key] = [z.replace(c, ' ') for z in corpus[key]]
    return corpus


def labelizeData(corpus):
    labelized = []
    for key in corpus.keys():
        for index, sentence in enumerate(corpus[key]):
            labelized.append(LabeledSentence(sentence, ['{}-{}'.format(key, index)]))
    return labelized


def get_dataset(root='', raw=False, test=0):
    print('loading data')
    base_dir = os.getcwd()
    # print(base_dir)
    if not os.path.exists(os.path.join(base_dir, 'match-data.json')):
        file_dir = '/Users/harry/Downloads/okoo-matches-NoTime'
        data = read_from_dir(file_dir)
    else:
        if test == 1:
            name = 'match-data-test.json'
        else:
            name = 'match-data.json'
        with open(os.path.join(base_dir, name)) as f:
            data = json.load(f)
    data = data_clean(data)
    print('loaded data')
    # data = np.array(data)
    if raw:
        return data
    else:
        return labelizeData(data)


def get_doc_vec(model, corpus, output_dim=-1):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, output_dim)) for z in corpus]
    return np.concatenate(vecs)
