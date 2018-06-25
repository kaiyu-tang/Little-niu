#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 下午8:15  
# @Author  : Kaiyu  
# @Site    :   
# @File    : load_data.py

import os
import re

import numpy as np
from gensim.models import Word2Vec
from gensim.models.doc2vec import LabeledSentence
import json
import jieba
from itertools import chain
from json.decoder import JSONDecodeError
import numpy as np
import torch
from torch.autograd import Variable

from Config import Config

PAD = Config.PAD

def clean(text,stop_chars = ''',?.!！;:(){}[]，。？；：（）【】 已在的中了．由—~'''):
    text = re.sub('[a-zA-Z]+', '', text)
    for c in stop_chars:
        text = text.replace(c,' ')
    text = ' '.join(jieba.lcut(text))
    return ' '.join(text.split())
def readfile(dir_path):
    res = []

    file_names = os.listdir(dir_path)
    index = 0
    for file_name in file_names:
        if file_name == 'result-new.json':
            continue
        with open(os.path.join(dir_path, file_name), 'rb') as f:
            data = json.load(f)
            for item in data['narrate']:
                item['text'] = clean(item['text'])
                print(index)
                index += 1
            res.extend(data['narrate'])
    return res


def load_sentence_data(data_path):
    with open(data_path) as f_:
        js_data = json.load(f_)['all']
        # for item in js_data:
        # item['text'] = jieba.lcut(item['text'])

    return js_data


class DataLoader(object):
    def __init__(self, src_sents, label, max_len, embed_dim, cuda=True, batch_size=64, shuffle=True, evaluation=False,clean=False):
        self.cuda = cuda
        self.sents_size = len(src_sents)
        self._step = 0
        self._batch_size = batch_size
        self._stop_step = self.sents_size // self._batch_size + 1
        self.evaluation = evaluation
        self._clean = clean

        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        self._embed_dim = embed_dim
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __len__(self):
        return self._batch_size

    def __next__(self):
        def pad_to_longest(insts, max_len):
            for index, inst in enumerate(insts):
                if self._clean:
                    inst = clean(''.join(inst)).split()
                inst_len = len(inst)
                if inst_len < max_len:
                    insts[index] = inst + [PAD] * (max_len - inst_len)
                elif inst_len > max_len:
                    insts[index] = inst[:max_len]
            return insts

        def convert_to_vectors(insts, max_len, embed_dim):
            word2vec_model = Word2Vec.load(os.path.join(Config.dir_model, Config.word2vec_model_name))
            insts_num = len(insts)
            data = np.zeros((insts_num, max_len, embed_dim))
            for index_0, inst in enumerate(insts):
                for index_1, word in enumerate(inst):
                    if word != PAD and word in word2vec_model:
                        data[index_0][index_1] = word2vec_model[word]
            data = Variable(torch.from_numpy(data), volatile=self.evaluation)
            return data

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration

        _start = self._step * self._batch_size
        _bsz = min(self._batch_size, self.sents_size - _start)
        self._step += 1
        data = pad_to_longest(self._src_sents[_start:_start + _bsz], self._max_len)
        data = convert_to_vectors(data, max_len=self._max_len, embed_dim=self._embed_dim)
        label = Variable(torch.from_numpy(self._label[_start:_start + _bsz]),
                         volatile=self.evaluation)
        if self.cuda:
            print(self.cuda)
            label = label.cuda()
            data = data.cuda()

        return data, label


if __name__ == '__main__':
    # path = '/Users/harry/PycharmProjects/toys/Text-classification/text-classify/okoo-match'
    # js_data = readfile(path)
    # with open('', 'w') as f:
    #     json.dump({'all': js_data}, f, ensure_ascii=False, indent=4, separators=(',', ': '))
    labels_dic = {}
    with open("label_doc.text") as f:
        for index, line in enumerate(f):
            for key in re.findall('(\d+)',line):
                labels_dic[''.join(key)] = index
    cur_true_label = index+1
    with open('okoo-labels.json') as f:
        data = json.load(f)["all"]
        for item in data:
            label = item['label']
            if label in labels_dic:
                item['merged_label'] = labels_dic[label]
            else:
                print(item)
                print(cur_true_label)
                item['merged_label'] = cur_true_label
                cur_true_label += 1
    with open('okoo-merged-labels.json','w') as f:
        json.dump({'all': data}, f, ensure_ascii=False, indent=4, separators=(',', ': '))



    # sentence = load_sentence_data('okoo-label.json')
