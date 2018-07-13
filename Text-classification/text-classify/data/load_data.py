#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 下午8:15  
# @Author  : Kaiyu  
# @Site    :   
# @File    : load_data.py

import re
import sys

import numpy as np
import torch
from torch.autograd import Variable

import thulac


class DataLoader(object):
    def __init__(self, src_sents, label, max_len, embed_dim, cuda=True, batch_size=64, shuffle=True, evaluation=False,
                 clean=False, big_word2vec_model='', small_word2vec_model='', PAD='*'):
        self.cuda = cuda
        self.sents_size = len(src_sents)
        self._step = 0
        self._batch_size = batch_size
        self._stop_step = self.sents_size // self._batch_size + 1
        self._evaluation = evaluation
        self._clean = clean
        self._thu0 = ''
        if self._clean:
            self._thu0 = thulac.thulac()
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        self._embed_dim = embed_dim
        self._PAD = PAD
        if shuffle:
            self._shuffle()
        # loading word2vec model
        self._big_word2vec_model = big_word2vec_model
        self._small_word2vec_model = small_word2vec_model
        self._process = Data_Process(big_word2vec_model=self._big_word2vec_model,
                                     small_word2vec_model=self._small_word2vec_model,
                                     cut_model=self._thu0, max_length=self._max_len, PAD=self._PAD,
                                     evaluation=self._evaluation)

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
        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration

        _start = self._step * self._batch_size
        _bsz = min(self._batch_size, self.sents_size - _start)
        self._step += 1
        data = self._process.pad_to_longest(self._src_sents[_start:_start + _bsz])
        data = self._process.convert_to_vectors_plus(data, embed_dim=self._embed_dim)
        label = Variable(torch.from_numpy(self._label[_start:_start + _bsz]),
                         volatile=self._evaluation)
        if self.cuda:
            print(self.cuda)
            label = label.cuda()
            data = data.cuda()

        return data, label


class Data_Process(object):
    def __init__(self, big_word2vec_model, small_word2vec_model, max_length, cut_model, PAD='*', clean=False,
                 evaluation=False):
        self._big_word2vec_model = big_word2vec_model
        self._small_word2vec_model = small_word2vec_model
        self._cut_model = cut_model
        self._clean = clean
        self._max_length = max_length
        self._PAD = PAD
        self._evaluation = evaluation
        if self._clean:
            self._thu0 = thulac.thulac()
        pass

    def clean(self, text, thu0, stop_chars=''',?.!！;:(){}[]，。？；：（）【】 已在的中了．由—~'''):
        text = re.sub('[a-zA-Z]+', '', text)
        for c in stop_chars:
            text = text.replace(c, ' ')
        text = thu0.fast_cut(text)
        return [t[0] for t in text]

    def pad_to_longest(self, insts):
        for index, inst in enumerate(insts):
            if self._clean:
                inst = self.clean(''.join(inst), self._thu0)
            inst_len = len(inst)
            if inst_len == self._max_length:
                insts[index] = inst
            elif inst_len < self._max_length:
                insts[index] = inst + [self._PAD] * (self._max_length - inst_len)
            else:
                insts[index] = inst[:self._max_length]
        return insts

    def _convert_to_vectors(self, insts, length, embed_dim, model):
        insts_num = len(insts)
        data = np.zeros((insts_num, length, embed_dim))
        for index_0, inst in enumerate(insts):
            for index_1, word in enumerate(inst):
                if word != self._PAD and word in model:
                    try:
                        data[index_0][index_1] = model[word]
                    except Exception as e:
                        print(e)
        data = Variable(torch.from_numpy(data), volatile=self._evaluation)
        return data

    def convert_to_vectors(self, insts, emded_dim, model):
        return self._convert_to_vectors(insts, self._max_length, embed_dim=emded_dim, model=model)

    def convert_to_vectors_plus(self, insts, embed_dim):
        big_word_vecs = self._convert_to_vectors(insts, self._max_length, embed_dim=embed_dim,
                                                 model=self._big_word2vec_model)
        small_word_vecs = self._convert_to_vectors(insts, self._max_length, embed_dim=embed_dim,
                                                   model=self._small_word2vec_model)
        data = big_word_vecs + small_word_vecs
        return data

    def convert_to_vectors_concate(self, insts, big_sequence_length, embed_dim):
        if big_sequence_length > self._max_length:
            sys.exit(1)
        small_sequence_length = self._max_length - big_sequence_length
        big_word_vecs = self._convert_to_vectors(insts, big_sequence_length, embed_dim=embed_dim,
                                                 model=self._big_word2vec_model)
        small_word_vecs = self._convert_to_vectors(insts, small_sequence_length, embed_dim=embed_dim,
                                                   model=self._small_word2vec_model)
        data = np.concatenate((big_word_vecs, small_word_vecs), axis=0)
        return data


if __name__ == '__main__':
    # path = '/Users/harry/PycharmProjects/toys/Text-classification/text-classify/okoo-match'
    # js_data = readfile(path)
    # with open('', 'w') as f:
    #     json.dump({'all': js_data}, f, ensure_ascii=False, indent=4, separators=(',', ': '))

    pass

    # sentence = load_sentence_data('okoo-label.json')
