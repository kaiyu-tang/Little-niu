#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/9/13 下午2:44  
# @Author  : Kaiyu  
# @Site    :   
# @File    : data_loader.py
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch.cuda
import pandas as pd
import os

path = os.path.dirname(__file__)


class MyDataset(Dataset):
    def __init__(self, data=None, cuda=torch.cuda.is_available(), embed_dim=300, seq_length=16,
                 text_path=os.path.join(path, "full-cut-clean.csv"),
                 vec_path=os.path.join(path, "word_embed/wiki.zh/wiki.zh.vec")):
        self._vec_path = vec_path
        self._text_path = text_path
        self._seq_length = seq_length
        self._cuda = cuda

        self._vec, self._word2idx, self._vocab_size = self.load_vec(self._vec_path)
        self._embed_dim = embed_dim
        if data == None:
            self._X, self._Y = self.read_file(self._text_path)
        else:
            self._X, self._Y = data[0], data[1]
        tmp = np.bincount(self._Y) / len(self._Y)
        self._weights = (np.ones((tmp.shape[0])) - tmp).astype(np.float32)
        self._Y = self._Y.astype(np.int64)

    def read_file(self, text_path):
        data = pd.read_csv(text_path)
        X = data["sentence"]
        Y = data["label"]
        return X.values, Y.values

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def get_weight(self):
        return self._weights

    def load_vec(self, vec_path):
        word2idx = {}
        count = 0
        vec = {}
        with open(vec_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
            for line in f:
                tokens = line.rstrip().split(" ")
                vec[tokens[0]] = np.asarray(list(map(float, tokens[1:])))
                if tokens[0] in word2idx:
                    word2idx[tokens[0]] = (word2idx[tokens[0]][0] + 1, word2idx[tokens[0]][1])
                else:
                    word2idx[tokens[0]] = (1, count)
                    count += 1
        for item in sorted(word2idx.items(), key=lambda x: x[1][0]):
            word2idx[item[0]] = item[1][1]
        return vec, word2idx, count

    def word2vec(self, word):
        if word in self._vec:
            return self._vec[word]
        else:
            return np.zeros(self._embed_dim)

    def get_embed(self):
        embed = np.zeros((self._vocab_size, self._embed_dim), dtype=np.float32)
        for word in self._word2idx:
            idx = self._word2idx[word]
            embed[idx] = self.word2vec(word)
        return embed

    def __getitem__(self, item):
        re = np.zeros((self._seq_length, self._embed_dim), dtype=np.float32)
        for index, word in enumerate(self._X[item]):
            if index < self._seq_length:
                re[index] = self.word2vec(word)
            else:
                break

        return re, self._Y[item]

        # return np.asarray([np.asarray(re, dtype=np.float32) +
        #                    np.zeros((self._seq_length, len(re[0])), dtype=np.float32),
        #                    self._Y[item]])

    def __len__(self):
        return len(self._X)


if __name__ == "__main__":
    path = "/Users/harry/PycharmProjects/toys/cut-video-with-text/Text_classification/text_classify/data/"
    data = MyDataset()
    mydataloader = DataLoader(dataset=data, batch_size=3, shuffle=True, num_workers=5)
    for i in mydataloader:
        print(i)
