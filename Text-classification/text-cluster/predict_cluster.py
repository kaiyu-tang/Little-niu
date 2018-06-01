#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/30 下午2:48  
# @Author  : Kaiyu  
# @Site    :   
# @File    : predict_cluster.py
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import json
import multiprocessing
import os
import sys
from itertools import chain
import train_cluster
import load_data
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.externals import joblib
from sklearn.cluster import KMeans

path_model = 'model'
if not os.path.exists(os.path.join(os.getcwd(), path_model)):
    os.makedirs(os.path.join(os.getcwd(), path_model))
path_data = 'data'
if not os.path.exists(os.path.join(os.getcwd(), path_data)):
    os.makedirs(os.path.join(os.getcwd(), path_data))

def visualize(tsne_model, vecs, labels):
    X_tsne = preprocessing.normalize(tsne_model.embedding_, norm='l2')
    num_vecs = len(vecs)
    for i in range(num_vecs):
        plt.plot(X_tsne[i][0], X_tsne[i][1], marksize=labels[i])
    plt.show()


def label_infer(model, vecs):
    return model.predict(vecs)


def save_labeled_data(corpus, labels, file_name):
    index = 0
    res = {}
    for key in corpus:
        tmp_re_ = []
        for text in corpus[key]:
            tmp_dict_ = {'text': text, 'label': str(labels[index])}
            tmp_re_.append(tmp_dict_)
            index += 1
        res[key] = tmp_re_
    with open(os.path.join(path_data, file_name), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    test = False

    doc2vec_dm_model = Doc2Vec.load(os.path.join(path_model, 'doc2vec_dm'))
    doc2vec_dbow_model = Doc2Vec.load(os.path.join(path_model, 'doc2vec_dbow'))
    km_dm_model = joblib.load(os.path.join(path_model, 'kmeans_dm.pkl'))
    km_dbow_model = joblib.load(os.path.join(path_model, 'kmeans_dbow.pkl'))
    corpus = load_data.get_dataset(test=test, cut=False)
    vecs_dm = load_data.get_doc_vec(doc2vec_dm_model, corpus)
    vecs_dbow = load_data.get_doc_vec(doc2vec_dbow_model, corpus)

    labels_dm = label_infer(km_dm_model, vecs_dm)
    labels_dbow = label_infer(km_dbow_model, vecs_dbow)
    corpus = load_data.get_dataset(raw=True, test=test)
    save_labeled_data(corpus, labels_dm, 'match-data-dm.json')
    save_labeled_data(corpus, labels_dm, 'match-data-dbow.json')

