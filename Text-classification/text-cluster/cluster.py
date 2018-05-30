#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/25 上午10:44
# @Author  : Kaiyu
# @Site    :
# @File    : cluster.py

import json
import os
from itertools import chain

import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import seaborn as sns

# doc2vec parameters
vector_size = 64  # 300维
window_size = 20
min_count = 4
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 30
dm = 0  # 0 = dbow; 1 = dmpv
worker_count = 64  # number of parallel processes
kmeans_clusters = 8
path_model = ''

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
    punctuation = ''',?;:(){}[]，？；：（）【】'''
    for key in corpus.keys():
        corpus[key] = [z.replace('。', ' ') for z in corpus[key]]
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


def get_dataset(root=''):
    print('loading data')
    base_dir = os.getcwd()
    if not os.path.exists(os.path.join(base_dir, 'match-data.json')):
        file_dir = '/Users/harry/Downloads/okoo-matches-NoTime'
        data = read_from_dir(file_dir)
    else:
        with open(os.path.join(base_dir, 'match-data.json')) as f:
            data = json.load(f)
    data = data_clean(data)
    data = labelizeData(data)
    print('loaded data')
    # data = np.array(data)
    return data


def train_doc2vec(train_data, dim=vector_size, epoch_num=train_epoch, window_size=window_size, workers=worker_count,
                  min_count=min_count, new_model=True):
    print('training model')
    all_data = train_data
    if not os.path.exists(os.path.join(path_model, 'model_dm')) or new_model:
        model_dm = Doc2Vec(dm=1, vector_size=dim, window=window_size,
                           work=workers, min_count=min_count, epoches=epoch_num)
    else:
        model_dm = Doc2Vec.load(os.path.join(path_model, 'model_dm'))
    if not os.path.exists(os.path.join(path_model,'model_dbow')) or new_model:
        model_dbow = Doc2Vec(dm=0, vector_size=dim, window=window_size,
                             work=workers, min_count=min_count, epoches=epoch_num)
    else:
        model_dbow = Doc2Vec.load(os.path.join(path_model, 'model_dbow'))
    if not new_model:
        return model_dm, model_dbow
    # build vocabulary
    model_dm.build_vocab(all_data)
    model_dbow.build_vocab(all_data)
    # all_data = np.array(all_data)
    # train model each epoch permutate the data
    model_dbow.train(all_data, epochs=model_dbow.epochs, total_examples=model_dbow.corpus_count)
    model_dm.train(all_data, epochs=model_dm.epochs, total_examples=model_dm.corpus_count)
    model_dm.save(os.path.join(path_model,'model_dm'))
    model_dbow.save(os.path.join('model_dbow'))
    print('finished train model')
    return model_dm, model_dbow


def get_doc_vec(model, corpus, output_dim=vector_size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, output_dim)) for z in corpus]
    return np.concatenate(vecs)


def Cluster(train_vecs, model_path=None, test_vecs=None):
    print('training cluster')
    kmeans_model = KMeans(n_clusters=kmeans_clusters, n_jobs=worker_count, )
    kmeans_model.fit(train_vecs)
    labels = kmeans_model.predict(train_vecs)
    cluster_centers = kmeans_model.cluster_centers_
    print('finished cluster')

    return labels, cluster_centers


def visualize(vecs, labels):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit(vecs)
    X_tsne = preprocessing.normalize(X_tsne.embedding_, norm='l2')
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(X_tsne.shape[0]):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / kmeans_clusters),
                 fontdict={'weight': 'bold', 'size': 3})
    plt.legend()
    plt.show()
    print()


if __name__ == '__main__':
    data = get_dataset()
    model_dm, model_dbow = train_doc2vec(data, new_model=True)
    vecs_dm = get_doc_vec(model_dm, data)
    vecs_dbow = get_doc_vec(model_dbow, data)
    labels_dm, cluster_centers_dm = Cluster(vecs_dm)
    labels_dbow, cluster_centers_dbow = Cluster(vecs_dbow)
    visualize(vecs_dm, labels_dm)
    print()
