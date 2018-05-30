#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/25 上午10:44
# @Author  : Kaiyu
# @Site    :
# @File    : train_cluster.py

import json
import multiprocessing
import os
import sys
from itertools import chain
import load_data
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.externals import joblib
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
train_epoch = 300
dm = 0  # 0 = dbow; 1 = dmpv
worker_count = multiprocessing.cpu_count()  # number of parallel processes
kmeans_clusters = 8
path_model = 'model'
if not os.path.exists(os.path.join(os.getcwd(), path_model)):
    os.makedirs(os.path.join(os.getcwd(), path_model))





def train_doc2vec(train_data, dim=vector_size, epoch_num=train_epoch, window_size=window_size, workers=worker_count,
                  min_count=min_count, new_model=True):
    print('training model')
    all_data = train_data
    if not os.path.exists(os.path.join(path_model, 'model_dm')) or new_model:
        model_dm = Doc2Vec(dm=1, vector_size=dim, window=window_size,
                           work=workers, min_count=min_count, epoches=epoch_num)
    else:
        model_dm = Doc2Vec.load(os.path.join(path_model, 'model_dm'))
    if not os.path.exists(os.path.join(path_model, 'model_dbow')) or new_model:
        model_dbow = Doc2Vec(dm=0, vector_size=dim, window=window_size,
                             work=workers, min_count=min_count, epoches=epoch_num)
    else:
        model_dbow = Doc2Vec.load(os.path.join(path_model, 'model_dbow'))
    if not new_model:
        print('finished train model')
        return model_dm, model_dbow
    # build vocabulary
    model_dm.build_vocab(all_data)
    model_dbow.build_vocab(all_data)
    # all_data = np.array(all_data)
    # train model each epoch permutate the data
    model_dbow.train(all_data, epochs=model_dbow.epochs, total_examples=model_dbow.corpus_count)
    model_dm.train(all_data, epochs=model_dm.epochs, total_examples=model_dm.corpus_count)
    model_dm.save(os.path.join(path_model, 'doc2vec_dm'))
    model_dbow.save(os.path.join(path_model, 'doc2vec_dbow'))
    print('finished train model')
    return model_dm, model_dbow





def train_cluster(train_vecs, model_name=None, test_vecs=None):
    print('training cluster')
    kmeans_model = KMeans(n_clusters=kmeans_clusters, n_jobs=worker_count, )
    kmeans_model.fit(train_vecs)
    labels = kmeans_model.predict(train_vecs)
    cluster_centers = kmeans_model.cluster_centers_
    joblib.dump(kmeans_model, os.path.join(path_model, model_name))
    # kmeans_model.save(os.path.join(path_model, model_name))
    print('finished cluster')

    return labels, cluster_centers


def train_tsne(vecs, labels, model_name):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit(vecs)
    joblib.dump(tsne, os.path.join(path_model, model_name))
    '''
    # tsne.save(os.path.join(path_model, 'tsne.h5'))
    X_tsne = preprocessing.normalize(X_tsne.embedding_, norm='l2')
    
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(X_tsne.shape[0]):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / kmeans_clusters),
                 fontdict={'weight': 'bold', 'size': 3})
    plt.legend()
    # plt.show()
    '''
    print()


if __name__ == '__main__':
    train_epoch = sys.argv[1]
    data = load_data.get_dataset()
    model_dm, model_dbow = train_doc2vec(data, new_model=True)
    vecs_dm = load_data.get_doc_vec(model_dm, data)
    vecs_dbow = load_data.get_doc_vec(model_dbow, data)
    labels_dm, cluster_centers_dm = train_cluster(vecs_dm, 'kmeans_dm.pkl')
    labels_dbow, cluster_centers_dbow = train_cluster(vecs_dbow, 'kmeans_dbow.pkl')
    train_tsne(vecs_dm, labels_dm, 'tsne_dm.pkl')
    train_tsne(vecs_dbow, labels_dbow, 'tsne_dbow.pkl')
    print()
