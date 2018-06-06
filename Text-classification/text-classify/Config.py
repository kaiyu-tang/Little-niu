#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 下午2:29  
# @Author  : Kaiyu  
# @Site    :   
# @File    : Config.py
import multiprocessing
import torch

class Config:
    # basic
    cuda = torch.cuda.is_available()
    dir_model = ''

    # textcnn
    sequence_length = 256
    word_embed_dim = 128
    class_num = 128
    kernel_num = 128
    kernel_size = (1, 2, 3, 5, 7, 8, 10)
    dropout = 0.5
    static = True
    lr = 0.001
    textcnn_epoches = 200
    log_interval = 20
    test_interval = 20
    save_best = True
    early_stop = 0

    # doc2vec
    doc2vec_epoch_num = 20
    dm_concat = 0  # very time consuming
    word2vec_net_size = 128
    train_epoch = 200
    window_size = 20
    works = multiprocessing.cpu_count()
    min_count = 2
