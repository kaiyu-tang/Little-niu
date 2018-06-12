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
    cuda = False
    dir_model = "./checkpoints"

    # textcnn
    sequence_length = 30
    word_embed_dim = 8
    class_num = 160
    kernel_num = 128
    kernel_size = (1, 2, 3, 5, 7, 8, 10)
    dropout = 0.5
    static = True
    lr = 0.001
    textcnn_epochs = 200
    log_interval = 20
    test_interval = 20
    save_interval = 3000
    save_best = True
    early_stop = 0
    train_proportion = 0.97

    # doc2vec
    word2vec_epoch_num = 12
    dm_concat = 0  # very time consuming
    word2vec_net_size = 128
    word2vec_train_epoch = 200
    window_size = 10
    works = multiprocessing.cpu_count()//3
    min_count = 2
    word2vec_sg = 0
    word2vec_model_name = '2word2vec-cbow'