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
    dir_model = "./checkpoints"

    PAD = "*"

    # textcnn
    sequence_length = 16
    word_embed_dim = 256
    class_num = 29
    kernel_num = 128
    kernel_size = (1, 2, 3, 5, 7, 8, 10)
    dropout = 0.5
    static = False
    lr = 0.001
    textcnn_epochs = 200
    log_interval = 50
    test_interval = 50
    save_interval = 3000
    save_best = 100
    early_stop = 0
    train_proportion = 0.98

    word_vec_train_epoch = 0

    # word2vec
    word2vec_dm_concat = 0  # very time consuming
    word2vec_net_size = 512
    word2vec_train_epoch = 20
    word2vec_window = 5
    word2vec_worker = multiprocessing.cpu_count()//3
    word2vec_min_count = 2
    word2vec_sg = 0
    word2vec_model_name = 'word2vec-cbow-5.pt'
    word2vec_negative = 0
    word2vec_iter = 20

    # fast_text
    fast_sg = 0
    fast_window = 5
    fast_min_count = 2
    fast_worker = multiprocessing.cpu_count()//3
    fast_iter = 20
    fast_model_name = 'fasttext-cbow-5.pt'
    # wordrank
    wordrank_window = 15
    wordrank_symmetric = 1
    wordrank_min_count = 2
    wordrank_iter = 100
    wordrank_worker = multiprocessing.cpu_count()
    wordrank_out_name = 'wordrank.pt'
    # thulac
    thulac_dict_path = ''
