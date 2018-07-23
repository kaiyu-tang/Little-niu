#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/29 下午3:06  
# @Author  : Kaiyu  
# @Site    :   
# @File    : train.py
import json
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from TextCNN import TextCNN
from Config import Config
from gensim.models import Word2Vec, FastText
from gensim.models.wrappers import Wordrank
from gensim.models.word2vec import LineSentence
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import thulac


def train_word_vectors(text_path, args):
    sentences = LineSentence(text_path)
    print("loading word2vec")
    # labels = [text['label'] for text in data]
    model_word2vec = Word2Vec(sentences=sentences, size=args.word_embed_dim, window=args.word2vec_window,
                              min_count=args.word2vec_min_count, workers=args.word2vec_worker, sg=args.word2vec_sg,
                              negative=args.word2vec_negative, iter=args.word2vec_iter, )
    print('loading fast text')
    model_fasttext = FastText(sentences=sentences, sg=args.fast_sg, size=args.word_embed_dim, window=args.fast_window,
                              min_count=args.fast_min_count, workers=args.fast_worker, iter=args.fast_iter, )
    # print('loading word rank')
    # model_wordrank = Wordrank.train(wr_path=args.dir_model, size=args.word_embed_dim, corpus_file=text_path,
    #                                 window=args.wordrank_window, out_name=args.wordrank_out_name,
    #                                 symmetric=args.wordrank_symmetric, min_count=args.wordrank_min_count,
    #                                 iter=args.wordrank_iter,
    #                                 np=args.wordrank_worker)

    # model_word2vec.build_vocab(sentences=sentences)
    # model_fasttext.build_vocab(sentences=sentences)

    print("start training")
    for epoch in range(args.word_vec_train_epoch):
        # random.shuffle(sentences)
        model_word2vec.train(sentences=sentences, epochs=model_word2vec.iter,
                             total_examples=model_word2vec.corpus_count)
        # model_fasttext.train(sentences=sentences, epochs=model_fasttext.iter,
        #                      total_examples=model_fasttext.corpus_count)
        print(epoch)
        if epoch % 20 == 0:
            model_word2vec.save(os.path.join(args.dir_model, str(epoch) + "-" + args.word2vec_model_name))
            # model_fasttext.save(os.path.join(args.dir_model, str(epoch) + "-" + args.fast_model_name))
    model_word2vec.save(os.path.join(args.dir_model, args.word2vec_model_name))
    model_fasttext.save(os.path.join(args.dir_model, args.fast_model_name))
    # model_wordrank.save(os.path.join(args.dir_model, str(epoch) + "-" + args.wordrank_model_name))
    print('finished training')

def eval_model(model, data_iter, args):
    model.eval()
    all_corrects, all_loss, all_size = 0, 0, 0
    step = 0
    model.double()
    y_true = []
    y_pred = []
    from sklearn import metrics
    for feature, target in data_iter:
        y_true.extend(map(int, target))
        logit = model(feature)
        y_pred.extend(map(int, torch.max(logit, 1)[1].view(target.size()).data))
    #####
    # then get the ground truth and the predict label named y_true and y_pred
    classify_report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    score = metrics.accuracy_score(y_true, y_pred)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('score: {0:f}'.format(score))

    return all_corrects


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path + ".pt")
    torch.save(model, save_path + ".pkl")
    print('Save Sucessful, path: {}'.format(save_path))


def train(model, train_iter, dev_iter, args, word2vec_path=''):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # word2vec_model = Word2Vec.load(os.path.join(Config.dir_model, Config.word2vec_model_name))
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    model.double()
    print('start training')
    for epoch in range(args.textcnn_epochs):
        for feature, target in train_iter:
            # feature = feature.data()
            # feature.data.t_()
            # target.data.sub_()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1

            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / target.shape[0]


            if steps % args.test_interval == 0:
                dev_acc = eval_model(model, dev_iter, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.dir_model, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                # save(model, args.dir_model, 'snapshot', steps)
                pass

###你好啊

if __name__ == '__main__':
    # data pre-process
    data_path = './data/okoo-merged-labels.json'
    data = json.load(open(data_path, encoding='utf-8'))['all']
    sentences = []
    labels = []
    for item in data:
        sentences.append(item['text'].split())
        labels.append(item['merged_label'])
    data_len = len(data)
    train_index = int(data_len * Config.train_proportion)
    # train word2vec
    text_path = 'data' + os.sep + 'okoo-merged-clean-cut-data.txt'
    # train_word_vectors(text_path, Config)

    # train text-Cnn
    print('loading textcnn model')
    textcnn = TextCNN()
    print('finished loading txtcnn model')
    print('Cuda: {}'.format(Config.cuda))
    print("loading data")
    from data.load_data import DataLoader
    train_iters = DataLoader(sentences[:train_index], labels[:train_index], Config.sequence_length,
                             cuda=Config.cuda, batch_size=1024, PAD=Config.PAD, embed_dim=Config.embed_dim,
                            )
    dev_iters = DataLoader(sentences[train_index:], labels[train_index:], Config.sequence_length,
                           cuda=Config.cuda, evaluation=True, batch_size=1024,
                           PAD=Config.PAD, embed_dim=Config.embed_dim,
                           )
    print("loading data successful")
    # start train
    train(textcnn, train_iters, dev_iters, Config)
