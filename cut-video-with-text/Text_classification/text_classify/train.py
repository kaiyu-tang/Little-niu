#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/29 下午3:06  
# @Author  : Kaiyu  
# @Site    :   
# @File    : train.py
import json
import os
import time

import torch
import torch.utils.data.dataloader
import torch.nn.functional as F
from Model import TextCNN, TextRNN, TextVDCNN
from Config import Config
from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pandas as pd
from data.data_loader import MyDataset
import numpy as np


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
    y_true = []
    y_pred = []
    hidden_state = None
    from sklearn import metrics
    for data_ in data_iter:
        feature, target = data_[0], data_[1]
        y_true = np.concatenate([y_true, target])
        if model.name == "TextCNN":
            logit = model(feature)
        elif model.name == "TextRNN":
            logit, hidden_state = model(feature, hidden_state)
        elif model.name == "TextVDCNN":
            logit = model(torch.transpose(feature, 1, 2))

        y_pred = np.concatenate([y_pred, torch.max(logit, 1)[1].view(target.size()).data])
        if len(y_pred) != len(y_true):
            print("{} {}".format(len(y_pred), len(y_true)))
    #####
    # then get the ground truth and the predict label named y_true and y_pred
    if len(y_pred) < len(y_true):
        print("changed")
        y_true = y_true[:len(y_pred)]

    classify_report = metrics.classification_report(y_true, y_pred)
    # confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    score = metrics.accuracy_score(y_true, y_pred)
    print('classify_report : \n', classify_report)
    # print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('score: {0:f}'.format(score))

    return average_accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path + ".pt")
    torch.save(model, save_path + ".pkl")
    print('Save Sucessful, path: {}'.format(save_path))


def train(model, train_iter, dev_iter, args, weights, best_acc=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.options[model.name]["lr"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.options[model.name]["lr"])
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # word2vec_model = Word2Vec.load(os.path.join(Config.dir_model, Config.word2vec_model_name))
    steps = 0
    last_step = 0
    model.train()
    option = args.options[model.name]
    print('start training {}'.format(model.name))
    #print(-1)
    torch.backends.cudnn.benchmark = True
        # weights = weights.cuda()
    for epoch in range(option["epoch"]):
        cur_time = time.time()
        for data_ in train_iter:
            feature, target = data_[0].long(), data_[1]
            if model.name == "TextRNN":
                pass
            if args.cuda:
                feature = feature.float().cuda()
                target = target.cuda()
            hidden_state = None
            optimizer.zero_grad()
            if model.name == "TextCNN":
                logit = model(feature)
            elif model.name == "TextRNN":
                logit, hidden_state = model(feature, hidden_state)
            elif model.name == "TextVDCNN":
                feature = torch.transpose(feature, 1, 2)
                logit = model(feature)

            loss = F.cross_entropy(logit, target, weight=weights)
            loss.backward()
            optimizer.step()
        end_time = time.time()
        steps += 1
        print("step: {} time: {} loss: {}".format(steps, end_time - cur_time, loss))

        if steps % option["test_interval"] == 0:
            dev_acc = eval_model(model, dev_iter, args)
            model.train()
            if dev_acc > best_acc:
                best_acc = dev_acc
                last_step = steps
                if option["save_best"]:
                    save(model, args.dir_model, 'best_acc{}'.format(best_acc), steps)
            else:
                # if steps - last_step >= args.early_stop:
                #     print('early stop by {} steps.'.format(args.early_stop))
                pass
        # elif steps % option["save_interval"] == 0:
        #     # save(model, args.dir_model, 'snapshot', steps)
        #     pass
    return best_acc


def prepare_sen_lab(test=True):
    # data pre-process
    data_path = './data/okoo-merged-3-label.json'
    data = json.load(open(data_path, encoding='utf-8'))
    sentences = []
    labels = []
    for item in data:
        sentences.append(item['text'].split())
        labels.append(item['merged_label'])
    data_path = './data/zhibo7m.json'
    data = json.load(open(data_path, encoding="utf-8"))
    al = len(data)
    count = 0
    for item_ in data:
        sentences.append(item_["msg"].split())
        try:
            labels.append(item_["t_label"])
        except KeyError as e:
            count += 1
            labels.append(0)
            print(item_["msg"])

    print("all: {} error: {}".format(al, count))

    return np.asarray(sentences), np.asarray(labels)


if __name__ == '__main__':
    data = MyDataset()
    # train word2vec
    # text_path = 'data' + os.sep + 'okoo-merged-clean-cut-data.txt'
    # train_word_vectors(text_path, Config)
    # train text-Cnn
    print(data.voca_size)
    print('loading text model')
    textcnn = TextCNN()
    textrnn = TextRNN()
    textvdcnn = TextVDCNN(voca_size=data.voca_size)
    print('finished loading txt model')
    print('Cuda: {}'.format(Config.cuda))
    print("loading data")
    # data = pd.read_csv("./data/full-cut-clean.csv")
    # sentences, labels = data["sentence"].values.astype(np.float32), data["label"].values
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.06)
    iters = 0
    best_acc = 0



    print("loaded data")
    weights = torch.from_numpy(data.get_weight())
    if torch.cuda.is_available():
        weights = weights.cuda()
        textcnn.cuda()
        textrnn.cuda()
        textvdcnn.cuda()
    for train_index, test_index in sss.split(data.X, data.Y):
        start_time = time.time()
        train_sampler = SubsetRandomSampler(train_index)
        dev_sampler = SubsetRandomSampler(test_index)
        train_iters = DataLoader(data, batch_size=64, num_workers=8, sampler=train_sampler,
                                 )

        dev_iters = DataLoader(data, batch_size=2, num_workers=8, sampler=dev_sampler,
                               )
        print("start")
        end_time = time.time()
        iters += 1
        # start train
        best_acc = train(textvdcnn, train_iters, dev_iters, Config, best_acc=best_acc, weights=weights)
        print("Iter: {} time: {} Loading data successful".format(iters, end_time - start_time))
