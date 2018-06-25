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
from gensim.models import Word2Vec
import random
from data.load_data import load_sentence_data, DataLoader
from matplotlib import pyplot as plt


def train_word2vec(sentences, args):
    print("loading data")

    print("finished load data")
    # labels = [text['label'] for text in data]
    model = Word2Vec(sentences=sentences, size=args.word_embed_dim, window=args.window_size,
                     min_count=args.min_count,
                     workers=args.works, sg=args.word2vec_sg)
    # model.build_vocab(sentences=sentences)
    print("finished build_vocab")
    for epoch in range(args.word2vec_train_epoch):
        random.shuffle(sentences)
        model.train(sentences=sentences, epochs=args.word2vec_epoch_num, total_examples=model.corpus_count)
        print(epoch)
        if epoch % 20 == 0:
            model.save(os.path.join(args.dir_model, str(epoch) + "-" + args.word2vec_model_name))
    model.save(os.path.join(args.dir_model, str(epoch) + "-" + args.word2vec_model_name))


def eval_model(model, data_iter,args):
    model.eval()
    corrects, avg_loss = 0, 0
    step = 0
    model.double()
    for feature, target in data_iter:
        step += 1
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        size = len(data_iter)
        avg_loss /= size
        accuracy = 100.0 * corrects / size
        print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
            avg_loss, accuracy / step, corrects, size))
    return accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path+".pt")
    torch.save(model, save_path+".pkl")
    print('Save Sucessful, path: {}'.format(save_path))


def train(model, train_iter, dev_iter, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    word2vec_model = Word2Vec.load(os.path.join(Config.dir_model, Config.word2vec_model_name))
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    model.double()
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
                print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
                    steps, loss.data[0], accuracy, corrects, 1))

            if steps % args.test_interval == 0:
                dev_acc = eval_model(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.dir_model, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.dir_model, 'snapshot', steps)


def predict(model, text, args):
    pass



if __name__ == '__main__':



    textcnn = TextCNN()
    data_path = './data/okoo-merged-labels.json'
    data = load_sentence_data(data_path)
    sentences = []
    labels = []

    #stati = [[] for i in range(140)]
    for text in data:
        sentence = text['text']
        sentences.append(sentence.split())
        label = int(text['merged_label'])
        #stati[label].append(sentence)
        if label > Config.class_num:
            Config.class_num = label
        labels.append(label)
    # static_ = []
    # for label, sentence in enumerate(stati):
    #     tmp_ = {"Num": len(sentence), "Label": label, "Text": sentence}
    #     static_.append(tmp_)
    # static_ = sorted(static_,key=lambda x: x["Num"])
    # #stati.reverse()
    # plt.plot([text["Num"] for text in static_])
    # plt.show()
    # with open('static.json', 'w') as f:
    #     json.dump({'all': static_}, f, ensure_ascii=False, indent=4, separators=(',', ': '))
    #     f.flush()
    #train_word2vec(sentences, Config)
    data_len = len(data)
    train_index = int(data_len * Config.train_proportion)
    print(Config.cuda)
    train_iters = DataLoader(sentences[:train_index], labels[:train_index], Config.sequence_length,
                             Config.word_embed_dim, cuda=Config.cuda, batch_size=1024)
    for i,j in train_iters:
        print()
    dev_iters = DataLoader(sentences[train_index:], labels[train_index:], Config.sequence_length,
                           Config.word_embed_dim, cuda=Config.cuda, evaluation=True, batch_size=1024)
    train(textcnn, train_iters, dev_iters, Config)
