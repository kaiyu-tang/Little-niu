#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/1 下午4:42  
# @Author  : Kaiyu  
# @Site    :   
# @File    : predict.py
import torch

from TextCNN import TextCNN
from Config import Config
from gensim.models import Word2Vec, FastText
import json
import random
from data.load_data import DataLoader
from train import eval_model
from lxml import etree
import torch.nn.functional as F
from sklearn.metrics import classification_report


class Predict(object):
    def __init__(self, data_iter, model, big_word2vec_model, small_word2vec_model, cuda=False):
        self.textcnn = TextCNN()
        self.textcnn.load_state_dict(torch.load(
            "/Users/harry/PycharmProjects/toys/Text-classification/text-classify/checkpoints/best_steps_350.pt",
            map_location='cpu'))
        self.textcnn.eval()
        self.textcnn.double()
        self.model = model
        self.big_word2vec_model = big_word2vec_model
        self.small_word2vec_model = small_word2vec_model
        self.data_iter = data_iter
        self.cuda = cuda

    def predict(self):
        predicteds = []
        logits = []

        all_corrects, all_loss, all_size = 0, 0, 0
        step = 0
        for feature, target in self.data_iter:
            step += 1
            # print(feature)
            if self.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = self.model(feature)
            predicted = torch.max(logit.data, 1)[1].view(target.size()).data
            # print(predicted)
            predicteds.extend(predicted)
            logits.extend(logit)
            loss = F.cross_entropy(logit, target, size_average=False)

            cur_loss = loss.data[0]
            all_loss += cur_loss
            cur_corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            all_corrects += cur_corrects
        print('Evaluation - average loss: {:.6f} average acc: {:.4f}%'.format(
            float(all_loss) / (int(all_size) + 1), 100 * float(all_corrects) / (int(all_size) + 1)))

        return predicteds, logits


if __name__ == "__main__":
    # texts, labels = get_text("")
    textcnn = TextCNN()
    textcnn.load_state_dict(torch.load(
        "/Users/harry/PycharmProjects/toys/Text-classification/text-classify/checkpoints/best_steps_350.pt",
        map_location='cpu'))
    textcnn.eval()
    textcnn.double()

    big_word2vec_path = '/Users/harry/PycharmProjects/toys/Text-classification/text-classify/wiki.zh/wiki.zh.bin'
    small_word2vec_path = '/Users/harry/PycharmProjects/toys/Text-classification/text-classify/checkpoints/fasttext-skim-clean-2.pt'
    big_word2vec_model = FastText.load_fasttext_format(big_word2vec_path)
    small_word2vec_model = Word2Vec.load(small_word2vec_path)
    # test_iters = DataLoader(texts, labels, Config.sequence_length, Config.word_embed_dim, cuda=Config.cuda,
    #                         batch_size=128, clean=True)
    # print("finish")

    data_path = './data/test001.json'
    sentences = []
    labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for chang in data:
            for text_ in chang:
                sentences.append(text_[1].split())
                labels.append(28)
    te_iters = DataLoader(sentences, labels, Config.sequence_length,
                          Config.word_embed_dim, cuda=Config.cuda, evaluation=True, batch_size=1024,
                          PAD=Config.PAD, clean=True,
                          big_word2vec_model=big_word2vec_model, small_word2vec_model=small_word2vec_model)

    # eval_model(textcnn, te_iters, Config)
    predict = Predict(te_iters, textcnn, big_word2vec_model, small_word2vec_model)
    predicts, targets = predict.predict()
    corects = 0
    res = [[] for i in range(30)]
    for text, predict_ in zip(sentences, predicts):
        res[int(predict_)].append(text)
        print("{}: {}".format(text, int(predict_)))
    print(res)
