#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/1 下午4:42  
# @Author  : Kaiyu  
# @Site    :   
# @File    : predict.py
import os
import sys
import torch
sys.path.append(os.path.dirname(__file__))
import TextCNN
from Config import Config
from gensim.models import Word2Vec, FastText
import json
import random

from train import eval_model
from lxml import etree
import torch.nn.functional as F
from sklearn.metrics import classification_report

base_path = os.path.dirname(TextCNN.__file__)

# default model loading
textcnn = TextCNN.TextCNN()
if torch.cuda.is_available():
    textcnn.load_state_dict(torch.load(os.path.join(base_path, 'checkpoints/best_steps_650.pt')))
    textcnn.cuda()
else:
    textcnn.load_state_dict(torch.load(os.path.join(base_path, 'checkpoints/best_steps_650.pt'), map_location='cpu'))
textcnn.eval()
# textcnn.double()

# big_word2vec_path = os.path.join(base_path, 'data/word_embed/wiki.zh/wiki.zh.bin')
# small_word2vec_path = os.path.join(base_path, 'data/word_embed/fasttext-skim-clean-2.pt')
# big_word2vec_model = FastText.load_fasttext_format(big_word2vec_path)
# small_word2vec_model = Word2Vec.load(small_word2vec_path)


class Predictor(object):
    def __init__(self, model=textcnn, cuda=False):
        self.model = model
        # self.big_word2vec_model = big_word2vec_model
        # self.small_word2vec_model = small_word2vec_model
        self._cuda = cuda

    def predict(self, data_iter):
        predicteds = []
        logits = []

        all_corrects, all_loss, all_size = 0, 0, 0
        step = 0
        for feature, target in data_iter:
            step += 1
            # print(feature)
            # if self._cuda:
            #     feature, target = feature.cuda(), target.cuda()

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
    from data.load_data import DataLoader
    te_iters = DataLoader(sentences, labels, Config.sequence_length,
                          Config.word_embed_dim, cuda=Config.cuda, evaluation=True, batch_size=1024,
                          PAD=Config.PAD, clean=True,)

    # eval_model(textcnn, te_iters, Config)
    predict = Predictor(te_iters)
    predicts, targets = predict.predict()
    corects = 0
    res = [[] for i in range(30)]
    for text, predict_ in zip(sentences, predicts):
        res[int(predict_)].append(text)
        print("{}: {}".format(text, int(predict_)))
    print(res)
