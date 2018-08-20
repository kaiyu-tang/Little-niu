#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/1 下午4:42  
# @Author  : Kaiyu  
# @Site    :   
# @File    : predict.py
import os
import sys
import torch
from pymongo import MongoClient

sys.path.append(os.path.dirname(__file__))
import Model
from Config import Config
from gensim.models import Word2Vec, FastText
import json
import random

from train import eval_model
import torch.nn.functional as F
from sklearn.metrics import classification_report
# from data.load_data import DataLoader
from data.load_data import DataProcess

base_path = os.path.dirname(TextCNN.__file__)


# default model loading
# _textcnn = TextCNN.TextCNN()
# if torch.cuda.is_available():
#     _textcnn.load_state_dict(torch.load(os.path.join(base_path, 'checkpoints/best_steps_650.pt')))
#     _textcnn.cuda()
# else:
#     _textcnn.load_state_dict(torch.load(os.path.join(base_path, 'checkpoints/best_steps_650.pt'), map_location='cpu'))
# _textcnn.eval()


# textcnn.double()

# big_word2vec_path = os.path.join(base_path, 'data/word_embed/wiki.zh/wiki.zh.bin')
# small_word2vec_path = os.path.join(base_path, 'data/word_embed/fasttext-skim-clean-2.pt')
# big_word2vec_model = FastText.load_fasttext_format(big_word2vec_path)
# small_word2vec_model = Word2Vec.load(small_word2vec_path)


class Predictor(object):
    # default model loading
    _textcnn = TextCNN.TextCNN()
    model_name = "best_acc0.9950530601482923_steps_49480.pkl"
    if torch.cuda.is_available():

        _textcnn = torch.load(os.path.join(base_path, "checkpoints",model_name),)
        # _textcnn = _textcnn.cuda()
        print("cuda")
    else:
        _textcnn = torch.load(os.path.join(base_path, "checkpoints",model_name),)

    _textcnn.eval()
    _data_processor = DataProcess()

    def __init__(self, model=_textcnn, cuda=torch.cuda.is_available()):
        self._model = model
        # self.big_word2vec_model = big_word2vec_model
        # self.small_word2vec_model = small_word2vec_model
        self._cuda = cuda
        self._data_processor = Predictor._data_processor

    def predicts(self, data_iter):
        """
        predict a batch of data
        :param data_iter: the data wrapped by DataLoader
        :return: predicts, logits (list,list) with the same length
        """
        predicteds = []
        logits = []

        all_corrects, all_loss, all_size = 0, 0, 0
        step = 0
        for feature, target in data_iter:
            step += 1
            # print(feature)
            # if self._cuda:
            #     feature, target = feature.cuda(), target.cuda()

            logit = self._model(feature)
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

    def predict(self, sentence):
        """
        predict a single sentence
        :param sentence: string (no any pre-process)
        :return: predict label and logit
        """
        sentence_padded = self._data_processor.pad_to_longest([sentence])
        sentence_vec = self._data_processor.convert_to_vectors_concate(sentence_padded, big_sequence_length=12,
                                                                       embed_dim=301)
        if torch.cuda.is_available():
            sentence_vec = sentence_vec.cuda()
        logit = self._model(sentence_vec)
        predicted = torch.max(logit.data, 1)[1][0]
        return int(predicted), list(map(float, logit[0]))


def label_zhibo360(predictor, zhibo360):
    """
       use the trained model to label the data of zhibo360
       :param predictor: the text classfy model
       :param bfwin007: the cursor
       :return: None
       """
    for item_ in zhibo360.find():
        live_texts_ = item_["live_text"]
        if 0 == len(live_texts_):
            zhibo360.delete_one({"_id": item_['_id']})
            continue
        for l_index_, l_item_ in enumerate(live_texts_):
            l_item_["p_label"] = predictor.predict(l_item_["text"])[0]
            live_texts_[l_index_] = l_item_
        zhibo360.update_one({"_id": item_['_id']}, {"$set": {"live_text": live_texts_}})


def label_bfwin007(predictor, bfwin007):
    """
    use the trained model to label the data of bfwin007
    :param predictor: the text classfy model
    :param bfwin007: the cursor
    :return: None
    """
    for item_ in bfwin007.find():
        live_texts_ = item_["live_texts"]
        if 0 == len(live_texts_):
            bfwin007.delete_one({"_id": item_["_id"]})
            continue
        for l_index_, l_item_ in enumerate(live_texts_):
            l_item_["p_label"] = predictor.predict(l_item_["live_text"])[0]
            live_texts_[l_index_] = l_item_
        bfwin007.update_one({"_id": item_['_id']}, {"$set": {"live_text": live_texts_}})


def label_7m(predictor, zhibo7m):
    """
        use the trained model to label the data of 7m.com
        :param predictor: the text classfy model
        :param bfwin007: the cursor
        :return: None
        """
    for item_ in zhibo7m.find():
        try:
            live_texts_ = item_["content"]["textFeed"]
        except Exception as e:
            zhibo7m.delete_one({"_id": item_['_id']})
            print("delete error id: {}".format(item_["_id"]))
            print(e)
        for l_index_, l_item_ in enumerate(live_texts_):
            l_item_["p_label"] = predictor.predict(l_item_["msg"])[0]
            live_texts_[l_index_] = l_item_
            # print(l_item_)
        zhibo7m.update_one({"_id": item_['_id']}, {"$set": {"textFeed": live_texts_}})


if __name__ == "__main__":
    # texts, labels = get_text("")

    # test_iters = DataLoader(texts, labels, Config.sequence_length, Config.word_embed_dim, cuda=Config.cuda,
    #                         batch_size=128, clean=True)
    # print("finish")
    predictor = Predictor()
    print(predictor.predict("点球进了！！！"))
    client = MongoClient()
    db = client["live_texts"]
    zhibo7m = db["zhibo7m"]
    label_7m(predictor, zhibo7m)
    bfwin007 = db["bfwin007"]
    label_bfwin007(predictor, bfwin007)
    # zhibo360 = db["zhibo360"]
    # label_zhibo360(predictor, zhibo360)
    sys.exit(0)
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
                          PAD=Config.PAD, clean=True, )

    # eval_model(textcnn, te_iters, Config)
    predicts, targets = predictor.predicts(te_iters)
    corects = 0
    res = [[] for i in range(30)]
    for text, predict_ in zip(sentences, predicts):
        res[int(predict_)].append(text)
        print("{}: {}".format(text, int(predict_)))
    print(res)
