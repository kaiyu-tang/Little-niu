#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/29 下午12:29  
# @Author  : Kaiyu  
# @Site    :   
# @File    : data-process.py
"""
this file is used to process data
"""
import csv
import json
import pandas as pd

import numpy as np
import os
import re
import sys

from gensim.models import FastText, Word2Vec
import thulac
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jieba


def okoo_merge_label(file_name):
    """
    merge the labels of the data of okoo.com to 7
    :param file_name:
    :return:
    """
    labels_dic = {}
    label = 0
    with open("label_doc_3", encoding='utf-8') as f:
        for line in f:
            if len(line) < 2:
                continue
            for key in re.findall('(\d+)', line):
                labels_dic[''.join(key)] = label
            label += 1
    cur_true_label = label + 1
    with open(file_name, encoding='utf-8') as f1:
        texts = []
        data = json.load(f1)['all']
        for text_ in data:
            label = text_['label']
            if label in labels_dic:
                text_['merged_label'] = labels_dic[label]
            else:
                print(text_)
                text_['merged_label'] = cur_true_label
            # text_['text'] = ' '.join([c[0] for c in thu0.fast_cut(text_['text'])])
            texts.append(text_)

    with open('okoo-merged-3-label.json', 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4, separators=(',', ': '))


def readfile(dir_path, out_file_name):
    res = []
    out_f = open(out_file_name, 'w', encoding='utf-8', errors='ignore')
    file_names = os.listdir(dir_path)
    index = 0
    for file_name in file_names:
        if file_name[:6] == 'result' or file_name == '.DS_Store':
            continue
        with open(os.path.join(dir_path, file_name), errors='ignore', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(e)
            for item in data['narrate']:
                text = item['text']
                out_f.write(text + "\n")
                # print(index)
                index += 1
            # res.extend(data['narrate'])
    return res


def chose_topk(sentences, labels, k):
    thu0 = thulac.thulac()
    labeled_sentences = [[] for i in range(max(labels) + 1)]

    def cut(sen):
        re = []
        for item in sen:
            tmp = [c[0] for c in thu0.fast_cut(item)]
            re.append(' '.join(tmp))
        return ' '.join(re)

    for sen_, label_ in zip(sentences, labels):
        labeled_sentences[label_].append(sen_)
    for label_, sen_ in enumerate(labeled_sentences):
        labeled_sentences[label_] = cut(sen_)
    with open('tf-idf-0.05.json', 'w', encoding='utf-8') as f:
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(labeled_sentences))
        words = vectorizer.get_feature_names()
        weights = tfidf.toarray()
        res = {}
        for label_ in range(len(weights)):
            # f.write("-------第{}类文本的tf-idf权重------\n".format(label_))
            tmp_ = []
            rank = [index for index, value in
                    sorted(list(enumerate(weights[label_])), key=lambda x: x[1], reverse=True)]
            for j in rank:
                if weights[label_][j] > 0.05:
                    tmp_.append((words[j], weights[label_][j]))
                    # f.write('{} : {}\n'.format(words[j], weights[label_][j]))
            res[label_] = tmp_
        json.dump(res, f, ensure_ascii=False, indent=2, separators=(',', ': '))


def clean_with_tf_idf(in_file_name, tf_idf_name):
    thu0 = thulac.thulac()
    data = json.load(open(in_file_name, encoding='utf-8'))
    tf_idf = json.load(open(tf_idf_name, encoding='utf-8'))
    for index_, item_ in enumerate(data):
        text_ = item_['text']
        label_ = item_['merged_label']
        tmp_ = ' '.join([c[0] for c in thu0.fast_cut(text_) if c[0] in tf_idf[str(label_)]])
        item_['text'] = tmp_
    json.dump(data, open('okoo-merged-clean-cut-data,json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2,
              separators=(',', ': '))


def process_zhibo360(zhibo360):
    """
    pre-process zhibo360 data in mongodb, update the live_text
    :param zhibo360: the cosur of collection
    :return: None
    """
    for item_ in zhibo360.find():
        title_ = item_['title']
        texts_ = item_['text'].split("。")
        time_ = item_['time']
        new_texts_ = []
        if len(texts_) > 5:
            for text_ in texts_:
                if "分钟" in text_:
                    if "一分钟" in text_:
                        text_ = text_.replace("一分钟", "1分钟")
                    if "两分钟" in text_ or "二分钟" in text_:
                        text_ = text_.replace("两分钟", "2分钟").replace("二分钟", "2分钟")
                    if "三分钟" in text_:
                        text_ = text_.replace("三分钟", "3分钟")
                    if "五分钟" in text_:
                        text_ = text_.replace("五分钟", "5分钟")
                    try:
                        live_time = re.findall("(\d+).*分钟", text_)[0]
                    except IndexError as e:
                        continue
                    new_texts_.append({"text": text_, "time": int(live_time)})
            new_texts_.sort(key=lambda x: x["time"])
        zhibo360.update_one({"_id": item_['_id']}, {"$set": {"live_text": new_texts_}})
        print(item_)


def process_bfwin007(bfwin007):
    """
    pre-process bfwin007 data in mongodb, update the live_text
    :param bfwin007: the collection cursor of bfwin007 data
    :return: None
    """
    for item_ in bfwin007.find():
        live_texts_ = item_["live_texts"]
        new_live_texts = []
        for live_text_ in live_texts_:
            try:
                time_ = sum(map(int, re.findall("(\d+)'", live_text_)))
            except TypeError as e:
                print(e)
            if time_ == 0:
                continue
            try:
                text_ = live_text_.split(":")[-1]
            except IndexError as e:
                print(e)
                continue
            new_live_texts.append({"live_text": text_, "time": time_})
            print({"live_text": text_, "time": time_})
        new_live_texts.sort(key=lambda x: x["time"])
        print()
        bfwin007.update_one({"_id": item_["_id"]}, {"$set": {"live_texts": new_live_texts}})


def clean_data(data_path):
    re = []
    thuo = thulac.thulac()
    stop_chars = ''',?.!！;:"(){}[]，。？-；'：（）【】 ．—~'''
    data = json.load(open(data_path, "r", encoding="utf-8"))
    with open(data_path, "w", encoding="utf-8") as f:
        index = 0
        for item_ in data:
            text_ = item_["msg"]
            for c in stop_chars:
                text_ = text_.replace(c, ' ')
            item_["cut_text"] = thuo.fast_cut(text_)
            index += 1
            print(index)
        print("dumping")
        json.dump(data, f, ensure_ascii=False, indent=2, separators=(',', ': '))


def count_word(data_path):
    data = []
    with open(data_path, "r") as f0:
        with open("all-corpus-sep.txt", 'w', encoding="utf-8") as f1:
            for line in f0:
                line.replace(" ", "")


def prepare_sen_lab(test=True):
    # data pre-process
    data_path = './okoo-merged-3-label.json'
    data = json.load(open(data_path, encoding='utf-8'))
    sentences = []
    labels = []
    for item in data:
        sentences.append(item['text'])
        labels.append(item['merged_label'])
    # data_path = './zhibo7m.json'
    # data = json.load(open(data_path, encoding="utf-8"))
    # al = len(data)
    # count = 0
    # for item_ in data:
    #     sentences.append(item_["msg"])
    #     try:
    #         labels.append(item_["t_label"])
    #     except KeyError as e:
    #         count += 1
    #         labels.append(0)
    #         print(item_["msg"])
    #
    # print("all: {} error: {}".format(al, count))

    return sentences[:100], labels[:100]


def word2vec(data_path):
    sentences, labels = prepare_sen_lab()
    big_embedding = []
    small_embedding = []
    big_model = FastText.load_fasttext_format(os.path.join(os.path.dirname(__file__),
                                                           'word_embed/wiki.zh/wiki.zh.bin'))
    small_model = Word2Vec.load(os.path.join(os.path.dirname(__file__),
                                             "word_embed/fasttext-skim-clean-2.pt"))
    jieba.load_userdict(os.path.join(os.path.dirname(__file__),
                                     "English_Cn_Name_Corpus(48W).txt"))
    big_em_ = {}
    small_em_ = {}
    print("load")
    with open("full-cut.csv", "w", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "sentence", "label"])
        for index, sent_ in enumerate(sentences):
            tmp = sent_
            # sent_ = re.sub("[a-zA-Z]", "某某", sent_)
            sent_ = "<bos> " + sent_ + " <eos>"
            # print(sent_)
            sent_ = jieba.cut(sent_)
            for word in sent_:
                if word not in big_em_:
                    big_em_[word] = (1, big_model[word])
                else:
                    big_em_[word] = (big_em_[word][0] + 1, big_em_[word][1])
                if word in small_em_:
                    small_em_.append(small_model[word])
                    if word not in small_em_:
                        small_em_[word] = (1, small_model[word])
                    else:
                        small_em_[word] = (small_em_[word][0] + 1, small_em_[word][1])

            writer.writerow([index, tmp, labels[index]])
    print("start sort")
    print(len(big_em_))
    big_em_ = sorted(big_em_.items(), key=lambda item: item[1][0], reverse=True)
    small_em_ = sorted(small_em_.items(), key=lambda item: item[1][0], reverse=True)
    print("end sort")
    with open("big_voc.vec", "w", encoding="utf-8") as big_f:
        for item in big_em_:
            big_f.write("{} {}\n".format(item[0], item[1][1]))
    with open("small_voc.vec", "w", encoding="utf-8") as small_f:
        for item in small_em_:
            small_f.write("{} {}\n".format(item[0], item[1][1]))

if __name__ == '__main__':
    print("sdasd")
    word2vec("aa")
