#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/29 下午12:29  
# @Author  : Kaiyu  
# @Site    :   
# @File    : data-process.py
"""
this file is used to process data
"""
import json
import os
import re
import sys
from functools import reduce

from pymongo import MongoClient
import thulac
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def okoo_merge_label(file_name):
    """
    merge the labels of the data of okoo.com to 7
    :param file_name:
    :return:
    """
    labels_dic = {}
    with open("label_doc_3", encoding='utf-8') as f:
        for line in f:
            if len(line) < 2:
                continue
            label = int(line.split(" ")[0])
            line = line[2:]
            for key in re.findall('(\d+)', line):
                labels_dic[''.join(key)] = label
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


def merge_7m_label(zhibo7m):
    labels_dict = {}
    label_file = "name.txt"
    with open(label_file, encoding="utf-8") as f_label:
        for line in f_label:
            if len(line) < 3:
                continue
            label = int(line.split()[0])
            for t_label_ in re.findall('''"(.+?)"''', line):
                labels_dict[t_label_] = label
    for item_ in zhibo7m.find():
        try:
            events_ = item_["content"]["event"]
            textFeeds_ = item_["textFeed"]
        except KeyError as e:
            zhibo7m.delete_one({"_id": item_["_id"]})
            print(e)
            print(item_["_id"])
            continue
        index_e_, index_t_ = 0, 0
        length_e_, length_t_ = len(events_), len(textFeeds_)
        while index_e_ < length_e_ and index_t_ < length_t_:
            time_e_, time_t_ = events_[index_e_]["time"], textFeeds_[index_t_]["time"]
            time_cha_ = time_e_ - time_t_
            name_e_ = events_[index_e_]["name"]
            label_e_ = labels_dict[name_e_]
            if abs(time_cha_) < 5:
                textFeeds_[index_t_]["name"] = name_e_
                textFeeds_[index_t_]["t_label"] = label_e_
                index_e_ += 1
                index_t_ += 1
            elif time_cha_ < 0:
                index_e_ += 1
            else:
                textFeeds_[index_t_]["name"] = ""
                textFeeds_[index_t_]["t_label"] = 0
                index_t_ += 1
        while index_t_ < length_t_:
            textFeeds_[index_t_]["t_label"] = 0
            textFeeds_[index_t_]["name"] = ""
            index_t_ += 1
            print(index_t_)

        zhibo7m.update_one({"_id": item_["_id"]}, {"$set": {"content.textFeed": textFeeds_}})


def extrace_zhibo7m(zhibo7m):
    with open("zhibo7m.json", "w", encoding="utf-8") as f:
        items = []
        for item_ in zhibo7m.find():
            text_feed_ = item_["content"]["textFeed"]
            items.extend(text_feed_)

        json.dump(items, f, ensure_ascii=False, indent=2, separators=(",", ": "))


if __name__ == '__main__':
    # thu0 = thulac.thulac()
    # t = thu0.fast_cut("   完赛  客观来说 切尔西今天比赛的场面并不是很好看 但拿到三分比什么都重要 另外 让穆帅感到欣慰的是核心阿扎尔状态的回升 比利时人今天的突破非常犀利")
    client = MongoClient()
    db = client["live_texts"]
    # # bfwin007 = db["bfwin007"]
    # # process_bfwin007(bfwin007)
    zhibo7m = db["zhibo7m"]
    merge_7m_label(zhibo7m)
    print("start merge")
    extrace_zhibo7m(zhibo7m)
    # okoo_merge_label("/Users/harry/PycharmProjects/toys/cut-video-with-text/Text_classification/text_classify/data/okoo-label.json")
    sys.exit(0)
    # data pre-process
    data_path = 'okoo-merged-clean-cut-data.json'
    data = json.load(open(data_path, encoding='utf-8'))['all']
    sentences = []
    labels = []
    for item in data:
        sentences.append(item['text'])
        labels.append(item['merged_label'])
    data_len = len(data)
    # train word2vec
    # text_path = 'data' + os.sep + 'all-corpus-seg-pure.txt'
    # train_word_vectors(text_path, Config)
    # clean data
    with open('okoo-merged-clean-cut-data.txt', 'w', encoding='utf-8') as f:
        for item_ in sentences:
            if len(item_) > 1:
                f.write('{}\n'.format(item_))
    sys.exit(0)
    # chose top k
    chose_topk(sentences, labels, 5)
