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
from pymongo import MongoClient
import thulac
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# import opencc

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
    thuo = thulac.thulac(T2S=True)
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
        json.dump(data, f, ensure_ascii=False)


if __name__ == '__main__':
    # f = open(
    #     ".././data/zhibo7m.json",
    #     "r",)
    # data = f.read()
    #
    # data = json.load(str(data))
    # f.close()
    # print("f close")
    # with open(
    #         ".././data/okoo-merged-3-label.json",
    #         "w", encoding="utf-8") as f1:
    #     json.dump(data, f1, ensure_ascii=False)
    # sys.exit()
    clean_data("./zhibo7m.json")
    sys.exit()
    # client = MongoClient()
    # db = client["live_texts"]
    # bfwin007 = db["bfwin007"]
    # with open(".././data/zhibo7m.json", "w", encoding="utf-8") as f:
    #     data = []
    #     for item_ in db["zhibo7m"].find():
    #         tmps = item_["content"]["textFeed"]
    #         new = []
    #         for tmp in tmps:
    #             tmp["msg"] = opencc.convert(tmp["msg"])
    #             new.append(tmp)
    #         data.append(new)
    #     json.dump(data, f, ensure_ascii=False)
    # process_bfwin007(bfwin007)
    #  okoo_merge_label("/Users/harry/PycharmProjects/toys/cut-video-with-text/Text_classification/text_classify/data/okoo-label.json")
    # sys.exit(0)
    # data pre-process
    data_path = './okoo-merged-3-label.json'
    data = json.load(open(data_path, encoding='utf-8'))
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
