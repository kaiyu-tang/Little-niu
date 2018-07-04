#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/29 下午12:29  
# @Author  : Kaiyu  
# @Site    :   
# @File    : data-process.py
import json
import os
import re
from gensim.corpora import WikiCorpus
import thulac


def okoo_merge_label(file_name):
    labels_dic = {}
    with open("label_doc.text",encoding='utf-8') as f:
        for index, line in enumerate(f):
            for key in re.findall('(\d+)', line):
                labels_dic[''.join(key)] = index
    cur_true_label = index + 1
    with open(file_name,encoding='utf-8') as f:
        data = json.load(f)["narrate"]
        for item in data:
            label = item['label']
            if label in labels_dic:
                item['merged_label'] = labels_dic[label]
            else:
                print(item)
                print(cur_true_label)
                item['merged_label'] = cur_true_label
                cur_true_label += 1
    with open('test000-merged-label.json', 'w',encoding='utf-8') as f:
        json.dump({'all': data}, f, ensure_ascii=False, indent=4, separators=(',', ': '))


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


if __name__ == '__main__':
    # thu1 = thulac.thulac(user_dict='English_Cn_Name_Corpus(48W).txt', deli='_', filt=False)
    # thu1.fast_cut_f('all-corpus.txt', 'all-corpus-seg-pure.txt')
    file_name = 'test000.json'
    okoo_merge_label(file_name)
