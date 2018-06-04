#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/15 下午1:15  
# @Author  : Kaiyu  
# @Site    :   
# @File    : qiniu-okooo.com-match.py
import random
import time
import requests
import json
import os
import re
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout

from threading import Thread


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return



headers = {
    'accept': "*/*",
    'accept-encoding': "gzip, deflate, br",
    'accept-language': "zh-CN,zh;q=0.9",
    'if-modified-since': "Thu, 17 May 2018 02:18:06 GMT",
    'origin': "http://cs.betradar.com",
    'referer': "http://cs.betradar.com/ls/widgets/?/okooo/zh/Asia:Shanghai/page/widgets_lmts",
    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36",

}

proxy = {'http': '125.46.0.62:53281'}


def get_match_info(url, header, re_match_basic, re_match_jieshuo, re_match_time, re_match_bifen, re_match_start_time,
                   re_match_label, connect_times=0):
    """
    :param url:需要爬的网址
    :param header: 请求头
    :return: 列表[(比赛基本信息)，[(文本，时间，当前比分)]]
    """

    res = {}
    try:
        page = requests.get(url, headers=header, timeout=2).content.decode('gb2312', 'ignore')
    except (ConnectionError, ConnectTimeout, ReadTimeout):
        time.sleep(random.randrange(0, 1.2))
        if connect_times < random.randint(1, 3):
            return get_match_info(url, header, re_match_basic, re_match_jieshuo, re_match_time, re_match_bifen,
                                  re_match_start_time, connect_times + 1)
        else:
            return res
    # option = webdriver.ChromeOptions()
    # option.add_argument('headless')
    # response = requests.get(url, headers=headers, proxies=proxy)
    # browser = webdriver.Chrome(chrome_options=option)
    # browser.add_cookie(cookie)
    # browser.get(url)
    # page = browser.page_source
    # match_basic = re.findall(r'<title>【(.+?)vs(.+?)\|(.+?)\s(\d+?)/(\d+?)】', page)
    # text = re.findall(r'<p class="float_l livelistcontext">(.+?)</p>', page, re.S | re.M)
    # timeline = re.findall(r'<b class="float_l livelistcontime">(\d+).</b>', page, re.S | re.M)
    # bifen = re.findall(r'<p class="float_l livelistconbifen"><b class=".+?">(\d)</b><b>-</b>'
    #                  r'<b class=".+?">(\d)</b></p>', page, re.S | re.M)
    text = re_match_jieshuo.findall(page)
    bifen = re_match_bifen.findall(page)
    if len(text) == 0 or len(bifen) == 0:
        return res
    match_basic = re_match_basic.findall(page)
    # Bifen = re_match_Bifen.findall(page)
    timeline = re_match_time.findall(page)
    start_time = re_match_start_time.findall(page)[0]
    labels = re_match_label.findall(page)
    timeline[-1] = "0'"
    timeline[0] = timeline[1]
    match_basic = match_basic[0]
    res["Url"] = url
    res["host"] = match_basic[0]
    res["visiting"] = match_basic[1]
    res["league"] = match_basic[2]
    res["round"] = match_basic[3]
    res["vs"] = bifen[0][0] + "-" + bifen[0][1]
    time_res = {}
    time_res['year'] = start_time[0]
    time_res['month'] = start_time[1]
    time_res['day'] = start_time[2]
    time_res['hour'] = start_time[3]
    time_res['minute'] = start_time[4]
    res['time'] = time_res

    jieshuo = []
    for txt, time_, bf, label in zip(text, timeline, bifen, labels):
        jieshuo.append({"text": txt, "time": time_[:-1], "vs": bf, 'label': label})
    jieshuo.reverse()
    res["narrate"] = jieshuo
    return res


if __name__ == "__main__":

    base = 'http://www.okooo.com/soccer/match/{}/'
    all_matches = {}
    threads = []
    match_start_id = 100000
    match_end_id = 9999999
    batch_size = 10
    count = 0
    base_dir = "okoo-match/{}.json"
    if not os.path.exists("okoo-match"):
        os.mkdir("okoo-match")

    re_match_basic = re.compile(r'<title>【(.+?)vs(.+?)\|(.+?)\s(.+?)】')
    # re_match_Bifen = re.compile(r'<div class="vs">\n\s+?<span class="vs_.+?">(\d+)</span>-<span class="vs_.+?">(\d+)</span>\n\s+?</div>')
    re_match_jieshuo = re.compile(r'<p class="float_l livelistcontext">(.+?)</p>')
    re_match_timeline = re.compile(r'<b class="float_l livelistcontime">(.+?)</b>')
    re_match_bifen = re.compile(r'<p class="float_l livelistconbifen"><b class=".*?">(\d)</b><b>-</b>'
                                r'<b class=".*?">(\d)</b></p>')
    re_match_start_time = re.compile(r'<div class="qbx_2">\s+?<p>(\d+)-(\d+)-(\d+?).*?(\d+):(\d+)</p>\s+?<p></p>\s+?'
                                     r'<p><span style="display:inline-block;margin-left:10px"></span></p>\s+?</div>')
    re_match_label = re.compile(r'<div class="livelistcon">\s+<span class="phrase_type_(\d+)"></span>\s+'
                                r'<p class="float_l livelistcontext">')
    res_f = open(base_dir.format('result-new'), 'w')
    res_f.write('start id: {} \n end_id: {} \nbatch_size: {} \n'.format(match_start_id, match_end_id, batch_size))
    for loop in range((match_end_id - match_start_id) // batch_size):
        time.sleep(random.randrange(0, 10))
        match_cur_start_id = match_start_id + loop * batch_size
        for match_id in range(match_cur_start_id, min(match_cur_start_id + batch_size, match_end_id)):
            url = base.format(match_id)
            start_time = time.clock()
            thread = ThreadWithReturnValue(target=get_match_info,
                                           args=(url, headers, re_match_basic, re_match_jieshuo, re_match_timeline,
                                                 re_match_bifen, re_match_start_time, re_match_label))
            # cur_res = get_match_info(url, headers,re_match_jieshuo,re_match_timeline,re_match_bifen)
            threads.append(thread)
            thread.start()
            end_time = time.clock()

            print("id: {} runtime: {}".format(match_id, end_time - start_time))
        for index, thread in enumerate(threads):
            cur_res = thread.join()
            try:
                if len(cur_res) != 0:
                    # print(cur_res)
                    with open(base_dir.format(cur_res["Url"].split("/")[-2]), 'w') as f_w:
                        print(cur_res["Url"])
                        res_f.write('Url: '+cur_res['Url']+"\n")
                        json.dump(cur_res, f_w, ensure_ascii=False, indent=4, separators=(',', ': '))
                        f_w.flush()
                    count += 1
            except TypeError as e:
                print(match_end_id + index)

                # all_matches[index + match_start_id] = cur_res
    print(count)
    res_f.write('total download url : {}'.format(count))
    res_f.flush()
    res_f.close()
    # print(len(all_matches))


