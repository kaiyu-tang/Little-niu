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
from requests.exceptions import ConnectionError, ConnectTimeout,ReadTimeout

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
    'Host': "www.okooo.com",
    'Connection': "keep-alive",
    'Cache-Control': "no-cache",
    'Upgrade-Insecure-Requests': "1",
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36",
    'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    'DNT': "1",
    'Referer': "http://www.okooo.com/soccer/league/17/",
    'Accept-Encoding': "gzip, deflate",
    'Accept-Language': "zh-CN,zh;q=0.9",
    # 'Cookie': "LastUrl=; FirstURL=www.okooo.com/soccer/match/974052/form/; FirstOKURL=http%3A//www.okooo.com/soccer/match/974052/trends/; First_Source=www.okooo.com; PHPSESSID=69aab6ed4361a2a859a65c0f60d78917e92801b4; Hm_lvt_5ffc07c2ca2eda4cc1c4d8e50804c94b=1526281950,1526361535; __utmc=56961525; pm=; OKSID=1080e5b35b01afd076b5b2250de9dd720dd18ffe; _ga=GA1.2.688543559.1526281950; __utmz=56961525.1526466285.13.2.utmcsr=okooo.com|utmccn=(referral)|utmcmd=referral|utmcct=/soccer/player/12994/; __utma=56961525.688543559.1526281950.1526527880.1526536384.16; LStatus=N; LoginStr=%7B%22welcome%22%3A%22%u60A8%u597D%uFF0C%u6B22%u8FCE%u60A8%22%2C%22login%22%3A%22%u767B%u5F55%22%2C%22register%22%3A%22%u6CE8%u518C%22%2C%22TrustLoginArr%22%3A%7B%22alipay%22%3A%7B%22LoginCn%22%3A%22%u652F%u4ED8%u5B9D%22%7D%2C%22tenpay%22%3A%7B%22LoginCn%22%3A%22%u8D22%u4ED8%u901A%22%7D%2C%22qq%22%3A%7B%22LoginCn%22%3A%22QQ%u767B%u5F55%22%7D%2C%22weibo%22%3A%7B%22LoginCn%22%3A%22%u65B0%u6D6A%u5FAE%u535A%22%7D%2C%22renren%22%3A%7B%22LoginCn%22%3A%22%u4EBA%u4EBA%u7F51%22%7D%2C%22baidu%22%3A%7B%22LoginCn%22%3A%22%u767E%u5EA6%22%7D%2C%22weixin%22%3A%7B%22LoginCn%22%3A%22%u5FAE%u4FE1%u767B%u5F55%22%7D%2C%22snda%22%3A%7B%22LoginCn%22%3A%22%u76DB%u5927%u767B%u5F55%22%7D%7D%2C%22userlevel%22%3A%22%22%2C%22flog%22%3A%22hidden%22%2C%22UserInfo%22%3A%22%22%2C%22loginSession%22%3A%22___GlobalSession%22%7D; Hm_lpvt_5ffc07c2ca2eda4cc1c4d8e50804c94b=1526537392; __utmb=56961525.18.8.1526537392599",
    'If-Modified-Since': "Thu, 17 May 2018 06:09:53 GMT",
    'Postman-Token': "3f54b9c5-32fe-4925-b88c-da29fbb55861"
}

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
                   connect_times=0):
    """
    :param url:需要爬的网址
    :param header: 请求头
    :return: 列表[(比赛基本信息)，[(文本，时间，当前比分)]]
    """

    res = {}
    try:
        page = requests.get(url, headers=header, timeout=0.9).content.decode('gb2312', 'ignore')
    except (ConnectionError, ConnectTimeout, ReadTimeout):
        time.sleep(1.5)
        if connect_times < 4:
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
    timeline[-1] = '0'
    timeline[0] = timeline[1]
    match_basic = match_basic[0]
    res["Url"] = url
    res["host"] = match_basic[0]
    res["visiting"] = match_basic[1]
    res["league"] = match_basic[2]
    res["round"] = match_basic[3]
    res["vs"] = bifen[0][0] + "-" + bifen[0][1]
    time_res= {}
    time_res['year']= start_time[0]
    time_res['month'] = start_time[1]
    time_res['day'] = start_time[2]
    time_res['hour'] = start_time[3]
    time_res['minute'] = start_time[4]
    res['time'] = time_res

    jieshuo = []
    for txt, time_, bf in zip(text, timeline, bifen):
        jieshuo.append({"text": txt, "time": time_, "vs": bf})
    jieshuo.reverse()
    res["narrate"] = jieshuo
    return res


if __name__ == "__main__":

    base = 'http://www.okooo.com/soccer/match/{}/'
    all_matches = {}
    threads = []
    match_start_id = 100000
    match_end_id =   9999999
    batch_size = 1000000
    count = 0
    base_dir = "okoo-matches/{}.json"
    if not os.path.exists("okoo-matches"):
        os.mkdir("okoo-matches")

    re_match_basic = re.compile(r'<title>【(.+?)vs(.+?)\|(.+?)\s(.+?)】')
    # re_match_Bifen = re.compile(r'<div class="vs">\n\s+?<span class="vs_.+?">(\d+)</span>-<span class="vs_.+?">(\d+)</span>\n\s+?</div>')
    re_match_jieshuo = re.compile(r'<p class="float_l livelistcontext">(.+?)</p>')
    re_match_timeline = re.compile(r'<b class="float_l livelistcontime">(.+?)</b>')
    re_match_bifen = re.compile(r'<p class="float_l livelistconbifen"><b class=".*?">(\d)</b><b>-</b>'
                                r'<b class=".*?">(\d)</b></p>')
    re_match_start_time = re.compile(r'<div class="qbx_2">\s+?<p>(\d+)-(\d+)-(\d+?).*?(\d+):(\d+)</p>\s+?<p></p>\s+?'
                                     r'<p><span style="display:inline-block;margin-left:10px"></span></p>\s+?</div>')
    for loop in range((match_end_id - match_start_id) // batch_size):
        time.sleep(random.randrange(0, 10))
        match_cur_start_id = match_start_id + loop * batch_size
        for match_id in range(match_cur_start_id, min(match_cur_start_id + batch_size, match_end_id)):
            url = base.format(match_id)
            start_time = time.clock()
            thread = ThreadWithReturnValue(target=get_match_info,
                                           args=(url, headers, re_match_basic, re_match_jieshuo, re_match_timeline,
                                                 re_match_bifen,re_match_start_time))
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
                        json.dump(cur_res, f_w, ensure_ascii=False, indent=4, separators=(',', ': '))
                        f_w.flush()
                    count += 1
            except TypeError as e:
                print(match_end_id + index)

                # all_matches[index + match_start_id] = cur_res
    print(count)

    # print(len(all_matches))
