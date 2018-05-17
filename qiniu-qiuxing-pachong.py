#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/15 下午1:15  
# @Author  : Kaiyu  
# @Site    :   
# @File    : qiniu-qiuxing-pachong.py
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import json
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import re
from bs4 import BeautifulSoup

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


def get_match_info(url, header, re_match_jieshuo, re_match_time, re_match_bifen):
    """
    :param url:需要爬的网址
    :param header: 请求头
    :return: 列表[(文本，时间，当前比分)]
    """

    res = []
    page = requests.get(url, headers=header).content.decode('gb2312', 'ignore')
    # option = webdriver.ChromeOptions()
    # option.add_argument('headless')
    # response = requests.get(url, headers=headers, proxies=proxy)
    # browser = webdriver.Chrome(chrome_options=option)
    # browser.add_cookie(cookie)
    # browser.get(url)
    # page = browser.page_source

    match_basic = re.findall(r'<title>【(.+?)vs(.+?)\|(.+?)\s(\d+?)/(\d+?)】', page)
    text = re.findall(r'<p class="float_l livelistcontext">(.+?)</p>', page, re.S | re.M)
    timeline = re.findall(r'<b class="float_l livelistcontime">(\d+).</b>', page, re.S | re.M)
    bifen = re.findall(r'<p class="float_l livelistconbifen"><b class=".+?">(\d)</b><b>-</b>'
                       r'<b class=".+?">(\d)</b></p>', page, re.S | re.M)

    # text = re_match_jieshuo.findall(page)
    # timeline = re_match_time.findall(page)
    # bifen = re_match_bifen.findall(page)
    for txt, time_, bf in zip(text, timeline, bifen):
        res.append((txt, time_, bf))
    res.reverse()
    if len(match_basic) == 0:
        return [], res
    return match_basic[0], res


def get_player_info(url, root="http://www.okooo.com"):
    browser = webdriver.Chrome()
    browser.get(url)
    page = browser.page_source
    matches = re.findall(
        r'<td><a href="/soccer/league/\d+?/schedule/" target="_blank">(.+?)</a><a></a></td>'  # match_name
        r'\n\s+?<td>(\d+)-(\d+)\s(\d+):(\d+)</td>'  # match time
        r'\n\s+?<td align="right"><a href=".+?" target="_blank" class=".+?">(.+?)</a>\s</td>'  # match team1
        r'\n\s+?<td><b><a\shref="(.+?)"\starget=".+?">(\d+) - (\d)</a></b></td>'  # match bifen
        r'\n\s+?<td align="left"> <a href=".+?" target=".+?">(.+?)</a></td>'  # match team2
        r'\n\s+?<td>(.+?)</td>'  # if shoufa
        r'\n\s+?<td>(\d+?).</td>'  # shangchang time
        r'\n\s+?<td>(\d)</td>', page)  # jinqiushu

    # print(page)
    # print(matches)
    res = []
    # for match in matches:
    #    res.append(get_match(root+match))
    return res


def get_player_url(url):
    browser = webdriver.Chrome()
    browser.get(url)

    page = browser.page_source
    ls_selector = Select(browser.find_element_by_id("select_ls"))
    ls_ids = {option.text: option.get_attribute('value') for option in ls_selector.options}
    team_ids = {}
    player_ids = {}
    # print(ls_selector.is_multiple)
    for sl_l in ls_selector.options:
        if sl_l.get_attribute('value') == '0':
            continue
        sl_l.click()
        print(sl_l.text)
        ls_selector = Select(browser.find_element_by_id("select_ls"))
        team_selector = Select(browser.find_element_by_id("select_team"))

        # print(team_selector.options[-1].text)
        for sl_t in team_selector.options:
            if sl_t.get_attribute('value') == "0":
                continue
            team_ids[sl_t.text] = sl_t.get_attribute('value')
            print(sl_t.text + " " + sl_t.get_attribute('value'))
            sl_t.click()
            player_selector = Select(browser.find_element_by_id("select_team"))
            for sl_p in player_selector.options:
                if sl_p.get_attribute('value') == "0":
                    continue
                player_ids[sl_p.text] = sl_p.get_attribute('value')

    # players = re.findall(r'<li><a href="(.+?)" target="_blank" title="(.+?)"><p><span><img src="(.+?)"\s'
    #                    r'alt=".+?" title=".+?"></span></p><i>.+?</i></a></li>', page)


if __name__ == "__main__":

    file_result = "result.json"
    base = 'http://www.okooo.com/soccer/match/{}/'
    all_matches = {}
    re_match_jieshuo = re.compile(r'<p class="float_l livelistcontext">(.+?)</p>')
    re_match_timeline = re.compile(r'<b class="float_l livelistcontime">(\d+).</b>')
    re_match_bifen = re.compile(r'<p class="float_l livelistconbifen"><b class=".+?">(\d)</b><b>-</b>'
                                r'<b class=".+?">(\d)</b></p>')
    threads = []
    match_start_id = 900000
    match_end_id = 999999
    for match_id in range(match_start_id, match_end_id):
        url = base.format(match_id)
        start_time = time.clock()
        thread = ThreadWithReturnValue(target=get_match_info,
                                       args=(url, headers, re_match_jieshuo, re_match_timeline, re_match_bifen))
        # cur_res = get_match_info(url, headers,re_match_jieshuo,re_match_timeline,re_match_bifen)
        threads.append(thread)
        thread.start()
        end_time = time.clock()
        # print('runtime of ' + url + " : " + str(end_time - start_time))
    with open(file_result, 'w') as f_w:
        for index, thread in enumerate(threads):
            cur_res = thread.join()
            if len(cur_res[1]) > 2:
                print(cur_res)
                json.dump({str(cur_res[0]): cur_res[1]}, f_w, ensure_ascii=False, indent=4, separators=(',', ': '))
                all_matches[index + match_start_id] = cur_res

    print(len(all_matches))
