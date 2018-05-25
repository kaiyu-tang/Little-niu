#!/usr/bin/env python  
# -*- coding: utf-8 -*-
# @Time    : 2018/5/17 下午6:06  
# @Author  : Kaiyu  
# @Site    :   
# @File    : qiniu-whoscored.com-player.py.py
import json
import os
import random
import time

import requests
from threading import Thread
import re
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout
from lxml import etree, html
from bs4 import BeautifulSoup


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


url = "https://www.whoscored.com/StatisticsFeed/1/GetPlayerStatistics"

querystring = {"category": "summary", "subcategory": "{}", "statsAccumulationType": "0", "isCurrent": "true",
               "playerId": "{}", "teamIds": "", "matchId": "", "stageId": "", "tournamentOptions": "",
               "sortBy": "Rating", "sortAscending": "", "age": "", "ageComparisonType": "", "appearances": "",
               "appearancesComparisonType": "", "field": "Overall", "nationality": "", "positionOptions": "",
               "timeOfTheGameEnd": "", "timeOfTheGameStart": "", "isMinApp": "false", "page": "",
               "includeZeroValues": "true", "numberOfPlayersToPick": ""}  # playerId and sybcategray is useful there

headers = headers = {
    'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    'Accept-Encoding': "gzip, deflate",
    'Accept-Language': "zh-CN,zh;q=0.9",
    'Cache-Control': "no-cache",
    'Connection': "keep-alive",
    # 'Cookie': "announceId=20180103001; JSESSIONID=831C24BC2C1F2A3A6B31E37B63B65823; Hm_lvt_b83b828716a7230e966a4555be5f6151=1526278774,1526374337,1526552611; Hm_lpvt_b83b828716a7230e966a4555be5f6151=1526552621",
    'DNT': "1",
    'Host': "www.tzuqiu.cc",
    'Referer': "http://www.tzuqiu.cc/stats.do",
    'Upgrade-Insecure-Requests': "1",
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36",
    # 'Postman-Token': "82796269-b881-47bb-a673-963c70be9c33"
}

proxies = {
    'http': 'http://115.223.255.75:9000'
}

def get_match_summary(url, connect_times=0):
    """
    :param url: the url requests
    :param connect_times: if connect_times larger than 2 than do not try
    :return: dic{'player name': , 'url':, 'matches':[]}
    """
    base_url = url.split('/')[2]
    res = {}
    summary_res = []
    offensive_res = []
    defensive_res = []
    pass_res = []
    try:
        response = requests.get(url, headers=headers, timeout=0.6,proxies=proxies)  # .decode('gb2312', 'ignore')
    except (ConnectionError, ConnectTimeout, ReadTimeout):
        time.sleep(0.8)
        if connect_times < 4:
            return get_match_summary(url, connect_times + 1)
        else:
            return res
    soup = BeautifulSoup(response.content, 'lxml')
    summary = soup.find_all(id='summaryTable')
    offensive = soup.find_all(id='offensiveTable')
    defensive = soup.find_all(id='defensiveTable')
    pas = soup.find_all(id='passTable')
    if len(summary) == 0 or len(offensive) == 0 or len(defensive) == 0 or len(pas) == 0:
        return res
    summary = summary[0].tbody.find_all('tr')[:-1]

    for tr in summary:
        summary_res_ = {}
        td_list = tr.find_all('td')
        summary_res_['url'] = base_url + td_list[0].a['href']
        summary_res_['Tournament'] = td_list[0].a.get_text().strip()
        summary_res_['Apps'] = ''.join(td_list[1].get_text().strip().split())
        summary_res_['Mins'] = td_list[2].get_text().strip()
        summary_res_['Goals'] = td_list[3].get_text().strip()
        summary_res_['Assists'] = td_list[4].get_text().strip()
        YR = td_list[5].find_all('span')
        if len(YR) == 2:
            summary_res_['Yellow'] = YR[0].get_text().strip()
            summary_res_['Red'] = YR[1].get_text().strip()
        else:
            summary_res_['Yellow'] = ''
            summary_res_['Red'] = ''
        summary_res_['PS%'] = td_list[6].get_text().strip()
        summary_res_['Create opportunity'] = td_list[7].get_text().strip()
        summary_res_['zhengding success'] = td_list[8].get_text().strip()
        summary_res_['best'] = td_list[9].get_text().strip()
        summary_res_['summary scores'] = td_list[10].get_text().strip()
        summary_res.append(summary_res_)
    # print(str(len(summary_res))+" : "+str(summary_res))
    offensive = offensive[0].tbody.find_all('tr')[:-1]
    for tr in offensive:
        offensive_res_ = {}
        td_list = tr.find_all('td')
        offensive_res_['shoot'] = td_list[4].get_text().strip()
        offensive_res_['shoot on target'] = td_list[5].get_text().strip()
        offensive_res_['crossover'] = td_list[6].get_text().strip()
        offensive_res_['offended'] = td_list[7].get_text().strip()
        offensive_res_['offside'] = td_list[8].get_text().strip()
        offensive_res_['stealed'] = td_list[9].get_text().strip()
        offensive_res_['faulty'] = td_list[10].get_text().strip()
        offensive_res_['offensive score'] = td_list[11].get_text().strip()
        offensive_res.append(offensive_res_)
    # print(str(len(offensive_res)) + " : " + str(offensive_res))
    defensive = defensive[0].tbody.find_all('tr')[:-1]
    for tr in defensive:
        defensive_res_ = {}
        td_list = tr.find_all('td')
        defensive_res_['steal'] = td_list[3].get_text().strip()
        defensive_res_['intercept'] = td_list[4].get_text().strip()
        defensive_res_['clearance kick'] = td_list[5].get_text().strip()
        defensive_res_['block off'] = td_list[6].get_text().strip()
        defensive_res_['offside trap'] = td_list[7].get_text().strip()
        defensive_res_['foul'] = td_list[8].get_text().strip()
        defensive_res_['beiguo'] = td_list[9].get_text().strip()
        defensive_res_['critical miss'] = td_list[10].get_text().strip()
        defensive_res_['defensive score'] = td_list[11].get_text().strip()
        defensive_res.append(defensive_res_)
    # print(str(len(defensive_res)) + " : " + str(defensive_res))
    pas = pas[0].tbody.find_all('tr')[:-1]
    for tr in pas:
        pass_res_ = {}
        td_list = tr.find_all('td')
        pass_res_['key pass'] = td_list[4].get_text().strip()
        pass_res_['pass'] = td_list[5].get_text().strip()
        pass_res_['PS%'] = td_list[6].get_text().strip()
        pass_res_['FTPS%'] = td_list[7].get_text().strip()
        pass_res_['crosses'] = td_list[8].get_text().strip()
        pass_res_['long pass'] = td_list[9].get_text().strip()
        pass_res_['zhisai'] = td_list[10].get_text().strip()
        pass_res_['pass score'] = td_list[11].get_text().strip()
        pass_res.append(pass_res_)
    # print(str(len(pass_res)) + " : " + str(pass_res))
    res_ = []
    for i in range(len(summary_res)):
        tmp_dict = summary_res[i].copy()
        tmp_dict.update(offensive_res[i])
        tmp_dict.update(defensive_res[i])
        tmp_dict.update(pass_res[i])
        res_.append(tmp_dict)
    res['player ch name'] = soup.head.title.get_text().split('|')[0]
    res['player en name'] = re.findall(r'<tr>\n\s+?<th>英文名: </th>\n\s+?<td>(.*?)</td>\n\s+?</tr>',response.text)
    res['url'] = url
    res['matches'] = res_
    # print(res)
    return res


if __name__ == "__main__":

    base = 'http://www.tzuqiu.cc/players/{}/show.do'
    all_matches = {}
    threads = []
    player_start_id = 1
    player_end_id = 50000
    batch_size = 500
    count = 0
    base_dir = "whoscored-matches/{}.json"
    if not os.path.exists("whoscored-matches"):
        os.mkdir("whoscored-matches")

    for loop in range((player_end_id - player_start_id) // batch_size):
        time.sleep(random.randrange(0, 10))
        player_cur_start_id = player_start_id + loop * batch_size
        for player_id in range(player_cur_start_id, min(player_cur_start_id + batch_size, player_end_id)):
            url = base.format(player_id)
            start_time = time.clock()
            thread = ThreadWithReturnValue(target=get_match_summary,
                                           args=(url, 0))
            threads.append(thread)
            thread.start()
            end_time = time.clock()

            print("id: {} runtime: {}".format(player_id, end_time - start_time))
        for index, thread in enumerate(threads):
            cur_res = thread.join()
            try:
                if len(cur_res) != 0:
                    # print(cur_res)
                    with open(base_dir.format(cur_res["url"].split("/")[-2]), 'w') as f_w:
                        print(cur_res["url"])
                        json.dump(cur_res, f_w, ensure_ascii=False, indent=4, separators=(',', ': '))
                    count += 1
            except TypeError as e:
                print(player_end_id + index)

    print(count)
