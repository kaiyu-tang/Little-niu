#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/22 上午10:41  
# @Author  : Kaiyu  
# @Site    :   
# @File    : qiniu-foxsports.com-matches.py

import json
import os
import random
import socket
import time
import requests
from threading import Thread
import re
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout
from lxml import etree, html
import copy
from bs4 import BeautifulSoup

import requests

# basic config

base_url = "https://www.foxsports.com/soccer/stats"
querystring = {"competition": "10", "season": "20172", "category": "STANDARD", "pos": "0", "team": "0", "sort": "1",
               "sortOrder": "0", "page": "1"}

headers = {
    'Host': "www.foxsports.com",
    'Connection': "keep-alive",
    'Upgrade-Insecure-Requests': "1",
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/66.0.3359.181 Safari/537.36",
    'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    'DNT': "1",
    'Accept-Encoding': "gzip, deflate, br",
    'Accept-Language': "zh-CN,zh;q=0.9",
    'Cookie': "tvp=0.8.100%230.8.100%40100%25; fpv=1; ajs_user_id=null; ajs_group_id=null; ajs_anonymous_"
              "id=%2226fedb47-8502-4fd7-b116-da4c89702a10%22; s_vi=[CS]v1|2D7C8873852A2111-60000106200043B7[CE]; __"
              "qca=P0-1462319940-1526272231601; aam_uuid=47781745402413259370396639518824597480; "
              "aam_dfp=aam%3DSoccerFan%2CScoreChecker%2CSoccerSuperFan; "
              "akacd_g1=1528690694~rv=100~id=a89867ec6686b4b3e57a790f08625bf7; __"
              "minFlavs__=1650709480; minFlavor=prodmi-1.10.2.9.js100; _"
              "minuid=e015a2e576-be71595583-f8e38a0960-9032ae2165-3d19a45430; _"
              "minTM=false; __min_duf=true; __min_du=true; minAnalytics={%22clicks%22:[]}; _minEE0=[]; _minEE1=[]; "
              "AMCVS_5BFD123F5245AECB0A490D45%40AdobeOrg=1; s_cc=true; ats-cid-AM-141116-sid=12525616x; "
              "janrainSSO_session=session; _parsely_session={"
              "%22sid%22:33%2C%22surl%22:%22https://www.foxsports.com/soccer/stats?competition=6&season=20172"
              "&category=DISCIPLINE&pos=0&team=0&sort=11&sortOrder=0%22%2C%22sref%22:%22https://www.foxsports.com"
              "/soccer/stats?competition=6&season=20170&category=DISCIPLINE&pos=0&team=0&isOpp=0&sort=5&sortOrder=0"
              "&page=0%22%2C%22sts%22:1528091753883%2C%22slts%22:1528086241354}; _parsely_visitor={"
              "%22id%22:%22604733dc-5745-42dd-871b-43967efd70ef%22%2C%22session_count%22:33%2C%22last_session_ts%22"
              ":1528091753883}; utag_vnum=1528864230785&vn=14; utag_invisit=true; utag_dslv_s=Less than 1 day; "
              "_minsid=a57b3e49da-54a1e826b3-579d55b797-8ab6f80da9-03944dfc48; minSessionAnalytics=true; "
              "AMCV_5BFD123F5245AECB0A490D45%40AdobeOrg=1406116232%7CMCIDTS%7C17666%7CMCMID"
              "%7C47561592802252768180418584408741321483%7CMCAAMLH-1528698351%7C11%7CMCAAMB-1528698351"
              "%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1528100751s%7CNONE%7CMCSYNCSOP"
              "%7C411-17694%7CMCAID%7C2D7C8873852A2111-60000106200043B7%7CvVersion%7C2.5.0; _minEE2=[]; "
              "utag_main=v_id:01635cea057a000f6a22c83208fe0407900850710093c$_sn:14$_ss:0$_st"
              ":1528095502525$vapi_domain:foxsports.com$_pn:7%3Bexp-session$ses_id:1528092683657%3Bexp-session; "
              "utag_dslv=1528093702535; akaas_fsas1=1528698698~rv=79~id=9621e03bdbc9c8887240434f1fea5fa4; "
              "s_sq=sportsfscomprod%3D%2526c.%2526a.%2526activitymap.%2526page%253Dfscom%25253Asoccer%25253Abig"
              "%252520board%25253Astats%2526link%253D2%2526region%253Dwisfoxbox%2526pageIDType%253D1%2526.activitymap"
              "%2526.a%2526.c%2526pid%253Dfscom%25253Asoccer%25253Abig%252520board%25253Astats%2526pidt%253D1%2526oid"
              "%253Dhttps%25253A%25252F%25252Fwww.foxsports.com%25252Fsoccer%25252Fstats%25253Fcompetition%25253D10"
              "%252526season%25253D20170%252526category%25253DSTANDARD%252526pos%25253D0%252526team%25253D0%252526is"
              "%2526ot%253DA",
    'Cache-Control': "no-cache",
    'Postman-Token': "fd52e9a9-2968-4270-9e31-a708feb4036c"
}
headers_sort_index = {
    'Host': "www.foxsports.com",
    'Connection': "keep-alive",
    'Cache-Control': "no-cache",
    'Upgrade-Insecure-Requests': "1",
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36",
    'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    'DNT': "1",
    'Referer': "https://www.foxsports.com/soccer/stats?competition=10&season=20170&category=STANDARD&pos=0&team=0&isOpp=0&sort=4&sortOrder=0&page=2",
    'Accept-Encoding': "gzip, deflate, br",
    'Accept-Language': "zh-CN,zh;q=0.9",
    'Cookie': "tvp=0.8.100%230.8.100%40100%25; fpv=1; ajs_user_id=null; ajs_group_id=null; ajs_anonymous_id=%2226fedb47-8502-4fd7-b116-da4c89702a10%22; s_vi=[CS]v1|2D7C8873852A2111-60000106200043B7[CE]; __qca=P0-1462319940-1526272231601; aam_uuid=47781745402413259370396639518824597480; aam_dfp=aam%3DSoccerFan%2CScoreChecker%2CSoccerSuperFan; akacd_g1=1528690694~rv=100~id=a89867ec6686b4b3e57a790f08625bf7; minFlavor=prodmi-1.10.2.9.js100; _minuid=e015a2e576-be71595583-f8e38a0960-9032ae2165-3d19a45430; _minTM=false; __min_duf=true; __min_du=true; minAnalytics={%22clicks%22:[]}; _minEE0=[]; _minEE1=[]; AMCVS_5BFD123F5245AECB0A490D45%40AdobeOrg=1; s_cc=true; ats-cid-AM-141116-sid=12525616x; janrainSSO_session=session; _dvp=TK:C0ObxjerU; _parsely_session={%22sid%22:34%2C%22surl%22:%22https://www.foxsports.com/soccer/stats?competition=1&season=20172&category=DISCIPLINE&pos=0&team=0&isOpp=0&sort=1&sortOrder=11&page=0%22%2C%22sref%22:%22%22%2C%22sts%22:1528109449855%2C%22slts%22:1528091753883}; _parsely_visitor={%22id%22:%22604733dc-5745-42dd-871b-43967efd70ef%22%2C%22session_count%22:34%2C%22last_session_ts%22:1528109449855}; utag_vnum=1528864230785&vn=18; utag_invisit=true; utag_dslv_s=Less than 1 day; _minsid=0daf08e0c9-901cd06b8a-8ad7a9c74c-331ad4bb2d-93bb558f21; minSessionAnalytics=true; AMCV_5BFD123F5245AECB0A490D45%40AdobeOrg=1406116232%7CMCIDTS%7C17666%7CMCMID%7C47561592802252768180418584408741321483%7CMCAAMLH-1528717361%7C11%7CMCAAMB-1528717361%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1528119761s%7CNONE%7CMCSYNCSOP%7C411-17694%7CMCAID%7C2D7C8873852A2111-60000106200043B7%7CvVersion%7C2.5.0; __minFlavs__=653637127; s_sq=%5B%5BB%5D%5D; _minEE2=[]; utag_main=v_id:01635cea057a000f6a22c83208fe0407900850710093c$_sn:18$_ss:0$_st:1528116638155$vapi_domain:foxsports.com$_pn:4%3Bexp-session$ses_id:1528112558757%3Bexp-session; utag_dslv=1528114838171; akaas_fsas1=1528719638~rv=79~id=4f125b71a9a8aa46474572647e365cb6",
    'Postman-Token': "3fdd2fe3-f934-44b3-b873-70cb4ab6f007"
}


# response = requests.request("GET", url, headers=headers, params=querystring)


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


def get_foxsport_match(querystring, proxies, connect_times=0, ):
    res = {}
    res['Querystring'] = querystring
    try:
        page = requests.get(base_url, headers=headers, params=querystring, timeout=5, proxies=proxies[random.randint(0, len(proxies))])
    except:  # (ConnectionError, ConnectTimeout, socket.timeout, ReadTimeout, TypeError):
        # print('connect_times: {}'.format(connect_times))
        time.sleep(random.randrange(0, 1))
        if connect_times < random.randint(1, 5):
            return get_foxsport_match(querystring, connect_times=connect_times + 1, proxies=proxies)
        else:
            return res
    html = etree.HTML(page.content)
    columns = []
    catergrays = html.xpath('//*[@id="wisfoxbox"]/section[2]/div[1]/table/thead/tr/th')
    for cater in catergrays[1:]:
        columns.append(cater.xpath('@title')[0])
    items = html.xpath('//*[@id="wisfoxbox"]/section[2]/div[1]/table/tbody/tr')
    players = []
    for item in items:
        full_name = item.xpath('td[1]/div/a/span[1]/text()')[0]
        name = item.xpath('td[1]/div/a/span[2]/text()')[0]
        tmp_dict_ = {'full-name': full_name, 'name': name}
        tmp_dict_['competition'], tmp_dict_['season'] = querystring['competition'], querystring['season']
        for index_, td_ in enumerate(item.xpath('td')[1:]):
            tmp_ = td_.xpath('text()')[0]
            tmp_dict_[columns[index_]] = tmp_
        players.append(tmp_dict_)
    if len(players) != 0:
        res['players'] = players
        # print(res)
    return res


def get_sort_index(querystring, proxies, connect_time=0, ):
    sort_index_ = -1
    page_num_ = 1
    res = [sort_index_, page_num_]
    if connect_time > 3:
        return res
    try:
        time.sleep(random.randrange(4))

        page = requests.get(base_url, params=querystring, headers=headers, timeout=4, proxies=proxies[random.randint(0, len(proxies))])
        page = etree.HTML(page.content)
        pkg_ = page.xpath('//*[@id="wisfoxbox"]/section[2]/div[1]/table/thead/tr/th'
                          '[@title="Penalty Kick Goals"]/a/@href')
        pk_ = page.xpath('//*[@id="wisfoxbox"]/section[2]/div[1]/table/thead/tr/th'
                         '[@title="Penalty Kick"]/a/@href')
        page_num_ = page.xpath('//*[@id="wisfoxbox"]/section[2]/div[3]/a[last()-1]/@href')
        if len(page_num_) != 0:
            res[1] = int(re.findall('page=(\d+)', page_num_[0])[0])

        if len(pk_) != 0:
            res[0] = int(re.findall('sort=(\d+?)', pk_[0])[0])
        elif len(pkg_) != 0:
            res[0] = int(re.findall('sort=(\d+?)', pkg_[0])[0])
        else:
            res = get_sort_index(querystring, connect_time=connect_time + 1, proxies=proxies)
    except:  # (ConnectionError, ConnectTimeout, socket.timeout, ReadTimeout, ConnectionError, TypeError):
        res = get_sort_index(querystring, connect_time=connect_time + 1, proxies=proxies)
    return res


def get_ip(url):
    # response = requests.get(url).text.split()
    response = {"174.120.70.232:80", "210.242.179.118:80", "120.52.32.46:80", "60.206.222.157:3128", "118.123.113.4:80",
                "218.85.133.62:80", "222.168.41.246:8090", "203.135.80.25：8080", "116.62.11.138:3128",
                "121.40.131.135:3128",  "116.241.162.35:3128"}
    # print(response)
    res = []
    for item in response:
        tmp = {}
        print(item)
        tmp['http'] = "http://" + item
        res.append(tmp)
    # print()
    return res


if __name__ == "__main__":
    # base config
    seasons = ['20140', '20141', '20142', '20150', '20151', '20152', '20160', '20161', '20162', '20170', '20171',
               '20172']  # ,
    category = ['DISCIPLINE', 'STANDARD', 'GOALKEEPING', 'CONTROL']
    competition_start_id = 0
    competition_end_id = 1000
    step = 0
    ip_url = "http://webapi.http.zhimacangku.com/getip?num=20&type=1&pro=0&city=0&yys=0&port=1&pack=22787&ts=0&ys=0&cs=0&lb=1&sb=0&pb=4&mr=1&regions="
    proxies = get_ip(ip_url)
    # batch_size = 100000

    base_dir = "foxsports-match/{}.json"
    if not os.path.exists("foxsports-match"):
        os.mkdir("foxsports-match")

    threads = []
    threads_length = 0
    for season in seasons:
        querystring['season'] = season
        connect_time = 0
        step = 0
        for competition_id in range(competition_start_id, competition_end_id):
            time.sleep(random.randint(2, 10))
            querystring['competition'] = str(competition_id)
            if step > 1000:
                break
            # get sort column index
            for category_ in category:
                querystring['category'] = category_
                print("seasion={} competition:{} category={} ".format(season, competition_id, category_))
                (sort_index, page_num) = get_sort_index(querystring, proxies)
                if sort_index < 0:
                    step += 1
                    continue
                print("seasion={} competition:{} category={} sort_index={} pagenum={}".format(season, competition_id,
                                                                                              category_, sort_index,
                                                                                              page_num))
                step = 0
                for page in range(1, page_num + 1):
                    querystring['sort'] = sort_index
                    querystring["page"] = str(page)
                    querystring_ = copy.deepcopy(querystring)
                    # url = base.format(competition_id, season, sort_index, page)
                    start_time = time.clock()
                    thread = ThreadWithReturnValue(target=get_foxsport_match,
                                                   args=(querystring_, proxies))
                    threads.append(thread)
                    threads_length += 1
                    thread.start()
                    end_time = time.clock()
                print("start saving")
                for thread in threads:
                    time.sleep(random.randint(2, 6))
                    cur_res = thread.join()
                    # print()
                    try:
                        if len(cur_res) > 1:
                            # print(cur_res)
                            querystring_ = cur_res['Querystring']
                            print(querystring_)
                            with open(base_dir.format(
                                    querystring_['season'] + "-" + querystring_['competition'] + "-" +
                                    querystring_['category'] + "-" + querystring_['page']), 'w') as f_w:
                                json.dump(cur_res, f_w, ensure_ascii=False, indent=4, separators=(',', ': '))
                    except TypeError as e:
                        print(e)
                threads = []
                print('finish saving')

                # print("Runtime: {} {}".format(end_time - start_time, querystring))
