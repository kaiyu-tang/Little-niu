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

import requests

# basic config

base_url = "https://www.foxsports.com/soccer/stats"
querystring = {"competition": "10", "season": "20172", "category": "OVERALL", "pos": "0", "team": "0", "sort": "3",
               "sortOrder": "0", "page": "1", "isOpp": "0"}

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
}
cookies = ['''tvp=0.8.100%230.8.100%40100%25; fpv=1; ajs_user_id=null; ajs_group_id=null; ajs_anonymous_id=%2226fedb47-8502-4fd7-b116-da4c89702a10%22; s_vi=[CS]v1|2D7C8873852A2111-60000106200043B7[CE]; __qca=P0-1462319940-1526272231601; aam_uuid=47781745402413259370396639518824597480; _minuid=e015a2e576-be71595583-f8e38a0960-9032ae2165-3d19a45430; minAnalytics={%22clicks%22:[]}; _minEE0=[]; _minEE1=[]; __minFlavs__=-301490876; minFlavor=productionmi-1.10.2.20.js100; AMCVS_5BFD123F5245AECB0A490D45%40AdobeOrg=1; s_cc=true; ats-cid-AM-141116-sid=34529944x; janrainSSO_session=session; akacd_g1=1530064755~rv=59~id=9aefc3af582734996f734d8102f70e6b; minVersion={"experiment":-389528770,"minFlavor":"session-testgroup-2mi-1.10.2.28.js50"}; __min_duf=true; __min_du=true; tvp=0.8.100%230.8.100%40100%25; aam_dfp=aam%3DCarEnthusiast%2CNASCARFan%2CSoccerFan%2CScoreChecker%2CNASCARSuperFan%2CSoccerSuperFan; _dvp=TK:C0ObxjerU; AMCV_5BFD123F5245AECB0A490D45%40AdobeOrg=1406116232%7CMCIDTS%7C17666%7CMCMID%7C47561592802252768180418584408741321483%7CMCAAMLH-1530088567%7C11%7CMCAAMB-1530088567%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1529490967s%7CNONE%7CMCSYNCSOP%7C411-17709%7CMCAID%7C2D7C8873852A2111-60000106200043B7%7CvVersion%7C2.5.0; utag_vnum=1531984466362&vn=6; _minEE2=[]; utag_main=v_id:01635cea057a000f6a22c83208fe0407900850710093c$_sn:30$_ss:0$_st:1529490784724$vapi_domain:foxsports.com$_pn:19%3Bexp-session$ses_id:1529487914858%3Bexp-session; utag_dslv=1529488984731; s_sq=%5B%5BB%5D%5D; _parsely_session={%22sid%22:47%2C%22surl%22:%22https://www.foxsports.com/soccer/players?competition=4&season=2017%22%2C%22sref%22:%22https://www.foxsports.com/soccer/players?competition=5&teamId=0&season=2017&position=0&country=0&grouping=0&weightclass=0&playerName=%22%2C%22sts%22:1529493962745%2C%22slts%22:1529487904992}; _parsely_visitor={%22id%22:%22604733dc-5745-42dd-871b-43967efd70ef%22%2C%22session_count%22:47%2C%22last_session_ts%22:1529493962745}; akaas_fsas1=1530099065~rv=79~id=8ff7ae4e6ade7d70dd28d860106a31f2''',
           '''tvp=0.8.100%230.8.100%40100%25; fpv=1; ajs_user_id=null; ajs_group_id=null; ajs_anonymous_id=%2226fedb47-8502-4fd7-b116-da4c89702a10%22; s_vi=[CS]v1|2D7C8873852A2111-60000106200043B7[CE]; __qca=P0-1462319940-1526272231601; aam_uuid=47781745402413259370396639518824597480; _minuid=e015a2e576-be71595583-f8e38a0960-9032ae2165-3d19a45430; minAnalytics={%22clicks%22:[]}; _minEE0=[]; _minEE1=[]; __minFlavs__=-301490876; minFlavor=productionmi-1.10.2.20.js100; AMCVS_5BFD123F5245AECB0A490D45%40AdobeOrg=1; s_cc=true; ats-cid-AM-141116-sid=34529944x; janrainSSO_session=session; akacd_g1=1530064755~rv=59~id=9aefc3af582734996f734d8102f70e6b; minVersion={"experiment":-389528770,"minFlavor":"session-testgroup-2mi-1.10.2.28.js50"}; __min_duf=true; __min_du=true; tvp=0.8.100%230.8.100%40100%25; aam_dfp=aam%3DCarEnthusiast%2CNASCARFan%2CSoccerFan%2CScoreChecker%2CNASCARSuperFan%2CSoccerSuperFan; _dvp=TK:C0ObxjerU; s_sq=%5B%5BB%5D%5D; _parsely_session={%22sid%22:47%2C%22surl%22:%22https://www.foxsports.com/soccer/players?competition=4&season=2017%22%2C%22sref%22:%22https://www.foxsports.com/soccer/players?competition=5&teamId=0&season=2017&position=0&country=0&grouping=0&weightclass=0&playerName=%22%2C%22sts%22:1529493962745%2C%22slts%22:1529487904992}; _parsely_visitor={%22id%22:%22604733dc-5745-42dd-871b-43967efd70ef%22%2C%22session_count%22:47%2C%22last_session_ts%22:1529493962745}; akaas_fsas1=1530099106~rv=79~id=ea37bf4005b33ce463996a4ceb204aa1; utag_main=v_id:01635cea057a000f6a22c83208fe0407900850710093c$_sn:31$_ss:1$_st:1529496106753$vapi_domain:foxsports.com$_pn:1%3Bexp-session$ses_id:1529494306753%3Bexp-session; utag_vnum=1531984466362&vn=7; utag_invisit=true; utag_dslv=1529494306778; utag_dslv_s=Less than 1 day; _minsid=27f1289e85-5216145310-65c34fd31f-8409c16f26-653e1382e1; _minTestModeEnabled=true; minSessionAnalytics=true; __min_dut=true; _minEE2=[]; AMCV_5BFD123F5245AECB0A490D45%40AdobeOrg=1406116232%7CMCIDTS%7C17666%7CMCMID%7C47561592802252768180418584408741321483%7CMCAAMLH-1530099108%7C11%7CMCAAMB-1530099108%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1529501508s%7CNONE%7CMCSYNCSOP%7C411-17709%7CMCAID%7C2D7C8873852A2111-60000106200043B7%7CvVersion%7C2.5.0''',
           '''tvp=0.8.100%230.8.100%40100%25; fpv=1; ajs_user_id=null; ajs_group_id=null; ajs_anonymous_id=%2226fedb47-8502-4fd7-b116-da4c89702a10%22; s_vi=[CS]v1|2D7C8873852A2111-60000106200043B7[CE]; __qca=P0-1462319940-1526272231601; aam_uuid=47781745402413259370396639518824597480; _minuid=e015a2e576-be71595583-f8e38a0960-9032ae2165-3d19a45430; minAnalytics={%22clicks%22:[]}; _minEE0=[]; _minEE1=[]; __minFlavs__=-301490876; minFlavor=productionmi-1.10.2.20.js100; AMCVS_5BFD123F5245AECB0A490D45%40AdobeOrg=1; s_cc=true; ats-cid-AM-141116-sid=34529944x; janrainSSO_session=session; akacd_g1=1530064755~rv=59~id=9aefc3af582734996f734d8102f70e6b; minVersion={"experiment":-389528770,"minFlavor":"session-testgroup-2mi-1.10.2.28.js50"}; __min_duf=true; __min_du=true; tvp=0.8.100%230.8.100%40100%25; aam_dfp=aam%3DCarEnthusiast%2CNASCARFan%2CSoccerFan%2CScoreChecker%2CNASCARSuperFan%2CSoccerSuperFan; _dvp=TK:C0ObxjerU; s_sq=%5B%5BB%5D%5D; _parsely_session={%22sid%22:47%2C%22surl%22:%22https://www.foxsports.com/soccer/players?competition=4&season=2017%22%2C%22sref%22:%22https://www.foxsports.com/soccer/players?competition=5&teamId=0&season=2017&position=0&country=0&grouping=0&weightclass=0&playerName=%22%2C%22sts%22:1529493962745%2C%22slts%22:1529487904992}; _parsely_visitor={%22id%22:%22604733dc-5745-42dd-871b-43967efd70ef%22%2C%22session_count%22:47%2C%22last_session_ts%22:1529493962745}; utag_vnum=1531984466362&vn=7; utag_invisit=true; utag_dslv_s=Less than 1 day; _minsid=27f1289e85-5216145310-65c34fd31f-8409c16f26-653e1382e1; _minTestModeEnabled=true; minSessionAnalytics=true; __min_dut=true; AMCV_5BFD123F5245AECB0A490D45%40AdobeOrg=1406116232%7CMCIDTS%7C17666%7CMCMID%7C47561592802252768180418584408741321483%7CMCAAMLH-1530099108%7C11%7CMCAAMB-1530099108%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1529501508s%7CNONE%7CMCSYNCSOP%7C411-17709%7CMCAID%7C2D7C8873852A2111-60000106200043B7%7CvVersion%7C2.5.0; _minEE2=[]; akaas_fsas1=1530099211~rv=79~id=c608ca5647756ee2a6592787cb58939e; utag_main=v_id:01635cea057a000f6a22c83208fe0407900850710093c$_sn:31$_ss:0$_st:1529496211584$vapi_domain:foxsports.com$_pn:2%3Bexp-session$ses_id:1529494306753%3Bexp-session; utag_dslv=1529494411602''',
           '''tvp=0.8.100%230.8.100%40100%25; fpv=1; ajs_user_id=null; ajs_group_id=null; ajs_anonymous_id=%2226fedb47-8502-4fd7-b116-da4c89702a10%22; s_vi=[CS]v1|2D7C8873852A2111-60000106200043B7[CE]; __qca=P0-1462319940-1526272231601; aam_uuid=47781745402413259370396639518824597480; _minuid=e015a2e576-be71595583-f8e38a0960-9032ae2165-3d19a45430; minAnalytics={%22clicks%22:[]}; _minEE0=[]; _minEE1=[]; __minFlavs__=-301490876; minFlavor=productionmi-1.10.2.20.js100; AMCVS_5BFD123F5245AECB0A490D45%40AdobeOrg=1; s_cc=true; ats-cid-AM-141116-sid=34529944x; janrainSSO_session=session; akacd_g1=1530064755~rv=59~id=9aefc3af582734996f734d8102f70e6b; minVersion={"experiment":-389528770,"minFlavor":"session-testgroup-2mi-1.10.2.28.js50"}; __min_duf=true; __min_du=true; tvp=0.8.100%230.8.100%40100%25; aam_dfp=aam%3DCarEnthusiast%2CNASCARFan%2CSoccerFan%2CScoreChecker%2CNASCARSuperFan%2CSoccerSuperFan; _dvp=TK:C0ObxjerU; s_sq=%5B%5BB%5D%5D; _parsely_session={%22sid%22:47%2C%22surl%22:%22https://www.foxsports.com/soccer/players?competition=4&season=2017%22%2C%22sref%22:%22https://www.foxsports.com/soccer/players?competition=5&teamId=0&season=2017&position=0&country=0&grouping=0&weightclass=0&playerName=%22%2C%22sts%22:1529493962745%2C%22slts%22:1529487904992}; _parsely_visitor={%22id%22:%22604733dc-5745-42dd-871b-43967efd70ef%22%2C%22session_count%22:47%2C%22last_session_ts%22:1529493962745}; utag_vnum=1531984466362&vn=7; utag_invisit=true; utag_dslv_s=Less than 1 day; _minsid=27f1289e85-5216145310-65c34fd31f-8409c16f26-653e1382e1; _minTestModeEnabled=true; minSessionAnalytics=true; __min_dut=true; AMCV_5BFD123F5245AECB0A490D45%40AdobeOrg=1406116232%7CMCIDTS%7C17666%7CMCMID%7C47561592802252768180418584408741321483%7CMCAAMLH-1530099108%7C11%7CMCAAMB-1530099108%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1529501508s%7CNONE%7CMCSYNCSOP%7C411-17709%7CMCAID%7C2D7C8873852A2111-60000106200043B7%7CvVersion%7C2.5.0; akaas_fsas1=1530099257~rv=79~id=f7a94095752aa55efd8b99d71ddf9a86; utag_main=v_id:01635cea057a000f6a22c83208fe0407900850710093c$_sn:31$_ss:0$_st:1529496258601$vapi_domain:foxsports.com$_pn:3%3Bexp-session$ses_id:1529494306753%3Bexp-session; utag_dslv=1529494458617; _minEE2=[]'''
           ]
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


def get_foxsport_match(querystring, proxies, connect_times=0):
    res = {}
    header = copy.deepcopy(headers)
    header['cookie'] = cookies[random.randint(0, len(cookies)-1)]
    res['Querystring'] = querystring
    try:
        #time.sleep(random.randint(6, 22))

        page = requests.get(base_url, headers=header, params=querystring, timeout=20,
                            proxies=proxies[random.randint(0, len(proxies)-1)])
    except:  # (ConnectionError, ConnectTimeout, socket.timeout, ReadTimeout, TypeError):
        # print('connect_times: {}'.format(connect_times))
        if connect_times < random.randint(10, 15):
            print(connect_times)
            return get_foxsport_match(querystring, proxies=proxies, connect_times=connect_times + 1)
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
        for index_, td_ in enumerate(item.xpath('td')[1:]):
            tmp_ = td_.xpath('text()')[0]
            tmp_dict_[columns[index_]] = tmp_
            tmp_dict_['competition'] = querystring['competition']
            tmp_dict_['season'] = querystring['season']
        players.append(tmp_dict_)
    if len(players) != 0:
        res['players'] = players
        # print(res)
    return res


def get_sort_index(querystring, proxies, connect_time=0, ):
    sort_index_ = -1
    page_num_ = 1
    res = [sort_index_, page_num_]
    if connect_time > 8:
        return res
    header = copy.deepcopy(headers)
    header['cookie'] = cookies[random.randint(0, len(cookies) - 1)]
    try:
        #time.sleep(random.randrange(20))
        page_ = requests.get(base_url, params=querystring, headers=headers, timeout=20,
                            proxies=proxies[random.randint(0, len(proxies)-1)])
        page = etree.HTML(page_.content)
        pkg_ = page.xpath('//*[@id="wisfoxbox"]/section[2]/div[1]/table/thead/tr/th'
                          '[@title="Penalty Kick Goals"]/a/@href')
        pk_ = page.xpath('//*[@id="wisfoxbox"]/section[2]/div[1]/table/thead/tr/th'
                         '[@title="Penalty Kick"]/a/@href')
        '//*[@id="wisfoxbox"]/section[2]/div[2]/a[4]'
        page_num_ = page.xpath('//*[@id="wisfoxbox"]/section[2]/div[3]/a[last()-1]/@href')
        if len(page_num_) != 0:
            res[1] = int(re.findall('page=(\d+)', page_num_[0])[0])
            res[0] = 3
        else:
            flag = page.xpath('//*[@id="wisfoxbox"]/section[2]/div[1]/table/tbody/tr')
            if len(flag)!=0:
                res[1] = 1
                res[0] = 3
        if len(pk_) != 0:
            res[0] = int(re.findall('sort=(\d+?)', pk_[0])[0])
        elif len(pkg_) != 0:
            res[0] = int(re.findall('sort=(\d+?)', pkg_[0])[0])
        if res[0] != -1:
            return res
        else:
            res = get_sort_index(querystring, proxies=proxies, connect_time=connect_time + 1)
    except:  # (ConnectionError, ConnectTimeout, socket.timeout, ReadTimeout, ConnectionError, TypeError):
        res = get_sort_index(querystring, proxies=proxies, connect_time=connect_time + 1)
    return res


def get_competition_id(url, proxies):
    try:
        header = copy.deepcopy(headers)
        header['cookie'] = cookies[random.randint(0, len(cookies) - 1)]
        page = requests.get(url, headers=headers, timeout=20, proxies=proxies[random.randint(0, len(proxies)-1)])
        page = etree.HTML(page.content)
        competitions = page.xpath('//*[@id="wisfoxbox"]/section[1]/div/div[2]/div[1]/div/a')
        competition_ids = [re.findall('competition=(\d+)', item.xpath('@href')[0])[0] for item in competitions]
        long_names = [item.xpath('span[1]/text()')[0] for item in competitions]
        short_names = [item.xpath('span[2]/text()')[0] for item in competitions]
        # re_ = []
        # for i,j,k in zip(competition_ids[1:],long_names[1:], short_names[1:]):
        #     tmp_ = {}
        #     tmp_['name'] = j
        #     tmp_['short_name'] = k
        #     tmp_['competition_id']=i
        #     re_.append(tmp_)
        #
        # with open('competition_ids.json', 'w') as f:
        #     json.dump({'all':re_},f,ensure_ascii=False, indent=4, separators=(',', ': '))
        return [long_names[1:], short_names[1:], competition_ids[1:]]
    except Exception as e:
        print("competition_id errors")
        print(e)
        return []


def get_season(url, proxies):
    try:
        header = copy.deepcopy(headers)
        header['cookie'] = cookies[random.randint(0, len(cookies) - 1)]
        page = requests.get(url, headers=headers, timeout=20, proxies=proxies[random.randint(0, len(proxies) - 1)])
        page = etree.HTML(page.content)
        season_ids = page.xpath('//*[@id="wisbb_ddlseason"]/option')
        season_ids = [item.xpath('@value')[0] for item in season_ids]
        return season_ids
    except:
        print("get season error")
        return []

def get_categray(querystring,proxies):
    try:
        header = copy.deepcopy(headers)
        header['cookie'] = cookies[random.randint(0, len(cookies) - 1)]
        page = requests.get(base_url, params=querystring, headers=headers, timeout=20,
                            proxies=proxies[random.randint(0, len(proxies) - 1)])
        page_ = etree.HTML(page.content)
        catorgries = page_.xpath('//*[@id="wisfoxbox"]/section[1]/div/div[2]/div[3]/div/a')
        catorgries = [category_.xpath('span[1]/text()')[0] for category_ in catorgries]
        if len(catorgries)==0:
            print()
        return catorgries
    except Exception as e:
        return []
        print(e)

def get_ip(url):
    # response = requests.get(url).text.split()
    response = {"174.120.70.232:80", "210.242.179.118:80", "120.52.32.46:80", "60.206.222.157:3128", "118.123.113.4:80",
                "218.85.133.62:80", "222.168.41.246:8090", "203.135.80.25：8080", "116.62.11.138:3128",
                "121.40.131.135:3128", "116.241.162.35:3128", "221.180.170.52:8080", "163.172.129.145:8118",
                "221.180.170.25:80",
                "221.180.170.112:8080",
                "221.180.170.15:80",
                "125.118.149.224:808",
                "114.219.26.77:8998",
                "221.180.170.2:8080",
                "221.180.170.14:80",
                "221.180.170.17:8080",
                "221.180.170.9:8080"}
    #print(response)
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
    ip_url = "http://webapi.http.zhimacangku.com/getip?num=20&type=1&pro=0&city=0&yys=0&port=1&pack=22787&ts=0&ys=0&cs=0&lb=1&sb=0&pb=4&mr=1&regions="
    proxies = get_ip(ip_url)
    seasons = ['20130', '20140', '20150', '20160', '20170', ]
    category = ['DISCIPLINE', 'STANDARD', 'GOALKEEPING', 'CONTROL']
    competition_ids = ['3', '12']#['2', '1', '5', '4', '43', '3', '12']
    competition_ids = get_competition_id("https://www.foxsports.com/soccer/stats?competition=0", proxies=proxies)
    if len(competition_ids) == 3:
        competition_ids = competition_ids[2]
        print(competition_ids)
    base_dir = "foxsports-match-auto/{}.json"
    if not os.path.exists("foxsports-match-auto"):
        os.mkdir("foxsports-match-auto")

    threads = []
    threads_length = 0
    for index,competition_id in enumerate(competition_ids[:]):
        querystring['competition'] = competition_id
        querystring['category'] = 'OVERALL'
        querystring['page'] = str(1)
        querystring['sort'] = '3'
        if index%3==0:
            sleep_time = random.randrange(400, 900)
        else:
            sleep_time = random.randrange(60, 100)
        #print("after waiting for {} seconds".format(sleep_time))
        #time.sleep(sleep_time)
        seasons = get_season("https://www.foxsports.com/soccer/stats?competition={}".format(competition_id),
                             proxies=proxies)
        sleep_time = random.randrange(30, 60)
        #print("waiting for {} seconds".format(sleep_time))
        print("competition id: {} seasons: {}".format(competition_id,seasons))
        for season in seasons:
            querystring['season'] = season
            # get sort column index
            sleep_time = random.randrange(35, 60)
            #print("waiting for {} seconds".format(sleep_time))
            category = get_categray(querystring, proxies)
            print("competition id: {} season: {} category: {}".format(competition_id, season, category))
            for category_ in category:
                if category_=="OVERALL":
                    continue
                sleep_time = random.randrange(20, 40)
                #print("waiting for {} seconds".format(sleep_time))
                querystring['category'] = category_
                print("competition:{} seasion={} category={} ".format(competition_id, season, category_))


                (sort_index, page_num) = get_sort_index(querystring, proxies=proxies)
                print("competition id: {} season: {} category: {} sort_index: {}".format(competition_id, season, category_, sort_index))
                if sort_index < 0:
                    continue
                querystring['sort'] = sort_index
                print("competition:{} seasion={} category={} sort_index={} pagenum={}".format(competition_id, season,
                                                                                              category_, sort_index,
                                                                                              page_num))
                for page in range(1, page_num + 1):

                    querystring_ = copy.deepcopy(querystring)
                    querystring_["page"] = str(page)
                    # url = base.format(competition_id, season, sort_index, page)
                    start_time = time.clock()
                    thread = ThreadWithReturnValue(target=get_foxsport_match, args=(querystring_, proxies, 0))
                    threads.append(thread)
                    threads_length += 1
                    thread.start()
                    end_time = time.clock()
                print("start saving")
                for thread in threads:
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
