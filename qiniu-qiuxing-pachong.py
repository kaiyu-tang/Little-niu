#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/15 下午1:15  
# @Author  : Kaiyu  
# @Site    :   
# @File    : qiniu-qiuxing-pachong.py

import requests
import json
from selenium import webdriver
import re

format_url = 'https://wlc.fn.sportradar.com/okooo/zh/Europe:Berlin/gismo/match_detailsextended/12055760'
headers = {
    "Host": "www.okooo.com",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "DNT": "1",
    "Referer": "http://www.okooo.com/soccer/player/12994/",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cookie": "LastUrl=; __utmz=56961525.1526281950.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); FirstURL=www.okooo.com/soccer/match/974052/form/; FirstOKURL=http%3A//www.okooo.com/soccer/match/974052/trends/; First_Source=www.okooo.com; PHPSESSID=69aab6ed4361a2a859a65c0f60d78917e92801b4; Hm_lvt_5ffc07c2ca2eda4cc1c4d8e50804c94b=1526281950,1526361535; __utmc=56961525; pm=; LStatus=N; LoginStr=%7B%22welcome%22%3A%22%u60A8%u597D%uFF0C%u6B22%u8FCE%u60A8%22%2C%22login%22%3A%22%u767B%u5F55%22%2C%22register%22%3A%22%u6CE8%u518C%22%2C%22TrustLoginArr%22%3A%7B%22alipay%22%3A%7B%22LoginCn%22%3A%22%u652F%u4ED8%u5B9D%22%7D%2C%22tenpay%22%3A%7B%22LoginCn%22%3A%22%u8D22%u4ED8%u901A%22%7D%2C%22qq%22%3A%7B%22LoginCn%22%3A%22QQ%u767B%u5F55%22%7D%2C%22weibo%22%3A%7B%22LoginCn%22%3A%22%u65B0%u6D6A%u5FAE%u535A%22%7D%2C%22renren%22%3A%7B%22LoginCn%22%3A%22%u4EBA%u4EBA%u7F51%22%7D%2C%22baidu%22%3A%7B%22LoginCn%22%3A%22%u767E%u5EA6%22%7D%2C%22weixin%22%3A%7B%22LoginCn%22%3A%22%u5FAE%u4FE1%u767B%u5F55%22%7D%2C%22snda%22%3A%7B%22LoginCn%22%3A%22%u76DB%u5927%u767B%u5F55%22%7D%7D%2C%22userlevel%22%3A%22%22%2C%22flog%22%3A%22hidden%22%2C%22UserInfo%22%3A%22%22%2C%22loginSession%22%3A%22___GlobalSession%22%7D; __utma=56961525.688543559.1526281950.1526361535.1526372145.5; OKSID=1080e5b35b01afd076b5b2250de9dd720dd18ffe; _ga=GA1.2.688543559.1526281950; mobile_app=1; Hm_lpvt_5ffc07c2ca2eda4cc1c4d8e50804c94b=1526372230; __utmb=56961525.8.8.1526372229947",
    "If-Modified-Since": "Tue, 15 May 2018 08:15:46 GMT"
}
proxy = {'http': '125.46.0.62:53281'}

cookie = [{'domain': '.okooo.com', 'expiry': 1526378450, 'httpOnly': False, 'name': '__utmb', 'path': '/', 'secure': False, 'value': '56961525.3.8.1526376650'}, {'domain': '.okooo.com', 'expiry': 1542144650, 'httpOnly': False, 'name': '__utmz', 'path': '/', 'secure': False, 'value': '56961525.1526376650.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)'}, {'domain': '.okooo.com', 'httpOnly': False, 'name': 'Hm_lpvt_5ffc07c2ca2eda4cc1c4d8e50804c94b', 'path': '/', 'secure': False, 'value': '1526376650'}, {'domain': '.okooo.com', 'expiry': 1557912650, 'httpOnly': False, 'name': 'Hm_lvt_5ffc07c2ca2eda4cc1c4d8e50804c94b', 'path': '/', 'secure': False, 'value': '1526376650'}, {'domain': '.okooo.com', 'httpOnly': False, 'name': '__utmc', 'path': '/', 'secure': False, 'value': '56961525'}, {'domain': '.okooo.com', 'httpOnly': False, 'name': 'PHPSESSID', 'path': '/', 'secure': False, 'value': '38f7d90a883d024e02001c3aa07e16b3b5c0a380'}, {'domain': '.okooo.com', 'expiry': 1589448650, 'httpOnly': False, 'name': '__utma', 'path': '/', 'secure': False, 'value': '56961525.1468799191.1526376650.1526376650.1526376650.1'}]
def get_matches(url,cookie):
    #response = requests.get(url, headers=headers, proxies=proxy)
    browser = webdriver.Chrome()
    #browser.add_cookie(cookie)
    browser.get(url)
    page = browser.page_source
    browser.close()
    print(page)
    m = re.findall("<p class=float_l livelistcontext>(.*)\.</p>", page)
    print(m)
    #print(browser.find_elements_by_xpath("float_r livebox"))
    while 1:
        pass
    #print(response.content)
    #result = json.loads(response.content.decode('gb2312'))
    #print(result)


# response = requests.get(format_url, headers=headers)
# result = json.loads(response.content.decode('utf-8'))
get_matches(r"http://www.okooo.com/soccer/match/974052/form/", cookie)
# print(result)
