#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 下午1:42
# @Author  : Kaiyu
# @Site    :
# @File    : youku.py
import os
import random
import sys
import time

import requests
from lxml import etree
import re
from multiprocessing import Pool


def get_video_url(url):
    page_ = requests.get(url)
    page = etree.HTML(page_.text)
    name = page.xpath('//*[@mid="001"]/h1/span/a/text()')[0]
    urls = []
    try:
        urls = page.xpath('//*[@data-sn="15"]/div[2]/div[1]/div[1]/div[1]/ul[1]')[0]
        urls = urls.xpath('li//*[@class="title"]/a/@href')
        urls = ['http:' + str(item) for item in urls]
    except Exception as e:
        print(e)
        print('name: {} url: {}'.format(name, url))

    return urls, str(name)


def download_videos(url):
    os.system('youtube-dl {}'.format(url))


def get_match_urls():
    page = '''<html class="gr__h5_m_youku_com"><head><script type="text/javascript" async="" src="http://g.alicdn.com/secdev/entry/index.js?t=212761" id="aplus-sufei"></script><script src="//g.alicdn.com/security/umscript/3.3.32/um.js" id="AWSC_umidPCModule"></script><script src="//af.alicdn.com/js/cj/110.js" id="AWSC_uabModule"></script><script async="" src="http://cmstool.youku.com/cms/tool/pub/get_putdata.json?securemode=2&amp;callback=jQuery111209694134727076091_1531885691718&amp;client=pc&amp;_=1531885691719"></script> 
  <meta charset="utf-8"> 
  <meta http-equiv="X-UA-Compatible" content="IE=Edge"> 
  <meta name="title" content="2018世界杯赛程"> 
  <meta name="keywords" content=""> 
  <meta name="description" content="2018世界杯赛程"> 
  <meta name="google-site-verification" content="F2zYXuMVH2X76NeYEdAiGokv0QFtgOB1ZgCSZPglQbs"> 
  <link rel="Shortcut Icon" href=" //static.youku.com/v1.0.166/index/img/favicon.ico"> 
  <link href="//img.alicdn.com/tfs/TB1u5jLkMoQMeJjy0FpXXcTxpXa-200-200.jpg" rel="apple-touch-icon-precomposed"> 
  <title>世界杯全部赛程</title> 
  <link type="text/css" href="//css.ykimg.com/youku/dist/css/find/g_78.css" rel="stylesheet"> 
  <link href="//css.ykimg.com/youku/dist/css/find/worldcup/schedule-view_9.css" type="text/css" rel="stylesheet"> 
  <meta name="data-spm" content="a2h8q">
  <meta name="data-scm" content="20140614.a2h4o.mqyxt1c968">
  <link type="image/x-icon" href="//static.youku.com/index/img/favicon.ico" rel="Shortcut Icon">
 <link id="YT-loginFrameCss" rel="stylesheet" href="//g.alicdn.com/static-es6/login/pc/login/css/main_1c820bfa.css"><style type="text/css" abt="234"></style><script>//console.log('a')
</script><script>//remove 17173 video ad
doAdblock();
function doAdblock(){
    (function() {
        function A() {}
        A.prototype = {
            rules: {
                '17173_in':{
                    'find':/http:\/\/f\.v\.17173cdn\.com\/(\d+\/)?flash\/PreloaderFile(Customer)?\.swf/,
                    'replace':"http://swf.adtchrome.com/17173_in_20150522.swf"
                },
                '17173_out':{
                    'find':/http:\/\/f\.v\.17173cdn\.com\/(\d+\/)?flash\/PreloaderFileFirstpage\.swf/,
                    'replace':"http://swf.adtchrome.com/17173_out_20150522.swf"
                },
                '17173_live':{
                    'find':/http:\/\/f\.v\.17173cdn\.com\/(\d+\/)?flash\/Player_stream(_firstpage)?\.swf/,
                    'replace':"http://swf.adtchrome.com/17173_stream_20150522.swf"
                },
                '17173_live_out':{
                    'find':/http:\/\/f\.v\.17173cdn\.com\/(\d+\/)?flash\/Player_stream_(custom)?Out\.swf/,
                    'replace':"http://swf.adtchrome.com/17173.out.Live.swf"
                }
            },
            _done: null,
            get done() {
                if(!this._done) {
                    this._done = new Array();
                }
                return this._done;
            },
            addAnimations: function() {
                var style = document.createElement('style');
                style.type = 'text/css';
                style.innerHTML = 'object,embed{\
                -webkit-animation-duration:.001s;-webkit-animation-name:playerInserted;\
                -ms-animation-duration:.001s;-ms-animation-name:playerInserted;\
                -o-animation-duration:.001s;-o-animation-name:playerInserted;\
                animation-duration:.001s;animation-name:playerInserted;}\
                @-webkit-keyframes playerInserted{from{opacity:0.99;}to{opacity:1;}}\
                @-ms-keyframes playerInserted{from{opacity:0.99;}to{opacity:1;}}\
                @-o-keyframes playerInserted{from{opacity:0.99;}to{opacity:1;}}\
                @keyframes playerInserted{from{opacity:0.99;}to{opacity:1;}}';
                document.getElementsByTagName('head')[0].appendChild(style);
            },
            animationsHandler: function(e) {
                if(e.animationName === 'playerInserted') {
                    this.replace(e.target);
                }
            },
            replace: function(elem) {
                if(this.done.indexOf(elem) != -1) return;
                this.done.push(elem);
                var player = elem.data || elem.src;
                if(!player) return;
                var i, find, replace = false;
                for(i in this.rules) {
                    find = this.rules[i]['find'];
                    if(find.test(player)) {
                        replace = this.rules[i]['replace'];
                        if('function' === typeof this.rules[i]['preHandle']) {
                            this.rules[i]['preHandle'].bind(this, elem, find, replace, player)();
                        }else{
                            this.reallyReplace.bind(this, elem, find, replace)();
                        }
                        break;
                    }
                }
            },
            reallyReplace: function(elem, find, replace) {
                elem.data && (elem.data = elem.data.replace(find, replace)) || elem.src && ((elem.src = elem.src.replace(find, replace)) && (elem.style.display = 'block'));
                var b = elem.querySelector("param[name='movie']");
                this.reloadPlugin(elem);
            },
            reloadPlugin: function(elem) {
                var nextSibling = elem.nextSibling;
                var parentNode = elem.parentNode;
                parentNode.removeChild(elem);
                var newElem = elem.cloneNode(true);
                this.done.push(newElem);
                if(nextSibling) {
                    parentNode.insertBefore(newElem, nextSibling);
                } else {
                    parentNode.appendChild(newElem);
                }
            },
            init: function() {
                var handler = this.animationsHandler.bind(this);
                document.body.addEventListener('webkitAnimationStart', handler, false);
                document.body.addEventListener('msAnimationStart', handler, false);
                document.body.addEventListener('oAnimationStart', handler, false);
                document.body.addEventListener('animationstart', handler, false);
                this.addAnimations();
            }
        };
        new A().init();
    })();
}
//remove baidu search ad
if(document.URL.indexOf('www.baidu.com') >= 0){
    if(document && document.getElementsByTagName && document.getElementById && document.body){
        var aa = function(){
            var all = document.body.querySelectorAll("#content_left div,#content_left table");
            for(var i = 0; i < all.length; i++){
                if(/display:\s?(table|block)\s!important/.test(all[i].getAttribute("style"))){all[i].style.display= "none";all[i].style.visibility='hidden';}
            }
            all = document.body.querySelectorAll('.result.c-container[id="1"]');
            //if(all.length == 1) return;
            for(var i = 0; i < all.length; i++){
                if(all[i].innerHTML && all[i].innerHTML.indexOf('广告')>-1){
                    all[i].style.display= "none";all[i].style.visibility='hidden';
                }
            }
        }
        aa();
        document.getElementById('wrapper_wrapper').addEventListener('DOMSubtreeModified',aa)
    };
}
//remove sohu video ad
if (document.URL.indexOf("tv.sohu.com") >= 0){
    if (document.cookie.indexOf("fee_status=true")==-1){document.cookie='fee_status=true'};
}
//remove 56.com video ad
if (document.URL.indexOf("56.com") >= 0){
    if (document.cookie.indexOf("fee_status=true")==-1){document.cookie='fee_status=true'};
}
//fore iqiyi enable html5 player function
if (document.URL.indexOf("iqiyi.com") >= 0){
    if (document.cookie.indexOf("player_forcedType=h5_VOD")==-1){
        document.cookie='player_forcedType=h5_VOD'
        if(localStorage.reloadTime && Date.now() - parseInt(localStorage.reloadTime)<60000){
            console.log('no reload')
        }else{
            location.reload()
            localStorage.reloadTime = Date.now();
        }
    }
}
</script><style type="text/css">object,embed{                -webkit-animation-duration:.001s;-webkit-animation-name:playerInserted;                -ms-animation-duration:.001s;-ms-animation-name:playerInserted;                -o-animation-duration:.001s;-o-animation-name:playerInserted;                animation-duration:.001s;animation-name:playerInserted;}                @-webkit-keyframes playerInserted{from{opacity:0.99;}to{opacity:1;}}                @-ms-keyframes playerInserted{from{opacity:0.99;}to{opacity:1;}}                @-o-keyframes playerInserted{from{opacity:0.99;}to{opacity:1;}}                @keyframes playerInserted{from{opacity:0.99;}to{opacity:1;}}</style><script src="//g.alicdn.com/fsp/tracker-patch/index.js?1531885691820" async="" crossorigin="true" id="tracker-patch"></script><script async="" src="//log.mmstat.com/eg.js"></script><script src="//g.alicdn.com/secdev/sufei_data/3.5.4/index.js" id="aplus-sufei"></script></head> 
 <body data-spm="11643819" data-gr-c-s-loaded="true" class=" black-skin w1080"> 
  <script>/*! 2018-07-13 17:06:31 v8.5.1 */
!function(e){function i(n){if(o[n])return o[n].exports;var r=o[n]={exports:{},id:n,loaded:!1};return e[n].call(r.exports,r,r.exports,i),r.loaded=!0,r.exports}var o={};return i.m=e,i.c=o,i.p="",i(0)}([function(e,i){"use strict";var o=window,n=document;!function(){var e=2,r="ali_analytics";if(o[r]&&o[r].ua&&e<=o[r].ua.version)return void(i.info=o[r].ua);var t,a,d,s,c,u,h,l,m,b,f,v,p,w,g,x,z,O=o.navigator,k=O.appVersion,T=O&&O.userAgent||"",y=function(e){var i=0;return parseFloat(e.replace(/\./g,function(){return 0===i++?".":""}))},_=function(e,i){var o,n;i[o="trident"]=.1,(n=e.match(/Trident\/([\d.]*)/))&&n[1]&&(i[o]=y(n[1])),i.core=o},N=function(e){var i,o;return(i=e.match(/MSIE ([^;]*)|Trident.*; rv(?:\s|:)?([0-9.]+)/))&&(o=i[1]||i[2])?y(o):0},P=function(e){return e||"other"},M=function(e){function i(){for(var i=[["Windows NT 5.1","winXP"],["Windows NT 6.1","win7"],["Windows NT 6.0","winVista"],["Windows NT 6.2","win8"],["Windows NT 10.0","win10"],["iPad","ios"],["iPhone;","ios"],["iPod","ios"],["Macintosh","mac"],["Android","android"],["Ubuntu","ubuntu"],["Linux","linux"],["Windows NT 5.2","win2003"],["Windows NT 5.0","win2000"],["Windows","winOther"],["rhino","rhino"]],o=0,n=i.length;o<n;++o)if(e.indexOf(i[o][0])!==-1)return i[o][1];return"other"}function r(e,i,n,r){var t,a=o.navigator.mimeTypes;try{for(t in a)if(a.hasOwnProperty(t)&&a[t][e]==i){if(void 0!==n&&r.test(a[t][n]))return!0;if(void 0===n)return!0}return!1}catch(e){return!1}}var t,a,d,s,c,u,h,l="",m=l,b=l,f=[6,9],v="{{version}}",p="<!--[if IE "+v+"]><s></s><![endif]-->",w=n&&n.createElement("div"),g=[],x={webkit:void 0,edge:void 0,trident:void 0,gecko:void 0,presto:void 0,chrome:void 0,safari:void 0,firefox:void 0,ie:void 0,ieMode:void 0,opera:void 0,mobile:void 0,core:void 0,shell:void 0,phantomjs:void 0,os:void 0,ipad:void 0,iphone:void 0,ipod:void 0,ios:void 0,android:void 0,nodejs:void 0,extraName:void 0,extraVersion:void 0};if(w&&w.getElementsByTagName&&(w.innerHTML=p.replace(v,""),g=w.getElementsByTagName("s")),g.length>0){for(_(e,x),s=f[0],c=f[1];s<=c;s++)if(w.innerHTML=p.replace(v,s),g.length>0){x[b="ie"]=s;break}!x.ie&&(d=N(e))&&(x[b="ie"]=d)}else((a=e.match(/AppleWebKit\/*\s*([\d.]*)/i))||(a=e.match(/Safari\/([\d.]*)/)))&&a[1]?(x[m="webkit"]=y(a[1]),(a=e.match(/OPR\/(\d+\.\d+)/))&&a[1]?x[b="opera"]=y(a[1]):(a=e.match(/Chrome\/([\d.]*)/))&&a[1]?x[b="chrome"]=y(a[1]):(a=e.match(/\/([\d.]*) Safari/))&&a[1]?x[b="safari"]=y(a[1]):x.safari=x.webkit,(a=e.match(/Edge\/([\d.]*)/))&&a[1]&&(m=b="edge",x[m]=y(a[1])),/ Mobile\//.test(e)&&e.match(/iPad|iPod|iPhone/)?(x.mobile="apple",a=e.match(/OS ([^\s]*)/),a&&a[1]&&(x.ios=y(a[1].replace("_","."))),t="ios",a=e.match(/iPad|iPod|iPhone/),a&&a[0]&&(x[a[0].toLowerCase()]=x.ios)):/ Android/i.test(e)?(/Mobile/.test(e)&&(t=x.mobile="android"),a=e.match(/Android ([^\s]*);/),a&&a[1]&&(x.android=y(a[1]))):(a=e.match(/NokiaN[^\/]*|Android \d\.\d|webOS\/\d\.\d/))&&(x.mobile=a[0].toLowerCase()),(a=e.match(/PhantomJS\/([^\s]*)/))&&a[1]&&(x.phantomjs=y(a[1]))):(a=e.match(/Presto\/([\d.]*)/))&&a[1]?(x[m="presto"]=y(a[1]),(a=e.match(/Opera\/([\d.]*)/))&&a[1]&&(x[b="opera"]=y(a[1]),(a=e.match(/Opera\/.* Version\/([\d.]*)/))&&a[1]&&(x[b]=y(a[1])),(a=e.match(/Opera Mini[^;]*/))&&a?x.mobile=a[0].toLowerCase():(a=e.match(/Opera Mobi[^;]*/))&&a&&(x.mobile=a[0]))):(d=N(e))?(x[b="ie"]=d,_(e,x)):(a=e.match(/Gecko/))&&(x[m="gecko"]=.1,(a=e.match(/rv:([\d.]*)/))&&a[1]&&(x[m]=y(a[1]),/Mobile|Tablet/.test(e)&&(x.mobile="firefox")),(a=e.match(/Firefox\/([\d.]*)/))&&a[1]&&(x[b="firefox"]=y(a[1])));t||(t=i());var z,O,T;if(!r("type","application/vnd.chromium.remoting-viewer")){z="scoped"in n.createElement("style"),T="v8Locale"in o;try{O=o.external||void 0}catch(e){}if(a=e.match(/360SE/))u="360";else if((a=e.match(/SE\s([\d.]*)/))||O&&"SEVersion"in O)u="sougou",h=y(a[1])||.1;else if((a=e.match(/Maxthon(?:\/)+([\d.]*)/))&&O){u="maxthon";try{h=y(O.max_version||a[1])}catch(e){h=.1}}else z&&T?u="360se":z||T||!/Gecko\)\s+Chrome/.test(k)||x.opera||x.edge||(u="360ee")}(a=e.match(/TencentTraveler\s([\d.]*)|QQBrowser\/([\d.]*)/))?(u="tt",h=y(a[2])||.1):(a=e.match(/LBBROWSER/))||O&&"LiebaoGetVersion"in O?u="liebao":(a=e.match(/TheWorld/))?(u="theworld",h=3):(a=e.match(/TaoBrowser\/([\d.]*)/))?(u="taobao",h=y(a[1])||.1):(a=e.match(/UCBrowser\/([\d.]*)/))&&(u="uc",h=y(a[1])||.1),x.os=t,x.core=x.core||m,x.shell=b,x.ieMode=x.ie&&n.documentMode||x.ie,x.extraName=u,x.extraVersion=h;var P=o.screen.width,M=o.screen.height;return x.resolution=P+"x"+M,x},S=function(e){function i(e){return Object.prototype.toString.call(e)}function o(e,o,n){if("[object Function]"==i(o)&&(o=o(n)),!o)return null;var r={name:e,version:""},t=i(o);if(o===!0)return r;if("[object String]"===t){if(n.indexOf(o)!==-1)return r}else if(o.exec){var a=o.exec(n);if(a)return a.length>=2&&a[1]?r.version=a[1].replace(/_/g,"."):r.version="",r}}var n={name:"other",version:""};e=(e||"").toLowerCase();for(var r=[["nokia",function(e){return e.indexOf("nokia ")!==-1?/\bnokia ([0-9]+)?/:/\bnokia([a-z0-9]+)?/}],["samsung",function(e){return e.indexOf("samsung")!==-1?/\bsamsung(?:[ \-](?:sgh|gt|sm))?-([a-z0-9]+)/:/\b(?:sgh|sch|gt|sm)-([a-z0-9]+)/}],["wp",function(e){return e.indexOf("windows phone ")!==-1||e.indexOf("xblwp")!==-1||e.indexOf("zunewp")!==-1||e.indexOf("windows ce")!==-1}],["pc","windows"],["ipad","ipad"],["ipod","ipod"],["iphone",/\biphone\b|\biph(\d)/],["mac","macintosh"],["mi",/\bmi[ \-]?([a-z0-9 ]+(?= build|\)))/],["hongmi",/\bhm[ \-]?([a-z0-9]+)/],["aliyun",/\baliyunos\b(?:[\-](\d+))?/],["meizu",function(e){return e.indexOf("meizu")>=0?/\bmeizu[\/ ]([a-z0-9]+)\b/:/\bm([0-9x]{1,3})\b/}],["nexus",/\bnexus ([0-9s.]+)/],["huawei",function(e){var i=/\bmediapad (.+?)(?= build\/huaweimediapad\b)/;return e.indexOf("huawei-huawei")!==-1?/\bhuawei\-huawei\-([a-z0-9\-]+)/:i.test(e)?i:/\bhuawei[ _\-]?([a-z0-9]+)/}],["lenovo",function(e){return e.indexOf("lenovo-lenovo")!==-1?/\blenovo\-lenovo[ \-]([a-z0-9]+)/:/\blenovo[ \-]?([a-z0-9]+)/}],["zte",function(e){return/\bzte\-[tu]/.test(e)?/\bzte-[tu][ _\-]?([a-su-z0-9\+]+)/:/\bzte[ _\-]?([a-su-z0-9\+]+)/}],["vivo",/\bvivo(?: ([a-z0-9]+))?/],["htc",function(e){return/\bhtc[a-z0-9 _\-]+(?= build\b)/.test(e)?/\bhtc[ _\-]?([a-z0-9 ]+(?= build))/:/\bhtc[ _\-]?([a-z0-9 ]+)/}],["oppo",/\boppo[_]([a-z0-9]+)/],["konka",/\bkonka[_\-]([a-z0-9]+)/],["sonyericsson",/\bmt([a-z0-9]+)/],["coolpad",/\bcoolpad[_ ]?([a-z0-9]+)/],["lg",/\blg[\-]([a-z0-9]+)/],["android",/\bandroid\b|\badr\b/],["blackberry",function(e){return e.indexOf("blackberry")>=0?/\bblackberry\s?(\d+)/:"bb10"}]],t=0;t<r.length;t++){var a=r[t][0],d=r[t][1],s=o(a,d,e);if(s){n=s;break}}return n},E=1;try{t=M(T),a=S(T),d=t.os,s=t.shell,c=t.core,u=t.resolution,h=t.extraName,l=t.extraVersion,m=a.name,b=a.version,v=d?d+(t[d]?t[d]:""):"",p=s?s+parseInt(t[s]):"",w=c,g=u,x=h?h+(l?parseInt(l):""):"",z=m+b}catch(e){}f={p:E,o:P(v),b:P(p),w:P(w),s:g,mx:x,ism:z},o[r]||(o[r]={}),o[r].ua||(o[r].ua={}),o.goldlog||(o.goldlog={}),i.info=o[r].ua=goldlog._aplus_client={version:e,ua_info:f}}()}]);/*! 2017-10-31 20:15:15 v0.2.4 */
!function(t){function e(o){if(n[o])return n[o].exports;var i=n[o]={exports:{},id:o,loaded:!1};return t[o].call(i.exports,i,i.exports,e),i.loaded=!0,i.exports}var n={};return e.m=t,e.c=n,e.p="",e(0)}([function(t,e,n){"use strict";!function(){var t=window.goldlog||(window.goldlog={});t._aplus_cplugin_utilkit||(t._aplus_cplugin_utilkit={status:"init"},n(1).init(t),t._aplus_cplugin_utilkit.status="complete")}()},function(t,e,n){"use strict";var o=n(2),i=n(4);e.init=function(t){t.setCookie=o.setCookie,t.getCookie=o.getCookie,t.on=i.on}},function(t,e,n){"use strict";var o=document,i=n(3),a=function(t){var e=new RegExp("(?:^|;)\\s*"+t+"=([^;]+)"),n=o.cookie.match(e);return n?n[1]:""};e.getCookie=a;var r=function(t,e,n){n||(n={});var i=new Date;return n.expires&&("number"==typeof n.expires||n.expires.toUTCString)?("number"==typeof n.expires?i.setTime(i.getTime()+24*n.expires*60*60*1e3):i=n.expires,e+="; expires="+i.toUTCString()):"session"!==n.expires&&(i.setTime(i.getTime()+63072e7),e+="; expires="+i.toUTCString()),e+="; path="+(n.path?n.path:"/"),e+="; domain="+n.domain,o.cookie=t+"="+e,a(t)};e.setCookie=function(t,e,n){try{if(n||(n={}),n.domain)r(t,e,n);else for(var o=i.getDomains(),a=0;a<o.length;)n.domain=o[a],r(t,e,n)?a=o.length:a++}catch(t){}}},function(t,e){"use strict";e.getDomains=function(){var t=[];try{for(var e=location.hostname,n=e.split("."),o=2;o<=n.length;)t.push(n.slice(n.length-o).join(".")),o++}catch(t){}return t}},function(t,e){"use strict";var n=window,o=document,i=!!o.attachEvent,a="attachEvent",r="addEventListener",c=i?a:r,u=function(t,e){var n=goldlog._$||{},o=n.meta_info||{},i=o.aplus_ctap||{};if(i&&"function"==typeof i.on)i.on(t,e);else{var a="ontouchend"in document.createElement("div"),r=a?"touchstart":"mousedown";s(t,r,e)}},s=function(t,e,o){return"tap"===e?void u(t,o):void t[c]((i?"on":"")+e,function(t){t=t||n.event;var e=t.target||t.srcElement;"function"==typeof o&&o(t,e)},!1)};e.on=s;var d=function(t){try{o.documentElement.doScroll("left")}catch(e){return void setTimeout(function(){d(t)},1)}t()},l=function(t){var e=0,n=function(){0===e&&t(),e++};"complete"===o.readyState&&n();var i;if(o.addEventListener)i=function(){o.removeEventListener("DOMContentLoaded",i,!1),n()},o.addEventListener("DOMContentLoaded",i,!1),window.addEventListener("load",n,!1);else if(o.attachEvent){i=function(){"complete"===o.readyState&&(o.detachEvent("onreadystatechange",i),n())},o.attachEvent("onreadystatechange",i),window.attachEvent("onload",n);var a=!1;try{a=null===window.frameElement}catch(t){}o.documentElement.doScroll&&a&&d(n)}};e.DOMReady=function(t){l(t)},e.onload=function(t){"complete"===o.readyState?t():s(n,"load",t)}}]);/*! 2017-12-19 12:10:24 v0.2.9 */
!function(o){function t(r){if(e[r])return e[r].exports;var a=e[r]={exports:{},id:r,loaded:!1};return o[r].call(a.exports,a,a.exports,t),a.loaded=!0,a.exports}var e={};return t.m=o,t.c=e,t.p="",t(0)}([function(o,t,e){"use strict";!function(){var o=window.goldlog||(window.goldlog={});o._aplus_cplugin_m||(o._aplus_cplugin_m=e(1).run())}()},function(o,t,e){"use strict";var r=e(2),a=e(3),n=e(4),s=navigator.sendBeacon?"post":"get";e(5).run(),t.run=function(){return{status:"complete",do_tracker_jserror:function(o){try{var t=new n({logkey:o?o.logkey:"",ratio:o&&"number"==typeof o.ratio&&o.ratio>0?o.ratio:r.jsErrorRecordRatio}),e=["Message: "+o.message,"Error object: "+o.error].join(" - "),c=goldlog.spm_ab||[],i=location.hostname+location.pathname;t.run({code:110,page:i,msg:"record_jserror_by"+s+"_"+o.message,spm_a:c[0],spm_b:c[1],c1:e,c2:o.filename,c3:location.protocol+"//"+i})}catch(o){a.logger({msg:o})}},do_tracker_lostpv:function(o){var t=!1;try{if(o&&o.page){var e=o.spm_ab?o.spm_ab.split("."):[],c="record_lostpv_by"+s+"_"+o.msg,i=new n({ratio:o.ratio||r.lostPvRecordRatio});i.run({code:102,page:o.page,msg:c,spm_a:e[0],spm_b:e[1],c1:o.duration,c2:o.page_url}),t=!0}}catch(o){a.logger({msg:o})}return t},do_tracker_obsolete_inter:function(o){var t=!1;try{if(o&&o.page){var e=o.spm_ab?o.spm_ab.split("."):[],c="record_obsolete interface be called by"+s,i=new n({ratio:o.ratio||r.obsoleteInterRecordRatio});i.run({code:109,page:o.page,msg:c,spm_a:e[0],spm_b:e[1],c1:o.interface_name,c2:o.interface_params}),t=!0}}catch(o){a.logger({msg:o})}return t},do_tracker_browser_support:function(o){var t=!1;try{if(o&&o.page){var e=o.spm_ab?o.spm_ab.split("."):[],c=new n({ratio:o.ratio||r.browserSupportRatio}),i=goldlog._aplus_client||{},g=i.ua_info||{};c.run({code:111,page:o.page,msg:o.msg+"_by"+s,spm_a:e[0],spm_b:e[1],c1:[g.o,g.b,g.w].join("_"),c2:o.etag||"",c3:o.cna||""}),t=!0}}catch(o){a.logger({msg:o})}return t}}}},function(o,t){"use strict";t.lostPvRecordRatio="0.01",t.obsoleteInterRecordRatio="0.01",t.jsErrorRecordRatio="0.01",t.browserSupportRatio="0.01",t.goldlogQueueRatio="0.01"},function(o,t){"use strict";var e=function(o){var t=o.level||"warn";window.console&&window.console[t]&&window.console[t](o.msg)};t.logger=e,t.assign=function(o,t){if("function"!=typeof Object.assign){var e=function(o){if(null===o)throw new TypeError("Cannot convert undefined or null to object");for(var t=Object(o),e=1;e<arguments.length;e++){var r=arguments[e];if(null!==r)for(var a in r)Object.prototype.hasOwnProperty.call(r,a)&&(t[a]=r[a])}return t};return e(o,t)}return Object.assign({},o,t)},t.makeCacheNum=function(){return Math.floor(268435456*Math.random()).toString(16)},t.obj2param=function(o){var t,e,r=[];for(t in o)o.hasOwnProperty(t)&&(e=""+o[t],r.push(t+"="+encodeURIComponent(e)));return r.join("&")}},function(o,t,e){var r=e(3),a={ratio:1,logkey:"fsp.1.1",gmkey:"",chksum:"H46747615"},n=function(o){o&&"object"==typeof o||(o=a),this.opts=o,this.opts.ratio=o.ratio||a.ratio,this.opts.logkey=o.logkey||a.logkey,this.opts.gmkey=o.gmkey||a.gmkey,this.opts.chksum=o.chksum||a.chksum},s=n.prototype;s.getRandom=function(){return Math.floor(100*Math.random())+1},s.run=function(o){var t,e,a={pid:"aplus",code:101,msg:"异常内容"},n="";try{var s=window.goldlog||{},c=s._$||{},i=c.meta_info||{},g=parseFloat(i["aplus-tracker-rate"]);if(t=this.opts||{},"number"==typeof g&&g+""!="NaN"||(g=t.ratio),e=this.getRandom(),e<=100*g){n="//gm.mmstat.com/"+t.logkey,o.rel=c.script_name+"@"+s.lver,o.type=o.code,o.uid=encodeURIComponent(s.getCookie("cna")),o=r.assign(a,o);var l=r.obj2param(o);s.tracker=s.send(n,{cache:r.makeCacheNum(),gokey:l,logtype:"2"},"POST")}}catch(o){r.logger({msg:"tracker.run() exec error: "+o})}},o.exports=n},function(o,t,e){"use strict";var r=e(6),a=function(o){var t=window.goldlog||{},e=t._$=t._$||{},r=t.spm_ab?t.spm_ab.join("."):"0.0",a=e.send_pv_count||0;if(a<1&&navigator&&navigator.sendBeacon){var n=window.goldlog_queue||(window.goldlog_queue=[]),s=location.hostname+location.pathname;n.push({action:["goldlog","_aplus_cplugin_m","do_tracker_lostpv"].join("."),arguments:[{page:s,page_url:location.protocol+"//"+s,duration:o,spm_ab:r,msg:"dom_state="+document.readyState}]})}};t.run=function(){var o=new Date;r.on(window,"beforeunload",function(){var t=new Date,e=t.getTime()-o.getTime();a(e)})}},function(o,t){"use strict";var e=window,r=document,a=!!r.attachEvent,n="attachEvent",s="addEventListener",c=a?n:s;t.getIframeUrl=function(o){var t,e="//g.alicdn.com";return t=goldlog&&"function"==typeof goldlog.getCdnPath?goldlog.getCdnPath()||e:e,(o||"https")+":"+t+"/alilog/aplus_cplugin/@@APLUS_CPLUGIN_VER/ls.html"},t.on=function(o,t,r){o[c]((a?"on":"")+t,function(o){o=o||e.event;var t=o.target||o.srcElement;"function"==typeof r&&r(o,t)},!1)},t.checkLs=function(){var o;try{window.localStorage&&(localStorage.setItem("test_log_cna","1"),"1"===localStorage.getItem("test_log_cna")&&(localStorage.removeItem("test_log_cna"),o=!0))}catch(t){o=!1}return o},t.tracker_iframe_status=function(o,t){var e=window.goldlog_queue||(window.goldlog_queue=[]),r=goldlog.spm_ab?goldlog.spm_ab.join("."):"",a="createIframe_"+t.status+"_id="+o;t.msg&&(a+="_"+t.msg),e.push({action:"goldlog._aplus_cplugin_m.do_tracker_browser_support",arguments:[{page:location.hostname+location.pathname,msg:a,browser_attr:navigator.userAgent,spm_ab:r,cna:t.duration||"",ratio:.01}]})},t.tracker_ls_failed=function(){var o=window.goldlog_queue||(window.goldlog_queue=[]),t=goldlog.spm_ab?goldlog.spm_ab.join("."):"";o.push({action:"goldlog._aplus_cplugin_m.do_tracker_browser_support",arguments:[{page:location.hostname+location.pathname,msg:"donot support localStorage",browser_attr:navigator.userAgent,spm_ab:t}]})},t.processMsgData=function(o){var t={};try{t="string"==typeof o?JSON.parse(o):o,t||(t={})}catch(o){t={}}return t},t.do_pub_fn=function(o,t){var e=window.goldlog_queue||(window.goldlog_queue=[]);e.push({action:"goldlog.aplus_pubsub.publish",arguments:[o,t]}),e.push({action:"goldlog.aplus_pubsub.cachePubs",arguments:[o,t]})}}]);/*! 2018-07-13 17:06:21 v8.5.1 */
!function(t){function e(o){if(n[o])return n[o].exports;var a=n[o]={exports:{},id:o,loaded:!1};return t[o].call(a.exports,a,a.exports,e),a.loaded=!0,a.exports}var n={};return e.m=t,e.c=n,e.p="",e(0)}([function(t,e,n){t.exports=n(1)},function(t,e,n){"use strict";!function(){var t=window,e=n(2),o=n(3),a=function(){n(87);var e=n(89),o=n(32);if(o.doPubMsg(["goldlogReady","running"]),document.getElementsByTagName("body").length){var r="g_tb_aplus_loaded";if(t[r])return;t[r]=1,n(97).initGoldlog(e)}else setTimeout(function(){a()},50)},r=function(n,o){try{var a=Math.floor(268435456*Math.random()).toString(16),r=t.location||{},i=["page="+encodeURIComponent(r.href),"info="+encodeURIComponent(n.message),"stack="+encodeURIComponent(n&&n.stack?n.stack:""),"filename=aplus_core","method="+o,"cache="+a].join("&"),s=r.protocol+"//gm.mmstat.com/mm.req.load?"+i;navigator&&navigator.sendBeacon?e.postData(s):e.sendImg(s)}catch(t){}};try{a()}catch(t){r(t,o.script_name+"@"+o.lver)}}()},function(t,e){"use strict";var n=window;e.sendImg=function(t,e){var o=new Image,a="_img_"+Math.random();n[a]=o;var r=function(){if(n[a])try{delete n[a]}catch(t){n[a]=void 0}};return o.onload=function(){r()},o.onerror=function(){r()},setTimeout(function(){window[a]&&(window[a].src="",r())},e||5e3),o.src=t,o=null,t},e.postData=function(t,e){for(var n in e)"cna"!==n&&(e[n]=encodeURIComponent(e[n]));return navigator.sendBeacon(t,JSON.stringify(e)),t}},function(t,e,n){"use strict";var o=n(4),a=n(5),r=n(6);e.APLUS_ENV="production",e.lver=a.lver,e.toUtVersion=a.toUtVersion,e.script_name=a.script_name,e.recordTypes=o.recordTypes,e.KEY=o.KEY,e.context=r.context,e.context_prepv=r.context_prepv,e.aplus_init=n(13).plugins_init,e.plugins_pv=n(30).plugins_pv,e.plugins_prepv=n(56).plugins_prepv,e.context_hjlj=n(61),e.plugins_hjlj=n(63).plugins_hjlj,e.beforeUnload=n(70),e.initLoad=n(72),e.spmException=n(78),e.goldlog_path=n(79),e.is_auto_pv="false",e.utilPvid=n(83),e.disablePvid="true",e.mustSpmE=!0,e.LS_CNA_KEY="APLUS_CNA"},function(t,e){"use strict";e.recordTypes={hjlj:"COMMON_HJLJ",uhjlj:"DATACLICK_HJLJ",pv:"PV",prepv:"PREPV"},e.KEY={NAME_STORAGE:{REFERRER:"wm_referrer",REFERRER_PV_ID:"refer_pv_id",LOST_PV_PAGE_DURATION:"lost_pv_page_duration",LOST_PV_PAGE_SPMAB:"lost_pv_page_spmab",LOST_PV_PAGE:"lost_pv_page",LOST_PV_PAGE_MSG:"lost_pv_page_msg"}}},function(t,e){"use strict";e.lver="8.5.1",e.toUtVersion="v20180713",e.script_name="aplus_o"},function(t,e,n){"use strict";e.context=n(7),e.context_prepv=n(12)},function(t,e,n){"use strict";function o(){return{compose:{maxTimeout:5500},etag:{egUrl:"//log.mmstat.com/eg.js",cna:i.getCookie("cna")},where_to_sendpv:{url:"//log.mmstat.com/yt.gif",urlRule:s.getBeaconSrc}}}function a(){return r.assign(new s.initConfig,new o)}var r=n(8),i=n(9),s=n(11);t.exports=a},function(t,e){"use strict";function n(t,e){return"function"!=typeof Object.assign?function(t){if(null===t)throw new TypeError("Cannot convert undefined or null to object");for(var e=Object(t),n=1;n<arguments.length;n++){var o=arguments[n];if(null!==o)for(var a in o)Object.prototype.hasOwnProperty.call(o,a)&&(e[a]=o[a])}return e}(t,e):Object.assign({},t,e)}function o(t){return"function"==typeof t}function a(t){return"string"==typeof t}function r(t){return"undefined"==typeof t}function i(t,e){return t.indexOf(e)>-1}var s=window;e.assign=n,e.makeCacheNum=function(){return Math.floor(268435456*Math.random()).toString(16)},e.each=function(t,e){var n,o=t.length;for(n=0;n<o;n++)e(t[n])},e.isStartWith=function(t,e){return 0===t.indexOf(e)},e.isEndWith=function(t,e){var n=t.length,o=e.length;return n>=o&&t.indexOf(e)==n-o},e.any=function(t,e){var n,o=t.length;for(n=0;n<o;n++)if(e(t[n]))return!0;return!1},e.isFunction=o,e.isArray=function(t){return Array.isArray?Array.isArray(t):/Array/.test(Object.prototype.toString.call(t))},e.isString=a,e.isNumber=function(t){return"number"==typeof t},e.isUnDefined=r,e.isContain=i;var u=function(t){var e,n=t.constructor===Array?[]:{};if("object"==typeof t){if(s.JSON&&s.JSON.parse)e=JSON.stringify(t),n=JSON.parse(e);else for(var o in t)n[o]="object"==typeof t[o]?u(t[o]):t[o];return n}};e.cloneObj=u,e.cloneDeep=u},function(t,e,n){"use strict";function o(t){var e=s.cookie.match(new RegExp("(?:^|;)\\s*"+t+"=([^;]+)"));return e?e[1]:""}function a(t,e,n){n||(n={});var a=new Date;return"session"===n.expires||(n.expires&&("number"==typeof n.expires||n.expires.toUTCString)?("number"==typeof n.expires?a.setTime(a.getTime()+24*n.expires*60*60*1e3):a=n.expires,e+="; expires="+a.toUTCString()):(a.setTime(a.getTime()+63072e7),e+="; expires="+a.toUTCString())),e+="; path="+(n.path?n.path:"/"),e+="; domain="+n.domain,s.cookie=t+"="+e,o(t)}function r(t,e,n){try{if(n||(n={}),n.domain)a(t,e,n);else for(var o=c.getDomains(),r=0;r<o.length;)n.domain=o[r],a(t,e,n)?r=o.length:r++}catch(t){}}function i(){var t={};return u.each(p,function(e){t[e]=o(e)}),t.cnaui=/\btanx\.com$/.test(l)?o("cnaui"):"",t}var s=document,u=n(8),c=n(10),l=location.hostname;e.getCookie=o,e.setCookie=r;var p=["tracknick","thw","cna"];e.getData=i,e.getHng=function(){return encodeURIComponent(o("hng")||"")}},function(t,e){"use strict";e.getDomains=function(){var t=[];try{for(var e=location.hostname,n=e.split("."),o=2;o<=n.length;)t.push(n.slice(n.length-o).join(".")),o++}catch(t){}return t}},function(t,e,n){"use strict";function o(t,e,n){var o=window.goldlog||{},s=o.getMetaInfo("aplus-ifr-pv")+""=="1";return e?r(t)?"yt":"m":n&&!s?a.isContain(t,"wrating.com")?"k":i(t)||"y":i(t)||"v"}var a=n(8),r=function(t){for(var e=["youku.com","soku.com","tudou.com","laifeng.com"],n=0;n<e.length;n++){var o=e[n];if(a.isContain(t,o))return!0}return!1},i=function(t){for(var e=[["scmp.com","sc"],["luxehomes.com.hk","sc"],["ays.com.hk","sc"],["cpjobs.com","sc"],["educationpost.com.hk","sc"],["cosmopolitan.com.hk","sc"],["elle.com.hk","sc"],["harpersbazaar.com.hk","sc"],["1688.com","6"],["youku.com","yt"],["soku.com","yt"],["tudou.com","yt"],["laifeng.com","yt"]],n=0;n<e.length;n++){var o=e[n];if(a.isContain(t,o[0]))return o[1]}return""};e.getBeaconSrc=o,e.initConfig=function(){return{compose:{},etag:{egUrl:"//log.mmstat.com/eg.js",cna:"",tag:"",stag:"",lstag:"-1",lscnastatus:""},can_to_sendpv:{flag:"NO"},userdata:{},what_to_sendpv:{pvdata:{},exparams:{}},what_to_pvhash:{hash:[]},what_to_sendpv_ut:{pvdataToUt:{}},what_to_sendpv_ut2:{isSuccess:!1,pvdataToUt:{}},when_to_sendpv:{aplusWaiting:""},where_to_sendpv:{url:"//log.mmstat.com/o.gif",urlRule:o},where_to_sendlog_ut:{aplusToUT:{},toUTName:"toUT"},hjlj:{what_to_hjlj:{logdata:{}},what_to_hjlj_ut:{logdataToUT:{}}},network:{connType:"UNKNOWN"},is_single:!1}}},function(t,e,n){"use strict";function o(){return{etag:{egUrl:"//log.mmstat.com/eg.js",cna:a.getCookie("cna"),tag:"",stag:""},compose:{},where_to_prepv:{url:"//log.mmstat.com/v.gif",urlRule:r.getBeaconSrc},userdata:{},what_to_prepv:{logdata:{}},what_to_hjlj_exinfo:{EXPARAMS_FLAG:"EXPARAMS",exinfo:[],exparams_key_names:["uidaplus","pc_i","pu_i"]},is_single:!1}}var a=n(9),r=n(11);t.exports=o},function(t,e,n){"use strict";e.plugins_init=[{name:"where_to_sendpv",enable:!0,path:n(14)},{name:"etag",enable:!0,path:n(27)},{name:"etag_sync",enable:!0,path:n(29)}]},function(t,e,n){"use strict";var o=n(15),a=n(22),r=n(23);t.exports=function(){return{init:function(t){this.options=t},getMetaInfo:function(){var t=a.getGoldlogVal("_$")||{},e=t.meta_info||r.getInfo();return e},getAplusMetaByKey:function(t){var e=this.getMetaInfo()||{};return e[t]},getGifPath:function(t,e){var n,r=a.getGoldlogVal("_$")||{};if("function"==typeof t)n=t(location.hostname,r.is_terminal,o.is_in_iframe)+".gif";else if(!n&&e){var i=e.match(/\/\w+\.gif/);i&&i.length>0&&(n=i[0])}return n||(n=r.is_terminal?"m.gif":"v.gif"),n},run:function(){var t=!!this.options.context.is_single;if(!t){var e=this.getAplusMetaByKey("aplus-rhost-v"),n=this.options.context.where_to_sendpv||{},a=n.url||"",r=this.getGifPath(n.urlRule,a),i=o.getPvUrl({metaName:"aplus-rhost-v",metaValue:e,gifPath:r,url:o.filterIntUrl(a)});n.url=i,this.options.context.where_to_sendpv=n}}}}},function(t,e,n){"use strict";function o(t){t=(t||"").split("#")[0].split("?")[0];var e=t.length,n=function(t){var e,n=t.length,o=0;for(e=0;e<n;e++)o=31*o+t.charCodeAt(e);return o};return e?n(e+"#"+t.charCodeAt(e-1)):-1}function a(t){for(var e=t.split("&"),n=0,o=e.length,a={};n<o;n++){var r=e[n],i=r.indexOf("="),s=r.substring(0,i),u=r.substring(i+1);a[s]=p.tryToDecodeURIComponent(u)}return a}function r(t){if("function"!=typeof t)throw new TypeError(t+" is not a function");return t}function i(t){var e,n,o,a=[],r=t.length;for(o=0;o<r;o++)e=t[o][0],n=t[o][1],a.push(l.isStartWith(e,v)?n:e+"="+encodeURIComponent(n));return a.join("&")}function s(t){var e,n,o,a={},r=t.length;for(o=0;o<r;o++)e=t[o][0],n=t[o][1],a[e]=n;return a}function u(t,e){var n,o,a,r=[];for(n in t)t.hasOwnProperty(n)&&(o=""+t[n],a=n+"="+encodeURIComponent(o),e?r.push(a):r.push(l.isStartWith(n,v)?o:a));return r.join("&")}function c(t,e){var n=t.indexOf("?")==-1?"?":"&",o=e?l.isArray(e)?i(e):u(e):"";return o?t+n+o:t}var l=n(8),p=n(16),g=n(18),f=parent!==self;e.is_in_iframe=f,e.makeCacheNum=l.makeCacheNum,e.isStartWith=l.isStartWith,e.isEndWith=l.isEndWith,e.any=l.any,e.each=l.each,e.assign=l.assign,e.isFunction=l.isFunction,e.isArray=l.isArray,e.isString=l.isString,e.isNumber=l.isNumber,e.isUnDefined=l.isUnDefined,e.isContain=l.isContain,e.sleep=n(19).sleep,e.makeChkSum=o,e.tryToDecodeURIComponent=p.tryToDecodeURIComponent,e.nodeListToArray=p.nodeListToArray,e.parseSemicolonContent=p.parseSemicolonContent,e.param2obj=a;var d=n(20),h=function(t){return/^(\/\/){0,1}(\w+\.){1,}\w+((\/\w+){1,})?$/.test(t)};e.hostValidity=h;var _=function(t,e){var n=/^(\/\/){0,1}(\w+\.){1,}\w+\/\w+\.gif$/.test(t),o=h(t),a="";return n?a="isGifPath":o&&(a="isHostPath"),a||d.logger({msg:e+": "+t+' is invalid, suggestion: "xxx.mmstat.com"'}),a},m=function(t){return!/^\/\/gj\.mmstat/.test(t)&&goldlog.isInternational()&&(t=t.replace(/^\/\/\w+\.mmstat/,"//gj.mmstat")),t};e.filterIntUrl=m,e.getPvUrl=function(t){t||(t={});var e,n,o=t.metaValue&&_(t.metaValue,t.metaName),a="";"isGifPath"===o?(e=/^\/\//.test(t.metaValue)?"":"//",a=e+t.metaValue):"isHostPath"===o&&(e=/^\/\//.test(t.metaValue)?"":"//",n=/\/$/.test(t.metaValue)?"":"/",a=e+t.metaValue+n+t.gifPath);var r;return a?r=a:(e=0===t.gifPath.indexOf("/")?t.gifPath:"/"+t.gifPath,r=t.url&&t.url.replace(/\/\w+\.gif/,e)),r},e.indexof=n(21).indexof,e.callable=r;var v="::-plain-::";e.mkPlainKey=function(){return v+Math.random()},e.s_plain_obj=v,e.mkPlainKeyForExparams=function(t){var e=t||v;return e+"exparams"},e.rndInt32=function(){return Math.round(2147483647*Math.random())},e.arr2param=i,e.arr2obj=s,e.obj2param=u,e.makeUrl=c,e.ifAdd=function(t,e){var n,o,a,r,i=e.length;for(n=0;n<i;n++)o=e[n],a=o[0],r=o[1],r&&t.push([a,r])},e.isStartWithProtocol=g.isStartWithProtocol,e.param2arr=function(t){for(var e,n=t.split("&"),o=0,a=n.length,r=[];o<a;o++)e=n[o].split("="),r.push([e.shift(),e.join("=")]);return r}},function(t,e,n){"use strict";function o(t,e){var n=e||"";if(t)try{n=decodeURIComponent(t)}catch(t){}return n}var a=n(17);e.tryToDecodeURIComponent=o,e.parseSemicolonContent=function(t,e,n){e=e||{};var r,i,s=t.split(";"),u=s.length;for(r=0;r<u;r++){i=s[r].split("=");var c=a.trim(i.slice(1).join("="));e[a.trim(i[0])||""]=n?c:o(c)}return e},e.nodeListToArray=function(t){var e,n;try{return e=[].slice.call(t)}catch(a){e=[],n=t.length;for(var o=0;o<n;o++)e.push(t[o]);return e}},e.nodeListToArray=function(t){var e,n;try{return e=[].slice.call(t)}catch(a){e=[],n=t.length;for(var o=0;o<n;o++)e.push(t[o]);return e}};var r={set:function(t,e){try{return localStorage.setItem(t,e),!0}catch(t){return!1}},get:function(t){return localStorage.getItem(t)},test:function(){var t="grey_test_key";try{return localStorage.setItem(t,1),localStorage.removeItem(t),!0}catch(t){return!1}},remove:function(t){localStorage.removeItem(t)}};e.store=r,e.getLsCna=function(t,e){var n="",o=r.get(t);if(o){var a=o.split("_")||[];n=e?a.length>1&&e===a[0]?a[1]:"":a.length>1?a[1]:""}return decodeURIComponent(n)},e.setLsCna=function(t,e,n){n&&r.set&&r.test()&&r.set(t,e+"_"+encodeURIComponent(n))},e.getUrl=function(t){var e=t||"//log.mmstat.com/eg.js";try{var n=goldlog.getMetaInfo("aplus-rhost-v"),o=/[[a-z|0-9\.]+[a-z|0-9]/,a=n.match(o);a&&a[0]&&(e=e.replace(o,a[0]))}catch(t){}return e}},function(t,e){"use strict";function n(t){return"string"==typeof t?t.replace(/^\s+|\s+$/g,""):""}e.trim=n},function(t,e,n){"use strict";var o=n(8),a=function(){var t=location.protocol;return"http:"!==t&&"https:"!==t&&(t="https:"),t};e.getProtocal=a,e.isStartWithProtocol=function(t){for(var e=["javascript:","tel:","sms:","mailto:","tmall://","#"],n=0,a=e.length;n<a;n++)if(o.isStartWith(t,e[n]))return!0;return!1}},function(t,e){"use strict";e.sleep=function(t,e){return setTimeout(function(){e()},t)}},function(t,e){"use strict";var n=function(){var t=!1;return"boolean"==typeof goldlog.aplusDebug&&(t=goldlog.aplusDebug),t};e.isDebugAplus=n;var o=function(t){t||(t={});var e=t.level||"warn";window.console&&window.console[e]&&window.console[e](t.msg)};e.logger=o},function(t,e){"use strict";e.indexof=function(t,e){var n=-1;try{n=t.indexOf(e)}catch(a){for(var o=0;o<t.length;o++)t[o]===e&&(n=o)}finally{return n}}},function(t,e){"use strict";var n=function(t){var e;try{window.goldlog||(window.goldlog={}),e=window.goldlog[t]}catch(t){e=""}finally{return e}};e.getGoldlogVal=n;var o=function(t,e){var n=!1;try{window.goldlog||(window.goldlog={}),t&&(window.goldlog[t]=e,n=!0)}catch(t){n=!1}finally{return n}};e.setGoldlogVal=o,e.getClientInfo=function(){return n("_aplus_client")||{}}},function(t,e,n){"use strict";function o(t){var e,n,o,a=t.length,r={};for(f._microscope_data=r,e=0;e<a;e++)n=t[e],"microscope-data"==l.tryToGetAttribute(n,"name")&&(o=l.tryToGetAttribute(n,"content"),u.parseSemicolonContent(o,r),f.is_head_has_meta_microscope_data=!0);f._microscope_data_params=u.obj2param(r),f.ms_data_page_id=r.pageId,f.ms_data_shop_id=r.shopId,f.ms_data_instance_id=r.siteInstanceId,f.ms_data_siteCategoryId=r.siteCategory,f.ms_prototype_id=r.prototypeId,f.site_instance_id_or_shop_id=f.ms_data_instance_id||f.ms_data_shop_id,f._atp_beacon_data={},f._atp_beacon_data_params=""}function a(t){var e,n=function(){var e;return document.querySelector&&(e=document.querySelector("meta[name=data-spm]")),c.each(t,function(t){"data-spm"===l.tryToGetAttribute(t,"name")&&(e=t)}),e},o=n();return o&&(e=l.tryToGetAttribute(o,"data-spm-protocol")),e}function r(t){var e=t.isonepage||"-1",n=e.split("|"),o=n[0],a=n[1]?n[1]:"";t.isonepage_data={isonepage:o,urlpagename:a}}function i(){var t=p.getMetaTags();o(t),c.each(t,function(t){var e=l.tryToGetAttribute(t,"name");/^aplus/.test(e)&&(f[e]=p.getMetaCnt(e))}),c.each(d,function(t){f[t]=p.getMetaCnt(t)}),f.spm_protocol=a(t);var e,n,i=["aplus-rate-ahot"],s=i.length;for(e=0;e<s;e++)n=i[e],f[n]=parseFloat(f[n]);return r(f),h=f||{},f}function s(){return h||i()}var u=n(15),c=n(8),l=n(24),p=n(25),g=n(26),f={},d=["ahot-aplus","isonepage","spm-id","data-spm","microscope-data"],h={};e.setMetaInfo=function(t,e){return h||(h={}),h[t]=e,!0};var _=function(t){return h||(h={}),h[t]||""};e.getMetaInfo=_,e.getInfo=i,e.qGet=s,e.appendMetaInfo=function(t,e){if(t&&e){var n,o=function(n){try{var o="string"==typeof e?JSON.parse(e):e;goldlog.setMetaInfo(t,c.assign(n,o))}catch(t){}},a=function(n){try{var o="string"==typeof e?JSON.parse(e):e;goldlog.setMetaInfo(t,n.concat(o))}catch(t){}},r=function(t){return"EXPARAMS"===t?g.getExparamsInfos("",t):t?t.split("&"):[]},i=function(n){try{var o=r(n),a=r(e);goldlog.setMetaInfo(t,o.concat(a).join("&"))}catch(t){}},s=function(t){t.constructor===Array?a(t):o(t)},u=goldlog.getMetaInfo(t);if("aplus-exinfo"===t&&(i(u),n=!0),u)if("object"==typeof u)s(u),n=!0;else try{var l=JSON.parse(u);"object"==typeof l&&(s(l),n=!0)}catch(t){}n||goldlog.setMetaInfo(t,e)}}},function(t,e,n){"use strict";function o(t,e){return t&&t.getAttribute?t.getAttribute(e)||"":""}function a(t,e,n){if(t&&t.setAttribute)try{t.setAttribute(e,n)}catch(t){}}function r(t,e){if(t&&t.removeAttribute)try{t.removeAttribute(e)}catch(n){a(t,e,"")}}function i(t,e,n){var o="script",a=g.createElement(o);a.type="text/javascript",a.async=!0;var r="https:"==location.protocol?e||t:t;0===r.indexOf("//")&&(r=u.getProtocal()+r),a.src=r,n&&(a.id=n);var i=g.getElementsByTagName(o)[0];s=s||g.getElementsByTagName("head")[0],i?i.parentNode.insertBefore(a,i):s&&s.appendChild(a)}var s,u=n(18),c=n(17),l=n(8),p=n(20),g=document;e.tryToGetAttribute=o,e.tryToSetAttribute=a,e.tryToRemoveAttribute=r,e.addScript=i,e.loadScript=function(t,e){function n(t){o.onreadystatechange=o.onload=o.onerror=null,o=null,e(t)}var o=g.createElement("script");if(s=s||g.getElementsByTagName("head")[0],o.async=!0,"onload"in o)o.onload=n;else{var a=function(){/loaded|complete/.test(o.readyState)&&n()};o.onreadystatechange=a,a()}o.onerror=function(t){n(t)},o.src=t,s.appendChild(o)},e.isTouch=function(){return"ontouchend"in document.createElement("div")},e.tryToGetHref=function(t){var e;try{e=c.trim(t.getAttribute("href",2))}catch(t){}return e||""};var f=function(){var t=goldlog&&goldlog._$?goldlog._$:{},e=t.meta_info||{};return e["aplus-exparams"]||""};e.getExParamsFromMeta=f,e.getExParams=function(t){var e=g.getElementById("beacon-aplus")||g.getElementById("tb-beacon-aplus"),n=o(e,"exparams"),a=d(n,f(),t)||"";return a&&a.replace(/&amp;/g,"&").replace(/\buserid=/,"uidaplus=")};var d=function(t,e,n){var o="aplus&sidx=aplusSidex",a=t||o;try{if(e){var r=n.param2obj(e),i=["aplus","cna","spm-cnt","spm-url","spm-pre","logtype","pre","uidaplus","asid","sidx","trid","gokey"];l.each(i,function(t){r.hasOwnProperty(t)&&(p.logger({msg:"Can not inject keywords: "+t}),delete r[t])}),delete r[""];var s="";if(t){var u=t.match(/aplus&/).index,c=u>0?n.param2obj(t.substring(0,u)):{};delete c[""],s=n.obj2param(l.assign(c,r))+"&"+t.substring(u,t.length)}else s=n.obj2param(r)+"&"+o;return s}return a}catch(t){return a}};e.mergeExparams=d},function(t,e,n){"use strict";function o(t){return i=i||document.getElementsByTagName("head")[0],s&&!t?s:i?s=i.getElementsByTagName("meta"):[]}function a(t){var e,n,a,r=o(),i=r.length;for(e=0;e<i;e++)n=r[e],u.tryToGetAttribute(n,"name")===t&&(a=u.tryToGetAttribute(n,"content"));return a||""}function r(t){var e={isonepage:"-1",urlpagename:""},n=t.qGet();if(n&&n.hasOwnProperty("isonepage_data"))e.isonepage=n.isonepage_data.isonepage,e.urlpagename=n.isonepage_data.urlpagename;else{var o=a("isonepage")||"-1",r=o.split("|");e.isonepage=r[0],e.urlpagename=r[1]?r[1]:""}return e}var i,s,u=n(24);e.getMetaTags=o,e.getMetaCnt=a,e.getOnePageInfo=r},function(t,e,n){"use strict";var o=n(15),a=n(24),r=n(21);e.getExparamsInfos=function(t,e){var n=[],i=t||["uidaplus","pc_i","pu_i"],s=a.getExParams(o)||"";s=s.replace(/&aplus&/,"&");for(var u=o.param2arr(s)||[],c=function(t){return r.indexof(i,t)>-1},l=0;l<u.length;l++){var p=u[l],g=p[0]||"",f=p[1]||"";g&&f&&("EXPARAMS"===e||c(g))&&n.push(g+"="+f)}return n}},function(t,e,n){"use strict";var o=n(9),a=n(24),r=n(16),i=n(28),s=n(18),u=n(22),c=n(3);t.exports=function(){return{init:function(t){this.options=t;var e=this.options.context.etag||{};this.cna=e.cna||o.getCookie("cna"),this.setTag(0),this.setStag(-1),this.setLsTag("-1"),this.setEtag(this.cna||""),this.requesting=!1,this.today=i.getFormatDate()},setLsTag:function(t){this.lstag=t,this.options.context.etag.lstag=t},setTag:function(t){this.tag=t,this.options.context.etag.tag=t},setStag:function(t){this.stag=t,this.options.context.etag.stag=t},setEtag:function(t){this.etag=t,this.options.context.etag.cna=t,o.getCookie("cna")!==t&&o.setCookie("cna",t)},setLscnaStatus:function(t){this.options.context.etag.lscnastatus=t},getUrl:function(){var t=this.options.context.etag||{};return r.getUrl(t.egUrl||"//log.mmstat.com/eg.js")},run:function(t,e){var n=this;if(n.cna)return void n.setTag(1);var o=null,i=this.getUrl();if(0===i.indexOf("//")){var l=s.getProtocal();i=l+i}n.requesting=!0;var p=function(){setTimeout(function(){e()},20),clearTimeout(o)};return a.loadScript(i,function(t){var e,o;if(t&&"error"===t.type?n.setStag(-3):(e=u.getGoldlogVal("Etag"),e&&n.setEtag(e),o=u.getGoldlogVal("stag"),"undefined"!=typeof o&&n.setStag(o)),n.requesting){if(2===o||4===o){var a=r.getLsCna(c.LS_CNA_KEY);a?(n.setLsTag(1),n.setEtag(a)):(n.setLsTag(0),r.setLsCna(c.LS_CNA_KEY,n.today,e))}p()}}),o=setTimeout(function(){n.requesting=!1,n.setStag(-2),e()},1500),2e3}}}},function(t,e){"use strict";function n(t,e,n){var o=""+Math.abs(t),a=e-o.length,r=t>=0;return(r?n?"+":"":"-")+Math.pow(10,Math.max(0,a)).toString().substr(1)+o}e.getFormatDate=function(t){var e=new Date;try{return[e.getFullYear(),n(e.getMonth()+1,2,0),n(e.getDate(),2,0)].join(t||"")}catch(t){return""}}},function(t,e,n){"use strict";var o=n(16),a=n(24),r=n(3),i=n(28),s=o.store||{};t.exports=function(){return{init:function(t){this.options=t,this.today=i.getFormatDate()},getUrl:function(){var t=this.options.context.etag||{};return o.getUrl(t.egUrl||"//log.mmstat.com/eg.js")},run:function(){var t=this;if(s.test()){var e=o.getLsCna(r.LS_CNA_KEY,this.today);e||setTimeout(function(){a.loadScript(t.getUrl(),function(e){e&&"error"!==e.type&&o.setLsCna(r.LS_CNA_KEY,t.today,goldlog.Etag)})},1e3)}}}}},function(t,e,n){"use strict";e.plugins_pv=[{name:"etag",enable:!0,path:n(31)},{name:"when_to_sendpv",enable:!0,path:n(33)},{name:"where_to_sendlog_ut",enable:!0,path:n(34)},{name:"is_single",enable:!0,path:n(36)},{name:"what_to_pvhash",enable:!0,path:n(39)},{name:"what_to_sendpv",enable:!0,path:n(40)},{name:"what_to_sendpv_userdata",enable:!0,path:n(44),deps:["what_to_sendpv"]},{name:"what_to_sendpv_etag",enable:!0,path:n(49),deps:["etag","what_to_sendpv"]},{name:"what_to_sendpv_ut",enable:!0,path:n(50),deps:["where_to_sendlog_ut","is_single"]},{name:"what_to_pv_slog",enable:!0,path:n(51),deps:["what_to_sendpv"]},{name:"can_to_sendpv",enable:!0,path:n(52)},{name:"where_to_sendpv",enable:!0,path:n(14),deps:["is_single"]},{name:"do_sendpv",enable:!0,path:n(53),deps:["is_single","what_to_sendpv","where_to_sendpv"]},{name:"do_sendpv_ut",enable:!0,path:n(54),deps:["what_to_sendpv_ut","where_to_sendlog_ut"]},{name:"after_pv",enable:!0,path:n(55)}]},function(t,e,n){"use strict";var o=n(32);t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=this;o.doSubOnceMsg("aplusInitContext",function(e){e.etag&&(t.options.context.etag=e.etag)})}}}},function(t,e){"use strict";var n=function(){var t=window.goldlog||{},e=t.aplus_pubsub||{},n="function"==typeof e.publish;return n?e:""};e.doPubMsg=function(t){var e=n();e&&e.publish.apply(e,t)},e.doCachePubs=function(t){var e=n();e&&"function"==typeof e.cachePubs&&e.cachePubs.apply(e,t)},e.doSubMsg=function(t,e){var o=n();o&&"function"==typeof o.subscribe&&o.subscribe(t,e)},e.doSubOnceMsg=function(t,e){var o=n();o&&"function"==typeof o.subscribeOnce&&o.subscribeOnce(t,e)}},function(t,e,n){"use strict";var o=n(22),a=n(19),r=n(23);t.exports=function(){return{init:function(t){this.options=t},getMetaInfo:function(){var t=o.getGoldlogVal("_$")||{},e=t.meta_info||r.getInfo();return e},getAplusWaiting:function(){var t=this.getMetaInfo()||{};return t["aplus-waiting"]},run:function(t,e){var n=this.options.config||{},o=this.getAplusWaiting();if(o&&n.is_auto)switch(o=this.getAplusWaiting()+"",this.options.context.when_to_sendpv={aplusWaiting:o},o){case"MAN":return"done";case"1":return this.options.context.when_to_sendpv.isWait=!0,a.sleep(6e3,function(){e()}),6e3;default:var r=1*o;if(r+""!="NaN")return this.options.context.when_to_sendpv.isWait=!0,a.sleep(r,function(){e()}),r}}}}},function(t,e,n){"use strict";var o=n(35);t.exports=function(){return{init:function(t){this.options=t},getAplusToUT:function(){return{toUT2:o.getAplusToUT("toUT2"),toUT:o.getAplusToUT("toUT")}},run:function(){var t=this.getAplusToUT();this.options.context.where_to_sendlog_ut.aplusToUT=t}}}},function(t,e){"use strict";var n=navigator.userAgent,o=/WindVane/i.test(n);e.is_WindVane=o;var a=function(){var t=goldlog.getMetaInfo("aplus_chnl");return!(!t||!t.isAvailable||"function"!=typeof t.toUT2&&"function"!=typeof t.toUT)&&t};e.isAplusChnl=a,e.getAplusToUT=function(t){var e={},n=a();if("object"==typeof n)e.bridgeName=n.bridgeName||"customBridge",e.isAvailable=n.isAvailable,e.toUT2=n.toUT2||n.toUT;else{var r=window.WindVane||{};if(o&&r&&r.isAvailable&&"function"==typeof r.call){var i=t||"toUT";e={bridgeName:"WindVane",isAvailable:!0,toUT2:function(t,e,n,o){return r.call("WVTBUserTrack",i,t,e,n,o)}}}}return e}},function(t,e,n){"use strict";var o=n(22),a=n(37),r=n(3);t.exports=function(){return{init:function(t){this.options=t,this._$=o.getGoldlogVal("_$")||{},this.isBoth="1"===this._$.meta_info["aplus-both-request"],this.is_WindVane=this._$.is_WindVane},isSingle_pv:function(t){return t?!this.isBoth:!(!this.is_WindVane||!a.isSingleUaVersion()||this.isBoth)},isSingle_hjlj:function(t,e){return e?!this.isBoth:!(!this.is_WindVane||!a.isSingleSendLog(t)||this.isBoth)},isSingle_uhjlj:function(t,e){return(!t||!/^\/aplus\.99\.(\d)+$/.test(t.logkey))&&(e?!this.isBoth:!(!(this.is_WindVane&&t&&t.logkey&&a.isSingleUaVersion())||this.isBoth))},run:function(){var t=this.options.context||{},e=this.options.config||{},n=t.where_to_sendlog_ut.aplusToUT||{},a=n.toUT||{},i=n.toUT2||{},s=o.getGoldlogVal("isUT4Aplus"),u=!!(a.isAvailable||i.isAvailable||s),c=t.userdata||{},l=!!t.is_single;switch(e.recordType){case r.recordTypes.uhjlj:l=this.isSingle_uhjlj(c,s);break;case r.recordTypes.hjlj:l=this.isSingle_hjlj(c,s);break;case r.recordTypes.pv:l=this.isSingle_pv(s);break;default:l=this.isSingle_pv(s)}this.options.context.is_single=u&&l}}}},function(t,e,n){"use strict";var o=n(38),a=function(t){var e=t.logkey.toLowerCase();0===e.indexOf("/")&&(e=e.substr(1));var n=t.gmkey.toUpperCase();switch(n){case"EXP":return"2201";case"CLK":return"2101";case"SLD":return"19999";case"OTHER":default:return"19999"}},r=function(){var t=!1;return t||goldlog.isUT4Aplus||o.webviewIsAbove({version_ios_tb:[5,11,7],version_ios_tm:[5,24,1],version_android_tb:[5,11,7],version_android_tm:[5,24,1]})};e.isSingleUaVersion=r,e.isSingleSendLog=function(t){return(!t||!/^\/fsp\.1\.1$/.test(t.logkey))&&!!(t&&t.logkey&&r())},e.getFunctypeValue=function(t){return e.isSingleSendLog(t)?a(t):"2101"},e.getFunctypeValue2=function(t){return a(t)}},function(t,e){"use strict";var n=function(t){var e=[0,0,0];try{if(t){var n=t[1],o=n.split(".");if(o.length>2)for(var a=0;a<o.length;)e[a]=parseInt(o[a]),a++}}catch(t){e=[0,0,0]}finally{return e}};e.parseVersion=n;var o=function(t,e){var n=!1;try{var o=t[0]>e[0],a=t[1]>e[1],r=t[2]>e[2],i=t[0]===e[0],s=t[1]===e[1],u=t[2]===e[2];n=!!o||(!(!i||!a)||(!!(i&&s&&r)||!!(i&&s&&u)))}catch(t){n=!1}finally{return n}};e.isAboveVersion=o,e.webviewIsAbove=function(t,e){var a=!1;try{e||(e=navigator.userAgent);var r=e.match(/AliApp\(TB\/(\d+[._]\d+[._]\d+)/i),i=n(r),s=e.match(/AliApp\(TM\/(\d+[._]\d+[._]\d+)/i),u=n(s),c=/iPhone|iPad|iPod|ios/i.test(e),l=/android/i.test(e);c?r&&i?a=o(i,t.version_ios_tb):s&&u&&(a=o(u,t.version_ios_tm)):l&&(r&&i?a=o(i,t.version_android_tb):s&&u&&(a=o(u,t.version_android_tm)))}catch(t){a=!1}return a}},function(t,e,n){"use strict";var o=n(22);t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=this.options.context.what_to_pvhash||{},e=o.getGoldlogVal("_$")||{},n=e.meta_info||{},a=n["aplus-pvhash"]||"",r=[];"1"===a&&(r=["_aqx_uri",encodeURIComponent(location.href)]),t.hash=r,this.options.context.what_to_pvhash=t}}}},function(t,e,n){"use strict";var o=n(15),a=n(8),r=n(24),i=n(22),s=n(9),u=n(41),c=n(42),l=n(43);t.exports=function(){return a.assign(l,{init:function(t){this.options=t,this.cookie_data||(this.cookie_data=s.getData()),this.client_info||(this.client_info=i.getClientInfo()||{});var e=location.hash;e&&0===e.indexOf("#")&&(e=e.substr(1)),this.loc_hash=e},getExParams:function(){var t=window,e=document,n=[],i=parent!==t.self,s=e.getElementById("beacon-aplus")||e.getElementById("tb-beacon-aplus"),c=r.tryToGetAttribute(s,"exparams"),l=r.mergeExparams(c,r.getExParamsFromMeta(),o)||"";l=l.replace(/&amp;/g,"&");var p,g,f=["taobao.com","tmall.com","etao.com","hitao.com","taohua.com","juhuasuan.com","alimama.com"];if(i){for(g=f.length,p=0;p<g;p++)if(o.isContain(location.hostname,f[p]))return n.push([o.mkPlainKeyForExparams(),l]),n;l=l.replace(/\buserid=\w*&?/,"")}l=l.replace(/\buserid=/,"uidaplus="),l&&n.push([o.mkPlainKeyForExparams(),l]);var d=a.makeCacheNum();return u.updateKey(n,"cache",d),n},getExtra:function(){var t=[],e=i.getGoldlogVal("_$")||{},n=e.meta_info||{},a=this.cookie_data||{},r=this.getClientInfo(!0)||[];return o.ifAdd(t,r),o.ifAdd(t,[["thw",a.thw],["bucket_id",c.getBucketId(n)],["urlokey",this.loc_hash],["wm_instanceid",n.ms_data_instance_id]]),t}})}},function(t,e){"use strict";function n(t,e,n){r(t,"spm-cnt",function(t){var o=t.split(".");return o[0]=goldlog.spm_ab[0],o[1]=goldlog.spm_ab[1],e?o[1]=o[1].split("/")[0]+"/"+e:o[1]=o[1].split("/")[0],n&&(o[4]=n),o.join(".")})}function o(t,e){var n=window.g_SPM&&g_SPM._current_spm;n&&r(t,"spm-url",function(){return[n.a,n.b,n.c,n.d].join(".")+(e?"."+e:"")},"spm-cnt")}function a(t,e){var n,o,a,r=-1;for(n=0,o=t.length;n<o;n++)if(a=t[n],a[0]===e){r=n;break}r>=0&&t.splice(r,1)}function r(t,e,n,o){var a,r,i=t.length,s=-1,u="function"==typeof n;for(a=0;a<i;a++){if(r=t[a],r[0]===e)return void(u?r[1]=n(r[1]):r[1]=n);o&&r[0]===o&&(s=a)}o&&(u&&(n=n()),s>-1?t.splice(s,0,[e,n]):t.push([e,n]))}t.exports={updateSPMCnt:n,updateSPMUrl:o,updateKey:r,removeKey:a}},function(t,e,n){"use strict";function o(t,e){var n,o=2146271213;for(n=0;n<t.length;n++)o=(o<<5)+o+t.charCodeAt(n);return(65535&o)%e}function a(t){var e,n=r.getCookie("t");return"3"!=t.ms_prototype_id&&"5"!=t.ms_prototype_id||(e=n?o(n,20):""),e}var r=n(9);e.getBucketId=a},function(t,e,n){"use strict";var o=n(15),a=n(8),r=n(22),i=n(35),s=n(9),u=n(3);t.exports={init:function(t){this.options=t,this.cookie_data||(this.cookie_data=s.getData())},getBasicParams:function(){var t=document,e=r.getGoldlogVal("_$")||{},n=e.spm||{},i=e.meta_info||{},u=i["aplus-ifr-pv"]+""=="1",c=o.is_in_iframe&&!u?0:1,l=[["logtype",c],["title",t.title],["pre",e.page_referrer],["cache",a.makeCacheNum()],["scr",screen.width+"x"+screen.height]],p=this.cookie_data||{},g=this.options.context||{},f=g.etag||{},d=f.cna||p.cna||s.getCookie("cna");d&&l.push([o.mkPlainKey(),"cna="+d]),p.tracknick&&l.push([o.mkPlainKey(),"nick="+p.tracknick]);var h=n.spm_url||"";return o.ifAdd(l,[["wm_pageid",i.ms_data_page_id],["wm_prototypeid",i.ms_prototype_id],["wm_sid",i.ms_data_shop_id],["spm-url",h],["spm-pre",n.spm_pre],["spm-cnt",n.spm_cnt],["cnaui",p.cnaui]]),l},getExParams:function(){return[]},getExtra:function(){return[]},getClientInfo:function(t){var e=[],n=r.getGoldlogVal("_$")||{},a=this.client_info||{},s=a.ua_info||{};if(t||!i.is_WindVane&&!i.isAplusChnl()){for(var c,l=[],p=["p","o","b","s","w","wx","ism"],g=0;c=p[g++];)s[c]&&l.push([c,s[c]]);o.ifAdd(e,l)}o.ifAdd(e,[["lver",goldlog.lver||u.lver],["jsver",n.script_name||u.script_name],["pver",goldlog.aplus_cplugin_ver]]);var f=this.options.config||{},d=f.is_auto;return d||o.ifAdd(e,[["mansndlog",1]]),e},processLodashDollar:function(){var t=r.getGoldlogVal("_$")||{};t.page_url!==location.href&&(t.page_referrer=t.page_url,t.page_url=location.href),r.setGoldlogVal("_$",t)},getLsParams:function(){var t=r.getGoldlogVal("_$")||{},e=[];return t.lsparams&&t.lsparams.spm_id&&(e.push(["lsparams",t.lsparams.spm_id]),e.push(["lsparams_pre",t.lsparams.current_url])),e},run:function(){var t=this.getBasicParams()||[],e=this.getExParams()||[],n=this.getExtra()||[];this.processLodashDollar();var o=this.getLsParams()||[],a=[].concat(t,e,n,o);this.options.context.what_to_sendpv.pvdata=a,this.options.context.what_to_sendpv.exparams=e}}},function(t,e,n){"use strict";var o=n(15),a=n(22),r=n(41),i=n(9),s=n(45);t.exports=function(){return{init:function(t){this.options=t},getPageId:function(){var t=this.options.config||{},e=this.options.context||{},n=e.userdata||{};
return t.page_id||t.pageid||t.pageId||n.page_id},getPageInfo:function(){var t;try{var e=top.location!==self.location;e&&void 0!==window.innerWidth&&(t={width:window.innerWidth,height:window.innerHeight})}catch(t){}return t},getUserdata:function(){var t=a.getGoldlogVal("_$")||{},e=t.spm||{},n=this.options.context||{},r=n.userdata||{},u=this.options.config||{},c=[];if(u&&!u.is_auto){u.gokey&&c.push([o.mkPlainKey(),u.gokey]);var l=e.data.b;if(l){var p=this.getPageId();l=p?l.split("/")[0]+"/"+p:l.split("/")[0],s.setB(l);var g=e.spm_cnt.split(".");g&&g.length>2&&(g[1]=l,e.spm_cnt=g.join("."))}}var f=function(t){if("object"==typeof t)for(var e in t)"object"!=typeof t[e]&&"function"!=typeof t[e]&&c.push([e,t[e]])};f(goldlog.getMetaInfo("aplus-cpvdata")),f(r);var d=i.getCookie("workno")||i.getCookie("emplId");d&&c.push(["workno",d]);var h=i.getHng();h&&c.push(["_hng",i.getHng()]);var _=this.getPageInfo();return _&&(c.push(["_pw",_.width]),c.push(["_ph",_.height])),c},processLodashDollar:function(){var t=this.options.config||{},e=a.getGoldlogVal("_$")||{};t&&t.referrer&&(e.page_referrer=t.referrer),a.setGoldlogVal("_$",e)},updatePre:function(t){var e=a.getGoldlogVal("_$")||{};return e.page_referrer&&r.updateKey(t,"pre",e.page_referrer),t},run:function(){var t=this.options.context.what_to_sendpv.pvdata,e=this.getUserdata();this.processLodashDollar();var n=t,o=this.options.context.what_to_pvhash.hash;o&&o.length>0&&n.push(o),n=n.concat(e),n=this.updatePre(n);var a=this.getPageId();a&&r.updateSPMCnt(n,a),this.options.context.what_to_sendpv.pvdata=n,this.options.context.userdata=e}}}},function(t,e,n){"use strict";function o(){if(!s.data.a||!s.data.b){var t=r._SPM_a,e=r._SPM_b;if(t&&e)return t=t.replace(/^{(\w+\/)}$/g,"$1"),e=e.replace(/^{(\w+\/)}$/g,"$1"),s.is_wh_in_page=!0,void c.setAB(t,e);var n=goldlog._$.meta_info;t=n["data-spm"]||n["spm-id"]||"0";var o=t.split(".");o.length>1&&(t=o[0],e=o[1]),c.setA(t),e&&c.setB(e);var a=i.getElementsByTagName("body");a=a&&a.length?a[0]:null,a&&(e=l.tryToGetAttribute(a,"data-spm"),e?c.setB(e):1===o.length&&c.setAB("0","0"))}}function a(){var t=s.data.a,e=s.data.b;t&&e&&(goldlog.spm_ab=[t,e])}var r=window,i=document,s={},u={};s.data=u;var c={},l=n(24),p=n(46),g=location.href,f=n(47).getRefer(),d=n(3);c.setA=function(t){s.data.a=t,a()},c.setB=function(t){s.data.b=t,a()},c.setAB=function(t,e){s.data.a=t,s.data.b=e,a()};var h=p.getSPMFromUrl,_=function(){var t=d.utilPvid.makePVId();return d.mustSpmE?t||goldlog.pvid||"":t||""},m=function(t,e){var n=t.goldlog||window.goldlog||{},a=n.meta_info||{};s.meta_protocol=a.spm_protocol;var r,i=n.spm_ab||[],u=i[0]||"0",c=i[1]||"0";"0"===u&&"0"===c&&(o(),u=s.data.a||"0",c=s.data.b||"0"),r=[s.data.a,s.data.b].join("."),s.spm_cnt=(r||"0.0")+".0.0";var l=t.send_pv_count>0?_():n.pvid;l&&(s.spm_cnt+="."+l),n._$.spm=s,"function"==typeof e&&e(l)};c.spaInit=function(t,e,n,o){var a="function"==typeof o?o:function(){},r=s.spm_url,i=window.g_SPM||{},u=t._$||{},c=u.send_pv_count;m({goldlog:t,meta_info:e,send_pv_count:c},function(t){s.spm_cnt=s.data.a+"."+s.data.b+".0.0"+(t?"."+t:"");var n=e["aplus-spm-fixed"];if("1"!==n){s.spm_pre=h(f),s.spm_url=h(location.href);var o=i._current_spm||{};o&&o.a&&"0"!==o.a&&o.b&&"0"!==o.b&&(s.spm_url=[o.a,o.b,o.c,o.d,o.e].join("."),s.spm_pre=r)}a()})},c.init=function(t,e,n){s.spm_url=h(g),s.spm_pre=h(f),m({goldlog:t,meta_info:e},function(){"function"==typeof n&&n()})},c.resetSpmCntPvid=function(){var t=goldlog.spm_ab;if(t&&2===t.length){var e=t.join(".")+".0.0",n=_();n&&(e=e+"."+n),s.spm_cnt=e,s.spm_url=e,goldlog._$.spm=s}},t.exports=c},function(t,e){"use strict";function n(t,e){if(!t||!e)return"";var n,o="";try{var a=new RegExp(t+"=([^&|#|?|/]+)");if("spm"===t||"scm"===t){var r=new RegExp("\\?.*"+t+"=([\\w\\.\\-\\*/]+)"),i=e.match(a),s=e.match(r),u=i&&2===i.length?i[1]:"",c=s&&2===s.length?s[1]:"";o=u>c?u:c,o=decodeURIComponent(o)}else n=e.match(a),o=n&&2===n.length?n[1]:""}catch(t){}finally{return o}}e.getParamFromUrl=n,e.getSPMFromUrl=function(t){return n("spm",t)}},function(t,e,n){"use strict";var o=null,a=n(48).nameStorage,r=n(4);e.getRefer=function(){if(null!==o)return o;var t=r.KEY||{},e=t.NAME_STORAGE||{};return o=document.referrer||a.getItem(e.REFERRER)||""}},function(t,e){"use strict";var n=function(){function t(){var t,e=[],r=!0;for(var l in p)p.hasOwnProperty(l)&&(r=!1,t=p[l]||"",e.push(c(l)+s+c(t)));n.name=r?o:a+c(o)+i+e.join(u)}function e(t,e,n){t&&(t.addEventListener?t.addEventListener(e,n,!1):t.attachEvent&&t.attachEvent("on"+e,function(e){n.call(t,e)}))}var n=window;if(n.nameStorage)return n.nameStorage;var o,a="nameStorage:",r=/^([^=]+)(?:=(.*))?$/,i="?",s="=",u="&",c=encodeURIComponent,l=decodeURIComponent,p={},g={};return function(t){if(t&&0===t.indexOf(a)){var e=t.split(/[:?]/);e.shift(),o=l(e.shift())||"";for(var n,i,s,c=e.join(""),g=c.split(u),f=0,d=g.length;f<d;f++)n=g[f].match(r),n&&n[1]&&(i=l(n[1]),s=l(n[2])||"",p[i]=s)}else o=t||""}(n.name),g.setItem=function(e,n){e&&"undefined"!=typeof n&&(p[e]=String(n),t())},g.getItem=function(t){return p.hasOwnProperty(t)?p[t]:null},g.removeItem=function(e){p.hasOwnProperty(e)&&(p[e]=null,delete p[e],t())},g.clear=function(){p={},t()},g.valueOf=function(){return p},g.toString=function(){var t=n.name;return 0===t.indexOf(a)?t:a+t},e(n,"beforeunload",function(){t()}),g}();e.nameStorage=n},function(t,e,n){"use strict";var o=n(41);t.exports=function(){return{init:function(t){this.options=t},updateBasicParams:function(){var t=this.options.context.what_to_sendpv.pvdata||[],e=this.options.context.etag||{};return e.cna&&(o.updateKey(t,"cna",e.cna),this.options.context.what_to_sendpv.pvdata=t),t},addTagParams:function(){var t=this.options.context.what_to_sendpv.pvdata||[],e=this.options.context.etag||{},n=[];(e.tag||0===e.tag)&&n.push(["tag",e.tag]),(e.stag||0===e.stag)&&n.push(["stag",e.stag]),(e.lstag||0===e.lstag)&&n.push(["lstag",e.lstag]),n.length>0&&(this.options.context.what_to_sendpv.pvdata=t.concat(n))},run:function(){this.updateBasicParams(),this.addTagParams()}}}},function(t,e,n){"use strict";function o(t){var e,n,o,a,i=[],s={};for(e=t.length-1;e>=0;e--)n=t[e],o=n[0],o&&o.indexOf(r.s_plain_obj)==-1&&s.hasOwnProperty(o)||(a=n[1],("aplus"==o||a)&&(i.unshift([o,a]),s[o]=1));return i}function a(t){var e,n,o,a,s=[],u={logtype:!0,cache:!0,scr:!0,"spm-cnt":!0};for(e=t.length-1;e>=0;e--)if(n=t[e],o=n[0],a=n[1],!(i.isStartWith(o,r.s_plain_obj)&&!i.isStartWith(o,r.mkPlainKeyForExparams())||u[o]))if(i.isStartWith(o,r.mkPlainKeyForExparams())){var c=r.param2arr(a);if("object"==typeof c&&c.length>0)for(var l=c.length-1;l>=0;l--){var p=c[l];p&&p[1]&&s.unshift([p[0],p[1]])}}else s.unshift([o,a]);return s}var r=n(15),i=n(8),s=n(22),u=n(25),c=n(23),l=n(3),p=n(9);t.exports=function(){return{init:function(t){this.options=t},getToUtData:function(t,e){var n,i=s.getGoldlogVal("_$")||{},g=i.spm||{},f=a(o(t)),d={};try{var h=r.arr2obj(f);h._toUT=1,h._bridgeName=e.bridgeName||"",n=JSON.stringify(h)}catch(t){n='{"_toUT":1}'}var _=u.getOnePageInfo(c);return d.functype="2001",d.urlpagename=_.urlpagename,d.url=location.href,d.spmcnt=g.spm_cnt||"",d.spmpre=g.spm_pre||"",d.lzsid="",d.cna=p.getCookie("cna"),d.extendargs=n,d.isonepage=_.isonepage,d.version=l.toUtVersion,d.lver=goldlog.lver||l.lver,d.jsver=l.script_name,d},run:function(){var t=this.options.context||{},e=t.what_to_sendpv||{},n=e.pvdata||[],o=t.what_to_sendpv_ut||{},a=t.where_to_sendlog_ut||{},r=a.aplusToUT||{},i=r.toUT||{};(i&&i.isAvailable&&"function"==typeof i.toUT2||goldlog.isUT4Aplus)&&(o.pvdataToUt=this.getToUtData(n,i),this.options.context.what_to_sendpv_ut=o)}}}},function(t,e){"use strict";t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=this.options.context||{},e=t.is_single?"1":"0";t.what_to_sendpv_ut2.pvdataToUt._slog=e,t.what_to_sendpv_ut.pvdataToUt._slog=e,t.what_to_sendpv.pvdata.push(["_slog",e])}}}},function(t,e,n){"use strict";var o=n(22);t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=o.getGoldlogVal("_$")||{},e=this.options.context.can_to_sendpv||{},n=t.send_pv_count||0,a=this.options.config||{};return a.is_auto&&n>0?"done":(e.flag="YES",this.options.context.can_to_sendpv=e,t.send_pv_count=++n,void o.setGoldlogVal("_$",t))}}}},function(t,e,n){"use strict";var o=n(22),a=n(15);t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=this.options.context||{},e=!!t.is_single;if(!e){var n=t.what_to_sendpv||{},r=t.where_to_sendpv||{},i=n.pvdata||[],s=goldlog.send(r.url,a.arr2obj(i));o.setGoldlogVal("req",s)}}}}},function(t,e){"use strict";t.exports=function(){return{init:function(t){this.options=t},run:function(t,e){var n=this,o=this.options.context||{},a=o.what_to_sendpv_ut||{},r=o.where_to_sendlog_ut||{},i=a.pvdataToUt||{},s=r.aplusToUT||{},u=s.toUT;if(goldlog.isUT4Aplus&&"UT4Aplus"===goldlog.getMetaInfo("aplus-toUT"))return s.toutflag="toUT",void(n.options.context.what_to_sendpv_ut.isSuccess=!0);if(u&&"function"==typeof u.toUT2&&u.isAvailable)try{s.toutflag="toUT",u.toUT2(i,function(){n.options.context.what_to_sendpv_ut.isSuccess=!0,e()},function(t){n.options.context.what_to_sendpv_ut.errorMsg=t,e()},2e3)}catch(t){e()}finally{return"pause"}}}}},function(t,e,n){"use strict";var o=n(32),a=n(22);t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=goldlog._$||{},e=this.options.context||{};a.setGoldlogVal("pv_context",e);var n=goldlog.spm_ab||[],r=n.join("."),i=t.send_pv_count,s={cna:e.etag.cna,count:i,spmab_pre:goldlog.spmab_pre};o.doPubMsg(["sendPV","complete",r,s]),o.doCachePubs(["sendPV","complete",r,s])}}}},function(t,e,n){"use strict";e.plugins_prepv=[{name:"etag",enable:!0,path:n(31)},{name:"what_to_hjlj_exinfo",enable:!0,path:n(57)},{name:"what_to_prepv",enable:!0,path:n(58),deps:["what_to_hjlj_exinfo"]},{name:"where_to_prepv",enable:!0,path:n(59),deps:["what_to_prepv"]},{name:"do_sendprepv",enable:!0,path:n(60),deps:["what_to_prepv","where_to_prepv"]}]},function(t,e,n){"use strict";var o=n(15),a=n(24),r=n(22),i=n(22),s=n(21),u=n(9);t.exports=function(){return{init:function(t){this.options=t},getCookieUserInfo:function(){var t=[],e=u.getCookie("workno")||u.getCookie("emplId");e&&t.push("workno="+e);var n=u.getHng();return n&&t.push("_hng="+u.getHng()),t},filterExinfo:function(t){var e="";try{t&&("string"==typeof t?e=t.replace(/&amp;/g,"&").replace(/\buserid=/,"uidaplus=").replace(/&aplus&/,"&"):"object"==typeof t&&(e=o.obj2param(t,!0)))}catch(t){e=t.message?t.message:""}return e},getExparamsFlag:function(){var t=this.options.context||{},e=t.what_to_hjlj_exinfo||{};return e.EXPARAMS_FLAG||"EXPARAMS"},getCustomExParams:function(t){var e="";return t!==this.getExparamsFlag()&&(e=this.filterExinfo(t)||""),e?e.split("&"):[]},getBeaconExparams:function(t,e){var n=[],r=a.getExParams(o)||"";r=r.replace(/&aplus&/,"&");for(var i=o.param2arr(r)||[],u=function(e){return s.indexof(t,e)>-1},c=0;c<i.length;c++){var l=i[c],p=l[0]||"",g=l[1]||"";p&&g&&(e===this.getExparamsFlag()||u(p))&&n.push(p+"="+g)}return n},getExinfo:function(t){var e=this.options.context||{},n=e.what_to_hjlj_exinfo||{},o=n.exparams_key_names||[],a=this.getBeaconExparams(o,t);return a},getExData:function(t){var e=[];if("object"==typeof t)for(var n in t){var o=t[n];n&&o&&"object"!=typeof o&&"function"!=typeof o&&e.push(n+"="+o)}return e},doConcatArr:function(t,e){return e&&e.length>0&&(t=t.concat(e)),t},run:function(){try{var t=this.options.context.what_to_hjlj_exinfo||{},e=r.getGoldlogVal("_$")||{},n=e.meta_info||{},o=n["aplus-exinfo"]||"",a=n["aplus-exdata"]||"",s=[];s=this.doConcatArr(s,t.exinfo||[]),s=this.doConcatArr(s,this.getExinfo(o)),s=this.doConcatArr(s,this.getCookieUserInfo()),s=this.doConcatArr(s,this.getCustomExParams(o)),s=this.doConcatArr(s,this.getExData(a)),t.exinfo=s.join("&"),this.options.context.what_to_hjlj_exinfo=t}catch(t){i.logger({msg:t?t.message:""})}}}}},function(t,e,n){"use strict";var o=n(15),a=n(8);t.exports=function(){return{init:function(t){this.options=t},getCna:function(){var t=this.options.context||{},e=t.etag||{};return e.cna||""},getParams:function(){var t=this.options.context||{},e=this.options.config||{},n=t.userdata||{},r=goldlog._$||{},i=r.spm||{},s=[["logtype",1],["cna",this.getCna()],["cache",a.makeCacheNum()],["spm-cnt",e["spm-cnt"]||"0.0.0.0"],["spm-url",e["spm-url"]||i.spm_cnt||"0.0.0.0"]];i.spm_url&&s.push(["spm-pre",i.spm_url||""]),s.push(["pre",location.href]),s.push(["aplus",""]),s.push(["_pseudo_url",e._pseudo_url]);var u=[];for(var c in n)u.push(c+"="+n[c]);var l=t.what_to_hjlj_exinfo||{},p=l.exinfo||"";if(p){var g=p.split("&")||[];o.each(g,function(t){var e=t.split("=")||[];2===e.length&&s.push(e)})}return s.push(["_extend_args",u.join("&")]),s.push(["_is_pseudo",1]),s},run:function(){this.options.context.what_to_prepv.logdata=this.getParams()}}}},function(t,e,n){"use strict";var o=n(15),a=n(22),r=n(23);t.exports=function(){return{init:function(t){this.options=t},getMetaInfo:function(){var t=a.getGoldlogVal("_$")||{},e=t.meta_info||r.getInfo();return e},getAplusMetaByKey:function(t){var e=this.getMetaInfo()||{};return e[t]},getGifPath:function(t,e){var n,r=a.getGoldlogVal("_$")||{};if("function"==typeof t)n=t(location.hostname,r.is_terminal,o.is_in_iframe)+".gif";else if(!n&&e){var i=e.match(/\/\w+\.gif/);i&&i.length>0&&(n=i[0])}return n||(n=r.is_terminal?"m.gif":"v.gif"),n},run:function(){var t=this.getAplusMetaByKey("aplus-rhost-v"),e=this.options.context.where_to_prepv||{},n=e.url||"",a=this.getGifPath(e.urlRule,n),r=o.getPvUrl({metaName:"aplus-rhost-v",metaValue:t,gifPath:a,url:o.filterIntUrl(n)});e.url=r,this.options.context.where_to_prepv=e}}}},function(t,e,n){"use strict";var o=n(22);t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=this.options.context||{},e=this.options.config||{},n=t.logger||{},a=t.what_to_prepv||{},r=t.where_to_prepv||{},i=a.logdata||{},s=r.url||"";s||"function"!=typeof n||n({msg:"warning: where_to_prepv.url is null, goldlog.record failed!"});var u=goldlog.send(r.url,i,e.method||"GET");o.setGoldlogVal("req",u)}}}},function(t,e,n){"use strict";function o(){var t=i.getGoldlogVal("_$")||{},e="//gm.mmstat.com/";return t.is_terminal&&(e="//wgo.mmstat.com/"),{where_to_hjlj:{url:e,ac_atpanel:"//ac.mmstat.com/",tblogUrl:"//log.mmstat.com/"}}}function a(){return r.assign(new s,new o)}var r=n(8),i=n(22),s=n(62);t.exports=a},function(t,e,n){"use strict";function o(){return{compose:{},basic_params:{cna:a.getCookie("cna")},where_to_hjlj:{url:"//gm.mmstat.com/",ac_atpanel:"//ac.mmstat.com/",tblogUrl:"//log.mmstat.com/"},userdata:{},what_to_hjlj:{logdata:{}},what_to_pvhash:{hash:[]},what_to_hjlj_exinfo:{EXPARAMS_FLAG:"EXPARAMS",exinfo:[],exparams_key_names:["uidaplus","pc_i","pu_i"]},what_to_hjlj_ut:{logdataToUT:{}},what_to_hjlj_ut2:{isSuccess:!1,logdataToUT:{}},where_to_sendlog_ut:{aplusToUT:{},toUTName:"toUT"},network:{connType:"UNKNOWN"},is_single:!1}}var a=n(9);t.exports=o},function(t,e,n){"use strict";e.plugins_hjlj=[{name:"where_to_sendlog_ut",enable:!0,path:n(34)},{name:"is_single",enable:!0,path:n(36)},{name:"what_to_hjlj_exinfo",enable:!0,path:n(57)},{name:"what_to_pvhash",enable:!0,path:n(39)},{name:"what_to_hjlj",enable:!0,path:n(64),deps:["what_to_hjlj_exinfo","what_to_pvhash"]},{name:"what_to_hjlj_ut",enable:!0,path:n(65),deps:["is_single","what_to_hjlj_exinfo"]},{name:"what_to_hjlj_slog",enable:!0,path:n(66),deps:["what_to_hjlj"]},{name:"where_to_hjlj",enable:!0,path:n(67),deps:["is_single","what_to_hjlj"]},{name:"do_sendhjlj",enable:!0,path:n(68),deps:["is_single","what_to_hjlj","where_to_hjlj"]},{name:"do_sendhjlj_ut",enable:!0,path:n(69),deps:["what_to_hjlj","what_to_hjlj_ut","where_to_sendlog_ut"]}]},function(t,e,n){"use strict";var o=n(24),a=n(15),r=n(8);t.exports=function(){return{init:function(t){this.options=t},getParams:function(){var t=this.options.context||{},e=t.userdata||{},n=t.basic_params||{},i=t.what_to_hjlj_exinfo||{},s=i.exinfo||"";e.gokey&&s&&0!==s.indexOf("&")&&(s="&"+s);var u=n.cna,c=e.gmkey,l=e.gokey||"";l+=s;var p=t.what_to_pvhash||{},g=p.hash||[];g.length&&(l+="&"+g.join("="));var f={cache:r.makeCacheNum(),gmkey:c,gokey:l,cna:u};e["spm-cnt"]&&(f["spm-cnt"]=e["spm-cnt"]),e["spm-pre"]&&(f["spm-pre"]=e["spm-pre"]);try{var d=o.getExParams(a),h=a.param2obj(d).uidaplus;h&&(f._gr_uid_=h)}catch(t){}return f},run:function(){this.options.context.what_to_hjlj.logdata=this.getParams()}}}},function(t,e,n){"use strict";var o=n(37),a=n(9),r=n(22),i=n(3);t.exports=function(){return{init:function(t){this.options=t},getToUtData:function(t,e){var n=r.getGoldlogVal("_$")||{},s=n.spm||{},u=this.options.context||{},c=u.userdata||{},l=u.what_to_hjlj_exinfo||{},p=l.exinfo||"";c.gokey&&p&&0!==p.indexOf("&")&&(p="&"+p);var g=c.gokey+p,f={gmkey:c.gmkey,gokey:g,lver:i.lver,jsver:i.script_name,version:i.toUtVersion,spm_cnt:s.spm_cnt||"",spm_url:s.spm_url||"",spm_pre:s.spm_pre||""};t&&(f._is_g2u_=1),f._bridgeName=e.bridgeName||"",f._toUT=1;try{f=JSON.stringify(f),"{}"==f&&(f="")}catch(t){f=""}var d=n.meta_info||{},h=d.isonepage_data||{},_={};return _.functype=o.getFunctypeValue({logkey:c.logkey,gmkey:c.gmkey,spm_ab:r.getGoldlogVal("spm_ab")}),_.spmcnt=s.spm_cnt||"",_.spmurl=s.spm_url||"",_.spmpre=s.spm_pre||"",_.logkey=c.logkey,_.logkeyargs=f,_.urlpagename=h.urlpagename,_.url=location.href,_.cna=a.getCookie("cna")||"",_.extendargs="",_.isonepage=h.isonepage,_},run:function(){var t=this.options.context||{},e=!!t.is_single,n=t.what_to_hjlj_ut||{},o=t.where_to_sendlog_ut||{},a=o.aplusToUT||{},r=a.toUT||{};n.logdataToUT=this.getToUtData(e,r),this.options.context.what_to_hjlj_ut=n}}}},function(t,e){"use strict";t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=this.options.context||{},e=t.is_single?"1":"0";t.what_to_hjlj_ut2.logdataToUT._slog=e,t.what_to_hjlj_ut.logdataToUT._slog=e,t.what_to_hjlj.logdata.gokey?t.what_to_hjlj.logdata.gokey+="&_slog="+e:t.what_to_hjlj.logdata.gokey="_slog="+e}}}},function(t,e,n){"use strict";var o=n(15),a=n(8),r=n(22),i=n(20),s=n(23);t.exports=function(){return{init:function(t){this.options=t},getMetaInfo:function(){var t=r.getGoldlogVal("_$")||{},e=t.meta_info||s.getInfo();return e},getAplusMetaByKey:function(t){var e=this.getMetaInfo()||{};return e[t]},cramUrl:function(t){var e=r.getGoldlogVal("_$")||{},n=e.spm||{},o=this.options.context.where_to_hjlj||{},i=o.ac_atpanel,s=o.tblogUrl,u=this.options.context.what_to_hjlj||{},c=this.options.context.userdata||{},l=!0,p=c.logkey;if(!p)return{url:t,logkey_available:!1};if("ac"==p)t=i+"1.gif";else if(a.isStartWith(p,"ac-"))t=i+p.substr(3);else if(a.isStartWith(p,"/")){t+=p.substr(1);var g=u.logdata||{};g["spm-cnt"]=n.spm_cnt,g.logtype=2;try{u.logdata=g,this.options.context.what_to_hjlj=u}catch(t){}}else a.isEndWith(p,".gif")?t=s+p:l=!1;return{url:t,logkey_available:l}},can_to_sendhjlj:function(t){var e=this.options.context||{},n=e.logger||function(){},o=this.options.context.userdata||{};return!!t.logkey_available||(n({msg:"logkey: "+o.logkey+" is not legal!"}),!1)},run:function(){var t=!!this.options.context.is_single;if(!t){var e,n,a=o.filterIntUrl(this.options.context.where_to_hjlj.url),r=this.getAplusMetaByKey("aplus-rhost-g"),s=r&&o.hostValidity(r);s&&(e=/^\/\//.test(r)?"":"//",n=/\/$/.test(r)?"":"/",a=e+r+n),r&&!s&&i.logger({msg:"aplus-rhost-g: "+r+' is invalid, suggestion: "xxx.mmstat.com"'});var u=this.cramUrl(a);return this.can_to_sendhjlj(u)?void(this.options.context.where_to_hjlj.url=u.url):"done"}}}}},function(t,e,n){"use strict";var o=n(22);t.exports=function(){return{init:function(t){this.options=t},run:function(){var t=this.options.context||{},e=this.options.config||{},n=!!t.is_single;if(!n){var a=t.logger||{},r=t.what_to_hjlj||{},i=t.where_to_hjlj||{},s=r.logdata||{},u=i.url||"";u||"function"!=typeof a||a({msg:"warning: where_to_hjlj.url is null, goldlog.record failed!"});var c=goldlog.send(i.url,s,e.method||"GET");o.setGoldlogVal("req",c)}}}}},function(t,e){"use strict";t.exports=function(){return{init:function(t){this.options=t},run:function(t,e){var n=this,o=this.options.context||{},a=o.what_to_hjlj_ut2.isSuccess,r=o.logger||function(){},i=!!o.is_single,s=o.where_to_sendlog_ut||{},u=o.what_to_hjlj_ut||{},c=u.logdataToUT||{},l=s.aplusToUT||{},p=l.toUT;if(goldlog.isUT4Aplus&&"UT4Aplus"===goldlog.getMetaInfo("aplus-toUT"))return l.toutflag="toUT",void(n.options.context.what_to_hjlj_ut.isSuccess=!0);if(!a&&p&&"function"==typeof p.toUT2&&p.isAvailable)try{l.toutflag="toUT",p.toUT2(c,function(){n.options.context.what_to_hjlj_ut.isSuccess=!0,e()},function(t){n.options.context.what_to_hjlj_ut.errorMsg=t,e()},3e3)}catch(t){i&&r({msg:"warning: singleSend toUTName = "+s.toUTName+" errorMsg:"+t.message})}finally{return"pause"}}}}},function(t,e,n){"use strict";function o(){var t,e,n=i.KEY||{},o=n.NAME_STORAGE||{};if(!c&&u){var a=location.href,l=u&&(a.indexOf("login.taobao.com")>=0||a.indexOf("login.tmall.com")>=0),p=s.getRefer();l&&p?(t=p,e=r.getItem(o.REFERRER_PV_ID)):(t=a,e=goldlog.pvid),r.setItem(o.REFERRER,t),r.setItem(o.REFERRER_PV_ID,e)}}var a=n(71),r=n(48).nameStorage,i=n(3),s=n(47),u="https:"==location.protocol,c=parent!==self;e.run=function(){a.on(window,"beforeunload",function(){o()})}},function(t,e){"use strict";function n(t,e){var n=goldlog._$||{},o=n.meta_info||{},a=o.aplus_ctap||{};if(a&&"function"==typeof a.on)a.on(t,e);else{var r="ontouchend"in document.createElement("div"),i=r?"touchstart":"mousedown";s(t,i,e)}}function o(t){try{c.documentElement.doScroll("left")}catch(e){return void setTimeout(function(){o(t)},1)}t()}function a(t){var e=0,n=function(){0===e&&t(),e++};"complete"===c.readyState&&n();var a;if(c.addEventListener)a=function(){c.removeEventListener("DOMContentLoaded",a,!1),n()},c.addEventListener("DOMContentLoaded",a,!1),window.addEventListener("load",n,!1);else if(c.attachEvent){a=function(){"complete"===c.readyState&&(c.detachEvent("onreadystatechange",a),n())},c.attachEvent("onreadystatechange",a),window.attachEvent("onload",n);var r=!1;try{r=null===window.frameElement}catch(t){}c.documentElement.doScroll&&r&&o(n)}}function r(t){"complete"===c.readyState?t():s(u,"load",t)}function i(t){if(!/touch|mouse|scroll|wheel/i.test(t))return!1;var e=!1;try{var n=Object.defineProperty({},"passive",{get:function(){e=!0}});u.addEventListener("test",null,n)}catch(t){}return e}function s(){var t=arguments;if(2===t.length)"DOMReady"===t[0]&&a(t[1]),"onload"===t[0]&&r(t[1]);else if(3===t.length){var e=t[0],o=t[1],s=t[2];"tap"===o?n(e,s):e[f]((l?"on":"")+o,function(t){t=t||u.event;var e=t.target||t.srcElement;"function"==typeof s&&s(t,e)},!!i(o)&&{passive:!0})}}var u=window,c=document,l=!!c.attachEvent,p="attachEvent",g="addEventListener",f=l?p:g;e.DOMReady=a,e.onload=r,e.on=s},function(t,e,n){"use strict";function o(){var t=window.goldlog||{},e=t._$||{},n=e.meta_info||{},o=(new Date).getTime(),s=Math.floor(o/72e5),u=a.getElementById("aplus-sufei"),c=t.getCdnPath(),l=c+"/alilog/aplus_plugin_xwj/index.js?t="+s,p=c+"/secdev/entry/index.js?t="+s,g=function(){var t=n["aplus-loadjs"];if(t)for(var e=0;e<t.length;e++)t[e].file&&r.addScript(t[e].file)},f=function(){setTimeout(function(){u&&"script"==u.tagName.toLowerCase()||r.addScript(p,"","aplus-sufei")},10)},d=function(){var t=n["aplus-rate-ahot"];t||0==t?t:t=.01,(Math.random()<t||n["ahot-aplus"])&&r.addScript(l),f()},h=function(){r.addScript(l),f()};g(),i.onload(function(){try{var t=n["aplus-xplug"];switch(t){case"NONE":break;case"ALL":h();break;default:d()}}catch(t){}})}var a=document,r=n(24),i=n(71),s=n(73);e.run=function(){s.init_aplusQueue(),o()},e.init_watchGoldlogQueue=function(){}},function(t,e,n){"use strict";function o(t){for(var e=[],n=[],o=[],a=[],r=[],i=[],s={};s=t.shift();)try{var u=s.action,c=s.arguments[0];/subscribe/.test(u)?"setMetaInfo"===c?n.push(s):"mw_change_pv"===c||"mw_change_hjlj"===c?e.push(s):o.push(s):/MetaInfo/.test(u)?a.push(s):r.push(s)}catch(t){r.push(s),l.do_tracker_jserror({message:t&&t.message,error:encodeURIComponent(t.stack),filename:"sortQueue"})}return i=e.concat(a),i=i.concat(o),i=i.concat(n,r)}var a=window,r=n(8),i=n(74),s=n(20),u=n(75),c=n(76),l=n(77),p=s.isDebugAplus();e.init_aplusQueue=function(){var t,e="_ap",n=a[e]=a[e]||[];n.push=t=function(){var t="0.0";window.goldlog&&window.goldlog.spm_ab&&(t=window.goldlog.spm_ab.join(".")),l.do_tracker_obsolete_inter({ratio:p?1:.01,page:location.hostname+location.pathname,spm_ab:t,interface_name:"win._ap",interface_params:JSON.stringify(arguments)});for(var e,o,a=0,s=arguments.length;a<s;a++)e=arguments[a],r.isString(e)?goldlog.send(i.hjlj()+e):r.isArray(e)&&"push"!=(o=e[0])&&(n[o]=n[o]||[]).push(e.slice(1))};for(var o;o=n.shift();)t(o)};var g="goldlog_queue",f=function(t,e,n){try{/_aplus_cplugin_track_deb/.test(t)||/_aplus_cplugin_m/.test(t)||l.do_tracker_jserror({message:n||'illegal task: goldlog_queue.push("'+JSON.stringify(e)+'")',error:JSON.stringify(e),filename:"processTask"})}catch(t){}},d=function(t,e){var n=t?t.action:"",o=t?t.arguments:"";try{if(n&&o&&r.isArray(o)){var i=n.split("."),s=a,u=a;if(3===i.length)s=a[i[0]][i[1]]||{},u=s[i[2]]?s[i[2]]:"";else for(;i.length;)if(u=s=s[i.shift()],!s)return void("function"==typeof e?e(t):f(n,t));"function"==typeof u&&u.apply(s,o)}else f(n,t)}catch(e){f(n,t,e.message)}},h=function(t){function e(){if(t&&r.isArray(t)&&t.length){for(var e=o(t),n={},a=[];n=e.shift();)d(n,function(t){a.push(t)});a.length>0&&setTimeout(function(){for(;n=a.shift();)d(n)},100)}}try{e()}catch(t){l.do_tracker_jserror({message:t&&t.message,error:encodeURIComponent(t.stack),filename:"processGoldlogQueue"})}};e.processGoldlogQueue=h;var _=u.extend({push:function(t){this.length++,d(t)}});e.init_watchGoldlogQueue=function(){c.init_loadAplusPlugin();try{var t=a[g]||[];a[g]=_.create({startLength:t.length,length:0}),h(t)}catch(t){l.do_tracker_jserror({message:t&&t.message,error:encodeURIComponent(t.stack),filename:"init_watchGoldlogQueue"})}}},function(t,e,n){"use strict";var o=n(18);e.hjlj=function(){var t=window.goldlog||(window.goldlog={}),e=t._$||{},n=e.script_name,a=e.meta_info||{},r=a["aplus-rhost-g"],i="//gm.mmstat.com/";return(e.is_terminal||"aplus_wap"===n)&&(i="//wgo.mmstat.com/"),"aplus_int"===n&&(i="//gj.mmstat.com/"),r&&(i="//"+r+"/"),o.getProtocal()+i}},function(t,e){"use strict";function n(){}n.prototype.extend=function(){},n.prototype.create=function(){},n.extend=function(t){return this.prototype.extend.call(this,t)},n.prototype.create=function(t){var e=new this;for(var n in t)e[n]=t[n];return e},n.prototype.extend=function(t){var e=function(){};try{"function"!=typeof Object.create&&(Object.create=function(t){function e(){}return e.prototype=t,new e}),e.prototype=Object.create(this.prototype);for(var n in t)e.prototype[n]=t[n];e.prototype.constructor=e,e.extend=e.prototype.extend,e.create=e.prototype.create}catch(t){console.log(t)}finally{return e}},t.exports=n},function(t,e,n){"use strict";var o=n(24),a=n(25),r=n(5),i=function(t,e){var n=a.getMetaCnt(t);return!(!n&&!e)},s=function(){var t=goldlog.getCdnPath();return{aplus_ae_path:t+"/alilog/s/"+r.lver+"/plugin/aplus_ae.js",aplus_ac_path:t+"/alilog/s/"+r.lver+"/plugin/aplus_ac.js"}},u=function(t,e){var n=s(),a=i(t,e),r={"aplus-auto-exp":n.aplus_ae_path,"aplus-auto-clk":n.aplus_ac_path};a&&r[t]&&o.addScript(r[t])};e.init_loadAplusPlugin=function(){!goldlog._aplus_auto_exp&&u("aplus-auto-exp"),!goldlog._aplus_ac&&u("aplus-auto-clk"),goldlog.aplus_pubsub.subscribe("setMetaInfo",function(t,e){"aplus-auto-exp"!==t||goldlog._aplus_auto_exp||u(t,e),"aplus-auto-clk"!==t||goldlog._aplus_ac||u(t,e)})}},function(t,e){"use strict";var n=function(t,e){var n=window.goldlog_queue||(window.goldlog_queue=[]);n.push({action:"goldlog._aplus_cplugin_track_deb.monitor",arguments:[{key:"APLUS_PLUGIN_DEBUG",title:"aplus_core",msg:["_error_:methodName="+e+",params="+JSON.stringify(t)],type:"updateMsg",description:e||"aplus_core"}]})},o=function(t,e,n){var o=window.goldlog_queue||(window.goldlog_queue=[]);o.push({action:["goldlog","_aplus_cplugin_m",e].join("."),arguments:[t,n]})};e.do_tracker_jserror=function(t,e){var a="do_tracker_jserror";o(t,a,e),n(t,a)},e.do_tracker_obsolete_inter=function(t,e){var a="do_tracker_obsolete_inter";o(t,a,e),n(t,a)},e.wrap=function(t){if("function"==typeof t)try{t()}catch(t){n({msg:t.message||t},"exception")}finally{}}},function(t,e){"use strict";function n(t,e){return t.indexOf(e)>-1}function o(t,e){for(var o=0,a=t.length;o<a;o++)if(n(e,t[o]))return!0;return!1}var a=location.host,r=["xiaobai.com","admin.taobao.org","aliloan.com","mybank.cn"],i=["tmc.admin.taobao.org","tmall.admin.taobao.org"];e.is_exception=o(r,a)&&!o(i,a)},function(t,e,n){"use strict";function o(){var t,e,n,o,a=c.getElementsByTagName("meta");for(t=0,e=a.length;t<e;t++)if(n=a[t],o=n.getAttribute("name"),"data-spm"===o||"spm-id"===o)return n}function a(){var t=c.createElement("meta");t.setAttribute("name","data-spm");var e=c.getElementsByTagName("head")[0];return e&&e.insertBefore(t,e.firstChild),t}function r(){var t=o();t||(t=a()),t.setAttribute("content",goldlog.spm_ab[0]||"");var e=c.getElementsByTagName("body")[0];e&&e.setAttribute("data-spm",goldlog.spm_ab[1]||"")}function i(){var t,e,n,o=c.getElementsByTagName("*");for(t=0,e=o.length;t<e;t++)n=o[t],n.getAttribute("data-spm-max-idx")&&n.setAttribute("data-spm-max-idx",""),n.getAttribute("data-spm-anchor-id")&&n.setAttribute("data-spm-anchor-id","")}function s(){var t=5e3;try{var e=goldlog.getMetaInfo("aplus-mmstat-timeout");if(e){var n=parseInt(e);n>=1e3&&n<=1e4&&(t=n)}}catch(t){}return t}var u=window,c=document,l=n(75),p=n(15),g=n(71),f=n(20),d=n(32),h=n(8),_=n(22),m=n(18),v=n(35),b=n(45),y=n(23),w=y.getInfo(),x=n(3),j=n(77),T=n(80),P=n(9),A=n(83),S=n(2),k=f.isDebugAplus(),E=[],U=[],C=[],I=[];e.run=l.extend({getCdnPath:function(){var t=c.getElementById("beacon-aplus")||c.getElementById("tb-beacon-aplus"),e="//g.alicdn.com",n=["//assets.alicdn.com/g","//g-assets.daily.taobao.net"];if(t)for(var o=0;o<n.length;o++){var a=new RegExp(n[o]);if(a.test(t.src)){e=n[o];break}}return e},isInternational:function(){this.cdnPath||(this.cdnPath=this.getCdnPath());var t="//assets.alicdn.com/g"===this.cdnPath||"//laz-g-cdn.alicdn.com"===this.cdnPath;return t||"int"===this.getMetaInfo("aplus-env")},getCookie:function(t){return P.getCookie(t)},getParam:function(t){var e=u.WindVane||{},n=v.isAplusChnl(),o="";n&&"object"==typeof n&&(o=n.bridgeName||"customBridge");var a=e.getParam?"WindVane":o,r=e&&"function"==typeof e.getParam?e.getParam(t):"",i=goldlog.spm_ab?goldlog.spm_ab.join("."):"0.0",s="sid="+t+"@valueIsEmpty="+!r;return a&&(s+="_bridgeName="+a),j.do_tracker_obsolete_inter({ratio:k?1:.01,page:location.hostname+location.pathname,spm_ab:i,interface_name:"goldlog.getParam",interface_params:s}),r},beforeSendPV:function(t){E.push(t)},afterSendPV:function(t){U.push(t)},send:function(t,e,n){var o;if(0===t.indexOf("//")){var a=m.getProtocal();t=a+t}return o="POST"===n&&navigator&&navigator.sendBeacon?S.postData(t,e):S.sendImg(p.makeUrl(t,e),s())},launch:function(t,e){var n;try{e=h.assign(e,t),n=goldlog._$._sendPV(e,t);var o=goldlog.spm_ab?goldlog.spm_ab.join("."):"0.0";j.do_tracker_obsolete_inter({page:location.hostname+location.pathname,spm_ab:o,interface_name:"goldlog.launch",interface_params:"userdata = "+JSON.stringify(t)+", config = "+JSON.stringify(e)})}catch(t){}finally{return f.logger({msg:"warning: This interface is deprecated, please use goldlog.sendPV instead! API: http://log.alibaba-inc.com/log/info.htm?type=2277&id=31"}),n}},_$:{_sendPV:function(t,e){if(t=t||{},h.any(E,function(e){return e(goldlog,t)===!1}))return!1;var o=n(85).SendPV,a=new o;return"undefined"==typeof t.recordType&&(t.recordType=x.recordTypes.pv),a.run(t,e,{fn_after_pv:U}),!0},_sendPseudo:function(t,e){t||(t={});var o=n(86).SendPrePV,a=new o;return"undefined"==typeof t.recordType&&(t.recordType=x.recordTypes.prepv),a.run(t,e,{},function(){d.doPubMsg(["sendPrePV","complete"])}),!0}},sendPV:function(t,e){return e=e||{},goldlog._$._sendPV(t,e)},beforeRecord:function(t){C.push(t)},afterRecord:function(t){
I.push(t)},record:function(t,e,n,o,a){if(!h.any(C,function(t){return t(goldlog)===!1}))return T.run({recordType:x.recordTypes.hjlj,method:"POST"===o?"POST":"GET"},{logkey:t,gmkey:e,gokey:n},{fn_after_record:I},function(){"function"==typeof a&&a()}),!0},recordUdata:function(t,e,n,o,a){var r=_.getGoldlogVal("_$")||{},i=r.spm||{};T.run({ignore_chksum:!0,method:"POST"===o?"POST":"GET",recordType:x.recordTypes.uhjlj},{logkey:t,gmkey:e,gokey:n,"spm-cnt":i.spm_cnt,"spm-pre":i.spm_pre},{},function(){h.isFunction(a)&&a()})},setPageSPM:function(t,e,n){var o=goldlog.getMetaInfo("aplus-spm-fixed"),a="function"==typeof n?n:function(){};goldlog.spm_ab=goldlog.spm_ab||[];var s=h.cloneObj(goldlog.spm_ab);if(t&&(goldlog.spm_ab[0]=""+t,goldlog._$.spm.data.a=""+t),e&&(goldlog.spm_ab[1]=""+e,goldlog._$.spm.data.b=""+e),b.spaInit(goldlog,w,s),"1"!==o){var u=s.join(".");goldlog.spmab_pre=u}var c=goldlog.spm_ab.join(".");d.doPubMsg(["setPageSPM",{spmab_pre:goldlog.spmab_pre,spmab:c}]),d.doCachePubs(["setPageSPM",{spmab_pre:goldlog.spmab_pre,spmab:c}]),r(),i(),a()},setMetaInfo:function(t,e){if(y.setMetaInfo(t,e)){var n=_.getGoldlogVal("_$")||{};n.meta_info=y.qGet();var o=_.setGoldlogVal("_$",n),a=A.isDisablePvid()+"";return"aplus-disable-pvid"===t&&a!==e+""&&b.resetSpmCntPvid(),d.doPubMsg(["setMetaInfo",t,e]),d.doCachePubs(["setMetaInfo",t,e]),o}},appendMetaInfo:y.appendMetaInfo,getMetaInfo:function(t){return y.getMetaInfo(t)},on:g.on,cloneDeep:h.cloneDeep,getPvId:A.getPvId})},function(t,e,n){"use strict";var o=n(8),a=n(22),r=n(32),i=n(20),s=n(81),u=n(82),c=n(3);e.run=function(t,e,n,l){var p=new u;p.init({middleware:[],config:t,plugins:c.plugins_hjlj});var g=p.run(),f=new c.context_hjlj;f.userdata=e,f.logger=i.logger;var d={context:f,pubsub:a.getGoldlogVal("aplus_pubsub"),pubsubType:"hjlj"},h=new s;h.create(d),h.wrap(g,function(){d.context.status="complete",r.doPubMsg(["mw_change_hjlj",d.context]),n&&n.fn_after_record&&o.each(n.fn_after_record,function(t){t(window.goldlog)}),"function"==typeof l&&l()})()}},function(t,e,n){"use strict";function o(){}var a=n(21),r=n(19),i=n(20),s=n(77),u=n(9);o.prototype.create=function(t){for(var e in t)"undefined"==typeof this[e]&&(this[e]=t[e]);return this},o.prototype.pubsubInfo=function(t,e){try{t&&t.pubsub&&t.pubsub.publish("mw_change_"+t.pubsubType,t.context,e)}catch(t){}},o.prototype.calledList=[],o.prototype.setCalledList=function(t){a.indexof(this.calledList,t)===-1&&this.calledList.push(t)},o.prototype.resetCalledList=function(){this.calledList=[]},o.prototype.wrap=function(t,e){var n=this,o=this.context||{},c=o.compose||{},l=c.maxTimeout||1e4;return function(o){var c,p=t.length,g=0,f=0,d=function(){if(n.pubsubInfo(n,t[g]),g===p)return o="done",n.resetCalledList(),"function"==typeof e&&e.call(n,o),void clearTimeout(c);if(a.indexof(n.calledList,g)===-1){if(n.setCalledList(g),!t[g]||"function"!=typeof t[g][0])return;try{o=t[g][0].call(n,o,function(){g++,f=1,clearTimeout(c),d(g)})}catch(e){s.do_tracker_jserror({message:e?e.message:"compose middleware error",error:encodeURIComponent(e.stack),filename:t[g][1]})}}var h="number"==typeof o;if("pause"===o||h){f=0;var _=h?o:l,m=t[g]?t[g][1]:"";c=r.sleep(_,function(){if(0===f){var t="jump the middleware about "+m+", because waiting timeout maxTimeout = "+_+"ms!";i.logger({msg:t});var e=window.goldlog_queue||(window.goldlog_queue=[]);e.push({action:"goldlog._aplus_cplugin_m.do_tracker_browser_support",arguments:[{msg:t,spmab:goldlog.spm_ab,page:location.href,etag:n.context?JSON.stringify(n.context.etag):"",cna:document.cookie?u.getCookie("cna"):""}]}),o=null,g++,d(g)}})}else"done"===o?(g=p,d(g)):(g++,d(g))};return n.calledList&&n.calledList.length>0&&n.resetCalledList(),d(g)}},t.exports=o},function(t,e,n){"use strict";var o=n(21);t.exports=function(){return{init:function(t){this.opts=t,t&&"object"==typeof t.middleware&&t.middleware.length>0?this.middleware=t.middleware:this.middleware=[],this.plugins_name=[]},pubsubInfo:function(t,e){try{var n=t.pubsub;n&&n.publish("plugins_change_"+t.pubsubType,e)}catch(t){}},checkPluginLoader:function(t,e){var n=!0;if("object"==typeof e.enable&&"function"==typeof e.enable.isEnable?n=e.enable.isEnable(e.name):"boolean"==typeof e.enable&&(n=!!e.enable),!n)return!1;if(n&&e.deps&&e.deps.length>0)for(var a=0;a<e.deps.length;a++)if(o.indexof(this.plugins_name,e.deps[a])===-1)return!1;return!0},run:function(t){t||(t=0);var e=this,n=this.middleware,o=this.opts||{},a=o.plugins;if(a&&"object"==typeof a&&a.length>0){var r=a[t];if(this.checkPluginLoader(a,r)&&(this.plugins_name.push(r.name),n.push([function(t,n){e.pubsubInfo(this,r);var a=new r.path;return a.init({context:this.context,config:o.config}),a.run(t,n)},r.name])),t++,a[t])return this.run(t)}else window.console&&console.log("aplus plugins "+JSON.stringify(a)+" must be object of array!");return n}}}},function(t,e,n){"use strict";function o(){var t="true"===l.disablePvid;try{var e=goldlog.getMetaInfo("aplus-disable-pvid")+"";"true"===e?t=!0:"false"===e&&(t=!1)}catch(t){}return t}function a(t){function e(t){var e="0123456789abcdefhijklmnopqrstuvwxyzABCDEFHIJKLMNOPQRSTUVWXYZ",n="0123456789abcdefghijkmnopqrstuvwxyzABCDEFGHIJKMNOPQRSTUVWXYZ";return 1==t?e.substr(Math.floor(60*Math.random()),1):2==t?n.substr(Math.floor(60*Math.random()),1):"0"}for(var n,o="",a="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",r=!1;o.length<t;)n=a.substr(Math.floor(62*Math.random()),1),!r&&o.length<=2&&("g"==n.toLowerCase()||"l"==n.toLowerCase())&&(0===o.length&&"g"==n.toLowerCase()?Math.random()<.5&&(n=e(1),r=!0):1==o.length&&"l"==n.toLowerCase()&&"g"==o.charAt(0).toLowerCase()&&(n=e(2),r=!0)),o+=n;return o}function r(t,e,n){return t?u.hash(encodeURIComponent(t)).substr(0,e):n}function i(){var t=a(8),e=t.substr(0,4),n=t.substr(0,6);return[r(location.href,4,e),r(document.title,4,e),n].join("")}function s(){var t=goldlog.pvid;return goldlog.pvid=i(),c.doPubMsg(["pvidChange",{pre_pvid:t,pvid:goldlog.pvid}]),c.doCachePubs(["pvidChange",{pre_pvid:t,pvid:goldlog.pvid}]),o()?"":goldlog.pvid}var u=n(84),c=n(32),l=n(3);e.isDisablePvid=o,e.makePVId=s,e.getPvId=function(){return o()?"":goldlog.pvid}},function(t,e){"use strict";var n=1315423911;e.hash=function(t,e){var o,a,r=e||n;for(o=t.length-1;o>=0;o--)a=t.charCodeAt(o),r^=(r<<5)+a+(r>>2);var i=(2147483647&r).toString(16);return i}},function(t,e,n){"use strict";var o=n(8),a=n(22),r=n(32),i=n(20),s=n(81),u=n(82),c=n(3),l=function(){};l.prototype.run=function(t,e,n){var l=new u;l.init({middleware:[],config:t,plugins:c.plugins_pv});var p=l.run(),g=new c.context;g.userdata=e,g.logger=i.logger;var f={context:g,pubsub:a.getGoldlogVal("aplus_pubsub"),pubsubType:"pv"},d=new s;d.create(f),d.wrap(p,function(){var e=f.context.can_to_sendpv||{};f.context.status="YES"===e.flag?"complete":"skip",r.doPubMsg(["mw_change_pv",f.context]),n&&n.fn_after_record&&o.each(n.fn_after_pv,function(e){e(window.goldlog,t)})})()},e.SendPV=l},function(t,e,n){"use strict";var o=n(8),a=n(22),r=n(32),i=n(20),s=n(81),u=n(82),c=n(3),l=function(){};l.prototype.run=function(t,e,n,l){var p=new u;p.init({middleware:[],config:t,plugins:c.plugins_prepv});var g=p.run(),f=new c.context_prepv;f.userdata=e,f.logger=i.logger;var d={context:f,pubsub:a.getGoldlogVal("aplus_pubsub"),pubsubType:"prepv"},h=new s;h.create(d),h.wrap(g,function(){d.context.status="complete",r.doPubMsg(["mw_change_prepv",d.context]),n&&n.fn_after_record&&o.each(n.fn_after_pv,function(e){e(window.goldlog,t)}),a.setGoldlogVal("prepv_context",f),"function"==typeof l&&l()})()},e.SendPrePV=l},function(t,e,n){"use strict";!function(){var t=window.goldlog||(window.goldlog={}),e=n(88);t.aplus_pubsub||(t.aplus_pubsub=e.create())}()},function(t,e,n){"use strict";function o(t){if("function"!=typeof t)throw new TypeError(t+" is not a function");return t}var a=n(75),r=function(t){for(var e=t.length,n=new Array(e-1),o=1;o<e;o++)n[o-1]=t[o];return n},i=a.extend({create:function(t){var e=new this;for(var n in t)e[n]=t[n];return e.handlers=[],e.pubs={},e},setHandlers:function(t){this.handlers=t},subscribe:function(t,e){o(e);var n=this,a=n.pubs||{},r=a[t]||[];if(r)for(var i=0;i<r.length;i++){var s=r[i]();e.apply(n,s)}var u=n.handlers||[];return t in u||(u[t]=[]),u[t].push(e),n.setHandlers(u),n},subscribeOnce:function(t,e){o(e);var n,a=this;return this.subscribe.call(this,t,n=function(){a.unsubscribe.call(a,t,n);var o=Array.prototype.slice.call(arguments);e.apply(a,o)}),this},unsubscribe:function(t,e){o(e);var n=this.handlers[t];if(!n)return this;if("object"==typeof n&&n.length>0){for(var a=0;a<n.length;a++){var r=e.toString(),i=n[a].toString();r===i&&n.splice(a,1)}this.handlers[t]=n}else delete this.handlers[t];return this},publish:function(t){var e=r(arguments),n=this.handlers||[],o=n[t]?n[t].length:0;if(o>0)for(var a=0;a<o;a++){var i=n[t][a];i&&"function"==typeof i&&i.apply(this,e)}return this},cachePubs:function(t){var e=this.pubs||{},n=r(arguments);e[t]||(e[t]=[]),e[t].push(function(){return n})}});t.exports=i},function(t,e,n){"use strict";var o=n(35),a=n(32),r=n(3);e.init=function(){n(90)(function(){var t=n(3),e=goldlog._$,i=navigator.userAgent;i.match(/AliApp\(([A-Z\-]+)\/([\d\.]+)\)/i)&&(e.is_ali_app=!0),t.utilPvid.makePVId();var s=n(45);e.spm=s,e.is_WindVane=o.is_WindVane;var u=e.meta_info;e.page_url=location.href,e.page_referrer=n(47).getRefer(),s.init(goldlog,u,function(){t.initLoad.init_watchGoldlogQueue();var e=n(3).spmException,o=e.is_exception;o||n(92),a.doPubMsg(["aplusReady","complete"]),a.doCachePubs(["aplusReady","complete"])}),goldlog.beforeSendPV(function(t,e){if(e.is_auto&&"1"===u["aplus-manual-pv"])return!1}),goldlog.afterSendPV(function(){window.g_SPM&&(g_SPM._current_spm="")}),r.is_auto_pv+""=="true"&&goldlog.sendPV({is_auto:!0}),t.initLoad.run(),t.beforeUnload.run()})}},function(t,e,n){"use strict";var o=n(32);t.exports=function(t){var e=n(91).AplusInit,a=new e;a.run({},function(e){o.doPubMsg(["aplusInitContext",e]),o.doCachePubs(["aplusInitContext",e]),"function"==typeof t&&t()})}},function(t,e,n){"use strict";var o=n(22),a=n(20),r=n(81),i=n(82),s=n(3),u=function(){};u.prototype.run=function(t,e){var n=new i;n.init({middleware:[],config:t,plugins:s.aplus_init});var u=n.run(),c=new s.context;c.logger=a.logger;var l={context:c,pubsub:o.getGoldlogVal("aplus_pubsub"),pubsubType:"aplusinit"},p=new r;p.create(l),p.wrap(u,function(){"function"==typeof e&&e(l.context)})()},e.AplusInit=u},function(t,e,n){"use strict";!function(){var t,e=n(8),o=n(22),a=n(93),r=function(){t=!0;var n=window.g_SPM||{};e.isFunction(n.getParam)||e.isFunction(n.spm)||a.run()},i=window.goldlog||(window.goldlog={});i.aplus_pubsub&&"function"==typeof i.aplus_pubsub.publish&&i.aplus_pubsub.subscribe("goldlogReady",function(e){"complete"!==e||t||r()});var s=0,u=function(){if(!t){var e=o.getGoldlogVal("_$")||{};"complete"===e.status?r():s<50&&(++s,setTimeout(function(){u()},200))}};u()}()},function(t,e,n){"use strict";var o=n(15),a=n(21),r=n(16),i=n(8),s=n(24),u=n(71),c=n(22),l=n(20),p=n(18),g=n(94),f=g.spmRender,d=parent!==self;e.run=function(){function t(t){var e=yt(t,pt),n=r.parseSemicolonContent(e)||{};return n}function e(){var t=H.spm.data;return[t.a,t.b].join(".")}function n(t,e){var n,o,a,r,i,s,u,c,l,p=[];for(n=bt(t.getElementsByTagName("a")),o=bt(t.getElementsByTagName("area")),r=n.concat(o),u=0,c=r.length;u<c;u++){for(s=!1,i=a=r[u];(i=i.parentNode)&&i!=t;)if(yt(i,ut)){s=!0;break}s||(l=yt(a,gt),e||"t"==l?e&&"t"==l&&p.push(a):p.push(a))}return p}function g(o,a,r,u){var c,l,p,g,f,d,h,_,m,v,b,y,w,j,P,S,k,U,C;if(a=a||o.getAttribute(ut)||"",a&&(c=n(o,u),0!==c.length)){if(p=a.split("."),k=jt(a,"110")&&3==p.length,k&&(U=p[2],p[2]="w"+(U||"0"),a=p.join(".")),Tt(w=e())&&w.match(/^[\w\-\*]+(\.[\w\-\*\/]+)?$/))if(i.isContain(a,".")){if(!jt(a,w)){for(g=w.split("."),p=a.split("."),P=0,j=g.length;P<j;P++)p[P]=g[P];a=p.join(".")}}else i.isContain(w,".")||(w+=".0"),a=w+"."+a;if(a.match&&a.match(/^[\w\-\*]+\.[\w\-\*\/]+\.[\w\-\*\/]+$/)){var I=u?dt:ft;for(C=parseInt(yt(o,I))||0,S=0,f=C,j=c.length;S<j;S++)if(l=c[S],d=s.tryToGetHref(l),h=s.tryToGetAttribute(l,pt),u||d||h)if(k&&l.setAttribute(_t,U),_=l.getAttribute(mt),_&&E(_))x(l,_,r);else{v=A(l.parentNode),v.a_spm_ab?(y=v.a_spm_ab,b=v.ab_idx):(y=void 0,f++,b=f);var M=t(l)||{},V=M.locaid||"";V?m=V:(m=T(l)||b,u&&(m="at"+((i.isNumber(m)?1e3:"")+m))),_=y?a+"-"+y+"."+m:a+"."+m,x(l,_,r)}o.setAttribute(I,f)}}}function h(t){for(var e=["mclick.simba.taobao.com","click.simba.taobao.com","click.tanx.com","click.mz.simba.taobao.com","click.tz.simba.taobao.com","redirect.simba.taobao.com","rdstat.tanx.com","stat.simba.taobao.com","s.click.taobao.com"],n=0;n<e.length;n++)if(t.indexOf(e[n])!=-1)return!0;return!1}function _(t){return t?!!t.match(/^[^\?]*\balipay\.(?:com|net)\b/i):J}function m(t){return t?!!t.match(/^[^\?]*\balipay\.(?:com|net)\/.*\?.*\bsign=.*/i):J}function v(t){for(var e;(t=t.parentNode)&&t.tagName!=ot;)if(e=yt(t,ct))return e;return""}function b(t,e){if(!goldlog.isUT4Aplus||"UT4Aplus"!==goldlog.getMetaInfo("aplus-toUT")){if(t&&/&?\bspm=[^&#]*/.test(t)&&(t=t.replace(/&?\bspm=[^&#]*/g,"").replace(/&{2,}/g,"&").replace(/\?&/,"?").replace(/\?$/,"")),!e)return t;var n,o,a,r,s,u,c,l="&";if(t.indexOf("#")!=-1&&(a=t.split("#"),t=a.shift(),o=a.join("#")),r=t.split("?"),s=r.length-1,a=r[0].split("//"),a=a[a.length-1].split("/"),u=a.length>1?a.pop():"",s>0&&(n=r.pop(),t=r.join("?")),n&&s>1&&n.indexOf("&")==-1&&n.indexOf("%")!=-1&&(l="%26"),t=t+"?spm="+ht+e+(n?l+n:"")+(o?"#"+o:""),c=i.isContain(u,".")?u.split(".").pop().toLowerCase():""){if({png:1,jpg:1,jpeg:1,gif:1,bmp:1,swf:1}.hasOwnProperty(c))return 0;!n&&s<=1&&(o||{htm:1,html:1,php:1,aspx:1}.hasOwnProperty(c)||(t+="&file="+u))}return t}}function y(t){return t&&Q.split("#")[0]==t.split("#")[0]}function w(t){var e=s.tryToGetHref(t),n=v(t),o=yt(t,ct),a="i"===(o||n||st);if(!e||h(e))return!0;var r=y(e)||p.isStartWithProtocol(e.toLowerCase())||_(e)||m(e);return!(a||!jt(e,"#")&&!r)||a}function x(t,n,a){if(!/^0\.0\.?/.test(n)){var r=s.tryToGetHref(t),i=e(),u=w(t);if(u){var c=o.param2obj(r);if(c.spm&&c.spm.split)for(var l=c.spm.split("."),p=n.split("."),g=0;g<3&&p[g]===l[g];g++)2===g&&l[3]&&(n=c.spm)}t.setAttribute(mt,n),Z=goldlog.getPvId(),Z&&(n+="."+Z),(Z||i&&i!=tt)&&(u||a||(r=b(r,n))&&j(t,r))}}function j(t,e){if(!goldlog.isUT4Aplus||"UT4Aplus"!==goldlog.getMetaInfo("aplus-toUT")){var n,o=t.innerHTML;o&&o.indexOf("<")==-1&&(n=F.createElement("b"),n.style.display="none",t.appendChild(n)),t.href=e,n&&t.removeChild(n)}}function T(t){var e,n=H.spm.data;return"0"==n.a&&"0"==n.b?e="0":(e=yt(t,ut),e&&e.match(/^d\w+$/)||(e="")),e}function P(t){for(var e,n,o=t;t&&t.tagName!=nt&&t.tagName!=ot&&t.getAttribute;){if(n=t.getAttribute(ut)){e=n,o=t;break}if(!(t=t.parentNode))break}return e&&!/^[\w\-\.\/]+$/.test(e)&&(e="0"),{spm_c:e,el:o}}function A(t){for(var e,n={},o="";t&&t.tagName!=nt&&t.tagName!=ot;){if(!o&&(o=yt(t,vt))){e=parseInt(yt(t,"data-spm-ab-max-idx"))||0,n.a_spm_ab=o,n.ab_idx=++e,t.setAttribute("data-spm-ab-max-idx",e);break}if(yt(t,ut))break;t=t.parentNode}return n}function S(t){var e;return t&&(e=t.match(/&?\bspm=([^&#]*)/))?e[1]:""}function k(t,n){var o=goldlog.getMetaInfo("aplus-getspmcd")||function(){},a=e(),r=s.tryToGetHref(t),i=S(r),u=null,c=a&&2==a.split(".").length;if(c){var l=o(t,null,a);return u=l&&"0"!==l.spm_c?[a,l.spm_c,l.spm_d]:[a,0,T(t)||0],void x(t,u.join("."),n)}r&&i&&(r=r.replace(/&?\bspm=[^&#]*/g,"").replace(/&{2,}/g,"&").replace(/\?&/,"?").replace(/\?$/,"").replace(/\?#/,"#"),j(t,r))}function E(t){var n=e(),o=t.split(".");return o[0]+"."+o[1]==n}function U(t,e){at&&N();var n,o,a=yt(t,mt);if(a&&E(a))x(t,a,e);else{if(n=P(t.parentNode),o=n.spm_c,!o)return void k(t,e);et&&(o="0"),g(n.el,o,e),g(n.el,o,e,!0)}}function C(t){if(t&&1==t.nodeType){xt(t,ft),xt(t,dt);var e,n=bt(t.getElementsByTagName("a")),o=bt(t.getElementsByTagName("area")),a=n.concat(o),r=a.length;for(e=0;e<r;e++)xt(a[e],mt)}}function I(n){var o=yt(n,mt);if(!o){var a=e(),r=n.parentNode;if(!r)return"";var i=t(n)||{},s=i.locaid||"",u=n.getAttribute(ut)||s,c=P(r),l=c.spm_c||0;l&&l.indexOf(".")!=-1&&(l=l.split("."),l=l[l.length-1]),o=At(a+"."+l,u)}return o}function M(t){var e,n=t.tagName;"A"!=n&&"AREA"!=n?e=I(t):(U(t,q),e=yt(t,mt)),e||(e="0.0.0.0");var o=goldlog.getPvId();return 4===e.split(".").length&&o&&(e+="."+o),"A"!=n&&"AREA"!=n&&wt(t,mt,e),e=e.split("."),{a:e[0],b:e[1],c:e[2],d:e[3],e:e[4]}}function V(t,e){if(e||(e=F),F.evaluate)return e.evaluate(t,F,null,9,null).singleNodeValue;for(var n,o=t.split("/");!n&&o.length>0;)n=o.shift();var a,r=/^.+?\[@id="(.+?)"]$/i,i=/^(.+?)\[(\d+)]$/i;return(a=n.match(r))?e=e.getElementById(a[1]):(a=n.match(i))&&(e=e.getElementsByTagName(a[1])[parseInt(a[2])-1]),e?0===o.length?e:V(o.join("/"),e):null}function N(){var t,e,n,o={};for(t in rt)rt.hasOwnProperty(t)&&(e=V(t),e&&(o[t]=1,n=rt[t],wt(e,ut,("A"==e.tagName?n.spmd:n.spmc)||"")));for(t in o)o.hasOwnProperty(t)&&delete rt[t]}function O(){if(!it&&W.spmData){it=q;var t,e,n,o,a=W.spmData.data;if(a&&Pt(a)){for(t=0,e=a.length;t<e;t++)n=a[t],o=n.xpath,o=o.replace(/^id\("(.+?)"\)(.*)/g,'//*[@id="$1"]$2'),rt[o]={spmc:n.spmc,spmd:n.spmd};N()}}}function G(){var t,e,n,o,a=F.getElementsByTagName("iframe"),r=a.length;for(e=0;e<r;e++)t=a[e],!t.src&&(n=yt(t,lt))&&(o=M(t),o?(o=[o.a,o.b,o.c,o.d,o.e].join("."),t.src=b(n,o)):t.src=n)}function R(){function t(){e++,e>10&&(n=3e3),G(),setTimeout(t,n)}var e=0,n=500;t()}function L(t,e,n){var o=r.parseSemicolonContent(e,{},!0),s=o.gostr||"",u=o.locaid||"",c=t.getAttribute(ut)||u,p="CLK",g=o.gokey||"",f=M(t),d=[f.a,f.b,f.c,c].join("."),h=s+"."+d;0!==h.indexOf("/")&&(h="/"+h);var _=[],m=["gostr","locaid","gmkey","gokey","spm-cnt","cna"];for(var v in o)o.hasOwnProperty(v)&&a.indexof(m,v)===-1&&_.push(v+"="+o[v]);_.push("_g_et="+n),_.push("autosend=1"),g&&_.length>0&&(g+="&"),g+=_.length>0?_.join("&"):"",goldlog&&i.isFunction(goldlog.recordUdata)?goldlog.recordUdata(h,p,g,"GET",function(){}):l.logger({msg:"goldlog.recordUdata is not function!"}),wt(t,mt,d)}function $(t,n){f.fetchSpmd(n);var o=n;W.g_SPM&&(g_SPM._current_spm=M(n));for(var a;n&&n.tagName!=nt;){a=yt(n,pt);{if(a){L(n,a,"mousedown"===t.type?t.button:"tap");break}n=n.parentNode}}if(!a){var r=e(),i=goldlog.getMetaInfo("aplus-getspmcd")||function(){};i(o,t,r)}}function B(t,e){for(var n;e&&(n=e.tagName);){if("A"==n||"AREA"==n){U(e,J);var o=window.g_SPM||(window.g_SPM={}),a=o._current_spm=M(e),r=[];try{r=[a.a,a.b,a.c,a.d];var i=a.e||goldlog.pvid||"";i&&r.push(i)}catch(t){}break}if(n==ot||n==nt)break;e=e.parentNode}}function D(t,e){var n=M(t),o=n.a+"."+n.b+"."+n.c+"."+n.d;return e&&(o+="."+n.e),o}f.renderSpmc();var W=window,F=document,K=location,q=!0,J=!1,H=c.getGoldlogVal("_$")||{},Y=H.meta_info,Q=K.href,X=H.is_terminal||/WindVane/i.test(navigator.userAgent),z=s.isTouch()||"1"===Y["aplus-touch"];W.g_SPM||(W.g_SPM={}),W.g_SPM.spm_d_for_ad={};var Z,tt="0.0",et=!1,nt="HTML",ot="BODY",at=J,rt={},it=J,st=Y.spm_protocol,ut="data-spm",ct="data-spm-protocol",lt="data-spm-src",pt="data-spm-click",gt="data-auto-spmd",ft="data-spm-max-idx",dt="data-auto-spmd-max-idx",ht="",_t="data-spm-wangpu-module-id",mt="data-spm-anchor-id",vt="data-spm-ab",bt=r.nodeListToArray,yt=s.tryToGetAttribute,wt=s.tryToSetAttribute,xt=s.tryToRemoveAttribute,jt=i.isStartWith,Tt=i.isString,Pt=i.isArray,At=function(t,e){if(!d&&e)return[t,e].join(".");if(t&&e)return t+".i"+e;var n=W.g_SPM||(W.g_SPM={}),o=n.spm_d_for_ad||{};return"number"==typeof o[t]?o[t]++:o[t]=0,n.spm_d_for_ad=o,t+".i"+o[t]};u.DOMReady(function(){O()}),X||R(),z?u.on(F,"tap",$):u.on(F,"mousedown",$),z?u.on(F,"tap",B):(u.on(F,"mousedown",B),u.on(F,"keydown",B)),W.g_SPM={resetModule:C,anchorBeacon:U,getParam:M,spm:D}}},function(t,e,n){e.confLoad=n(95),e.spmRender=n(96)},function(t,e){t.exports={setMeta:function(){},load:function(t){t()}}},function(t,e){e.renderSpmc=function(){},e.fetchSpmd=function(){}},function(t,e,n){"use strict";var o=n(48),a=n(32),r=n(3),i=n(23),s=n(94),u=s.confLoad,c=i.getInfo(),l=navigator.userAgent,p="complete";e.initGoldlog=function(t){function e(){var e=/TB\-PD/i.test(l),i=n._$=n._$||{};i.meta_info=c,i.is_terminal="aplus_wap"===r.script_name||e||"1"==c["aplus-terminal"],i.send_pv_count=0,i.status=p,i.script_name=r.script_name,n.lver=r.lver,n.nameStorage=o.nameStorage,n.isUT4Aplus=/UT4Aplus/i.test(l),a.doPubMsg(["goldlogReady",p]),a.doCachePubs(["goldlogReady",p]),t.init()}var n=window.goldlog||(window.goldlog={}),i=r.goldlog_path.run.create();for(var s in i)n[s]=i[s];return u.load(e),n}}]);/*! 2018-07-13 17:06:33 v8.5.1 */
!function(t){function e(n){if(o[n])return o[n].exports;var r=o[n]={exports:{},id:n,loaded:!1};return t[n].call(r.exports,r,r.exports,e),r.loaded=!0,r.exports}var o={};return e.m=t,e.c=o,e.p="",e(0)}([function(t,e,o){!function(){function t(t){for(var e=[{name:"youku.com",fn:o(22)},{name:"soku.com",fn:o(22)},{name:"tudou.com",fn:o(27)},{name:"laifeng.com",fn:o(28)}],r=s.getMetaCnt("aplus-urchin2-logrule"),i=0;i<e.length;i++){var a=e[i].name;if(n.isContain(t,a)||a===r)return e[i]}return!1}var e=window.goldlog||(window.goldlog={});if(!e.aplus_urchin2){var n=o(1),r=o(2),i=o(4),a=o(14),s=o(21),u=function(){e.setMetaInfo("aplus-getspmcd",function(t,e,o){var n=r.getSpmCD(t,o);return r.doSendHjlj(t,e,o,n),n})},c=function(t){var e=window.goldlog_queue||(window.goldlog_queue=[]);e.push({action:"goldlog.aplus_pubsub.subscribe",arguments:["goldlogReady",function(e){"complete"===e&&(u(t),i.init())}]});for(var o=function(e,o){return{action:"goldlog.aplus_pubsub.subscribe",arguments:["goldlogReady",function(n){"complete"===n&&t.fn(t.name,e,o)}]}},n=!1,r=0;r<e.length;r++)if("goldlog.sendPV"===e[r].action&&!n){n=!0,e.splice(r,0,o({is_auto:!0}));break}n||e.push(o({is_auto:!0})),a.init_watchGoldlogQueue()},p=t(window.location.hostname);p?p&&"function"==typeof p.fn&&c(p):(a.init_watchGoldlogQueue(),e.sendPV({is_auto:!0}))}}()},function(t,e){"use strict";function o(t,e){return"function"!=typeof Object.assign?function(t){if(null===t)throw new TypeError("Cannot convert undefined or null to object");for(var e=Object(t),o=1;o<arguments.length;o++){var n=arguments[o];if(null!==n)for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}(t,e):Object.assign({},t,e)}function n(t){return"function"==typeof t}function r(t){return"string"==typeof t}function i(t){return"undefined"==typeof t}function a(t,e){return t.indexOf(e)>-1}var s=window;e.assign=o,e.makeCacheNum=function(){return Math.floor(268435456*Math.random()).toString(16)},e.each=function(t,e){var o,n=t.length;for(o=0;o<n;o++)e(t[o])},e.isStartWith=function(t,e){return 0===t.indexOf(e)},e.isEndWith=function(t,e){var o=t.length,n=e.length;return o>=n&&t.indexOf(e)==o-n},e.any=function(t,e){var o,n=t.length;for(o=0;o<n;o++)if(e(t[o]))return!0;return!1},e.isFunction=n,e.isArray=function(t){return Array.isArray?Array.isArray(t):/Array/.test(Object.prototype.toString.call(t))},e.isString=r,e.isNumber=function(t){return"number"==typeof t},e.isUnDefined=i,e.isContain=a;var u=function(t){var e,o=t.constructor===Array?[]:{};if("object"==typeof t){if(s.JSON&&s.JSON.parse)e=JSON.stringify(t),o=JSON.parse(e);else for(var n in t)o[n]="object"==typeof t[n]?u(t[n]):t[n];return o}};e.cloneObj=u,e.cloneDeep=u},function(t,e,o){"use strict";var n=o(3),r={UL:1,OL:2,LI:3,INPUT:4,DIV:5,BODY:6,STRONG:7,SPAN:8,FORM:9,BUTTON:10,CAPTION:11,FIELDSET:12,COLGROUP:13,TFOOT:14,LABEL:15,LEGEND:16,THEAD:17,OPTGROUP:18,OPTION:19,SELECT:20,TABLE:21,TBODY:22,IFRAME:0,SCRIPT:0,OBJECT:0,EMBED:0,IMG:0};e.doSendHjlj=function(t,e,o,r){var i,a;try{var s=function(){var t=window.goldlog||{},n=[o,r.spm_c,r.spm_d],s=n.join(".");i="//gm.mmstat.com/yt/preclk."+s,a={autosend:1,pos_co:"#"+r.spm_c+"~"+r.spm_d,eventtype:e?e.type:""},t.sendPositionCode(i,a)};"0"!==r.spm_c&&e&&u(e)&&s()}catch(t){var c="doSendHjlj error: "+t.message+",spm_ab="+o+",params"+JSON.stringify(a)+",req_path="+i;n.do_tracker_jserror({message:c,error:t})}};var i="HTML",a="BODY",s=function(t){for(var e,o={A:1,INPUT:1,BUTTON:1};t&&t.tagName!==i&&t.tagName!==a&&t.getAttribute;){if(e=t.getAttribute("data-stat-role"),"ck"===e)return t;if(o[t.tagName])return t;if(!(t=t.parentNode))break}return!(!o[t.nodeName]&&(t=t.parentNode||{},!o[t.nodeName]))&&t},u=function(t){var e;e="touchstart"===t.type?t.touches[0]:t;var o,n;return e.target||(e.target=e.srcElement||document),!e.pageX&&e.clientX&&(o=document.documentElement,n=document.body,e.pageX=(o&&o.scrollLeft||n&&n.scrollLeft||0)-(o&&o.clientLeft||n&&n.clientLeft||0),e.pageY=(o&&o.scrollTop||n&&n.scrollTop||0)-(o&&o.clientTop||n&&n.clientTop||0)),!("undefined"==typeof e.pageX||!s(e.target))};e.getSpmCD=function(t){var e={spm_c:"0",spm_d:"0"},o=[];if(t.id)return e.spm_c=t.id,e;for(var n,i;t&&(n=t.nodeName.toUpperCase(),"BODY"!=n);t=t.parentNode){if(t.id&&"i"!=t.getAttribute("cs"))return e.spm_c=t.id,e;i=0;for(var a=t.previousSibling;a;a=a.previousSibling){var s=a.nodeName.toUpperCase();0!==r[s]&&s==n&&++i}n=r[n]||n;var u=i?"!"+(i+1):"";o.unshift(n+u),e.spm_d=o.join("~")}return e}},function(t,e){"use strict";var o=function(t,e){var o=window.goldlog_queue||(window.goldlog_queue=[]);o.push({action:"goldlog._aplus_cplugin_track_deb.monitor",arguments:[{key:"APLUS_PLUGIN_DEBUG",title:"aplus_core",msg:["_error_:methodName="+e+",params="+JSON.stringify(t)],type:"updateMsg",description:e||"aplus_core"}]})},n=function(t,e,o){var n=window.goldlog_queue||(window.goldlog_queue=[]);n.push({action:["goldlog","_aplus_cplugin_m",e].join("."),arguments:[t,o]})};e.do_tracker_jserror=function(t,e){var r="do_tracker_jserror";n(t,r,e),o(t,r)},e.do_tracker_obsolete_inter=function(t,e){var r="do_tracker_obsolete_inter";n(t,r,e),o(t,r)},e.wrap=function(t){if("function"==typeof t)try{t()}catch(t){o({msg:t.message||t},"exception")}finally{}}},function(t,e,o){"use strict";var n=o(5),r=o(12),i=o(10),a=o(1),s=o(3),u=o(13);e.init=function(){var t={},e=goldlog.aplus_pubsub;return e.subscribe("mw_change_pv",function(e){t=e}),goldlog.setPvExtdParam=function(t){var e=window.goldlog||(window.goldlog={});try{var o=e.spm_ab?e.spm_ab.join("."):"0.0";s.do_tracker_obsolete_inter({page:location.hostname+location.pathname,spm_ab:o,interface_name:"goldlog.setPvExtdParam",interface_params:"userdata = "+JSON.stringify(t)})}catch(t){i({msg:t}),s.do_tracker_jserror({message:"goldlog.setPvExtdParam error:"+t.message,error:t})}finally{return!0}},goldlog.sendPositionCode=function(o,i){var s=r.getGoldlogVal("pv_context")||{},c=s.what_to_sendpv||{},p=t.what_to_sendpv||{},l=p.pvdata||c.pvdata||[],d=[];u.updateKey(l,"logtype",2),u.updateKey(l,"cache",a.makeCacheNum());for(var g in i)d.push([g,i[g]]);var f=[].concat(l,d),m=goldlog.send(n.makeUrl(o,f));return e&&e.publish("sendPositionCode",m),m},goldlog.sendPseudo=function(t,e){var o=goldlog._$||{};t||(t={}),t._pseudo_url?o._sendPseudo({"spm-cnt":t._pseudo_spm||"","spm-url":t.element_spm_id||"",_pseudo_url:t._pseudo_url},e):i.logger({msg:"_pseudo_url can not be null!"})},goldlog.sendPV=function(t,e){e=e||{},goldlog.aplus_pubsub.publish("YOUTU_READY_SENDPV",{config:t,userdata:e}),goldlog.aplus_pubsub.cachePubs("YOUTU_READY_SENDPV",{config:t,userdata:e})},(window.goldlog_queue||(window.goldlog_queue=[])).push({action:"goldlog.aplus_pubsub.subscribe",arguments:["PLUGIN_DO_SENDPV",function(t){t&&t.config&&t.userdata&&goldlog._$._sendPV(t.config,t.userdata)}]}),goldlog}},function(t,e,o){"use strict";function n(t){t=(t||"").split("#")[0].split("?")[0];var e=t.length,o=function(t){var e,o=t.length,n=0;for(e=0;e<o;e++)n=31*n+t.charCodeAt(e);return n};return e?o(e+"#"+t.charCodeAt(e-1)):-1}function r(t){for(var e=t.split("&"),o=0,n=e.length,r={};o<n;o++){var i=e[o],a=i.indexOf("="),s=i.substring(0,a),u=i.substring(a+1);r[s]=l.tryToDecodeURIComponent(u)}return r}function i(t){if("function"!=typeof t)throw new TypeError(t+" is not a function");return t}function a(t){var e,o,n,r=[],i=t.length;for(n=0;n<i;n++)e=t[n][0],o=t[n][1],r.push(p.isStartWith(e,v)?o:e+"="+encodeURIComponent(o));return r.join("&")}function s(t){var e,o,n,r={},i=t.length;for(n=0;n<i;n++)e=t[n][0],o=t[n][1],r[e]=o;return r}function u(t,e){var o,n,r,i=[];for(o in t)t.hasOwnProperty(o)&&(n=""+t[o],r=o+"="+encodeURIComponent(n),e?i.push(r):i.push(p.isStartWith(o,v)?n:r));return i.join("&")}function c(t,e){var o=t.indexOf("?")==-1?"?":"&",n=e?p.isArray(e)?a(e):u(e):"";return n?t+o+n:t}var p=o(1),l=o(6),d=o(8),g=parent!==self;e.is_in_iframe=g,e.makeCacheNum=p.makeCacheNum,e.isStartWith=p.isStartWith,e.isEndWith=p.isEndWith,e.any=p.any,e.each=p.each,e.assign=p.assign,e.isFunction=p.isFunction,e.isArray=p.isArray,e.isString=p.isString,e.isNumber=p.isNumber,e.isUnDefined=p.isUnDefined,e.isContain=p.isContain,e.sleep=o(9).sleep,e.makeChkSum=n,e.tryToDecodeURIComponent=l.tryToDecodeURIComponent,e.nodeListToArray=l.nodeListToArray,e.parseSemicolonContent=l.parseSemicolonContent,e.param2obj=r;var f=o(10),m=function(t){return/^(\/\/){0,1}(\w+\.){1,}\w+((\/\w+){1,})?$/.test(t)};e.hostValidity=m;var y=function(t,e){var o=/^(\/\/){0,1}(\w+\.){1,}\w+\/\w+\.gif$/.test(t),n=m(t),r="";return o?r="isGifPath":n&&(r="isHostPath"),r||f.logger({msg:e+": "+t+' is invalid, suggestion: "xxx.mmstat.com"'}),r},_=function(t){return!/^\/\/gj\.mmstat/.test(t)&&goldlog.isInternational()&&(t=t.replace(/^\/\/\w+\.mmstat/,"//gj.mmstat")),t};e.filterIntUrl=_,e.getPvUrl=function(t){t||(t={});var e,o,n=t.metaValue&&y(t.metaValue,t.metaName),r="";"isGifPath"===n?(e=/^\/\//.test(t.metaValue)?"":"//",r=e+t.metaValue):"isHostPath"===n&&(e=/^\/\//.test(t.metaValue)?"":"//",o=/\/$/.test(t.metaValue)?"":"/",r=e+t.metaValue+o+t.gifPath);var i;return r?i=r:(e=0===t.gifPath.indexOf("/")?t.gifPath:"/"+t.gifPath,i=t.url&&t.url.replace(/\/\w+\.gif/,e)),i},e.indexof=o(11).indexof,e.callable=i;var v="::-plain-::";e.mkPlainKey=function(){return v+Math.random()},e.s_plain_obj=v,e.mkPlainKeyForExparams=function(t){var e=t||v;return e+"exparams"},e.rndInt32=function(){return Math.round(2147483647*Math.random())},e.arr2param=a,e.arr2obj=s,e.obj2param=u,e.makeUrl=c,e.ifAdd=function(t,e){var o,n,r,i,a=e.length;for(o=0;o<a;o++)n=e[o],r=n[0],i=n[1],i&&t.push([r,i])},e.isStartWithProtocol=d.isStartWithProtocol,e.param2arr=function(t){for(var e,o=t.split("&"),n=0,r=o.length,i=[];n<r;n++)e=o[n].split("="),i.push([e.shift(),e.join("=")]);return i}},function(t,e,o){"use strict";function n(t,e){var o=e||"";if(t)try{o=decodeURIComponent(t)}catch(t){}return o}var r=o(7);e.tryToDecodeURIComponent=n,e.parseSemicolonContent=function(t,e,o){e=e||{};var i,a,s=t.split(";"),u=s.length;for(i=0;i<u;i++){a=s[i].split("=");var c=r.trim(a.slice(1).join("="));e[r.trim(a[0])||""]=o?c:n(c)}return e},e.nodeListToArray=function(t){var e,o;try{return e=[].slice.call(t)}catch(r){e=[],o=t.length;for(var n=0;n<o;n++)e.push(t[n]);return e}},e.nodeListToArray=function(t){var e,o;try{return e=[].slice.call(t)}catch(r){e=[],o=t.length;for(var n=0;n<o;n++)e.push(t[n]);return e}};var i={set:function(t,e){try{return localStorage.setItem(t,e),!0}catch(t){return!1}},get:function(t){return localStorage.getItem(t)},test:function(){var t="grey_test_key";try{return localStorage.setItem(t,1),localStorage.removeItem(t),!0}catch(t){return!1}},remove:function(t){localStorage.removeItem(t)}};e.store=i,e.getLsCna=function(t,e){var o="",n=i.get(t);if(n){var r=n.split("_")||[];o=e?r.length>1&&e===r[0]?r[1]:"":r.length>1?r[1]:""}return decodeURIComponent(o)},e.setLsCna=function(t,e,o){o&&i.set&&i.test()&&i.set(t,e+"_"+encodeURIComponent(o))},e.getUrl=function(t){var e=t||"//log.mmstat.com/eg.js";try{var o=goldlog.getMetaInfo("aplus-rhost-v"),n=/[[a-z|0-9\.]+[a-z|0-9]/,r=o.match(n);r&&r[0]&&(e=e.replace(n,r[0]))}catch(t){}return e}},function(t,e){"use strict";function o(t){return"string"==typeof t?t.replace(/^\s+|\s+$/g,""):""}e.trim=o},function(t,e,o){"use strict";var n=o(1),r=function(){var t=location.protocol;return"http:"!==t&&"https:"!==t&&(t="https:"),t};e.getProtocal=r,e.isStartWithProtocol=function(t){for(var e=["javascript:","tel:","sms:","mailto:","tmall://","#"],o=0,r=e.length;o<r;o++)if(n.isStartWith(t,e[o]))return!0;return!1}},function(t,e){"use strict";e.sleep=function(t,e){return setTimeout(function(){e()},t)}},function(t,e){"use strict";var o=function(){var t=!1;return"boolean"==typeof goldlog.aplusDebug&&(t=goldlog.aplusDebug),t};e.isDebugAplus=o;var n=function(t){t||(t={});var e=t.level||"warn";window.console&&window.console[e]&&window.console[e](t.msg)};e.logger=n},function(t,e){"use strict";e.indexof=function(t,e){var o=-1;try{o=t.indexOf(e)}catch(r){for(var n=0;n<t.length;n++)t[n]===e&&(o=n)}finally{return o}}},function(t,e){"use strict";var o=function(t){var e;try{window.goldlog||(window.goldlog={}),e=window.goldlog[t]}catch(t){e=""}finally{return e}};e.getGoldlogVal=o;var n=function(t,e){var o=!1;try{window.goldlog||(window.goldlog={}),t&&(window.goldlog[t]=e,o=!0)}catch(t){o=!1}finally{return o}};e.setGoldlogVal=n,e.getClientInfo=function(){return o("_aplus_client")||{}}},function(t,e){"use strict";function o(t,e,o){i(t,"spm-cnt",function(t){var n=t.split(".");return n[0]=goldlog.spm_ab[0],n[1]=goldlog.spm_ab[1],e?n[1]=n[1].split("/")[0]+"/"+e:n[1]=n[1].split("/")[0],o&&(n[4]=o),n.join(".")})}function n(t,e){var o=window.g_SPM&&g_SPM._current_spm;o&&i(t,"spm-url",function(){return[o.a,o.b,o.c,o.d].join(".")+(e?"."+e:"")},"spm-cnt")}function r(t,e){var o,n,r,i=-1;for(o=0,n=t.length;o<n;o++)if(r=t[o],r[0]===e){i=o;break}i>=0&&t.splice(i,1)}function i(t,e,o,n){var r,i,a=t.length,s=-1,u="function"==typeof o;for(r=0;r<a;r++){if(i=t[r],i[0]===e)return void(u?i[1]=o(i[1]):i[1]=o);n&&i[0]===n&&(s=r)}n&&(u&&(o=o()),s>-1?t.splice(s,0,[e,o]):t.push([e,o]))}t.exports={updateSPMCnt:o,updateSPMUrl:n,updateKey:i,removeKey:r}},function(t,e,o){"use strict";function n(t){for(var e=[],o=[],n=[],r=[],i=[],a=[],s={};s=t.shift();)try{var u=s.action,c=s.arguments[0];/subscribe/.test(u)?"setMetaInfo"===c?o.push(s):"mw_change_pv"===c||"mw_change_hjlj"===c?e.push(s):n.push(s):/MetaInfo/.test(u)?r.push(s):i.push(s)}catch(t){i.push(s),p.do_tracker_jserror({message:t&&t.message,error:encodeURIComponent(t.stack),filename:"sortQueue"})}return a=e.concat(r),a=a.concat(n),a=a.concat(o,i)}var r=window,i=o(1),a=o(15),s=o(10),u=o(16),c=o(17),p=o(3),l=s.isDebugAplus();e.init_aplusQueue=function(){var t,e="_ap",o=r[e]=r[e]||[];o.push=t=function(){var t="0.0";window.goldlog&&window.goldlog.spm_ab&&(t=window.goldlog.spm_ab.join(".")),p.do_tracker_obsolete_inter({ratio:l?1:.01,page:location.hostname+location.pathname,spm_ab:t,interface_name:"win._ap",interface_params:JSON.stringify(arguments)});for(var e,n,r=0,s=arguments.length;r<s;r++)e=arguments[r],i.isString(e)?goldlog.send(a.hjlj()+e):i.isArray(e)&&"push"!=(n=e[0])&&(o[n]=o[n]||[]).push(e.slice(1))};for(var n;n=o.shift();)t(n)};var d="goldlog_queue",g=function(t,e,o){try{/_aplus_cplugin_track_deb/.test(t)||/_aplus_cplugin_m/.test(t)||p.do_tracker_jserror({message:o||'illegal task: goldlog_queue.push("'+JSON.stringify(e)+'")',error:JSON.stringify(e),filename:"processTask"})}catch(t){}},f=function(t,e){var o=t?t.action:"",n=t?t.arguments:"";try{if(o&&n&&i.isArray(n)){var a=o.split("."),s=r,u=r;if(3===a.length)s=r[a[0]][a[1]]||{},u=s[a[2]]?s[a[2]]:"";else for(;a.length;)if(u=s=s[a.shift()],!s)return void("function"==typeof e?e(t):g(o,t));"function"==typeof u&&u.apply(s,n)}else g(o,t)}catch(e){g(o,t,e.message)}},m=function(t){function e(){if(t&&i.isArray(t)&&t.length){for(var e=n(t),o={},r=[];o=e.shift();)f(o,function(t){r.push(t)});r.length>0&&setTimeout(function(){for(;o=r.shift();)f(o)},100)}}try{e()}catch(t){p.do_tracker_jserror({message:t&&t.message,error:encodeURIComponent(t.stack),filename:"processGoldlogQueue"})}};e.processGoldlogQueue=m;var y=u.extend({push:function(t){this.length++,f(t)}});e.init_watchGoldlogQueue=function(){c.init_loadAplusPlugin();try{var t=r[d]||[];r[d]=y.create({startLength:t.length,length:0}),m(t)}catch(t){p.do_tracker_jserror({message:t&&t.message,error:encodeURIComponent(t.stack),filename:"init_watchGoldlogQueue"})}}},function(t,e,o){"use strict";var n=o(8);e.hjlj=function(){var t=window.goldlog||(window.goldlog={}),e=t._$||{},o=e.script_name,r=e.meta_info||{},i=r["aplus-rhost-g"],a="//gm.mmstat.com/";return(e.is_terminal||"aplus_wap"===o)&&(a="//wgo.mmstat.com/"),"aplus_int"===o&&(a="//gj.mmstat.com/"),i&&(a="//"+i+"/"),n.getProtocal()+a}},function(t,e){"use strict";function o(){}o.prototype.extend=function(){},o.prototype.create=function(){},o.extend=function(t){return this.prototype.extend.call(this,t)},o.prototype.create=function(t){var e=new this;for(var o in t)e[o]=t[o];return e},o.prototype.extend=function(t){var e=function(){};try{"function"!=typeof Object.create&&(Object.create=function(t){function e(){}return e.prototype=t,new e}),e.prototype=Object.create(this.prototype);for(var o in t)e.prototype[o]=t[o];e.prototype.constructor=e,e.extend=e.prototype.extend,e.create=e.prototype.create}catch(t){console.log(t)}finally{return e}},t.exports=o},function(t,e,o){"use strict";var n=o(18),r=o(19),i=o(20),a=function(t,e){var o=r.getMetaCnt(t);return!(!o&&!e)},s=function(){var t=goldlog.getCdnPath();return{aplus_ae_path:t+"/alilog/s/"+i.lver+"/plugin/aplus_ae.js",aplus_ac_path:t+"/alilog/s/"+i.lver+"/plugin/aplus_ac.js"}},u=function(t,e){var o=s(),r=a(t,e),i={"aplus-auto-exp":o.aplus_ae_path,"aplus-auto-clk":o.aplus_ac_path};r&&i[t]&&n.addScript(i[t])};e.init_loadAplusPlugin=function(){!goldlog._aplus_auto_exp&&u("aplus-auto-exp"),!goldlog._aplus_ac&&u("aplus-auto-clk"),goldlog.aplus_pubsub.subscribe("setMetaInfo",function(t,e){"aplus-auto-exp"!==t||goldlog._aplus_auto_exp||u(t,e),"aplus-auto-clk"!==t||goldlog._aplus_ac||u(t,e)})}},function(t,e,o){"use strict";function n(t,e){return t&&t.getAttribute?t.getAttribute(e)||"":""}function r(t,e,o){if(t&&t.setAttribute)try{t.setAttribute(e,o)}catch(t){}}function i(t,e){if(t&&t.removeAttribute)try{t.removeAttribute(e)}catch(o){r(t,e,"")}}function a(t,e,o){var n="script",r=d.createElement(n);r.type="text/javascript",r.async=!0;var i="https:"==location.protocol?e||t:t;0===i.indexOf("//")&&(i=u.getProtocal()+i),r.src=i,o&&(r.id=o);var a=d.getElementsByTagName(n)[0];s=s||d.getElementsByTagName("head")[0],a?a.parentNode.insertBefore(r,a):s&&s.appendChild(r)}var s,u=o(8),c=o(7),p=o(1),l=o(10),d=document;e.tryToGetAttribute=n,e.tryToSetAttribute=r,e.tryToRemoveAttribute=i,e.addScript=a,e.loadScript=function(t,e){function o(t){n.onreadystatechange=n.onload=n.onerror=null,n=null,e(t)}var n=d.createElement("script");if(s=s||d.getElementsByTagName("head")[0],n.async=!0,"onload"in n)n.onload=o;else{var r=function(){/loaded|complete/.test(n.readyState)&&o()};n.onreadystatechange=r,r()}n.onerror=function(t){o(t)},n.src=t,s.appendChild(n)},e.isTouch=function(){return"ontouchend"in document.createElement("div")},e.tryToGetHref=function(t){var e;try{e=c.trim(t.getAttribute("href",2))}catch(t){}return e||""};var g=function(){var t=goldlog&&goldlog._$?goldlog._$:{},e=t.meta_info||{};return e["aplus-exparams"]||""};e.getExParamsFromMeta=g,e.getExParams=function(t){var e=d.getElementById("beacon-aplus")||d.getElementById("tb-beacon-aplus"),o=n(e,"exparams"),r=f(o,g(),t)||"";return r&&r.replace(/&amp;/g,"&").replace(/\buserid=/,"uidaplus=")};var f=function(t,e,o){var n="aplus&sidx=aplusSidex",r=t||n;try{if(e){var i=o.param2obj(e),a=["aplus","cna","spm-cnt","spm-url","spm-pre","logtype","pre","uidaplus","asid","sidx","trid","gokey"];p.each(a,function(t){i.hasOwnProperty(t)&&(l.logger({msg:"Can not inject keywords: "+t}),delete i[t])}),delete i[""];var s="";if(t){var u=t.match(/aplus&/).index,c=u>0?o.param2obj(t.substring(0,u)):{};delete c[""],s=o.obj2param(p.assign(c,i))+"&"+t.substring(u,t.length)}else s=o.obj2param(i)+"&"+n;return s}return r}catch(t){return r}};e.mergeExparams=f},function(t,e,o){"use strict";function n(t){return a=a||document.getElementsByTagName("head")[0],s&&!t?s:a?s=a.getElementsByTagName("meta"):[]}function r(t){var e,o,r,i=n(),a=i.length;for(e=0;e<a;e++)o=i[e],u.tryToGetAttribute(o,"name")===t&&(r=u.tryToGetAttribute(o,"content"));return r||""}function i(t){var e={isonepage:"-1",urlpagename:""},o=t.qGet();if(o&&o.hasOwnProperty("isonepage_data"))e.isonepage=o.isonepage_data.isonepage,e.urlpagename=o.isonepage_data.urlpagename;else{var n=r("isonepage")||"-1",i=n.split("|");e.isonepage=i[0],e.urlpagename=i[1]?i[1]:""}return e}var a,s,u=o(18);e.getMetaTags=n,e.getMetaCnt=r,e.getOnePageInfo=i},function(t,e){"use strict";e.lver="8.5.1",e.toUtVersion="v20180713",e.script_name="aplus_urchin2"},function(t,e){"use strict";e.getPvid=function(t){var e=["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],o=0,n="",r=new Date;for(o=0;o<t;o++){var i=parseInt(Math.random()*Math.pow(10,6))%e.length;n+=e[i]}return r.getTime()+n},e.getMetaCnt=function(t){var e="";if(t&&document.querySelector){var o=document.querySelector("[name="+t+"]");o&&(e=o.getAttribute("content"))}return e}},function(t,e,o){"use strict";var n=o(23),r=o(1),i=o(24),a=o(21),s=o(26);t.exports=function(t,e,o){var u=n(),c=r.assign(u,{params:{yft:""},urchinTracker:function(t,e,o){var n=this,r=n.params||{};r.ysuid=n._yCookie("__ysuid")||"",r.yuid=n._yCookie("_l_lgi")||0,n._yCookie("__ayft")?r.yft=n._yCookie("__ayft"):r.yft=(new Date).valueOf();var i=!1;n._yCookie("__aysid")?r.ysid=n._yCookie("__aysid"):(r.ysid=n._yGetPvid(3),i=!0),r.pvid=n._yGetPvid(6),r.rpvid=n._yGetRPvid(),r.ycid=window.cateStr||"",r.rycid=n._yCookie("__arycid")||"",r.ycms=window.cateStr||"",r.rcms=n._yCookie("__arcms")||"",r.unc=navigator.cookieEnabled?0:1,r.frame=top.location!=self.location?1:0,r.ikuins=n._yGetIkuId(),r.dev=navigator.platform||"",r.mtype=n._yGetMType(),r.from=n._yGetQueryString("from"),r.abt=n._yGetMType(),r.cpid=window._stat_topics_cpid||"";var a=n._yCookie("__ayscnt");if(i){var s=parseInt(a)||0;r.yscnt=s+1}else r.yscnt=parseInt(a)||1;var u=parseInt(n._yCookie("__aypstp"))||0;r.ypstp=u+1;var c=parseInt(n._yCookie("__ayspstp"))||0;r.yspstp=c+1,r.yvstp=parseInt(n._yCookie("__ayvstp"))||0,r.ysvstp=parseInt(n._yCookie("__aysvstp"))||0,this.params=r,e||(e={}),n._yInfo(t.firstTime,e,o),n._yReset()},_yReset:function(){var t=this,e=t.params;t._yCookie("__ayft",e.yft,{expires:"session"}),t._yCookie("__aysid",e.ysid,{expires:2}),t._yCookie("__arpvid",e.pvid+"-"+(new Date).getTime(),{expires:"session"}),t._yCookie("__arycid",e.ycid,{expires:"session"}),t._yCookie("__ayscnt",parseInt(e.yscnt),{expires:"session"}),t._yCookie("__arcms",e.ycid,{expires:"session"}),t._yCookie("__aypstp",parseInt(e.ypstp),{expires:"session"}),t._yCookie("__ayspstp",parseInt(e.yspstp),{expires:2})}});s.setSpmAB(location.hostname);var p=function(e,o,n){var s,u=r.assign({l_v:3,p_v:3,dmid:1},n||{}),p=i.getCookie("P_gck");"youku.com"===t?(s=i.getCookie("__ysuid"),s||(s=a.getPvid(3),i.setCookie("__ysuid",s))):"soku.com"===t&&(s=i.getCookie("SOKUSESSID")),u.pc_i=s||"",u.pu_i=p||"",c.appendExinfo(u),c.urchinTracker(r.assign({firstTime:"yft",yuid:"P_gck"},e),o,u)};p({},e||{is_auto:!1},o||{});var l=window.goldlog||(window.goldlog={});l.aplus_urchin2=c,window.UrchinAplus=c,l.aplus_pubsub.subscribe("YOUTU_READY_SENDPV",function(t){"object"==typeof t&&p({is_auto:!1},t.config,t.userdata)})}},function(t,e,o){"use strict";function n(t,e){return t+Math.floor(Math.random()*(e-t+1))}var r,i=o(24),a=o(10),s=window.navigator.userAgent,u=function(t){var e=document.createElement("script"),o=(new Date).getTime();t+="&_"+o,e.setAttribute("src",t),document.getElementsByTagName("body")[0].appendChild(e)};t.exports=function(){return{params:{ysuid:"",yuid:"",yft:"",ysid:"",pvid:"",rpvid:"",ycid:"",rycid:"",ypstp:"",yspstp:"",yscnt:"",ycms:"",rcms:"",unc:"",frame:"",ikuins:"",dev:"",mtype:"",from:"",abt:"",cpid:"",yvstp:"",ysvstp:""},urchinTracker:function(t,e,o){var a=this,s=a._yGetPvid(6),c=function(){a.params.ycid=window.cateStr||"",a.params.yuid=a._yCookie(t.yuid),a.params.ycms=window.cateStr||"",a.params.unc=navigator.cookieEnabled?0:1,a.params.frame=top.location!=self.location?1:0,a.params.ikuins=a._yGetIkuId(),a.params.dev=navigator.platform||"",a.params.mtype=a._yGetMType(),a.params.from=a._yGetQueryString("from"),a.params.abt=a._yGetMType(),a.params.cpid=window._stat_topics_cpid||"",a.params.yvstp=parseInt(a._yCookie("__ayvstp"))||0,a.params.ysvstp=parseInt(a._yCookie("__aysvstp"))||0,e||(e={}),a._yInfo(t.firstTime,e,o)};if(r)c();else{var p=(new Date).getTime(),l=n(1e4,99999),d="sck_callback"+p+"_"+l;goldlog[d]=function(e){r=!0;var o=e.YOUKUSESSID||"",n=e.CNA||"";o&&a._yCookie("__ysuid",o,{domain:t.domain,expires:"session"}),n&&i.setCookie("cna",n,{domain:t.domain}),a.params.ysuid=e.YOUKUSESSID,a.params[t.firstTime]=e[t.firstTime],a.params.ysid=e.ysid,a.params.pvid=s,a.params.rpvid=e.rpvid,a.params.rycid=e.rcid,a.params.ypstp=e.pstp,a.params.yspstp=e.spstp,a.params.yscnt=e.scnt,goldlog[d]="",c()},u("//lstat.youku.com/sck.php?pvid="+s+"&jsoncallback=goldlog."+d)}},_yCookie:function(t,e,o){if(t){if(!e||"undefined"==typeof e){var n=i.getCookie(t);return this._yTrim(decodeURIComponent(n))}i.setCookie(t,encodeURIComponent(e),o)}},_yInfoBase:function(t){var e=this,o=e.params||{};return t||(t=[]),t.push("ysid="+o.ysid),t.push("pvid="+o.pvid),t.push("rpvid="+o.rpvid),t.push("ycid="+o.ycid),t.push("rycid="+o.rycid),t.push("ypstp="+o.ypstp),t.push("yspstp="+o.yspstp),t.push("yscnt="+o.yscnt),t.push("ycms="+o.ycms),t.push("rcms="+o.rcms),t.push("unc="+o.unc),t.push("frame="+o.frame),t.push("ikuins="+o.ikuins),t.push("dev="+o.dev),t.push("mtype="+o.mtype),t.push("from="+o.from),t.push("abt="+o.abt),t.push("cpid="+o.cpid),t.join("&")},_yInfo:function(t,e,o){var n=this,r=n.params||{};try{var i=[t+"="+r[t]],s=n._yInfoBase(i);o||(o={}),o.extd=s,goldlog.aplus_pubsub.publish("PLUGIN_DO_SENDPV",{plugin_name:"aplus_urchin2",config:e,userdata:o})}catch(t){a.logger({msg:t})}},_yVvlogInfo:function(){var t=this,e=t.params;e.ysvstp=parseInt(e.ysvstp)+1,e.yvstp=parseInt(e.yvstp)+1;var o={};return o.pc_i=e.ysuid,o.pc_u=e.yuid,o.yvft=e.yft,o.seid=e.ysid,o.svstp=e.ysvstp,o.vsidc=e.yscnt,o.vstp=e.yvstp,o.pvid=e.pvid,o.rvpvid=e.rpvid,o.ycid=e.ycid,o.rycid=e.rycid,t.params=e,t._yResetVV(),o},_yResetVV:function(){var t=this,e=t.params;t._yCookie("__ayvstp",parseInt(e.yvstp),{expires:"session"}),t._yCookie("__aysvstp",parseInt(e.ysvstp),{expires:2})},_yTrim:function(t){return t=t.replace(/(\s*|　*)$/,""),t=t.replace(/^(\s*|　*)/,"")},_yGetPvid:function(t){var e=["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],o=0,n="",r=new Date;for(o=0;o<t;o++){var i=parseInt(Math.random()*Math.pow(10,6))%e.length;n+=e[i]}return r.getTime()+n},_yGetRPvid:function(){try{var t=this._yCookie("__arpvid")||"";return""==t?"":(t=t.split("-"),t[0]||"")}catch(t){return""}},_yGetIkuId:function(){var t=null;if(s.indexOf("MSIE")!=-1&&window.ActiveXObject)try{t=new window.ActiveXObject("iKuAgent.KuAgent2")}catch(t){}return t?t.Youku_Hao:0},_yGetMType:function(){var t="";return t=s.indexOf("Android")!==-1?"adr":s.indexOf("iPad")!==-1?"ipa":s.indexOf("iPhone")!==-1?"iph":s.indexOf("iPod")!==-1?"ipo":"oth"},_yGetQueryString:function(t){var e=new RegExp("(^|&)"+t+"=([^&]*)(&|$)"),o=window.location.search.substr(1).match(e);return null!==o?unescape(o[2]):""},_yGetCna:function(){var t=i.getCookie("cna")||"";return t},_yGetRandStr:function(t){if("number"!=typeof t||t<=0)return"";var e=this._yGetPvid(t);return e},isAppendExinfo:!0,appendExinfo:function(t){if(this.isAppendExinfo){var e=t.pc_i||"",o=t.pu_i||"";e&&o&&(this.isAppendExinfo=!1);var n=window.goldlog_queue||(window.goldlog_queue=[]);n.push({action:"goldlog.appendMetaInfo",arguments:["aplus-exinfo","pc_i="+e+"&pu_i="+o]})}}}}},function(t,e,o){"use strict";function n(t){var e=s.cookie.match(new RegExp("(?:^|;)\\s*"+t+"=([^;]+)"));return e?e[1]:""}function r(t,e,o){o||(o={});var r=new Date;return"session"===o.expires||(o.expires&&("number"==typeof o.expires||o.expires.toUTCString)?("number"==typeof o.expires?r.setTime(r.getTime()+24*o.expires*60*60*1e3):r=o.expires,e+="; expires="+r.toUTCString()):(r.setTime(r.getTime()+63072e7),e+="; expires="+r.toUTCString())),e+="; path="+(o.path?o.path:"/"),e+="; domain="+o.domain,s.cookie=t+"="+e,n(t)}function i(t,e,o){try{if(o||(o={}),o.domain)r(t,e,o);else for(var n=c.getDomains(),i=0;i<n.length;)o.domain=n[i],r(t,e,o)?i=n.length:i++}catch(t){}}function a(){var t={};return u.each(l,function(e){t[e]=n(e)}),t.cnaui=/\btanx\.com$/.test(p)?n("cnaui"):"",t}var s=document,u=o(1),c=o(25),p=location.hostname;e.getCookie=n,e.setCookie=i;var l=["tracknick","thw","cna"];e.getData=a,e.getHng=function(){return encodeURIComponent(n("hng")||"")}},function(t,e){"use strict";e.getDomains=function(){var t=[];try{for(var e=location.hostname,o=e.split("."),n=2;n<=o.length;)t.push(o.slice(o.length-n).join(".")),n++}catch(t){}return t}},function(t,e,o){"use strict";var n=o(3);e.setSpmAB=function(t){var e,o,r,i,a={"pd.youku.com":"pd","tv.youku.com":"tv","fun.youku.com":"fu","heyi.youku.com":"hy","game.youku.com":"gm","tvs.youku.com":"ts","mapp.youku.com":"mp","zipindao.youku.com":"zp","hz.youku.com":"hz","iku.youku.com":"ik","fashion.youku.com":"fs","live.youku.com":"lv","static.youku.com":"st","faxian.youku.com":"fx","laiwan.youku.com":"lw","yuanxian.youku.com":"yx","zy.youku.com":"zy","vr.youku.com":"vr","dv.youku.com":"dv","paike.youku.com":"pk","i.youku.com":"i9","guanghe.youku.com":"gh","movie.youku.com":"mv","news.youku.com":"nw","svip.youku.com":"sv","comic.youku.com":"cm","mobile.youku.com":"mb","finance.youku.com":"fc","3g.youku.com":"3g","tech.youku.com":"tc","child.youku.com":"ch","original.youku.com":"or","jilupian.youku.com":"jl","auto.youku.com":"at","music.youku.com":"ms","top.youku.com":"tp","life.youku.com":"lf","gongyi.youku.com":"gy","travel.youku.com":"tr","www.youku.com":"ww","baby.youku.com":"bb","sports.youku.com":"sp","edu.youku.com":"ed","epg.youku.com":"ep","c.youku.com":"c9","ent.youku.com":"et","culture.youku.com":"32"};try{var s=goldlog.spm_ab||[],u=s[0]||"0",c=s[1]||"0";"0"!==u&&"0"!==c||(e=a[t]||"0","0"===e&&/^https?:\/\/youku.com/.test(location.href)&&(e=a["www.youku.com"]),o=window.pageIdNum,o=/^\d{1,}$/.test(o)?parseInt(o):"0","0"!==e&&"0"!==o&&(r="a2h"+e,i=2e7+o,goldlog.setPageSPM(r+"",i+"")))}catch(a){try{var p="autoSpmAB.setSpmAB error hostname="+t;p+=",spm_a="+e,p+=",spm_b="+o,p+=",temp_spma="+r,p+=",temp_spmab="+i+" !"+a.message,n.do_tracker_jserror({message:p,error:a})}catch(t){n.do_tracker_jserror({message:"autoSpmAB.setSpmAB trycatch error",error:t})}}}},function(t,e,o){"use strict";var n=o(23),r=o(1),i=o(24);t.exports=function(t,e,o){var a=n(),s=r.assign(a,{params:{ysuid:"",yuid:"",tft:"",ysid:"",pvid:"",rpvid:"",ycid:"",rycid:"",ypstp:"",yspstp:"",yscnt:"",ycms:"",rcms:"",unc:"",frame:"",ikuins:"",dev:"",mtype:"",from:"",abt:"",cpid:"",yvstp:"",ysvstp:""}}),u=function(t,e,o){var n=r.assign({l_v:3,p_v:3,dmid:1},o||{});n.pc_i=i.getCookie("__ysuid"),n.pu_i=i.getCookie("P_gck"),s.appendExinfo(n),s.urchinTracker(r.assign({domain:".tudou.com",firstTime:"tft",yuid:"P_gck"},t),e,n)};u({},e||{is_auto:!1},o||{});var c=window.goldlog||(window.goldlog={});c.aplus_urchin2=s,window.UrchinAplus=s,c.aplus_pubsub.subscribe("YOUTU_READY_SENDPV",function(t){"object"==typeof t&&u({is_auto:!1},t.config,t.userdata)})}},function(t,e,o){"use strict";var n=o(23),r=o(1),i=o(24),a=o(21);t.exports=function(t,e,o){var s=n(),u=r.assign(s,{params:{ysuid:"",yuid:"",lft:"",ysid:"",pvid:"",rpvid:"",ycid:"",rycid:"",ypstp:"",yspstp:"",yscnt:"",ycms:"",rcms:"",unc:"",frame:"",ikuins:"",dev:"",mtype:"",from:"",abt:"",cpid:"",yvstp:"",ysvstp:""}}),c=function(t,e,o){var n=r.assign({l_v:3,p_v:3,dmid:1},o||{}),s=i.getCookie("__ysuid");s||(s=a.getPvid(3),i.setCookie("__ysuid",s)),n.pc_i=s,n.pu_i=i.getCookie("P_gck"),u.appendExinfo(n),u.urchinTracker(r.assign({domain:".laifeng.com",firstTime:"lft",yuid:"P_gck"},t),e,n)};c({},e||{is_auto:!1},o||{});var p=window.goldlog||(window.goldlog={});p.aplus_urchin2=u,window.UrchinAplus=u,p.aplus_pubsub.subscribe("YOUTU_READY_SENDPV",function(t){"object"==typeof t&&c({is_auto:!1},t.config,t.userdata)})}}]);</script><script src="//g.alicdn.com/alilog/mlog/aplus_o.js"></script>
  <div class="window"> 
   <div class="schedule-wrap back-p3"> 
    <div class="mod-tab-wrap schedule-tab"> 
     <ul class="tab-nav clearfix"> 
      <li class=""> <a rel="1" href="javascript:void(0);">淘汰赛</a> </li> 
      <li class="current"> <a rel="2" href="javascript:void(0);" data-spm-anchor-id="a2h8q.11643819.0.0">按日期展示</a> </li> 
      <li> <a rel="3" href="javascript:void(0);">按组别展示</a> </li> 
     </ul> 
     <div class="tab-panel tab-1" style="display: none;"> 
      <div class="schedule-vs"><div class="wrap-svg"></div> 
       <div class="group-item-bg"> 
        <span class="group-item-title">世界杯决赛</span> 
       </div> 
       <div class="vs-item vs-4">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000975&amp;matchId=49" data-showcode="7feeedff60f046b0ba1f" data-showid="405066" data-livestatus="2" class="match-card status2  team-94  team-96" id="match49">
<h2>
1/8决赛 6月30日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_7feeedff60f046b0ba1f"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-96">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E966A35FB10B250A7DC7">
阿根廷        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000976&amp;matchId=50" data-showcode="5be891eea5e5433aa6e4" data-showid="405067" data-livestatus="2" class="match-card status2  team-84  team-87" id="match50">
<h2>
1/8决赛 7月1日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5be891eea5e5433aa6e4"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-87">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF28A118830A9A0070B1">
葡萄牙        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000979&amp;matchId=53" data-showcode="ee40c336522f42e3b2c1" data-showid="405070" data-livestatus="2" class="match-card status2  team-101  team-105" id="match53">
<h2>
1/8决赛 7月2日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_ee40c336522f42e3b2c1"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-105">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEF4A118830AB00CE1F0">
墨西哥        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000980&amp;matchId=54" data-showcode="38c7b4f64855400fa9f9" data-showid="405071" data-livestatus="2" class="match-card status2  team-108  team-113" id="match54">
<h2>
1/8决赛 7月3日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_38c7b4f64855400fa9f9"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-113">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF40A35FB10B20025E80">
日本        
</li>
</ul>
</div>
    
</div> 
       <div class="vs-item vs-2 vs-2-top">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000983&amp;matchId=57" data-showcode="04d9f025cc914809a660" data-showid="405074" data-livestatus="2" class="match-card status2  team-84  team-94" id="match57">
<h2>
1/4决赛 7月6日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_04d9f025cc914809a660"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000984&amp;matchId=58" data-showcode="818dd7284d7b4f1fb922" data-showid="405075" data-livestatus="2" class="match-card status2  team-101  team-108" id="match58">
<h2>
1/4决赛 7月7日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_818dd7284d7b4f1fb922"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时        
</li>
</ul>
</div>
    
</div> 
       <div class="vs-item vs-1 vs-1-top">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000987&amp;matchId=61" data-showcode="5882db0d0cb64392b1a5" data-showid="405078" data-livestatus="2" class="match-card status2  team-94  team-108" id="match61">
<h2>
半决赛 7月11日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5882db0d0cb64392b1a5"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时        
</li>
</ul>
</div>
    
</div> 
       <div class="vs-item vs-0">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000989&amp;matchId=63" data-showcode="1eee64c5fa354319aa14" data-showid="405081" data-livestatus="2" class="match-card status2  team-108  team-107" id="match63">
<h2>
季军赛 7月14日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_1eee64c5fa354319aa14"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000990&amp;matchId=64" data-showcode="0c3a385642cd44269f0e" data-showid="405082" data-livestatus="2" class="match-card status2  team-94  team-97" id="match64">
<h2>
决赛 7月15日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_0c3a385642cd44269f0e"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚        
</li>
</ul>
</div>
    
</div> 
       <div class="vs-item vs-1 vs-1-bottom">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000988&amp;matchId=62" data-showcode="70c0140e7ccf46228b34" data-showid="405079" data-livestatus="2" class="match-card status2  team-97  team-107" id="match62">
<h2>
半决赛 7月12日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_70c0140e7ccf46228b34"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
    
</div> 
       <div class="vs-item vs-2">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000986&amp;matchId=60" data-showcode="d138eed5202f470eb8ea" data-showid="405077" data-livestatus="2" class="match-card status2  team-83  team-97" id="match60">
<h2>
1/4决赛 7月8日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯
</li>
<li class="vs">
<div class="vs-score">
<span>5&nbsp;&nbsp;:&nbsp;&nbsp;6</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_d138eed5202f470eb8ea"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000985&amp;matchId=59" data-showcode="4e2c29305bac45928c6c" data-showid="405076" data-livestatus="2" class="match-card status2  team-104  team-107" id="match59">
<h2>
1/4决赛 7月7日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_4e2c29305bac45928c6c"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
    
</div> 
       <div class="vs-item vs-4">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000977&amp;matchId=51" data-showcode="c5bb120f712f497cbd35" data-showid="405068" data-livestatus="2" class="match-card status2  team-88  team-83" id="match51">
<h2>
1/8决赛 7月1日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-88">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F058A118830AA905C916">
西班牙
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;5</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_c5bb120f712f497cbd35"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000978&amp;matchId=52" data-showcode="b76fa8316fed4646a8cf" data-showid="405069" data-livestatus="2" class="match-card status2  team-97  team-91" id="match52">
<h2>
1/8决赛 7月2日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_b76fa8316fed4646a8cf"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-91">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDE2A118830A9C0CBE7E">
丹麦        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000981&amp;matchId=55" data-showcode="002c5a09e9eb4fe6a4b2" data-showid="405072" data-livestatus="2" class="match-card status2  team-104  team-99" id="match55">
<h2>
1/8决赛 7月3日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_002c5a09e9eb4fe6a4b2"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-99">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF73A35FB10B2300DB4F">
瑞士        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000982&amp;matchId=56" data-showcode="7343cf1f493249f2aeea" data-showid="405073" data-livestatus="2" class="match-card status2  team-112  team-107" id="match56">
<h2>
1/8决赛 7月4日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-112">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE58A118830AA2026B1C">
哥伦比亚
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;5</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_7343cf1f493249f2aeea"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
    
</div> 
      </div> 
     </div> 
     <div class="tab-panel tab-2" style=""> 
      <div class="schedule-date">
<dl>
  
<dt data-spm-anchor-id="a2h8q.11643819.0.i2">6月14日</dt>
<dd class="date-item" data-date="2018-06-14">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000927&amp;matchId=1" data-showcode="925ab3466bd24fc49055" data-showid="404846" data-livestatus="2" class="match-card status2  team-83  team-86" id="match1">
<h2>
世界杯A组 6月14日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯
</li>
<li class="vs">
<div class="vs-score">
<span>5&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_925ab3466bd24fc49055?spm=a2h8q.11643819.match1.1~3!2~A" data-spm-anchor-id="a2h8q.11643819.match1.1~3!2~A"> 
<span class="btn-span" data-spm-anchor-id="a2h8q.11643819.0.i0">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-86">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F005A35FB10B2806141C">
沙特阿拉伯        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月15日</dt>
<dd class="date-item" data-date="2018-06-15">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000928&amp;matchId=2" data-showcode="859f2a46d72a4a9abada" data-showid="404878" data-livestatus="2" class="match-card status2  team-85  team-84" id="match2">
<h2>
世界杯A组 6月15日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-85">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E7F5A118830AA30C6CAB">
埃及
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_859f2a46d72a4a9abada"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000929&amp;matchId=3" data-showcode="eec087398ceb441e8ade" data-showid="404888" data-livestatus="2" class="match-card status2  team-89  team-90" id="match3">
<h2>
世界杯B组 6月15日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-89">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEDFA35FB10B1A008A7C">
摩洛哥
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_eec087398ceb441e8ade"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-90">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F06DA35FB10B1A0B99AF">
伊朗        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月16日</dt>
<dd class="date-item" data-date="2018-06-16">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000930&amp;matchId=4" data-showcode="60bd661c2c274835a955" data-showid="404889" data-livestatus="2" class="match-card status2  team-87  team-88" id="match4">
<h2>
世界杯B组 6月16日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-87">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF28A118830A9A0070B1">
葡萄牙
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_60bd661c2c274835a955"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-88">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F058A118830AA905C916">
西班牙        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000991&amp;matchId=5" data-showcode="5d481c1703d24371905a" data-showid="404909" data-livestatus="2" class="match-card status2  team-94  team-93" id="match5">
<h2>
世界杯C组 6月16日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5d481c1703d24371905a"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-93">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECA3A35FB10B2C0D51ED">
澳大利亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000932&amp;matchId=6" data-showcode="209edadb526b4c0bb226" data-showid="404921" data-livestatus="2" class="match-card status2  team-96  team-95" id="match6">
<h2>
世界杯D组 6月16日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-96">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E966A35FB10B250A7DC7">
阿根廷
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_209edadb526b4c0bb226"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-95">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDA0A35FB10B1D004B78">
冰岛        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月17日</dt>
<dd class="date-item" data-date="2018-06-17">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000933&amp;matchId=7" data-showcode="32ab6e9e666845bfb4a8" data-showid="404910" data-livestatus="2" class="match-card status2  team-92  team-91" id="match7">
<h2>
世界杯C组 6月17日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-92">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEC2A118830AAB07FAE9">
秘鲁
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_32ab6e9e666845bfb4a8"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-91">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDE2A118830A9C0CBE7E">
丹麦        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000934&amp;matchId=8" data-showcode="1fbc52a94fbf44eaac13" data-showid="404922" data-livestatus="2" class="match-card status2  team-97  team-98" id="match8">
<h2>
世界杯D组 6月17日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_1fbc52a94fbf44eaac13"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-98">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF0CA35FB10B1D0B0051">
尼日利亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000935&amp;matchId=9" data-showcode="45312fbed14d485a8b3d" data-showid="404842" data-livestatus="2" class="match-card status2  team-102  team-100" id="match9">
<h2>
世界杯E组 6月17日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-102">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE73A35FB10B280C60F1">
哥斯达黎加
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_45312fbed14d485a8b3d"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-100">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFA8A118830AA00978A9">
塞尔维亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000936&amp;matchId=10" data-showcode="881934cf4e484b109255" data-showid="404876" data-livestatus="2" class="match-card status2  team-103  team-105" id="match10">
<h2>
世界杯F组 6月17日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-103">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE05A35FB10B220021FC">
德国
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_881934cf4e484b109255"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-105">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEF4A118830AB00CE1F0">
墨西哥        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月18日</dt>
<dd class="date-item" data-date="2018-06-18">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000937&amp;matchId=11" data-showcode="d704397c8ebf4790ad80" data-showid="404867" data-livestatus="2" class="match-card status2  team-101  team-99" id="match11">
<h2>
世界杯E组 6月18日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_d704397c8ebf4790ad80"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-99">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF73A35FB10B2300DB4F">
瑞士        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000938&amp;matchId=12" data-showcode="2888947210db4b8c957a" data-showid="404879" data-livestatus="2" class="match-card status2  team-104  team-106" id="match12">
<h2>
世界杯F组 6月18日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2888947210db4b8c957a"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-106">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE8AA118830AA70EE2A3">
韩国        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000939&amp;matchId=13" data-showcode="efdb1cbcb04f4c1885cd" data-showid="404904" data-livestatus="2" class="match-card status2  team-108  team-110" id="match13">
<h2>
世界杯G组 6月18日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_efdb1cbcb04f4c1885cd"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-110">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECD4A118830AAD0E8744">
巴拿马        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月19日</dt>
<dd class="date-item" data-date="2018-06-19">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000940&amp;matchId=14" data-showcode="e28a30bf555940c4a5d2" data-showid="404905" data-livestatus="2" class="match-card status2  team-109  team-107" id="match14">
<h2>
世界杯G组 6月19日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-109">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F01FA118830AA507BA25">
突尼斯
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_e28a30bf555940c4a5d2"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000941&amp;matchId=15" data-showcode="af4ef90bef5d48f19ae3" data-showid="404928" data-livestatus="2" class="match-card status2  team-112  team-113" id="match15">
<h2>
世界杯H组 6月19日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-112">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE58A118830AA2026B1C">
哥伦比亚
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_af4ef90bef5d48f19ae3"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-113">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF40A35FB10B20025E80">
日本        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000942&amp;matchId=16" data-showcode="2d04c1450c7f4651b5c0" data-showid="404930" data-livestatus="2" class="match-card status2  team-111  team-114" id="match16">
<h2>
世界杯H组 6月19日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-111">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDC9A35FB10B1F0CEB45">
波兰
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2d04c1450c7f4651b5c0"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-114">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFCDA35FB10B26017CF1">
塞内加尔        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月20日</dt>
<dd class="date-item" data-date="2018-06-20">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000943&amp;matchId=17" data-showcode="488c0a1a568349c08b08" data-showid="404880" data-livestatus="2" class="match-card status2  team-83  team-85" id="match17">
<h2>
世界杯A组 6月20日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_488c0a1a568349c08b08"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-85">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E7F5A118830AA30C6CAB">
埃及        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000944&amp;matchId=18" data-showcode="ed873aa1a72c41f2a934" data-showid="404891" data-livestatus="2" class="match-card status2  team-87  team-89" id="match18">
<h2>
世界杯B组 6月20日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-87">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF28A118830A9A0070B1">
葡萄牙
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_ed873aa1a72c41f2a934"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-89">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEDFA35FB10B1A008A7C">
摩洛哥        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000945&amp;matchId=19" data-showcode="5da3c5d595724115b8d0" data-showid="404881" data-livestatus="2" class="match-card status2  team-84  team-86" id="match19">
<h2>
世界杯A组 6月20日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5da3c5d595724115b8d0"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-86">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F005A35FB10B2806141C">
沙特阿拉伯        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月21日</dt>
<dd class="date-item" data-date="2018-06-21">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000946&amp;matchId=20" data-showcode="156c198bbcfe49c586dd" data-showid="404892" data-livestatus="2" class="match-card status2  team-90  team-88" id="match20">
<h2>
世界杯B组 6月21日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-90">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F06DA35FB10B1A0B99AF">
伊朗
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_156c198bbcfe49c586dd"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-88">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F058A118830AA905C916">
西班牙        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000947&amp;matchId=21" data-showcode="2d48dd021b9e4f0fb318" data-showid="404913" data-livestatus="2" class="match-card status2  team-91  team-93" id="match21">
<h2>
世界杯C组 6月21日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-91">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDE2A118830A9C0CBE7E">
丹麦
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2d48dd021b9e4f0fb318"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-93">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECA3A35FB10B2C0D51ED">
澳大利亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000948&amp;matchId=22" data-showcode="30f6a209a5f44428b290" data-showid="404916" data-livestatus="2" class="match-card status2  team-94  team-92" id="match22">
<h2>
世界杯C组 6月21日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_30f6a209a5f44428b290"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-92">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEC2A118830AAB07FAE9">
秘鲁        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月22日</dt>
<dd class="date-item" data-date="2018-06-22">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000949&amp;matchId=23" data-showcode="5b5be61f51554c329b6b" data-showid="404923" data-livestatus="2" class="match-card status2  team-96  team-97" id="match23">
<h2>
世界杯D组 6月22日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-96">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E966A35FB10B250A7DC7">
阿根廷
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5b5be61f51554c329b6b"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000950&amp;matchId=24" data-showcode="957da6beb9e74a69b06c" data-showid="404870" data-livestatus="2" class="match-card status2  team-101  team-102" id="match24">
<h2>
世界杯E组 6月22日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_957da6beb9e74a69b06c"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-102">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE73A35FB10B280C60F1">
哥斯达黎加        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000951&amp;matchId=25" data-showcode="c2eae348b37e4b16a38d" data-showid="404924" data-livestatus="2" class="match-card status2  team-98  team-95" id="match25">
<h2>
世界杯D组 6月22日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-98">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF0CA35FB10B1D0B0051">
尼日利亚
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_c2eae348b37e4b16a38d"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-95">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDA0A35FB10B1D004B78">
冰岛        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月23日</dt>
<dd class="date-item" data-date="2018-06-23">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000952&amp;matchId=26" data-showcode="39fe585ef93b4b7db715" data-showid="404872" data-livestatus="2" class="match-card status2  team-100  team-99" id="match26">
<h2>
世界杯E组 6月23日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-100">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFA8A118830AA00978A9">
塞尔维亚
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_39fe585ef93b4b7db715"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-99">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF73A35FB10B2300DB4F">
瑞士        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000953&amp;matchId=27" data-showcode="44d3b3b350ae4c7b967d" data-showid="404906" data-livestatus="2" class="match-card status2  team-108  team-109" id="match27">
<h2>
世界杯G组 6月23日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时
</li>
<li class="vs">
<div class="vs-score">
<span>5&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_44d3b3b350ae4c7b967d"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-109">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F01FA118830AA507BA25">
突尼斯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000954&amp;matchId=28" data-showcode="334e90e97e2740fdb811" data-showid="404883" data-livestatus="2" class="match-card status2  team-106  team-105" id="match28">
<h2>
世界杯F组 6月23日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-106">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE8AA118830AA70EE2A3">
韩国
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_334e90e97e2740fdb811"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-105">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEF4A118830AB00CE1F0">
墨西哥        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月24日</dt>
<dd class="date-item" data-date="2018-06-24">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000955&amp;matchId=29" data-showcode="9b1980d656c14aaca13f" data-showid="404887" data-livestatus="2" class="match-card status2  team-103  team-104" id="match29">
<h2>
世界杯F组 6月24日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-103">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE05A35FB10B220021FC">
德国
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_9b1980d656c14aaca13f"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000956&amp;matchId=30" data-showcode="20eebce657314c9d8554" data-showid="404908" data-livestatus="2" class="match-card status2  team-107  team-110" id="match30">
<h2>
世界杯G组 6月24日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰
</li>
<li class="vs">
<div class="vs-score">
<span>6&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_20eebce657314c9d8554"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-110">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECD4A118830AAD0E8744">
巴拿马        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000957&amp;matchId=31" data-showcode="9d6e86a899a94599a0a7" data-showid="404931" data-livestatus="2" class="match-card status2  team-113  team-114" id="match31">
<h2>
世界杯H组 6月24日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-113">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF40A35FB10B20025E80">
日本
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_9d6e86a899a94599a0a7"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-114">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFCDA35FB10B26017CF1">
塞内加尔        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月25日</dt>
<dd class="date-item" data-date="2018-06-25">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000958&amp;matchId=32" data-showcode="04b6763610fa4cbca283" data-showid="404932" data-livestatus="2" class="match-card status2  team-111  team-112" id="match32">
<h2>
世界杯H组 6月25日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-111">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDC9A35FB10B1F0CEB45">
波兰
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_04b6763610fa4cbca283"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-112">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE58A118830AA2026B1C">
哥伦比亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000959&amp;matchId=33" data-showcode="3c2e677db9594269bfd9" data-showid="404884" data-livestatus="2" class="match-card status2  team-84  team-83" id="match33">
<h2>
世界杯A组 6月25日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_3c2e677db9594269bfd9"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000960&amp;matchId=34" data-showcode="fa7d1951861b45089005" data-showid="404886" data-livestatus="2" class="match-card status2  team-86  team-85" id="match34">
<h2>
世界杯A组 6月25日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-86">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F005A35FB10B2806141C">
沙特阿拉伯
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_fa7d1951861b45089005"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-85">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E7F5A118830AA30C6CAB">
埃及        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月26日</dt>
<dd class="date-item" data-date="2018-06-26">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000961&amp;matchId=35" data-showcode="414ac9940d5d49a88ebd" data-showid="404896" data-livestatus="2" class="match-card status2  team-88  team-89" id="match35">
<h2>
世界杯B组 6月26日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-88">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F058A118830AA905C916">
西班牙
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_414ac9940d5d49a88ebd"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-89">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEDFA35FB10B1A008A7C">
摩洛哥        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000962&amp;matchId=36" data-showcode="8a7da9de819d467a9ccd" data-showid="404895" data-livestatus="2" class="match-card status2  team-90  team-87" id="match36">
<h2>
世界杯B组 6月26日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-90">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F06DA35FB10B1A0B99AF">
伊朗
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_8a7da9de819d467a9ccd"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-87">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF28A118830A9A0070B1">
葡萄牙        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000963&amp;matchId=37" data-showcode="2c17f27c83524912b9b2" data-showid="404920" data-livestatus="2" class="match-card status2  team-93  team-92" id="match37">
<h2>
世界杯C组 6月26日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-93">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECA3A35FB10B2C0D51ED">
澳大利亚
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2c17f27c83524912b9b2"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-92">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEC2A118830AAB07FAE9">
秘鲁        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000964&amp;matchId=38" data-showcode="995a13ca537e4d6a9726" data-showid="404918" data-livestatus="2" class="match-card status2  team-91  team-94" id="match38">
<h2>
世界杯C组 6月26日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-91">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDE2A118830A9C0CBE7E">
丹麦
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_995a13ca537e4d6a9726"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月27日</dt>
<dd class="date-item" data-date="2018-06-27">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000965&amp;matchId=39" data-showcode="2af644b2c4e44b65bf08" data-showid="404925" data-livestatus="2" class="match-card status2  team-98  team-96" id="match39">
<h2>
世界杯D组 6月27日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-98">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF0CA35FB10B1D0B0051">
尼日利亚
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2af644b2c4e44b65bf08"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-96">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E966A35FB10B250A7DC7">
阿根廷        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000966&amp;matchId=40" data-showcode="d038b5fb9d8d467788ef" data-showid="404927" data-livestatus="2" class="match-card status2  team-95  team-97" id="match40">
<h2>
世界杯D组 6月27日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-95">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDA0A35FB10B1D004B78">
冰岛
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_d038b5fb9d8d467788ef"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000967&amp;matchId=41" data-showcode="a2ea6655e85c48b99f37" data-showid="404897" data-livestatus="2" class="match-card status2  team-105  team-104" id="match41">
<h2>
世界杯F组 6月27日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-105">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEF4A118830AB00CE1F0">
墨西哥
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_a2ea6655e85c48b99f37"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000968&amp;matchId=42" data-showcode="8ab41da4295648629da0" data-showid="404893" data-livestatus="2" class="match-card status2  team-106  team-103" id="match42">
<h2>
世界杯F组 6月27日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-106">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE8AA118830AA70EE2A3">
韩国
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_8ab41da4295648629da0"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-103">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE05A35FB10B220021FC">
德国        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月28日</dt>
<dd class="date-item" data-date="2018-06-28">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000969&amp;matchId=43" data-showcode="2ba938cb384a434185b6" data-showid="404875" data-livestatus="2" class="match-card status2  team-99  team-102" id="match43">
<h2>
世界杯E组 6月28日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-99">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF73A35FB10B2300DB4F">
瑞士
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2ba938cb384a434185b6"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-102">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE73A35FB10B280C60F1">
哥斯达黎加        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000970&amp;matchId=44" data-showcode="596cd2d206a54136a8a4" data-showid="404873" data-livestatus="2" class="match-card status2  team-100  team-101" id="match44">
<h2>
世界杯E组 6月28日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-100">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFA8A118830AA00978A9">
塞尔维亚
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_596cd2d206a54136a8a4"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000971&amp;matchId=45" data-showcode="4c9babf9e1ef41208ed4" data-showid="404935" data-livestatus="2" class="match-card status2  team-113  team-111" id="match45">
<h2>
世界杯H组 6月28日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-113">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF40A35FB10B20025E80">
日本
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_4c9babf9e1ef41208ed4"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-111">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDC9A35FB10B1F0CEB45">
波兰        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000972&amp;matchId=46" data-showcode="8675d1b2e7954312b592" data-showid="404936" data-livestatus="2" class="match-card status2  team-114  team-112" id="match46">
<h2>
世界杯H组 6月28日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-114">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFCDA35FB10B26017CF1">
塞内加尔
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_8675d1b2e7954312b592"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-112">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE58A118830AA2026B1C">
哥伦比亚        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月29日</dt>
<dd class="date-item" data-date="2018-06-29">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000973&amp;matchId=47" data-showcode="f1c6f7f5b72e481d8966" data-showid="404914" data-livestatus="2" class="match-card status2  team-110  team-109" id="match47">
<h2>
世界杯G组 6月29日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-110">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECD4A118830AAD0E8744">
巴拿马
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_f1c6f7f5b72e481d8966"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-109">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F01FA118830AA507BA25">
突尼斯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000974&amp;matchId=48" data-showcode="f9b0f880b61944d5adf5" data-showid="404911" data-livestatus="2" class="match-card status2  team-107  team-108" id="match48">
<h2>
世界杯G组 6月29日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_f9b0f880b61944d5adf5"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时        
</li>
</ul>
</div>
    
</dd>
  
<dt>6月30日</dt>
<dd class="date-item" data-date="2018-06-30">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000975&amp;matchId=49" data-showcode="7feeedff60f046b0ba1f" data-showid="405066" data-livestatus="2" class="match-card status2  team-94  team-96" id="match49">
<h2>
1/8决赛 6月30日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_7feeedff60f046b0ba1f"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-96">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E966A35FB10B250A7DC7">
阿根廷        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月01日</dt>
<dd class="date-item" data-date="2018-07-01">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000976&amp;matchId=50" data-showcode="5be891eea5e5433aa6e4" data-showid="405067" data-livestatus="2" class="match-card status2  team-84  team-87" id="match50">
<h2>
1/8决赛 7月1日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5be891eea5e5433aa6e4"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-87">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF28A118830A9A0070B1">
葡萄牙        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000977&amp;matchId=51" data-showcode="c5bb120f712f497cbd35" data-showid="405068" data-livestatus="2" class="match-card status2  team-88  team-83" id="match51">
<h2>
1/8决赛 7月1日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-88">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F058A118830AA905C916">
西班牙
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;5</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_c5bb120f712f497cbd35"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月02日</dt>
<dd class="date-item" data-date="2018-07-02">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000978&amp;matchId=52" data-showcode="b76fa8316fed4646a8cf" data-showid="405069" data-livestatus="2" class="match-card status2  team-97  team-91" id="match52">
<h2>
1/8决赛 7月2日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_b76fa8316fed4646a8cf"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-91">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDE2A118830A9C0CBE7E">
丹麦        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000979&amp;matchId=53" data-showcode="ee40c336522f42e3b2c1" data-showid="405070" data-livestatus="2" class="match-card status2  team-101  team-105" id="match53">
<h2>
1/8决赛 7月2日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_ee40c336522f42e3b2c1"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-105">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEF4A118830AB00CE1F0">
墨西哥        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月03日</dt>
<dd class="date-item" data-date="2018-07-03">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000980&amp;matchId=54" data-showcode="38c7b4f64855400fa9f9" data-showid="405071" data-livestatus="2" class="match-card status2  team-108  team-113" id="match54">
<h2>
1/8决赛 7月3日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_38c7b4f64855400fa9f9"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-113">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF40A35FB10B20025E80">
日本        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000981&amp;matchId=55" data-showcode="002c5a09e9eb4fe6a4b2" data-showid="405072" data-livestatus="2" class="match-card status2  team-104  team-99" id="match55">
<h2>
1/8决赛 7月3日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_002c5a09e9eb4fe6a4b2"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-99">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF73A35FB10B2300DB4F">
瑞士        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月04日</dt>
<dd class="date-item" data-date="2018-07-04">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000982&amp;matchId=56" data-showcode="7343cf1f493249f2aeea" data-showid="405073" data-livestatus="2" class="match-card status2  team-112  team-107" id="match56">
<h2>
1/8决赛 7月4日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-112">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE58A118830AA2026B1C">
哥伦比亚
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;5</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_7343cf1f493249f2aeea"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月06日</dt>
<dd class="date-item" data-date="2018-07-06">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000983&amp;matchId=57" data-showcode="04d9f025cc914809a660" data-showid="405074" data-livestatus="2" class="match-card status2  team-84  team-94" id="match57">
<h2>
1/4决赛 7月6日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_04d9f025cc914809a660"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月07日</dt>
<dd class="date-item" data-date="2018-07-07">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000984&amp;matchId=58" data-showcode="818dd7284d7b4f1fb922" data-showid="405075" data-livestatus="2" class="match-card status2  team-101  team-108" id="match58">
<h2>
1/4决赛 7月7日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_818dd7284d7b4f1fb922"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000985&amp;matchId=59" data-showcode="4e2c29305bac45928c6c" data-showid="405076" data-livestatus="2" class="match-card status2  team-104  team-107" id="match59">
<h2>
1/4决赛 7月7日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_4e2c29305bac45928c6c"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月08日</dt>
<dd class="date-item" data-date="2018-07-08">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000986&amp;matchId=60" data-showcode="d138eed5202f470eb8ea" data-showid="405077" data-livestatus="2" class="match-card status2  team-83  team-97" id="match60">
<h2>
1/4决赛 7月8日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯
</li>
<li class="vs">
<div class="vs-score">
<span>5&nbsp;&nbsp;:&nbsp;&nbsp;6</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_d138eed5202f470eb8ea"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月11日</dt>
<dd class="date-item" data-date="2018-07-11">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000987&amp;matchId=61" data-showcode="5882db0d0cb64392b1a5" data-showid="405078" data-livestatus="2" class="match-card status2  team-94  team-108" id="match61">
<h2>
半决赛 7月11日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5882db0d0cb64392b1a5"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月12日</dt>
<dd class="date-item" data-date="2018-07-12">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000988&amp;matchId=62" data-showcode="70c0140e7ccf46228b34" data-showid="405079" data-livestatus="2" class="match-card status2  team-97  team-107" id="match62">
<h2>
半决赛 7月12日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_70c0140e7ccf46228b34?spm=a2h8q.11643819.match62.1~3!2~A" data-spm-anchor-id="a2h8q.11643819.match62.1~3!2~A"> 
<span class="btn-span" data-spm-anchor-id="a2h8q.11643819.0.i1">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月14日</dt>
<dd class="date-item" data-date="2018-07-14">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000989&amp;matchId=63" data-showcode="1eee64c5fa354319aa14" data-showid="405081" data-livestatus="2" class="match-card status2  team-108  team-107" id="match63">
<h2>
季军赛 7月14日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_1eee64c5fa354319aa14"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
    
</dd>
  
<dt>7月15日</dt>
<dd class="date-item" data-date="2018-07-15">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000990&amp;matchId=64" data-showcode="0c3a385642cd44269f0e" data-showid="405082" data-livestatus="2" class="match-card status2  team-94  team-97" id="match64">
<h2>
决赛 7月15日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>4&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_0c3a385642cd44269f0e?spm=a2h8q.11643819.match64.1~3!2~A" data-spm-anchor-id="a2h8q.11643819.match64.1~3!2~A"> 
<span class="btn-span" data-spm-anchor-id="a2h8q.11643819.0.i3">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚        
</li>
</ul>
</div>
    
</dd>
    
</dl></div> 
     </div> 
     <div class="tab-panel tab-3" style="display: none;"> 
      <div class="schedule-group">
<dl>
  
<div class="group-item-bg">
<span class="group-item-title">小组赛A组</span>
</div>
<dd class="group-item" data-group="A">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000927&amp;matchId=1" data-showcode="925ab3466bd24fc49055" data-showid="404846" data-livestatus="2" class="match-card status2  team-83  team-86" id="match1">
<h2>
世界杯A组 6月14日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯
</li>
<li class="vs">
<div class="vs-score">
<span>5&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_925ab3466bd24fc49055"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-86">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F005A35FB10B2806141C">
沙特阿拉伯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000928&amp;matchId=2" data-showcode="859f2a46d72a4a9abada" data-showid="404878" data-livestatus="2" class="match-card status2  team-85  team-84" id="match2">
<h2>
世界杯A组 6月15日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-85">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E7F5A118830AA30C6CAB">
埃及
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_859f2a46d72a4a9abada"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000943&amp;matchId=17" data-showcode="488c0a1a568349c08b08" data-showid="404880" data-livestatus="2" class="match-card status2  team-83  team-85" id="match17">
<h2>
世界杯A组 6月20日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_488c0a1a568349c08b08"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-85">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E7F5A118830AA30C6CAB">
埃及        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000945&amp;matchId=19" data-showcode="5da3c5d595724115b8d0" data-showid="404881" data-livestatus="2" class="match-card status2  team-84  team-86" id="match19">
<h2>
世界杯A组 6月20日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5da3c5d595724115b8d0"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-86">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F005A35FB10B2806141C">
沙特阿拉伯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000959&amp;matchId=33" data-showcode="3c2e677db9594269bfd9" data-showid="404884" data-livestatus="2" class="match-card status2  team-84  team-83" id="match33">
<h2>
世界杯A组 6月25日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-84">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F03CA35FB10B2C00650E">
乌拉圭
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_3c2e677db9594269bfd9"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-83">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE20A118830A9F0EAB69">
俄罗斯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000960&amp;matchId=34" data-showcode="fa7d1951861b45089005" data-showid="404886" data-livestatus="2" class="match-card status2  team-86  team-85" id="match34">
<h2>
世界杯A组 6月25日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-86">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F005A35FB10B2806141C">
沙特阿拉伯
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_fa7d1951861b45089005"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-85">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E7F5A118830AA30C6CAB">
埃及        
</li>
</ul>
</div>
    
</dd>									
  
<div class="group-item-bg">
<span class="group-item-title">小组赛B组</span>
</div>
<dd class="group-item" data-group="B">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000929&amp;matchId=3" data-showcode="eec087398ceb441e8ade" data-showid="404888" data-livestatus="2" class="match-card status2  team-89  team-90" id="match3">
<h2>
世界杯B组 6月15日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-89">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEDFA35FB10B1A008A7C">
摩洛哥
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_eec087398ceb441e8ade"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-90">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F06DA35FB10B1A0B99AF">
伊朗        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000930&amp;matchId=4" data-showcode="60bd661c2c274835a955" data-showid="404889" data-livestatus="2" class="match-card status2  team-87  team-88" id="match4">
<h2>
世界杯B组 6月16日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-87">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF28A118830A9A0070B1">
葡萄牙
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_60bd661c2c274835a955"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-88">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F058A118830AA905C916">
西班牙        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000944&amp;matchId=18" data-showcode="ed873aa1a72c41f2a934" data-showid="404891" data-livestatus="2" class="match-card status2  team-87  team-89" id="match18">
<h2>
世界杯B组 6月20日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-87">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF28A118830A9A0070B1">
葡萄牙
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_ed873aa1a72c41f2a934"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-89">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEDFA35FB10B1A008A7C">
摩洛哥        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000946&amp;matchId=20" data-showcode="156c198bbcfe49c586dd" data-showid="404892" data-livestatus="2" class="match-card status2  team-90  team-88" id="match20">
<h2>
世界杯B组 6月21日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-90">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F06DA35FB10B1A0B99AF">
伊朗
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_156c198bbcfe49c586dd"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-88">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F058A118830AA905C916">
西班牙        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000961&amp;matchId=35" data-showcode="414ac9940d5d49a88ebd" data-showid="404896" data-livestatus="2" class="match-card status2  team-88  team-89" id="match35">
<h2>
世界杯B组 6月26日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-88">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F058A118830AA905C916">
西班牙
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_414ac9940d5d49a88ebd"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-89">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEDFA35FB10B1A008A7C">
摩洛哥        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000962&amp;matchId=36" data-showcode="8a7da9de819d467a9ccd" data-showid="404895" data-livestatus="2" class="match-card status2  team-90  team-87" id="match36">
<h2>
世界杯B组 6月26日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-90">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F06DA35FB10B1A0B99AF">
伊朗
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_8a7da9de819d467a9ccd"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-87">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF28A118830A9A0070B1">
葡萄牙        
</li>
</ul>
</div>
    
</dd>									
  
<div class="group-item-bg">
<span class="group-item-title">小组赛C组</span>
</div>
<dd class="group-item" data-group="C">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000991&amp;matchId=5" data-showcode="5d481c1703d24371905a" data-showid="404909" data-livestatus="2" class="match-card status2  team-94  team-93" id="match5">
<h2>
世界杯C组 6月16日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5d481c1703d24371905a"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-93">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECA3A35FB10B2C0D51ED">
澳大利亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000933&amp;matchId=7" data-showcode="32ab6e9e666845bfb4a8" data-showid="404910" data-livestatus="2" class="match-card status2  team-92  team-91" id="match7">
<h2>
世界杯C组 6月17日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-92">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEC2A118830AAB07FAE9">
秘鲁
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_32ab6e9e666845bfb4a8"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-91">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDE2A118830A9C0CBE7E">
丹麦        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000947&amp;matchId=21" data-showcode="2d48dd021b9e4f0fb318" data-showid="404913" data-livestatus="2" class="match-card status2  team-91  team-93" id="match21">
<h2>
世界杯C组 6月21日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-91">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDE2A118830A9C0CBE7E">
丹麦
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2d48dd021b9e4f0fb318"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-93">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECA3A35FB10B2C0D51ED">
澳大利亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000948&amp;matchId=22" data-showcode="30f6a209a5f44428b290" data-showid="404916" data-livestatus="2" class="match-card status2  team-94  team-92" id="match22">
<h2>
世界杯C组 6月21日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_30f6a209a5f44428b290"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-92">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEC2A118830AAB07FAE9">
秘鲁        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000963&amp;matchId=37" data-showcode="2c17f27c83524912b9b2" data-showid="404920" data-livestatus="2" class="match-card status2  team-93  team-92" id="match37">
<h2>
世界杯C组 6月26日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-93">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECA3A35FB10B2C0D51ED">
澳大利亚
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2c17f27c83524912b9b2"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-92">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEC2A118830AAB07FAE9">
秘鲁        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000964&amp;matchId=38" data-showcode="995a13ca537e4d6a9726" data-showid="404918" data-livestatus="2" class="match-card status2  team-91  team-94" id="match38">
<h2>
世界杯C组 6月26日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-91">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDE2A118830A9C0CBE7E">
丹麦
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_995a13ca537e4d6a9726"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-94">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE3EA35FB10B2508134C">
法国        
</li>
</ul>
</div>
    
</dd>									
  
<div class="group-item-bg">
<span class="group-item-title">小组赛D组</span>
</div>
<dd class="group-item" data-group="D">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000932&amp;matchId=6" data-showcode="209edadb526b4c0bb226" data-showid="404921" data-livestatus="2" class="match-card status2  team-96  team-95" id="match6">
<h2>
世界杯D组 6月16日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-96">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E966A35FB10B250A7DC7">
阿根廷
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_209edadb526b4c0bb226"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-95">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDA0A35FB10B1D004B78">
冰岛        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000934&amp;matchId=8" data-showcode="1fbc52a94fbf44eaac13" data-showid="404922" data-livestatus="2" class="match-card status2  team-97  team-98" id="match8">
<h2>
世界杯D组 6月17日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_1fbc52a94fbf44eaac13"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-98">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF0CA35FB10B1D0B0051">
尼日利亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000949&amp;matchId=23" data-showcode="5b5be61f51554c329b6b" data-showid="404923" data-livestatus="2" class="match-card status2  team-96  team-97" id="match23">
<h2>
世界杯D组 6月22日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-96">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E966A35FB10B250A7DC7">
阿根廷
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_5b5be61f51554c329b6b"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000951&amp;matchId=25" data-showcode="c2eae348b37e4b16a38d" data-showid="404924" data-livestatus="2" class="match-card status2  team-98  team-95" id="match25">
<h2>
世界杯D组 6月22日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-98">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF0CA35FB10B1D0B0051">
尼日利亚
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_c2eae348b37e4b16a38d"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-95">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDA0A35FB10B1D004B78">
冰岛        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000965&amp;matchId=39" data-showcode="2af644b2c4e44b65bf08" data-showid="404925" data-livestatus="2" class="match-card status2  team-98  team-96" id="match39">
<h2>
世界杯D组 6月27日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-98">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF0CA35FB10B1D0B0051">
尼日利亚
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2af644b2c4e44b65bf08"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-96">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14E966A35FB10B250A7DC7">
阿根廷        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000966&amp;matchId=40" data-showcode="d038b5fb9d8d467788ef" data-showid="404927" data-livestatus="2" class="match-card status2  team-95  team-97" id="match40">
<h2>
世界杯D组 6月27日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-95">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDA0A35FB10B1D004B78">
冰岛
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_d038b5fb9d8d467788ef"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-97">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEA7A35FB10B2C0E859A">
克罗地亚        
</li>
</ul>
</div>
    
</dd>									
  
<div class="group-item-bg">
<span class="group-item-title">小组赛E组</span>
</div>
<dd class="group-item" data-group="E">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000935&amp;matchId=9" data-showcode="45312fbed14d485a8b3d" data-showid="404842" data-livestatus="2" class="match-card status2  team-102  team-100" id="match9">
<h2>
世界杯E组 6月17日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-102">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE73A35FB10B280C60F1">
哥斯达黎加
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_45312fbed14d485a8b3d"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-100">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFA8A118830AA00978A9">
塞尔维亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000937&amp;matchId=11" data-showcode="d704397c8ebf4790ad80" data-showid="404867" data-livestatus="2" class="match-card status2  team-101  team-99" id="match11">
<h2>
世界杯E组 6月18日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_d704397c8ebf4790ad80"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-99">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF73A35FB10B2300DB4F">
瑞士        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000950&amp;matchId=24" data-showcode="957da6beb9e74a69b06c" data-showid="404870" data-livestatus="2" class="match-card status2  team-101  team-102" id="match24">
<h2>
世界杯E组 6月22日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_957da6beb9e74a69b06c"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-102">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE73A35FB10B280C60F1">
哥斯达黎加        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000952&amp;matchId=26" data-showcode="39fe585ef93b4b7db715" data-showid="404872" data-livestatus="2" class="match-card status2  team-100  team-99" id="match26">
<h2>
世界杯E组 6月23日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-100">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFA8A118830AA00978A9">
塞尔维亚
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_39fe585ef93b4b7db715"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-99">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF73A35FB10B2300DB4F">
瑞士        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000969&amp;matchId=43" data-showcode="2ba938cb384a434185b6" data-showid="404875" data-livestatus="2" class="match-card status2  team-99  team-102" id="match43">
<h2>
世界杯E组 6月28日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-99">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF73A35FB10B2300DB4F">
瑞士
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2ba938cb384a434185b6"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-102">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE73A35FB10B280C60F1">
哥斯达黎加        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000970&amp;matchId=44" data-showcode="596cd2d206a54136a8a4" data-showid="404873" data-livestatus="2" class="match-card status2  team-100  team-101" id="match44">
<h2>
世界杯E组 6月28日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-100">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFA8A118830AA00978A9">
塞尔维亚
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_596cd2d206a54136a8a4"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-101">
<img class="team-logo" src="https://r1.ykimg.com/051000005B152CA2A35FB10B27057946">
巴西        
</li>
</ul>
</div>
    
</dd>									
  
<div class="group-item-bg">
<span class="group-item-title">小组赛F组</span>
</div>
<dd class="group-item" data-group="F">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000936&amp;matchId=10" data-showcode="881934cf4e484b109255" data-showid="404876" data-livestatus="2" class="match-card status2  team-103  team-105" id="match10">
<h2>
世界杯F组 6月17日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-103">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE05A35FB10B220021FC">
德国
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_881934cf4e484b109255"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-105">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEF4A118830AB00CE1F0">
墨西哥        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000938&amp;matchId=12" data-showcode="2888947210db4b8c957a" data-showid="404879" data-livestatus="2" class="match-card status2  team-104  team-106" id="match12">
<h2>
世界杯F组 6月18日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2888947210db4b8c957a"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-106">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE8AA118830AA70EE2A3">
韩国        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000954&amp;matchId=28" data-showcode="334e90e97e2740fdb811" data-showid="404883" data-livestatus="2" class="match-card status2  team-106  team-105" id="match28">
<h2>
世界杯F组 6月23日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-106">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE8AA118830AA70EE2A3">
韩国
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_334e90e97e2740fdb811"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-105">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEF4A118830AB00CE1F0">
墨西哥        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000955&amp;matchId=29" data-showcode="9b1980d656c14aaca13f" data-showid="404887" data-livestatus="2" class="match-card status2  team-103  team-104" id="match29">
<h2>
世界杯F组 6月24日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-103">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE05A35FB10B220021FC">
德国
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_9b1980d656c14aaca13f"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000967&amp;matchId=41" data-showcode="a2ea6655e85c48b99f37" data-showid="404897" data-livestatus="2" class="match-card status2  team-105  team-104" id="match41">
<h2>
世界杯F组 6月27日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-105">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EEF4A118830AB00CE1F0">
墨西哥
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_a2ea6655e85c48b99f37"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-104">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF5BA118830A9D08FB36">
瑞典        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000968&amp;matchId=42" data-showcode="8ab41da4295648629da0" data-showid="404893" data-livestatus="2" class="match-card status2  team-106  team-103" id="match42">
<h2>
世界杯F组 6月27日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-106">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE8AA118830AA70EE2A3">
韩国
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_8ab41da4295648629da0"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-103">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE05A35FB10B220021FC">
德国        
</li>
</ul>
</div>
    
</dd>									
  
<div class="group-item-bg">
<span class="group-item-title">小组赛G组</span>
</div>
<dd class="group-item" data-group="G">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000939&amp;matchId=13" data-showcode="efdb1cbcb04f4c1885cd" data-showid="404904" data-livestatus="2" class="match-card status2  team-108  team-110" id="match13">
<h2>
世界杯G组 6月18日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时
</li>
<li class="vs">
<div class="vs-score">
<span>3&nbsp;&nbsp;:&nbsp;&nbsp;0</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_efdb1cbcb04f4c1885cd"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-110">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECD4A118830AAD0E8744">
巴拿马        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000940&amp;matchId=14" data-showcode="e28a30bf555940c4a5d2" data-showid="404905" data-livestatus="2" class="match-card status2  team-109  team-107" id="match14">
<h2>
世界杯G组 6月19日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-109">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F01FA118830AA507BA25">
突尼斯
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_e28a30bf555940c4a5d2"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000953&amp;matchId=27" data-showcode="44d3b3b350ae4c7b967d" data-showid="404906" data-livestatus="2" class="match-card status2  team-108  team-109" id="match27">
<h2>
世界杯G组 6月23日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时
</li>
<li class="vs">
<div class="vs-score">
<span>5&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_44d3b3b350ae4c7b967d"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-109">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F01FA118830AA507BA25">
突尼斯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000956&amp;matchId=30" data-showcode="20eebce657314c9d8554" data-showid="404908" data-livestatus="2" class="match-card status2  team-107  team-110" id="match30">
<h2>
世界杯G组 6月24日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰
</li>
<li class="vs">
<div class="vs-score">
<span>6&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_20eebce657314c9d8554"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-110">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECD4A118830AAD0E8744">
巴拿马        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000973&amp;matchId=47" data-showcode="f1c6f7f5b72e481d8966" data-showid="404914" data-livestatus="2" class="match-card status2  team-110  team-109" id="match47">
<h2>
世界杯G组 6月29日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-110">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ECD4A118830AAD0E8744">
巴拿马
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_f1c6f7f5b72e481d8966"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-109">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F01FA118830AA507BA25">
突尼斯        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000974&amp;matchId=48" data-showcode="f9b0f880b61944d5adf5" data-showid="404911" data-livestatus="2" class="match-card status2  team-107  team-108" id="match48">
<h2>
世界杯G组 6月29日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-107">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14F084A118830AAF053DE2">
英格兰
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_f9b0f880b61944d5adf5"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-108">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14ED80A118830A9802AC22">
比利时        
</li>
</ul>
</div>
    
</dd>									
  
<div class="group-item-bg">
<span class="group-item-title">小组赛H组</span>
</div>
<dd class="group-item" data-group="H">
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000941&amp;matchId=15" data-showcode="af4ef90bef5d48f19ae3" data-showid="404928" data-livestatus="2" class="match-card status2  team-112  team-113" id="match15">
<h2>
世界杯H组 6月19日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-112">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE58A118830AA2026B1C">
哥伦比亚
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_af4ef90bef5d48f19ae3"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-113">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF40A35FB10B20025E80">
日本        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000942&amp;matchId=16" data-showcode="2d04c1450c7f4651b5c0" data-showid="404930" data-livestatus="2" class="match-card status2  team-111  team-114" id="match16">
<h2>
世界杯H组 6月19日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-111">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDC9A35FB10B1F0CEB45">
波兰
</li>
<li class="vs">
<div class="vs-score">
<span>1&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_2d04c1450c7f4651b5c0"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-114">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFCDA35FB10B26017CF1">
塞内加尔        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000957&amp;matchId=31" data-showcode="9d6e86a899a94599a0a7" data-showid="404931" data-livestatus="2" class="match-card status2  team-113  team-114" id="match31">
<h2>
世界杯H组 6月24日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-113">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF40A35FB10B20025E80">
日本
</li>
<li class="vs">
<div class="vs-score">
<span>2&nbsp;&nbsp;:&nbsp;&nbsp;2</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_9d6e86a899a94599a0a7"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-114">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFCDA35FB10B26017CF1">
塞内加尔        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000958&amp;matchId=32" data-showcode="04b6763610fa4cbca283" data-showid="404932" data-livestatus="2" class="match-card status2  team-111  team-112" id="match32">
<h2>
世界杯H组 6月25日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-111">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDC9A35FB10B1F0CEB45">
波兰
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;3</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_04b6763610fa4cbca283"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-112">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE58A118830AA2026B1C">
哥伦比亚        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000971&amp;matchId=45" data-showcode="4c9babf9e1ef41208ed4" data-showid="404935" data-livestatus="2" class="match-card status2  team-113  team-111" id="match45">
<h2>
世界杯H组 6月28日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-113">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EF40A35FB10B20025E80">
日本
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_4c9babf9e1ef41208ed4"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-111">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EDC9A35FB10B1F0CEB45">
波兰        
</li>
</ul>
</div>
  
<div data-liveurl="http://vku.youku.com/live/ilproom?id=8000972&amp;matchId=46" data-showcode="8675d1b2e7954312b592" data-showid="404936" data-livestatus="2" class="match-card status2  team-114  team-112" id="match46">
<h2>
世界杯H组 6月28日已结束
</h2>
<ul class="match-team">
<li class="team-item home" data-team="team-114">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EFCDA35FB10B26017CF1">
塞内加尔
</li>
<li class="vs">
<div class="vs-score">
<span>0&nbsp;&nbsp;:&nbsp;&nbsp;1</span>
</div>


<a target="_blank" class="cup-live-btn reback" href="//v.youku.com/v_nextstage/id_8675d1b2e7954312b592"> 
<span class="btn-span">回看</span>
</a>


</li>
<li class="team-item  guest" data-team="team-112">
<img class="team-logo" src="https://r1.ykimg.com/051000005B14EE58A118830AA2026B1C">
哥伦比亚        
</li>
</ul>
</div>
    
</dd>									
  
</dl>  </div> 
     </div> 
    </div> 
   </div> 
  </div> 
  <script type="text/javascript" src="//d3js.org/d3.v3.js"></script> 
  <script src="//js.ykimg.com/youku/dist/js/lib_15.js" id="libjsnode" charset="utf-8"></script> 
  <script type="text/javascript" src="//js.ykimg.com/youku/dist/js/g_84.js"></script> 
  <script type="text/javascript" src="//js.ykimg.com/youku/dist/js/page/find/g_133.js"></script> 
  <script type="text/javascript" src="//js.ykimg.com/youku/dist/js/page/find/external/init_3.js"></script> 
  <script language="javascript" type="text/javascript" src="//js.ykimg.com/youku/dist/js/page/find/worldcup/scheduleView_30.js"></script>  
 
<div id="sideBar" class="side-bar" style="display: block;"><div class="feedback"><a href="//csc.youku.com/feedback-web/web/" target="_blank"></a></div><div id="goTop" data-stat-role="ck" style="display: none;"><img width="29" height="65" src="//r1.ykimg.com/05100000575CCAF767BC3D4B250267B7"></div></div><script></script></body></html>'''
    urls = re.findall('<a target="_blank" class="cup-live-btn reback" href="(.*?)"', page)
    urls = list(map(lambda a: 'http:' + a, set(urls)))

    return urls


if __name__ == '__main__':
    match_urls = get_match_urls()
    with open('urlss.txt', 'w') as f:
        for url_ in match_urls:
            f.write(url_ + "\n")
    P = Pool()
    os.makedirs(os.path.join(os.getcwd(), 'video'))
    os.chdir(os.path.join(os.getcwd(), 'video'))
    for match_url_ in match_urls:
        video_urls, name = get_video_url(match_url_)
        path = os.path.join(os.getcwd(), 'video', name)
        os.makedirs(path)
        for video_url_ in video_urls:
            download_videos(video_url_)
            # P.apply_async(download_videos, args=(video_url_,))
    P.close()
    P.join()
