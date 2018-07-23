from lxml import etree
import sys
html = ''''' 
<div>
    <ul>
         <li class="item-0"><a href="link1.html">first item</a></li>
         <li class="item-1"><a href="link2.html">second item</a></li>
         <li class="item-inactive"><a href="link3.html">third item</a></li>
         <li class="item-1"><a href="link4.html">fourth item</a></li>
         <li class="item-0"><a href="link5.html">fifth item</a>
     </ul>
 </div>
'''

page = etree.HTML(html)  # 先确保html经过了utf-8解码，即code = html.decode('utf-8', 'ignore')，否则会出现解析出错情况。因为中文被编码成utf-8之后变成 '/u2541'　之类的形式，lxml一遇到　“/”就会认为其标签结束。
hrefs = page.xpath("//a")  # 它会找到整个html代码里的所有a标签，如果想精确地获取，可以采用下面获取p标签的方法
print()