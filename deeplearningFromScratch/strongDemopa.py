
import requests
import re

#封装的加强版get_urls
def get_urls(destination,feature):
    '''  
    1.找到要爬的目标网页
    2.用正则来匹配不同图片的地址，产生一个匹配之后的结果
    3.通过源代码和匹配之后的结果来找到获取图片的地址
    '''
    response = requests.get(destination)
    '''
    .匹配不换行的字符，
    #任意数量,
    ?:非贪婪匹配，匹配尽可能短的字符
    ():不仅要匹配字符，字符还要使用
    '''
    if(feature == 'jpg'):
        url_add = r'<img.*?src="(.*?.jpg)" .*?/>'
    if(feature == 'gif'):
         url_add = r'<gif.*?src="(.*?.jpg)" .*?/>'
    if(feature == 'avi'):
         url_add =r'<avi.*?src="(.*?.jpg)" .*?/>'
    #在网页源代码中查找图片的地址，第二个参数要求网页源代码
    url_list = re.findall(url_add,response.text)
    print(url_list)
    return url_list



#下载数据
def get_jpg(url,name,roadline):
    response = requests.get(url)
    #下载路径
    with open(roadline+'p%d.jpg' %name,'wb') as ft:
        ft.write(response.content)

def  pa(web,feature,roadline = 'D:\\others\\resultset\\'):
    url_list = get_urls(web,feature)
    name = 1
    for url in url_list:
        com_url = 'https:' + url
        get_jpg(com_url,name,roadline)
        name += 1
        print(com_url)