# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from time import sleep

# driver = webdriver.Edge()
# driver.get("https://www.baidu.com")

# driver.find_element(By.ID, "kw").send_keys("selenium")
# sleep(2)

# driver.find_element(By.ID, "su").click()
# sleep(5)

# content  = driver.find_element(By.ID, "content_left").text
# print(content)

# driver.quit()

# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from time import sleep
# driver = webdriver.Edge()

# driver.get("https://yqfkgl.zju.edu.cn/_web/_customizes/ykt/index2.jsp")


# input("Press Enter to continue...")

# page_source = driver.page_source

# print(page_source)

# driver.quit()

import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup, Comment

driver = webdriver.Edge()
LOGIN_KEY = ["认证","登录"]
record_hrefs = [] # 记录已经爬取的链接 由于某些链接会自行跳转，在其父页面中记录的链接不一定是最终的链接

def crawl(url)->BeautifulSoup:
    try:
        driver.get(url)
    except Exception as e:
        print("Error: ",e)
        return None
    for keyword in LOGIN_KEY:
        if keyword in driver.title:
            input("Press Enter to continue...")# 人工认证 之后再继续
            break

    page_source = driver.page_source
    return BeautifulSoup(page_source, 'html.parser')


def clean_html(soup: BeautifulSoup)->BeautifulSoup:
    # 移除注释
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # 移除 <script> 和 <style>
    for tag in soup(["script", "style","path"]):
        tag.extract()

    # 移除包含特定 class 或 id 的标签（根据需要定制）
    for unwanted_tag in soup.find_all(['div', 'span'], class_=['ad', 'banner']):
        unwanted_tag.extract()

    for unwanted_tag in soup.find_all(['img']):
        if unwanted_tag.has_attr('src') :
            del unwanted_tag['src']

    return soup

def recursive_crawl(url:str, depth:int=10):
    """
    递归爬取网页链接
    url是从父页面中提取的链接，实际过程中可能会发生跳转，重复url的判断以跳转后的url为准
    depth用于限制递归深度，depth每次递归后自减
    """
    if depth <= 0:
        return
    if url in record_hrefs: # 防止重复爬取
        return
    page_source = crawl(url) # 爬取页面 这里已经进入了新地址
    
    print("processing",url)
    
    if driver.current_url in record_hrefs: # 防止重复爬取
        return
    page_source = clean_html(page_source)
    record_hrefs.append(driver.current_url) # 记录已经爬取的链接

    #保存html 文件名与网址需要使用json同步


    hrefs = [] #记录所有次级地址
    elements = driver.find_elements("xpath","//*[@href]") #//*[@href]是xpath表达式，用于寻找所有含有href属性的元素
    for element in elements:
        href = element.get_attribute("href")
        if href.endswith(".css") or href.endswith(".js") or 'javascript:' in href or href in hrefs: # 过滤掉不需要的链接
            continue
        hrefs.append(href)
    
    # 开始递归
    for href in hrefs:
        recursive_crawl(href,depth-1)



if __name__ == '__main__':
    url = "https://myvpn.zju.edu.cn/"
    # page_source = crawl(url)
    # page_source = clean_html(page_source)
    # print(page_source.prettify())

    # # 查找所有包含href属性的元素
    # elements = driver.find_elements("xpath", "//*[@href]")
    # # 提取所有href属性的值
    # # hrefs = [element.get_attribute("href") for element in elements]
    # # hrefs = []
    # for element in elements:
    #     href = element.get_attribute("href")
    #     if href.endswith(".css") or href.endswith(".js") or 'javascript:' in href or href in hrefs: # 过滤掉不需要的链接
    #         continue
    #     hrefs.append(href)
    # print(hrefs)

    recursive_crawl(url=url,depth=20)

    print(record_hrefs)

    input("Press Enter to continue...")
