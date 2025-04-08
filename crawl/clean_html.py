from bs4 import BeautifulSoup, Comment
import os

def clean_html(soup: BeautifulSoup):
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



if __name__ == '__main__':
    filelist = os.listdir('html')
    for file in filelist:
        if file.endswith('.html'):
            with open(os.path.join('html',file),'r',encoding='utf-8') as input_file:
                html_content = input_file.read()

            soup = BeautifulSoup(html_content,'html.parser')

            cleaned_soup = clean_html(soup)

            with open(os.path.join('html',file), 'w', encoding='utf-8') as output_file:
                output_file.write(cleaned_soup.prettify())