import json
import os
#设置环境变量
# os.environ["LANGSMITH_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "description"
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import pandas as pd

MODEL = "qwen2:72b"
OLLAMA_URL = "http://10.214.149.209:14005"
HTML_PATH = "html"


# class GraphState(TypedDict):
#     qusetion: str
#     answer: str 
#     description: json # 包含所有子页面的描述信息

llm = ChatOllama(model=MODEL,base_url=OLLAMA_URL,temperature=0)
sync_json = json.load(open('sync.json','r',encoding='utf-8'))

prompt =  PromptTemplate(
    template="""You are an AI assistant that helps users to generate descriptions for web pages according to the html content, web title and web name.\n
    This is the web html: \n{web_html}.\n
    This is the web title: {web_title}.\n
    This is the web name: {web_name}.\n
    Please generate a description for this web page within 200 words in Chinese. Focus on what this page is mainly used for, ignoring the page's layout information and technical details.\n
    If this description contains private information, please anonymise it. You don't need to tell me if anonymisation has taken place.\n
    Just tell me the description, do not tell me anything else.\n""",
    input_variables=["web_name","web_title","web_html"],
)

chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    df = pd.read_csv("csv/first_level.csv", encoding="utf-8")
    app_name = df.get('应用名称')
    link = df.get('link')
    description = df.get('描述')
    for i in range(len(app_name)):
        c_df = pd.read_csv(f"csv/{app_name[i]}/sync.csv", encoding="utf-8")
        c_link = c_df.get('link')
        html_file = c_df.get('html_file')
        title = c_df.get('title')
        c_description = c_df.get('描述')
        for j in range(len(c_link)):
            web_name = app_name[i]
            web_title = title[j]
            with open (f"csv/{app_name[i]}/{html_file[j]}",'r',encoding='utf-8') as f:
                web_html = f.read()
            soup = BeautifulSoup(web_html, 'html.parser')
            web_html = soup.decode()   # 在这里去除缩进
            now_description = chain.invoke({
                "web_name":web_name,
                "web_title":web_title,
                "web_html":web_html
            })
            c_description[j] = now_description
            print(f"{html_file[j]}:\n{now_description}")
        c_df['描述'] = c_description
        c_df.to_csv(f"csv/{app_name[i]}/sync.csv", encoding="utf-8", index=False)
        print(f"{app_name[i]} has been updated.")
    # for app_name in sync_json:
    #     for i,web in enumerate(sync_json[app_name]):
    #         web_name = app_name
    #         web_title = web["title"]
    #         with open (f"{HTML_PATH}/{web['html_file']}",'r',encoding='utf-8') as f:
    #             web_html = f.read()
    #         soup = BeautifulSoup(web_html, 'html.parser')
    #         web_html = soup.decode()   # 在这里去除缩进
    #         # print(i,web_name,web_title,web_html)
    #         # with open('temp.html','w',encoding='utf-8') as f:
    #         #     f.write(web_html)
    #         description = chain.invoke({
    #             "web_name":web_name,
    #             "web_title":web_title,
    #             "web_html":web_html
    #             })
    #         sync_json[app_name][i]["description"] = description
    #         print(f"{web['html_file']}:\n{description}")
    # json.dump(
    #     sync_json,
    #     open(os.path.join('html','description','sync.json'),'w',encoding='utf-8'),
    #     ensure_ascii=False,
    #     indent=2)