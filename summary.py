import json
import os
#设置环境变量
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "description"
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup

MODEL = "qwen2:72b"
OLLAMA_URL = "http://10.214.149.209:14005"
# HTML_PATH = "html"


# class GraphState(TypedDict):
#     qusetion: str
#     answer: str 
#     description: json # 包含所有子页面的描述信息

llm = ChatOllama(model=MODEL,base_url=OLLAMA_URL,temperature=0)
sync_json = json.load(open('csv/description/sync.json','r',encoding='utf-8'))

prompt =  PromptTemplate(
    template="""You are an AI assistant that helps users to summarise the functionality of a website based on the descriptions of all the subpages.\n
    This is the website's name: {web_name}.\n
    This is the subpages' description: \n{subpages}.\n
    Please summarise the description for this website within 200 words in Chinese. Focus on what the website is mainly used for.\n
    If this description contains private information, please anonymise it. You don't need to tell me if anonymisation has taken place.\n
    Just tell me the summary, do not tell me anything else.\n""",
    input_variables=["web_name","subpages"],
)

chain = prompt | llm | StrOutputParser()

website_summary = {}

if __name__ == "__main__":
    for app_name in sync_json:
        subpages_description = sync_json[app_name]
        # print(app_name,subpages_description)
        summary = chain.invoke({
            "web_name":app_name,
            "subpages":subpages_description
        })
        print(f"{app_name}:\n{summary}")
        website_summary[app_name] = summary
    json.dump(
        website_summary,
        open('csv/description/summary.json','w',encoding='utf-8'),
        ensure_ascii=False,
        indent=4
    )
