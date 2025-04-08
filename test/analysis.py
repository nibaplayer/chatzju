from bs4 import BeautifulSoup
import os 

from langchain_experimental.llms.ollama_functions import OllamaFunctions
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
from typing import Literal
from langchain_core.messages import BaseMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pprint import pprint
from langchain_community.chat_models import ChatOllama

MODEL = "qwen2:7b-instruct-fp16"

soup = BeautifulSoup(open("/root/code/chatzju/html/card.html"), "html.parser")

for script in soup(["script", "style", "head"]):
    script.decompose()
html_content = soup.prettify()
print(soup.prettify())

#TODO 先判断是否为充值页面，若否再分析如何进入下一级页面
#TODO COT 

def analyze_html(html_content: str, question: str):
    '''
    分析html内容，返回与问题相关的元素
    '''
    llm = ChatOllama(model=MODEL, temperature=0)
    analyze_prompt = PromptTemplate(
        template="""You are an expert at programing selenium python program to manipulate the web based on the user's question and the html content. \n
        You need to return the python code to fullfill the task asked by user. \n
        HTML content: {html_content}. \n
        Question: {question}""",
        input_variables=["question", "html_content"],
    )
    chain = analyze_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "html_content": html_content})
    return result

if __name__ == "__main__":
    result = analyze_html(html_content, "我想充100块网费")
    print(result)