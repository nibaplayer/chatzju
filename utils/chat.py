import os
#设置环境变量
# os.environ["LANGSMITH_API_KEY"] = "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"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "main"
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
EMBEDDING_MODEL = "nomic-embed-text"
KNOWLEDGE_DIR = "knowledge"

def action_chain(question: str):
    # TODO 解决错误率的问题
    # LLM
    llm = ChatOllama(model=MODEL, format="json", temperature=0)
    action_prompt = PromptTemplate(
        template="""
        You are an expert at selecting what to do next based on the user's question. \n
        You can choose to open a webpage or answer the user's question. \n
        Just return a JSON with the key 'action' and the value 'open' or 'answer'. \n
        Question: {question}
        """,
        input_variables=["question"],
    )
    chain = action_prompt | llm | JsonOutputParser()
    result = chain.invoke({"question": question})
    # print(result)
    if "action" in result:
        if result["action"] == "open":
            return result["action"]
        elif result["action"] == "answer":
            return result["action"]

def navigate_chain(question: str):
    # LLM
    llm = ChatOllama(model=MODEL, format="json", temperature=0)
    select_prompt = PromptTemplate(
        template="""You are an expert at selecting related content for a user's question based on the document list {list}. \n
        Do not be stringent with the keywords in the question; focus on related topics. \n
        Provide up to three options related to the user's question from the document list {list}.\n
        Return a JSON with the key 'url' and no preamble or explanation. \n
        If there are no related documents, return a JSON with the key 'option' and the value 'none'. \n
        Question: {question}""",
        input_variables=["question"],
    )
    d_list = []
    d_dict = {}
    with open('apps.json','r') as json_file:
        loaded_data = json.load(json_file)
        for value in loaded_data.values():
            d_list.append(value['app'])
            d_dict[value['app']] = value['link']
    chain = select_prompt | llm | JsonOutputParser()
    result = chain.invoke({"question": question, "list": d_list})
    try:
        url = d_dict.get(result['url'])
        pass
    except:
        raise("The url is not found")
    return url




if __name__ == '__main__':
    question = "打开电子校园卡"
    action = action_chain(question)
    # print(action)
    result = {'action':action}
    if action == 'open':
        content = navigate_chain(question)
        result['content'] = content #TODO 这里answer修改为content前端需要修改
        print(result)