import os
#设置环境变量
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chat"

from flask import Flask, request, Response, jsonify
from flask_cors import CORS

from utils.chat import navigate_chain, action_chain

from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

from pprint import pprint
import json

#TODO 现在可以先不接入，先在main.py中测试，
#TODO 需要考虑在前端的调用形式 目前预估只能连贯执行完成，不能中断 （考虑使用扩展插件的形式）
MODEL = "qwen2:7b-instruct-fp16"
EMBEDDING_MODEL = "nomic-embed-text"
KNOWLEDGE_DIR = "knowledge"
OLLAMA_URL = "http://10.214.149.209:14005"

SUBPAGES_DESCRIPTION = json.load(
    open(os.path.join('csv','description','sync.json'),'r',encoding='utf-8')
)
WEBSITE_SUMMARY = json.load(
    open(os.path.join('csv','description','summary.json'),'r',encoding='utf-8')
)

df_web = pd.read_csv("csv/first_level.csv", encoding="utf-8")
df_web_name = df_web["应用名称"]
df_web_url = df_web["link"]
website_dict = dict(zip(df_web_name,df_web_url))

#从pdf加载knowledge
filelist = os.listdir(KNOWLEDGE_DIR)
merge_pages = []
for file in filelist:
    if file.endswith('.pdf'):
        file_path = os.path.join(KNOWLEDGE_DIR, file)
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        merge_pages += pages

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=400, add_start_index=True
)
all_splitter = text_splitter.split_documents(merge_pages)
vectorstore = Chroma.from_documents(all_splitter,embedding=OllamaEmbeddings(model=EMBEDDING_MODEL,base_url=OLLAMA_URL))

def selece_best_three_website(grade:dict,threshold=5):
    # 选择最好的三个website 要设置一个阈值
    best_three_website = sorted(grade.items(),key=lambda x:x[1],reverse=True)[:3]
    best_three_website = [x[0] for x in best_three_website if x[1] >= threshold]
    return best_three_website

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[Document]
    action: str

#NODE
def open_navigation(state: GraphState):
    """
    Open the webpage based on the question.
    Args:
        state (dict): The current graph state
    Returns:
        state with url in generation
    """
    # print("---OPEN NAVIGATION---")
    question = state["question"]
    # LLM
    llm = ChatOllama(model=MODEL, format="json", temperature=0, base_url=OLLAMA_URL)
    grading_prompt =  PromptTemplate(
        template="""You are a grader assessing relevant of user's instructions to the website.\n
            Here is the user's instruction: {instruction}.\n
            Here is the website's name: {web_name}.\n
            Here is the website's summary: \n{summary}.\n
            Please remember that the user is staff or student of Zhejiang University.
            Please grade the relevance of the instruction to the website on a scale of 0 to 10.\n
            If the instruction is irrelevant to the website, please give a score of 0.\n
            If the instruction is highly relevant to the website, please give a score of 10.\n
            If the instruction is somewhat relevant to the website, please give a score between 0 and 10.\n
            Provide the score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["instruction","web_name","summary"],
    )
    grading_chain = grading_prompt | llm | JsonOutputParser()
    second_grading_prompt = PromptTemplate(
        template="""You are a grader assessing the relevance of user's instructions to the website according to the description of the websites' subpages.\n
            Here is the user's instruction: {instruction}.\n
            Here is the website's name: {web_name}.\n
            Here is the description of a subpage: \n{subpages}.\n
            Please remember that the user is staff or student of Zhejiang University.
            Please grade the relevance of the instruction to the website on a scale of 0 to 10.\n
            If the instruction is irrelevant to the website, please give a score of 0.\n
            If the instruction is highly relevant to the website, please give a score of 10.\n
            If the instruction is somewhat relevant to the website, please give a score between 0 and 10.\n
            Provide the score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["instruction","web_name","subpages"],
    )
    second_grading_chain = second_grading_prompt | llm | JsonOutputParser()
    select_prompt = PromptTemplate(
        template="""You are a selector selecting the best subpage related to the user's instruction.\n
            Here is the user's instruction: {instruction}.\n
            Here is the subpages' url and description: \n{subpages}.\n
            Please select the most relevant subpage to the user's instruction.\n
            Provide the selected subpage's url as a JSON with a single key 'url' and no premable or explanation.""",
    )
    select_chain = select_prompt | llm | JsonOutputParser()
    grade = {}
    # 第一轮打分
    for web_name in WEBSITE_SUMMARY:
        score = grading_chain.invoke({
            "instruction":question,
            "web_name":web_name,
            "summary":WEBSITE_SUMMARY[web_name]
        })
        grade[web_name] = score['score']
    best_trhee_website = selece_best_three_website(grade) #这里可能会是一个空列表，此时可以提前退出

    # print("best_trhee_website:",best_trhee_website)

    # 第二轮打分，给子页面打分
    grade = {} # 清空
    description = {}

    for web_name in best_trhee_website:
        for subpages in SUBPAGES_DESCRIPTION[web_name]:
            score = second_grading_chain.invoke({
                "instruction":question,
                "web_name":web_name,
                "subpages":subpages
            })
            grade[subpages['url']] = score['score']
            description[subpages['url']] = subpages['description']
    best_three_subpages = selece_best_three_website(grade)
    description = {url:description[url] for url in best_three_subpages}
    # print(description)
    # TODO 两次都可能输出空列表，提前退出，统一做在这里
    if len(best_three_subpages) == 0:
        return {
            "question": question, 
            "generation": {'action':'answer','content':'万分抱歉，我当前的能力无法帮助您解决该问题。'}, 
            "documents": state["documents"], 
            "action": "answer"
        }
    # 选择最符合的子页面
    select_subpage = select_chain.invoke({
        "instruction": question,
        "subpages":description
    })
    # print(select_subpage['url'])
    return {
        "question": question, 
        "generation": {'action':'open','content':select_subpage['url']}, 
        "documents": state["documents"], 
        "action": "open"
    }
    # navigation_prompt = PromptTemplate(
    #     template="""You are an expert at selecting related content for a user's question based on the document list {list}. \n
    #     Do not be stringent with the keywords in the question; focus on related topics. \n
    #     Provide up to three options related to the user's question from the document list {list}.\n
    #     Return a JSON with the key 'url' and no preamble or explanation. \n
    #     If there are no related documents, return a JSON with the key 'option' and the value 'none'. \n
    #     Question: {question}""",
    #     input_variables=["question","list"],
    # )
    # d_list = []
    # d_dict = {}
    # with open('apps.json','r') as json_file:
    #     loaded_data = json.load(json_file)
    #     for value in loaded_data.values():
    #         d_list.append(value['app'])
    #         d_dict[value['app']] = value['link']
    # navigation_chain = navigation_prompt | llm | JsonOutputParser()
    # navigation_result = navigation_chain.invoke({"question": question, "list": d_list})
    # try:
    #     url = d_dict.get(navigation_result['url'])
    # except:
    #     raise ValueError("The url is not found")
    # generation = {'action':'open','content':url}
    # print("generation:", generation)
    # return {"question": question, "generation": generation, "documents": state["documents"], "action": "open"}

def retieve_docs(state: GraphState):
    """
    Retrieve documents based on the question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    # print('---RETRIEVE DOCS---')
    question = state["question"]
    retiever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 50}
    ) #retiever定义在内部，根据情况可以更改参数
    documents = retiever.get_relevant_documents(question)
    # print("documents:", documents)
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    # print("---GRADE DOCS---")
    question = state["question"]
    llm = ChatOllama(model=MODEL, format="json", temperature=0, base_url=OLLAMA_URL)
    documents = state["documents"]
    retrieval_grade_prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["document", "question"],
    )
    retrieval_grade_chain = retrieval_grade_prompt | llm | JsonOutputParser()
    filtered_documents = []
    for d in documents:
        score = retrieval_grade_chain.invoke({"document": d.page_content, "question": question})
        try:
            grade = score["score"]
        except:
            raise ValueError("The score is not found")
        if grade == "yes":
            filtered_documents.append(d)
    # print("filtered_documents:", filtered_documents)
    return {"documents": filtered_documents, "question": question}

def generate_answer(state: GraphState):
    """
    Generate answer
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    # print("---GENERATE ANSWER---")
    question = state["question"]
    context = ""
    for doc in state["documents"]:
        context += doc.page_content + "\n\n"
    llm = ChatOllama(model=MODEL, temperature=0,base_url=OLLAMA_URL)
    answer_prompt = PromptTemplate(
        template="""
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Answer using the same language as in the question. Some sentences in the context are redundant. If you don't know the answer, just say that you don't know. Keep the answer concise.\n
            Question: {question} \n
            Context: {context} \n
        """,
        input_variables=["question", "context"],
    )
    answer_chain = answer_prompt | llm | StrOutputParser()
    generation = answer_chain.invoke({"question": question, "context": context})
    # print("generation:", generation)
    new_generation = {"action":"answer","content":generation}
    return {"question": question, "generation": new_generation, "documents": state["documents"], "action": "answer"}
    

# EDGE
def route_action(state: GraphState):
    """
    Decide the next action based on the current state.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """
    # print("---ROUTE ACTION---")
    qusetion = state["question"]
    # print("question:", qusetion)
    #创建route_chain
    llm = ChatOllama(model=MODEL, format="json", temperature=0,base_url=OLLAMA_URL)
    route_prompt = PromptTemplate(
        template="""
        You are an expert at selecting what to do next based on the user's input. \n
        You can choose to open a webpage or answer the user's question. \n
        Here are some websites you can open: {websites}\n
        Just return a JSON with the key 'action' and the value 'open' or 'answer'. \n
        Question: {question}
        """,
        input_variables=["question","websites"],
    )
    route_chain = route_prompt | llm | JsonOutputParser()
    route_result = route_chain.invoke({"question": qusetion,"websites":website_dict})
    #返回一个json，包含action，值为open或answer
    # print("route_result:", route_result)
    if "action" in route_result:
        if route_result["action"] == "open":
            return "open"
        elif route_result["action"] == "answer":
            return "answer"
    raise ValueError("The action is not found")

chat_workflow = StateGraph(GraphState)
#define nodes
chat_workflow.add_node("open", open_navigation)
chat_workflow.add_node("retrieve", retieve_docs)
chat_workflow.add_node("grade", grade_documents)
chat_workflow.add_node("answer", generate_answer)

#build graph
chat_workflow.add_conditional_edges(
    START,
    route_action,
    {
        "open": "open",
        "answer": "retrieve",
    },
)
chat_workflow.add_edge("open", END)
chat_workflow.add_edge("retrieve", "grade")
chat_workflow.add_edge("grade", "answer")
chat_workflow.add_edge("answer", END)

chat_app = chat_workflow.compile()

'''
1. 问答系统
2. 网页导航系统
    一级网页通过建立字典，让LLM直接做选择题，不做embedding
    二级网页需要使用LLM分析一级页面的元素
        对于二级网页来说，一种是事先分析，将所有相关信息存储在vectorstore中，然后通过相似性检索；另一种是实时分析
    部分网页涉及到用户认证，整个LLM思考链条中需要能够及时停止，等待用户执行相关操作后再继续的能力
        另一个需求是查询系统内部的信息，无需接入数据库，通过网页方式查询。不过这种方式需要有分析页面元素的能力
3. 网页导航plus
    除了跳转的功能，还需要进行type。需要分析页面中的输入框，将指定信息填入
'''

#接入flask
# app = Flask(__name__)
# CORS(app)


# @app.route('/chat', methods = ['POST'])
# def chat():
#     # 未设置历史记录
#     print('get http request for record!')
#     message = request.json['message']
#     # print('message:',message)
#     action = action_chain(message)
#     # print('action:',action)
#     result = {'action':action}
#     if action == 'open':
#         answer = navigate_chain(message)
#         print('answer:',answer)
#         result['answer'] = answer
#         # 添加验证步骤
#     return jsonify(result)

# @app.route('/chatgraph', methods = ['POST'])
# def chatgraph():
#     print('get http request for record!')
#     message = request.json['message']
#     llm_result = chat_app.invoke({"question": message}) 
#     return_result = llm_result['generation'] 
#     #generation 已经是json形式，含有action和content
#     return_result['rag'] = llm_result['documents']
#     # TODO 
#     # 记录时间开销 仅调用大模型的时间开销
#     return jsonify(return_result)

import requests
import pandas as pd
import time
import concurrent.futures
df = pd.read_csv("测试用例.csv", encoding="utf-8")
input = df["输入"]
output = df["输出"]
total_time = df["总时间开销"]
rag_doc = df['rag']
# warm up
for i in range(3):
    llm_result = chat_app.invoke({"question": input[i]})
    print(f"warm up {i}")

def function(question:str):
    start_time = time.time()
    llm_result = chat_app.invoke({"question": question})
    end_time = time.time()
    print({"input":question,"time":end_time-start_time})
    return {"input":question,"time":end_time-start_time}

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results = executor.map(function,input)
    
    
# for i,question in enumerate(input):
#     start_time = time.time()
#     llm_result = chat_app.invoke({"question": question})
#     end_time = time.time()
#     print(llm_result)
#     total_time[i] = end_time - start_time
#     if 'documents' in llm_result:
#         rag_doc[i] = str(llm_result['documents'])
#     output[i] = llm_result['generation']['content']

# df["输出"] = output
# df["rag"] = rag_doc
# df["总时间开销"] = total_time
# df.to_csv("测试用例.csv", encoding="utf-8",index=False)
