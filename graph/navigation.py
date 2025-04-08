# chroma 客户端
from .chroma import CHROMA_CLIENT, MyEmbeddingFunction

collection_summary = CHROMA_CLIENT.get_or_create_collection(
    name="summary", 
    embedding_function=MyEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"}, # 只有 "l2", "ip, "or "cosine" 默认是 l2
)

collection_subpages = CHROMA_CLIENT.get_or_create_collection(
    name="subpages", 
    embedding_function=MyEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"}, # 只有 "l2", "ip, "or "cosine" 默认是 l2
)

# langgraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from typing import List
import time

MODEL = "qwen2:72b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:14005"
ONE_TURN = 5 # 一次retrive的数量
PRINT_FLAG = True
THRESHOLD = 5 # 评分阈值 低于这个阈值需要重新检索 满分为10

class FIRST_RETRIEVAL(TypedDict):
    name: str # 网页名称
    description: str # 网页描述
    score: int # LLM给分

class SECOND_RETRIEVAL(TypedDict):
    name: str # 网页名称
    description: str # 网页描述
    title: str # 网页标题
    url: str
    score: int # LLM给分

class NavigationState(TypedDict):
    """代表图的状态"""
    question: str
    answer: str
    first_retrieval: List[FIRST_RETRIEVAL] # 这两个一开始没赋值前是一个None
    second_retrieval: List[SECOND_RETRIEVAL]
    start_time: float
    end_time: float
    breakdown: List[str]
    breakdown_time: List[float]


def first_retriever(state: NavigationState):
    start_time = time.time()
    if state['start_time'] is None:
        state['start_time'] = start_time
    # 一级检索，从summary中检索
    question = state['question']
    first_retrieval = state["first_retrieval"] if state["first_retrieval"] is not None else []# 一开始为None
    retrieval = collection_summary.query(
        query_texts = [question],
        n_results = ONE_TURN + len(first_retrieval)
    ).get('metadatas')[0]
    #避免重复加入
    for r in retrieval:
        if r['web_name'] not in [f['name'] for f in first_retrieval]:
            first_retrieval.append({
                "name":r['web_name'],
                "description":r['description'],
                "score": -1, # -1表示未评分
            })
    if PRINT_FLAG:
        print("---FIRST RETRIVER---")
        print("retrieval:",retrieval)
    end_time = time.time()
    if state["end_time"] is None: # 仅有这个结点需要判断，其他结点无需进行这一步判断
        consume_time = end_time - start_time
    else:
        consume_time = end_time - state["end_time"]
    if state["breakdown"] is None:
        state["breakdown"] = [] #初始化
    if state["breakdown_time"] is None:
        state["breakdown_time"] = [] #初始化
    state["end_time"] = end_time
    state["breakdown"].append("first_retrieval")
    state["breakdown_time"].append(consume_time)
    state["first_retrieval"] = first_retrieval
    return state

def first_grader(state: NavigationState):
    start_time = time.time()
    # 一级评分
    first_retrieval = state["first_retrieval"]
    question = state['question']
    # 需要评分的部分会小于等于ONE_TURN 先收集，一起输入给LLM
    need_grading = [{"name":f['name'],"description":f['description']} for f in first_retrieval if f['score'] == -1]
    llm = ChatOllama(model=MODEL, format="json", temperature=0, base_url=OLLAMA_URL)
    grading_prompt =  PromptTemplate(
        template="""You are a grader assessing relevant of user's instructions to the website.\n
            Here is the user's instruction: {instruction}.\n
            Here is the website's name and description: {need_grading}.\n
            Please remember that the user is staff or student of Zhejiang University.
            Please grade the relevance of the instruction to the website on a scale of 0 to 10.\n
            If the instruction is irrelevant to the website, please give a score of 0.\n
            If the instruction is highly relevant to the website, please give a score of 10.\n
            If the instruction is somewhat relevant to the website, please give a score between 0 and 10.\n
            Provide the score as a JSON with all web name mentioned above and no premable or explanation. You must provide the JSON like this: {{"web_name":score}}""",
        input_variables=["instruction","need_grading"],
    )
    # 评分
    grading_chain = grading_prompt | llm | JsonOutputParser()
    grading_results = grading_chain.invoke({"instruction":question,"need_grading":need_grading})
    # 更新评分
    for f in first_retrieval: #这是引用传递的 没问题
        if f['name'] in grading_results:
            f['score'] = grading_results[f['name']]
    # 更新评分
    if PRINT_FLAG:
        print("---FIRST GRADER---")
        print("need_grading:",need_grading)
        print("grading_results:",grading_results)
        # print("first_retrieval:",first_retrieval)
    end_time = time.time()
    consume = end_time - state["end_time"] # 与前一个结点的时间差 有些conditional_edge的时间会包含在结点里面
    state["end_time"] = end_time # 更新时间
    state["breakdown"].append("first_grader")
    state["breakdown_time"].append(consume)
    state["first_retrieval"] = first_retrieval
    return state


def first_filter(state: NavigationState):
    # 一级过滤，判断是否需要重新进行一级检索 还要考虑已经没有更多的需要检索了
    first_retrival = state["first_retrieval"]
    total_num = collection_summary.count()
    flag = False
    for i in first_retrival:
        if i['score'] >= THRESHOLD:
            flag = True
            break
    if flag:
        return "yes"
    elif len(first_retrival) >= total_num:
        return "full" # 已经没有更多的需要检索了 可以直接跳过二级检索直接结束
    else:
        return "no"
        
def second_retriever(state: NavigationState):
    start_time = time.time()
    # 二级检索
    # 从一级检索的结果中找到评分超出阈值的web，然后从subpages中检索
    question = state["question"]
    first_retrieval = state["first_retrieval"]
    second_retrieval = state["second_retrieval"] if state["second_retrieval"] is not None else []
    # flag = False # 如果没有改变，则没有超出阈值的web, 这不用判断，这在first_filter中已经判断过一次
    for f in first_retrieval:
        if f['score'] >= THRESHOLD:
            web_name = f['name']
            # 判断该网页的所有子页面是否已经都被检索
            # subpages = collection_subpages.get(where={'web_name':web_name}) 
            # total_num = len(subpages['ids'])
            now_num = len([sub['url'] for sub in second_retrieval if sub["name"] == web_name]) #
            # if now_num >= total_num:
            #     #所有子页面已经被检索
            #     #这个应该被放到second_filter中

            retrieval = collection_subpages.query(
                query_texts=[question],
                n_results=now_num+ONE_TURN,
                where={'web_name':web_name}
            ).get('metadatas')[0]
            # 避免重复加入
            for r in retrieval:
                if r['url'] not in [subpage['url'] for subpage in second_retrieval]:
                    second_retrieval.append(
                        {
                            "name":r['web_name'],
                            "description":r['description'],
                            'title':r['title'],
                            'url':r['url'],
                            'score':-1,#-1 表示未评分
                        }
                    )
    if PRINT_FLAG:
        print("---SECOND RETRIEVER---")
        print("second_retrieval:",second_retrieval)
    end_time = time.time()
    consume = end_time - state["end_time"]
    state["end_time"] = end_time
    state["breakdown"].append("second_retriever")
    state["breakdown_time"].append(consume)
    state["second_retrieval"] = second_retrieval
    state["first_retrieval"] = first_retrieval 
    return state

def second_grader(state: NavigationState):
    start_time = time.time()
    # 二级评分
    question = state["question"]
    second_retrieval = state["second_retrieval"]
    first_retrieval = state['first_retrieval']
    need_grading = [{"url":s["url"],"name":s['name'],"title":s['title'],"description":s['description']} for s in second_retrieval if s['score'] == -1]  #这里最多会有 ONE_TURN*ONE_TURN个

    llm = ChatOllama(model=MODEL, format="json", temperature=0, base_url=OLLAMA_URL)
    grading_prompt =  PromptTemplate(
        template="""You are a grader assessing relevant of user's instructions to the website.\n
            Here is the user's instruction: {instruction}.\n
            Here is the website's url, name, title and description: {need_grading}.\n
            Please remember that the user is staff or student of Zhejiang University.
            Please grade the relevance of the instruction to the website on a scale of 0 to 10.\n
            If the instruction is irrelevant to the website, please give a score of 0.\n
            If the instruction is highly relevant to the website and , please give a score of 10.\n
            If the instruction is somewhat relevant to the website, please give a score between 0 and 10.\n
            Provide the score as a JSON with all web url mentioned above and no premable or explanation. You must provide the JSON like this: {{"url":score}}""",
        input_variables=["instruction","need_grading"],
    )
    grading_chain = grading_prompt | llm | JsonOutputParser()
    for need_grading_chunk in [need_grading[i:i+ONE_TURN] for i in range(0,len(need_grading),ONE_TURN)]:
        # 切片 以防一次性输入过多
        
        grading_results = grading_chain.invoke({"instruction":question,"need_grading":need_grading_chunk})

        for s in second_retrieval: # 列表中的元素是引用传递的 这没问题
            if s['url'] in grading_results:
                s['score'] = grading_results[s['url']]
        
        if PRINT_FLAG:
            print("---SECOND GRADER CHUNK---")
            print("need_grading:",need_grading_chunk)
            print("grading_results:",grading_results)

    # 在这里判断某一网站的所有子页面是否都不超过阈值，这个网站的评分需要被修改为阈值-1
    for f in first_retrieval:
        if f["score"] >= THRESHOLD:
            web_name = f['name']
            subpages = collection_subpages.get(where={'web_name':web_name})
            total_num = len(subpages['ids']) # 该网站所有的子页面
            now_num = len([sub['url'] for sub in second_retrieval if sub['name'] == web_name]) # 该网站目前已经被检索的子页面数量 
            if now_num >= total_num:
                # 该网站的所有子页面已经被检索 修改评分=阈值-1以作标记
                # 子页面中有评分超过阈值的也没关系 它不会返回一级检索
                f['score'] = THRESHOLD - 1 if THRESHOLD - 1 > 0 else 0

    end_time = time.time()
    consume = end_time - state["end_time"]
    state["end_time"] = end_time
    state["breakdown"].append("second_grader")
    state["breakdown_time"].append(consume)

    state["second_retrieval"] = second_retrieval
    state["first_retrieval"] = first_retrieval 
    return state

def second_filter(state: NavigationState):
    # 二级过滤
    # 判断是否需要重新进行二级检索 以及是否需要进行一级检索
    question = state["question"]
    first_retrieval = state["first_retrieval"]
    second_retrieval = state["second_retrieval"]
    graded = [{'name':s['name'],'description':s['description'],'score':s['score']} for s in second_retrieval if s['score'] >= THRESHOLD] # 一级评分超过阈值的 也是二级评分的对象
    flag = False # 先判断是否又超过评分阈值的子页面   
    for g in graded:
        if g['score'] >= THRESHOLD:
            flag = True
            break
    if flag:
        #有超过评分阈值的子页面
        return "yes"
    # 没有评分超过阈值的子页面  考虑回去进一步检索 flag 复用 现在还是False
    # 考虑是否还有子页面待查
    for g in graded:
        web_name = g['name']
        subpages = collection_subpages.get(where={'web_name':web_name})
        total_num = len(subpages['ids']) # 该网站的子页面数量
        now_num = len([sub['url'] for sub in second_retrieval if sub['name'] == web_name]) #该网站目前已经被检索的子页面数量
        if now_num < total_num:
            flag = True # 有子页面待查 返回二级检索 可以提前退出
            break
        #否则 返回一级检索

    if flag:
        return "second"
    else:
        return "first"
            
def generate_answer(state: NavigationState):
    start_time = time.time()
    # 生成答案
    question = state["question"]
    second_retrieval = state["second_retrieval"]
    first_retrieval = state["first_retrieval"]
    llm = ChatOllama(model=MODEL, format="json", temperature=0, base_url=OLLAMA_URL)
    answer_prompt =  PromptTemplate(
        template="""You are a selector selecting the best subpage related to the user's instruction.\n
            Here is the user's instruction: {instruction}.\n
            Here is the subpages' url, title, web name and description: \n{subpages}.\n
            Please select the most relevant subpage to the user's instruction.\n
            Provide the selected subpage's url as a JSON with a single key 'url' and no premable or explanation. You must provide the JSON like this: {{"content":url}}""",
        input_variables=["instruction","subpages"],
    )
    answer_chain = answer_prompt | llm | JsonOutputParser()
    # 选择最相关的子页面 前面的RAG可能会返回不止一个页面
    subpages = [{"url":s['url'],"title":s['title'],"name":s['name'],"description":s['description']} for s in second_retrieval if s['score'] >= THRESHOLD] # 把评分超过阈值的子页面传入，前面的步骤保证了这里不会是个空
    answer = answer_chain.invoke({"instruction":question,"subpages":subpages})
    answer['action'] = 'open' # 添加一个动作
    if PRINT_FLAG:
        print("---GENERATE ANSWER---")
        print("answer:",answer)


    end_time = time.time()
    consume = end_time - state["end_time"]
    state["end_time"] = end_time
    state["breakdown"].append("generate_answer")
    state["breakdown_time"].append(consume)

    state["answer"] = answer
    state["second_retrieval"] = second_retrieval
    state["first_retrieval"] = first_retrieval 
    return {
        "answer":answer,
        "question":question,
        "first_retrieval":first_retrieval,
        "second_retrieval":second_retrieval
    }
    

def without_retrieval(state: NavigationState):
    # 没有检索到符合要求的页面
    start_time = time.time()
    end_time = time.time()
    consume = end_time - state["end_time"]
    state["end_time"] = end_time
    state["breakdown"].append("without_retrieval")
    state["breakdown_time"].append(consume)
    state["answer"] = {'action':'answer','content':'请给我一个更详细的描述。'}
    return state

navigation_workflow = StateGraph(NavigationState)

navigation_workflow.add_node("first_retriever",first_retriever)
navigation_workflow.add_node("first_grader",first_grader)
navigation_workflow.add_node("without_retrieval",without_retrieval)
navigation_workflow.add_node("second_retriever",second_retriever)
navigation_workflow.add_node("second_grader",second_grader)
navigation_workflow.add_node("generate_answer",generate_answer)

navigation_workflow.add_edge(START,"first_retriever")
navigation_workflow.add_edge("first_retriever","first_grader")
navigation_workflow.add_conditional_edges(
    "first_grader",
    first_filter,
    {
        "yes":"second_retriever", # 这里需要进入二级检索
        "no":"first_retriever",
        "full":"without_retrieval", # 已经没有更多的需要检索了 可以直接跳过二级检索直接结束
    },
)
navigation_workflow.add_edge("without_retrieval",END)
navigation_workflow.add_edge("second_retriever","second_grader")
# navigation_workflow.add_edge("second_grader",END)
navigation_workflow.add_conditional_edges(
    "second_grader",
    second_filter,
    {
        "yes":"generate_answer", # 这里表示有符合要求的子页面 最后接入生成回答的节点
        "first":"first_retriever",
        "second":"second_retriever",
    }
)
navigation_workflow.add_edge("generate_answer",END)

navigator = navigation_workflow.compile()