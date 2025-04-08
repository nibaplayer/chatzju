# 使用这个来进行embedding并将结果存入数据库
import os
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "nomic-embed-text"
KNOWLEDGE_DIR = "knowledge"

# connect to the database
chroma_client = chromadb.HttpClient(host="localhost",port= 14009)

filelist = os.listdir(KNOWLEDGE_DIR)
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
vectorestrore = Chroma.from_documents(all_splitter,embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),client=chroma_client,collection_name="chatzju")