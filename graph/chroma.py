# 这里构建chroma客户端
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb.config
from langchain_community.embeddings import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:14005"

embed_model = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBEDDING_MODEL)

# 定义自己的embedding function
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return embed_model.embed_documents(input)      

CHROMA_CLIENT = chromadb.PersistentClient(
    path='/root/code/chatzju/chroma_data'
)