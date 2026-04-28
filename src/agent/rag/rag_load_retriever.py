# src/agent/rag/rag_load_retriever.py
from dotenv import load_dotenv
from .vectorstore_utils import get_retriever
#RAG_PDF Part2
#作向量库的加载&检索
# ========= 配置 =========

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"

load_dotenv()
def rag_load_retriever(k: int = 4):
    """
    加载向量库并创建 retriever

    Args:
        k: 返回最相关的 chunk 数

    Returns:
        retriever 对象
    """
    # 直接使用公共函数，这里使用相同的 embedding 配置
    retriever = get_retriever(
        k=k,
        embedding_model="text-embedding-3-small"
    )
    return retriever