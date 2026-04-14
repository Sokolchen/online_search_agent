# src/agent/rag/rag_load_retriever.py
from dotenv import load_dotenv
from .vectorstore_utils import get_retriever
#RAG_PDF Part2
#作向量库的加载&检索
# ========= 配置 =========

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"

load_dotenv()


# ========= ⭐ 新增：最小 document router =========
def guess_pdf(query: str):
    """
    根据用户问题选择对应 PDF（用于二次检索第一阶段）

    Args:
        query: 用户问题

    Returns:
        source_file 或 None
    """
    if not query:
        return None

    query_lower = query.lower()

    return None


def rag_load_retriever(k: int = 4, query: str = None):
    """
    加载向量库并创建 retriever

    Args:
        k: 返回最相关的 chunk 数
        query: 用户输入问题（用于二次检索时选择对应 PDF）

    Returns:
        retriever 对象
    """
    # ========= ⭐ 二次检索关键：先选 PDF =========
    source_file = guess_pdf(query)


    # 直接使用公共函数，这里使用默认的 embedding 配置（即无 base_url）
    retriever = get_retriever(
        k=k,
        embedding_model="text-embedding-3-small",
        source_file=source_file   # ⭐关键：限制在指定 PDF 内检索
    )

    return retriever