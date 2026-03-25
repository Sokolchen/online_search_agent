# src/agent/rag/rag_retriever.py

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ========= 配置 =========

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"

load_dotenv()


def rag_load_retriever(k: int = 10):
    """
    加载向量库并创建 retriever

    参数:
        k: 返回最相关的 chunk 数

    返回:
        retriever 对象
    """

    # ========= 检查向量库是否存在 =========

    index_file = os.path.join(
        VECTOR_DB_PATH,
        "index.faiss"
    )

    if not os.path.exists(index_file):

        raise FileNotFoundError(
            "FAISS index not found.\n"
            "请先运行 build_pdf_vectorstore() 创建向量库。"
        )

    # ========= 加载 embedding =========

    embeddings = OpenAIEmbeddings()

    # ========= 加载向量库 =========

    print("Loading FAISS vectorstore...")

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Vectorstore loaded successfully.")

    # ========= 创建 retriever =========

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    print(f"Retriever created (top {k}).")

    return retriever