# src/agent/rag/vectorstore_utils.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# ⭐ FIX：FAISS → Chroma
from langchain_chroma import Chroma

from typing import Optional

# RAG_PDF Part0
# 初始化vectorstore的封装函数 与 创建检索器的封装函数

load_dotenv()

# 向量库路径常量
VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/chroma_db"

# ⭐ 统一从 .env 读取 API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


def load_vectorstore(
    embedding_model: str = "text-embedding-3-small"
) -> Chroma:
    """
    加载 Chroma 向量库

    参数:
        embedding_model: 使用的 embedding 模型名称

    返回:
        Chroma 向量库对象
    """

    # ⭐ 强制统一 embedding 配置（避免 URL 拼接错误）
    embeddings = OpenAIEmbeddings(
        model=embedding_model,  # ✅ 使用形参
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    print("Loading Chroma vectorstore...")

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="pdf_collection"
    )

    print("Vectorstore loaded successfully.")

    return vectorstore


def get_retriever(
    k: int = 4,
    embedding_model: str = "text-embedding-3-small",
    source_file: Optional[str] = None
):
    """
    创建检索器

    参数:
        k: 返回文档数量
        embedding_model: embedding 模型
        source_file: 指定只检索某个 PDF

    返回:
        retriever 对象
    """

    vectorstore = load_vectorstore(
        embedding_model
    )

    # ========= ⭐ Chroma 原生 filter（替代你原来的 Python过滤） =========
    if source_file is None:

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 20,
                "lambda_mult": 0.5
            }
        )

        print(f"Retriever created (global MMR top {k}).")
        return retriever

    # ========= ⭐ Chroma metadata filter（核心优化） =========
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "filter": {
                "source_file": source_file
            }
        }
    )

    print(f"Retriever created (filtered by {source_file}, top {k}).")

    return retriever