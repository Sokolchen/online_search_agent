# src/agent/rag/vectorstore_utils.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


from langchain_chroma import Chroma
from typing import Optional

# RAG_PDF Part0
# 初始化vectorstore的封装函数 与 创建检索器的封装函数

load_dotenv()

# 向量库路径常量
VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


def load_vectorstore(
    embedding_model: str = "text-embedding-3-small"
) -> Chroma:
    """
    加载 Chroma 向量库

    Args:
        embedding_model: 使用的 embedding 模型名称

    Returns:
        Chroma 向量库对象
    """

    #统一 embedding 配置
    embeddings = OpenAIEmbeddings(
        model=embedding_model,  #使用形参
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    print("加载Chroma向量库...")

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="pdf_collection"
    )

    print("Vectorstore配置已完成")

    return vectorstore


def get_retriever(
    k: int = 5,
    embedding_model: str = "text-embedding-3-small",
    source_file: Optional[str] = None
):
    """
    创建检索器

    Args:
        k: 返回文档数量
        embedding_model: embedding 模型
        source_file: 指定只检索某个 PDF（可选）

    Returns:
        retriever 对象
    """

    vectorstore = load_vectorstore(
        embedding_model
    )

    # =========Chroma filter=========

    #设定search_kwargs——配置向量检索器的参数
    search_kwargs = {
        "k": k,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
    """
    先从数据库中取 fetch_k=20 个最相似的文档作为候选池，并以 lambda_mult=0.5 的比例，在“文档相关性”和“内容多样性”之间取得平衡。
    lambda_mult越接近0多样性越高，检索内容会尽量覆盖不同主题，避免内容高度重复
    越接近1则相关性越高，更专注查找更匹配的文档
    """
    #仅在合法字符串时添加 filter
    if source_file is not None:
        search_kwargs["filter"] = {
            "source_file": source_file
        }
        print(f"Retriever 建立完成 (启用了文件名 {source_file} 过滤, top {k}).")
    else:
        print(f"Retriever 建立完成 (使用了默认 MMR 检索 top {k}).")

    #接收完所有参数，统一创建 retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",#mmr让算法把所有结果视为一个整体来考虑，尽量确保信息多元与匹配
        search_kwargs=search_kwargs
    )

    #DEBUG
    print("search_kwargs =", search_kwargs)

    return retriever