# src/agent/rag/vectorstore_utils.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#RAG_PDF Part0
#初始化vectorstore的封装函数 与 创建检索器的封装函数
load_dotenv()
# 向量库路径常量
VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"

def load_vectorstore(embedding_model: str = "text-embedding-3-small",
                     base_url: str = None,
                     api_key: str = None) -> FAISS:
    """
    加载 FAISS 向量库

    参数:
        embedding_model: 使用的 embedding 模型名称
        base_url: API 基础 URL（可选，用于自定义 API 端点）
        api_key: API 密钥（可选，默认从环境变量读取）

    返回:
        FAISS 向量库对象
    """
    # 检查向量库文件是否存在
    index_file = os.path.join(
        VECTOR_DB_PATH, "index.faiss"
    )
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            "FAISS index not found.\n"
            "请先运行 build_pdf_vectorstore() 创建向量库。"
        )

    # 创建 embedding 对象
    embedding_kwargs = {"model": embedding_model}
    if base_url:
        embedding_kwargs["base_url"] = base_url
    if api_key:
        embedding_kwargs["api_key"] = api_key
    embeddings = OpenAIEmbeddings(**embedding_kwargs)

    # 加载向量库
    print("Loading FAISS vectorstore...")
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vectorstore loaded successfully.")
    return vectorstore

def get_retriever(k: int = 10,
                  embedding_model: str = "text-embedding-3-small",
                  base_url: str = None,
                  api_key: str = None):
    """
    创建检索器

    参数:
        k: 返回文档数量
        embedding_model: embedding 模型
        base_url: API 基础 URL
        api_key: API 密钥

    返回:
        retriever 对象
    """
    vectorstore = load_vectorstore(embedding_model, base_url, api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print(f"Retriever created (top {k}).")
    return retriever