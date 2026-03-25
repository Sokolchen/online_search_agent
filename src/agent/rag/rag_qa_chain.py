# src/agent/rag/rag_qa_chain.py

import os
from dotenv import load_dotenv

from agent.my_llm import deepseek_llm

from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAIEmbeddings


# ========= 环境变量 =========

load_dotenv()


# ========= 向量库路径 =========

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"


# ========= 创建 QA Chain =========

def rag_qa_chain():

    """
    加载向量库并创建 RetrievalQA 链

    返回:
        qa_chain
    """

    index_file = os.path.join(
        VECTOR_DB_PATH,
        "index.faiss"
    )

    if not os.path.exists(index_file):

        raise FileNotFoundError(
            "FAISS index not found.\n"
            "请先运行 build_pdf_vectorstore() 创建向量库。"
        )

    print("Loading FAISS vectorstore...")

    # ========= embedding =========

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://api.shubiaobiao.com/v1"
    )

    # ========= 加载向量库 =========

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )

    print("FAISS loaded successfully.")


    # ========= retriever =========

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )


    # ========= QA chain =========

    qa_chain = RetrievalQA.from_chain_type(
        llm=deepseek_llm,
        retriever=retriever,
        chain_type="stuff"
    )

    print("QA chain created successfully.")

    return qa_chain


# ========= 提供单独问答函数 =========

def ask_pdf(question: str):

    """
    对 PDF 知识库提问

    参数:
        question: 用户问题

    返回:
        answer: 回答字符串
    """

    qa = rag_qa_chain()

    result = qa.invoke(
        {"query": question}
    )

    return result["result"]