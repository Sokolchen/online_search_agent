from agent.my_llm import deepseek_llm

from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()


# ========= 向量库路径 =========

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"


# ========= 创建 embedding =========

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://api.shubiaobiao.com/v1"
)


# ========= 加载 FAISS =========

index_file = os.path.join(
    VECTOR_DB_PATH,
    "index.faiss"
)

if not os.path.exists(index_file):
    raise FileNotFoundError(
        "FAISS index not found. "
        "请先运行 build_pdf_vectorstore() 创建向量库。"
    )

print("Loading FAISS vectorstore...")

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embedding,
    allow_dangerous_deserialization=True
)

print("FAISS loaded successfully.")


# ========= 创建 retriever =========

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# ========= （可选）调试查看 chunk =========

# query = "什么是 CET_SET_RULES？"
# docs = vectorstore.similarity_search(query, k=5)
# for i, doc in enumerate(docs):
#     print(f"Chunk {i}:")
#     print("Source:",
#           doc.metadata.get("source_file"))
#     print(doc.page_content[:200])
#     print("-" * 50)


# ========= 创建 RetrievalQA =========

qa_chain = RetrievalQA.from_chain_type(
    llm=deepseek_llm,
    retriever=retriever,
    chain_type="stuff",  # 最常用
)


# ========= 提问 =========

query = "什么是 CET_SET_RULES？"

print("\nQuestion:")
print(query)


# ========= 获取回答 =========

result = qa_chain.invoke(
    {"query": query}
)

answer = result["result"]


# ========= 输出 =========

print("\nAnswer:\n")

print(answer)