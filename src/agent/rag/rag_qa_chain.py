# src/agent/rag/rag_qa_chain.py

from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever, Document
from typing import List, Optional
from agent.my_llm import deepseek_llm
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from agent.rag.vectorstore_utils import get_retriever
from langchain_core.prompts import PromptTemplate

# RAG_PDF Part3
# 定义核心问答工具

# ========= 环境变量 =========
load_dotenv()

# ========= ⭐ FIX：已迁移到 Chroma（不再使用 FAISS） =========
VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/chroma_db"


# 清洗数据
def clean_chunk_for_qa(chunk: str) -> str:
    lines = chunk.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("Figure") or line.startswith("Table"):
            continue

        if "|" in line:
            continue

        if line.startswith("[") and line.endswith("]"):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# 清洗类定义
class CleanRetriever(BaseRetriever):

    base: BaseRetriever

    # ⭐ FIX：统一使用 invoke（Chroma retriever兼容）
    def _get_relevant_documents(
        self,
        query: str,
        run_manager: Optional = None
    ) -> List[Document]:

        docs = self.base.invoke(query)

        for doc in docs:
            doc.page_content = clean_chunk_for_qa(doc.page_content)

        return docs


# QA_PROMPT
QA_PROMPT = """
你是一个文档问答助手。

如果用户问题类似：
- 这个文件讲了什么
- 这是什么文件

只用 3~5 句话概括。

不要列出详细条目。
不要重复内容。
总字数不超过150字。

文档：
{context}

问题：
{question}

回答：
"""

prompt = PromptTemplate(
    template=QA_PROMPT,
    input_variables=["context", "question"]
)


# ========= 创建 QA Chain =========
def rag_qa_chain():
    """
    加载向量库并创建 RetrievalQA 链
    """

    # ========= ⭐ Chroma retriever =========
    base_retriever = get_retriever(
        k=4,
        embedding_model="text-embedding-3-small",
        source_file=None   # ⭐ 不强制绑定PDF（由router控制更合理）
    )

    cleaned_retriever = CleanRetriever(base=base_retriever)

    qa_chain = RetrievalQA.from_chain_type(
        llm=deepseek_llm,
        retriever=cleaned_retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt
        }
    )

    print("QA chain created successfully.")

    return qa_chain


# ========= 提供单独问答函数 =========
def ask_pdf(question: str):
    """
    对 PDF 知识库提问
    """

    qa = rag_qa_chain()

    retriever = qa.retriever

    docs = retriever.invoke(question)

    for i, d in enumerate(docs):
        print("\n====================")
        print(f"[{i}] source:", d.metadata.get("source_file"))
        print(f"page:", d.metadata.get("page"))
        print(d.page_content[:300])

    result = qa.invoke({"query": question})

    return result["result"]