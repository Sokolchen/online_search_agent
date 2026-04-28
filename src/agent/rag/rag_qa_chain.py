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
    lines = chunk.split("\n")#将 chunk 按换行符 "\n" 拆分成多个字符串
    cleaned = []

    for line in lines:
        line = line.strip()#删除首尾空白

        if not line:#跳过空字符串
            continue

        if line.startswith("Figure") or line.startswith("Table"):#过滤图形与图表
            continue

        if "|" in line:#过滤有表格符号的行
            continue

        if line.startswith("[") and line.endswith("]"):#过滤形为[......]的行
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# 清洗类定义
class CleanRetriever(BaseRetriever):

    base: BaseRetriever


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
你是一个严格的文档问答助手。你必须**仅依据**下方【文档】内容回答，不得使用任何外部知识或猜测。

## 回答规则
1. 如果用户问“这个文件讲了什么”或类似概括性问题：
   - 用 3~5 句话概括文档主旨。
   - 不列条目、不重复、不超过150字。

2. 对于其他具体问题：
   - 如果在【文档】中找到直接答案，请引用相关原文并给出准确回答。
   - 如果【文档】中**完全没有提及**问题的关键信息，请直接回答：
     “文档未提及相关内容，无法基于文档作答。”
   - 如果【文档】中只有**部分相关**内容，请先说明“文档仅提到……”，再说明“未涉及……”。

3. 绝对禁止补充文档中没有的专业知识、技术对比或背景介绍。

## 文档内容
{context}

## 用户问题
{question}

## 回答
"""

prompt = PromptTemplate(
    template=QA_PROMPT,
    input_variables=["context", "question"]
)


# ========= 创建 QA Chain =========
def rag_qa_chain(source_file: str = None):
    """
    加载向量库并创建 RetrievalQA 链
    """

    # ========= Chroma retriever =========
    base_retriever = get_retriever(
        k=6,
        embedding_model="text-embedding-3-small",
        source_file=source_file
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
def ask_pdf(question: str, source_file: str = None):
    """
    对 PDF 知识库提问
    """

    qa = rag_qa_chain(source_file=source_file)

    retriever = qa.retriever

    docs = retriever.invoke(question)

    for i, d in enumerate(docs):#debug用
        print("\n====================")
        print(f"[{i}] source:", d.metadata.get("source_file"))
        print(f"page:", d.metadata.get("page"))
        print(d.page_content[:300])

    result = qa.invoke({"query": question})

    return result["result"]