# src/agent/rag/rag_qa_chain.py
from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever,Document
from typing import List, Optional
from agent.my_llm import deepseek_llm
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from agent.rag.vectorstore_utils import get_retriever

#RAG_PDF Part3
#定义核心问答工具
# ========= 环境变量 =========

load_dotenv()


# ========= 向量库路径 =========

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"
#清洗数据
def clean_chunk_for_qa(chunk: str) -> str:#接收chunk处理后返回（字符串）
    lines = chunk.split("\n")#将输入字符串chunk按照 \n 拆分成一个列表，每个元素是原始文本中的一行。
    cleaned = []#存放清洗后的行
    for line in lines:
        line = line.strip()#移除当前行开头和结尾的空白字符
        if not line:
            continue#此行为空或只包含空白字符时跳过
        if line.startswith("Figure") or line.startswith("Table"):
            continue#如果行以 "Figure" 或 "Table" 开头，则认为是图表标题，跳过该行
        if "|" in line:
            continue#如果行中包含竖线字符 |，则认为是表格的一部分，跳过该行
        if line.startswith("[") and line.endswith("]"):
            continue#如果行的开头是 [并且结尾是]，即整行内容为 [xxx] 格式（如 [1]、[citation]），则视为引用标记，跳过该行
        cleaned.append(line)
    return "\n".join(cleaned)
#清洗类定义-使用继承base
class CleanRetriever(BaseRetriever):#这是一个检索器组件
    # 使用Pydantic字段注解，声明一个类属性base，类型注解为BaseRetriever表示该检索器内部包含一个“基础”检索器对象
    #实例化时需要传入base参数
    base: BaseRetriever

    def _get_relevant_documents(#重写 _get_relevant_documents 方法
            self,
            query: str,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        docs = self.base._get_relevant_documents(query, run_manager=run_manager)#调用被重写的方法
        for doc in docs:
            doc.page_content = clean_chunk_for_qa(doc.page_content)
        return docs
    #最后，对每个文档的 page_content 属性进行修改：调用函数 clean_chunk_for_qa
    #传入原始文本内容，得到清洗后的文本，再赋值回 doc.page_content，返回清洗文档docs



# ========= 创建 QA Chain =========

def rag_qa_chain():

    """
    加载向量库并创建 RetrievalQA 链

    返回:
        qa_chain
    """
#加载vectorstore/faiss_index/index.faiss

    base_retriever = get_retriever(
        k=10,
        embedding_model="text-embedding-3-small",
        base_url="https://api.shubiaobiao.com/v1"
    )

    cleaned_retriever = CleanRetriever(base=base_retriever)
#清洗由检索器检索回的数据，在不改变原始数据下过滤对查找原理类型问题的干扰
    # ========= QA chain =========

    qa_chain = RetrievalQA.from_chain_type(
        llm=deepseek_llm,
        retriever=cleaned_retriever,
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