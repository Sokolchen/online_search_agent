# src/agent/rag/rag_pdf_input.py
import os
from langchain.tools import tool
from agent.rag.pdf_indexer import build_pdf_vectorstore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"


def list_existing_pdfs():
    """获取当前向量库中的 PDF 文件列表。

    Returns:
        list[str]: 当前向量库中已存在的 PDF 文件路径列表
    """
    index_file = os.path.join(
        VECTOR_DB_PATH,
        "index.faiss"
    )

    if not os.path.exists(index_file):
        return []

    try:
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )
        docs = vectorstore.docstore._dict
        files = set()  # 创建不允许重复的集合，保证是1个PDF对多个chunk

        for k in docs:
            metadata = docs[k].metadata
            if "source_file" in metadata:
                files.add(metadata["source_file"])

        return sorted(files)  # 排序
    except Exception:
        return []


@tool("rag_pdf_input", parse_docstring=True)
def rag_pdf_input(pdf_path: str) -> str:
    """向本地知识库添加新的 PDF 文件。

    当用户希望向知识库添加新的PDF文件时使用该工具。
    例如：
    - 添加新的PDF
    - 导入新的文件
    - 更新知识库
    - 增加新的文档
    用户需输入 PDF 文件路径。

    Args:
        pdf_path (str): PDF 文件的绝对路径

    Returns:
        str: 处理结果消息，包括成功添加、当前知识库列表或错误信息
    """
    pdf_path = pdf_path.strip()  # 删除空格

    if not os.path.exists(pdf_path):
        return (
            "❌ 解析失败：路径不存在。\n"
            "可能原因：\n"
            "1. 路径拼写错误\n"
            "2. 文件不存在\n"
            "3. 未使用绝对路径\n"
        )

    if not pdf_path.lower().endswith(".pdf"):  # 文件名小写化检查是否为PDF
        return (
            "❌ 解析失败：文件不是 PDF。\n"
            "请提供 .pdf 文件路径。"
        )

    try:  # 使用[]以支持批量处理
        build_pdf_vectorstore([pdf_path])
        files = list_existing_pdfs()

        if files:  # 分隔符.join格式+数据
            file_list_text = "\n".join(f"- {f}" for f in files)
            return (
                "✅ PDF 添加成功！\n\n"
                "当前本地知识库包含：\n"
                f"{file_list_text}"
            )
        else:
            return "⚠ PDF 已添加，但无法读取当前库列表。"

    except Exception as e:
        return (
            "❌ PDF 解析失败。\n"
            "可能原因：\n"
            "1. PDF 文件损坏\n"
            "2. Java 未正确安装（OpenDataLoader 依赖）\n"
            "3. API Key 或 Embedding 失败\n"
            "4. PDF 内容为空\n\n"
            f"错误信息：{str(e)}"
        )