# src/agent/rag/rag_pdf_input.py

import os
from langchain.tools import tool
from agent.rag.pdf_indexer import build_pdf_vectorstore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma


VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/chroma_db"


def list_existing_pdfs():
    """获取当前向量库中的 PDF 文件列表。

    Returns:
        list[str]: 当前向量库中已存在的 PDF 文件列表
    """

    try:
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url="https://api.shubiaobiao.com/v1"
        )

        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding,
            collection_name="pdf_collection"
        )

        # ⭐ Chroma 官方方式获取 metadata
        results = vectorstore.get(include=["metadatas"])

        metadatas = results.get("metadatas", [])

        files = set()

        for meta in metadatas:
            if not meta:
                continue
            if "source_file" in meta:
                files.add(meta["source_file"])

        return sorted(files)

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

    if not pdf_path.lower().endswith(".pdf"):
        return (
            "❌ 解析失败：文件不是 PDF。\n"
            "请提供 .pdf 文件路径。"
        )

    try:
        build_pdf_vectorstore([pdf_path])

        files = list_existing_pdfs()

        if files:
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
            "2. Embedding API 失败\n"
            "3. PDF 内容为空\n\n"
            f"错误信息：{str(e)}"
        )