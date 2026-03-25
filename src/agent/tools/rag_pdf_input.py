# src/agent/rag/rag_pdf_input.py

"""
PDF 动态导入工具

作用：
允许用户在运行时输入 PDF 路径，
并自动加入 FAISS 向量库。

成功后：
显示当前已有 PDF 文件列表

失败后：
提示可能原因
"""

import os
from langchain.tools import tool

from agent.rag.pdf_indexer import build_pdf_vectorstore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ========= 向量库路径 =========

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"


def list_existing_pdfs():
    """
    获取当前向量库中的 PDF 文件列表
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

        files = set()

        for k in docs:

            metadata = docs[k].metadata

            if "source_file" in metadata:
                files.add(
                    metadata["source_file"]
                )

        return sorted(files)

    except Exception:

        return []


@tool
def rag_pdf_input(_: str = "") -> str:
    """
    当用户希望向知识库添加新的PDF文件时使用该工具。

    例如：
    - 添加新的PDF
    - 导入新的文件
    - 更新知识库
    - 增加新的文档

    工具会请求用户输入 PDF 文件路径。
    """

    print("\n========== PDF IMPORT TOOL ==========\n")

    # ========= 请求用户输入路径 =========

    pdf_path = input(
        "请输入要添加的 PDF 文件路径：\n"
    )

    pdf_path = pdf_path.strip()

    # ========= 检查路径 =========

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

        print("\n正在解析 PDF...\n")

        # ========= 调用 indexer =========

        build_pdf_vectorstore(
            [pdf_path]
        )

        print("\nPDF 添加完成。\n")

        # ========= 显示当前库 =========

        files = list_existing_pdfs()

        if files:

            file_list_text = "\n".join(
                f"- {f}"
                for f in files
            )

            return (
                "✅ PDF 添加成功！\n\n"
                "当前本地知识库包含：\n"
                f"{file_list_text}"
            )

        else:

            return (
                "⚠ PDF 已添加，但无法读取当前库列表。"
            )

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