# src/agent/rag/rag_manage_tools.py

import os
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/chroma_db"


def load_vectorstore():
    """
    加载 Chroma 向量库。
    """

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://api.shubiaobiao.com/v1"
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding,
        collection_name="pdf_collection"
    )

    return vectorstore


@tool("rag_list_vectorstore", parse_docstring=True)
def rag_list_vectorstore(_: str = "") -> str:
    """查看当前向量库中存储的 PDF 文件及每个文件占用的 chunk 数量。

    Args:
        _ (str): 可选参数，暂未使用

    Returns:
        str: 当前向量库状态描述，包括每个文件的 chunk 数量和总 chunk 数
    """
    try:
        vectorstore = load_vectorstore()

        # ⭐ Chroma 正确方式：get all data
        results = vectorstore.get(include=["metadatas"])

        metadatas = results.get("metadatas", [])

        file_chunks = {}

        for meta in metadatas:
            file_name = meta.get("source_file", "未知文件")
            file_chunks[file_name] = file_chunks.get(file_name, 0) + 1

        if not file_chunks:
            return "当前向量库为空，没有存储任何 PDF 文件。"

        result_lines = ["当前向量库状态："]
        total_chunks = 0

        for f, c in file_chunks.items():
            result_lines.append(f"- {f}: {c} chunks")
            total_chunks += c

        result_lines.append(f"总 chunk 数量: {total_chunks}")

        return "\n".join(result_lines)

    except Exception as e:
        return f"查看向量库失败:\n{str(e)}"


@tool("rag_delete_pdf", parse_docstring=True)
def rag_delete_pdf(file_name: str) -> str:
    """删除指定 PDF 文件在向量库中的向量数据。

    Args:
        file_name (str): 需要删除的 PDF 文件名

    Returns:
        str: 删除结果描述，包括删除的 chunk 数量或错误信息
    """
    try:
        vectorstore = load_vectorstore()

        # ⭐ Chroma 原生删除方式（关键升级）
        vectorstore.delete(
            where={
                "source_file": file_name
            }
        )

        vectorstore.persist()

        return f"文件 {file_name} 的向量数据已成功删除。"

    except Exception as e:
        return (
            f"删除失败，可能原因:\n"
            f"1. 文件不存在\n2. 向量库损坏\n"
            f"错误信息: {str(e)}"
        )