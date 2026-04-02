# src/agent/rag/rag_manage_tools.py
import os
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"


def load_vectorstore() -> FAISS:
    """
    加载 FAISS 向量库。

    Returns:
        FAISS: 已加载的 FAISS 向量库实例

    Raises:
        FileNotFoundError: 如果 FAISS index 文件不存在
    """
    index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index 不存在，请先创建向量库")

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://api.shubiaobiao.com/v1"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )
    return vectorstore


@tool("rag_list_vectorstore", parse_docstring=True)
def rag_list_vectorstore(_: str = "") -> str:
    """
    查看当前向量库中存储的 PDF 文件及每个文件占用的 chunk 数量。

    Args:
        _ (str): 可选参数，暂未使用

    Returns:
        str: 当前向量库状态描述，包括每个文件的 chunk 数量和总 chunk 数
    """
    try:
        vectorstore = load_vectorstore()
        docs = vectorstore.docstore._dict  # 内部存储

        file_chunks = {}
        for k, doc in docs.items():
            metadata = doc.metadata
            file_name = metadata.get("source_file", "未知文件")
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
        return f"查看向量库失败，可能原因:\n1. 向量库未创建\n2. 文件损坏\n错误信息: {str(e)}"


@tool("rag_delete_pdf", parse_docstring=True)
def rag_delete_pdf(file_name: str) -> str:
    """
    删除指定 PDF 文件在向量库中的向量数据。

    Args:
        file_name (str): 需要删除的 PDF 文件名

    Returns:
        str: 删除结果描述，包括删除的 chunk 数量或错误信息
    """
    try:
        vectorstore = load_vectorstore()
        docs_dict = vectorstore.docstore._dict

        # 找出属于该文件的 doc ids
        ids_to_delete = [
            doc_id
            for doc_id, doc in docs_dict.items()
            if doc.metadata.get("source_file") == file_name
        ]

        if not ids_to_delete:
            return f"未找到名为 {file_name} 的文件在向量库中。"

        # 删除并保存
        vectorstore.delete(ids=ids_to_delete)
        vectorstore.save_local(VECTOR_DB_PATH)

        return (
            f"文件 {file_name} 的向量数据已成功删除，"
            f"删除 {len(ids_to_delete)} 个 chunk。"
        )

    except Exception as e:
        return (
            f"删除失败，可能原因:\n"
            f"1. 文件不存在\n2. 向量库损坏\n"
            f"错误信息: {str(e)}"
        )