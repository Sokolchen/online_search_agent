# src/agent/rag/rag_manage_tools.py
import os
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"

#管理本地PDF向量库，包括删除与查询本地存储情况
#
def load_vectorstore():
    """
    加载 FAISS 向量库，返回 vectorstore
    """
    if not os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
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
    查看当前向量库中存储了哪些文件以及每个文件占用多少 chunks
    """
    try:
        vectorstore = load_vectorstore()
        # key -> Document 对象
        docs = vectorstore.docstore._dict  # 访问内部存储
        file_chunks = {}
        for k in docs:#遍历文档从metadata中提取indexer中定义的"source_file"来获取文件名
            metadata = docs[k].metadata
            file_name = metadata.get("source_file", "未知文件")
            file_chunks[file_name] = file_chunks.get(file_name, 0) + 1
            #利用字典 file_chunks 累加每个文件对应的 chunk 数量。
            #get方法在检索遇到某个文件名时返回0，后续遇到时返回已有计数值，+1代表当前检索到的chunk属于该文件
            #将计算后的新值赋给字典中键为 file_name 的条目。如果该键原本不存在，则创建并赋值；如果已存在，则更新
        if not file_chunks:
            return "当前向量库为空，没有存储任何 PDF 文件。"
        #打印状态
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
    删除指定 PDF 的向量数据
    """
    try:
        vectorstore = load_vectorstore()

        docs_dict = vectorstore.docstore._dict

        # 找出属于该文件的 doc ids
        #利用列表推导式：创建一个新列表，里面放的是：在docs中满足条件-名字与要删除的文件名相同的元素。
        ids_to_delete = [
            doc_id
            for doc_id, doc in docs_dict.items()
            if doc.metadata.get("source_file") == file_name
        ]

        if not ids_to_delete:
            return f"未找到名为 {file_name} 的文件在向量库中。"

        # 正确删除（关键）
        vectorstore.delete(ids=ids_to_delete)

        # 保存
        vectorstore.save_local(VECTOR_DB_PATH)
        #本地保存，否则删除只在内存生效，重启后恢复

        return (
            f"文件 {file_name} 的向量数据已成功删除，"
            f"删除 {len(ids_to_delete)} 个 chunk。"
        )

    except Exception as e:
        return (
            f"删除失败，可能原因:\n"
            f"1. 文件不存在\n"
            f"2. 向量库损坏\n"
            f"错误信息: {str(e)}"
        )