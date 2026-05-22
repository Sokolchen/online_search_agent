# src/agent/rag/rag_manage_tools.py
#包含了“查看本地向量存储情况”与“基于文件名metadata的向量删除”相关TOOL
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from agent.rag.vectorstore_utils import load_vectorstore



#Langchain默认tool需要一个接收参数，此处留空
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

        #Chroma式查询：get all data 遍历metadata获取file_names
        results = vectorstore.get(include=["metadatas"])

        metadatas = results.get("metadatas", [])#metadata不存在则返回[]

        file_chunks = {}

        for meta in metadatas:#一个meta对应一个chunk，准备获取每个pdf的chunk数量
            if not meta or not isinstance(meta, dict):
                continue
            #若metadata为空或格式异常，则跳过

            file_name = meta.get("source_file")
            if not file_name:
                file_name = "未知文件"
                #若metadata中没有source_file则归类为“未知文件”
            else:
                file_name = str(file_name).strip()
                #转为字符串并去除空格，避免同名不同格式问题

            file_chunks[file_name] = file_chunks.get(file_name, 0) + 1
            #统计逻辑：若文件已存在+1, 若不存在从0开始计数,以此查看本地有多少chunks，最后整体+1
        if not file_chunks:
            return "当前向量库为空，没有存储任何 PDF 文件。"

        result_lines = ["当前向量库状态："]
        total_chunks = 0

        #按chunk数降序，同chunk则按字母排序，把打印的信息加入total_chunks
        for f, c in sorted(file_chunks.items(), key=lambda x: (-x[1], x[0])):
            result_lines.append(f"- {f}: {c} chunks")
            total_chunks += c
            """
            假如文件是("A.pdf", 5)则转换→(-5, "A.pdf")，
            负数是为了实现“chunk大的排前面，sorted默认从小到大，这样可以把绝对值最大值置顶”
            """
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

        # 删除前先统计 chunk 数
        # 更新：直接给出删除处理后的向量库情况
        results = vectorstore.get(
            where={"source_file": file_name},
            include=["metadatas"]
        )
        deleted_count = len(results.get("ids", []))

        if deleted_count == 0:
            return (
                f"未找到文件 {file_name} 的向量数据，可能已被删除或文件名不匹配。\n"
                f"提示：请先使用本地向量库列表功能确认文件是否已入库。"
            )

        # 执行删除
        vectorstore.delete(where={"source_file": file_name})

        # 统计删除后向量库状态
        all_results = vectorstore.get(include=["metadatas"])
        remaining_files: set[str] = set()
        for meta in all_results.get("metadatas", []):
            if meta and isinstance(meta, dict):
                name = meta.get("source_file")
                if name:
                    remaining_files.add(name)

        remaining_text = (
            "\n".join(f"- {f}" for f in sorted(remaining_files))
            if remaining_files
            else "(空)"
        )

        return (
            f"✅ 已删除文件 {file_name}\n"
            f"共移除 {deleted_count} 个 chunks\n\n"
            f"当前向量库剩余文件：\n{remaining_text}"
        )

    except Exception as e:
        return (
            f"删除失败，可能原因:\n"
            f"1. 文件不存在\n2. 向量库损坏\n"
            f"错误信息: {str(e)}"
        )