# src/agent/rag/rag_qa_tool.py
from langchain.tools import tool
from agent.rag.rag_qa_chain import ask_pdf
from dotenv import load_dotenv
load_dotenv()
@tool("rag_qa_tool", parse_docstring=True)
def rag_qa_tool(question: str,source_file: str = None) -> str:
    """处理用户关于本地文档内容的问题，并从PDF知识库中检索答案。

    可回答来自向量库文件中的问题，例如：
    - 什么是 **某个名词或现象**？
    - **为了得出某个原理**需要做什么？
    - 某文档中的**定义或规则**是什么？

    Args:
        question (str):用户提出的问题
        source_file (str):用户需要查询的文件名称

    Returns:
        str:从PDF知识库中检索并生成的答案，如果查询失败则返回错误信息
    """

    print("\n[rag_qa_tool] Question received:")
    print(question)

    try:
        answer = ask_pdf(question, source_file=source_file)
        return answer
    except Exception as e:
        return f"PDF 查询失败: {str(e)}"