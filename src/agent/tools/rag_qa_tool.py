# src/agent/rag/rag_qa_tool.py
from langchain.tools import tool
from agent.rag.rag_qa_chain import ask_pdf

@tool("rag_qa_tool", parse_docstring=True)
def rag_qa_tool(question: str) -> str:
    """
    当用户的问题涉及本地PDF文档内容时使用该工具。
    可回答来自PDF文件中的问题，例如：
    - 什么是 **某个名词或现象**？
    - **为了得出某个原理**需要做什么？
    - 某文档中的**定义或规则**是什么？

    输入:
        question: 用户问题

    输出:
        从PDF知识库中检索并生成的答案
    """

    print("\n[rag_qa_tool] Question received:")
    print(question)

    try:

        answer = ask_pdf(question)

        return answer

    except Exception as e:

        return f"PDF 查询失败: {str(e)}"