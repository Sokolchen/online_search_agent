from langchain_core.tools import tool

from agent.my_llm import zhipu_ai_client

@tool("web_search",parse_docstring=True)
def web_search(query:str) -> str:
    """联网搜索工具，搜索所有网络公开信息

    Args:
        query:所有需要联网搜索的信息

    Returns:
        返回搜索的结果，类型为文本字符串
    """
    try:
        resp=zhipu_ai_client.web_search.web_search(
            search_engine="search_std",
            search_query=query,
        )
        if resp.search_result:
            return "\n\n".join([searched_messages.content for searched_messages in resp.search_result])
        return"无相关搜索内容"
    except Exception as e:
        return f"Error: {e}"