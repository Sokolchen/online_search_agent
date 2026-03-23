from typing import Optional, Tuple
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool

#基于bs4与WebBaseLoader实现的简单推文网站爬取Tool
@tool("web_crawl",parse_docstring=True)
def web_crawl(
        url:str,
        css_classes:Optional[Tuple[str,...]]=None
)->str:
    """从用户提供的指定网站爬取内容，支持用户按照自定义CSS类名的过滤

    Args:
        url:要爬取的网页的url
        css_classes:用户主动提供的优先提取的css类名元组，用于过滤页面元素，
        若用户未主动提供，则默认爬取整个网页的内容

    """
    try:
        bs_kwargs = {}
        if css_classes:
            if isinstance(css_classes, str):#检查是否为字符串（用户输入大概率为字符串）
                css_classes = (css_classes,)#转换为元组
            bs_kwargs["parse_only"] = bs4.SoupStrainer(class_=css_classes)
        #转换完成，把参数赋给WebBaseLoader，在有提供类名情况下，工具只解析包含指定 CSS 类名的标签
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs=bs_kwargs,
        )
        docs = loader.load()
        if not docs:
            return "未从对应网址爬取到任何内容"
        return "\n\n".join(doc.page_content for doc in docs)#实现以两个空行分割的完整字符串输出
    except Exception as e:
        return f"爬取失败：{e}"

