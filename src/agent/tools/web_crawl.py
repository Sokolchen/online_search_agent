from typing import Optional, Tuple
import bs4
from agent.my_llm import deepseek_llm
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableConfig

# 会话历史存储
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@tool("web_crawl", parse_docstring=True)
def web_crawl(
        url: str,
        question: str,
        css_classes: Optional[Tuple[str, ...]] = None,
        session_id: str = "default"
) -> str:
    """从指定网站爬取内容并进行RAG问答，可被Agent调用

    Args:
        url: 要爬取的网页URL
        css_classes: 可选，用户提供的CSS类名元组，用于过滤页面元素
        question: 用户提出的问题
        session_id: 会话ID，用于保存多轮问答历史

    Returns:
        返回基于RAG检索后的中文答案
    """
    try:
        # 1. 爬取网页
        bs_kwargs = {}
        if css_classes:
            if isinstance(css_classes, str):
                css_classes = (css_classes,)
            bs_kwargs["parse_only"] = bs4.SoupStrainer(class_=css_classes)
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs=bs_kwargs,
        )
        docs = loader.load()
        if not docs:
            return "未从对应网址爬取到任何内容"

        # 2. 文本切割
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        # 3. 向量化
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()

        # 4. 系统提示
        system_prompt = """
        你是一个能处理问答任务的智能助手，用户会使用中文来提问，根据用户提示的网址与要求来爬取网站
        你必须遵守以下流程：如果目标网站爬取到的为非中文内容，先把用户输入的中文转为与目标网站相同的语言进行理解，再查询文档，最后把文档总结出的内容重新翻译为中文回答。
        假如用户只让你爬取网站内容而不提出任何问题，直接爬取并记忆内容，等待用户进一步的命令。
        如果不知道答案，就直接说不知道。使用不多于150字的中文回答，在用户提出要减少或者提升总结字数时忽略前一句话的字数限制 。\n

        {context}
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # 5. chain1
        chain_1 = create_stuff_documents_chain(deepseek_llm, prompt)

        # 6. 历史上下文处理
        contextualize_q_system_prompt = """给定聊天历史和最新用户问题（可能引用历史上下文），
请整理成一个独立可理解的问题，不要回答，直接改写问题即可。"""
        retriever_history_temp = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_chain = create_history_aware_retriever(deepseek_llm, retriever, retriever_history_temp)

        # 7. 父chain组合
        main_chain = create_retrieval_chain(history_chain, chain_1)
        result_chain = RunnableWithMessageHistory(
            main_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # 8. 调用RAG返回答案
        config = RunnableConfig(
            configurable={
                "session_id": session_id
            }
        )

        resp = result_chain.invoke(
            {
                "input": question
            },
            config=config
        )
        return resp["answer"]

    except Exception as e:
        return f"爬取或RAG处理失败：{e}"

