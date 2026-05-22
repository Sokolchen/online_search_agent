from typing import Optional, Tuple
import hashlib
import json
import shutil
import time
from pathlib import Path

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

# ======================== 缓存配置 ========================

CACHE_ROOT = Path("cache") / "web_crawl"
CACHE_TTL = 3600  # 会话缓存有效期（秒），默认 1 小时


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _cache_dir(session_id: str, url: str) -> Path:
    return CACHE_ROOT / session_id / _url_hash(url)


def _cache_valid(cache_dir: Path) -> bool:
    """缓存存在且未过期"""
    meta_file = cache_dir / "meta.json"
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return (time.time() - meta["created_at"]) < CACHE_TTL
    except (json.JSONDecodeError, KeyError):
        return False


def _cleanup_expired(session_id: str) -> None:
    """清理当前会话中已过期的缓存目录"""
    session_dir = CACHE_ROOT / session_id
    if not session_dir.exists():
        return
    for item in session_dir.iterdir():
        if item.is_dir():
            meta_file = item / "meta.json"
            if not meta_file.exists():
                shutil.rmtree(item, ignore_errors=True)
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if time.time() - meta["created_at"] > CACHE_TTL:
                    shutil.rmtree(item, ignore_errors=True)
            except (json.JSONDecodeError, KeyError):
                shutil.rmtree(item, ignore_errors=True)


# ======================== 会话历史 ========================

store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ======================== 核心工具 ========================

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
        # 0. 清理过期缓存
        _cleanup_expired(session_id)

        cache_dir = _cache_dir(session_id, url)
        embeddings = OpenAIEmbeddings()

        # ========== 1. 获取或重建向量库 ==========
        if _cache_valid(cache_dir):
            # 命中缓存 → 直接加载已有向量库
            vectorstore = Chroma(
                persist_directory=str(cache_dir / "chroma_db"),
                embedding_function=embeddings,
            )
        else:
            # 未命中 → 爬取网页
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

            # 文本切割
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            split_docs = splitter.split_documents(docs)

            # 向量化 + 持久化到磁盘
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=str(cache_dir / "chroma_db"),
            )

            # 写入缓存元数据
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "meta.json").write_text(
                json.dumps(
                    {"url": url, "created_at": time.time()},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        # ========== 2. 构建 RAG 问答链路 ==========
        retriever = vectorstore.as_retriever()

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
                ("human", "{input}"),
            ]
        )

        chain_1 = create_stuff_documents_chain(deepseek_llm, prompt)

        contextualize_q_system_prompt = """你的任务是把用户的追问改写为一个独立、完整的问题。
        规则：
        1. 结合聊天历史，把指代词（如"那""他们""这个"）替换为具体内容
        2. 只输出一句改写后的问题，不超过30字
        3. 绝对不要回答问题，不要复述，不要总结"""
        retriever_history_temp = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_chain = create_history_aware_retriever(
            deepseek_llm, retriever, retriever_history_temp
        )

        main_chain = create_retrieval_chain(history_chain, chain_1)
        result_chain = RunnableWithMessageHistory(
            main_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # ========== 3. 执行问答 ==========
        config = RunnableConfig(
            configurable={"session_id": session_id}
        )

        resp = result_chain.invoke(
            {"input": question},
            config=config,
        )
        return resp["answer"]

    except Exception as e:
        return f"爬取或RAG处理失败：{e}"