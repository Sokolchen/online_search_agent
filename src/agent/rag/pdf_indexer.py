# src/agent/rag/pdf_indexer.py

import os
import pathlib
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

#使用语义切分（新增）
from langchain_experimental.text_splitter import SemanticChunker

from langchain_chroma import Chroma
import re
# RAG_PDF Part1
# ========= 配置 =========

#
VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/chroma_db"

load_dotenv()


# ========= 主函数 =========
#采用了SemanticChunker 语义相似度拆分，只在语义差异最大的 5% 位置切分，可能导致chunk过多
#这个函数需要重写以优化在拥有了文本类型选择函数后，根据返回值确定切分策略，此函数保留为一种通用切割方法
def build_pdf_vectorstore(pdf_paths):
    """
    建立或追加 PDF 向量数据库。
    """

    if not pdf_paths:
        print("No PDF paths provided.")
        return None

    all_chunks = []

    # 固定embedding模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # =========语义切分器 =========
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    # ========= 处理每个 PDF =========
    for pdf_path in pdf_paths:

        pdf_path_obj = pathlib.Path(pdf_path).resolve()
        pdf_path_str = pdf_path_obj.as_posix()
        pdf_name = pdf_path_obj.name

        print(f"\n正在处理 PDF: {pdf_name}")

        already_exists = False

        for chunk in all_chunks:
            if chunk.metadata.get("source_file") == pdf_name:
                already_exists = True
                break

        if already_exists:
            print(f"\n⚠️ PDF已被处理: {pdf_name}")
            print("⛔ 正在跳过并停止此PDF的处理")
            continue

        loader = OpenDataLoaderPDFLoader(
            file_path=pdf_path_str,
            format="markdown",
            quiet=True
        )

        try:
            documents = loader.load()
        except Exception as e:
            print(f"Error loading {pdf_name}: {e}")
            continue

        if not documents:
            print(f"Warning: {pdf_name} returned 0 pages.")
            continue

        print(f"Loaded pages: {len(documents)}")

        # =========语义切分=========
        chunks = splitter.split_documents(documents)

        print("\n===== CHUNKS DEBUG =====")

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i} ---")
            print("metadata:", chunk.metadata)

        print(f"Chunks created: {len(chunks)}")

        # =========过滤低质量 chunk=========
        filtered_chunks = []

        for i, chunk in enumerate(chunks):

            text = chunk.page_content.strip()

            # 过滤空/过短 chunk
            if len(text) < 50:
                continue

            # 过滤 citation/reference chunk
            if re.match(r'^\s*-\s*\[\d+\]', text):
                continue

            # ========= ⭐ FIX：确保 metadata 是独立副本 =========
            chunk.metadata = dict(chunk.metadata)

            # ========= ⭐ FIX：统一写入 source_file =========
            chunk.metadata["source_file"] = pdf_name
            chunk.metadata["chunk_index"] = i

            filtered_chunks.append(chunk)

        all_chunks.extend(filtered_chunks)

    if not all_chunks:
        print("No chunks generated.")
        return None

    for chunk in all_chunks:
        if "source_file" not in chunk.metadata:
            chunk.metadata["source_file"] = "unknown"
    # ========= ⭐ Chroma 加载或创建 =========

    if os.path.exists(VECTOR_DB_PATH):

        print("\nLoading existing Chroma vectorstore...")

        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings,
            collection_name="pdf_collection"
        )

        print("Appending new documents...")

        vectorstore.add_documents(all_chunks)

    else:

        print("\nCreating new vectorstore...")

        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH,
            collection_name="pdf_collection"
        )

    print(
        f"\nChroma vectorstore saved successfully."
        f"\nTotal new chunks added: {len(all_chunks)}"
    )

    return vectorstore

