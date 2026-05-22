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
# RAG_PDF Part1 接收传入路径并处理pdf
# ========= 配置 =========

#需要优化：此代码需要加入PDF分类函数，区分：守则规则类/论文类/普通文本类等输入文本，根据不同的类型文本做不同切割分类来得到更好数据
from agent.rag.vectorstore_utils import VECTOR_DB_PATH
load_dotenv()


# ========= 主函数 =========
def build_pdf_vectorstore(pdf_paths):
    """
    建立或追加 PDF 向量数据库。
    """

    if not pdf_paths:
        print("No PDF paths provided.")
        return None

    all_chunks = []

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    # ========= 从向量库获取已入库的 PDF 列表 =========
    # 等同于一次SQL查询
    existing_files: set[str] = set()
    if os.path.exists(VECTOR_DB_PATH):
        try:
            existing_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings,
                collection_name="pdf_collection"
            )
            # 只返回metadata字段，不返回向量
            results = existing_store.get(include=["metadatas"])
            for meta in results.get("metadatas", []):
                #遍历所有 metadata 记录，如果 "metadatas" 键不存在就用空列表兜底
                if meta and isinstance(meta, dict):
                    #防止空metadata或类型意外
                    #集合自动去重
                    name = meta.get("source_file")
                    if name:
                        existing_files.add(name)
            if existing_files:
                print(f"向量库中已存在 {len(existing_files)} 个 PDF 文件")
        except Exception as e:
            print(f"Warning: 无法读取已有向量库元数据: {e}")

    # ========= 处理每个 PDF =========
    for pdf_path in pdf_paths:

        pdf_path_obj = pathlib.Path(pdf_path).resolve()
        pdf_path_str = pdf_path_obj.as_posix()
        pdf_name = pdf_path_obj.name

        print(f"\nProcessing PDF: {pdf_name}")

        # ========= 1. 查向量库：是否已存在 =========
        if pdf_name in existing_files:
            print(f"\n⚠️ PDF already exists in vectorstore: {pdf_name}")
            print("⛔ Skipping...")
            continue

        # ========= 2. 查当前批次：是否重复传入 =========
        batch_dup = any(
            chunk.metadata.get("source_file") == pdf_name
            for chunk in all_chunks
        )
        if batch_dup:
            print(f"\n⚠️ Duplicate in this batch: {pdf_name}")
            print("⛔ Skipping...")
            continue

        # ========= 3. 加载 PDF =========
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

        # ========= 4. 语义切分 =========
        chunks = splitter.split_documents(documents)
        print(f"Chunks created: {len(chunks)}")

        # ========= 5. 过滤低质量 chunk =========
        filtered_chunks = []

        for i, chunk in enumerate(chunks):

            text = chunk.page_content.strip()

            if len(text) < 50:
                continue

            if re.match(r'^\s*-\s*\[\d+\]', text):
                continue

            chunk.metadata = dict(chunk.metadata)
            chunk.metadata["source_file"] = pdf_name
            chunk.metadata["chunk_index"] = i

            filtered_chunks.append(chunk)

        all_chunks.extend(filtered_chunks)
        existing_files.add(pdf_name)   # ← 标记为已处理，避免批次内重复检查时漏掉

    if not all_chunks:
        print("No new chunks to add.")
        return None

    for chunk in all_chunks:
        if "source_file" not in chunk.metadata:
            chunk.metadata["source_file"] = "unknown"

    # ========= Chroma 加载或创建 =========
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

