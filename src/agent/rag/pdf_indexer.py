# src/agent/rag/pdf_indexer.py

import os
import pathlib
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ========= 配置 =========

VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"

load_dotenv()


# ========= 主函数 =========

def build_pdf_vectorstore(pdf_paths):
    """
    建立或追加 PDF 向量数据库。

    参数:
        pdf_paths: list[str]
            PDF 文件路径列表

    metadata:
        source_file: 文件名
        chunk_index: chunk 编号
    """

    if not pdf_paths:
        print("No PDF paths provided.")
        return None

    all_chunks = []

    embeddings = OpenAIEmbeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    # ========= 处理每个 PDF =========

    for pdf_path in pdf_paths:

        pdf_path_obj = pathlib.Path(pdf_path).resolve()
        pdf_path_str = pdf_path_obj.as_posix()

        pdf_name = pdf_path_obj.name

        print(f"\nProcessing PDF: {pdf_name}")

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

        # ========= 切分 =========

        chunks = splitter.split_documents(documents)

        print(f"Chunks created: {len(chunks)}")

        # ========= 添加 metadata =========

        for i, chunk in enumerate(chunks):

            chunk.metadata["source_file"] = pdf_name
            chunk.metadata["chunk_index"] = i

        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks generated.")
        return None

    # ========= 创建或追加向量库 =========

    if os.path.exists(VECTOR_DB_PATH):

        print("\nLoading existing vectorstore...")

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("Appending new documents...")

        vectorstore.add_documents(all_chunks)

    else:

        print("\nCreating new vectorstore...")

        vectorstore = FAISS.from_documents(
            all_chunks,
            embeddings
        )

    # ========= 保存 =========

    vectorstore.save_local(VECTOR_DB_PATH)

    print(
        f"\nVectorstore saved successfully."
        f"\nTotal new chunks added: {len(all_chunks)}"
    )

    return vectorstore