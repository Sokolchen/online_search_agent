# src/agent/rag/pdf_indexer.py
import os
import pathlib
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

#RAG_PDF Part1
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

    all_chunks = []#预定义最终存储用向量库的列表

    embeddings = OpenAIEmbeddings()#设定openai-embeddings模型。
    #文本切割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    # ========= 处理每个 PDF =========

    for pdf_path in pdf_paths:

        pdf_path_obj = pathlib.Path(pdf_path).resolve() #将路径绝对化并返回Pure path
        pdf_path_str = pdf_path_obj.as_posix()#返回使用正斜杠（/）的路径字符串:

        pdf_name = pdf_path_obj.name#对Pure path使用.name返回文件名

        print(f"\nProcessing PDF: {pdf_name}")
        #默认按照页切割
        loader = OpenDataLoaderPDFLoader(
            file_path=pdf_path_str,
            format="markdown",
            quiet=True
        )

        try:
            documents = loader.load()#转化为document的一个list

        except Exception as e:
            print(f"Error loading {pdf_name}: {e}")
            continue

        if not documents:#document转换失败时报错
            print(f"Warning: {pdf_name} returned 0 pages.")
            continue

        print(f"Loaded pages: {len(documents)}")

        # ========= 切分 =========

        chunks = splitter.split_documents(documents)#切分文件为数个chunks

        print(f"Chunks created: {len(chunks)}")

        # ========= 添加 metadata =========
        #追踪文件向量去向
        for i, chunk in enumerate(chunks):#同时获取可迭代对象（如列表、元组等）中每个元素的索引和值‌的内置函数

            chunk.metadata["source_file"] = pdf_name
            chunk.metadata["chunk_index"] = i

        all_chunks.extend(chunks) #带有元数据的chunk存入开头定义的all_chunks

    if not all_chunks:#报错提示
        print("No chunks generated.")
        return None

    # ========= 创建或追加向量库 =========

    if os.path.exists(VECTOR_DB_PATH):#检查向量库路径
#重要：首次执行时需保证vectorstore下无其他文件
        print("\nLoading existing vectorstore...")
#FAISS.load_local(向量路径, embeddings模型, 显式设置开)
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("Appending new documents...")

        vectorstore.add_documents(all_chunks)#向已有的向量库中添加新文档

    else:
#无目录时直接依照现在的chunks创建新目录
        print("\nCreating new vectorstore...")

        vectorstore = FAISS.from_documents(
            all_chunks,
            embeddings
        )

    # ========= 保存 =========

    vectorstore.save_local(VECTOR_DB_PATH)#保存于本地

    print(
        f"\nVectorstore saved successfully."
        f"\nTotal new chunks added: {len(all_chunks)}"
    )

    return vectorstore