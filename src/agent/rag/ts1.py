# test_pdf_indexer.py

import os

from agent.rag.pdf_indexer import build_pdf_vectorstore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# ========= 配置区域 ========= #

PDF_PATH = r"E:\Re\online_search_agent\data\pdfs\CET_SET_RULES.pdf"

VECTOR_DB_PATH = r"E:\Re\online_search_agent\vectorstore\chroma_db"

COLLECTION_NAME = "pdf_collection"


# ========= 测试函数 ========= #

def test_pdf_ingestion():
    """
    测试 PDF 是否能够成功导入向量数据库。
    """

    print("\n========== PDF Indexer Test ==========")

    # ---------- Step1 检查PDF存在 ---------- #

    if not os.path.exists(PDF_PATH):
        print("❌ PDF 不存在:")
        print(PDF_PATH)
        return

    print("✅ PDF 存在")

    # ---------- Step2 调用 pdf_indexer ---------- #

    try:
        print("\n📥 正在导入 PDF...")

        build_pdf_vectorstore([PDF_PATH])

        print("✅ PDF 导入完成")

    except Exception as e:

        print("\n❌ PDF 导入失败:")
        print(str(e))
        return

    # ---------- Step3 验证 Chroma 写入 ---------- #

    try:

        print("\n📊 正在检查向量库内容...")

        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url="https://api.shubiaobiao.com/v1"
        )

        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding,
            collection_name=COLLECTION_NAME
        )

        results = vectorstore.get(include=["metadatas"])

        metadatas = results.get("metadatas", [])

        if not metadatas:

            print("❌ 向量库为空（未写入）")
            return

        print(f"✅ 成功写入 {len(metadatas)} 个 chunks")

        # ---------- Step4 显示文件 ---------- #

        files = set()

        for meta in metadatas:
            if "source_file" in meta:
                files.add(meta["source_file"])

        print("\n📄 当前库包含文件:")

        for f in files:
            print("-", f)

        print("\n🎉 PDF Indexer 测试成功！")

    except Exception as e:

        print("\n❌ 向量库检查失败:")
        print(str(e))


# ========= 主程序 ========= #

if __name__ == "__main__":
    test_pdf_ingestion()