# rebuild_vectorstore.py

from src.agent.rag.pdf_indexer import build_pdf_vectorstore

# ========= PDF路径 =========
PDF_PATHS = [
    r"E:/Re/online_search_agent/data/pdfs/HOW MANY PAGES.pdf"
]


def main():

    print("\n==============================")
    print("🚀 Rebuilding Chroma Vectorstore")
    print("==============================\n")

    # ========= 强制重建 =========
    vectorstore = build_pdf_vectorstore(PDF_PATHS)

    if vectorstore is None:
        print("❌ build failed")
        return

    print("\n==============================")
    print("✅ Vectorstore rebuild complete")
    print("==============================\n")

    # ========= 简单验证 =========
    query = "how many pages"

    docs = vectorstore.similarity_search(query, k=3)

    print("\n🔍 Test retrieval:\n")

    for i, doc in enumerate(docs):
        print(f"--- Result {i+1} ---")
        print(doc.page_content[:150])
        print(doc.metadata)
        print("\n------------------\n")


if __name__ == "__main__":
    main()