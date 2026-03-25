from agent.rag.pdf_indexer import build_pdf_vectorstore


if __name__ == "__main__":

    build_pdf_vectorstore(
        ["E:/Re/online_search_agent/data/pdfs/集成电路设计与布局实验指导书.pdf"]
    )