from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
VECTOR_DB_PATH = "E:/Re/online_search_agent/vectorstore/faiss_index"

def show_sources():

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 读取 metadata
    docs = vectorstore.docstore._dict

    files = set()

    for k in docs:

        metadata = docs[k].metadata

        if "source_file" in metadata:
            files.add(metadata["source_file"])

    print("\n已加载的 PDF 文件：")

    for f in files:
        print("-", f)

    print(f"\n共 {len(files)} 个 PDF")

if __name__ == "__main__":
    show_sources()