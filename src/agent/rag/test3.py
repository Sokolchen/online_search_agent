# src/agent/rag/test_min_rag_qa.py

"""
最小 RAG QA 测试文件

作用：
验证 rag_qa_chain 是否能正常：
1. 加载 FAISS
2. 创建 RetrievalQA
3. 调用 LLM
4. 返回答案
"""

from agent.rag.rag_qa_chain import ask_pdf


def main():

    # ========= 测试问题 =========

    question = "介绍一下EM-7028"

    print("\n========== RAG QA TEST ==========\n")

    print("Question:")
    print(question)


    # ========= 调用 RAG =========

    answer = ask_pdf(question)


    # ========= 输出答案 =========

    print("\nAnswer:\n")

    print(answer)

    print("\n========== TEST END ==========\n")


if __name__ == "__main__":
    main()