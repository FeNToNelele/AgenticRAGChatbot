import argparse, os
from huggingface_hub import login
from core.embeddings import fetch_embedding, build_database
from core.model import load_llm
from core.graph import build_graph

def initialize():
    print("Initializing...")
    HF_API_KEY = os.environ.get("HF_PWC_CHATBOT_APIKEY")
    if HF_API_KEY:
        login(token=HF_API_KEY)
    else:
        raise Exception("Please set HF_PWC_CHATBOT_APIKEY environment variable.")

def rag_chain_executor(rag_chain, question: str):
    state = {"question": question, "context": [], "answer": ""}
    final_state = rag_chain.invoke(state)
    return final_state["answer"]

def main():
    initialize()
    cached_embedding = fetch_embedding()
    db = build_database(cached_embedding)
    retriever = db.as_retriever()
    llm = load_llm()
    rag_chain = build_graph(retriever, llm)

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", help="Ask a single question")
    args = parser.parse_args()

    if args.question:
        print(rag_chain_executor(rag_chain, args.question))
    else:
        while True:
            q = input("Ask me something (or 'exit'): ")
            if q.lower() == "exit":
                break
            print(rag_chain_executor(rag_chain, q))

if __name__ == "__main__":
    main()
