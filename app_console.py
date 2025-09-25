from core.embeddings import fetch_embedding, build_chroma_db
from core.model import load_llm
from core.graph import build_graph

def build_rag_chain():
    """Initialize prerequisits for RAG Chain."""

    cached_embedding = fetch_embedding()
    db = build_chroma_db(cached_embedding)
    retriever = db.as_retriever()
    llm = load_llm()
    return build_graph(retriever, llm)

def rag_chain_executor(rag_chain, question: str):
    """Run one question through the graph."""
    
    state = {"question": question, "context": [], "answer": ""}
    final_state = rag_chain.invoke(state)
    return final_state["answer"]

def chat():
    """Simple console chat loop."""
    print("Session started. Type exit/quit/bye to leave.")

    rag_chain = build_rag_chain()

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit", "bye"]:
            print("Bye!")
            break
        try:
            response = rag_chain_executor(rag_chain, question)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    chat()
