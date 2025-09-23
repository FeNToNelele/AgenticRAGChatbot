from core.tools import rag_chain_executor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def chat():
    print("Session started. Type exit/quit/bye to leave.")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit", "bye"]:
            print("Bye!")
            break
        try:
            response = rag_chain_executor(question)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    chat()