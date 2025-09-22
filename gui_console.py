from core.tools import history_aware_retriever, question_answer_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    RunnablePassthrough.assign(context=history_aware_retriever)
    | question_answer_chain
    | StrOutputParser()
)

def chat():
    print("Session started. You can leave chat by typing exit/quit/bye anytime.")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        chat_history = []
        
        response = chain.invoke({"input": question, "chat_history": chat_history})
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()