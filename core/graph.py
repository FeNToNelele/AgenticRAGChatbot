from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retriever_node_factory(retriever):
    def node(state: RAGState) -> RAGState:
        print("[Retriever] Searching for documents...")
        docs = retriever.invoke(state["question"])
        return {**state, "context": docs}
    return node

def chatbot_node_factory(llm):
    def node(state: RAGState) -> RAGState:
        print("[Chatbot Node] Generating answer...")

        prompt_text = """Use the following context to answer the user's question in a friendly, 
            helpful, short and concise manner. Do not leave "Note: etc." texts after you answered."""

        context_text = "\n\n".join(doc.page_content for doc in state.get("context", []))
        if context_text:
            prompt_text = f"""{prompt_text}
            Context: {context_text}
            Question: {state["question"]}
            Answer:"""
        else:
            prompt_text = f"""Answer the question in a helpful, short, concise manner. Use The knowledge you have.
            Question: {state["question"]}
            Answer:"""

        response = llm.invoke(prompt_text).strip()

        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        for prefix in ["System:", "Human:", "AI:", "Assistant:", "Bot:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        return {**state, "answer": response}
    return node

def controller_node(state: RAGState) -> str:
    question = state["question"]
    if len(question.split()) < 5:
        print("[Controller] No need for retrieval.")
        return {"next": "chatbot"}
    else:
        print("[Controller] Use retrieval.")
        return {"next": "retrieve"}

def build_graph(retriever, llm):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retriever_node_factory(retriever))
    graph.add_node("chatbot", chatbot_node_factory(llm))
    graph.add_node("controller", controller_node)

    graph.set_entry_point("controller")
    graph.add_conditional_edges(
        "controller",
        lambda state: state["next"],
        {
            "retrieve": "retrieve",
            "chatbot": "chatbot",
        },
    )
    graph.add_edge("retrieve", "chatbot")
    graph.add_edge("chatbot", END)

    return graph.compile()
