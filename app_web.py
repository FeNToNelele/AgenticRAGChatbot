import streamlit as st
from core.embeddings import fetch_embedding, build_chroma_db
from core.model import load_llm
from core.graph import build_graph

@st.cache_resource
def build_rag_chain():
    """Initialize everything once and cache across Streamlit sessions."""
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



st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.header("Chatbot")
rag_chain = build_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Write your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain_executor(rag_chain, prompt)
            except Exception as e:
                response = f"[Error] {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
