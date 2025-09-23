import streamlit as st
from core.tools import rag_chain_executor

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.header("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
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
                response = rag_chain_executor(prompt)
            except Exception as e:
                response = f"[Error] {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})