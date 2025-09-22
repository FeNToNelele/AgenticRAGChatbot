import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage

from core.tools import history_aware_retriever, question_answer_chain

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.header("Chatbot")

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
            # Create input for retriever
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(("human", msg["content"]))
                else:
                    chat_history.append(("ai", msg["content"]))

            retrieved_docs = history_aware_retriever.invoke({
                "input": prompt,
                "chat_history": chat_history
            })

            # Prepare final prompt with context
            qa_input = {
                "input": prompt,
                "context": retrieved_docs,
                "chat_history": chat_history
            }

            # Invoke the question-answer chain to get the final response
            response = question_answer_chain.invoke(qa_input)
            # response = response_chain.content

        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})