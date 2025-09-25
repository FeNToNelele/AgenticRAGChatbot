# Agentic RAG Chatbot Prototype

This repository contains an **Agentic Retrieval-Augmented Generation (RAG)** chatbot prototype built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://www.langchain.com/), and open-source [HuggingFace models](https://huggingface.co/).  
The goal is to demonstrate **agentic behavior**, modular pipeline design, and potential scalability, rather than provide a production-ready system.

---

## Main Features

- Console and Web GUI for demo usage.
- Optional notebook view to see how the system works high-level.

---

## Architecture Overview

User → Controller Node → (optional) Retriever Node → QA Node → Answer

- The **Controller Node** implements agentic logic.
- If the query is short/simple → goes directly to QA Node.
- If complex → Retriever Node fetches context chunks before QA Node answers.

---

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/FeNToNelele/AgenticRAGChatbot.git
cd agentic-rag-chatbot
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure HuggingFace Access
Export your HuggingFace token:
```bash
export HF_PWC_CHATBOT_APIKEY=your_token_here
```

### 4. Run any UI you wish
```bash
python gui_console.py
streamlit run gui_web.py
```
or open the Notebook project.

### Note: First run of the project takes a while. This is due to downloading model, caching and loading vector store.
