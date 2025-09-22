from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import torch, os

from datasets import load_dataset

VECTOR_DB_DIR = "chroma_db"
EMBEDDING_CACHED_DIR = "embedding_cache"

os.makedirs(EMBEDDING_CACHED_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

GROQ_API_KEY = os.environ.get('AgenticRAGChatbot_APIKEY')

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

underlying_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': device})
store = LocalFileStore(EMBEDDING_CACHED_DIR)

cached_embedding = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings,
    document_embedding_cache=store,
    namespace=underlying_embeddings.model_name
)

if not os.path.exists(VECTOR_DB_DIR):
    print("Vector store not found. Creating a new one...")
    
    print("Loading Wikipedia dataset...")
    streaming_dataset = load_dataset("legacy-datasets/wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    
    # Keep demo hardware-friendly, using only subset
    subset_dataset = streaming_dataset.take(1000)

    docs = []
    for item in subset_dataset:
        docs.append(Document(page_content=item['text'], metadata={'title': item['title']}))

    print(f"Loaded {len(docs)} documents from Wikipedia.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    db = Chroma.from_documents(chunks, cached_embedding, persist_directory=VECTOR_DB_DIR)
    print(f"Vector store created and saved to '{VECTOR_DB_DIR}'.")
else:
    print(f"Loading existing vector store from '{VECTOR_DB_DIR}'...")
    db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=cached_embedding)
    print("Done.")


print("Getting retriever...")
retriever = db.as_retriever()
print("Done.")

contextualizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history."),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualizer_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the provided context only. If you don't know the answer, just say you don't know, don't try to make up an answer. \n\n{context}"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

def rag_chain_executor(state):
    """Executes the RAG chain to retrieve context and generate a response."""
    question = state["messages"][-1].content
    retrieved_docs = history_aware_retriever.invoke({"input": question, "chat_history": state["messages"][:-1]})
    response = question_answer_chain.invoke({"input": question, "context": retrieved_docs, "chat_history": state["messages"][:-1]})
    return {"messages": [("ai", response.content)]}