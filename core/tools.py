from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import torch, os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

from langchain.chains.combine_documents import create_stuff_documents_chain


from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Tuple
from datasets import load_dataset

class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

VECTOR_DB_DIR = "chroma_db"
EMBEDDING_CACHED_DIR = "embedding_cache"

os.makedirs(EMBEDDING_CACHED_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

GROQ_API_KEY = os.environ.get('AgenticRAGChatbot_APIKEY')

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': device})
store = LocalFileStore(EMBEDDING_CACHED_DIR)

cached_embedding = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store,
    namespace=embeddings.model_name
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

selected_model = "meta-llama/Llama-3.1-8B-Instruct"

print(f"Loading model")
tokenizer = AutoTokenizer.from_pretrained(selected_model)
model = AutoModelForCausalLM.from_pretrained(
    selected_model,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
)

llm = HuggingFacePipeline(pipeline=pipe)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the provided context only.\n\n{context}"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

def retrieve_node(state: RAGState) -> RAGState:
    print("[Retriever] Searching for documents...")
    docs = retriever.invoke(state["question"])
    state["context"] = docs
    return state

def qa_node(state: RAGState) -> RAGState:
    print("[QA] Generating answer...")
    response = qa_chain.invoke({
    "input": state["question"],
    "context": state.get("context", [])
    })
    state["answer"] = response.content
    return state

# Only for demonstration purposes
# If question is shorter than 5 words, use QA, else retrieve info.
def controller_node(state: RAGState) -> str:
    if len(state["question"].split()) < 5:
        print("[Controller] No need for retrieval.")
        return "qa"
    else:
        print("[Controller] Use retrieval.")
        return "retrieve"

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("qa", qa_node)

graph.set_entry_point(controller_node)
graph.add_edge("retrieve", "qa")
graph.add_edge("qa", END)

rag_chain = graph.compile()

def rag_chain_executor(question: str):
    state = {"question": question, "context": [], "answer": ""}
    final_state = rag_chain.invoke(state)
    return final_state["answer"]