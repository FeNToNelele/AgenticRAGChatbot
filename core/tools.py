from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import torch, os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

from huggingface_hub import login

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
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

HF_API_KEY = os.environ.get('HF_PWC_CHATBOT_APIKEY')
if HF_API_KEY:
    login(token=HF_API_KEY)
else:
    raise Exception("Please create API key for HuggingFace to download LLaMa for the first time.")

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

selected_model = "meta-llama/Llama-3.2-3B-Instruct"

print(f"Loading model")
tokenizer = AutoTokenizer.from_pretrained(selected_model)
model = AutoModelForCausalLM.from_pretrained(
    selected_model,
    device_map="auto",
    load_in_4bit=True, # To save VRAM, works with CUDA.
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    return_full_text=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = HuggingFacePipeline(pipeline=pipe)

def retrieve_node(state: RAGState) -> RAGState:
    print("[Retriever] Searching for documents...")
    docs = retriever.invoke(state["question"])
    return {**state, "context": docs}

def chatbot_node(state: RAGState) -> RAGState:
    print("[Chatbot Node] Generating answer...")

    context_text = "\n\n".join([doc.page_content for doc in state.get("context", [])])

    if context_text:
        prompt_text = f"""You are a helpful AI assistant. Use the following context to answer the user's question in a friendly, 
        helpful, short and concise manner. Do not leave "Note: etc." texts after you answered.
        Context: {context_text}
        Question: {state["question"]}
        Answer:"""
    else:
        prompt_text = f"""You are a helpful AI assistant. Answer the user's question in a friendly, 
        helpful, short and concise manner. Do not leave "Note: etc." texts after you answered.
        Question: {state["question"]}
        Answer:"""
 
    response = llm.invoke(prompt_text)

    if prompt_text in response:
        response = response.replace(prompt_text, "").strip()

    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip() 

    prefixes_to_remove = ["System:", "Human:", "AI:", "Assistant:", "Bot:"]
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
        
    return {**state, "answer": response}

# Only for demonstration purposes
# If question has 5 words at least, use retrieval. Lot more sophisticated logic in production.
def controller_node(state: RAGState) -> str:
    question = state["question"]
    if len(question.split()) < 5:
        print("[Controller] No need for retrieval.")
        return {"next" : "chatbot"}
    else:
        print("[Controller] Use retrieval.")
        return {"next" : "retrieve"}

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("chatbot", chatbot_node)
graph.add_node("controller", controller_node)

graph.set_entry_point("controller")
graph.add_conditional_edges(
    "controller",
    lambda state: state["next"],
    {
        "retrieve": "retrieve",
        "chatbot": "chatbot",
    }
)

graph.add_edge("retrieve", "chatbot")
graph.add_edge("chatbot", END)

rag_chain = graph.compile()

def rag_chain_executor(question: str):
    state = {"question": question, "context": [], "answer": ""}
    final_state = rag_chain.invoke(state)
    return final_state["answer"]