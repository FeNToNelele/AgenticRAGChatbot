import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import torch

VECTOR_DB_DIR = "chroma_db"
EMBEDDING_CACHED_DIR = "embedding_cache"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def fetch_embedding():
    """Return cache-backed embedding function."""
    os.makedirs(EMBEDDING_CACHED_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": get_device()},
    )
    store = LocalFileStore(EMBEDDING_CACHED_DIR)

    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=store,
        namespace=embeddings.model_name,
    )


def build_database(cached_embedding):
    """Create or load a Chroma vector store."""
    if not os.path.exists(VECTOR_DB_DIR):
        print("Vector store not found. Creating a new one...")

        streaming_dataset = load_dataset(
            "legacy-datasets/wikipedia",
            "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        subset_dataset = streaming_dataset.take(1000)

        docs = [
            Document(page_content=item["text"], metadata={"title": item["title"]})
            for item in subset_dataset
        ]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        db = Chroma.from_documents(chunks, cached_embedding, persist_directory=VECTOR_DB_DIR)
        print(f"Vector store created and saved to '{VECTOR_DB_DIR}'.")
    else:
        print(f"Loading existing vector store from '{VECTOR_DB_DIR}'...")
        db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=cached_embedding)

    return db
