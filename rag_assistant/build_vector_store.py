"""
build_vector_store.py
TORUS-2.0: Step 4 - Index knowledge base docs into a local FAISS vector store.

Place this file at: rag_assistant/build_vector_store.py
Run it from repo root:  python rag_assistant/build_vector_store.py

Uses:
  - sentence-transformers (all-MiniLM-L6-v2) for FREE local embeddings,
    no API key required.
  - FAISS for a local, file-based vector store (no server, no compilation
    issues on Windows).

Output: a folder ./faiss_index/ containing the serialized vector store.
Re-run this any time you add/edit docs in docs/knowledge_base/.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_BASE_DIR = "docs/knowledge_base"
INDEX_SAVE_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents():
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {KNOWLEDGE_BASE_DIR}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_and_save_index(chunks):
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Embedding chunks and building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(INDEX_SAVE_PATH)
    print(f"Vector store saved to ./{INDEX_SAVE_PATH}/")


def main():
    docs = load_documents()
    if not docs:
        raise RuntimeError(
            f"No .md files found in {KNOWLEDGE_BASE_DIR}. "
            "Make sure the knowledge base docs are saved there first."
        )
    chunks = split_documents(docs)
    build_and_save_index(chunks)
    print("\nDone. Next: run rag_assistant/ask_question.py to test retrieval.")


if __name__ == "__main__":
    main()
