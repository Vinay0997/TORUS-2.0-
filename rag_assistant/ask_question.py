"""
ask_question.py
TORUS-2.0: Step 4 - RAG retrieval + generation chain.

Place this file at: rag_assistant/ask_question.py
Run it from repo root:  python rag_assistant/ask_question.py

Works both locally (env var) and on Streamlit Community Cloud (st.secrets).

Requires a free Hugging Face fine-grained access token with the
"Make calls to Inference Providers" permission enabled:
  Local:  $env:HUGGINGFACEHUB_API_TOKEN = "your_token_here"
  Cloud:  add HUGGINGFACEHUB_API_TOKEN under app Settings -> Secrets

To switch to OpenAI GPT-4 later, replace the ChatHuggingFace block with:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_REPO_ID = "meta-llama/Llama-3.2-3B-Instruct"
TOP_K = 4


def get_hf_token():
    """Checks env var first (local dev), then Streamlit secrets (cloud deploy)."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        if token:
            return token
    except Exception:
        pass
    raise RuntimeError(
        "HUGGINGFACEHUB_API_TOKEN not found.\n"
        "Local: $env:HUGGINGFACEHUB_API_TOKEN = \"your_token_here\"\n"
        "Streamlit Cloud: add it under app Settings -> Secrets as:\n"
        '  HUGGINGFACEHUB_API_TOKEN = "your_token_here"'
    )


HF_TOKEN = get_hf_token()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN  # ensure it's set for the HF client

SYSTEM_PROMPT = """You are the TORUS 2.0 project assistant. You help \
technicians and engineers understand the TORUS pipeline, its data \
model, FHIR/HL7 concepts, device troubleshooting steps, and the ML \
models used in this project.

Rules you must always follow:
1. Answer ONLY using the provided context below. If the context does \
not contain the answer, say "I don't have enough information in the \
knowledge base to answer that" - do not guess or use outside knowledge.
2. NEVER provide a clinical diagnosis or medical advice about real \
patients. This system uses only synthetic, non-PHI data for portfolio \
and educational purposes.
3. Focus on explaining workflows, data fields, architecture, and \
troubleshooting steps - not clinical interpretation.
4. Be concise and specific. Reference actual file/table/column names \
from the context when relevant.

Context:
{context}
"""


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def build_chain():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm_endpoint = HuggingFaceEndpoint(
        repo_id=HF_MODEL_REPO_ID,
        provider="auto",
        temperature=0.1,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask_question(question: str, chain=None, retriever=None, show_sources: bool = True):
    if chain is None or retriever is None:
        chain, retriever = build_chain()

    answer = chain.invoke(question)

    if show_sources:
        retrieved_docs = retriever.invoke(question)
        sources = sorted(set(d.metadata.get("source", "unknown") for d in retrieved_docs))
        print("\n--- Retrieved from ---")
        for s in sources:
            print(f"  {s}")

    return answer


def main():
    print(f"Loading vector store and connecting to Hugging Face ({HF_MODEL_REPO_ID})...")
    chain, retriever = build_chain()

    test_questions = [
        "How is device telemetry ingested into TORUS?",
        "Explain the FHIR ImagingStudy resource used in this system.",
        "What escalation threshold should a technician use for anomaly rate?",
        "Which two models are compared for follow-up prediction and how is the winner chosen?",
    ]

    for q in test_questions:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        answer = ask_question(q, chain=chain, retriever=retriever)
        print(f"\nA: {answer}")


if __name__ == "__main__":
    main()
