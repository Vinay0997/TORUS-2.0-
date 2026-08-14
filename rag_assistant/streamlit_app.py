"""
streamlit_app.py
TORUS-2.0: Step 4 - Chat UI for the RAG assistant.

Place this file at: rag_assistant/streamlit_app.py
Run it from repo root:  python -m streamlit run rag_assistant/streamlit_app.py

Requires:
  - faiss_index/ already built (run build_vector_store.py first)
  - HUGGINGFACEHUB_API_TOKEN environment variable set before launching
    Streamlit (Streamlit runs in a fresh process, so set the env var in
    the SAME terminal session before running the streamlit command)
"""

import os
import sys
import streamlit as st

# Force the token into the environment before any other imports/caching runs
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Temporary debug line - remove after confirming this works
st.write(f"Token loaded: {len(os.environ.get('HUGGINGFACEHUB_API_TOKEN', ''))} characters")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask_question import build_chain, HF_MODEL_REPO_ID

st.set_page_config(page_title="TORUS Assistant", page_icon="🩺", layout="centered")

st.title("🩺 TORUS 2.0 Assistant")
st.caption(
    f"RAG assistant · Model: `{HF_MODEL_REPO_ID}` via Hugging Face Inference API · "
    "Answers are grounded in the TORUS knowledge base only."
)

with st.expander("ℹ️ About this assistant", expanded=False):
    st.markdown(
        """
This assistant answers questions about the **TORUS 2.0** pipeline —
architecture, data ingestion, FHIR/HL7 concepts, device troubleshooting,
the follow-up prediction model, and MLflow experiment tracking.

**It will not provide medical diagnoses or advice about real patients.**
All data in this project is synthetic. If a question falls outside the
knowledge base, the assistant will say so rather than guessing.

Note: this runs against the Hugging Face free-tier Inference API, so the
first response after startup may be slower while the model "wakes up."
        """
    )

EXAMPLE_QUESTIONS = [
    "How is device telemetry ingested into TORUS?",
    "Explain the FHIR ImagingStudy resource used in this system.",
    "What escalation threshold should a technician use for anomaly rate?",
    "Which two models are compared for follow-up prediction and how is the winner chosen?",
]

st.markdown("**Try an example:**")
cols = st.columns(2)
selected_example = None
for i, q in enumerate(EXAMPLE_QUESTIONS):
    if cols[i % 2].button(q, use_container_width=True):
        selected_example = q

if "chain" not in st.session_state:
    with st.spinner("Loading vector store and connecting to Hugging Face..."):
        try:
            st.session_state.chain, st.session_state.retriever = build_chain()
            st.session_state.load_error = None
        except Exception as e:
            st.session_state.chain, st.session_state.retriever = None, None
            st.session_state.load_error = str(e)

if st.session_state.get("load_error"):
    st.error(
        "Failed to load the RAG chain. Common causes:\n\n"
        "1. `faiss_index/` doesn't exist yet - run "
        "`python rag_assistant/build_vector_store.py` first.\n"
        "2. `HUGGINGFACEHUB_API_TOKEN` isn't set in this terminal session - "
        "set it with `$env:HUGGINGFACEHUB_API_TOKEN = \"your_token_here\"` "
        "BEFORE running the streamlit command.\n"
        "3. Rate limit or cold-start timeout on the free Hugging Face tier - "
        "wait ~20 seconds and refresh the page.\n\n"
        f"Error detail: {st.session_state.load_error}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources retrieved"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

user_input = st.chat_input("Ask about TORUS architecture, FHIR/HL7, troubleshooting, or the models...")
query = selected_example or user_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer (may take a moment on free tier)..."):
            try:
                retrieved_docs = st.session_state.retriever.invoke(query)
                sources = sorted(set(
                    d.metadata.get("source", "unknown") for d in retrieved_docs
                ))
                answer = st.session_state.chain.invoke(query)
                st.markdown(answer)
                with st.expander("📄 Sources retrieved"):
                    for s in sources:
                        st.markdown(f"- `{s}`")
            except Exception as e:
                answer = f"Error generating answer: {e}\n\nThis may be a temporary Hugging Face rate limit - try again in ~20 seconds."
                sources = []
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })

if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
