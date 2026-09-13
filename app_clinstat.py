import os
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from rag_pipe import load_vector_store, run_agent, EMBED_MODEL

load_dotenv()

st.set_page_config(page_title="ClinStat Agent", layout="centered")
st.title("🤖 ClinStat Research Agent")

@st.cache_resource
def get_embedder():
    return TextEmbedding(model_name=EMBED_MODEL)

@st.cache_resource
def get_collection():
    return load_vector_store()

embedder = get_embedder()
collection = get_collection()

st.caption(f"{collection.count()} chunks indexed.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": ...}

# Render prior turns
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

query = st.chat_input("Ask a question about your documents...")

if st.button("Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources, used_search = run_agent(
                query, collection, embedder, history=st.session_state.chat_history
            )
        st.write(answer)
        if used_search:
            with st.expander("Sources referenced"):
                for src in set(sources):
                    st.write(f"- {src}")

    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Keep only the last 4 messages (2 user/assistant pairs) — retrieved context
# already adds significant tokens per turn, so history needs to stay lean
st.session_state.chat_history = st.session_state.chat_history[-4:]