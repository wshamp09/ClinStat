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

st.caption(f"{collection.count()} chunks indexed. Agent decides when to search your documents vs. answer from general knowledge.")

query = st.text_area(
    "Your question",
    height=100,
    placeholder="e.g. Briefly describe adaptive approaches",
)

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        answer, sources, used_search = run_agent(query, collection, embedder)

    st.markdown("### Answer")
    st.write(answer)

    if used_search:
        with st.expander("Documents searched"):
            for src in set(sources):
                st.write(f"- {src}")
    else:
        st.caption("_Answered from general knowledge — no document search was needed._")