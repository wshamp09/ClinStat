import os
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from rag_pipe import load_vector_store, run_agent, expand_with_example, retrieve_context, ask_groq, EMBED_MODEL

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
    placeholder="e.g. Briefly describe adaptive approaches, then show me R code for a simple adaptive design simulation",
)

col1, col2 = st.columns([2, 1])
with col1:
    ask_clicked = st.button("Ask")
with col2:
    want_example = st.checkbox("Also generate code example", value=False)
    language = st.selectbox("Language", ["R", "Python", "SAS"], disabled=not want_example)

if ask_clicked and query:
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

    if want_example:
        with st.spinner(f"Generating {language} example..."):
            docs, doc_sources = retrieve_context(collection, embedder, query)
            example = expand_with_example(query, answer, docs, doc_sources, language=language)

        st.markdown(f"### {language} example (generated — not sourced from documents)")
        st.code(example, language=language.lower())