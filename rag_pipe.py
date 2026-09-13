import os
import glob
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from fastembed import TextEmbedding
import chromadb
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
KNOWLEDGE_BANK_DIR = os.path.expanduser("~/Documents/KnowledgeBank")
SUBFOLDERS = ["RegulatoryDocs", "StatBooks", "StatPapers"]
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500       # target characters per chunk
CHUNK_OVERLAP = 50
GROQ_MODEL = "openai/gpt-oss-120b"

# ---------------------------------------------------------------
# Auth helper (local .env or Streamlit Cloud secrets)
# ---------------------------------------------------------------
def get_groq_api_key():
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")

# ---------------------------------------------------------------
# PDF loading — page-aware extraction
# ---------------------------------------------------------------
def extract_pages(path):
    """Returns a list of (page_number, page_text) tuples, 1-indexed."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages

# ---------------------------------------------------------------
# Chunking — paragraph-aware, tracks the page each chunk starts on
# ---------------------------------------------------------------
def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits page text into paragraphs, then groups paragraphs into chunks
    up to ~chunk_size characters. Each chunk records the page number it
    started on. Avoids cutting mid-paragraph where possible.
    """
    chunks = []  # list of (text, start_page)
    buffer = ""
    buffer_start_page = None

    for page_num, page_text in pages:
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        if not paragraphs:
            # fall back to line splits if no double-newline paragraphs found
            paragraphs = [p.strip() for p in page_text.split("\n") if p.strip()]

        for para in paragraphs:
            if buffer_start_page is None:
                buffer_start_page = page_num

            if len(buffer) + len(para) + 1 <= chunk_size:
                buffer = f"{buffer}\n{para}".strip()
            else:
                if buffer:
                    chunks.append((buffer, buffer_start_page))
                # start new buffer; carry a small overlap from the end of the previous buffer
                overlap_text = buffer[-overlap:] if overlap and buffer else ""
                buffer = f"{overlap_text}\n{para}".strip()
                buffer_start_page = page_num

    if buffer:
        chunks.append((buffer, buffer_start_page))

    return chunks

# ---------------------------------------------------------------
# Embedding + vector store
# ---------------------------------------------------------------
def build_vector_store():
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("pdf_docs")

    total_indexed = 0
    for category in SUBFOLDERS:
        folder_path = os.path.join(KNOWLEDGE_BANK_DIR, category)
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

        if not pdf_files:
            print(f"No PDFs found in '{folder_path}'. Skipping.")
            continue

        for path in pdf_files:
            filename = os.path.basename(path)
            print(f"Processing [{category}] {filename}...")

            pages = extract_pages(path)
            total_text_len = sum(len(t) for _, t in pages)
            if total_text_len < 100:
                print(f"⚠️  Warning: {filename} extracted almost no text — may be scanned/image-based.")
                continue

            chunk_tuples = chunk_pages(pages)
            if not chunk_tuples:
                continue

            texts = [c[0] for c in chunk_tuples]
            embeddings = [emb.tolist() for emb in embedder.embed(texts)]
            ids = [f"{category}_{filename}_{i}" for i in range(len(chunk_tuples))]
            metadatas = [
                {"source": filename, "category": category, "chunk": i, "page": page_num}
                for i, (_, page_num) in enumerate(chunk_tuples)
            ]

            collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            total_indexed += 1

    print(f"Indexed {total_indexed} PDF(s) across {len(SUBFOLDERS)} categories into '{CHROMA_DIR}'.")
    return collection

def load_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection("pdf_docs")

# ---------------------------------------------------------------
# Retrieval — now includes page numbers in the source label
# ---------------------------------------------------------------
def retrieve_context(collection, embedder, query, top_k=3, category=None):
    query_embedding = list(embedder.embed([query]))[0].tolist()
    where_filter = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )
    docs = results["documents"][0]
    sources = [
        f"{m['category']}/{m['source']}, p.{m.get('page', '?')}"
        for m in results["metadatas"][0]
    ]
    return docs, sources

# ---------------------------------------------------------------
# Agent (with conversation memory + page-cited answers)
# ---------------------------------------------------------------
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the indexed knowledge bank (regulatory documents, statistics textbooks, "
            "and statistics papers) for relevant context. Use this when the question could be "
            "answered or informed by the user's specific document set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "category": {
                    "type": "string",
                    "enum": SUBFOLDERS,
                    "description": "Optional: restrict to one category if the question clearly points to a source type."
                }
            },
            "required": ["query"]
        }
    }
}

SYSTEM_PROMPT = (
    "You are a clinical/regulatory research assistant with access to a knowledge bank "
    "containing three categories: RegulatoryDocs, StatBooks, and StatPapers. "
    "You have a search_documents tool to search this knowledge bank, optionally filtered by category. "
    "Use it when the question relates to specifics that might be in those documents. "
    "Write one cohesive, natural answer that blends retrieved document content with your own "
    "expertise — do not split your response into separate 'from documents' and 'general knowledge' "
    "sections. Instead, cite sources inline as you use them, e.g. 'per the FDA adaptive design "
    "guidance (RegulatoryDocs/filename.pdf, p.12), ...'. If a claim doesn't come from a specific "
    "retrieved passage, just state it plainly without a citation. This is a multi-turn conversation — "
    "use prior turns for context on follow-up questions (e.g. 'what about the 3-period version')."
)

def run_agent(user_query, collection, embedder, history=None, max_tool_rounds=2):
    """
    history: list of prior {"role": ..., "content": ...} messages (user/assistant turns only,
    no tool-call internals) to carry conversation context across turns.
    Returns (answer_text, sources_used, used_search).
    """
    client = Groq(api_key=get_groq_api_key())

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    all_sources = []
    all_docs = []
    used_search = False

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=[SEARCH_TOOL],
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content, all_sources, used_search

        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            search_query = args.get("query", user_query)
            category = args.get("category")

            docs, sources = retrieve_context(collection, embedder, search_query, category=category)
            used_search = True
            all_sources.extend(sources)
            all_docs.extend(docs)

            tool_result = "\n\n---\n\n".join(
                f"[Source: {src}]\n{chunk}" for chunk, src in zip(docs, sources)
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result or "No relevant documents found."
            })

    # Fallback: fresh plain completion to avoid tool_choice conflicts
    context_block = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src in zip(all_docs, all_sources)
    )
    history_block = ""
    if history:
        history_block = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    fallback_prompt = f"""Conversation so far:
{history_block}

Based on the following retrieved document context, write one cohesive answer to the latest question.
Blend document content naturally into your explanation, citing sources inline
(e.g. "(RegulatoryDocs/filename.pdf, p.12)") only where a specific claim comes from that source.

Retrieved context:
{context_block}

Latest question: {user_query}
"""
    final = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fallback_prompt},
        ],
        temperature=0.3,
    )
    return final.choices[0].message.content, all_sources, used_search

# ---------------------------------------------------------------
# CLI entry point (ingestion)
# ---------------------------------------------------------------
if __name__ == "__main__":
    collection = build_vector_store()