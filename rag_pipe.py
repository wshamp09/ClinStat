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
CHUNK_SIZE = 500
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
# PDF loading + chunking
# ---------------------------------------------------------------
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

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
            print(f"Processing [{category}] {os.path.basename(path)}...")
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)
            if not chunks:
                continue

            embeddings = [emb.tolist() for emb in embedder.embed(chunks)]
            filename = os.path.basename(path)
            ids = [f"{category}_{filename}_{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": filename, "category": category, "chunk": i}
                for i in range(len(chunks))
            ]

            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
            total_indexed += 1

    print(f"Indexed {total_indexed} PDF(s) across {len(SUBFOLDERS)} categories into '{CHROMA_DIR}'.")
    return collection

def load_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection("pdf_docs")

# ---------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------
def retrieve_context(collection, embedder, query, top_k=4, category=None):
    query_embedding = list(embedder.embed([query]))[0].tolist()
    where_filter = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )
    docs = results["documents"][0]
    sources = [f"{m['category']}/{m['source']}" for m in results["metadatas"][0]]
    return docs, sources

# ---------------------------------------------------------------
# Direct Groq call
# ---------------------------------------------------------------
def ask_groq(query, context_chunks, sources):
    client = Groq(api_key=get_groq_api_key())

    context_block = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src in zip(context_chunks, sources)
    )

    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context_block}

Question: {query}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

def expand_with_example(query, answer, context_chunks, sources, language="R"):
    client = Groq(api_key=get_groq_api_key())

    context_block = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src in zip(context_chunks, sources)
    )

    prompt = f"""You previously answered a question using document context. Now write a
{language} code example that illustrates the concept in a practical way.

Original question: {query}

Grounded answer (from documents):
{answer}

Supporting document context:
{context_block}

Write a self-contained, runnable {language} example with brief comments explaining each step.
If the documents don't contain enough detail to make the example specific, use reasonable
general knowledge of the method, but stay consistent with what the documents describe.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": f"You are an expert statistical programmer who writes clear, well-commented {language} code."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------
# Agent
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

def run_agent(user_query, collection, embedder, max_tool_rounds=3):
    client = Groq(api_key=get_groq_api_key())

    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical/regulatory research assistant with access to a knowledge bank "
                "containing three categories: RegulatoryDocs, StatBooks, and StatPapers. "
                "You have a search_documents tool to search this knowledge bank, optionally filtered by category. "
                "Use it when the question relates to specifics that might be in those documents. "
                "If the question is general knowledge, you may answer directly, or search first to ground "
                "your answer, then supplement with general knowledge. Clearly separate: (1) what comes from "
                "retrieved documents (cite source filename and category), and (2) general knowledge/synthesis."
            )
        },
        {"role": "user", "content": user_query}
    ]

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

    # Hit max rounds — sidestep tool-choice conflicts entirely by starting a
    # fresh, plain (non-tool) completion using everything gathered so far.
    context_block = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src in zip(all_docs, all_sources)
    )
    fallback_prompt = f"""Based on the following retrieved document context, answer the original question.
Clearly separate what comes from the documents (cite source) from general knowledge.

Retrieved context:
{context_block}

Original question: {user_query}
"""
    final = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a clinical/regulatory research assistant."},
            {"role": "user", "content": fallback_prompt},
        ],
        temperature=0.3,
    )
    return final.choices[0].message.content, all_sources, used_search

# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    collection = build_vector_store()
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    query = input("\nAsk a question: ")
    answer, sources, used_search = run_agent(query, collection, embedder)

    print("\n--- Answer ---")
    print(answer)
    if used_search:
        print("\n--- Sources used ---")
        print(set(sources))