import os
from dotenv import load_dotenv
from fastembed import TextEmbedding
from rag_pipe import load_vector_store, run_agent, EMBED_MODEL

load_dotenv()

def main():
    print("Loading embedder and vector store...")
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    collection = load_vector_store()
    print(f"Ready. {collection.count()} chunks indexed.\n")

    while True:
        query = input("Ask a question (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break

        answer, sources, used_search = run_agent(query, collection, embedder)

        print("\n--- Answer ---")
        print(answer)
        if used_search:
            print("\n--- Sources ---")
            print(set(sources))
        print()

if __name__ == "__main__":
    main()