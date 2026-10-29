"""Terminal chat with the ClinStat agent.

    python clinstat_query.py
"""
from __future__ import annotations

import sys

from clinstat.agent import run_agent
from clinstat.config import settings
from clinstat.llm import GroqChat, MissingAPIKey
from clinstat.runtime import build_retriever


def main() -> int:
    print("Loading index and models ...")
    retriever = build_retriever()
    if retriever.index.is_empty:
        print("Index is empty - run `python ingest.py` first.", file=sys.stderr)
        return 1
    try:
        llm = GroqChat()
    except MissingAPIKey as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Ready: {len(retriever.index.documents)} documents, {len(retriever.index.chunks)} chunks. "
          f"Model {llm.model}. Type 'quit' to exit.\n")

    history: list[dict] = []
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", ""):
            break

        def on_event(e: dict) -> None:
            if e["type"] == "tool_start":
                print(f"   ... {e['tool']} {e['args'][:120]}")

        result = run_agent(query, retriever, llm, history=history, on_event=on_event)
        if result.error:
            print(f"\n[error] {result.error}\n")
            continue
        print(f"\nClinStat:\n{result.answer}\n")
        cited = set(result.cited)
        for s in result.citations.sources:
            if s.tag in cited:
                print(f"  [{s.tag}] {s.label}")
        for c in result.citations.calcs:
            print(f"  [{c.tag}] {c.result.calculator}: {c.result.outputs}")
        print()
        history += [{"role": "user", "content": query}, {"role": "assistant", "content": result.answer}]
        history = history[-2 * settings.history_turns:]
    return 0


if __name__ == "__main__":
    sys.exit(main())
