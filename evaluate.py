"""Score retrieval quality on eval/questions.json.

    python evaluate.py            # all modes, k = 5
    python evaluate.py --k 8 --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

from clinstat.evaluation import DEFAULT_QUESTIONS, evaluate, load_questions
from clinstat.runtime import build_retriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--csv", help="write per-question details to this CSV")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    retriever = build_retriever(with_reranker=not args.no_rerank)
    if retriever.index.is_empty:
        print("Index is empty - run `python ingest.py` first.", file=sys.stderr)
        return 1
    questions = load_questions(args.questions)
    print(f"Evaluating {len(questions)} questions against {len(retriever.index.chunks)} chunks (k={args.k})")
    result = evaluate(retriever, questions, k=args.k)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(result["details"][0].keys()))
            writer.writeheader()
            writer.writerows(result["details"])
        print(f"Details written to {args.csv}")
    misses = [d for d in result["details"] if not d["doc_hit"] and d["mode"] == result["summary"][-1]["mode"]]
    if misses:
        print(f"\nMisses for '{result['summary'][-1]['mode']}':")
        for d in misses:
            print(f"  {d['id']}: {d['question']}  (expected {d['expected_doc']}; got {d['top_sources']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
