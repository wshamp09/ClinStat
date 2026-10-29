"""Retrieval evaluation harness.

Scores each retrieval mode on a fixed question set so that changes to
chunking, models or the knowledge bank can be checked objectively.

Metrics (per mode, averaged over questions):
* doc_hit@k  - a passage from the expected document is in the top k
* page_hit@k - a passage within +/-1 page of an expected page is in the top k
               (only questions that list expected_pages)
* MRR        - mean reciprocal rank of the first passage from the expected document
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .config import PROJECT_DIR
from .retrieval import Retriever

DEFAULT_QUESTIONS = PROJECT_DIR / "eval" / "questions.json"


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["questions"] if isinstance(data, dict) else data


def _doc_match(chunk: dict, expected: str) -> bool:
    e = expected.lower()
    return chunk["source"].lower() == e or e in chunk["source"].lower()


def _page_match(chunk: dict, pages: list[int]) -> bool:
    return any(chunk["page"] - 1 <= p <= chunk["page_end"] + 1 for p in pages)


def evaluate(
    retriever: Retriever,
    questions: list[dict],
    k: int = 5,
    modes: list[tuple[str, bool]] | None = None,
    log: Callable[[str], None] = print,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Run every (mode, rerank) configuration. Returns {"summary": [...], "details": [...]}."""
    if modes is None:
        modes = [("bm25", False), ("dense", False), ("hybrid", False)]
        if retriever.reranker is not None:
            modes.append(("hybrid", True))
    indexed = {d["source"].lower() for d in retriever.index.documents.values()}
    usable = [q for q in questions if any(q["expected_doc"].lower() in s for s in indexed)]
    missing = len(questions) - len(usable)
    if missing:
        log(f"Skipping {missing} question(s) whose expected document is not in the index.")

    summary, details = [], []
    total_steps = max(len(modes) * len(usable), 1)
    step = 0
    for mode, rerank in modes:
        name = mode + (" + rerank" if rerank else "")
        t0 = time.time()
        doc_hits = page_hits = page_n = 0
        rr_sum = 0.0
        for q in usable:
            step += 1
            if progress:
                progress(step / total_steps, f"{name}: {q['id']}")
            hits = retriever.search(q["question"], top_k=k, mode=mode, rerank=rerank)
            rank = next((r for r, p in enumerate(hits, start=1) if _doc_match(p.chunk, q["expected_doc"])), None)
            doc_hit = rank is not None
            doc_hits += doc_hit
            rr_sum += 1.0 / rank if rank else 0.0
            page_hit = None
            if q.get("expected_pages"):
                page_n += 1
                page_hit = any(_doc_match(p.chunk, q["expected_doc"]) and _page_match(p.chunk, q["expected_pages"]) for p in hits)
                page_hits += page_hit
            details.append({
                "mode": name, "id": q["id"], "question": q["question"], "expected_doc": q["expected_doc"],
                "doc_hit": doc_hit, "first_rank": rank, "page_hit": page_hit,
                "top_sources": "; ".join(f"{p.chunk['source']} p.{p.chunk['page']}" for p in hits[:3]),
            })
        n = max(len(usable), 1)
        row = {
            "mode": name,
            f"doc_hit@{k}": round(doc_hits / n, 3),
            f"page_hit@{k}": round(page_hits / page_n, 3) if page_n else None,
            "MRR": round(rr_sum / n, 3),
            "questions": len(usable),
            "seconds": round(time.time() - t0, 1),
        }
        summary.append(row)
        log(f"  {name:16s} doc_hit@{k}={row[f'doc_hit@{k}']:.3f}  MRR={row['MRR']:.3f}"
            + (f"  page_hit@{k}={row[f'page_hit@{k}']:.3f}" if page_n else "") + f"  ({row['seconds']}s)")
    return {"summary": summary, "details": details, "k": k}
