"""Hybrid retrieval: BM25 keyword search + dense vectors, fused and re-ranked.

Why hybrid: dense embeddings capture meaning ("what to do about patients who
stop treatment" ~ "intercurrent events"), while BM25 nails exact terms the
embedding model has rarely seen ("Grizzle", "PROC GLIMMIX", "E9(R1)", "TOST").
Reciprocal-rank fusion combines the two rankings without having to calibrate
their scores, and an optional cross-encoder re-reads the top candidates
together with the query for the final order.
"""
from __future__ import annotations

import math
import re
import time
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .index_store import KnowledgeIndex

_STOP = set(
    "a an and are as at be been being by can could did do does for from had has have how if in into is it its "
    "may might more most no not of on or our should so such than that the their them then there these they this "
    "those to under use used using was we were what when where which while who why will with would you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.extend(_normalize_token(tok))
    return out


@lru_cache(maxsize=500_000)
def _normalize_token(tok: str) -> tuple[str, ...]:
    """Split hyphenated words (cross-over -> cross, over, crossover), drop stopwords, stem.

    Cached: the corpus has far fewer distinct words than word occurrences, so this
    makes building the keyword index several times faster.
    """
    parts = tok.replace("'", "").split("-")
    cands = parts + ["".join(parts)] if len(parts) > 1 else [tok]
    return tuple(_stem(t) for t in cands if t not in _STOP and (len(t) > 1 or t.isdigit()))


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


class BM25:
    """Okapi BM25 over the in-memory chunk list (pure numpy)."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            lengths[i] = len(toks)
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        self.avgdl = float(lengths.mean()) if self.n else 0.0
        self.norm = (k1 * (1 - b + b * lengths / max(self.avgdl, 1e-9))).astype(np.float32)
        self.postings = {
            t: (np.array([d for d, _ in p], dtype=np.int32), np.array([f for _, f in p], dtype=np.float32))
            for t, p in postings.items()
        }
        self.idf = {
            t: math.log(1 + (self.n - len(p[0]) + 0.5) / (len(p[0]) + 0.5)) for t, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            docs, tfs = self.postings[term]
            s[docs] += self.idf[term] * tfs * (self.k1 + 1) / (tfs + self.norm[docs])
        return s


@dataclass
class Passage:
    chunk: dict
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def label(self) -> str:
        c = self.chunk
        pages = f"p.{c['page']}" if c["page"] == c["page_end"] else f"pp.{c['page']}-{c['page_end']}"
        return f"{c['category']}/{c['source']}, {pages}"


def _top_indices(scores: np.ndarray, mask: np.ndarray, k: int) -> list[int]:
    s = np.where(mask, scores, -np.inf)
    valid = int(np.isfinite(s).sum())
    k = min(k, valid)
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    return [int(i) for i in idx if np.isfinite(s[i])]


class Retriever:
    MODES = ("hybrid", "dense", "bm25")

    def __init__(self, index: KnowledgeIndex, embedder, reranker=None, settings: Settings | None = None):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings or Settings()
        self._bm25: BM25 | None = None
        self.last_timing: dict[str, float] = {}

    def with_settings(self, settings: Settings) -> "Retriever":
        """A per-session copy with different settings that shares the index, models and BM25."""
        r = Retriever(self.index, self.embedder, self.reranker, settings)
        r._bm25 = self._bm25
        return r

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([f"{c['title']} {c['section']} {c['text']}" for c in self.index.chunks])
        return self._bm25

    def search(
        self,
        query: str,
        top_k: int | None = None,
        categories: list[str] | None = None,
        sources: list[str] | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
    ) -> list[Passage]:
        if self.index.is_empty or not query.strip():
            return []
        st = self.settings
        top_k = top_k or st.top_k
        rerank = st.use_reranker if rerank is None else rerank
        mask = self.index.mask(categories, sources)
        if not mask.any():
            return []
        n_cand = st.candidates_per_retriever
        t0 = time.perf_counter()
        timing = {"dense_s": 0.0, "bm25_s": 0.0, "rerank_s": 0.0}

        dense_ids: list[int] = []
        bm25_ids: list[int] = []
        if mode in ("hybrid", "dense"):
            t = time.perf_counter()
            qv = self.embedder.embed_query(query)
            dense_ids = _top_indices(self.index.dense_scores(qv), mask, n_cand)
            timing["dense_s"] = time.perf_counter() - t
        if mode in ("hybrid", "bm25"):
            t = time.perf_counter()
            bm = self.bm25.scores(query)
            bm25_ids = [i for i in _top_indices(bm, mask, n_cand) if bm[i] > 0]
            timing["bm25_s"] = time.perf_counter() - t

        fused: dict[int, float] = defaultdict(float)
        d_rank = {i: r for r, i in enumerate(dense_ids, start=1)}
        b_rank = {i: r for r, i in enumerate(bm25_ids, start=1)}
        for i, r in d_rank.items():
            fused[i] += 1.0 / (st.rrf_k + r)
        for i, r in b_rank.items():
            fused[i] += 1.0 / (st.rrf_k + r)
        ranked = sorted(fused, key=lambda i: -fused[i])

        if rerank and self.reranker is not None and ranked:
            pool = ranked[: st.rerank_pool]
            t = time.perf_counter()
            try:
                scores = self.reranker.score(
                    query, [f"{self.index.chunks[i]['title']}\n{self.index.chunks[i]['text']}" for i in pool]
                )
                order = sorted(range(len(pool)), key=lambda j: -scores[j])
                ranked = [pool[j] for j in order] + ranked[st.rerank_pool:]
                final_scores = {pool[j]: scores[j] for j in range(len(pool))}
            except Exception:
                final_scores = fused
            timing["rerank_s"] = time.perf_counter() - t
        else:
            final_scores = fused

        out: list[Passage] = []
        per_page: Counter = Counter()
        for i in ranked:
            c = self.index.chunks[i]
            key = (c["doc_id"], c["page"])
            if per_page[key] >= 2:  # overlapping neighbours add little
                continue
            per_page[key] += 1
            out.append(Passage(chunk=c, score=float(final_scores.get(i, 0.0)), dense_rank=d_rank.get(i), bm25_rank=b_rank.get(i)))
            if len(out) >= top_k:
                break
        timing["total_s"] = time.perf_counter() - t0
        self.last_timing = timing
        return out
