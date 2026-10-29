"""Embedding and re-ranking models (fastembed / ONNX - no torch required)."""
from __future__ import annotations

import os
import time
from typing import Callable, Iterable, Protocol

import numpy as np

ProgressFn = Callable[[int, int], None]


def model_cache_dir() -> str:
    """Keep downloaded ONNX models in the project (git-ignored) instead of a temp dir."""
    from .config import PROJECT_DIR

    return os.environ.get("FASTEMBED_CACHE_PATH", str(PROJECT_DIR / ".cache" / "fastembed"))


class EmbedderProtocol(Protocol):
    model_name: str

    def embed_passages(self, texts: list[str], progress: ProgressFn | None = None) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FastEmbedder:
    """BGE-style embedder. Queries and passages are embedded differently:
    bge models expect an instruction prefix on the query side, which
    fastembed's ``query_embed`` adds automatically."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=model_cache_dir())

    def embed_passages(self, texts: list[str], progress: ProgressFn | None = None) -> np.ndarray:
        out: list[np.ndarray] = []
        total = len(texts)
        for i, vec in enumerate(self._model.passage_embed(texts, batch_size=self.batch_size), start=1):
            out.append(vec)
            if progress and (i % self.batch_size == 0 or i == total):
                progress(i, total)
        if not out:
            return np.zeros((0, 384), dtype=np.float32)
        return _normalize(np.vstack(out))

    def embed_query(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.query_embed(text)))
        return _normalize(np.asarray(vec)[None, :])[0]


class FastReranker:
    """Cross-encoder re-ranker (reads query and passage together)."""

    def __init__(self, model_name: str, batch_size: int = 16):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=model_cache_dir())

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, passages, batch_size=self.batch_size)]


class HashingEmbedder:
    """Deterministic, dependency-free stand-in used by the unit tests.

    It is NOT a semantic model; it hashes word unigrams/bigrams into a fixed
    number of dimensions so the plumbing can be exercised offline.
    """

    def __init__(self, dim: int = 256):
        self.model_name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        import re
        import zlib

        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"[a-z0-9]+", text.lower())
        for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        return v

    def embed_passages(self, texts: Iterable[str], progress: ProgressFn | None = None) -> np.ndarray:
        texts = list(texts)
        mat = np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        if progress:
            progress(len(texts), len(texts))
        return _normalize(mat)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]


def timed_progress(label: str, log: Callable[[str], None] = print, every_s: float = 5.0) -> ProgressFn:
    """Progress callback that logs at most every ``every_s`` seconds (and at the end)."""
    start = time.time()
    last = [0.0]

    def _cb(done: int, total: int) -> None:
        now = time.time()
        if now - last[0] >= every_s or done == total:
            last[0] = now
            rate = done / max(now - start, 1e-9)
            eta = (total - done) / rate if rate > 0 else 0
            log(f"    {label}: {done}/{total} ({100 * done / max(total, 1):.0f}%) - {rate:.1f}/s, ~{eta:.0f}s left")

    return _cb
