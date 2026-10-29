"""Helpers that assemble the index, models and retriever (shared by the app and CLIs)."""
from __future__ import annotations

import logging

from .config import Settings, settings as default_settings
from .embeddings import FastEmbedder, FastReranker
from .index_store import KnowledgeIndex
from .retrieval import Retriever

log = logging.getLogger("clinstat")


def load_index(cfg: Settings | None = None) -> KnowledgeIndex:
    cfg = cfg or default_settings
    return KnowledgeIndex.load(cfg.index_dir)


def load_embedder(cfg: Settings | None = None, index: KnowledgeIndex | None = None) -> FastEmbedder:
    cfg = cfg or default_settings
    model = (index.manifest.get("embed_model") if index is not None else None) or cfg.embed_model
    return FastEmbedder(model)


def load_reranker(cfg: Settings | None = None) -> FastReranker | None:
    cfg = cfg or default_settings
    if not cfg.use_reranker:
        return None
    try:
        return FastReranker(cfg.rerank_model)
    except Exception as exc:  # model download blocked, etc. - retrieval still works without it
        log.warning("Re-ranker unavailable (%s); continuing with hybrid retrieval only.", exc)
        return None


def build_retriever(cfg: Settings | None = None, with_reranker: bool = True) -> Retriever:
    cfg = cfg or default_settings
    index = load_index(cfg)
    embedder = load_embedder(cfg, index)
    reranker = load_reranker(cfg) if with_reranker else None
    return Retriever(index, embedder, reranker, cfg)
