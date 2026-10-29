"""On-disk knowledge index and incremental ingestion.

Layout of ``index/``::

    manifest.json        documents, file hashes, model name, skipped files, last run report
    chunks.jsonl.gz      one JSON record per chunk (text + metadata)
    embeddings.npy       float16 matrix, one L2-normalised row per chunk (same order)

The whole index is loaded into memory: at ~10k chunks x 384 dims this is ~8 MB
and exact cosine search is a single matrix-vector product.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .chunking import chunk_pages
from .config import INDEX_SCHEMA_VERSION, Settings
from .embeddings import EmbedderProtocol, timed_progress
from .pdf_text import extract_document

Log = Callable[[str], None]

MANIFEST = "manifest.json"
CHUNKS = "chunks.jsonl.gz"
EMBEDDINGS = "embeddings.npy"
ROOT_CATEGORY = "General"


# --------------------------------------------------------------------------
# Index container
# --------------------------------------------------------------------------


@dataclass
class KnowledgeIndex:
    index_dir: Path
    chunks: list[dict] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    manifest: dict = field(default_factory=dict)

    # ---- loading / saving ----------------------------------------------
    @classmethod
    def load(cls, index_dir: Path) -> "KnowledgeIndex":
        idx = cls(index_dir=index_dir)
        mpath = index_dir / MANIFEST
        if not mpath.exists():
            idx.manifest = _empty_manifest()
            return idx
        idx.manifest = json.loads(mpath.read_text())
        cpath, epath = index_dir / CHUNKS, index_dir / EMBEDDINGS
        if cpath.exists() and epath.exists():
            with gzip.open(cpath, "rt", encoding="utf-8") as fh:
                idx.chunks = [json.loads(line) for line in fh if line.strip()]
            idx.embeddings = np.load(epath).astype(np.float32)
            if len(idx.chunks) != idx.embeddings.shape[0]:
                raise ValueError(
                    f"Index is inconsistent: {len(idx.chunks)} chunks vs {idx.embeddings.shape[0]} vectors. "
                    "Rebuild it with `python ingest.py --rebuild`."
                )
        return idx

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_dir / ".tmp"
        tmp.mkdir(exist_ok=True)
        with gzip.open(tmp / CHUNKS, "wt", encoding="utf-8", compresslevel=6) as fh:
            for c in self.chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        emb = self.embeddings if self.embeddings.size else np.zeros((0, 0), dtype=np.float32)
        np.save(tmp / EMBEDDINGS, emb.astype(np.float16))
        (tmp / MANIFEST).write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False))
        for name in (CHUNKS, EMBEDDINGS, MANIFEST):  # manifest last = commit point
            os.replace(tmp / name, self.index_dir / name)
        tmp.rmdir()

    # ---- queries --------------------------------------------------------
    @property
    def documents(self) -> dict[str, dict]:
        return self.manifest.get("documents", {})

    @property
    def categories(self) -> list[str]:
        return sorted({d["category"] for d in self.documents.values()})

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def schema_ok(self) -> bool:
        return self.manifest.get("schema_version") == INDEX_SCHEMA_VERSION

    def mask(self, categories: list[str] | None = None, sources: list[str] | None = None) -> np.ndarray:
        m = np.ones(len(self.chunks), dtype=bool)
        if categories:
            cats = set(categories)
            m &= np.array([c["category"] in cats for c in self.chunks], dtype=bool)
        if sources:
            wanted = {s.lower() for s in sources}
            m &= np.array([_matches_source(c, wanted) for c in self.chunks], dtype=bool)
        return m

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        if self.embeddings.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.embeddings @ query_vec.astype(np.float32)

    def page_text(self, source: str, page: int) -> list[dict]:
        """All chunks that touch ``page`` of the document whose file name/title matches ``source``."""
        wanted = {source.lower()}
        hits = [c for c in self.chunks if _matches_source(c, wanted) and c["page"] <= page <= c["page_end"]]
        return sorted(hits, key=lambda c: c["chunk"])

    def resolve_source(self, name: str) -> str | None:
        """Map a loose document reference (file name, stem or title fragment) to a doc_id."""
        key = name.lower().strip()
        for doc_id, d in self.documents.items():
            if key in (doc_id.lower(), d["source"].lower(), Path(d["source"]).stem.lower()):
                return doc_id
        for doc_id, d in self.documents.items():
            if key in d["source"].lower() or key in d.get("title", "").lower():
                return doc_id
        return None


def _matches_source(chunk: dict, wanted: set[str]) -> bool:
    src = chunk["source"].lower()
    doc_id = chunk["doc_id"].lower()
    stem = Path(src).stem
    return any(w in (src, doc_id, stem) or (len(w) >= 4 and w in src) for w in wanted)


def _empty_manifest() -> dict:
    return {"schema_version": INDEX_SCHEMA_VERSION, "embed_model": None, "documents": {}, "skipped": {}, "last_run": None}


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_knowledge_bank(kb_dir: Path) -> dict[str, tuple[str, Path]]:
    """Return {doc_id: (category, path)} for every PDF. Sub-folder name = category."""
    found: dict[str, tuple[str, Path]] = {}
    if not kb_dir.is_dir():
        return found
    for path in sorted(kb_dir.rglob("*")):
        if path.suffix.lower() != ".pdf" or path.name.startswith("."):
            continue
        rel = path.relative_to(kb_dir)
        if any(part.startswith(("_", ".")) for part in rel.parts[:-1]):
            continue  # e.g. _Removed/ holds PDFs taken out of the index from the app
        category = rel.parts[0] if len(rel.parts) > 1 else ROOT_CATEGORY
        found[rel.as_posix()] = (category, path)
    return found


def _year_from_name(name: str) -> int | None:
    m = re.search(r"(19[5-9]\d|20[0-4]\d)", name)
    return int(m.group(1)) if m else None


def build_chunk_records(doc_id: str, category: str, path: Path, settings: Settings) -> tuple[list[dict], dict]:
    """Extract and chunk one PDF. Returns (chunk_records, doc_info). Raises ValueError if unusable."""
    doc = extract_document(path, settings.min_chars_per_page)
    if doc.warning:
        raise ValueError(doc.warning)
    chunks = chunk_pages(doc.pages, settings.chunk_chars, settings.chunk_overlap_chars, settings.min_chunk_chars)
    if not chunks:
        raise ValueError("no indexable text after cleaning")
    file_key = hashlib.sha1(doc_id.encode()).hexdigest()[:10]
    records = [
        {
            "id": f"{file_key}-{i:05d}",
            "doc_id": doc_id,
            "source": path.name,
            "category": category,
            "title": doc.title,
            "chunk": i,
            "page": c.page,
            "page_end": c.page_end,
            "section": c.section,
            "text": c.text,
        }
        for i, c in enumerate(chunks)
    ]
    info = {
        "source": path.name,
        "category": category,
        "title": doc.title,
        "year": _year_from_name(path.name),
        "pages": doc.n_pages,
        "chunks": len(records),
        "skipped_pages": doc.skipped_page_counts(),
    }
    return records, info


def embedding_text(record: dict) -> str:
    """Text actually embedded: the document title gives each chunk its context."""
    return f"{record['title']}\n{record['text']}"


def ingest(
    settings: Settings,
    embedder: EmbedderProtocol,
    rebuild: bool = False,
    log: Log = print,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Bring the index in line with the Knowledge Bank folder.

    New or changed PDFs are (re-)indexed, deleted PDFs are dropped, unchanged
    PDFs are left alone. Returns a run report (also stored in the manifest).
    """
    t0 = time.time()
    kb_dir, index_dir = settings.kb_dir, settings.index_dir
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge Bank folder not found: {kb_dir} (set CLINSTAT_KB_DIR)")

    index = KnowledgeIndex.load(index_dir)
    model_changed = index.manifest.get("embed_model") not in (None, embedder.model_name)
    if rebuild or not index.schema_ok() or model_changed:
        if not rebuild:
            log("Index schema or embedding model changed - rebuilding from scratch.")
        index = KnowledgeIndex(index_dir=index_dir, manifest=_empty_manifest())
    index.manifest["embed_model"] = embedder.model_name

    on_disk = scan_knowledge_bank(kb_dir)
    docs = index.manifest["documents"]
    skipped_before = index.manifest.get("skipped", {})
    hashes = {}
    for doc_id, (_, path) in on_disk.items():
        hashes[doc_id] = file_sha256(path)

    removed = [d for d in docs if d not in on_disk]
    todo = []
    unchanged = []
    for doc_id in on_disk:
        prev = docs.get(doc_id) or skipped_before.get(doc_id)
        if prev and prev.get("sha256") == hashes[doc_id]:
            unchanged.append(doc_id)
        else:
            todo.append(doc_id)

    log(f"Knowledge Bank: {len(on_disk)} PDF(s) in {kb_dir}")
    log(f"  unchanged: {len(unchanged)} | to index: {len(todo)} | removed: {len(removed)}")

    drop = set(removed) | {d for d in todo if d in docs}
    keep_rows = [i for i, c in enumerate(index.chunks) if c["doc_id"] not in drop]
    chunks = [index.chunks[i] for i in keep_rows]
    emb = index.embeddings[keep_rows] if index.embeddings.size else None
    for d in removed:
        docs.pop(d, None)
        log(f"  - removed {d}")
    skipped = {k: v for k, v in skipped_before.items() if k in on_disk and k not in todo}

    report = {"added": [], "updated": [], "removed": removed, "unchanged": len(unchanged), "skipped": []}
    new_records: list[dict] = []
    for n, doc_id in enumerate(todo, start=1):
        category, path = on_disk[doc_id]
        t_doc = time.time()
        if progress:
            progress((n - 1) / max(len(todo), 1), f"Reading {doc_id}")
        try:
            records, info = build_chunk_records(doc_id, category, path, settings)
        except Exception as exc:
            reason = str(exc)
            skipped[doc_id] = {"sha256": hashes[doc_id], "reason": reason, "category": category, "source": path.name}
            docs.pop(doc_id, None)
            report["skipped"].append({"doc_id": doc_id, "reason": reason})
            log(f"  [{n}/{len(todo)}] SKIPPED {doc_id}: {reason}")
            continue
        info["sha256"] = hashes[doc_id]
        info["indexed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (report["updated"] if doc_id in docs else report["added"]).append(doc_id)
        docs[doc_id] = info
        new_records.extend(records)
        skipped_pages = ", ".join(f"{v} {k}" for k, v in info["skipped_pages"].items()) or "none"
        log(
            f"  [{n}/{len(todo)}] {doc_id}: {info['pages']} pages -> {info['chunks']} chunks "
            f"(pages left out: {skipped_pages}) [{time.time() - t_doc:.1f}s]"
        )

    if new_records:
        log(f"Embedding {len(new_records)} new chunks with {embedder.model_name} ...")
        if progress:
            progress(0.9, f"Embedding {len(new_records)} chunks")
        vecs = embedder.embed_passages(
            [embedding_text(r) for r in new_records], progress=timed_progress("embedded", log)
        )
        chunks.extend(new_records)
        emb = vecs if emb is None or emb.size == 0 else np.vstack([emb, vecs])

    index.chunks = chunks
    index.embeddings = emb if emb is not None else np.zeros((0, 0), dtype=np.float32)
    report["seconds"] = round(time.time() - t0, 1)
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index.manifest["skipped"] = skipped
    index.manifest["last_run"] = report
    index.manifest["schema_version"] = INDEX_SCHEMA_VERSION
    index.save()
    if progress:
        progress(1.0, "Done")
    log(
        f"Done in {report['seconds']}s: {len(report['added'])} added, {len(report['updated'])} updated, "
        f"{len(removed)} removed, {len(report['skipped'])} skipped. Index now holds "
        f"{len(docs)} documents / {len(chunks)} chunks."
    )
    if skipped:
        log("Files not indexed (fix and re-run):")
        for doc_id, s in skipped.items():
            log(f"  - {doc_id}: {s['reason']}")
    return report


def remove_document(index_dir: Path, doc_id: str) -> bool:
    """Drop a document from the index without touching the PDF."""
    index = KnowledgeIndex.load(index_dir)
    if doc_id not in index.documents:
        return False
    keep = [i for i, c in enumerate(index.chunks) if c["doc_id"] != doc_id]
    index.chunks = [index.chunks[i] for i in keep]
    index.embeddings = index.embeddings[keep] if index.embeddings.size else index.embeddings
    index.manifest["documents"].pop(doc_id)
    index.save()
    return True
