"""Build or update the knowledge index from the Knowledge Bank folder.

    python ingest.py                 # incremental: only new / changed / removed PDFs
    python ingest.py --rebuild       # re-index everything from scratch
    python ingest.py --kb-dir PATH   # use a different Knowledge Bank folder

Each sub-folder of the Knowledge Bank becomes a category (e.g. RegulatoryDocs,
StatBooks, StatPapers, Ophthalmic). Folders starting with "_" are ignored.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from clinstat.config import Settings
from clinstat.embeddings import FastEmbedder
from clinstat.index_store import ingest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="discard the existing index and rebuild")
    parser.add_argument("--kb-dir", type=Path, help="Knowledge Bank folder (default: CLINSTAT_KB_DIR or ~/Documents/KnowledgeBank)")
    parser.add_argument("--index-dir", type=Path, help="index output folder (default: ./index)")
    args = parser.parse_args()

    cfg = Settings()
    if args.kb_dir:
        cfg = dataclasses.replace(cfg, kb_dir=args.kb_dir.expanduser().resolve())
    if args.index_dir:
        cfg = dataclasses.replace(cfg, index_dir=args.index_dir.expanduser().resolve())

    print(f"Knowledge Bank : {cfg.kb_dir}")
    print(f"Index folder   : {cfg.index_dir}")
    print(f"Embedding model: {cfg.embed_model} (first run downloads it)")
    embedder = FastEmbedder(cfg.embed_model)
    try:
        ingest(cfg, embedder, rebuild=args.rebuild, log=lambda m: print(m, flush=True))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
