"""Sentence-aware chunking that keeps track of section headings and pages.

Each chunk is ~``chunk_chars`` characters, ends on a sentence boundary, carries
``overlap_chars`` of trailing sentences into the next chunk, and records the
section it falls under. Chunks may span a page break (paragraphs often do);
``page`` is where the chunk starts and ``page_end`` where it finishes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf_text import CleanPage

# Split after . ? ! when followed by whitespace and an upper-case letter, digit or
# opening bracket - but not after common abbreviations or single initials.
_ABBREV = (
    r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bal\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bNo\.)"
    r"(?<!\bp\.)(?<!\bpp\.)(?<!\bSec\.)(?<!\bref\.)(?<!\bcf\.)(?<!\b[A-Z]\.)"
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])" + _ABBREV + r"\s+(?=[A-Z0-9(\[\"“])")


@dataclass
class Chunk:
    text: str
    page: int
    page_end: int
    section: str


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _hard_wrap(sentence: str, limit: int) -> list[str]:
    """Break an over-long 'sentence' (tables, formula dumps) at whitespace."""
    if len(sentence) <= limit:
        return [sentence]
    out, cur = [], ""
    for word in sentence.split():
        if cur and len(cur) + len(word) + 1 > limit:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_pages(
    pages: list[CleanPage],
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
    min_chunk_chars: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading_stack: dict[int, str] = {}

    cur: list[tuple[str, int]] = []  # (sentence, page)
    cur_len = 0
    cur_section = ""

    def section_label() -> str:
        levels = sorted(heading_stack)
        return " > ".join(heading_stack[l] for l in levels[-2:])

    def emit(carry_overlap: bool) -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        text = " ".join(s for s, _ in cur).strip()
        if len(text) >= min_chunk_chars or not chunks:
            chunks.append(Chunk(text=text, page=cur[0][1], page_end=cur[-1][1], section=cur_section))
        elif chunks:  # tiny tail: append to the previous chunk instead of dropping it
            prev = chunks[-1]
            chunks[-1] = Chunk(text=f"{prev.text} {text}", page=prev.page, page_end=cur[-1][1], section=prev.section)
        if carry_overlap and overlap_chars > 0:
            kept: list[tuple[str, int]] = []
            total = 0
            for sent, pg in reversed(cur):
                if total + len(sent) > overlap_chars and kept:
                    break
                if len(sent) > overlap_chars * 1.5:
                    break
                kept.insert(0, (sent, pg))
                total += len(sent) + 1
            cur = kept
            cur_len = sum(len(s) + 1 for s, _ in cur)
        else:
            cur, cur_len = [], 0

    for page in pages:
        if page.kind != "content":
            continue
        for block in page.blocks:
            if block.kind == "heading":
                # Close the running chunk at a section boundary if it has substance.
                if cur_len >= min_chunk_chars:
                    emit(carry_overlap=False)
                for lvl in [l for l in heading_stack if l >= block.level]:
                    del heading_stack[lvl]
                heading_stack[block.level] = block.text.strip()
                if not cur:
                    cur_section = section_label()
                continue
            for sentence in split_sentences(block.text):
                for piece in _hard_wrap(sentence, chunk_chars):
                    if cur_len + len(piece) + 1 > chunk_chars and cur_len > 0:
                        emit(carry_overlap=True)
                        cur_section = section_label()
                    if not cur:
                        cur_section = section_label()
                    cur.append((piece, page.page))
                    cur_len += len(piece) + 1
    emit(carry_overlap=False)
    return chunks
