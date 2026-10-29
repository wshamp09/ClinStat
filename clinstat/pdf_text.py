"""PDF text extraction and cleaning.

pypdf returns one string per page with a hard line break at the end of every
visual line and almost never a blank line between paragraphs. This module turns
that into clean paragraph blocks and classifies each page so that tables of
contents, back-of-book indexes and bibliographies can be left out of the index
(they match almost any query lexically and crowd out real content).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class Block:
    """A paragraph, heading or list item within a page."""

    text: str
    kind: str = "para"  # "para" | "heading"
    level: int = 0      # heading depth (1 = top level); 0 for paragraphs


@dataclass
class CleanPage:
    page: int                       # 1-based page number in the PDF
    kind: str                       # "content" | "toc" | "index" | "references" | "blank"
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


@dataclass
class ExtractedDocument:
    path: Path
    title: str
    n_pages: int
    pages: list[CleanPage]
    total_chars: int
    warning: str | None = None

    @property
    def content_pages(self) -> list[CleanPage]:
        return [p for p in self.pages if p.kind == "content"]

    def skipped_page_counts(self) -> dict[str, int]:
        return dict(Counter(p.kind for p in self.pages if p.kind != "content"))


# --------------------------------------------------------------------------
# Regular expressions
# --------------------------------------------------------------------------

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_BULLET = re.compile(r"^\s*(?:[•●▪◦‣\-–]|\(?[a-z]\)|\(?[ivx]{1,4}\)|\d{1,2}\))\s+")
_NUMBERED_HEADING = re.compile(
    r"^(?P<num>(?:[A-Z]\.)?\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(?P<title>[A-Z][^\n]{2,90})$"
)
_ROMAN_HEADING = re.compile(r"^(?P<num>[IVX]{1,5}|[A-H])\.\s+(?P<title>[A-Z][^\n]{2,90})$")
_TOC_LINE = re.compile(r"(?:\.\s?){4,}\s*\d{1,4}\s*$|\s{2,}\d{1,4}\s*$")
_INDEX_LINE = re.compile(r"[A-Za-z)\]]\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?)*\s*$")
_TRAILING_LINE_NO = re.compile(r"^(?P<body>.*\S)\s+(?P<no>\d{1,4})\s*$")
_LOWER_TAIL = {"the", "a", "an", "of", "and", "or", "to", "for", "in", "with", "using", "by", "on", "at", "from", "is", "are", "be"}
_YEAR = re.compile(r"\(?(?:19|20)\d{2}[a-z]?\)?[.,;:]")
_REF_START = re.compile(
    r"^(?:\[\d{1,3}\]|\d{1,3}\.\s+[A-Z]|[A-Z][A-Za-z'’\-]+,\s+(?:[A-Z]\.\s?-?){1,3}|[A-Z][A-Za-z'’\-]+\s+[A-Z]{1,3}[,.])"
)
_REF_MARKERS = re.compile(r"et al\.|J(?:ournal)?\b|Stat(?:istics)?\b|Biometri|\d+\s*[:(]\s*\d+|pp\.\s*\d|doi", re.I)
_RUNNING_HEAD = re.compile(r"^(?:\d{1,4}\s+[A-Z][A-Z0-9 ,:×\-–'’()&]{3,}|[A-Z][A-Z0-9 ,:×\-–'’()&]{3,}\s+\d{1,4})$")
_PAGE_NUMBER_ONLY = re.compile(r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.I)
_SENTENCE_END = re.compile(r"[.!?:;)\]\"”]$")


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def read_pdf_pages(path: Path) -> tuple[list[str], str]:
    """Return (page_texts, title). Raises ValueError for unreadable files."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # corrupt file
        raise ValueError(f"could not open PDF ({exc.__class__.__name__}: {exc})") from exc
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise ValueError("PDF is password-protected")
        except Exception as exc:
            raise ValueError(f"PDF is encrypted ({exc})") from exc

    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")

    title = ""
    try:
        meta_title = (reader.metadata or {}).get("/Title")
        if meta_title and len(str(meta_title).strip()) > 4:
            title = str(meta_title).strip()
    except Exception:
        pass
    return texts, title


def title_from_filename(path: Path) -> str:
    stem = re.sub(r"[_\-]+", " ", path.stem)
    return re.sub(r"\s+", " ", stem).strip()


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("­", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _line_signature(line: str) -> str:
    """Normalise a line so that running headers differing only by page number match."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_repeated_margin_lines(pages_lines: list[list[str]], margin: int = 3) -> set[str]:
    """Signatures of lines that recur in the top/bottom margin of many pages."""
    n = len(pages_lines)
    if n < 4:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        sigs = {_line_signature(l) for l in lines[:margin] + lines[-margin:] if l.strip()}
        counts.update(sigs)
    threshold = max(3, int(0.08 * n))
    return {sig for sig, c in counts.items() if c >= threshold and len(sig) > 1}


def _heading_level(line: str) -> int:
    """Return the heading depth if the line looks like a section heading, else 0."""
    s = line.strip()
    if len(s) < 4 or len(s) > 100 or s.endswith((",", ";")):
        return 0
    if _TOC_LINE.search(s):
        return 0
    m = _NUMBERED_HEADING.match(s)
    if m:
        title = m.group("title")
        # Reject sentences ("3 patients were ...") and numeric-looking rows from tables.
        if title.endswith(".") and len(title.split()) > 8:
            return 0
        if sum(ch.isdigit() for ch in title) > len(title) * 0.3:
            return 0
        if len(title.split()) > 10 or title.split()[-1].lower() in _LOWER_TAIL:
            return 0
        if re.search(r"[a-z]\.\s+[A-Z]", title):  # two sentences: prose, not a heading
            return 0
        if "www." in title or "http" in title or title.endswith("-"):
            return 0  # a numbered reference entry, not a heading
        # Depth ordering: ROMAN (1) > LETTER (2) > 1. (3) > 1.1 (4) > 1.1.1 (5) ...
        return min(m.group("num").count(".") + 3, 8)
    m = _ROMAN_HEADING.match(s)
    if m and not s.endswith(".") and len(m.group("title").split()) <= 10 and not re.search(r"[a-z]\.\s+[A-Z]", s):
        return 1 if re.fullmatch(r"[IVX]{1,5}", m.group("num")) else 2
    words = s.split()
    if (
        2 <= len(words) <= 8
        and s.isupper()
        and not any(ch.isdigit() for ch in s)
        and sum(c.isalpha() for c in s) >= 4
        and sum(len(w) for w in words) / len(words) >= 3
        and not s.endswith(".")
    ):
        return 1
    return 0


def strip_line_numbers(lines: list[str]) -> list[str]:
    """Remove the margin line numbers that FDA draft guidances print on every line.

    Applied only when most lines on the page end in an integer and those integers
    mostly increase, so ordinary text ending in a number is left alone.
    """
    nums = []
    for line in lines:
        m = _TRAILING_LINE_NO.match(line.strip())
        nums.append(int(m.group("no")) if m else None)
    found = [n for n in nums if n is not None]
    non_empty = sum(1 for l in lines if l.strip())
    if non_empty < 6 or len(found) < 0.5 * non_empty:
        return lines
    increasing = sum(1 for a, b in zip(found, found[1:]) if 0 < b - a <= 3)
    if increasing < 0.6 * (len(found) - 1):
        return lines
    out = []
    for line, n in zip(lines, nums):
        out.append(_TRAILING_LINE_NO.match(line.strip()).group("body") if n is not None else line)
    return out


def classify_page(lines: list[str]) -> str:
    content = [l.strip() for l in lines if l.strip()]
    if sum(len(l) for l in content) < 40:
        return "blank"
    n = len(content)
    toc = sum(bool(_TOC_LINE.search(l)) for l in content)
    if n >= 5 and toc / n >= 0.35:
        return "toc"
    if any(re.fullmatch(r"(table of )?contents", l.lower()) for l in content[:6]) and toc / n >= 0.2:
        return "toc"
    idx = sum(bool(_INDEX_LINE.search(l)) for l in content)
    if n >= 8 and idx / n >= 0.35:
        return "index"
    if n >= 8 and re.fullmatch(r"(subject |author )?index( \d+)?", content[0].lower()) and idx / n >= 0.2:
        return "index"
    years = sum(bool(_YEAR.search(l)) for l in content)
    starts = sum(bool(_REF_START.match(l)) for l in content)
    markers = sum(bool(_REF_MARKERS.search(l)) for l in content)
    head = " ".join(content[:3]).lower()
    ref_header = any(w in head for w in ("references", "bibliography", "literature cited"))
    if n >= 6:
        if years / n >= 0.3 and starts / n >= 0.2 and markers / n >= 0.25:
            return "references"
        if ref_header and years / n >= 0.2:
            return "references"
    return "content"


def _lines_to_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    para: list[str] = []
    typical = sorted(len(l) for l in lines if l.strip())
    typical_len = typical[int(len(typical) * 0.75)] if typical else 80

    def flush() -> None:
        if para:
            text = " ".join(para).strip()
            if text:
                blocks.append(Block(text=text))
            para.clear()

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            flush()
            continue
        level = _heading_level(line)
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if level and nxt[:1].islower() and len(line) > 0.6 * typical_len:
            level = 0  # a numbered list item whose sentence continues on the next line
        if level:
            flush()
            blocks.append(Block(text=line, kind="heading", level=level))
            continue
        if _BULLET.match(line):
            flush()
        if para and para[-1].endswith("-") and len(para[-1]) > 1 and para[-1][-2].isalpha() and line[:1].islower():
            para[-1] = para[-1][:-1] + line  # re-join a hyphenated word
        else:
            para.append(line)
        # A short line ending a sentence usually closes a paragraph.
        if _SENTENCE_END.search(line) and len(line) < 0.6 * typical_len:
            flush()
    flush()
    return blocks


def clean_pages(page_texts: list[str]) -> list[CleanPage]:
    pages_lines = [normalize_text(t).splitlines() for t in page_texts]
    repeated = find_repeated_margin_lines(pages_lines)

    cleaned: list[CleanPage] = []
    for i, lines in enumerate(pages_lines, start=1):
        kept = []
        for j, line in enumerate(lines):
            in_margin = j < 3 or j >= len(lines) - 3
            if in_margin and _line_signature(line) in repeated:
                continue
            if (j < 2 or j >= len(lines) - 2) and _RUNNING_HEAD.match(line.strip()):
                continue  # book running head: "18 THE 2 x 2 CROSS-OVER TRIAL"
            if _PAGE_NUMBER_ONLY.match(line):
                continue
            kept.append(line)
        kind = classify_page(kept)  # classify first: TOC page numbers look like line numbers
        if kind == "content":
            kept = strip_line_numbers(kept)
            kind = classify_page(kept)
        blocks = _lines_to_blocks(kept) if kind == "content" else []
        cleaned.append(CleanPage(page=i, kind=kind, blocks=blocks))
    return _mark_trailing_reference_runs(cleaned)


def _mark_trailing_reference_runs(pages: list[CleanPage]) -> list[CleanPage]:
    """A content page sandwiched between reference pages is almost always references too."""
    for i in range(1, len(pages) - 1):
        if pages[i].kind == "content" and pages[i - 1].kind == "references" and pages[i + 1].kind == "references":
            pages[i].kind = "references"
            pages[i].blocks = []
    return pages


def extract_document(path: Path, min_chars_per_page: int = 200) -> ExtractedDocument:
    texts, meta_title = read_pdf_pages(path)
    total = sum(len(t.strip()) for t in texts)
    title = meta_title or title_from_filename(path)
    warning = None
    if not texts:
        warning = "PDF has no pages"
    elif total / max(len(texts), 1) < min_chars_per_page:
        warning = (
            f"only {total / max(len(texts), 1):.0f} characters of text per page - likely scanned or "
            "image-based. Run OCR (e.g. `ocrmypdf in.pdf out.pdf`) and add the OCR'd copy."
        )
    pages = clean_pages(texts) if warning is None else []
    return ExtractedDocument(path=path, title=title, n_pages=len(texts), pages=pages, total_chars=total, warning=warning)
