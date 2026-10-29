"""Tools the agent can call, their JSON schemas, and a citation registry.

Every passage or calculator result handed to the model gets a short tag
([S1], [S2], ... for passages; [C1], ... for calculations) that the model
cites inline. The registry maps tags back to the underlying chunk or
calculation so the app can show, verify and export them.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .index_store import KnowledgeIndex
from .retrieval import Passage, Retriever
from .stats_tools import CALCULATORS, CalcResult

# --------------------------------------------------------------------------
# Citation registry
# --------------------------------------------------------------------------


@dataclass
class SourceRef:
    tag: str
    chunk: dict

    @property
    def label(self) -> str:
        c = self.chunk
        pages = f"p.{c['page']}" if c["page"] == c["page_end"] else f"pp.{c['page']}-{c['page_end']}"
        return f"{c['category']}/{c['source']}, {pages}"

    def to_dict(self) -> dict:
        c = self.chunk
        return {"tag": self.tag, "label": self.label, "source": c["source"], "category": c["category"],
                "doc_id": c["doc_id"], "page": c["page"], "page_end": c["page_end"],
                "section": c.get("section", ""), "text": c["text"]}


@dataclass
class CalcRef:
    tag: str
    result: CalcResult

    def to_dict(self) -> dict:
        return {"tag": self.tag, "text": self.result.to_text(), **self.result.to_dict()}


@dataclass
class Citations:
    sources: list[SourceRef] = field(default_factory=list)
    calcs: list[CalcRef] = field(default_factory=list)
    _by_chunk: dict[str, SourceRef] = field(default_factory=dict)

    def add_chunk(self, chunk: dict) -> SourceRef:
        if chunk["id"] in self._by_chunk:
            return self._by_chunk[chunk["id"]]
        ref = SourceRef(tag=f"S{len(self.sources) + 1}", chunk=chunk)
        self.sources.append(ref)
        self._by_chunk[chunk["id"]] = ref
        return ref

    def add_calc(self, result: CalcResult) -> CalcRef:
        ref = CalcRef(tag=f"C{len(self.calcs) + 1}", result=result)
        self.calcs.append(ref)
        return ref

    def valid_tags(self) -> set[str]:
        return {s.tag for s in self.sources} | {c.tag for c in self.calcs}


_TAG_GROUP = re.compile(r"\[((?:[SC]\d+)(?:\s*[,;]\s*[SC]?\d+)*)\]")


def normalize_citations(text: str, valid: set[str]) -> tuple[str, list[str], list[str]]:
    """Normalise ``[S1, S3]`` -> ``[S1][S3]``, drop tags that were never issued.

    Returns (clean_text, cited_tags_in_order, invalid_tags).
    """
    cited: list[str] = []
    invalid: list[str] = []

    def repl(m: re.Match) -> str:
        parts = re.split(r"\s*[,;]\s*", m.group(1))
        prefix = "S"
        out = []
        for p in parts:
            if p[0] in "SC":
                prefix = p[0]
                tag = p
            else:
                tag = prefix + p
            if tag in valid:
                out.append(f"[{tag}]")
                if tag not in cited:
                    cited.append(tag)
            else:
                invalid.append(tag)
        return "".join(out)

    return _TAG_GROUP.sub(repl, text or ""), cited, invalid


# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


def tool_schemas(categories: list[str]) -> list[dict]:
    cat_prop = {"type": "array", "items": {"type": "string", "enum": categories} if categories else {"type": "string"},
                "description": "Optional: restrict to these knowledge-bank categories."}
    num = {"type": "number"}
    return [
        _fn(
            "search_knowledge_bank",
            "Hybrid keyword + semantic search over the indexed knowledge bank (regulatory guidance, statistics "
            "textbooks, methods papers). Returns tagged passages [S#] with document and page. Use focused queries; "
            "run several searches for multi-part questions.",
            {
                "query": {"type": "string", "description": "Search query - specific terms work best."},
                "categories": cat_prop,
                "documents": {"type": "array", "items": {"type": "string"},
                              "description": "Optional: restrict to documents (file names or distinctive parts of them)."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Passages to return (default 5)."},
            },
            ["query"],
        ),
        _fn("list_documents", "List the documents in the knowledge bank (optionally one category).",
            {"category": {"type": "string"}}),
        _fn(
            "read_page",
            "Read the full indexed text of one page of a document - use when a passage is cut off or you need the "
            "surrounding context.",
            {"document": {"type": "string", "description": "File name (or distinctive part of it)."},
             "page": {"type": "integer", "minimum": 1}},
            ["document", "page"],
        ),
        _fn(
            "continuous_sample_size",
            "Sample size (omit n) or power (give n) for a continuous endpoint. Designs: parallel (sd = between-"
            "subject SD), crossover_2x2 (sd = within-subject SD; n = total subjects), paired (sd = SD of paired "
            "differences; e.g. contralateral-eye designs). Hypotheses: superiority, noninferiority (higher is "
            "better; margin > 0), equivalence (margin = symmetric bound, or lower_margin/upper_margin). For ratio/"
            "bioequivalence-type endpoints use scale='ratio' with gmr and cv (margins as ratios, default 0.8-1.25).",
            {
                "design": {"type": "string", "enum": ["parallel", "crossover_2x2", "paired"]},
                "hypothesis": {"type": "string", "enum": ["superiority", "noninferiority", "equivalence"]},
                "delta": {**num, "description": "True difference treatment - control."},
                "sd": num, "alpha": num, "power": num,
                "n": {"type": "integer", "description": "Give to compute power. Parallel: control group size."},
                "margin": num, "lower_margin": num, "upper_margin": num,
                "ratio": {**num, "description": "Allocation treatment:control (parallel only)."},
                "sides": {"type": "integer", "enum": [1, 2]},
                "scale": {"type": "string", "enum": ["difference", "ratio"]},
                "gmr": num, "cv": num,
            },
            ["design", "hypothesis"],
        ),
        _fn(
            "binary_sample_size",
            "Sample size (omit n_control) or power for two independent proportions (risk difference). "
            "Superiority: pooled z-test (optional continuity correction). Non-inferiority: Wald test, margin > 0.",
            {
                "p_control": num, "p_treatment": num,
                "hypothesis": {"type": "string", "enum": ["superiority", "noninferiority"]},
                "alpha": num, "power": num, "n_control": {"type": "integer"}, "margin": num, "ratio": num,
                "continuity_correction": {"type": "boolean"},
            },
            ["p_control", "p_treatment"],
        ),
        _fn(
            "correlated_eyes_adjustment",
            "Design effect for ophthalmic trials where both eyes of a subject get the SAME treatment: "
            "DE = 1 + (m-1)*ICC. Give n_independent_eyes to get subjects needed, or n_subjects to get effective n.",
            {"n_independent_eyes": {"type": "integer"}, "n_subjects": {"type": "integer"},
             "eyes_per_subject": num, "icc": num},
            ["icc"],
        ),
        _fn("inflate_for_dropout", "Inflate an evaluable sample size for expected dropout.",
            {"n": {"type": "integer"}, "dropout_rate": num,
             "multiple_of": {"type": "integer", "description": "Round up to a multiple (e.g. sequences x block)."}},
            ["n", "dropout_rate"]),
        _fn(
            "adjust_pvalues",
            "Multiplicity-adjusted p-values: bonferroni, holm, hochberg, bh (FDR), fixed_sequence (order given).",
            {"pvalues": {"type": "array", "items": num},
             "method": {"type": "string", "enum": ["bonferroni", "holm", "hochberg", "bh", "fixed_sequence"]},
             "alpha": num, "labels": {"type": "array", "items": {"type": "string"}}},
            ["pvalues"],
        ),
    ]


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def format_passage(ref: SourceRef, max_chars: int | None = None) -> str:
    c = ref.chunk
    head = f"[{ref.tag}] {ref.label}"
    if c.get("section"):
        head += f" | section: {c['section']}"
    text = c["text"] if not max_chars else c["text"][:max_chars]
    return f"{head}\n{text}"


@dataclass
class ToolExecutor:
    retriever: Retriever
    citations: Citations
    default_top_k: int = 5
    default_categories: list[str] | None = None
    default_sources: list[str] | None = None
    char_limit: int = 9000
    log: list[dict] = field(default_factory=list)

    @property
    def index(self) -> KnowledgeIndex:
        return self.retriever.index

    def run(self, name: str, raw_args: str | dict) -> str:
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments were not valid JSON ({exc}). Retry with valid JSON."
        args = {k: v for k, v in args.items() if v is not None}
        entry = {"tool": name, "args": args}
        t0 = time.perf_counter()
        try:
            if name == "search_knowledge_bank":
                out = self._search(**args)
            elif name == "list_documents":
                out = self._list_documents(**args)
            elif name == "read_page":
                out = self._read_page(**args)
            elif name in CALCULATORS:
                result = CALCULATORS[name](**args)
                ref = self.citations.add_calc(result)
                out = f"[{ref.tag}]\n{result.to_text()}"
            else:
                out = f"ERROR: unknown tool '{name}'."
        except TypeError as exc:
            out = f"ERROR: bad arguments for {name}: {exc}"
        except ValueError as exc:
            out = f"ERROR: {exc}"
        entry["summary"] = out.splitlines()[0][:160] if out else ""
        entry["seconds"] = round(time.perf_counter() - t0, 2)
        if name == "search_knowledge_bank":
            entry["rerank_s"] = round(self.retriever.last_timing.get("rerank_s", 0.0), 2)
        self.log.append(entry)
        if len(out) > self.char_limit:
            out = out[: self.char_limit] + "\n[... truncated]"
        return out

    def _search(self, query: str, categories: list[str] | None = None, documents: list[str] | None = None,
                top_k: int | None = None) -> str:
        cats = categories or self.default_categories
        srcs = documents or self.default_sources
        k = max(1, min(int(top_k or self.default_top_k), 10))
        passages: list[Passage] = self.retriever.search(query, top_k=k, categories=cats, sources=srcs)
        if not passages and documents:
            passages = self.retriever.search(query, top_k=k, categories=cats)  # unknown document name: widen
        if not passages:
            return "No relevant passages found in the knowledge bank."
        per_passage = max(400, self.char_limit // max(len(passages), 1) - 120)
        return "\n\n---\n\n".join(format_passage(self.citations.add_chunk(p.chunk), per_passage) for p in passages)

    def _list_documents(self, category: str | None = None) -> str:
        docs = self.index.documents
        rows = [
            f"- {d['category']}/{d['source']} ({d.get('pages', '?')} pages)"
            + (f" - {d['title']}" if d.get("title") and d["title"].lower() not in d["source"].lower() else "")
            for d in sorted(docs.values(), key=lambda d: (d["category"], d["source"]))
            if not category or d["category"].lower() == category.lower()
        ]
        return "\n".join(rows) or "No documents found."

    def _read_page(self, document: str, page: int) -> str:
        doc_id = self.index.resolve_source(document)
        if not doc_id:
            return f"ERROR: no document matching '{document}'. Use list_documents."
        chunks = self.index.page_text(self.index.documents[doc_id]["source"], int(page))
        if not chunks:
            return f"No indexed text on page {page} of {doc_id} (it may be a table of contents, index or references page)."
        return "\n\n".join(format_passage(self.citations.add_chunk(c)) for c in chunks)
