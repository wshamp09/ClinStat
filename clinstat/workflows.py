"""Structured workflows: guidance comparison and section checking."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .llm import GroqChat, friendly_error
from .retrieval import Retriever
from .tools import Citations, format_passage, normalize_citations

CHECKLIST_DIR = Path(__file__).parent / "checklists"
STATUSES = ("met", "partial", "missing", "not_applicable")


@dataclass
class WorkflowResult:
    kind: str
    title: str
    body: str
    citations: Citations
    items: list[dict] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    model: str = ""
    fallbacks: list[dict] = field(default_factory=list)
    error: str | None = None
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_record(self) -> dict:
        _, cited, _ = normalize_citations(self.body + json.dumps(self.items), self.citations.valid_tags())
        cited_set = set(cited)
        return {
            "type": self.kind,
            "title": self.title,
            "question": self.inputs.get("topic") or self.inputs.get("checklist_title", ""),
            "body": self.body,
            "items": self.items,
            "inputs": self.inputs,
            "sources": [dict(s.to_dict(), cited=s.tag in cited_set) for s in self.citations.sources],
            "calcs": [],
            "model": self.model,
            "fallbacks": self.fallbacks,
            "created": self.created,
            "error": self.error,
        }


# --------------------------------------------------------------------------
# Compare guidance
# --------------------------------------------------------------------------

COMPARE_PROMPT = """Compare what the following documents say about: "{topic}".

Use ONLY the tagged passages below for document-specific statements and cite them inline ([S#]). \
If a document's passages do not address an aspect, write "Not addressed in retrieved passages" \
(do not guess what the document says).

Write in markdown:
1. **Overview** - 2-4 sentences.
2. **Comparison table** - rows are 4-8 aspects that matter for this topic; one column per document \
(use these short names: {short_names}); concise cells with citations.
3. **Where they agree** - bullets.
4. **Differences and tensions** - bullets, noting draft vs final status and dates where relevant.
5. **Implications for a trial statistician** - 2-5 practical bullets (e.g. what to write in a protocol or SAP).

Passages:
{passages}"""


def _short_name(source: str) -> str:
    stem = Path(source).stem
    return re.sub(r"[_\-]+", " ", stem)[:40]


def compare_documents(
    topic: str,
    documents: list[str],
    retriever: Retriever,
    llm: GroqChat,
    per_doc_k: int = 4,
    passage_chars: int = 900,
    reasoning_effort: str | None = "medium",
) -> WorkflowResult:
    citations = Citations()
    blocks, names = [], []
    index = retriever.index
    for doc in documents:
        doc_id = index.resolve_source(doc)
        if not doc_id:
            continue
        source = index.documents[doc_id]["source"]
        names.append(_short_name(source))
        passages = retriever.search(topic, top_k=per_doc_k, sources=[source])
        blocks.append(f"### Document: {_short_name(source)} ({doc_id})")
        if not passages:
            blocks.append("(no relevant passages retrieved)")
        for p in passages:
            blocks.append(format_passage(citations.add_chunk(p.chunk), passage_chars))
    title = f"Comparison: {topic}"
    inputs = {"topic": topic, "documents": documents, "per_doc_k": per_doc_k}
    if len(names) < 2:
        return WorkflowResult("comparison", title, "", citations, inputs=inputs, model=llm.model, fallbacks=list(getattr(llm, "fallbacks", [])),
                              error="Pick at least two documents that are in the index.")
    prompt = COMPARE_PROMPT.format(topic=topic, short_names="; ".join(names), passages="\n\n".join(blocks))
    try:
        msg = llm.complete(
            [{"role": "system", "content": "You are a precise regulatory and biostatistics analyst."},
             {"role": "user", "content": prompt}],
            reasoning_effort=reasoning_effort,
        )
        body, _, _ = normalize_citations(msg.content or "", citations.valid_tags())
        error = None if body.strip() else "The model returned an empty comparison."
    except Exception as exc:
        body, error = "", friendly_error(exc)
    return WorkflowResult("comparison", title, body, citations, inputs=inputs, model=llm.model, fallbacks=list(getattr(llm, "fallbacks", [])), error=error)


# --------------------------------------------------------------------------
# Section checker
# --------------------------------------------------------------------------


def load_checklists() -> dict[str, dict]:
    out = {}
    for path in sorted(CHECKLIST_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        out[data["id"]] = data
    return out


CHECK_PROMPT = """You are reviewing a draft {doc_type} section against a regulatory checklist.

Checklist: {checklist_title}

For EACH checklist item decide whether the draft section satisfies it:
- "met": clearly and specifically addressed
- "partial": mentioned but vague, incomplete or inconsistent
- "missing": not addressed
- "not_applicable": genuinely does not apply to this section/design (say why in "gap")

Rules: base the status only on the draft text. "evidence" must be a short verbatim quote from the draft \
(or "" if none). Use the guidance passages ([S#]) to justify "gap"/"suggestion" and list the tags you used in \
"citations". Suggestions should be concrete wording the author could add.

Return JSON only, exactly in this shape:
{{"summary": "<3-5 sentence overall assessment>",
  "items": [{{"id": "<item id>", "status": "met|partial|missing|not_applicable", "evidence": "...", \
"gap": "...", "suggestion": "...", "citations": ["S1"]}}]}}

Checklist items:
{items}

Guidance passages:
{passages}

Draft section:
\"\"\"
{text}
\"\"\""""


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def check_section(
    text: str,
    checklist: dict,
    retriever: Retriever,
    llm: GroqChat,
    passages_per_item: int = 1,
    passage_chars: int = 600,
    max_text_chars: int = 12000,
    reasoning_effort: str | None = "medium",
) -> WorkflowResult:
    citations = Citations()
    notes = []
    if len(text) > max_text_chars:
        notes.append(f"Section truncated to the first {max_text_chars:,} characters.")
        text = text[:max_text_chars]

    item_lines, passage_blocks = [], []
    for item in checklist["items"]:
        item_lines.append(f"- {item['id']}: {item['title']} - {item['requirement']}")
        hits = retriever.search(item["query"], top_k=passages_per_item, sources=item.get("documents") or None)
        if not hits and item.get("documents"):
            hits = retriever.search(item["query"], top_k=passages_per_item)
        for p in hits:
            ref = citations.add_chunk(p.chunk)
            block = format_passage(ref, passage_chars)
            if block not in passage_blocks:
                passage_blocks.append(block)

    prompt = CHECK_PROMPT.format(
        doc_type=checklist.get("doc_type", "protocol/SAP"),
        checklist_title=checklist["title"],
        items="\n".join(item_lines),
        passages="\n\n".join(passage_blocks) or "(no guidance passages retrieved)",
        text=text,
    )
    title = f"Section check: {checklist['title']}"
    inputs = {"checklist_id": checklist["id"], "checklist_title": checklist["title"], "chars": len(text)}
    try:
        msg = llm.complete(
            [{"role": "system", "content": "You are a meticulous regulatory biostatistics reviewer. Reply with JSON only."},
             {"role": "user", "content": prompt}],
            json_mode=True,
            reasoning_effort=reasoning_effort,
        )
        data = _parse_json(msg.content)
    except Exception as exc:
        return WorkflowResult("checklist", title, "", citations, inputs=inputs, model=llm.model, fallbacks=list(getattr(llm, "fallbacks", [])), error=friendly_error(exc))

    valid = citations.valid_tags()
    by_id = {str(i.get("id")): i for i in data.get("items", []) if isinstance(i, dict)}
    items = []
    for item in checklist["items"]:
        got = by_id.get(item["id"], {})
        status = str(got.get("status", "")).lower().replace(" ", "_")
        items.append({
            "id": item["id"],
            "title": item["title"],
            "requirement": item["requirement"],
            "status": status if status in STATUSES else "not_assessed",
            "evidence": got.get("evidence", ""),
            "gap": normalize_citations(got.get("gap", ""), valid)[0],
            "suggestion": normalize_citations(got.get("suggestion", ""), valid)[0],
            "citations": [c for c in got.get("citations", []) if c in valid],
        })
    summary = normalize_citations(str(data.get("summary", "")), valid)[0]
    counts = {s: sum(i["status"] == s for i in items) for s in STATUSES + ("not_assessed",)}
    tally = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v)
    body = f"**Result:** {tally}.\n\n{summary}"
    if notes:
        body += "\n\n_" + " ".join(notes) + "_"
    return WorkflowResult("checklist", title, body, citations, items=items, inputs=inputs, model=llm.model, fallbacks=list(getattr(llm, "fallbacks", [])))
