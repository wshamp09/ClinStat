"""The ClinStat research agent: a Groq tool-calling loop with verified citations."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .config import Settings, settings as default_settings
from .llm import GroqChat
from .retrieval import Retriever
from .tools import Citations, ToolExecutor, normalize_citations, tool_schemas

EventFn = Callable[[dict], None]

SYSTEM_PROMPT = """You are ClinStat, a research assistant for a biostatistician who designs and analyses \
clinical trials (often ophthalmic / contact lens trials, including crossover and contralateral-eye designs). \
You have tools to search a curated knowledge bank and vetted statistical calculators.

How to work
- Search the knowledge bank for anything about regulatory expectations, definitions, methods or study design \
that the documents may cover. Use several focused searches for multi-part questions (e.g. one per guidance or \
concept). Use the `documents` filter when the user names a document. Use read_page if a passage is cut off.
- For ANY number that needs computing (sample size, power, adjusted p-values, design effects, dropout \
inflation) call a calculator - never do the arithmetic yourself. If an input is missing, choose a sensible \
value, say so, and show how the answer changes if that matters.
- Cite as you write: put the tag right after the claim it supports, e.g. "... must be pre-specified [S2]." \
Calculator results are cited as [C1]. Only use tags you were given. Do not add a reference list at the end - \
the app shows the sources.
- Write one cohesive answer that blends the documents with your own expertise. Statements from general \
knowledge carry no tag. If the knowledge bank does not cover the question, say so in one sentence and answer \
from general knowledge.
- Regulatory care: say whether a guidance is draft or final and give its year when it matters; distinguish \
FDA, ICH and EMA positions; never invent guidance wording, section numbers or page numbers.
- Be precise and concise. Use headings, bullets or a table only when they help. Formulas may use inline \
LaTeX ($...$).
- This is a multi-turn conversation: use earlier turns to resolve follow-up questions."""


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: Citations
    cited: list[str] = field(default_factory=list)
    invalid_tags: list[str] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    model: str = ""
    timings: dict = field(default_factory=dict)
    fallbacks: list[dict] = field(default_factory=list)
    error: str | None = None
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def used_search(self) -> bool:
        return bool(self.citations.sources)

    def to_record(self) -> dict:
        """Serializable form used by the UI and exports."""
        cited = set(self.cited)
        return {
            "type": "answer",
            "title": self.question[:90],
            "question": self.question,
            "body": self.answer,
            "sources": [dict(s.to_dict(), cited=s.tag in cited) for s in self.citations.sources],
            "calcs": [dict(c.to_dict(), cited=c.tag in cited) for c in self.citations.calcs],
            "tool_log": self.tool_log,
            "invalid_tags": self.invalid_tags,
            "model": self.model,
            "timings": self.timings,
            "fallbacks": self.fallbacks,
            "created": self.created,
            "error": self.error,
        }


def _catalog(retriever: Retriever, limit: int = 40) -> str:
    docs = retriever.index.documents
    if not docs or len(docs) > limit:
        return f"The knowledge bank holds {len(docs)} documents; call list_documents to see them."
    lines = [f"- {d['category']}/{d['source']}" for d in sorted(docs.values(), key=lambda d: (d["category"], d["source"]))]
    return "Documents in the knowledge bank:\n" + "\n".join(lines)


def _assistant_message(msg) -> dict:
    """Convert an SDK message with tool calls into a plain dict we can send back."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in msg.tool_calls
        ],
    }


def run_agent(
    question: str,
    retriever: Retriever,
    llm: GroqChat,
    history: list[dict] | None = None,
    categories: list[str] | None = None,
    documents: list[str] | None = None,
    top_k: int | None = None,
    reasoning_effort: str | None = "medium",
    on_event: EventFn | None = None,
    cfg: Settings | None = None,
) -> AgentResult:
    cfg = cfg or default_settings
    citations = Citations()
    executor = ToolExecutor(
        retriever=retriever,
        citations=citations,
        default_top_k=top_k or cfg.top_k,
        default_categories=categories or None,
        default_sources=documents or None,
        char_limit=cfg.tool_result_char_limit,
    )
    scope = ""
    if categories or documents:
        scope = "\n\nThe user restricted searches to: " + ", ".join((categories or []) + (documents or [])) + "."
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + _catalog(retriever) + scope}]
    for turn in (history or [])[-2 * cfg.history_turns:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    tools = tool_schemas(retriever.index.categories)

    def emit(event: dict) -> None:
        if on_event:
            try:
                on_event(event)
            except Exception:
                pass

    if hasattr(llm, "on_fallback"):
        llm.on_fallback = lambda f: emit({"type": "fallback", **f})

    answer = ""
    t_start = time.perf_counter()
    model_s = 0.0

    def ask(tool_choice: str):
        nonlocal model_s
        t = time.perf_counter()
        try:
            return llm.complete(messages, tools=tools, tool_choice=tool_choice, reasoning_effort=reasoning_effort)
        finally:
            model_s += time.perf_counter() - t

    try:
        for round_no in range(cfg.max_tool_rounds):
            msg = ask("auto")
            if not getattr(msg, "tool_calls", None):
                answer = msg.content or ""
                break
            messages.append(_assistant_message(msg))
            for tc in msg.tool_calls:
                emit({"type": "tool_start", "tool": tc.function.name, "args": tc.function.arguments, "round": round_no + 1})
                result = executor.run(tc.function.name, tc.function.arguments)
                emit({"type": "tool_end", "tool": tc.function.name, "summary": executor.log[-1].get("summary", ""),
                      "seconds": executor.log[-1].get("seconds", 0.0)})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            emit({"type": "final", "detail": "tool budget used - writing the answer"})
            msg = ask("none")
            answer = msg.content or ""
        if not answer.strip():
            raise RuntimeError("The model returned an empty answer. Try rephrasing or lowering reasoning effort.")
        error = None
    except Exception as exc:  # surfaced in the UI rather than crashing the app
        from .llm import friendly_error

        error = friendly_error(exc)
        answer = answer or ""

    tool_s = sum(e.get("seconds", 0.0) for e in executor.log)
    timings = {
        "total_s": round(time.perf_counter() - t_start, 1),
        "model_s": round(model_s, 1),
        "search_s": round(sum(e.get("seconds", 0.0) for e in executor.log if e["tool"] == "search_knowledge_bank"), 1),
        "rerank_s": round(sum(e.get("rerank_s", 0.0) for e in executor.log), 1),
        "tools_s": round(tool_s, 1),
    }
    clean, cited, invalid = normalize_citations(answer, citations.valid_tags())
    return AgentResult(
        question=question,
        answer=clean,
        citations=citations,
        cited=cited,
        invalid_tags=invalid,
        tool_log=executor.log,
        model=llm.model,
        timings=timings,
        fallbacks=list(getattr(llm, "fallbacks", [])),
        error=error,
    )
