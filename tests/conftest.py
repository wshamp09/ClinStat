"""Shared fixtures: a tiny Knowledge Bank of generated PDFs, an index built with
an offline hashing embedder, and a scripted stand-in for the Groq client."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clinstat.config import Settings
from clinstat.embeddings import HashingEmbedder
from clinstat.index_store import KnowledgeIndex, ingest
from clinstat.retrieval import Retriever

GUIDANCE_PAGES = [
    [
        "Contains Nonbinding Recommendations",
        "I. INTRODUCTION",
        "This guidance describes estimands for clinical trials. An estimand is a precise description of the 1",
        "treatment effect reflecting the clinical question posed by the trial objective. 2",
        "A. Intercurrent Events 3",
        "Intercurrent events are events occurring after treatment initiation that affect either the 4",
        "interpretation or the existence of the measurements. Discontinuation of treatment is an example. 5",
        "Rescue medication use is another common intercurrent event in ophthalmic studies. 6",
    ],
    [
        "Contains Nonbinding Recommendations",
        "B. Strategies 7",
        "Under the treatment policy strategy the occurrence of the intercurrent event is considered irrelevant 8",
        "in defining the treatment effect of interest. Under the hypothetical strategy a scenario is envisaged 9",
        "in which the intercurrent event would not occur. 10",
        "C. Sensitivity Analysis 11",
        "A sensitivity analysis explores the robustness of inferences from the main estimator to deviations 12",
        "from its underlying modelling assumptions and limitations in the data. 13",
    ],
]

BOOK_PAGES = [
    [
        "2.7 Consequences of preliminary testing",
        "Grizzle proposed a two-stage procedure in which a preliminary test for carryover determines whether",
        "both periods or only the first period are analysed. The two-stage procedure inflates the type I",
        "error rate and is no longer recommended for the AB/BA cross-over trial.",
        "2.8 Washout periods",
        "An adequate washout period between treatment periods removes the need to model carryover in most",
        "contact lens studies, where lens effects do not persist after removal of the lens.",
    ],
    [
        "References",
        "Grizzle, J. E. (1965). The two-period change-over design and its use in clinical trials. Biometrics, 21:467-480.",
        "Senn, S. J. (1993). Cross-over Trials in Clinical Research. Wiley: Chichester.",
        "Freeman, P. (1989). The performance of the two-stage analysis of two-treatment, two-period crossover trials. Stat Med 8:1421-1432.",
        "Jones, B. and Kenward, M. G. (2015). Design and Analysis of Cross-Over Trials. CRC Press, 3rd edition.",
        "Hills, M. and Armitage, P. (1979). The two-period cross-over clinical trial. Br J Clin Pharmacol, 8:7-20.",
        "Kenward, M. G. and Roger, J. H. (2010). The use of baseline covariates in crossover studies. Biostatistics, 11:1-17.",
    ],
]


def make_pdf(path: Path, pages: list[list[str]], title: str | None = None) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    if title:
        c.setTitle(title)
    for lines in pages:
        y = 740
        for line in lines:
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()


@pytest.fixture()
def kb(tmp_path: Path) -> Path:
    root = tmp_path / "KnowledgeBank"
    make_pdf(root / "RegulatoryDocs" / "Estimand_Guidance_2020.pdf", GUIDANCE_PAGES, title="Estimands guidance")
    make_pdf(root / "StatBooks" / "Crossover_Book_2015.pdf", BOOK_PAGES)
    make_pdf(root / "StatPapers" / "scanned.pdf", [[""], [""]])  # no text layer
    make_pdf(root / "_Removed" / "ignored.pdf", GUIDANCE_PAGES)
    return root


@pytest.fixture()
def cfg(kb: Path, tmp_path: Path) -> Settings:
    return dataclasses.replace(Settings(), kb_dir=kb, index_dir=tmp_path / "index", use_reranker=False,
                               min_chars_per_page=50, min_chunk_chars=40)


@pytest.fixture()
def built(cfg: Settings) -> tuple[Settings, KnowledgeIndex, dict]:
    report = ingest(cfg, HashingEmbedder(), log=lambda m: None)
    return cfg, KnowledgeIndex.load(cfg.index_dir), report


@pytest.fixture()
def retriever(built) -> Retriever:
    cfg, index, _ = built
    return Retriever(index, HashingEmbedder(), None, cfg)


# ---------------------------------------------------------------- fake LLM
def tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def message(content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class FakeLLM:
    """Returns scripted messages in order and records every request."""

    model = "fake-model"

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], "tool_choice": tool_choice, **kwargs})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
