"""ClinStat - research agent for clinical trial statistics (Streamlit app).

    ./run_app.sh            (or: venv/bin/python -m streamlit run app_clinstat.py)
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="ClinStat", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------
# Start-up check: is this Python the project's venv, with the packages installed?
# A clear message here beats a ModuleNotFoundError from deep inside the package.
# --------------------------------------------------------------------------
_REQUIRED = {"fastembed": "fastembed", "groq": "groq", "pypdf": "pypdf", "numpy": "numpy", "scipy": "scipy",
             "pandas": "pandas", "docx": "python-docx", "onnxruntime": "onnxruntime"}
_missing = [pkg for mod, pkg in _REQUIRED.items() if importlib.util.find_spec(mod) is None]
if _missing:
    _here = Path(__file__).resolve().parent
    _venv_python = _here / "venv" / "bin" / "python"
    st.error("ClinStat can't start: required packages are missing from the Python that is running it.")
    st.markdown(
        f"- Running with: `{sys.executable}`\n"
        f"- Missing: {', '.join(f'`{m}`' for m in _missing)}\n"
        + (f"- Project venv found at `{_venv_python}`, but the app was **not started with it**.\n"
           if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve() else "")
    )
    st.markdown("**Fix:** stop the app (`Ctrl+C`) and start it with the project script, which always uses the venv:")
    st.code(f"cd \"{_here}\"\n./run_app.sh", language="bash")
    st.markdown("Or install the packages into the Python shown above:")
    st.code(f"\"{sys.executable}\" -m pip install --only-binary=:all: -r \"{_here / 'requirements.txt'}\"", language="bash")
    st.stop()

import pandas as pd  # noqa: E402

from clinstat import __version__  # noqa: E402
from clinstat.agent import run_agent  # noqa: E402
from clinstat.config import _setting, settings  # noqa: E402
from clinstat.embeddings import FastEmbedder  # noqa: E402
from clinstat.evaluation import evaluate, load_questions  # noqa: E402
from clinstat.export import records_to_docx, records_to_markdown  # noqa: E402
from clinstat.index_store import ingest, remove_document  # noqa: E402
from clinstat.index_store import KnowledgeIndex  # noqa: E402
from clinstat.llm import REASONING_EFFORTS, GroqChat, friendly_error, has_groq_api_key  # noqa: E402
from clinstat.retrieval import Retriever  # noqa: E402
from clinstat.runtime import load_reranker  # noqa: E402
from clinstat.stats_tools import (  # noqa: E402
    adjust_pvalues,
    binary_sample_size,
    continuous_sample_size,
    correlated_eyes_adjustment,
    inflate_for_dropout,
)
from clinstat.ui import render_record  # noqa: E402
from clinstat.workflows import check_section, compare_documents, load_checklists  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ==========================================================================
# Cached resources
# ==========================================================================


def _index_stamp() -> float:
    m = settings.index_dir / "manifest.json"
    return m.stat().st_mtime if m.exists() else 0.0


@st.cache_resource(show_spinner="Loading knowledge index ...")
def get_index(stamp: float) -> KnowledgeIndex:
    return KnowledgeIndex.load(settings.index_dir)


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it) ...")
def get_embedder(model_name: str) -> FastEmbedder:
    return FastEmbedder(model_name)


@st.cache_resource(show_spinner="Loading re-ranker ...")
def get_reranker():
    return load_reranker(settings)


@st.cache_resource(show_spinner="Building keyword index ...")
def get_retriever(stamp: float) -> Retriever:
    index = get_index(stamp)
    embedder = get_embedder(index.manifest.get("embed_model") or settings.embed_model)
    r = Retriever(index, embedder, get_reranker(), settings)
    _ = r.bm25  # build once, up front
    return r


def get_llm() -> GroqChat:
    return GroqChat()


def reset_index_caches() -> None:
    get_index.clear()
    get_retriever.clear()


# ==========================================================================
# Session state
# ==========================================================================

ss = st.session_state
ss.setdefault("chat", [])          # [{"role": "user", "content": str} | {"role": "assistant", "record": dict}]
ss.setdefault("compare", None)
ss.setdefault("check", None)
ss.setdefault("eval", None)
ss.setdefault("pending_question", None)

stamp = _index_stamp()
index = get_index(stamp)

# ==========================================================================
# Sidebar
# ==========================================================================

with st.sidebar:
    st.title("📊 ClinStat")
    st.caption(f"Research agent for clinical trial statistics · v{__version__}")

    if index.is_empty:
        st.error("The index is empty. Run `python ingest.py` locally (or use the Knowledge Bank tab).")
    elif not index.schema_ok():
        st.warning("The index was built by an older version. Rebuild it: `python ingest.py --rebuild`.")
    else:
        st.success(f"{len(index.documents)} documents · {len(index.chunks):,} passages")

    key_ok = has_groq_api_key()
    st.caption("Models: " + " → ".join(m.split("/")[-1] for m in settings.groq_models)
               + ("  (falls back on rate limits)" if len(settings.groq_models) > 1 else ""))
    if not key_ok:
        st.error("GROQ_API_KEY not found. Add it to Streamlit secrets or a local .env file.")

    st.subheader("Answer settings")
    reasoning = st.select_slider("Reasoning effort", options=list(REASONING_EFFORTS), value="medium",
                                 help="Higher = more thorough, slower and uses more tokens.")
    top_k = st.slider("Passages per search", 3, 10, settings.top_k,
                      help="Fewer passages keep requests under Groq's per-minute token limits.")
    rerank_available = get_reranker() is not None
    use_rerank = st.toggle("Cross-encoder re-ranking", value=rerank_available, disabled=not rerank_available,
                           help="Re-reads the top candidates with the query for a better final order.")

    st.subheader("Search scope")
    scope_cats = st.multiselect("Categories", index.categories, placeholder="All categories")
    doc_options = sorted(d["source"] for d in index.documents.values()
                         if not scope_cats or d["category"] in scope_cats)
    scope_docs = st.multiselect("Documents", doc_options, placeholder="All documents")

    st.divider()
    records = [m["record"] for m in ss.chat if m["role"] == "assistant"]
    if records:
        st.download_button("⬇ Conversation (Markdown)", lambda: records_to_markdown(records), file_name="clinstat_session.md",
                           mime="text/markdown", on_click="ignore")
        st.download_button("⬇ Conversation (Word)", lambda: records_to_docx(records), file_name="clinstat_session.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           on_click="ignore")
    if st.button("Clear conversation"):
        ss.chat = []
        st.rerun()

if not index.is_empty:
    retriever = get_retriever(stamp).with_settings(dataclasses.replace(settings, use_reranker=use_rerank))
else:
    retriever = None

tab_ask, tab_compare, tab_check, tab_calc, tab_kb, tab_eval = st.tabs(
    ["💬 Ask", "⚖️ Compare guidance", "✅ Check a section", "🧮 Calculators", "📚 Knowledge Bank", "📏 Evaluate"]
)

# ==========================================================================
# Ask
# ==========================================================================

EXAMPLES = [
    "Which intercurrent-event strategies does ICH E9(R1) describe, and which fits lens discontinuation in a comfort study?",
    "Should I test for carryover before analysing a 2x2 crossover? What do Jones & Kenward recommend?",
    "Sample size for a contralateral-eye study: difference 3 points, SD of differences 10, 90% power?",
    "When does the Hochberg procedure control the familywise error rate, per the FDA multiple endpoints guidance?",
]

with tab_ask:
    if retriever is None:
        st.info("Build the index first (Knowledge Bank tab or `python ingest.py`).")
    if not ss.chat:
        st.markdown("#### Ask about guidance, methods or design")
        st.caption("Answers cite the knowledge bank inline ([S1]) and use vetted calculators for numbers ([C1]).")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex_{i}", width="stretch"):
                ss.pending_question = ex
                st.rerun()

    for n, msg in enumerate(ss.chat):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_record(msg["record"], index, key=f"chat{n}")

    question = st.chat_input("Ask a question about your documents, methods or a design ...",
                             disabled=retriever is None or not key_ok)
    question = question or ss.pop("pending_question", None)
    if question and retriever is not None:
        ss.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Researching ...", expanded=False) as status:
                def on_event(e: dict) -> None:
                    if e["type"] == "tool_start":
                        try:
                            args = json.loads(e["args"] or "{}")
                        except Exception:
                            args = {}
                        detail = args.get("query") or args.get("document") or ", ".join(f"{k}={v}" for k, v in args.items())
                        status.write(f"**{e['tool'].replace('_', ' ')}** - {str(detail)[:140]}")
                    elif e["type"] == "tool_end":
                        status.write(f"   done in {e.get('seconds', 0):.1f} s")
                    elif e["type"] == "fallback":
                        nxt = e.get("to") or "none left"
                        status.write(f"⚠️ {e['from']} is rate-limited - switching to **{nxt}**")
                    elif e["type"] == "final":
                        status.write("Writing the answer ...")

                history = []
                for m in ss.chat[:-1]:
                    if m["role"] == "user":
                        history.append({"role": "user", "content": m["content"]})
                    elif m["record"].get("body"):
                        history.append({"role": "assistant", "content": m["record"]["body"]})
                try:
                    result = run_agent(question, retriever, get_llm(), history=history, categories=scope_cats,
                                       documents=scope_docs, top_k=top_k, reasoning_effort=reasoning, on_event=on_event)
                    record = result.to_record()
                except Exception as exc:
                    record = {"type": "answer", "title": question[:90], "question": question, "body": "",
                              "sources": [], "calcs": [], "error": friendly_error(exc)}
                status.update(label="Done" if not record.get("error") else "Problem", state="complete" if not record.get("error") else "error")
        ss.chat.append({"role": "assistant", "record": record})
        st.rerun()

# ==========================================================================
# Compare guidance
# ==========================================================================

with tab_compare:
    st.markdown("#### Compare what documents say about a topic")
    st.caption("Retrieves the most relevant passages from each document separately, then builds a cited comparison table.")
    c1, c2 = st.columns([0.6, 0.4])
    topic = c1.text_input("Topic", placeholder="e.g. handling of missing data; use of external controls; type I error control")
    all_docs = sorted(d["source"] for d in index.documents.values())
    default_docs = [d for d in all_docs if d.lower().startswith(("ich_e9", "fda_"))][:2]
    chosen = c2.multiselect("Documents (2-4)", all_docs, default=default_docs, max_selections=4)
    per_doc = st.slider("Passages per document", 2, 6, 4)
    if st.button("Compare", type="primary", disabled=not (topic and len(chosen) >= 2 and key_ok and retriever)):
        with st.spinner("Retrieving and comparing ..."):
            try:
                res = compare_documents(topic, chosen, retriever, get_llm(), per_doc_k=per_doc, reasoning_effort=reasoning)
                ss.compare = res.to_record()
            except Exception as exc:
                ss.compare = {"type": "comparison", "title": topic, "body": "", "sources": [], "error": friendly_error(exc)}
    if ss.compare:
        st.divider()
        render_record(ss.compare, index, key="compare")

# ==========================================================================
# Section checker
# ==========================================================================

with tab_check:
    st.markdown("#### Check a protocol / SAP section against guidance")
    st.caption("Each checklist item is graded Met / Partial / Missing with a quote from your draft, the gap, suggested "
               "wording, and the guidance passages behind it.")
    checklists = load_checklists()
    order = ["estimands_e9r1", "sap_core_e9", "multiple_endpoints", "pro_endpoint", "external_control",
             "adaptive_design", "master_protocol"]
    cl_ids = [k for k in order if k in checklists] + [k for k in checklists if k not in order]
    cl_id = st.selectbox("Checklist", cl_ids, format_func=lambda k: checklists[k]["title"])
    cl = checklists[cl_id]
    with st.expander(f"{len(cl['items'])} items · {cl.get('source_note', '')}"):
        for it in cl["items"]:
            st.markdown(f"**{it['id']} {it['title']}** - {it['requirement']}")
    upload = st.file_uploader("Upload a section (.docx, .txt, .md) or paste below", type=["docx", "txt", "md"])
    pasted = ""
    if upload is not None:
        if upload.name.lower().endswith(".docx"):
            from docx import Document

            doc = Document(upload)
            pasted = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            pasted = upload.read().decode("utf-8", errors="ignore")
    text = st.text_area("Draft section", value=pasted, height=260,
                        placeholder="Paste the estimand / analysis section of your protocol or SAP ...")
    st.caption(f"{len(text):,} characters (up to 12,000 are sent).")
    ppi = st.slider("Guidance passages per item", 1, 3, 1, help="More context per item uses more tokens.")
    if st.button("Run check", type="primary", disabled=not (text.strip() and key_ok and retriever)):
        with st.spinner("Checking ..."):
            try:
                res = check_section(text, cl, retriever, get_llm(), passages_per_item=ppi, reasoning_effort=reasoning)
                ss.check = res.to_record()
            except Exception as exc:
                ss.check = {"type": "checklist", "title": cl["title"], "body": "", "sources": [], "error": friendly_error(exc)}
    if ss.check:
        st.divider()
        render_record(ss.check, index, key="check")
    st.caption("This is a drafting aid, not a regulatory review. Confirm every finding against the guidance itself.")

# ==========================================================================
# Calculators
# ==========================================================================


def show_result(fn, **kwargs) -> None:
    try:
        r = fn(**kwargs)
    except ValueError as exc:
        st.error(str(exc))
        return
    main = {k: v for k, v in r.outputs.items() if not isinstance(v, list)}
    cols = st.columns(min(len(main), 4) or 1)
    for i, (k, v) in enumerate(main.items()):
        cols[i % len(cols)].metric(k.replace("_", " "), f"{v:.4g}" if isinstance(v, float) else str(v))
    st.code(r.to_text(), language=None)


with tab_calc:
    st.markdown("#### Vetted calculators")
    st.caption("The same functions the agent calls. Power formulas are validated against simulation in tests/test_stats_tools.py.")
    calc = st.radio("Calculator", ["Continuous endpoint", "Binary endpoint", "Correlated eyes", "Dropout inflation",
                                   "Multiplicity"], horizontal=True)

    if calc == "Continuous endpoint":
        c1, c2, c3 = st.columns(3)
        design = c1.selectbox("Design", ["parallel", "crossover_2x2", "paired"],
                              help="paired = contralateral-eye or other within-subject pairs")
        hyp = c2.selectbox("Hypothesis", ["superiority", "noninferiority", "equivalence"])
        solve = c3.radio("Solve for", ["sample size", "power"], horizontal=True)
        scale = st.radio("Scale", ["difference", "ratio (GMR & CV)"], horizontal=True)
        kw: dict = {"design": design, "hypothesis": hyp}
        c1, c2, c3, c4 = st.columns(4)
        if scale == "difference":
            kw["delta"] = c1.number_input("True difference (trt - ctl)", value=0.5 if hyp == "superiority" else 0.0, format="%.4f")
            sd_label = {"parallel": "SD", "crossover_2x2": "Within-subject SD", "paired": "SD of differences"}[design]
            kw["sd"] = c2.number_input(sd_label, value=1.0, min_value=1e-6, format="%.4f")
            if hyp != "superiority":
                kw["margin"] = c3.number_input("Margin (> 0)", value=0.5, min_value=0.0, format="%.4f")
        else:
            kw["scale"] = "ratio"
            kw["gmr"] = c1.number_input("GMR", value=0.95, min_value=0.01, format="%.4f")
            kw["cv"] = c2.number_input("CV (within-subject for crossover)", value=0.30, min_value=0.001, format="%.4f")
            if hyp != "superiority":
                kw["lower_margin"] = c3.number_input("Lower margin (ratio)", value=0.80, format="%.4f")
                if hyp == "equivalence":
                    kw["upper_margin"] = c4.number_input("Upper margin (ratio)", value=1.25, format="%.4f")
        c1, c2, c3 = st.columns(3)
        default_alpha = 0.05 if hyp in ("superiority", "equivalence") else 0.025
        kw["alpha"] = c1.number_input("Alpha" + (" (two-sided)" if hyp == "superiority" else " (one-sided)"),
                                      value=default_alpha, min_value=0.0001, max_value=0.5, format="%.4f")
        if solve == "sample size":
            kw["power"] = c2.number_input("Target power", value=0.8, min_value=0.5, max_value=0.999, format="%.3f")
        else:
            kw["n"] = int(c2.number_input("n (parallel: control group; otherwise total)", value=40, min_value=2, step=1))
        if design == "parallel":
            kw["ratio"] = c3.number_input("Allocation trt:ctl", value=1.0, min_value=0.1, format="%.2f")
        show_result(continuous_sample_size, **kw)

    elif calc == "Binary endpoint":
        c1, c2, c3 = st.columns(3)
        hyp = c1.selectbox("Hypothesis", ["superiority", "noninferiority"], key="bin_h")
        pc = c2.number_input("Control proportion", value=0.60, min_value=0.001, max_value=0.999, format="%.3f")
        pt = c3.number_input("Treatment proportion", value=0.80, min_value=0.001, max_value=0.999, format="%.3f")
        c1, c2, c3, c4 = st.columns(4)
        kw = {"p_control": pc, "p_treatment": pt, "hypothesis": hyp}
        kw["alpha"] = c1.number_input("Alpha", value=0.05 if hyp == "superiority" else 0.025, format="%.4f", key="bin_a")
        solve = c2.radio("Solve for", ["sample size", "power"], key="bin_s")
        if solve == "sample size":
            kw["power"] = c3.number_input("Target power", value=0.8, format="%.3f", key="bin_p")
        else:
            kw["n_control"] = int(c3.number_input("n control", value=100, min_value=2, key="bin_n"))
        kw["ratio"] = c4.number_input("Allocation trt:ctl", value=1.0, min_value=0.1, key="bin_r")
        if hyp == "noninferiority":
            kw["margin"] = st.number_input("NI margin (risk difference)", value=0.10, min_value=0.001, format="%.3f")
        else:
            kw["continuity_correction"] = st.checkbox("Continuity correction (Fleiss-Tytun-Ury)")
        show_result(binary_sample_size, **kw)

    elif calc == "Correlated eyes":
        st.caption("Both eyes of a subject receive the same treatment. For contralateral designs use the paired design.")
        c1, c2, c3 = st.columns(3)
        n_ind = int(c1.number_input("Eyes needed if independent", value=100, min_value=1))
        m = c2.number_input("Eyes per subject", value=2.0, min_value=1.0, max_value=2.0, step=0.1)
        icc = c3.number_input("Inter-eye correlation (ICC)", value=0.5, min_value=0.0, max_value=1.0)
        show_result(correlated_eyes_adjustment, n_independent_eyes=n_ind, eyes_per_subject=m, icc=icc)

    elif calc == "Dropout inflation":
        c1, c2, c3 = st.columns(3)
        n = int(c1.number_input("Evaluable n", value=64, min_value=1))
        d = c2.number_input("Dropout rate", value=0.10, min_value=0.0, max_value=0.95, format="%.3f")
        mult = int(c3.number_input("Round up to a multiple of", value=1, min_value=1,
                                   help="e.g. number of sequences x block size"))
        show_result(inflate_for_dropout, n=n, dropout_rate=d, multiple_of=mult)

    else:
        c1, c2 = st.columns([0.7, 0.3])
        raw = c1.text_input("Unadjusted p-values (comma separated, in testing order for fixed sequence)", "0.012, 0.041, 0.030, 0.004")
        method = c2.selectbox("Method", ["holm", "hochberg", "bonferroni", "bh", "fixed_sequence"])
        labels_raw = st.text_input("Labels (optional, comma separated)", "")
        alpha = st.number_input("Alpha", value=0.05, format="%.4f", key="mult_a")
        try:
            pvals = [float(x) for x in re.split(r"[,\s;]+", raw.strip()) if x]
            labels = [x.strip() for x in labels_raw.split(",") if x.strip()] or None
            if labels and len(labels) != len(pvals):
                st.warning("Number of labels does not match the number of p-values; using H1, H2, ...")
                labels = None
            show_result(adjust_pvalues, pvalues=pvals, method=method, alpha=alpha, labels=labels)
        except ValueError:
            st.error("Could not read the p-values.")

# ==========================================================================
# Knowledge Bank
# ==========================================================================


def admin_unlocked() -> bool:
    password = _setting("ADMIN_PASSWORD", "")
    if not password:
        return True
    entered = st.text_input("Admin password", type="password", key="admin_pw")
    return entered == password


with tab_kb:
    st.markdown("#### Knowledge Bank")
    docs = index.documents
    last = index.manifest.get("last_run") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(docs))
    c2.metric("Passages", f"{len(index.chunks):,}")
    c3.metric("Categories", len(index.categories))
    c4.metric("Last indexed", (last.get("finished_at") or "-")[:10])
    st.caption(f"Embedding model: {index.manifest.get('embed_model') or settings.embed_model} · index folder: {settings.index_dir}")

    if docs:
        df = pd.DataFrame([
            {"Category": d["category"], "File": d["source"], "Title": d.get("title", ""), "Year": d.get("year"),
             "Pages": d.get("pages"), "Passages": d.get("chunks"),
             "Pages left out": ", ".join(f"{v} {k}" for k, v in (d.get("skipped_pages") or {}).items()),
             "Indexed": (d.get("indexed_at") or "")[:10]}
            for d in sorted(docs.values(), key=lambda d: (d["category"], d["source"]))
        ])
        st.dataframe(df, hide_index=True, width="stretch")
    skipped = index.manifest.get("skipped") or {}
    if skipped:
        st.warning(f"{len(skipped)} file(s) could not be indexed:")
        for doc_id, s in skipped.items():
            st.markdown(f"- `{doc_id}` - {s['reason']}")

    st.divider()
    st.markdown("##### Manage")
    if not settings.kb_available:
        st.info(
            f"Read-only here: the Knowledge Bank folder ({settings.kb_dir}) is not available - this is expected on "
            "Streamlit Cloud. Add PDFs on your Mac, run `python ingest.py`, then commit and push the `index/` folder."
        )
    elif admin_unlocked():
        cats = sorted({*index.categories, *[p.name for p in settings.kb_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Add PDFs**")
            cat_choice = st.selectbox("Category", cats + ["+ New category"])
            if cat_choice == "+ New category":
                cat_choice = st.text_input("New category name", placeholder="e.g. Ophthalmic")
            files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
            if st.button("Save to Knowledge Bank", disabled=not (files and cat_choice)):
                folder = settings.kb_dir / re.sub(r"[^A-Za-z0-9 _\-]", "", cat_choice).strip()
                folder.mkdir(parents=True, exist_ok=True)
                for f in files:
                    safe = re.sub(r"[^A-Za-z0-9._\- ()]", "_", os.path.basename(f.name))
                    (folder / safe).write_bytes(f.getbuffer())
                st.success(f"Saved {len(files)} file(s) to {folder}. Now update the index.")
        with c2:
            st.markdown("**Update index**")
            rebuild = st.checkbox("Rebuild from scratch", help="Only needed after changing chunking settings or models.")
            if st.button("Update index", type="primary"):
                bar = st.progress(0.0, text="Starting ...")
                with st.status("Indexing ...", expanded=True) as status:
                    try:
                        ingest(settings, get_embedder(settings.embed_model), rebuild=rebuild, log=status.write,
                               progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
                        status.update(label="Index updated", state="complete")
                    except Exception as exc:
                        status.update(label=f"Indexing failed: {exc}", state="error")
                reset_index_caches()
                st.button("Reload app")
            st.markdown("**Remove a document**")
            to_remove = st.selectbox("Document", ["-"] + sorted(docs), key="rm_doc")
            confirm = st.checkbox("Move the PDF to _Removed/ and drop it from the index", key="rm_ok")
            if st.button("Remove", disabled=to_remove == "-" or not confirm):
                src = settings.kb_dir / to_remove
                if src.exists():
                    dest = settings.kb_dir / "_Removed" / to_remove
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest))
                remove_document(settings.index_dir, to_remove)
                reset_index_caches()
                st.success(f"Removed {to_remove}.")
                st.rerun()
    else:
        st.caption("Enter the admin password to add or remove documents.")

# ==========================================================================
# Evaluate
# ==========================================================================

with tab_eval:
    st.markdown("#### Retrieval evaluation")
    st.caption("Scores how often the right document (and page) is retrieved for a fixed question set - run it after "
               "adding documents or changing settings. No LLM calls are made.")
    custom = st.file_uploader("Optional: use a custom question set (JSON)", type=["json"], key="eval_q")
    try:
        questions = json.loads(custom.read())["questions"] if custom else load_questions()
    except Exception as exc:
        questions = []
        st.error(f"Could not read questions: {exc}")
    st.caption(f"{len(questions)} questions loaded. Format: see eval/questions.json.")
    c1, c2 = st.columns(2)
    k_eval = c1.slider("k (passages retrieved)", 3, 10, 5)
    mode_opts = ["bm25", "dense", "hybrid"] + (["hybrid + rerank"] if rerank_available else [])
    modes_sel = c2.multiselect("Modes", mode_opts, default=mode_opts)
    if st.button("Run evaluation", disabled=not (questions and retriever and modes_sel)):
        bar = st.progress(0.0)
        modes = [("hybrid", True) if m == "hybrid + rerank" else (m, False) for m in modes_sel]
        ss.eval = evaluate(retriever, questions, k=k_eval, modes=modes, log=lambda m: None,
                           progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    if ss.eval:
        st.dataframe(pd.DataFrame(ss.eval["summary"]), hide_index=True, width="stretch")
        det = pd.DataFrame(ss.eval["details"])
        only_miss = st.checkbox("Show misses only", value=True)
        st.dataframe(det[~det["doc_hit"]] if only_miss else det, hide_index=True, width="stretch")
        st.download_button("⬇ Details (CSV)", det.to_csv(index=False), file_name="clinstat_eval.csv", mime="text/csv")
