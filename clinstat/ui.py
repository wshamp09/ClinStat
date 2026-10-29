"""Streamlit rendering helpers shared by the app's tabs."""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from .config import settings
from .export import STATUS_LABEL, record_to_docx, record_to_markdown

STATUS_ICON = {"met": "✅", "partial": "🟡", "missing": "❌", "not_applicable": "➖", "not_assessed": "❔"}


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:n] or "clinstat"


def style_citations(text: str) -> str:
    """Render [S1]/[C1] tags as small superscripts."""
    return re.sub(r"\[([SC]\d+)\]", r"<sup>[\1]</sup>", text or "")


@st.cache_data(show_spinner=False, max_entries=64)
def render_pdf_page(pdf_path: str, page: int, scale: float = 1.6):
    """PNG bytes of one PDF page, or None if it cannot be rendered."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        if not (1 <= page <= len(pdf)):
            return None
        image = pdf[page - 1].render(scale=scale).to_pil()
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def local_pdf_path(doc_id: str) -> Path | None:
    p = settings.kb_dir / doc_id
    return p if p.is_file() else None


@st.dialog("Source", width="large")
def source_dialog(src: dict, page_chunks: list[dict]) -> None:
    st.markdown(f"**[{src['tag']}] {src['label']}**")
    if src.get("section"):
        st.caption(f"Section (auto-detected): {src['section']}")
    st.markdown("**Cited passage**")
    st.info(src["text"])
    pdf = local_pdf_path(src["doc_id"])
    tab_text, tab_page = st.tabs(["Page text (indexed)", "Original page"])
    with tab_text:
        if page_chunks:
            st.markdown("\n\n".join(c["text"] for c in page_chunks))
        else:
            st.caption("No other indexed text on this page.")
    with tab_page:
        img = render_pdf_page(str(pdf), int(src["page"])) if pdf else None
        if img:
            st.image(img, caption=f"{src['source']} - page {src['page']}")
        else:
            st.caption("The PDF is not available here (e.g. on Streamlit Cloud). Open it locally to view the page.")


def render_sources(record: dict, index, key: str) -> None:
    sources = record.get("sources", [])
    calcs = record.get("calcs", [])
    if not sources and not calcs:
        return
    cited = [s for s in sources if s.get("cited")]
    other = [s for s in sources if not s.get("cited")]
    label = f"Sources ({len(cited)} cited"
    label += f", {len(other)} also retrieved)" if other else ")"
    with st.expander(label, expanded=False):
        for group, title in ((cited, None), (other, "Retrieved but not cited")):
            if not group:
                continue
            if title:
                st.caption(title)
            for i, s in enumerate(group):
                c1, c2 = st.columns([0.85, 0.15])
                sec = f" · _{s['section']}_" if s.get("section") else ""
                c1.markdown(f"**[{s['tag']}]** {s['label']}{sec}")
                c1.caption(s["text"][:280] + ("…" if len(s["text"]) > 280 else ""))
                if c2.button("View", key=f"{key}_src_{s['tag']}_{i}"):
                    chunks = index.page_text(s["source"], int(s["page"])) if index is not None else []
                    source_dialog(s, chunks)
    if calcs:
        with st.expander(f"Calculations ({len(calcs)})", expanded=False):
            for c in calcs:
                st.markdown(f"**[{c['tag']}]**")
                st.code(c["text"], language=None)


def render_checklist(items: list[dict]) -> None:
    import pandas as pd

    if not items:
        return
    df = pd.DataFrame([
        {
            "": STATUS_ICON.get(i["status"], ""),
            "ID": i["id"],
            "Item": i["title"],
            "Status": STATUS_LABEL.get(i["status"], i["status"]),
            "Evidence in draft": i.get("evidence", ""),
            "Gap": i.get("gap", ""),
            "Suggested wording": i.get("suggestion", ""),
            "Guidance": " ".join(f"[{c}]" for c in i.get("citations", [])),
        }
        for i in items
    ])
    st.dataframe(df, hide_index=True, width="stretch")


def download_buttons(record: dict, key: str) -> None:
    name = _slug(record.get("title", "clinstat"))
    c1, c2, _ = st.columns([0.2, 0.2, 0.6])
    # Files are built only when a button is clicked (callable data), not on every rerun.
    c1.download_button("⬇ Markdown", lambda: record_to_markdown(record, with_passages=True), file_name=f"{name}.md",
                       mime="text/markdown", key=f"{key}_md", on_click="ignore")
    c2.download_button("⬇ Word", lambda: record_to_docx(record), file_name=f"{name}.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       key=f"{key}_docx", on_click="ignore")


def render_record(record: dict, index, key: str) -> None:
    if record.get("error"):
        st.error(record["error"])
    if record.get("body"):
        st.markdown(style_citations(record["body"]), unsafe_allow_html=True)
    if record.get("items"):
        render_checklist(record["items"])
    if record.get("invalid_tags"):
        st.caption(f"Removed {len(record['invalid_tags'])} citation tag(s) that did not match a retrieved source.")
    if record.get("tool_log"):
        steps = [t["tool"].replace("_", " ") for t in record["tool_log"]]
        st.caption("Steps: " + " → ".join(steps))
    tm = record.get("timings") or {}
    parts = []
    if tm:
        parts += [f"⏱ {tm.get('total_s', 0)} s total", f"Groq {tm.get('model_s', 0)} s"]
        if tm.get("search_s"):
            parts.append(f"search {tm['search_s']} s" + (f" (re-rank {tm['rerank_s']} s)" if tm.get("rerank_s") else ""))
    if record.get("model"):
        model_txt = f"model: {record['model']}"
        if record.get("fallbacks"):
            model_txt += " (fallback: primary model was rate-limited)"
        parts.append(model_txt)
    if parts:
        st.caption(" · ".join(parts))
    render_sources(record, index, key)
    if record.get("body"):
        download_buttons(record, key)
