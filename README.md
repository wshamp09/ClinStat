# ClinStat

A research agent for clinical trial statistics. It answers questions from a curated
Knowledge Bank (regulatory guidance, statistics textbooks, methods papers), with
inline citations to the exact document and page. Every number comes from vetted
calculators, never from model arithmetic.

| Tab | What it does |
|---|---|
| 💬 **Ask** | Multi-turn chat. The agent searches the Knowledge Bank, reads whole pages when a passage is cut off, and calls calculators. Answers cite passages `[S1]` and calculations `[C1]`. Click **View** on any source to see the passage, the full page text and the original PDF page. |
| ⚖️ **Compare guidance** | Pick 2–4 documents and a topic. It gets the best passages from each document separately, then writes a cited comparison table, agreements, differences and practical implications. |
| ✅ **Check a section** | Paste or upload a protocol/SAP section and pick a checklist (ICH E9(R1) estimands, ICH E9 core SAP content, FDA multiple endpoints, PRO endpoints, external controls, adaptive designs, master/platform protocols). Each item is graded Met / Partial / Missing, with a quote from your draft, the gap, suggested wording and the supporting guidance passages. |
| 🧮 **Calculators** | Sample size and power for continuous endpoints (parallel, 2×2 crossover, paired/contralateral; superiority, NI, equivalence; difference or GMR/CV scale), binary endpoints, the correlated-eyes design effect, dropout inflation and multiplicity adjustment. |
| 📚 **Knowledge Bank** | Documents indexed, pages left out, files that failed to index and why. Locally you can also upload PDFs, update the index incrementally and remove documents. |
| 📏 **Evaluate** | Scores retrieval (document hit@k, page hit@k, MRR) on `eval/questions.json` for keyword, semantic, hybrid and re-ranked search. No LLM calls. |

Every answer, comparison and check can be downloaded as **Markdown** or **Word**, with sources and the cited passages. The whole conversation can be downloaded the same way.

## How it works

```
PDF ─► page text ─► clean ─► classify pages ─► sentence-aware chunks ─► embed ─► index/
        (pypdf)     ligatures, hyphenation,    content / TOC / index /   ~1,200 chars,       bge-small
                    running heads, draft-      references / blank        200-char overlap,  (fastembed)
                    guidance line numbers      (only content indexed)    section + page

question ─► Groq agent (gpt-oss-120b) ─► tools ─► search_knowledge_bank ─► BM25 + dense ─► RRF ─► cross-encoder
                                                  read_page / list_documents
                                                  calculators (scipy)
          ◄── answer with [S#]/[C#] tags, checked against the sources actually retrieved
```

* **Chunking.** Text is split at sentence boundaries into ~1,200-character chunks with 200 characters of overlap. Each chunk records its start and end page and its auto-detected section heading. Tables of contents, back-of-book indexes and bibliographies are detected and left out, because they match almost any query on keywords alone.
* **Retrieval.** BM25 finds exact terms ("Grizzle", "E9(R1)", "PROC GLIMMIX"). bge-small embeddings find meaning, using bge's query-side instruction for queries. Reciprocal-rank fusion merges the two lists, and a small ONNX cross-encoder re-orders the top 20. No more than two chunks per page are returned.
* **Index.** `index/` holds `chunks.jsonl.gz`, `embeddings.npy` (float16) and `manifest.json`, about 8–10 MB for the current Knowledge Bank. Search is an exact in-memory cosine product, which takes milliseconds at this scale. Ingestion is incremental: file hashes are compared, so only new or changed PDFs are processed, and deleted PDFs are dropped.
* **Citations.** Every passage or calculation sent to the model gets a tag. After the answer comes back, tags that were never issued are removed and the UI reports them, so a citation always points to text the model actually saw.
* **Calculators.** Continuous designs use exact noncentral-t power. Equivalence uses the noncentral-t approximation, the same as PowerTOST `method="nct"`. Binary designs use the Fleiss pooled z-test (optional Fleiss–Tytun–Ury correction) for superiority and the unpooled Wald test for non-inferiority. `tests/test_stats_tools.py` checks each formula against published values (64/group for d = 0.5; n = 40 for CV 30%, GMR 0.95 in a 2×2 TOST; 82 and 91/group for 0.6 vs 0.8) and against Monte Carlo simulation of the actual test.

## Setup (Mac, Python 3.11)

Step-by-step first-time setup, including GitHub and Streamlit Cloud, is in **`SETUP.md`**. In short:

```bash
cd ~/Documents/GitHub/ClinStat
python3.11 -m venv venv && source venv/bin/activate
pip install --only-binary=:all: -r requirements-dev.txt   # wheels only; requirements.txt alone skips test tools
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your GROQ_API_KEY

open ~/Documents/GitHub/ClinStat/.streamlit

```

The key is read from `.streamlit/secrets.toml` or a local `.env` file. Both are git-ignored, and the code contains no keys.

### Build the index

PDFs live outside the repo, in `~/Documents/KnowledgeBank/<Category>/*.pdf`. Each sub-folder becomes a category, so you can add new ones such as `Ophthalmic`, `DeviceRegulatory` or `ReportingGuidelines`. Folders whose names start with `_` are ignored.

```bash
python ingest.py              # incremental; progress is printed per file and during embedding
python ingest.py --rebuild    # from scratch (after changing chunking settings or models)
```

The first run downloads the embedding model (~70 MB) and the re-ranker (~90 MB) into `.cache/fastembed/`. The run ends with a list of any files that could not be indexed. A scanned PDF needs OCR first, for example `ocrmypdf in.pdf out.pdf`.

### Run

```bash
./run_app.sh                      # web app (always uses the project venv)
python clinstat_query.py          # terminal chat
python check_models.py            # check each Groq model in the fallback chain
python evaluate.py                # retrieval scores (writes misses to the console)
pytest                            # tests (offline; ~2 s)
```

## Deploy to Streamlit Community Cloud

1. Commit the code and the `index/` folder. Never commit `.env`, `secrets.toml`, `venv/` or PDFs; `.gitignore` covers all of these.
2. In the app's **Settings → Secrets**, add `GROQ_API_KEY = "..."`.
3. Under **Advanced settings**, choose Python 3.11 to match your Mac.
4. In the cloud the Knowledge Bank tab is read-only, since the PDFs aren't there. To add documents, update the index on your Mac and push `index/`.

> **Repository visibility:** `index/chunks.jsonl.gz` contains the extracted text of every indexed document, textbooks included. Keep the GitHub repo **private** (Community Cloud can deploy from private repos) unless you index only public documents.

> **Confidential material:** questions and retrieved passages are sent to Groq's API. Don't index internal or confidential study documents in this app.

## Configuration

All settings live in `clinstat/config.py` and can be overridden by environment variables or secrets:

| Setting | Default | Notes |
|---|---|---|
| `CLINSTAT_KB_DIR` | `~/Documents/KnowledgeBank` | Source PDFs |
| `CLINSTAT_INDEX_DIR` | `index` | Relative to the project folder |
| `CLINSTAT_GROQ_MODELS` | `openai/gpt-oss-120b,qwen/qwen3.8-27b` | Groq models tried in order; the next is used when one is rate-limited. Each has its own free quota. |
| `CLINSTAT_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Changing it triggers a full rebuild |
| `CLINSTAT_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | fastembed cross-encoder |
| `CLINSTAT_USE_RERANKER` | `1` | `0` disables re-ranking (fastest on older CPUs) |
| `ADMIN_PASSWORD` | unset | Requires a password for Knowledge Bank changes in the app |

### Model fallback (free tier)

Groq's free tier gives **each model its own quota**, roughly 8K tokens/minute and 200K tokens/day for
`gpt-oss-120b`. ClinStat sends every request to the first model in `CLINSTAT_GROQ_MODELS`. If that model is
rate-limited, overloaded or retired, it retries the same request on the next model, and keeps using that
model for the rest of the answer. Each answer shows which model wrote it and whether a fallback happened.
`python check_models.py` tests every model in the chain (tool calling and quota left) with one tiny request
each. If all models are limited, lower **Passages per search** or **Reasoning effort** in the sidebar, or wait
for the per-minute window to reset.

## Project layout

```
app_clinstat.py          Streamlit app (six tabs)
run_app.sh               starts the app with the project venv
clinstat_query.py        terminal chat
check_models.py          checks each Groq model in the fallback chain
ingest.py                build / update the index
evaluate.py              retrieval evaluation CLI
clinstat/
  config.py              settings
  pdf_text.py            extraction, cleaning, page classification
  chunking.py            sentence-aware chunking with sections and pages
  embeddings.py          fastembed embedder and cross-encoder (ONNX, no torch)
  index_store.py         on-disk index and incremental ingestion
  retrieval.py           BM25, dense search, RRF fusion, re-ranking
  llm.py                 Groq client with model fallback chain, key handling, friendly errors
  tools.py               agent tool schemas, execution and citation registry
  agent.py               tool-calling loop
  stats_tools.py         calculators
  workflows.py           guidance comparison and section checker
  checklists/*.json      checklist definitions (edit or add your own)
  evaluation.py          retrieval metrics
  export.py              Markdown and Word export
  ui.py                  Streamlit rendering helpers
eval/questions.json      evaluation questions (add some for each new document)
tests/                   offline tests (fake LLM, generated PDFs, simulation checks)
KNOWLEDGE_BANK_ADDITIONS.md   suggested documents to add
```

## Adding a checklist

Drop a JSON file into `clinstat/checklists/` with `id`, `title`, `doc_type`, `source_note` and `items`. Each item needs an `id`, `title`, `requirement`, a retrieval `query` and an optional `documents` list of file-name fragments. It appears in the app automatically.
