# ClinStat — first-time setup

From a clean start on your Mac (Intel, macOS) through a private GitHub repo and a
deployed Streamlit Cloud app. Run the commands in **Terminal**, one block at a time.
Lines starting with `#` are comments.

**What you need:** the `ClinStat_v2.zip` download, your new Knowledge Bank folder with
its sub-folders, a GitHub account, and a Groq account.

---

## 1. Revoke the old Groq key and create a new one

The old repo had a committed `.env` file containing a Groq key.

1. Go to <https://console.groq.com/keys>.
2. **Delete** every key that was ever in that `.env`. Replacing a key doesn't disable it; only deleting does.
3. Click **Create API Key** and name it `clinstat`. Copy it somewhere temporary; you'll paste it in steps 6 and 13.

## 2. Delete the old Streamlit app and GitHub repo

1. **Streamlit Cloud:** go to <https://share.streamlit.io>. On the old ClinStat app, click **⋮ → Delete**.
2. **GitHub:** go to <https://github.com/wshamp09/ClinStat> → **Settings** → scroll to **Danger Zone** → **Delete this repository**, and type the name to confirm.

## 3. Move the old local folder out of the way

Keep it until the new app works, then delete it.

```bash
mv ~/Documents/GitHub/ClinStat ~/Documents/GitHub/ClinStat_old
```

## 4. Unpack the new project

```bash
cd ~/Documents/GitHub
unzip ~/Downloads/ClinStat_v2.zip -d ~/Documents/GitHub/
# If Safari already unzipped it to ~/Downloads/ClinStat, use this instead:
# mv ~/Downloads/ClinStat ~/Documents/GitHub/

cd ~/Documents/GitHub/ClinStat
ls -a
# expect: app_clinstat.py  clinstat/  eval/  index/  tests/  ingest.py  requirements.txt  .gitignore  .streamlit/ ...
```

## 5. Create a fresh Python 3.11 environment

```bash
python3.11 --version
# If that says "command not found":
#   brew install python@3.11
```

If Python 3.11 came from the **python.org installer**, give it the standard SSL certificates once.
Without them, model downloads can fail with `CERTIFICATE_VERIFY_FAILED`. Skip this if you
installed Python with Homebrew.

```bash
open "/Applications/Python 3.11/Install Certificates.command"
```

Create the environment and install. `--only-binary=:all:` tells pip to use only ready-made
packages (wheels) and never compile, so it automatically picks versions built for your Mac.

```bash
cd ~/Documents/GitHub/ClinStat
python3.11 -m venv venv
source venv/bin/activate              # your prompt now starts with (venv)
python -m pip install --upgrade pip
pip install --only-binary=:all: -r requirements-dev.txt   # app + test dependencies; a few minutes
```

Check the install:

```bash
python -c "import streamlit, fastembed, groq, scipy, onnxruntime, pypdfium2, docx; print('OK', streamlit.__version__, onnxruntime.__version__, scipy.__version__)"
# expect on an Intel Mac: OK 1.63.0 1.19.2 1.15.x
python -c "import ssl, urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=10); print('SSL OK')"
# expect: SSL OK   (if not, see Troubleshooting: CERTIFICATE_VERIFY_FAILED)
```

> Run `source venv/bin/activate` from `~/Documents/GitHub/ClinStat` in **every new Terminal window** before using the project.

## 6. Add the Groq key for local runs

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
open -e .streamlit/secrets.toml
```

In TextEdit, replace `your-key-here` with the new key, so the line reads
`GROQ_API_KEY = "gsk_..."`. Save and close. This file is git-ignored and will never be uploaded.

Check that the key works and that every model in the fallback chain is available on your account.
This sends one tiny request per model:

```bash
python check_models.py
# expect, for each model: reachable: yes · tool calling: OK {...} · requests/day left ... ; then "All models ready."
```

If a model fails with "model not found" or "decommissioned", look up the current name on Groq's Models page
and set it in `.streamlit/secrets.toml`, for example:
`CLINSTAT_GROQ_MODELS = "openai/gpt-oss-120b,qwen/qwen3.8-27b"`.

## 7. Fill the Knowledge Bank

Each sub-folder becomes a category in the app. Move the old folder aside and copy the new one in:

```bash
mv ~/Documents/KnowledgeBank ~/Documents/KnowledgeBank_old 2>/dev/null
cp -R "/PATH/TO/YOUR/NEW/KnowledgeBank" ~/Documents/KnowledgeBank
```

Replace `/PATH/TO/YOUR/NEW/KnowledgeBank` with the real path; dragging the folder into Terminal pastes it.
You can also just drag the folder in Finder.

Expected layout. Category names are up to you; folders whose names start with `_` are ignored.

```
~/Documents/KnowledgeBank/
├── RegulatoryDocs/      *.pdf
├── DeviceRegulatory/    *.pdf
├── Ophthalmic/          *.pdf
├── ReportingGuidelines/ *.pdf
├── StatBooks/           *.pdf
└── StatPapers/          *.pdf
```

Check what will be indexed:

```bash
find ~/Documents/KnowledgeBank -name "*.pdf" -not -path "*/_*" | sed "s|$HOME/Documents/KnowledgeBank/||" | sort
find ~/Documents/KnowledgeBank -name "*.pdf" -not -path "*/_*" | wc -l
```

## 8. Run the tests (optional, ~2 seconds)

```bash
pytest
# expect: 48 passed
```

## 9. Build the index

```bash
python ingest.py --rebuild
```

What you'll see:

* The first run downloads the embedding model (~70 MB) to `.cache/fastembed/`.
* One line per PDF: pages, number of passages, and which pages were left out (TOC, index, references).
* Progress readouts while embedding, with a rate and an estimated time left. Expect a few minutes for ~30 books and guidances.
* A closing summary, then **"Files not indexed (fix and re-run)"** if any PDFs had no text layer.

**Fixing scanned PDFs.** Run OCR on each file listed as skipped, then index again. Only new or changed files are processed.

```bash
brew install ocrmypdf          # one time
cd ~/Documents/KnowledgeBank/StatBooks                                         # folder of the skipped file
ocrmypdf --skip-text "Some_Book.pdf" "Some_Book_ocr.pdf" && mv "Some_Book_ocr.pdf" "Some_Book.pdf"
cd ~/Documents/GitHub/ClinStat
python ingest.py               # incremental
```

Check the index size; GitHub's limit is 100 MB per file:

```bash
du -sh index/*
```

## 10. Check retrieval quality

```bash
python evaluate.py
```

It prints `doc_hit@5`, `page_hit@5` and `MRR` for keyword, semantic, hybrid and re-ranked search,
then any misses. Questions whose expected document isn't in your new Knowledge Bank are skipped.
Edit `eval/questions.json` to add a question or two for each new document; the file has the format.

## 11. Run the app locally

```bash
cd ~/Documents/GitHub/ClinStat
./run_app.sh
```

`run_app.sh` always starts the app with the project's own Python (`venv/`), even in a Terminal window where
the venv isn't activated, and installs anything missing first. `./run_app.sh --check` only verifies the
environment. If the app is started with a different Python, it stops with a message naming that Python and
the fix, rather than failing with `ModuleNotFoundError`.

The app opens at <http://localhost:8501>. The first launch downloads the re-ranker (~90 MB). A quick check:

1. The sidebar shows a green **"N documents · N passages"** box and no API-key error.
2. **Ask:** click an example question. You should see the research steps, an answer with `[S1]`-style citations, and **Sources** → **View** opens the passage and the PDF page.
3. **Calculators:** Continuous endpoint with the defaults (d = 0.5, SD = 1) → 64 per group.
4. **Knowledge Bank:** your documents are listed, with any skipped files explained.

Stop the app with `Ctrl+C`.

## 12. Create the private GitHub repo and push

First, a safety check. Make sure nothing secret or bulky will be committed:

```bash
cd ~/Documents/GitHub/ClinStat
git init -b main
git add .
git status --short | grep -E "\.env$|secrets\.toml$|venv/|\.pdf$|\.cache/" && echo "STOP: something sensitive is staged" || echo "Clean: nothing sensitive staged"
git grep -n "gsk_" -- . ':!SETUP.md' && echo "STOP: key found" || echo "No keys in tracked files"
git status --short | head -60      # expect code, docs, index/, eval/, tests/ — no venv, no PDFs, no secrets.toml
```

Commit:

```bash
git commit -m "ClinStat: hybrid retrieval agent, calculators, guidance workflows"
```

Create the **private** repo and push. Choose **one** option.

*Option A — GitHub CLI (simplest):*

```bash
brew install gh                    # one time
gh auth login                      # GitHub.com → HTTPS → log in with a browser
gh repo create ClinStat --private --source=. --remote=origin --push
```

*Option B — website:* go to <https://github.com/new>, set the name to `ClinStat`, choose **Private**, and leave
README/.gitignore/licence **unchecked**. Click **Create repository**, then:

```bash
git remote add origin https://github.com/wshamp09/ClinStat.git
git push -u origin main
```

Confirm on GitHub that the repo shows **Private**, and that there's no `.env`, `venv/` or `secrets.toml` in it.

## 13. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> → **Create app** → deploy from a GitHub repo. The code stays private even though the app gets a URL.
   * If the private repo isn't listed, open your profile menu → **Settings → Linked accounts → GitHub** and allow access to **private repositories**.
2. Fill in **Repository** `wshamp09/ClinStat`, **Branch** `main`, **Main file path** `app_clinstat.py`, and pick an app URL.
3. **Advanced settings:**
   * **Python version:** `3.11`
   * **Secrets:** paste
     ```toml
     GROQ_API_KEY = "gsk_your_new_key"
     ```
4. Click **Deploy**. The first boot installs packages and downloads both models, which takes a few minutes.
5. In the cloud, the Knowledge Bank tab is read-only because the PDFs stay on your Mac. That's expected.

Optional: to limit who can open the app, go to the app's **Settings → Sharing** and invite only specific emails.

## 14. Day-to-day: adding or changing documents

```bash
cd ~/Documents/GitHub/ClinStat && source venv/bin/activate
# 1. drop new PDFs into ~/Documents/KnowledgeBank/<Category>/
python ingest.py                  # incremental; read the skipped list at the end
python evaluate.py                # optional
git add index eval
git commit -m "Index: add <documents>"
git push                          # Streamlit Cloud redeploys automatically
```

Once everything works, remove the old copies:

```bash
rm -rf ~/Documents/GitHub/ClinStat_old ~/Documents/KnowledgeBank_old
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install` tries to **build** a package ("Building wheel for …", "Failed building wheel") | You left out `--only-binary=:all:`. Re-run `pip install --only-binary=:all: -r requirements-dev.txt`. |
| `No matching distribution found for …` | Check that the venv is Python 3.11 (`python --version`); recreate it with `rm -rf venv && python3.11 -m venv venv`. If it names **pypdfium2**, delete the two `pypdfium2` lines from `requirements.txt` and re-run. It's optional: without it, the source viewer shows page text but no page image. |
| `CERTIFICATE_VERIFY_FAILED` (install, `ingest.py` or the app) | python.org Python: run `open "/Applications/Python 3.11/Install Certificates.command"`. Otherwise: `pip install --upgrade certifi && export SSL_CERT_FILE=$(python -m certifi)`, and add that `export` line to `~/.zshrc` to make it permanent. On a corporate network or VPN that inspects traffic, run setup off the VPN. |
| Sidebar: **"The index is empty"** | Run `python ingest.py` from the project folder. On the cloud, check that `index/` was pushed: `git ls-files index`. |
| Sidebar: **"GROQ_API_KEY not found"** | Locally, check `.streamlit/secrets.toml`. On the cloud, check App → Settings → Secrets. |
| Status shows **"… is rate-limited - switching to …"** | Normal: the next model in the chain took over. The answer's caption shows which model wrote it. |
| Answer shows **"Every Groq model in the fallback chain is rate-limited"** | Wait a minute (per-minute limits) and retry, or lower **Passages per search** / **Reasoning effort**. If it persists all day, the daily free quotas are used up; they reset within 24 hours. `python check_models.py` shows what's left. |
| **401** from Groq | The key is wrong or was deleted. Create a new one and update the secrets. |
| `ingest.py`: **"Knowledge Bank folder not found"** | The folder must be `~/Documents/KnowledgeBank`, or pass `--kb-dir "/other/path"`. |
| Cloud build fails | Open **Manage app → logs**. The most common cause is a Python version other than 3.11. Delete the app and redeploy with 3.11. |
| `ModuleNotFoundError` (e.g. `dotenv`) or **"ClinStat can't start: required packages are missing"** | The app was started with a Python other than the project venv, usually a plain `streamlit run` in a window without `(venv)`. Start it with `./run_app.sh`. |
| App feels slow | Under each answer, a line shows `⏱ total · Groq · search (re-rank)`. **Groq** dominates: set **Reasoning effort** to `low` in the sidebar. **Re-rank** dominates (common on Intel Macs): switch off **Cross-encoder re-ranking** in the sidebar, or set `CLINSTAT_USE_RERANKER=0` in `.env`. The first page load after starting the app also loads both models and builds the keyword index once; later clicks reuse them. |
| Models re-download every run | They're cached in `.cache/fastembed/` in the project folder. Don't delete it, and run commands from the project folder. |
