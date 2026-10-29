#!/usr/bin/env bash
# Start ClinStat with the project's own Python (venv), whatever Terminal state you're in.
#   ./run_app.sh            -> web app
#   ./run_app.sh --check    -> only verify the environment
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x "venv/bin/python" ]; then
  echo "No venv found here. Create it first:"
  echo "  python3.11 -m venv venv && venv/bin/python -m pip install --only-binary=:all: -r requirements-dev.txt"
  exit 1
fi

if ! venv/bin/python -c "import dotenv, streamlit, fastembed, groq, scipy, onnxruntime, docx, pypdf" 2>/dev/null; then
  echo "Some packages are missing from the venv - installing them now ..."
  venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
fi
echo "Environment OK: $(venv/bin/python --version) at $(pwd)/venv"

if [ "${1:-}" = "--check" ]; then exit 0; fi
exec venv/bin/python -m streamlit run app_clinstat.py "$@"
