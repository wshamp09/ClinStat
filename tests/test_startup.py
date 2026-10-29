"""Start-up robustness: .env loading without python-dotenv."""
from __future__ import annotations

import os
import sys

from clinstat.config import load_env_file


def test_env_file_loads_without_python_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "\n"
        "GROQ_API_KEY=gsk_test_value\n"
        "export CLINSTAT_KB_DIR=\"~/Documents/My Knowledge Bank\"\n"
        "CLINSTAT_USE_RERANKER=0   # inline comment\n"
        "ALREADY_SET=from_file\n"
        "QUOTED='single quoted'\n"
    )
    monkeypatch.setitem(sys.modules, "dotenv", None)  # simulate python-dotenv not installed
    for k in ("GROQ_API_KEY", "CLINSTAT_KB_DIR", "CLINSTAT_USE_RERANKER", "QUOTED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALREADY_SET", "from_shell")
    load_env_file(env)
    assert os.environ["GROQ_API_KEY"] == "gsk_test_value"
    assert os.environ["CLINSTAT_KB_DIR"] == "~/Documents/My Knowledge Bank"
    assert os.environ["CLINSTAT_USE_RERANKER"] == "0"
    assert os.environ["QUOTED"] == "single quoted"
    assert os.environ["ALREADY_SET"] == "from_shell"  # real environment wins


def test_missing_env_file_is_fine(tmp_path):
    load_env_file(tmp_path / "nope.env")
