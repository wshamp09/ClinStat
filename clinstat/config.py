"""Central configuration.

Every setting can be overridden with an environment variable (or a key of the
same name in Streamlit secrets / a local ``.env``). Paths are resolved relative
to the project folder, so the app behaves the same whichever directory it is
launched from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (existing variables win).

    Uses python-dotenv when installed; otherwise a small built-in reader that handles
    comments, blank lines, ``export KEY=...`` and quoted values. Either way a missing
    python-dotenv never stops the app from starting.
    """
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(path)
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] in ("'", '"') and value.endswith(value[:1]) and len(value) >= 2:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(PROJECT_DIR / ".env")


def _setting(name: str, default: str) -> str:
    """Environment first, then Streamlit secrets, then the default."""
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit secrets raise if no secrets file exists
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def _path(value: str) -> Path:
    p = Path(os.path.expanduser(value))
    return p if p.is_absolute() else (PROJECT_DIR / p)


@dataclass(frozen=True)
class Settings:
    # --- locations -------------------------------------------------------
    kb_dir: Path = field(default_factory=lambda: _path(_setting("CLINSTAT_KB_DIR", "~/Documents/KnowledgeBank")))
    index_dir: Path = field(default_factory=lambda: _path(_setting("CLINSTAT_INDEX_DIR", "index")))

    # --- models ----------------------------------------------------------
    embed_model: str = field(default_factory=lambda: _setting("CLINSTAT_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: _setting("CLINSTAT_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    # Groq models tried in order; the next one is used when the current one is rate-limited.
    # Each Groq model has its own free-tier quota, so the chain adds capacity.
    groq_models: tuple[str, ...] = field(default_factory=lambda: tuple(
        m.strip() for m in _setting("CLINSTAT_GROQ_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b").split(",") if m.strip()
    ))

    # --- chunking --------------------------------------------------------
    chunk_chars: int = 1200          # target characters per chunk (~300 tokens; bge-small reads 512)
    chunk_overlap_chars: int = 200   # carried over at sentence boundaries
    min_chunk_chars: int = 120       # fragments shorter than this merge into a neighbour
    min_chars_per_page: int = 200    # below this on average the PDF is treated as scanned

    # --- retrieval -------------------------------------------------------
    top_k: int = 5                   # passages returned to the LLM per search
    candidates_per_retriever: int = 40
    rerank_pool: int = 20           # candidates re-read by the cross-encoder (CPU cost scales with this)
    use_reranker: bool = field(default_factory=lambda: _setting("CLINSTAT_USE_RERANKER", "1") != "0")
    rrf_k: int = 60

    # --- agent -----------------------------------------------------------
    max_tool_rounds: int = 4
    history_turns: int = 3           # user/assistant pairs kept as context
    tool_result_char_limit: int = 9000
    temperature: float = 0.2

    @property
    def kb_available(self) -> bool:
        return self.kb_dir.is_dir()


settings = Settings()

INDEX_SCHEMA_VERSION = 2
