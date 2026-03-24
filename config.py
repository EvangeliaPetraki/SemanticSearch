from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def _get_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _get_path(name: str, default: str) -> Path:
    value = os.environ.get(name, default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


PDF_PATH = _get_path("SEMANTIC_SEARCH_PDF_PATH", "data/input.pdf")
SUMMARY_FILE = _get_path("SEMANTIC_SEARCH_SUMMARY_FILE", "data/summary.txt")
CHROMA_PATH = _get_path("SEMANTIC_SEARCH_CHROMA_PATH", "data/chroma_db")
COLLECTION_NAME = _get_str("SEMANTIC_SEARCH_COLLECTION_NAME", "document_chunks")
EMBED_MODEL = _get_str("SEMANTIC_SEARCH_EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = _get_str("SEMANTIC_SEARCH_OLLAMA_MODEL", "llama3")
CHUNK_SIZE = _get_int("SEMANTIC_SEARCH_CHUNK_SIZE", 2000)
CHUNK_OVERLAP = _get_int("SEMANTIC_SEARCH_CHUNK_OVERLAP", 500)
TOP_K = _get_int("SEMANTIC_SEARCH_TOP_K", 5)
