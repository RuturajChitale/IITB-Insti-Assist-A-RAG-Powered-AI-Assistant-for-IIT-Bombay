"""
Central configuration for the IITB Insti-Assist RAG pipeline.
Keeping every tunable in one place makes it easy to swap models,
change chunk sizes, or point the index somewhere else without
hunting through every script.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
INDEX_PATH = PROCESSED_DIR / "faiss.index"
INDEX_META_PATH = PROCESSED_DIR / "faiss_meta.json"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS = 1000        # target characters per chunk
CHUNK_OVERLAP_CHARS = 150      # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 4
# If the best chunk's similarity score falls below this, we treat the
# question as "not covered" by our knowledge base and refuse to answer.
MIN_SIMILARITY_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    """
    Resolves a secret from, in order: environment variable (local .env via
    python-dotenv, or any host that injects env vars), then Streamlit's own
    secrets store (used by Streamlit Community Cloud's "Secrets" panel).
    """
    value = os.environ.get(key, "")
    if value:
        return value
    try:
        import streamlit as st  # imported lazily - not a dependency for CLI scripts

        return st.secrets.get(key, default)
    except Exception:
        return default


GROQ_API_KEY = _get_secret("GROQ_API_KEY")
GROQ_MODEL = _get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are IITB Insti-Assist, an academic-rules assistant for \
students of IIT Bombay. You answer ONLY using the CONTEXT provided below, \
which is extracted from official IIT Bombay academic documents (course \
info booklet, academic calendar, timetable, fee circulars, procedures, \
rules for medals/prizes, etc).

Rules you must follow strictly:
1. Answer using only facts present in the CONTEXT. Do not use outside \
knowledge, do not guess, and do not make anything up.
2. If the CONTEXT does not contain enough information to answer the \
question, reply exactly with: "I don't know based on the available IIT \
Bombay academic documents." Do not try to be helpful by guessing.
3. When you do answer, be precise and cite which source document(s) the \
answer relies on (you will be given source names alongside each context \
chunk - refer to them by name).
4. Keep answers concise and student-friendly, using bullet points for \
multi-part rules (e.g. grade lists, categories, dates, fee amounts).
"""
