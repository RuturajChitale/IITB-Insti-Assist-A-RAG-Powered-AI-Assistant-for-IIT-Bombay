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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are IITB Insti-Assist, an academic-rules assistant for \
students of IIT Bombay. You answer ONLY using the CONTEXT provided below, \
which is extracted from official IIT Bombay academic documents (UG/PG/PhD \
rulebooks, grading policy, academic calendar).

Rules you must follow strictly:
1. Answer using only facts present in the CONTEXT. Do not use outside \
knowledge, do not guess, and do not make anything up.
2. If the CONTEXT does not contain enough information to answer the \
question, reply exactly with: "I don't know based on the available IIT \
Bombay academic documents." Do not try to be helpful by guessing.
3. When you do answer, be precise and cite which source document(s) the \
answer relies on (you will be given source names alongside each context \
chunk - refer to them by name, e.g. "According to the UG Rule Book...").
4. Keep answers concise and student-friendly, using bullet points for \
multi-part rules (e.g. grade lists, categories, dates).
"""

# ---------------------------------------------------------------------------
# Source documents (official, public IIT Bombay academic documents)
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "id": "ug_rulebook",
        "title": "UG Rule Book (B.Tech/B.S./Dual Degree Rules & Regulations)",
        "url": "https://acad.iitb.ac.in/files/UG_RULE_BOOK.pdf",
    },
    {
        "id": "phd_rules",
        "title": "Ph.D. Programme Rules & Regulations",
        "url": "https://acad.iitb.ac.in/files/phdRules_3.pdf",
    },
    {
        "id": "mtech_rules",
        "title": "M.Tech./M.Phil./Dual Degree PG Rules & Regulations",
        "url": "https://www.iitb.ac.in/newacadhome/MTechRulesupdate201805July.pdf",
    },
    {
        "id": "meng_rules",
        "title": "M.Eng. Programme Rules",
        "url": "https://www.iitb.ac.in/newacadhome/MEngRules.pdf",
    },
    {
        "id": "academic_calendar_2025_26",
        "title": "Academic Calendar 2025-26",
        "url": "https://acad.iitb.ac.in/sites/default/files/Academic%20Calendar%202025-26_FINAL.pdf",
    },
    {
        "id": "grading_policy",
        "title": "Grading Policy Summary (Gymkhana Students' Council Portal)",
        "url": "https://gymkhana.iitb.ac.in/~scp/scp/pdfs/grading.pdf",
    },
]
