"""
Ingestion step: parse every PDF found in data/raw/ into text, clean it up,
and split it into overlapping chunks suitable for embedding.

Any PDF placed in data/raw/ is picked up automatically - there's no fixed
list of expected filenames, so you can add/remove/rename source documents
just by changing what's in that folder and re-running this script.

Output: data/processed/chunks.jsonl, one JSON object per chunk:
    {
        "chunk_id": "some-document_0007",
        "source_id": "some-document",
        "source_title": "Some Document",
        "page": 12,
        "text": "..."
    }

Usage:
    python scripts/ingest.py
"""

import json
import re
import sys
from pathlib import Path

import pdfplumber

sys.path.append(str(Path(__file__).resolve().parent.parent))
from rag.config import (  # noqa: E402
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    CHUNKS_PATH,
    PROCESSED_DIR,
    RAW_DIR,
)

WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
TITLE_SEP_RE = re.compile(r"[_\-]+")
TITLE_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Collapse repeated whitespace/newlines left behind by PDF extraction."""
    text = WHITESPACE_RE.sub(" ", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def prettify_title(stem: str) -> str:
    """Turns a filename stem like 'Hostel_Fee_Circular_New' into a readable title."""
    title = TITLE_SEP_RE.sub(" ", stem)
    title = TITLE_SPACE_RE.sub(" ", title).strip()
    return title


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text) for a PDF file."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = clean_text(raw)
            if cleaned:
                pages.append((i, cleaned))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Simple sliding-window chunker over characters, but it snaps chunk
    boundaries to the nearest sentence/paragraph break where possible so
    chunks don't end mid-sentence. This keeps retrieval results readable
    when we show them back to the user as "sources".
    """
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)

        if end < n:
            # try to snap to the last sentence boundary within the window
            window = text[start:end]
            boundary = max(window.rfind(". "), window.rfind(".\n"), window.rfind("\n\n"))
            if boundary > size * 0.4:  # only snap if it doesn't shrink the chunk too much
                end = start + boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)  # ensure forward progress

    return chunks


def build_chunks() -> list[dict]:
    all_chunks = []
    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"[warn] no PDFs found in {RAW_DIR}. Add PDFs there and re-run.")
        return all_chunks

    for pdf_path in pdf_paths:
        source_id = pdf_path.stem
        source_title = prettify_title(source_id)

        print(f"[parse] {source_title} ({pdf_path.name})")
        pages = extract_pages(pdf_path)
        chunk_counter = 0

        for page_num, page_text in pages:
            for piece in chunk_text(page_text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS):
                chunk_counter += 1
                all_chunks.append(
                    {
                        "chunk_id": f"{source_id}_{chunk_counter:04d}",
                        "source_id": source_id,
                        "source_title": source_title,
                        "page": page_num,
                        "text": piece,
                    }
                )
        print(f"        -> {chunk_counter} chunks from {len(pages)} pages")

    return all_chunks


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()

    if not chunks:
        print(f"\nNo chunks produced. Add PDFs to {RAW_DIR} and re-run.")
        return

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(chunks)} chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()

