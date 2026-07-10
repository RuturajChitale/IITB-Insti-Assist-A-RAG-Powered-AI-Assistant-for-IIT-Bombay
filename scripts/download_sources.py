"""
NOTE: This project now expects PDFs to be placed directly in data/raw/
(and committed to the repo) rather than auto-downloaded, because IITB's
website blocks/redirects requests coming from cloud hosts like Streamlit
Community Cloud.

This script is kept only as a convenience for local use: it just reports
what's currently in data/raw/ so you can sanity-check before running
ingest.py. It does not fetch anything from the internet.

Usage:
    python scripts/download_sources.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from rag.config import RAW_DIR  # noqa: E402


def download_all() -> None:
    """
    Kept with this name so app.py / older instructions that call it still
    work - it no longer downloads anything, it just reports what's present.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))

    if not pdfs:
        print(
            f"No PDFs found in {RAW_DIR}.\n"
            "Add your source PDFs there (any filename works - ingest.py "
            "picks up every *.pdf automatically), then run:\n"
            "  python scripts/ingest.py\n"
            "  python scripts/build_index.py"
        )
        return

    print(f"Found {len(pdfs)} PDF(s) in {RAW_DIR}:")
    for p in pdfs:
        size_kb = p.stat().st_size / 1024
        print(f"  - {p.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    download_all()
