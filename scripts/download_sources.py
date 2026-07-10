"""
Downloads the real, official IIT Bombay academic-rules PDFs listed in
rag/config.py::SOURCES into data/raw/.

We deliberately do NOT commit the PDFs themselves to the git repo (they are
copyrighted institute documents, and the files are large) - instead, this
script re-downloads them from the official acad.iitb.ac.in / iitb.ac.in
URLs at setup time. This also means the assistant always ingests the
*current* version of each document.

Usage:
    python scripts/download_sources.py
"""

import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from rag.config import RAW_DIR, SOURCES  # noqa: E402

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IITB-Insti-Assist/1.0; "
        "academic-project; +https://github.com/)"
    )
}


def download_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []

    for src in SOURCES:
        dest = RAW_DIR / f"{src['id']}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {src['id']} already downloaded -> {dest.name}")
            ok += 1
            continue

        print(f"[fetch] {src['title']}\n        {src['url']}")
        try:
            resp = requests.get(src["url"], headers=HEADERS, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"        saved -> {dest} ({len(resp.content) / 1024:.1f} KB)")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"        FAILED: {exc}")
            failed.append(src)

    print(f"\nDone. {ok}/{len(SOURCES)} documents available in {RAW_DIR}/")
    if failed:
        print("\nCould not auto-download the following (network / link may have")
        print("changed). Please download them manually from the URL and save")
        print("under data/raw/<id>.pdf:")
        for src in failed:
            print(f"  - {src['id']}: {src['url']}")


if __name__ == "__main__":
    download_all()
