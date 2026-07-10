"""
Embeds every chunk in data/processed/chunks.jsonl with a sentence-transformer
model and builds a FAISS index for fast cosine-similarity search.

Output:
    data/processed/faiss.index      - the FAISS index (vectors only)
    data/processed/faiss_meta.json  - list of chunk metadata, same order as
                                       vectors in the index, so index i in
                                       FAISS <-> faiss_meta.json[i]

Usage:
    python scripts/build_index.py
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from rag.config import (  # noqa: E402
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    INDEX_META_PATH,
    INDEX_PATH,
)


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run scripts/ingest.py first."
        )
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (first run downloads it)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nIndexed {index.ntotal} vectors (dim={dim}).")
    print(f"Saved index      -> {INDEX_PATH}")
    print(f"Saved chunk meta -> {INDEX_META_PATH}")


if __name__ == "__main__":
    main()
