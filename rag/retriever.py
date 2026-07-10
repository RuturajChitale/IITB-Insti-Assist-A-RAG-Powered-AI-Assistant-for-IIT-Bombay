"""
Loads the FAISS index built by scripts/build_index.py and exposes a
simple `retrieve(query, top_k)` function that returns the most relevant
chunks along with a cosine-similarity score for each.
"""

import json
import threading
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL_NAME, INDEX_META_PATH, INDEX_PATH, TOP_K


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_id: str
    source_title: str
    page: int
    text: str
    score: float


class Retriever:
    """
    Thin wrapper around a FAISS index + the sentence-transformer used to
    embed queries. Loading the embedding model is the slow part, so we
    keep a single lazily-initialized instance (see get_retriever() below)
    instead of reloading it on every Streamlit rerun.
    """

    def __init__(self):
        if not INDEX_PATH.exists() or not INDEX_META_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run scripts/ingest.py and "
                "scripts/build_index.py before starting the app."
            )

        self.index = faiss.read_index(str(INDEX_PATH))
        with open(INDEX_META_PATH, encoding="utf-8") as f:
            self.chunk_meta = json.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.chunk_meta[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    source_id=meta["source_id"],
                    source_title=meta["source_title"],
                    page=meta["page"],
                    text=meta["text"],
                    score=float(score),
                )
            )
        return results


_retriever_lock = threading.Lock()
_retriever_instance: Retriever | None = None


def get_retriever() -> Retriever:
    """Lazy singleton so Streamlit reruns don't reload the model/index each time."""
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                _retriever_instance = Retriever()
    return _retriever_instance
