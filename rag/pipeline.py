"""
The full RAG pipeline: retrieve -> groundedness check -> generate.
This is the single function app.py (or any other frontend) needs to call.
"""

from dataclasses import dataclass, field

from rag.config import MIN_SIMILARITY_THRESHOLD, TOP_K
from rag.llm import answer_question
from rag.retriever import RetrievedChunk, get_retriever


@dataclass
class RAGResult:
    answer: str
    grounded: bool
    sources: list[RetrievedChunk] = field(default_factory=list)


def ask(question: str, top_k: int = TOP_K) -> RAGResult:
    retriever = get_retriever()
    chunks = retriever.retrieve(question, top_k=top_k)

    # Groundedness gate: if even the best match is a weak semantic match,
    # the question is almost certainly out of scope for our corpus, so we
    # refuse rather than let the LLM improvise from retrieved noise.
    best_score = chunks[0].score if chunks else 0.0
    is_grounded = best_score >= MIN_SIMILARITY_THRESHOLD

    if not is_grounded:
        return RAGResult(
            answer="I don't know based on the available IIT Bombay academic documents.",
            grounded=False,
            sources=[],
        )

    answer = answer_question(question, chunks)

    # Treat the LLM's own "I don't know" as ungrounded too, so the UI can
    # style it differently (and skip showing sources for it).
    refused = "i don't know" in answer.lower()[:60]

    return RAGResult(
        answer=answer,
        grounded=not refused,
        sources=[] if refused else chunks,
    )
