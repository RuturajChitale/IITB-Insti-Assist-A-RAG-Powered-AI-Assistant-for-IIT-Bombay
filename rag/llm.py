"""
Thin wrapper around the Groq API (Llama models) for the generation step.
Groq is used because it has a generous free tier and doesn't hit the quota
walls that Gemini's free tier does.
"""

from groq import Groq

from rag.config import GROQ_API_KEY, GROQ_MODEL, SYSTEM_PROMPT
from rag.retriever import RetrievedChunk


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Formats retrieved chunks into a labelled context block for the prompt."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {c.source_title}, page {c.page}]\n{c.text}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Calls the Groq chat completion endpoint with the retrieved context
    injected into the prompt, and returns the model's grounded answer.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or export it in your shell before running the app."
        )

    if not chunks:
        # No relevant context at all - don't even call the LLM, just refuse.
        return "I don't know based on the available IIT Bombay academic documents."

    context_block = build_context_block(chunks)

    user_prompt = f"""CONTEXT:
{context_block}

QUESTION: {question}

Answer using ONLY the CONTEXT above. If it's insufficient, say you don't know."""

    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temperature: we want faithful, non-creative answers
        max_tokens=700,
    )
    return completion.choices[0].message.content.strip()
