"""
IITB Insti-Assist - Streamlit front-end.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from rag.config import INDEX_META_PATH, INDEX_PATH  # noqa: E402
from rag.pipeline import ask  # noqa: E402

st.set_page_config(page_title="IITB Insti-Assist", page_icon="🎓", layout="centered")

st.title("🎓 IITB Insti-Assist")
st.caption(
    "A RAG-powered academic assistant for IIT Bombay — answers grounded in "
    "official UG/PG/PhD rulebooks, grading policy, and the academic calendar."
)


@st.cache_resource(show_spinner=False)
def bootstrap_knowledge_base() -> None:
    """
    On a fresh deployment (e.g. Streamlit Community Cloud) there's no way
    to run the setup scripts by hand first, so build the index once here,
    the first time the app boots. @st.cache_resource means this only runs
    once per running instance, not on every rerun.
    """
    from scripts import build_index, download_sources, ingest

    download_sources.download_all()
    ingest.main()
    build_index.main()


if not INDEX_PATH.exists() or not INDEX_META_PATH.exists():
    with st.spinner(
        "First-time setup: downloading official IITB documents, chunking "
        "them, and building the search index. This can take a couple of "
        "minutes and only happens once."
    ):
        bootstrap_knowledge_base()
    st.rerun()

with st.sidebar:
    st.header("About this scope")
    st.markdown(
        "**Scope:** Academic Assistant\n\n"
        "Covers course registration, grading/CPI rules, academic "
        "categories, examination rules, and the academic calendar.\n\n"
        "The assistant will say **\"I don't know\"** if your question "
        "isn't covered by the ingested documents (e.g. hostel rules, "
        "placements) — it will not guess."
    )
    st.header("Source documents")
    from rag.config import SOURCES  # noqa: E402

    for src in SOURCES:
        st.markdown(f"- [{src['title']}]({src['url']})")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, RAGResult)

question = st.chat_input("Ask about IITB academic rules, grading, or the calendar…")

for past_q, past_result in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(past_q)
    with st.chat_message("assistant"):
        st.markdown(past_result.answer)
        if past_result.sources:
            with st.expander(f"📄 Sources ({len(past_result.sources)})"):
                for s in past_result.sources:
                    st.markdown(
                        f"**{s.source_title}** — page {s.page} "
                        f"(similarity {s.score:.2f})"
                    )
                    st.caption(s.text[:400] + ("…" if len(s.text) > 400 else ""))

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant rules and asking the LLM…"):
            try:
                result = ask(question)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        st.markdown(result.answer)
        if result.sources:
            with st.expander(f"📄 Sources ({len(result.sources)})"):
                for s in result.sources:
                    st.markdown(
                        f"**{s.source_title}** — page {s.page} "
                        f"(similarity {s.score:.2f})"
                    )
                    st.caption(s.text[:400] + ("…" if len(s.text) > 400 else ""))
        else:
            st.info(
                "No confident match was found in the academic documents, "
                "so no sources are shown for this answer."
            )

    st.session_state.history.append((question, result))
