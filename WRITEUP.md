# Project Write-up: IITB Insti-Assist

**Scope:** Academic Assistant
**Author:** Ruturaj, B.Tech Chemical Engineering, IIT Bombay
**Project:** WnCC Learners' Space — Machine Learning (LS-26), Final Project

---

## 1. Chosen scope and why

I chose the **Academic Assistant** scope over Hostel/Campus, Council/Club,
or a General assistant for three reasons:

1. **Document quality and stability.** IIT Bombay's Academic Office
   publishes its rulebooks and calendars as clean, well-structured,
   text-based PDFs (not scanned images), which makes chunking and
   retrieval far more reliable than scraping less-structured sources like
   council websites or hostel notice boards.
2. **High-stakes correctness.** Grading, CPI/SPI calculation, and academic
   category rules are exactly the kind of information where a wrong guess
   from a generic LLM is actively harmful — a student who trusts a
   hallucinated answer about, say, Category III registration rules could
   make a costly mistake. This makes the "refuse if not grounded"
   requirement of the project genuinely meaningful, not just a formality.
3. **Personal relevance.** As a second-year student who has already had to
   parse these rulebooks for course registration, I know firsthand how
   dense and hard-to-search the actual documents are — this is a tool I'd
   use myself.

## 2. Data sources used

Six official, publicly available IIT Bombay academic documents (see
`rag/config.py` for live URLs, and `README.md` for the full list):

| # | Document | Why it's included |
|---|----------|-------------------|
| 1 | UG Rule Book | Core B.Tech/Dual Degree grading, registration category, course-drop rules |
| 2 | Ph.D. Programme Rules | Extends coverage to research-scholar academic rules |
| 3 | M.Tech./PG Rules | Extends coverage to Master's programme rules |
| 4 | M.Eng. Programme Rules | Covers the working-professional M.Eng. track |
| 5 | Academic Calendar 2025-26 | Concrete dates: registration windows, exam dates, feedback periods |
| 6 | Grading Policy Summary | A shorter, more skimmable restatement of the grade-point table, useful as a cross-check against the UG Rule Book |

These are downloaded at setup time (`scripts/download_sources.py`) rather
than committed to the repo, both to keep the repo small and to always
ingest the current official version rather than a copy that could go
stale.

## 3. Chunking strategy and why

Documents are parsed page-by-page with `pdfplumber`, then split with a
**character-based sliding window (1000 chars, 150-char overlap)** that
snaps to the nearest sentence or paragraph boundary within the window
rather than cutting mid-sentence. I considered a few alternatives:

- **Fixed-size chunking with no sentence-snapping** — simplest, but often
  cuts a rule definition (e.g. a grade-category description) exactly in
  half, which hurts both retrieval quality and the readability of the
  "sources" shown to the user.
- **Paragraph-based chunking** — respects document structure better, but
  these rulebooks have very unevenly sized paragraphs (some are a single
  clause, others are a full page of numbered sub-rules), which would
  produce wildly inconsistent chunk sizes and hurt embedding quality.
- **Chosen approach (sentence-snapped sliding window)** — a middle ground:
  consistent chunk sizes for stable embeddings, while avoiding the worst
  mid-sentence cuts. The overlap (150 chars) means a rule that happens to
  fall near a chunk boundary is very likely to still appear whole in at
  least one chunk.

Each chunk retains its **source document title and page number** as
metadata, which is what powers the "Sources" panel in the UI (functional
requirement #6).

## 4. Retrieval, grounding, and generation

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` — small, fast,
  good enough quality for a focused, single-domain corpus like this one,
  and it runs on CPU without needing a GPU or API key.
- **Vector index:** FAISS `IndexFlatIP` over L2-normalized vectors, which
  makes inner product equivalent to cosine similarity. Exact (flat) search
  is used rather than an approximate index since the corpus is only a few
  thousand chunks — well within the range where exact search is still
  fast.
- **Groundedness gate:** before ever calling the LLM, the top retrieved
  chunk's cosine similarity is checked against a threshold
  (`MIN_SIMILARITY_THRESHOLD = 0.30`). Below it, the app returns "I don't
  know" immediately — this is what makes the assistant correctly refuse
  out-of-scope questions (e.g. about hostel mess bills) instead of forcing
  the LLM to improvise from irrelevant context.
- **Generation:** Groq's hosted Llama 3.3 70B is given the retrieved
  chunks (labelled by source + page) and a system prompt that explicitly
  instructs it to answer only from context and to say "I don't know" if
  the context is insufficient — a second layer of grounding on top of the
  similarity gate, since even relevant-looking chunks sometimes don't
  actually answer the specific question asked.

## 5. Known limitations / what I'd improve with more time

- **Table extraction is imperfect.** `pdfplumber`'s plain-text extraction
  flattens multi-column tables (e.g. the full grade-point table) into
  fairly jumbled text. A dedicated table-extraction pass (or an
  OCR/layout-aware parser) would improve answers to questions that hinge
  on tabular data specifically.
- **The similarity threshold is a heuristic, not a calibrated confidence
  score.** It was tuned by manual spot-checking rather than a labelled
  validation set. A small held-out set of "in-scope" vs. "out-of-scope"
  questions would let this be tuned properly and reported with real
  precision/recall numbers.
- **No cross-document conflict detection.** If two rulebooks describe an
  overlapping rule slightly differently (e.g. UG vs PG re-examination
  policy), the assistant currently just answers from whichever chunks
  rank highest, without flagging the discrepancy explicitly.
- **Single-domain scope by design.** The assistant will correctly say "I
  don't know" for anything outside academic rules (hostels, clubs,
  placements) — this is intentional per the project scope, but a real
  "General Insti Assistant" would need per-domain routing to multiple
  corpora.
- **Stretch goals not attempted:** live user PDF upload, and highlighting
  the exact quoted sentence within a source chunk (currently the whole
  chunk is shown, which is coarser than exact-quote highlighting).
