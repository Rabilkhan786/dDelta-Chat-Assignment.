# Delta Chat

Compares two revisions of an engineering document (native PDF or scanned PDF),
computes a structured delta between them, writes a delta report, and answers
questions about either revision or the delta with cited, grounded chat.

## How to run

### CLI

```bash
uv sync
uv run python main.py run
uv run python main.py chat "What changed near the compressor?"
uv run python -m eval.run_eval
```

Requirements: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and the
`tesseract-ocr` system package (`sudo apt-get install tesseract-ocr` on
Debian/Ubuntu, `brew install tesseract` on macOS) for scanned-PDF support.
Chat needs a free [Groq](https://console.groq.com) API key — copy
`.env.example` to `.env` and set `GROQ_API_KEY`.

`run` ingests the two configured PDFs, writes their canonical JSON to
`data/output/`, writes the delta report to `data/reports/`, and builds a
Chroma retrieval index in `data/chroma_db/` (downloads a small embedding
model the first time it runs, so it needs network access once).

### Web UI (FastAPI + Streamlit)

The same pipeline is also exposed as a small web app: upload any two PDF
revisions (native text or scanned/OCR — auto-detected, same as the CLI's
`--adapter auto`) through the browser instead of editing `config.yaml`.

```bash
uv sync
uv run uvicorn src.api.main:app --reload --port 8000   # terminal 1: backend
uv run streamlit run streamlit_app.py                   # terminal 2: frontend
```

or with `make`:

```bash
make api   # terminal 1
make ui    # terminal 2
```

Open the Streamlit URL it prints (defaults to `http://localhost:8501`),
upload Revision A and Revision B as PDFs, and click **Process documents**.
That calls `POST /api/upload` on the FastAPI backend, which runs the exact
same `DeltaPipeline` as `main.py run` (ingest → align → delta → report →
Chroma index) on the uploaded files, so both a native PID and a scanned/OCR
PID work. Once processing finishes, the delta summary and changed entries
are shown, and the chat panel below is ready — each question goes through
`POST /api/chat`, which runs `GroundedChatService` and returns a cited
answer exactly like `main.py chat`.

Backend endpoints:

- `GET /api/health` — `{"status": "ok", "ready": bool}`.
- `POST /api/upload` — multipart form with `revision_a` and `revision_b` PDF
  files; returns the delta summary, entries, and indexed excerpt count.
- `GET /api/report` — the last generated delta report (JSON).
- `POST /api/chat` — `{"question": "..."}` → `{"answer": "...", "citations": [...]}`.

The Streamlit app reads the backend URL from `DELTA_CHAT_API_URL` (defaults
to `http://localhost:8000`), so the two processes can be deployed separately
(e.g. FastAPI on one host, Streamlit on another) by setting that env var.

## Architecture

```
PDF (native or scanned) -> ingest adapter -> CanonicalDocument
                                                   |
                                    align elements (RapidFuzz + page/type/position)
                                                   |
                                  classify added / removed / modified + confidence
                                                   |
                                  delta report (Markdown + JSON)  -----.
                                                   |                   |
                        canonical text + delta report  --------> Chroma index
                                                                        |
                                              question -> retrieve top-k -> LLM -> cited answer
```

- **`src/ingest/`** — one `FormatAdapter` interface, two working implementations.
  `pdf_native.py` reads the text/vector layer directly with PyMuPDF.
  `pdf_scanned.py` renders each page to an image and runs **Tesseract OCR**
  (`pytesseract`), keeping per-word bounding boxes and confidence.
  `registry.py` picks between them automatically by counting selectable
  characters. `dwg.py` is a real stub behind the same interface (raises a
  clear "not implemented" error) so a real DWG parser can be added later
  without touching anything downstream.
- **`src/canonical/model.py`** — the format-agnostic model
  (`CanonicalDocument` → `Page` → `Element`) everything else works against.
  This is the seam that makes ingestion pluggable.
- **`src/delta/`** — `align.py` matches elements between revisions by page,
  type, nearby position, and RapidFuzz text similarity (matching is the hard
  part, not diffing). `engine.py` turns matches/non-matches into
  `added`/`removed`/`modified` entries with a confidence score (OCR matches
  are discounted by OCR confidence). No LLM in this path — it's deterministic
  and reproducible. `report.py` renders Markdown and JSON.
- **`src/chat/`** — `index.py` embeds PID A, PID B, and every delta entry as
  separate, source-labelled excerpts into a Chroma vector store
  (`langchain-chroma` + a local `sentence-transformers` embedding model).
  `answer.py` retrieves the top-k excerpts for a question, builds a prompt
  that requires a citation for every claim, and calls the configured LLM
  (`llm.py`, currently Groq). If retrieval finds nothing, chat says so
  instead of guessing; if the LLM call fails, the retrieved citations are
  still returned.
- **`src/observability/logging.py`** — JSON logs with a shared request ID and
  a `stage()` context manager that times and logs the start/end/failure of
  every pipeline stage (ingest, alignment, delta, report, retrieval, LLM
  call). See below for why this over a dedicated tracing SDK.
- **`src/api/main.py`** — a thin FastAPI wrapper (`/api/upload`, `/api/report`,
  `/api/chat`, `/api/health`) around `DeltaPipeline` and `GroundedChatService`
  so the identical pipeline runs on uploaded files instead of the paths in
  `config.yaml`. No pipeline logic lives here.
- **`streamlit_app.py`** — the browser UI: upload two PDFs, call the API to
  process them, show the delta summary/table, then a chat panel over the
  result.

## Design decisions & trade-offs

- **Deterministic delta, LLM only for chat.** The delta engine is plain
  RapidFuzz + geometry — no LLM. That makes it reproducible and cheap to run
  in eval, and keeps the one non-deterministic part of the system (the LLM)
  isolated to the final answer-generation call.
- **One retriever, not a hybrid.** The chat index is a single Chroma
  similarity search. An earlier version of this project combined BM25 and
  semantic search with Reciprocal Rank Fusion — it worked, but it was a lot
  of bespoke code (a hand-rolled tokenizer, RRF math, two indexes to keep in
  sync) to maintain for a two-document assignment. LangChain + Chroma cover
  semantic retrieval in a few lines; see "what's next" for when hybrid would
  earn its complexity back.
- **Tesseract over a heavier OCR stack.** PaddleOCR/EasyOCR give slightly
  better accuracy but pull in large ML dependencies. Tesseract is a single
  system package, has a stable Python wrapper (`pytesseract`), and its `TSV`
  output already gives per-word bounding boxes and confidence — exactly what
  the canonical model needs.
- **Groq for the LLM.** Fast, has a free tier, and `llama-3.1-8b-instant` is
  plenty for "answer from this evidence, cite it." Swapping providers means
  implementing the two-method `ChatProvider` protocol in `src/chat/llm.py`.

## What I cut, and why

- **DWG** is a real stub (`src/ingest/dwg.py`) behind the same
  `FormatAdapter` interface as the PDF adapters, but does not parse DWG
  files. Native PDF + scanned PDF (via Tesseract) are the two formats
  demonstrated end-to-end, which satisfies "at least two of three" with room
  to add a real DWG parser (e.g. `ezdxf`) later without touching delta/chat.
- **Delta markup (bonus)** is not implemented — `src/markup/` is an empty
  seam. Given the time budget, a genuinely useful delta engine, grounded
  chat, observability, and eval mattered more than a visual overlay.
- **Hybrid BM25 + semantic retrieval** was cut in favor of plain semantic
  search (see above) — less code, easier to reason about, and precision
  wasn't a bottleneck at this scale (two documents, ~500 excerpts).
- **Langfuse-based tracing** was cut in favor of the homegrown JSON `stage()`
  logger. Langfuse is a fine choice, but wiring in a second observability
  tool on top of structured JSON logs would have been redundant for a system
  this size — see "Observability" below.

## Observability

Every pipeline stage (ingest, alignment, delta classification, report
generation, retrieval, LLM call) is wrapped in
`src/observability/logging.py`'s `stage()` context manager, which logs a
`stage_started` and `stage_completed`/`stage_failed` JSON record with a
duration in milliseconds. All logs in one `run` or `chat` invocation share a
`request_id` (bound via `request_context()`), so `grep`-ing one ID in
`logs/project.log` gives the full trace of that request. The LLM call
additionally logs the model name, input/output token counts, and an
estimated cost computed from the per-token prices in `config.yaml`. Failures
(a corrupt PDF, a missing Tesseract binary, a failed LLM call) are logged
with `stage_failed`/`llm` exception details and never silently swallowed —
`GroundedChatService` still returns the retrieved citations if the LLM call
itself fails.

This is a homegrown tracer rather than OpenTelemetry/Langfuse because the
system is small enough that one JSON-lines log file with a request ID
already answers "what did this request do and how long did each step take,"
and it adds zero new services to run. The trade-off: no built-in UI for
browsing traces — `logs/project.log` is `jq`-able but not a dashboard. See
"what's next."

## Evaluation

```bash
uv run python -m eval.run_eval
```

- **Delta metrics.** `eval/run_eval.py` derives a stable ID for every
  non-`unchanged` delta entry from the generated `delta_report.json` and
  compares it against `eval/datasets/ground_truth.json`'s
  `expected_change_ids` (precision/recall/F1). Run with
  `--write-candidates` first to generate candidate IDs, review them against
  the report, and paste approved ones into `ground_truth.json` — the
  harness never fabricates labels.
- **Chat metrics.** For each `qa_cases` entry (question + expected keywords
  + expected citation fragments like `"pid_b | page 1"`), the harness runs
  real grounded chat and scores answer correctness (are the expected
  keywords in the answer?) and citation accuracy (do the citations mention
  the expected source/page?).
- **On the default sample pair** (`data/samples/synthetic_revision/`, see
  provenance below), the engine finds exactly the 3 edits that were made —
  1 modified (a renamed PSV tag), 1 added (a new note), 1 removed (a deleted
  callout) — against 494 unchanged elements. `eval/datasets/ground_truth.json`
  is filled in with those 3 IDs and one QA case.
- **Known failure case**, documented honestly in `ground_truth.json`: if
  content moves further than `align.max_bbox_distance` (75pt) between
  revisions, alignment reports it as a remove + add instead of a modify. A
  larger search radius or a second matching pass keyed on text similarity
  alone (ignoring position) would catch this, at the cost of more false
  positive matches.

## Sample data

Three pairs, generated/documented in `data/samples/` (see each folder's
`PROVENANCE.md`):

1. **`synthetic_revision/`** — the supplied `Lift Gas compressor-P&ID.pdf` as
   Revision A, plus a Revision B with 3 concrete edits applied by
   `data/samples/make_samples.py` (regenerate with
   `uv run python data/samples/make_samples.py`). This is the default pair
   and the one eval is scored against, since it's a genuine revision pair.
2. **`different_systems/`** — the two originally supplied P&IDs, which
   turned out to be two *different* compressor systems, not two revisions of
   one document. Kept as an honest edge-case: the delta on this pair is
   large and not meaningful, which is the correct behavior for unrelated
   inputs.
3. **`scanned/`** — page 1 of Revision A rasterized with no text layer, to
   exercise `ScannedPDFAdapter` end-to-end with real Tesseract OCR.

No key or PII is committed; `.env.example` lists the one required variable.

## What's next

- **DWG support** via `ezdxf`, mapping entities/blocks/text to the same
  `CanonicalDocument` model.
- **Delta markup** — draw the delta's bounding boxes back onto the PDF with
  PyMuPDF annotations.
- **Hybrid retrieval** if a larger document set shows semantic-only search
  missing exact tag/number lookups (BM25 is strong for that).
- **LLM-as-judge** for answer correctness, validated against a small
  human-labeled subset, to reduce how much of chat quality depends on exact
  keyword matches in eval.
- **A metrics dashboard** (or a served `/metrics` endpoint) instead of
  reading `logs/project.log` by hand.
