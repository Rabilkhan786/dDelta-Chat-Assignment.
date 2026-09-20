# Document Delta & Grounded Chat

Compare two revisions of an engineering document, write a structured delta
report, then answer questions from the two revisions and the report with exact
citations.

The project intentionally has a small, explainable design. It supports two
formats end to end:

- Native PDF: PyMuPDF reads the selectable text layer.
- Scanned PDF: PyMuPDF renders each page and Tesseract OCR reads it.

DWG is recognised through the same adapter interface, but is a deliberately
honest stub. It raises a clear `NotImplementedError`; it does not pretend to
parse a drawing. The assignment asks for two of three formats, and the PDF/OCR
pair is the chosen scope.

## Quick start

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Tesseract
available on your `PATH`. On Windows, install Tesseract and restart the shell.

```bash
uv sync
uv run python main.py run
uv run python main.py chat "What changed on PSV-9066?"
```

The first run downloads the local sentence-transformer model used by Chroma.
Copy `.env.example` to `.env` and set `GROQ_API_KEY` before using real LLM
answers. The delta and retrieval steps work without a Groq key; chat returns
retrieved evidence and a clear provider failure if the key is missing.

Useful commands:

```bash
make run       # native/OCR routing → canonical JSON → report → index
make chat      # one grounded question
make test      # unit and integration tests
make eval      # labelled delta, retrieval, and chat scorecard
make serve     # optional FastAPI demo at http://127.0.0.1:8000/docs
```

## What happens in a run

```text
Revision A PDF + Revision B PDF
             ↓
native text check → Native PDF adapter or OCR adapter
             ↓
line-level CanonicalDocument
             ↓
revision-compatibility warning + deterministic alignment
             ↓
added / removed / modified / moved delta entries
             ↓
JSON + Markdown report, then hybrid retrieval index
```

Every extractor produces `CanonicalDocument → Page → Element`. An element is a
**line**, not a native-PDF block or a single OCR word. This makes native and
scanned pages comparable. Each element keeps its page, bounding box, source,
and OCR confidence where relevant.

`src/ingest/classify.py` applies only simple, auditable rules:

- `dimension` for recognised units such as `10 bar` or `150 mm`;
- `note` for leading note/install/warning text;
- `symbol` for standalone technical tags such as `PSV-9066` or `P-101`;
- `title` for short, large native-PDF text;
- otherwise `text`.

It does **not** claim table-cell parsing, CAD geometry detection, or symbol
recognition from pixels.

## Delta logic

The delta path never calls an LLM. It is deterministic RapidFuzz plus simple
geometry:

1. Match same-page, same-type lines that are nearby and textually similar.
2. Match still-unmatched same-page lines only when their text similarity is
   very high. This catches labels that moved farther than the first position
   threshold.
3. Classify the result as `added`, `removed`, `modified`, `unchanged`, or
   `moved`. A moved entry has an unchanged label but a meaningfully different
   bounding-box position.

Before alignment, the pipeline calculates token-overlap document similarity.
Low overlap does not silently stop the run, but the report carries a visible
warning that the pair may be two different systems rather than revisions. This
is deliberately a warning signal, not proof of document identity.

The report is written to:

- `data/reports/delta_report.json`
- `data/reports/delta_report.md`
- `data/reports/revision_b_markup.pdf` (a simple bounding-box overlay)

Entries include type, page, bounding box, confidence, current/previous element
IDs, and location-change metadata. The JSON report is the authoritative
machine-readable artifact.

## Grounded chat and hybrid retrieval

The index contains one excerpt for every element in Revision A, Revision B,
and the delta report. Its metadata includes source, PID, revision, page,
element ID, element type, change type, bounding box, and confidence.

Retrieval is intentionally small and local:

```text
question
 ├─ BM25 keyword search       → exact tags, values, dimensions
 └─ Chroma semantic search    → natural-language meaning
                ↓
       Reciprocal Rank Fusion
                ↓
cross-encoder reranker (small candidate list)
                ↓
       cited evidence → Groq answer
```

BM25 is important for identifiers such as `PSV-9066`, `P-101`, `DN150`, and
`10 bar`; semantic search is useful for questions such as “what pressure
changes happened?”. A compact cross-encoder (`ms-marco-MiniLM-L-6-v2`) then
reranks only the post-RRF candidates. It is configurable in `config.yaml` and
can be disabled for a faster, RRF-only demo. The question router only *prefers* the delta report for
change questions or one PID for explicit revision questions. It never uses an
agent framework and comparison questions keep all sources available.

Weak results are dropped when neither keyword nor semantic retrieval clears the
configured quality threshold. With no evidence, chat refuses rather than asks
the model to guess. The Groq provider sits behind the two-method `ChatProvider`
interface and is used only after retrieval.

## Optional HTTP API

`make serve` starts a single FastAPI process. It is a thin wrapper around the
same pipeline, not a microservice system.

- `POST /compare` accepts `revision_a`, `revision_b`, and `adapter`.
- `POST /chat` accepts `question`.
- `GET /report` returns the latest report.
- `GET /health` confirms the API is running.

See the interactive API documentation at `/docs` after starting it.

## Observability

`src/observability/logging.py` writes JSON lines to `logs/project.log`. Each
pipeline or chat request receives a request ID. Stages emit start, completed,
or failed events with duration milliseconds. Groq calls record the model,
prompt, response, input/output tokens, and an estimated configured cost.
Errors such as bad PDFs, OCR failures, missing API credentials, and provider
errors remain visible in the same trace instead of being swallowed.

The implementation uses JSON logs rather than a hosted tracing service so the
demo needs no extra account or process. The trade-off is that traces are read
from a file rather than a dashboard.

## Evaluation

Run `make eval` after `make run`. The dataset lives in
`eval/datasets/ground_truth.json`; labels are not generated by the evaluator.
`--write-candidates` can create proposed delta IDs for a person to review, but
does not edit the ground truth.

The scorecard separates:

- delta precision, recall, and F1 against human-reviewed change IDs;
- retrieval Recall@K and MRR against labelled relevant excerpt IDs;
- generated-answer keyword correctness;
- citation accuracy/groundedness against expected citation fragments.

Known failures stay in the dataset. In particular, a major re-layout can still
confuse the simple line matcher, OCR can misread small dense drawing text, and
document-token overlap is only a warning—not a revision guarantee.

## Repository map

```text
src/
  canonical/       shared Pydantic representation and JSON writer
  ingest/          native PDF, scanned OCR, routing, DWG seam, line/classifier helpers
  delta/           compatibility check, alignment, deterministic delta, reports
  chat/            hybrid retrieval, prompt, swappable Groq provider, answer service
  observability/   structured JSON request traces
  api.py           optional FastAPI wrapper
eval/              label-driven metrics and scorecard
tests/             routing, normalization, delta, retrieval, chat, logging tests
data/samples/      sample pairs and provenance notes
```

## Scope cuts and next steps

This is deliberately not a CAD system. DWG parsing, table structure, image
symbols, and cross-page alignment are not implemented. The included PDF markup
only boxes the reliable text regions already found by the delta engine; it is
not a visual-diff or geometry-detection system. The next practical improvements
would be a reviewed DWG adapter, page matching for inserted cover sheets, and
better OCR/layout for dense drawings.

Read [DEMO.md](DEMO.md) for a short walkthrough and the acceptance-criteria
checklist.
