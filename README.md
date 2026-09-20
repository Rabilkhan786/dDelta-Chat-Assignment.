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
uv run python main.py run --question "What changed on PSV-9066?"
```

This one command ingests both revisions, writes reports, builds the index, and
answers the question under one request ID. Use `run` without `--question` for
comparison only, or `chat "your question"` to reuse the latest index.

The first run downloads the local embedding and cross-encoder models.
Copy `.env.example` to `.env` and set `GROQ_API_KEY` before using real LLM
answers. The delta and retrieval steps work without a Groq key; chat returns
retrieved evidence and a clear provider failure if the key is missing.

Useful commands:

```bash
make run       # native/OCR routing → canonical JSON → report → index
make demo      # same pipeline, then a cited question under one request ID
make chat      # one grounded question
make test      # unit and integration tests
make check     # lint, unused imports, and formatting checks
make format    # apply import ordering and consistent formatting
make eval      # labelled delta, retrieval, and chat scorecard
make serve     # optional FastAPI demo at http://127.0.0.1:8000/docs
```

On Windows without Make, use the equivalent `uv run` commands in the Makefile.
Configuration lives in `src/config/config.yaml`; credentials belong only in `.env`.

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
**line**, not a native-PDF block or a single OCR word. Native extraction and
OCR can still split or read the same line differently; shared granularity does
not eliminate those errors. Each element keeps its page, bounding box, source,
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
The configured threshold is 0.60: the supplied unrelated pair scores 0.465,
whereas the default revision pair scores 0.989. Those two observations are
smoke checks, not calibration; heavy OCR errors can trigger false warnings.

The report is written to:

- `data/reports/delta_report.json`
- `data/reports/delta_report.md`
- `data/reports/revision_b_markup.pdf` (a simple bounding-box overlay)

Entries include type, page, bounding box, confidence, current/previous element
IDs, and location-change metadata. The JSON report is the authoritative
machine-readable artifact.
Matched-element confidence uses the weaker confidence of the two revisions,
so low-confidence OCR in A is not hidden by a clean native PDF in B.

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
can be disabled for a faster, RRF-only demo. `retrieval.top_k` is the single final
result limit; `candidate_k` controls the short list sent to the reranker.

### Query handling: no rewriting

The retriever uses the user's question as written. There is no query-rewrite,
query-expansion, synonym-generation, multi-query agent, or conversation-memory
step.

BM25 tokenization handles common technical identifier formatting without
changing the query itself. For example, compact, spaced, and hyphenated forms
such as `PSV9066A`, `PSV 9066A`, and `PSV-9066A` expose compatible lexical
tokens. Chroma semantic search and the cross-encoder receive the original
question unchanged.

A small source preference remains inside retrieval: change questions prefer the
delta report, explicit Revision A/B questions prefer that revision, and
comparison questions search all sources equally. This is an RRF score boost,
not a hard filter.

### Unsupported questions and citations

Keyword retrieval requires actual token overlap and semantic retrieval uses a
configured distance-based cutoff. The cross-encoder only reorders the short
post-RRF candidate list; it does not apply a second rejection threshold. The
distance conversion and reranker score are ranking heuristics, **not
probabilities**.

With no evidence, chat returns `status: unsupported` without an LLM call. After
generation, every bracketed citation must match supplied evidence, and at least
one citation is required. Only citations actually used are returned. This checks
source provenance, not whether every claim is entailed by its citation.

The Groq provider sits behind the single-method `ChatProvider.complete()`
interface and is used only for answer generation. Provider failure returns
`status: provider_error`; available evidence references are diagnostic, not a
successful answer.

## Optional HTTP API

`make serve` starts a single FastAPI process. It is a thin wrapper around the
same pipeline, not a microservice system.

- `POST /compare` accepts `revision_a`, `revision_b`, and `adapter`.
- `POST /chat` accepts `question`.
- `GET /report` returns the latest report.
- `GET /health` confirms the API is running.

See the interactive API documentation at `/docs` after starting it.
This is a trusted-local, single-pair demo: each comparison replaces the active
index and report. Do not expose it publicly or run concurrent comparisons; it
has no authentication, upload isolation, or per-user storage.

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
For delta and retrieval only, use `uv run python -m eval.run_eval --skip-generation`.
Without a configured provider, generation is explicitly reported as not run.

The scorecard separates:

- delta precision, recall, and F1 against the existing labelled change IDs;
- retrieval Recall@K and MRR against labelled relevant excerpt IDs;
- generated-answer keyword correctness;
- citation precision and coverage against expected citation fragments.

The current file has only three expected changes and one QA case. Existing labels
are preserved, but their historical human-review status has not been independently
verified in this cleanup. Keyword coverage is not semantic answer correctness;
source-fragment matching is not factual entailment. This is a regression smoke
test, not proof of general accuracy. See [eval/README.md](eval/README.md) for metric
definitions, review steps, and remaining gaps.

Known failures stay in the dataset. In particular, a major re-layout can still
confuse the simple line matcher, OCR can misread small dense drawing text, and
document-token overlap is only a warning—not a revision guarantee.
The supplied dense scanned/native same-content pair produced 853 false changes
in the local stress check; it is **not a supported accuracy benchmark**. See
[eval/RESULTS.md](eval/RESULTS.md) for the measured results and generation gap.

## Repository map

```text
src/
  canonical/       shared Pydantic representation and JSON writer
  ingest/          native PDF, scanned OCR, routing, DWG seam, line/classifier helpers
  delta/           compatibility check, alignment, deterministic delta, reports
  chat/            hybrid retrieval, reranking, provider, grounded answers
  config/          YAML defaults and typed settings
  markup/          optional PDF bounding-box overlay
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

Read [DEMO.md](DEMO.md) for the walkthrough and
[ASSIGNMENT_CHECKLIST.md](ASSIGNMENT_CHECKLIST.md) for the HTML acceptance review,
including unvalidated and incomplete items.
