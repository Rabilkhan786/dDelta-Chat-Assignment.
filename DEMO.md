# Demo walkthrough

This is a two-to-four minute walkthrough of what the repository actually runs.

## 1. Install and compare the supplied revision pair

```bash
uv sync
uv run python main.py run
```

The command detects each input PDF. The supplied default pair has selectable
text, so it uses the native adapter. The resulting line-level canonical files
are saved under `data/output/`, and the report is written to
`data/reports/delta_report.json` and `.md`.

Open the JSON report and point out:

- the `revision_compatibility` score and any warning;
- summary counts for added, removed, modified, and moved elements;
- a changed entry's page, bounding box, confidence, element ID, and
  `location_changed` flag.

Open `data/reports/revision_b_markup.pdf` too. It adds simple colored boxes to
the changed text regions: green for added, red for removed, orange for modified,
and purple for moved. This is a bbox overlay, not a claimed pixel or CAD diff.

The compare path is deterministic: it does not call Groq or any other LLM.

## 2. Show scanned-PDF routing

```bash
uv run python main.py run \
  --revision-a data/samples/scanned/lift_gas_scanned.pdf \
  --revision-b data/samples/synthetic_revision/revision_a.pdf
```

The JSON trace records `ScannedPDFAdapter` for the raster-only input. OCR words
are grouped into lines before they become canonical elements, so they have the
same granularity as lines extracted from a native PDF. Each OCR element includes
a bounding box and confidence.

## 3. Show cited hybrid chat

```bash
cp .env.example .env
# Set GROQ_API_KEY in .env
uv run python main.py chat "What changed on PSV-9066?"
```

The retriever combines BM25 exact matching with Chroma semantic matching using
Reciprocal Rank Fusion, then re-ranks the short candidate list with a compact
cross-encoder. The final answer is prompted only with retrieved
evidence and returns citations in this stable form:

```text
[delta_report | PID revision_b | page 1 | delta-3]
```

Ask an unsupported question too. If nothing passes the retrieval quality
threshold, the service says it cannot support an answer instead of guessing.

## 4. Show a request trace

Open `logs/project.log` and filter one request ID. It contains structured JSON
for routing, ingestion, compatibility, alignment, delta, reporting, index
build, retrieval, and the LLM call. LLM records include model, prompt,
response, token counts, cost estimate, and failures.

## 5. Run the checks

```bash
make test
make eval
```

The evaluation command keeps document-delta metrics separate from retrieval
Recall@K/MRR and answer/citation scoring. Labels live in the repository and are
never fabricated by the evaluator. Known failure cases are printed from the
dataset.

## Assignment checklist

| Requirement | Implementation |
| --- | --- |
| Two real formats | Native PDF and scanned PDF/OCR |
| Format seam | `FormatAdapter`; DWG stub is honest and callable |
| Structured delta | Typed, located, confidence-scored JSON + Markdown + PDF bbox markup |
| Grounded chat | Hybrid retrieval over PID A, PID B, delta report; citations |
| Determinism | No LLM in compatibility, alignment, or delta classification |
| Observability | JSON request traces, stage timings, token/cost, visible failures |
| Evaluation | Label-driven delta, retrieval, answer, citation metrics |
| Single-process API | Optional FastAPI `/compare`, `/chat`, `/report`, `/health` |

The intentional scope cuts are DWG parsing, table/cell understanding, and
geometry recognition. They are documented rather than overstated.
