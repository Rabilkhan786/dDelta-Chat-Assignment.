# Delta Chat

Delta Chat compares two engineering-document revisions, writes a deterministic delta report, and supports cited retrieval-grounded chat.

## Run

```powershell
uv sync --locked
uv run python main.py run --adapter native
uv run python main.py chat "What changed near the compressor?"
uv run python -m eval.run_eval
```

`run` ingests the configured pair, writes canonical JSON to `data/output/`, writes Markdown and JSON reports to `data/reports/`, and builds a local BM25 retrieval index in `data/bm25/`. `chat` requires `OPENAI_API_KEY` in `.env`; it returns only retrieved, cited evidence.

## Architecture

`ingest/` adapters normalize each source into `canonical/CanonicalDocument`. `delta/align.py` deterministically matches same-page, same-type, nearby elements with RapidFuzz text similarity. `delta/engine.py` classifies matched and unmatched elements without an LLM. `delta/report.py` emits JSON and Markdown. `chat/` indexes PID A, PID B, and report entries as separate source-labelled excerpts, retrieves with BM25, then calls the configured provider only for grounded answer generation. `observability/` emits JSON logs with request IDs and stage durations.

Native PDF is demonstrated by the supplied documents. `ScannedPDFAdapter` is complete behind the same interface but requires the optional `paddleocr` package; DWG remains an explicit stub. This is a deliberate scope cut because no scanned PDF or DWG sample was supplied. The Assignment Specification and Project Workflow Guide are external session references and are intentionally not copied into this repository.

## Sample-data provenance and limitations

`data/input/Export Gas Compressor-P&ID (1).pdf` and `data/input/Lift Gas compressor-P&ID.pdf` are the supplied sample documents. They depict different compressor systems rather than confirmed revisions, so the generated report is an objective structural comparison, not claimed revision ground truth. The provided sources do not specify human-labelled expected deltas or Q&A; `eval/datasets/ground_truth.template.json` is the schema a reviewer should fill before using precision/recall/F1 as a quality claim.

## Observability and evaluation

Each pipeline stage emits structured JSON with a shared request ID, duration, failures, delta counts, and retrieval-hit count. The LLM provider logs model and token counts; pricing is intentionally `null` because no provider-pricing source is configured. The evaluation harness computes exact-ID delta precision, recall, and F1 and records known failures rather than fabricating labels.

## Production considerations

For larger drawing sets, batch canonical ingestion and index per document version; constrain OCR concurrency; persist report/version metadata; add human review for low-confidence changes; and export trace metrics to an observability backend. Keep `.env` private and rotate provider keys. The current lexical retriever is predictable and inspectable, but a production system may add a separately evaluated semantic retriever.

## Interview discussion

Why deterministic delta? It makes structural changes reproducible, debuggable, and measurable. Why a canonical model? It decouples downstream comparison/retrieval from source formats. The principal limitation is matching moved/reflowed content and unlabelled sample data; the next step is a reviewer-labelled revision dataset and false-positive analysis.
