# Delta Chat

Delta Chat compares two engineering-document revisions, writes a deterministic delta report, and supports cited retrieval-grounded chat.

## Run

```powershell
uv sync --locked
uv run python main.py run
uv run python main.py chat "What changed near the compressor?"
uv run python -m eval.run_eval
```

`run` automatically inspects each PDF for meaningful selectable text. PDFs with at least the configured `ingest.native_text_threshold` characters use `NativePDFAdapter`; otherwise `AutomaticPDFAdapter` routes to `ScannedPDFAdapter`. The command writes canonical JSON to `data/output/`, Markdown and JSON reports to `data/reports/`, and a local BM25 retrieval index in `data/bm25/`. `chat` requires `OPENAI_API_KEY` in `.env`; it returns only retrieved, cited evidence.

## Architecture

`ingest/` adapters normalize each source into `canonical/CanonicalDocument`. `delta/align.py` deterministically matches same-page, same-type, nearby elements with RapidFuzz text similarity. `delta/engine.py` classifies matched and unmatched elements without an LLM. `delta/report.py` emits JSON and Markdown. `chat/` indexes PID A, PID B, and report entries as separate source-labelled excerpts, retrieves with BM25, then calls the configured provider only for grounded answer generation. `observability/` emits JSON logs with request IDs and stage durations.

Native PDF is demonstrated by the supplied documents. Scanned PDFs are automatically detected when selectable text is below the configured threshold and are routed to `ScannedPDFAdapter` (also exposed as `OCRPDFAdapter`). OCR preserves page numbers and approximate bounding boxes in the same `CanonicalDocument` schema as native parsing. Install the optional dependency with `uv add paddleocr paddlepaddle` and then `uv sync`. If it is unavailable, ingestion fails with that actionable command and never fabricates OCR output. DWG remains an explicit stub. No scanned PDF or DWG sample was supplied, so real OCR end-to-end coverage is a documented skipped test until a scanned sample is available. The Assignment Specification and Project Workflow Guide are external session references and are intentionally not copied into this repository.

## Sample-data provenance and limitations

`data/input/Export Gas Compressor-P&ID (1).pdf` and `data/input/Lift Gas compressor-P&ID.pdf` are the supplied sample documents. They depict different compressor systems rather than confirmed revisions, so the generated report is an objective structural comparison, not claimed revision ground truth. The provided sources do not specify human-labelled expected deltas or Q&A; `eval/datasets/ground_truth.template.json` is the schema a reviewer should fill before using precision/recall/F1 as a quality claim.

## Observability and evaluation

Each pipeline stage emits structured JSON with a shared request ID, duration, failures, delta counts, and retrieval-hit count. The LLM provider logs model, token counts, and an estimated cost calculated from the input/output token prices in `src/config/config.yaml`. For the configured `gpt-4.1-mini`, the default $0.40 input and $1.60 output prices per million tokens follow the [official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini). Update configuration if the provider/model or pricing changes.

The evaluation harness derives predicted IDs from `data/reports/delta_report.json` and compares them only with human-labelled `expected_change_ids`. Generate candidate IDs with `uv run python -m eval.run_eval --write-candidates`, review them, copy approved labels into `eval/datasets/ground_truth.json`, then run `uv run python -m eval.run_eval`. It does not fabricate precision, recall, F1, chat correctness, or citation-accuracy results.

## Configuration, environment, and validation

`src/config/config.yaml` centralizes document paths, alignment thresholds, automatic-detection threshold, OCR DPI, retrieval settings, model selection, and token pricing. Create a local `.env` and set `OPENAI_API_KEY` for live grounded-chat validation; Langfuse variables remain optional. The validation sequence is `uv run pytest -q`, `uv run python main.py run`, and then `uv run python main.py chat "What changed near the compressor?"`. Confirm the resulting JSON log includes the answer, citations, request ID, stage duration, token counts, and `estimated_cost_usd`.

If the API key is absent, invalid, rate-limited, or out of quota, chat logs the provider failure and returns a clear validation message with the retrieved evidence citations; it does not fabricate an answer, token counts, or cost. Restore provider availability and rerun the command to validate live answer, token, and cost telemetry.

## Assignment mapping

| Assignment section | Status | Implementation |
| --- | --- | --- |
| What you'll build A: format-agnostic ingestion | Complete for Native; OCR adapter/routing implemented | `src/ingest/` adapters return `CanonicalDocument`; automatic selection is logged. |
| What you'll build B-C: deterministic delta and reports | Complete | `src/delta/` and generated Markdown/JSON reports. |
| What you'll build D: grounded chat with citations | Implemented; live provider validation depends on API access | `src/chat/` retrieves PID A, PID B, and report excerpts before one provider call. |
| Observability | Implemented; external provider telemetry awaits live validation | JSON request IDs, stage timing, token and estimated-cost fields. |
| Evaluation harness | Implemented; quality score awaits human labels | `eval/` derives predictions and requires reviewer labels. |
| Technical requirements: 2-3 document pairs and two formats demonstrated | Awaiting external samples | Only one native pair is supplied; no scanned PDF or DWG sample is available. |

## Production considerations

For larger drawing sets, batch canonical ingestion and index per document version; constrain OCR concurrency; persist report/version metadata; add human review for low-confidence changes; and export trace metrics to an observability backend. Keep `.env` private and rotate provider keys. The current lexical retriever is predictable and inspectable, but a production system may add a separately evaluated semantic retriever.

## Interview discussion

Why deterministic delta? It makes structural changes reproducible, debuggable, and measurable. Why a canonical model? It decouples downstream comparison/retrieval from source formats. The principal limitation is matching moved/reflowed content and unlabelled sample data; the next step is a reviewer-labelled revision dataset and false-positive analysis.
