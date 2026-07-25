# Demo walkthrough

This walkthrough demonstrates the required pipeline with the supplied P&ID PDFs. It is designed for a short screen recording or terminal walkthrough.

1. Install the locked environment.

   ```powershell
   uv sync --locked
   ```

2. Run the automatic PDF-detection pipeline.

   ```powershell
   uv run python main.py run
   ```

   The router counts selectable text. It selects `NativePDFAdapter` for meaningful text and `ScannedPDFAdapter` for an image-only PDF, logging the selected adapter. Both paths return the same `CanonicalDocument` schema. The structured JSON logs show one request ID spanning detection, ingestion, document alignment, deterministic delta comparison, delta-report generation, and retrieval-index building. The report artifacts are `data/reports/delta_report.md` and `data/reports/delta_report.json`.

   To use OCR after supplying a scanned PDF, install `paddleocr` and `paddlepaddle` with `uv add paddleocr paddlepaddle`, run `uv sync`, and pass the scanned PDF with `--revision-a` or `--revision-b`. If PaddleOCR is unavailable, the OCR adapter returns an actionable installation error and does not generate output.

3. Inspect the report. For the supplied drawings, the result is an objective comparison, not a revision-quality claim: 226 additions, 159 modifications, and 101 removals were detected. The documents represent different compressor systems, so the large delta is expected.

4. Configure the LLM only for grounded chat.

   ```powershell
   # Create .env and set OPENAI_API_KEY
   uv run python main.py chat "What changed near the compressor?"
   ```

   The chat service retrieves source-labelled excerpts from PID A, PID B, and the delta report before invoking the provider. The prompt requires a citation for every factual statement. Inspect the JSON logs for the request ID, `grounded_chat` and `llm_completion` durations, input/output tokens, and `estimated_cost_usd`.

5. Run the regression tests and evaluation harness.

   ```powershell
   uv run pytest -q
   uv run python -m eval.run_eval
   ```

   First run `uv run python -m eval.run_eval --write-candidates` to generate deterministic candidate IDs from the report. A human reviewer places approved expected IDs in `eval/datasets/ground_truth.json` using the provided template, then reruns the scorecard. The harness deliberately reports that labels are required until then; it does not fabricate delta or citation quality scores.
