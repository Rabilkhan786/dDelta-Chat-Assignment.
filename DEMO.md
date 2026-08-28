# Demo walkthrough

1. **Install.**

   ```bash
   uv sync
   sudo apt-get install tesseract-ocr   # or: brew install tesseract
   ```

2. **Run the pipeline** on the default sample pair (a genuine revision pair —
   see `data/samples/synthetic_revision/PROVENANCE.md`).

   ```bash
   uv run python main.py run
   ```

   The JSON logs show one request ID spanning native-PDF ingestion for both
   revisions, alignment, delta classification, report generation, and
   retrieval-index building, each with a stage duration. On this pair, the
   delta engine finds exactly the 3 edits that were actually made, against
   494 unchanged elements:

   ```
   summary: {"total_entries": 497, "actual_changes": 3, "unchanged": 494,
             "modified": 1, "added": 1, "removed": 1}

   modified | text | page 1 | confidence 0.89 | 'PSV 9066A' -> 'PSV 9066C'
   removed  | text | page 1 | confidence 1.00 | 'MECHANICAL INTERLOCK'
   added    | text | page 1 | confidence 1.00 | 'NOTE 24: NEW BLOWDOWN VALVE ADDED PER REV B.'
   ```

   Full reports: `data/reports/delta_report.md` and `.json`.

3. **Try the scanned-PDF adapter** (Tesseract OCR, no text layer in the
   input):

   ```bash
   uv run python main.py run \
     --revision-a data/samples/scanned/lift_gas_scanned.pdf \
     --revision-b data/samples/synthetic_revision/revision_a.pdf \
     --adapter auto
   ```

   The router logs `pdf_adapter_selected` with 0 selectable characters for
   the scanned file, routes it to `ScannedPDFAdapter`, and OCR recovers text,
   per-word bounding boxes, and confidence into the same canonical schema
   native PDFs use.

4. **Grounded chat.**

   ```bash
   cp .env.example .env   # set GROQ_API_KEY
   uv run python main.py chat "What changed on the PSV valve?"
   ```

   Retrieval embeds PID A, PID B, and the delta report into Chroma
   (downloads a small embedding model the first time), retrieves the top-k
   matches for the question, and requires the LLM to cite one of them for
   every claim. If nothing relevant is retrieved, chat says so instead of
   guessing; if the LLM call itself fails, the retrieved citations are
   still returned. Inspect `logs/project.log` for the request ID, retrieval
   hit count, model name, token counts, and `estimated_cost_usd`.

5. **Tests and eval.**

   ```bash
   uv run pytest -q
   uv run python -m eval.run_eval
   ```

   `eval/datasets/ground_truth.json` is filled in with the 3 real change IDs
   from step 2 and one chat QA case. A real run prints:

   ```
   delta_scorecard: precision=1.0 recall=1.0 f1=1.0 predicted=3 expected=3
   ```

   followed by the chat scorecard (answer correctness + citation accuracy),
   which needs `GROQ_API_KEY` and network access — the harness logs a clear
   `chat_evaluation_failed` and still exits 0 rather than fabricating a
   score if either is unavailable.
