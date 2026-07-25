# Demo walkthrough

This walkthrough demonstrates the required pipeline with the supplied P&ID PDFs. It is designed for a short screen recording or terminal walkthrough.

1. Install the locked environment.

   ```powershell
   uv sync --locked
   ```

2. Run the native-PDF pipeline.

   ```powershell
   uv run python main.py run --adapter native
   ```

   The structured JSON logs show one request ID spanning native ingestion, document alignment, deterministic delta comparison, delta-report generation, and retrieval-index building. The report artifacts are `data/reports/delta_report.md` and `data/reports/delta_report.json`.

3. Inspect the report. For the supplied drawings, the result is an objective comparison, not a revision-quality claim: 226 additions, 159 modifications, and 101 removals were detected. The documents represent different compressor systems, so the large delta is expected.

4. Configure the LLM only for grounded chat.

   ```powershell
   Copy-Item .env.example .env
   # Set OPENAI_API_KEY in .env
   uv run python main.py chat "What changed near the compressor?"
   ```

   The chat service retrieves source-labelled excerpts from PID A, PID B, and the delta report before invoking the provider. The prompt requires a citation for every factual statement.

5. Run the regression tests and evaluation harness.

   ```powershell
   uv run pytest -q
   uv run python -m eval.run_eval
   ```

   The evaluation harness deliberately reports that labels are required until a human reviewer creates `eval/datasets/ground_truth.json` from the provided template. It does not fabricate delta or citation quality scores.
