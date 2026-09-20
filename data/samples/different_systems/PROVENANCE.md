# Sample 2: two originally supplied P&IDs (edge case, not a revision pair)

- `export_gas_compressor.pdf` — supplied `Export Gas Compressor-P&ID (1).pdf`.
- `lift_gas_compressor.pdf` — supplied `Lift Gas compressor-P&ID.pdf`.

These are two **different** compressor systems, not two revisions of the same
document. They are kept as a stress-test pair: running the pipeline on them
produces a large, mostly-unhelpful delta, which is the expected and honest
result for documents that were never the same drawing. Their token-overlap score
is 0.465. The old 0.12 threshold incorrectly accepted them; the configured 0.60
threshold now warns. This is one regression case, not a general identity test.
Do not use these documents to judge revision-delta accuracy.
