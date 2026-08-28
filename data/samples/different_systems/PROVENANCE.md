# Sample 2: two originally supplied P&IDs (edge case, not a revision pair)

- `export_gas_compressor.pdf` — supplied `Export Gas Compressor-P&ID (1).pdf`.
- `lift_gas_compressor.pdf` — supplied `Lift Gas compressor-P&ID.pdf`.

These are two **different** compressor systems, not two revisions of the same
document. They are kept as a stress-test pair: running the pipeline on them
produces a large, mostly-unhelpful delta, which is the expected and honest
result for documents that were never the same drawing. Useful for confirming
the pipeline doesn't crash or fabricate a false "these are similar" answer on
unrelated inputs, not for judging delta quality.
