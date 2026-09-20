# Sample pairs and provenance

| Pair | Inputs | Purpose |
| --- | --- | --- |
| Synthetic revision | `synthetic_revision/revision_a.pdf`, `revision_b.pdf` | Same drawing with one tag edit, one removed callout, one added note |
| Native/OCR same content | `scanned/lift_gas_scanned.pdf`, `synthetic_revision/revision_a.pdf` | OCR routing and extraction-error stress test |
| Different systems | `different_systems/export_gas_compressor.pdf`, `lift_gas_compressor.pdf` | Negative compatibility case, not a revision pair |

Each folder contains a provenance note. Only the first pair currently has
delta/QA labels. The scanned PDF is synthesized by rasterizing the supplied
drawing; it is not a real physical scan. `make_samples.py` regenerates the
synthetic edits and scan and overwrites those sample files.

These are three test pairs, not three independently human-reviewed benchmarks.
