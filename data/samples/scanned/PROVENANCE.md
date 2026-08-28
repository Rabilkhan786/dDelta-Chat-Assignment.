# Sample 3: scanned PDF (exercises the Tesseract OCR adapter)

- `lift_gas_scanned.pdf` — page 1 of `revision_a.pdf`, rendered to a raster image at 2x
  and re-embedded into a fresh PDF with **no text layer**, by
  `data/samples/make_samples.py`. `NativePDFAdapter` finds 0 selectable characters on
  it, so `AutomaticPDFAdapter` routes it to `ScannedPDFAdapter`, which runs Tesseract
  and recovers text with bounding boxes and per-word confidence.

No physical scanner was available, so this simulates one the way the assignment FAQ
suggests: "export a PDF twice... or photograph/scan a printout." Rasterizing removes
the text layer the same way printing and re-scanning would.

Compare it against `synthetic_revision/revision_a.pdf` to see native vs. OCR extraction
of the same page:

```
uv run python main.py run --revision-a data/samples/scanned/lift_gas_scanned.pdf \
  --revision-b data/samples/synthetic_revision/revision_a.pdf --adapter auto
```
