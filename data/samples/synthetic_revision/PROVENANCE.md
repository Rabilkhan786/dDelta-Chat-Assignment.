# Sample 1: synthetic revision pair (primary demo pair)

- `revision_a.pdf` — the supplied `Lift Gas compressor-P&ID.pdf`, unmodified.
- `revision_b.pdf` — the same file with 3 edits applied by `data/samples/make_samples.py`
  (PyMuPDF redaction + text insertion), regenerate with `uv run python data/samples/make_samples.py`:
  1. **Modified** — pressure safety valve tag `PSV 9066A` renamed to `PSV 9066C`.
  2. **Removed** — the `MECHANICAL INTERLOCK` callout near the compressor.
  3. **Added** — a new note, `NOTE 24: NEW BLOWDOWN VALVE ADDED PER REV B.`, in a blank
     corner of the sheet.

This is a genuine revision pair (same drawing, same layout, a handful of real changes),
so it is the default pair in `src/config/config.yaml` and the one the delta engine and
eval harness are exercised against.
