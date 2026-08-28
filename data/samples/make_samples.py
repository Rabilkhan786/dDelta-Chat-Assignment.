"""Regenerates the synthetic sample pairs from the one supplied native PDF.

Run with: uv run python data/samples/make_samples.py

What this produces (see PROVENANCE.md in each folder for details):
  synthetic_revision/revision_b.pdf  - revision_a.pdf with 3 concrete edits applied
  scanned/lift_gas_scanned.pdf       - revision_a.pdf rasterized with no text layer,
                                        used to exercise the Tesseract OCR adapter
"""

from pathlib import Path

import fitz

SAMPLES = Path(__file__).parent
SOURCE = SAMPLES / "synthetic_revision" / "revision_a.pdf"


def make_revision_b() -> None:
    """Apply one modify, one removal, and one addition to page 1 of revision_a.pdf."""
    doc = fitz.open(SOURCE)
    page = doc[0]

    # Modify: rename a pressure safety valve tag.
    modify_rect = fitz.Rect(853, 75, 868, 86)
    page.add_redact_annot(modify_rect)

    # Remove: drop the "MECHANICAL INTERLOCK" callout entirely.
    remove_rect = fitz.Rect(931, 171, 962, 182)
    page.add_redact_annot(remove_rect)

    page.apply_redactions()
    page.insert_text((853, 82), "PSV\n9066C", fontsize=6)

    # Add: a new note in a blank corner of the sheet.
    page.insert_text((100, 30), "NOTE 24: NEW BLOWDOWN VALVE ADDED PER REV B.", fontsize=7)

    out_path = SAMPLES / "synthetic_revision" / "revision_b.pdf"
    doc.save(out_path)
    doc.close()
    print(f"wrote {out_path}")


def make_scanned_sample() -> None:
    """Rasterize page 1 of revision_a.pdf into an image-only PDF (no text layer)."""
    doc = fitz.open(SOURCE)
    original_page = doc[0]
    original_rect = original_page.rect
    pixmap = original_page.get_pixmap(matrix=fitz.Matrix(2, 2))
    doc.close()

    # Keep the new page the same physical size (in points) as the original so
    # recovered OCR bounding boxes stay comparable to the native PDF's coordinates.
    scanned = fitz.open()
    page = scanned.new_page(width=original_rect.width, height=original_rect.height)
    page.insert_image(page.rect, pixmap=pixmap)

    out_dir = SAMPLES / "scanned"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lift_gas_scanned.pdf"
    scanned.save(out_path)
    scanned.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    make_revision_b()
    make_scanned_sample()
