"""Small readable drawing tests; the dense supplied scan remains a separate stress case."""

import pymupdf

from src.config.settings import settings
from src.pipeline import DeltaPipeline, adapter_for


def test_scanned_and_native_revisions_use_the_same_delta_pipeline(tmp_path, monkeypatch):
    native_a = tmp_path / "native_a.pdf"
    native_b = tmp_path / "native_b.pdf"
    scanned_a = tmp_path / "scanned_a.pdf"
    for destination, pressure in [(native_a, 10), (native_b, 12)]:
        with pymupdf.open() as document:
            page = document.new_page(width=400, height=300)
            for y, text in [
                (50, "PSV-9066A"),
                (100, f"Pressure {pressure} bar"),
                (150, "NOTE 1: INSPECT VALVE"),
            ]:
                page.insert_text((40, y), text, fontsize=11)
            document.save(destination)
    with pymupdf.open(native_a) as native, pymupdf.open() as scan:
        page = scan.new_page(width=400, height=300)
        page.insert_image(page.rect, pixmap=native[0].get_pixmap(matrix=pymupdf.Matrix(3, 3)))
        scan.save(scanned_a)

    for field, filename in [
        ("canonical_a", "a.json"),
        ("canonical_b", "b.json"),
        ("delta_json", "custom_delta.json"),
        ("delta_markdown", "custom_delta.md"),
        ("delta_markup", "markup.pdf"),
    ]:
        monkeypatch.setattr(settings.paths, field, str(tmp_path / filename))
    # Model-backed indexing has a separate real-model smoke check. This test
    # verifies ingestion through report/markup and the canonical indexing seam.
    indexed = []

    def capture_index(old, new, deltas):
        indexed.append((old, new, deltas))
        return len(deltas)

    monkeypatch.setattr("src.pipeline.build_index", capture_index)
    result = DeltaPipeline(adapter_for("auto")).run(scanned_a, native_b)
    assert all(e.source == "ocr" for e in result.pid_a.pages[0].elements)
    changes = [entry for entry in result.deltas if entry.change_type.value != "unchanged"]
    assert len(changes) == 1
    assert changes[0].change_type.value == "modified"
    assert "10 bar" in changes[0].description and "12 bar" in changes[0].description
    assert result.report["revision_compatibility"]["compatible"]
    assert len(result.report["entries"]) == 1
    assert result.report["entries"][0]["delta_id"] == "delta-1"
    assert result.report["entries"][0]["location_revision"] == "B"
    assert indexed[0][0] == result.pid_a
    assert (tmp_path / "custom_delta.json").exists()
    assert (tmp_path / "custom_delta.md").exists()
    assert result.markup_path.exists()
