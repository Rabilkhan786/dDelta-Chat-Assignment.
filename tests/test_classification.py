"""Document the technical types the deterministic rules actually support."""

from pathlib import Path

import pytest

from src.ingest.base import FormatAdapter
from src.ingest.classify import classify_text
from src.ingest.dwg import DWGStubAdapter


@pytest.mark.parametrize(
    ("text", "font_size", "expected"),
    [
        ("10 bar", None, "dimension"),
        ("NOTE 24: NEW VALVE", None, "note"),
        ("PSV-9066A", None, "symbol"),
        ("COMPRESSOR PLAN", 18, "title"),
        ("unstructured table-looking text", None, "text"),
    ],
)
def test_supported_text_classification(text, font_size, expected):
    assert classify_text(text, font_size).value == expected


def test_dwg_seam_is_real_but_explicitly_unimplemented():
    adapter = DWGStubAdapter()
    assert isinstance(adapter, FormatAdapter)
    assert adapter.supports(Path("drawing.DWG"))
    with pytest.raises(NotImplementedError, match="not implemented"):
        adapter.parse(Path("drawing.DWG"))
