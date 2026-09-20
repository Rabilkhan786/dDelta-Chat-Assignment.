"""Tests for converting Tesseract words into canonical text lines."""

import pytest

from src.ingest.lines import ocr_lines


def test_ocr_words_are_grouped_sorted_and_scaled() -> None:
    data = {
        "text": ["VALVE", "OPEN", "", "ignored"],
        "conf": [90, 80, 99, -1],
        "block_num": [1, 1, 1, 2],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 1],
        "left": [20, 80, 0, 0],
        "top": [40, 40, 0, 0],
        "width": [40, 40, 0, 0],
        "height": [20, 20, 0, 0],
    }

    lines = ocr_lines(data, scale=2)

    assert len(lines) == 1
    assert lines[0]["text"] == "VALVE OPEN"
    assert lines[0]["confidence"] == pytest.approx(0.85)
    assert lines[0]["x0"] == 10.0
    assert lines[0]["y0"] == 20.0
    assert lines[0]["x1"] == 60.0
    assert lines[0]["y1"] == 30.0
