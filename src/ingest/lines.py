"""Convert words into one stable canonical line per OCR line."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def ocr_lines(data: dict[str, list[Any]], scale: float) -> list[dict[str, float | str]]:
    """Group Tesseract TSV words by its page/block/paragraph/line fields."""
    groups: dict[tuple[int, int, int], list[tuple[str, float, float, float, float, float]]] = (
        defaultdict(list)
    )
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        confidence = float(data["conf"][index])
        if not text or confidence < 0:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        left, top, width, height = (
            float(data[name][index]) / scale for name in ("left", "top", "width", "height")
        )
        groups[key].append((text, confidence / 100, left, top, width, height))

    lines: list[dict[str, float | str]] = []
    for words in groups.values():
        words.sort(key=lambda word: word[2])
        left = min(word[2] for word in words)
        top = min(word[3] for word in words)
        right = max(word[2] + word[4] for word in words)
        bottom = max(word[3] + word[5] for word in words)
        lines.append(
            {
                "text": " ".join(word[0] for word in words),
                "confidence": sum(word[1] for word in words) / len(words),
                "x0": left,
                "y0": top,
                "x1": right,
                "y1": bottom,
            }
        )
    return sorted(lines, key=lambda line: (float(line["y0"]), float(line["x0"])))
