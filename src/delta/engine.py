"""Turn an AlignmentResult into structured, confidence-scored DeltaEntry objects."""

import re

from src.canonical.model import Alignment, AlignmentResult, BoundingBox, DeltaEntry, DeltaType, Element
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class DeltaEngine:
    """Rule-based delta classification. No LLM involved, so results are reproducible."""

    def compare(self, alignment_result: AlignmentResult) -> list[DeltaEntry]:
        with stage(logger, "delta_comparison"):
            return self._compare(alignment_result)

    def _compare(self, alignment_result: AlignmentResult) -> list[DeltaEntry]:
        deltas: list[DeltaEntry] = []

        for match in alignment_result.matches:
            if match.matched_after_move and self._normalize(match.left.text) == self._normalize(match.right.text):
                deltas.append(self._moved(match))
            elif self._normalize(match.left.text) != self._normalize(match.right.text):
                deltas.append(self._modified(match))
            else:
                deltas.append(self._unchanged(match))

        deltas += [self._removed(element) for element in alignment_result.unmatched_left]
        deltas += [self._added(element) for element in alignment_result.unmatched_right]
        deltas = self._join_adjacent_fragments(deltas)

        counts = {change_type.value: sum(1 for delta in deltas if delta.change_type == change_type) for change_type in DeltaType}
        logger.info("delta_comparison_completed", extra={**counts, "total": len(deltas)})
        return deltas

    @staticmethod
    def _join_adjacent_fragments(deltas: list[DeltaEntry]) -> list[DeltaEntry]:
        """Join touching same-column text fragments into one meaningful callout.

        PDF writers sometimes emit a two-word label on separate visual lines.
        This preserves line-level extraction while keeping the report useful.
        """
        joined: list[DeltaEntry] = []
        for delta in deltas:
            previous = joined[-1] if joined else None
            if previous and DeltaEngine._can_join(previous, delta):
                joined[-1] = previous.model_copy(update={
                    "description": DeltaEngine._join_description(previous.description, delta.description),
                    "region": BoundingBox(x0=min(previous.region.x0, delta.region.x0),
                        y0=min(previous.region.y0, delta.region.y0), x1=max(previous.region.x1, delta.region.x1),
                        y1=max(previous.region.y1, delta.region.y1)),
                    "confidence": min(previous.confidence, delta.confidence),
                })
            else:
                joined.append(delta)
        return joined

    @staticmethod
    def _can_join(left: DeltaEntry, right: DeltaEntry) -> bool:
        if left.change_type not in {DeltaType.ADDED, DeltaType.REMOVED} or left.change_type != right.change_type:
            return False
        if left.element_type != right.element_type or left.page_number != right.page_number:
            return False
        if not left.region or not right.region:
            return False
        same_column = abs(left.region.x0 - right.region.x0) <= 3
        touching_lines = right.region.y0 - left.region.y1 <= 3
        return same_column and touching_lines

    @staticmethod
    def _join_description(left: str, right: str) -> str:
        """Combine the quoted text parts of two matching added/removed entries."""
        prefix = left.split("'", maxsplit=1)[0]
        left_text = left.rsplit("'", maxsplit=1)[0].split("'", maxsplit=1)[-1]
        right_text = right.rsplit("'", maxsplit=1)[0].split("'", maxsplit=1)[-1]
        return f"{prefix}'{left_text} {right_text}'"

    def _unchanged(self, match: Alignment) -> DeltaEntry:
        return DeltaEntry(change_type=DeltaType.UNCHANGED, element_type=match.right.type, page_number=match.right.page_number,
            region=match.right.bbox, description=f"Unchanged {match.right.type.value}: '{match.right.text}'",
            confidence=self._confidence(match.similarity, match.right), element_id=match.right.id)

    def _modified(self, match: Alignment) -> DeltaEntry:
        return DeltaEntry(change_type=DeltaType.MODIFIED, element_type=match.right.type, page_number=match.right.page_number,
            region=match.right.bbox, description=f"{match.right.type.value} changed from '{match.left.text}' to '{match.right.text}'",
            confidence=self._confidence(match.similarity, match.right), element_id=match.right.id,
            previous_element_id=match.left.id, location_changed=match.matched_after_move)

    def _moved(self, match: Alignment) -> DeltaEntry:
        """Report an unchanged label that moved beyond the first-pass distance limit."""
        return DeltaEntry(change_type=DeltaType.MOVED, element_type=match.right.type,
            page_number=match.right.page_number, region=match.right.bbox,
            description=f"Moved {match.right.type.value}: '{match.right.text}'",
            confidence=self._confidence(match.similarity, match.right), element_id=match.right.id,
            previous_element_id=match.left.id, location_changed=True)

    def _removed(self, element: Element) -> DeltaEntry:
        return DeltaEntry(change_type=DeltaType.REMOVED, element_type=element.type, page_number=element.page_number,
            region=element.bbox, description=f"Removed {element.type.value}: '{element.text}'",
            confidence=self._confidence(100.0, element), element_id=element.id)

    def _added(self, element: Element) -> DeltaEntry:
        return DeltaEntry(change_type=DeltaType.ADDED, element_type=element.type, page_number=element.page_number,
            region=element.bbox, description=f"Added {element.type.value}: '{element.text}'",
            confidence=self._confidence(100.0, element), element_id=element.id)

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse whitespace so re-wrapped or re-flowed text isn't reported as modified."""
        return re.sub(r"\s+", " ", text.strip())

    @staticmethod
    def _confidence(similarity: float, element: Element) -> float:
        """Native text: confidence is just similarity. OCR text: discount it by OCR confidence."""
        normalized_similarity = max(0.0, min(similarity / 100.0, 1.0))
        if element.source == "native" or element.ocr_confidence is None:
            confidence = normalized_similarity
        else:
            confidence = normalized_similarity * element.ocr_confidence
        return round(min(confidence, 1.0), 2)
