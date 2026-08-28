"""Turn an AlignmentResult into structured, confidence-scored DeltaEntry objects."""

from src.canonical.model import Alignment, AlignmentResult, DeltaEntry, DeltaType, Element
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
            if match.left.text.strip() != match.right.text.strip():
                deltas.append(self._modified(match))
            else:
                deltas.append(self._unchanged(match))

        deltas += [self._removed(element) for element in alignment_result.unmatched_left]
        deltas += [self._added(element) for element in alignment_result.unmatched_right]

        counts = {change_type.value: sum(1 for delta in deltas if delta.change_type == change_type) for change_type in DeltaType}
        logger.info("delta_comparison_completed", extra={**counts, "total": len(deltas)})
        return deltas

    def _unchanged(self, match: Alignment) -> DeltaEntry:
        return DeltaEntry(change_type=DeltaType.UNCHANGED, element_type=match.right.type, page_number=match.right.page_number,
            region=match.right.bbox, description=f"Unchanged {match.right.type.value}: '{match.right.text}'",
            confidence=self._confidence(match.similarity, match.right))

    def _modified(self, match: Alignment) -> DeltaEntry:
        return DeltaEntry(change_type=DeltaType.MODIFIED, element_type=match.right.type, page_number=match.right.page_number,
            region=match.right.bbox, description=f"{match.right.type.value} changed from '{match.left.text}' to '{match.right.text}'",
            confidence=self._confidence(match.similarity, match.right))

    def _removed(self, element: Element) -> DeltaEntry:
        return DeltaEntry(change_type=DeltaType.REMOVED, element_type=element.type, page_number=element.page_number,
            region=element.bbox, description=f"Removed {element.type.value}: '{element.text}'",
            confidence=self._confidence(100.0, element))

    def _added(self, element: Element) -> DeltaEntry:
        return DeltaEntry(change_type=DeltaType.ADDED, element_type=element.type, page_number=element.page_number,
            region=element.bbox, description=f"Added {element.type.value}: '{element.text}'",
            confidence=self._confidence(100.0, element))

    @staticmethod
    def _confidence(similarity: float, element: Element) -> float:
        """Native text: confidence is just similarity. OCR text: discount it by OCR confidence."""
        normalized_similarity = max(0.0, min(similarity / 100.0, 1.0))
        if element.source == "native" or element.ocr_confidence is None:
            confidence = normalized_similarity
        else:
            confidence = normalized_similarity * element.ocr_confidence
        return round(min(confidence, 1.0), 2)
