"""Match elements between two document revisions before they are classified.

Matching strategy, in order:
1. Same page
2. Same element type
3. Nearby bounding boxes
4. Highest text similarity (RapidFuzz)
"""

from __future__ import annotations

from math import sqrt

from rapidfuzz import fuzz

from src.canonical.model import Alignment, AlignmentResult, BoundingBox, CanonicalDocument, Element
from src.config.settings import settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class Aligner:
    """Deterministically pair up elements from an old and a new revision."""

    def __init__(self, similarity_threshold: float | None = None, max_bbox_distance: float | None = None) -> None:
        self.similarity_threshold = similarity_threshold or settings.align.similarity_threshold
        self.max_bbox_distance = max_bbox_distance or settings.align.max_bbox_distance

    def align(self, old_document: CanonicalDocument, new_document: CanonicalDocument) -> AlignmentResult:
        with stage(logger, "document_alignment"):
            return self._align(old_document, new_document)

    def _align(self, old_document: CanonicalDocument, new_document: CanonicalDocument) -> AlignmentResult:
        result = AlignmentResult()
        old_elements = self._flatten(old_document)
        new_elements = self._flatten(new_document)
        matched_new: set[int] = set()

        for old in old_elements:
            best_index, best_similarity, best_distance = None, 0.0, float("inf")

            for index, new in enumerate(new_elements):
                if index in matched_new or old.page_number != new.page_number or old.type != new.type:
                    continue

                distance = self._bbox_distance(old.bbox, new.bbox)
                if distance > self.max_bbox_distance:
                    continue

                similarity = fuzz.WRatio(old.text.strip(), new.text.strip())
                if similarity > best_similarity or (similarity == best_similarity and distance < best_distance):
                    best_index, best_similarity, best_distance = index, similarity, distance

            if best_index is not None and best_similarity >= self.similarity_threshold:
                matched = new_elements[best_index]
                result.matches.append(Alignment(left=old, right=matched, similarity=best_similarity, bbox_distance=best_distance))
                matched_new.add(best_index)
            else:
                result.unmatched_left.append(old)

        result.unmatched_right = [element for index, element in enumerate(new_elements) if index not in matched_new]

        logger.info("alignment_completed", extra={"matches": len(result.matches),
            "unmatched_left": len(result.unmatched_left), "unmatched_right": len(result.unmatched_right)})
        return result

    @staticmethod
    def _flatten(document: CanonicalDocument) -> list[Element]:
        return [element for page in document.pages for element in page.elements]

    @staticmethod
    def _bbox_distance(left: BoundingBox | None, right: BoundingBox | None) -> float:
        """Euclidean distance between bounding-box centers."""
        if left is None or right is None:
            return float("inf")
        left_x, left_y = (left.x0 + left.x1) / 2, (left.y0 + left.y1) / 2
        right_x, right_y = (right.x0 + right.x1) / 2, (right.y0 + right.y1) / 2
        return sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2)
