"""Match elements between two document revisions before they are classified.

Matching strategy, in order:
1. Same page, type, nearby position, and best text similarity.
2. For unmatched items, same page and very high text similarity even if moved.
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

    def __init__(
        self,
        similarity_threshold: float | None = None,
        max_bbox_distance: float | None = None,
        moved_similarity_threshold: float | None = None,
    ) -> None:
        self.similarity_threshold = (
            settings.align.similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )
        self.max_bbox_distance = (
            settings.align.max_bbox_distance if max_bbox_distance is None else max_bbox_distance
        )
        self.moved_similarity_threshold = (
            settings.align.moved_similarity_threshold
            if moved_similarity_threshold is None
            else moved_similarity_threshold
        )

    def align(
        self, old_document: CanonicalDocument, new_document: CanonicalDocument
    ) -> AlignmentResult:
        with stage(logger, "document_alignment"):
            return self._align(old_document, new_document)

    def _align(
        self, old_document: CanonicalDocument, new_document: CanonicalDocument
    ) -> AlignmentResult:
        result = AlignmentResult()
        old_elements = self._flatten(old_document)
        new_elements = self._flatten(new_document)
        matched_new: set[int] = set()
        unmatched_left: list[Element] = []
        for old in old_elements:
            candidate = self._best_candidate(old, new_elements, matched_new, nearby_only=True)
            if candidate and candidate[1] >= self.similarity_threshold:
                index, similarity, distance = candidate
                result.matches.append(
                    Alignment(
                        left=old,
                        right=new_elements[index],
                        similarity=similarity,
                        bbox_distance=distance,
                    )
                )
                matched_new.add(index)
            else:
                unmatched_left.append(old)

        still_unmatched: list[Element] = []
        for old in unmatched_left:
            candidate = self._best_candidate(old, new_elements, matched_new, nearby_only=False)
            if candidate and candidate[1] >= self.moved_similarity_threshold:
                index, similarity, distance = candidate
                result.matches.append(
                    Alignment(
                        left=old,
                        right=new_elements[index],
                        similarity=similarity,
                        bbox_distance=distance,
                        matched_after_move=True,
                    )
                )
                matched_new.add(index)
            else:
                still_unmatched.append(old)

        result.unmatched_left = still_unmatched
        result.unmatched_right = [
            item for index, item in enumerate(new_elements) if index not in matched_new
        ]

        logger.info(
            "alignment_completed",
            extra={
                "matches": len(result.matches),
                "unmatched_left": len(result.unmatched_left),
                "unmatched_right": len(result.unmatched_right),
            },
        )
        return result

    def _best_candidate(
        self, old: Element, candidates: list[Element], used: set[int], nearby_only: bool
    ) -> tuple[int, float, float] | None:
        """Find one unused same-page, same-type candidate using explicit rules."""
        best: tuple[int, float, float] | None = None
        for index, new in enumerate(candidates):
            if index in used or old.page_number != new.page_number or old.type != new.type:
                continue
            distance = self._bbox_distance(old.bbox, new.bbox)
            if nearby_only and distance > self.max_bbox_distance:
                continue
            # Plain ratio avoids WRatio's subset bias (for example, matching
            # the tag "9066A" to an unrelated one-character label "9").
            similarity = fuzz.ratio(old.text.strip(), new.text.strip())
            if (
                best is None
                or similarity > best[1]
                or (similarity == best[1] and distance < best[2])
            ):
                best = (index, similarity, distance)
        return best

    @staticmethod
    def _flatten(document: CanonicalDocument) -> list[Element]:
        return [element for page in document.pages for element in page.elements]

    @staticmethod
    def _bbox_distance(left: BoundingBox | None, right: BoundingBox | None) -> float:
        """Euclidean distance between bounding-box centers."""
        if left is None or right is None:
            # Missing coordinates mean "unknown", not "moved". Text and type
            # can still align through the normal first pass.
            return 0.0
        left_x, left_y = (left.x0 + left.x1) / 2, (left.y0 + left.y1) / 2
        right_x, right_y = (right.x0 + right.x1) / 2, (right.y0 + right.y1) / 2
        return sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2)
