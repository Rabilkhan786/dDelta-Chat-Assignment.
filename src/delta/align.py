from __future__ import annotations

from math import sqrt

from rapidfuzz import fuzz
from src.config.settings import settings
from src.observability.logging import get_logger, stage

from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    Element,
    Alignment,
    AlignmentResult,
)

logger = get_logger(__name__)


class Aligner:
    """
    Align elements between two document revisions.

    Matching Strategy

    1. Same page
    2. Same element type
    3. Nearby bounding boxes
    4. Highest text similarity
    """

    def __init__(
        self,
        similarity_threshold = settings.align.similarity_threshold,
        max_bbox_distance = settings.align.max_bbox_distance,
    ) -> None:

        self.similarity_threshold = similarity_threshold
        self.max_bbox_distance = max_bbox_distance

    def align(
        self,
        old_document: CanonicalDocument,
        new_document: CanonicalDocument,
    ) -> AlignmentResult:

        with stage(logger, "document_alignment"):
            return self._align(old_document, new_document)

    def _align(
        self,
        old_document: CanonicalDocument,
        new_document: CanonicalDocument,
    ) -> AlignmentResult:
        """Match elements deterministically by page, type, location and text."""

        result = AlignmentResult()

        old_elements = self._flatten_document(old_document)
        new_elements = self._flatten_document(new_document)

        logger.info(
            "Loaded %d old elements and %d new elements",
            len(old_elements),
            len(new_elements),
        )

        matched_new: set[int] = set()

        for old in old_elements:

            best_index = None
            best_similarity = 0.0
            best_distance = float("inf")

            for index, new in enumerate(new_elements):

                if index in matched_new:
                    continue

                # Match only same page
                if old.page_number != new.page_number:
                    continue

                # Match only same element type
                if old.type != new.type:
                    continue

                # Compare bounding boxes
                distance = self._bbox_distance(
                    old.bbox,
                    new.bbox,
                )

                if distance > self.max_bbox_distance:
                    continue

                # Compare text
                similarity = fuzz.WRatio(
                    old.text.strip(),
                    new.text.strip(),
                )

                if (
                    similarity > best_similarity
                    or (
                        similarity == best_similarity
                        and distance < best_distance
                    )
                ):
                    best_similarity = similarity
                    best_distance = distance
                    best_index = index

            if (
                best_index is not None
                and best_similarity >= self.similarity_threshold
            ):

                matched_element = new_elements[best_index]

                result.matches.append(
                    Alignment(
                        left=old,
                        right=matched_element,
                        similarity=best_similarity,
                        bbox_distance=best_distance,
                    )
                )

                matched_new.add(best_index)

                logger.debug(
                    "Matched old=%s new=%s similarity=%.2f distance=%.2f",
                    old.id,
                    matched_element.id,
                    best_similarity,
                    best_distance,
                )

            else:

                result.unmatched_left.append(old)

                logger.debug(
                    "Unmatched old element id=%s page=%d type=%s",
                    old.id,
                    old.page_number,
                    old.type.value,
                )

        for index, element in enumerate(new_elements):

            if index not in matched_new:

                result.unmatched_right.append(element)

                logger.debug(
                    "Unmatched new element id=%s page=%d type=%s",
                    element.id,
                    element.page_number,
                    element.type.value,
                )

        logger.info(
            (
                "Alignment completed | "
                "Matches=%d | "
                "Unmatched Left=%d | "
                "Unmatched Right=%d"
            ),
            len(result.matches),
            len(result.unmatched_left),
            len(result.unmatched_right),
        )

        return result

    @staticmethod
    def _flatten_document(
        document: CanonicalDocument,
    ) -> list[Element]:

        elements: list[Element] = []

        for page in document.pages:
            elements.extend(page.elements)

        return elements

    @staticmethod
    def _bbox_distance(
        left: BoundingBox | None,
        right: BoundingBox | None,
    ) -> float:
        """
        Euclidean distance between bbox centers.
        """

        if left is None or right is None:
            return float("inf")

        left_x = (left.x0 + left.x1) / 2
        left_y = (left.y0 + left.y1) / 2

        right_x = (right.x0 + right.x1) / 2
        right_y = (right.y0 + right.y1) / 2

        return sqrt(
            (left_x - right_x) ** 2
            + (left_y - right_y) ** 2
        )
