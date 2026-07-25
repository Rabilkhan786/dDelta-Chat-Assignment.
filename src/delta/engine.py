"""
Delta Engine

Converts AlignmentResult into structured DeltaEntry objects.
"""

from src.observability.logging import get_logger, stage

from src.canonical.model import (
    Alignment,
    AlignmentResult,
    DeltaEntry,
    DeltaType,
    Element,
)

logger = get_logger(__name__)


class DeltaEngine:
    """
    Rule-based delta detection engine.
    """

    def compare(
        self,
        alignment_result: AlignmentResult,
    ) -> list[DeltaEntry]:
        """
        Compare two aligned documents and return detected changes.
        """

        with stage(logger, "delta_comparison"):
            return self._compare(alignment_result)

    def _compare(self, alignment_result: AlignmentResult) -> list[DeltaEntry]:
        """Perform deterministic classification over an existing alignment."""

        deltas: list[DeltaEntry] = []

        unchanged_count = 0
        modified_count = 0
        added_count = 0
        removed_count = 0

        # Matched Elements 
       

        for match in alignment_result.matches:

            if self._is_modified(match):

                delta = self._create_modified_delta(match)

                modified_count += 1

                logger.debug(
                    "Modified element old=%s new=%s confidence=%.2f",
                    match.left.id,
                    match.right.id,
                    delta.confidence,
                )

            else:

                delta = self._create_unchanged_delta(match)

                unchanged_count += 1

                logger.debug(
                    "Unchanged element old=%s new=%s confidence=%.2f",
                    match.left.id,
                    match.right.id,
                    delta.confidence,
                )

            deltas.append(delta)

        
        # Removed Elements
        

        for element in alignment_result.unmatched_left:

            delta = self._create_removed_delta(element)

            deltas.append(delta)

            removed_count += 1

            logger.debug(
                "Removed element id=%s confidence=%.2f",
                element.id,
                delta.confidence,
            )

        # Added Elements
        

        for element in alignment_result.unmatched_right:

            delta = self._create_added_delta(element)

            deltas.append(delta)

            added_count += 1

            logger.debug(
                "Added element id=%s confidence=%.2f",
                element.id,
                delta.confidence,
            )

        logger.info(
            (
                "Delta comparison completed | "
                "Unchanged=%d | "
                "Modified=%d | "
                "Added=%d | "
                "Removed=%d | "
                "Total=%d"
            ),
            unchanged_count,
            modified_count,
            added_count,
            removed_count,
            len(deltas),
        )

        return deltas

    @staticmethod
    def _is_modified(match: Alignment) -> bool:
        """
        Returns True if two aligned elements differ.
        """

        return (
            match.left.text.strip()
            !=
            match.right.text.strip()
        )

    def _create_unchanged_delta(
        self,
        match: Alignment,
    ) -> DeltaEntry:

        return DeltaEntry(
            change_type=DeltaType.UNCHANGED,
            element_type=match.right.type,
            page_number=match.right.page_number,
            region=match.right.bbox,
            description=(
                f"Unchanged {match.right.type.value}: "
                f"'{match.right.text}'"
            ),
            confidence=self._calculate_confidence(
                similarity=match.similarity,
                old_element=match.left,
                new_element=match.right,
            ),
        )

    def _create_modified_delta(
        self,
        match: Alignment,
    ) -> DeltaEntry:

        return DeltaEntry(
            change_type=DeltaType.MODIFIED,
            element_type=match.right.type,
            page_number=match.right.page_number,
            region=match.right.bbox,
            description=(
                f"{match.right.type.value} changed "
                f"from '{match.left.text}' "
                f"to '{match.right.text}'"
            ),
            confidence=self._calculate_confidence(
                similarity=match.similarity,
                old_element=match.left,
                new_element=match.right,
            ),
        )

    def _create_removed_delta(
        self,
        element: Element,
    ) -> DeltaEntry:

        return DeltaEntry(
            change_type=DeltaType.REMOVED,
            element_type=element.type,
            page_number=element.page_number,
            region=element.bbox,
            description=(
                f"Removed {element.type.value}: "
                f"'{element.text}'"
            ),
            confidence=self._calculate_confidence(
                similarity=100.0,
                old_element=element,
                new_element=None,
            ),
        )

    def _create_added_delta(
        self,
        element: Element,
    ) -> DeltaEntry:

        return DeltaEntry(
            change_type=DeltaType.ADDED,
            element_type=element.type,
            page_number=element.page_number,
            region=element.bbox,
            description=(
                f"Added {element.type.value}: "
                f"'{element.text}'"
            ),
            confidence=self._calculate_confidence(
                similarity=100.0,
                old_element=None,
                new_element=element,
            ),
        )

    @staticmethod
    def _calculate_confidence(
        similarity: float,
        old_element: Element | None,
        new_element: Element | None,
    ) -> float:
        """
        Calculate confidence for a detected change.

        Native PDF:
            confidence = similarity

        OCR:
            confidence = similarity × OCR confidence
        """

        source = None
        ocr_confidence = None

        if new_element is not None:
            source = new_element.source
            ocr_confidence = new_element.ocr_confidence

        elif old_element is not None:
            source = old_element.source
            ocr_confidence = old_element.ocr_confidence

        normalized_similarity = max(0.0, min(similarity / 100.0, 1.0))
        if source == "native" or ocr_confidence is None:
            confidence = normalized_similarity

        else:
            confidence = normalized_similarity * ocr_confidence

        return round(min(confidence, 1.0), 2)
