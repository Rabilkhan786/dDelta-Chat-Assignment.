from src.canonical.model import (
    Alignment,
    AlignmentResult,
    BoundingBox,
    DeltaEntry,
    DeltaType,
    Element,
    ElementType,
)
from src.chat.llm import GroqChatProvider
from src.delta.engine import DeltaEngine


def test_modified_native_elements_have_normalized_confidence() -> None:
    """RapidFuzz's 0-100 score must become a report confidence from 0 to 1."""
    old = Element(
        id="a",
        page_number=1,
        type=ElementType.TEXT,
        text="pressure 10",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )
    new = Element(
        id="b",
        page_number=1,
        type=ElementType.TEXT,
        text="pressure 12",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )
    delta = DeltaEngine().compare(
        AlignmentResult(matches=[Alignment(left=old, right=new, similarity=90, bbox_distance=0)])
    )[0]
    assert delta.change_type.value == "modified"
    assert delta.confidence == 0.9


def test_rewrapped_whitespace_is_not_reported_as_modified() -> None:
    """Re-flowed line breaks/spacing alone shouldn't count as a real content change."""
    old = Element(
        id="a",
        page_number=1,
        type=ElementType.TEXT,
        text="pressure  10\nbar",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )
    new = Element(
        id="b",
        page_number=1,
        type=ElementType.TEXT,
        text="pressure 10 bar",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )
    delta = DeltaEngine().compare(
        AlignmentResult(matches=[Alignment(left=old, right=new, similarity=100, bbox_distance=0)])
    )[0]
    assert delta.change_type.value == "unchanged"


def test_ocr_confidence_discounts_similarity() -> None:
    """An OCR match should be less confident than an identical native-text match."""
    old = Element(
        id="a", page_number=1, type=ElementType.TEXT, text="valve", source="ocr", ocr_confidence=0.5
    )
    new = Element(
        id="b", page_number=1, type=ElementType.TEXT, text="valve", source="ocr", ocr_confidence=0.5
    )
    delta = DeltaEngine().compare(
        AlignmentResult(matches=[Alignment(left=old, right=new, similarity=100, bbox_distance=0)])
    )[0]
    assert delta.change_type.value == "unchanged"
    assert delta.confidence == 0.5


def test_llm_cost_estimate_uses_configured_token_rates() -> None:
    """Cost telemetry must be deterministic without making a provider call."""
    assert GroqChatProvider._estimate_cost(1_000_000, 1_000_000) == 0.13


def test_low_ocr_confidence_in_revision_a_is_not_lost_when_b_is_native():
    old = Element(
        id="a",
        page_number=1,
        type=ElementType.TEXT,
        text="10 bar",
        source="ocr",
        ocr_confidence=0.2,
    )
    new = Element(id="b", page_number=1, type=ElementType.TEXT, text="12 bar")
    match = Alignment(left=old, right=new, similarity=90, bbox_distance=0)
    delta = DeltaEngine().compare(AlignmentResult(matches=[match]))[0]
    assert delta.confidence == 0.18


def test_removed_labels_far_above_each_other_are_not_joined():
    lower = DeltaEntry(
        change_type=DeltaType.REMOVED,
        element_type=ElementType.TEXT,
        page_number=1,
        description="Removed text: 'lower'",
        confidence=1,
        region=BoundingBox(x0=10, y0=200, x1=50, y1=210),
    )
    upper = lower.model_copy(update={"region": BoundingBox(x0=10, y0=10, x1=50, y1=20)})
    assert not DeltaEngine._can_join(lower, upper)
