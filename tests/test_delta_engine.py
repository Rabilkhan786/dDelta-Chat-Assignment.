from src.canonical.model import Alignment, AlignmentResult, BoundingBox, Element, ElementType
from src.delta.engine import DeltaEngine


def test_modified_native_elements_have_normalized_confidence() -> None:
    """RapidFuzz's 0-100 score must become a report confidence from 0 to 1."""
    old = Element(id="a", page_number=1, type=ElementType.TEXT, text="pressure 10", bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1))
    new = Element(id="b", page_number=1, type=ElementType.TEXT, text="pressure 12", bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1))
    delta = DeltaEngine().compare(AlignmentResult(matches=[Alignment(left=old, right=new, similarity=90, bbox_distance=0)]))[0]
    assert delta.change_type.value == "modified"
    assert delta.confidence == 0.9
