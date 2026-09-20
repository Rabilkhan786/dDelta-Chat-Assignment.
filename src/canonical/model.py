"""The format-agnostic representation every ingestion adapter normalizes into."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ElementType(StrEnum):
    """Types of elements extracted from a document."""

    TEXT = "text"
    TITLE = "title"
    NOTE = "note"
    TABLE = "table"
    DIMENSION = "dimension"
    SYMBOL = "symbol"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Coordinates of an element on a page."""

    x0: float
    y0: float
    x1: float
    y1: float


class Element(BaseModel):
    """One line of text or a supported technical tag in a document."""

    id: str
    page_number: int = Field(ge=1)
    type: ElementType
    text: str
    bbox: BoundingBox | None = None
    source: str = "native"  # "native" or "ocr"
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Page(BaseModel):
    """One page of a document."""

    page_number: int = Field(ge=1)
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)
    elements: list[Element] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Information about the document."""

    document_id: str
    pid: str
    file_name: str
    file_type: str
    revision: str | None = None


class CanonicalDocument(BaseModel):
    """Standard representation used by the whole pipeline."""

    metadata: DocumentMetadata
    pages: list[Page] = Field(default_factory=list)


class Alignment(BaseModel):
    """Represents one aligned pair of elements."""

    left: Element
    right: Element
    similarity: float = Field(ge=0.0, le=100.0)
    bbox_distance: float = Field(ge=0.0)
    matched_after_move: bool = False


class AlignmentResult(BaseModel):
    """Output of the alignment stage."""

    matches: list[Alignment] = Field(default_factory=list)
    unmatched_left: list[Element] = Field(default_factory=list)
    unmatched_right: list[Element] = Field(default_factory=list)


class DeltaType(StrEnum):
    """Type of change detected between document revisions."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MOVED = "moved"
    UNCHANGED = "unchanged"


class DeltaEntry(BaseModel):
    """Represents one detected change between two document revisions."""

    change_type: DeltaType
    element_type: ElementType
    page_number: int = Field(ge=1)
    region: BoundingBox | None = None
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    element_id: str | None = None
    previous_element_id: str | None = None
    location_changed: bool = False
