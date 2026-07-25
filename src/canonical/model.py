from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """
    Types of elements extracted from a document.
    """

    TEXT = "text"
    TITLE = "title"
    NOTE = "note"
    TABLE = "table"
    DIMENSION = "dimension"
    SYMBOL = "symbol"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """
    Coordinates of an element on a page.
    """

    x0: float
    y0: float
    x1: float
    y1: float


class Element(BaseModel):
    """
    Smallest unit extracted from a document.
    """

    id: str

    page_number: int

    type: ElementType

    text: str

    bbox: Optional[BoundingBox] = None

    # Source of extracted text
    source: str = "native"

    # OCR confidence (None for native PDFs)
    ocr_confidence: Optional[float] = None


class Page(BaseModel):
    """
    One page of a document.
    """

    page_number: int

    width: float

    height: float

    elements: List[Element] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """
    Information about the document.
    """

    document_id: str
    
    pid: str

    file_name: str

    file_type: str

    revision: Optional[str] = None
    


class CanonicalDocument(BaseModel):
    """
    Standard representation used by the whole pipeline.
    """

    metadata: DocumentMetadata

    pages: List[Page] = Field(default_factory=list)
    
    
class Alignment(BaseModel):
    """
    Represents one aligned pair of elements.
    """

    left: Element
    right: Element
    similarity: float
    bbox_distance: float
    
    

class AlignmentResult(BaseModel):
    """
    Output of the alignment stage.
    """

    matches: list[Alignment] = field(default_factory=list)
    unmatched_left: list[Element] = field(default_factory=list)
    unmatched_right: list[Element] = field(default_factory=list)
    
    

class DeltaType(str, Enum):
    """
    Type of change detected between document revisions.
    """

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    
    
class DeltaEntry(BaseModel):
    """
    Represents one detected change between two document revisions.
    """

    change_type: DeltaType

    element_type: ElementType

    page_number: int

    region: BoundingBox | None = None

    description: str

    confidence: float
