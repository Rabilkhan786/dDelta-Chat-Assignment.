from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from src.canonical.model import DeltaEntry , CanonicalDocument
from src.config.settings import settings

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Handles indexing of:

    - Old Revision PDF
    - New Revision PDF
    - Delta Report JSON

    into:

    - Chroma Vector Database (Dense Retrieval)
    - BM25 Index (Sparse Retrieval)
    """

    def __init__(self) -> None:
        """
        Initialize embedding model, vector store,
        and BM25 storage.
        """

        logger.info("Initializing Document Indexer...")

        
        # Embedding Model

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding.model,
            model_kwargs={
                "device": settings.embedding.device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        logger.info(
            "Loaded embedding model: %s",
            settings.embedding.model,
        )

     
        # Chroma Vector Database
       

        self.vector_store = Chroma(
            collection_name=settings.chroma.collection_name,
            persist_directory=settings.chroma.persist_directory,
            embedding_function=self.embedding_model,
        )

        logger.info(
            "Connected to Chroma collection: %s",
            settings.chroma.collection_name,
        )

      
        # BM25 Storage Paths
       

        self.bm25_index_path = Path(
            settings.paths.bm25_index
        )

        self.documents_path = Path(
            settings.paths.bm25_documents
        )

        # Create directory if it doesn't exist
        self.bm25_index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # BM25 Runtime Objects
        

        self.bm25_index: BM25Okapi | None = None
        self.documents: List[Document] = []

        logger.info("Document Indexer initialized successfully.")
        


    def create_documents(
        self,
        document: CanonicalDocument,
    ) -> List[Document]:
        """
        Convert a CanonicalDocument into LangChain Documents.

        Parameters
        ----------
        document
            Canonical representation of one document revision.

        Returns
        -------
        List[Document]
            List of LangChain documents.
        """

        logger.info(
            "Creating LangChain documents for '%s'",
            document.metadata.file_name,
        )

        documents: List[Document] = []

        for page in document.pages:

            logger.debug(
                "Processing page %d",
                page.page_number,
            )

            for element in page.elements:

                text = element.text.strip()

                if not text:
                    continue

                if len(text) == 1 and text.isalpha():
                    continue

                langchain_document = Document(
                    page_content=text,
                    metadata={
                        "document_id": document.metadata.document_id,
                        "pid": document.metadata.pid,
                        "file_name": document.metadata.file_name,
                        "revision": document.metadata.revision,
                        "page_number": element.page_number,
                        "element_id": element.id,
                        "element_type": element.type.value,
                        "source": getattr(element, "source", "native"),
                        "ocr_confidence": getattr(
                            element,
                            "ocr_confidence",
                            None,
                        ),
                        "bbox": {
                            "x0": element.bbox.x0,
                            "y0": element.bbox.y0,
                            "x1": element.bbox.x1,
                            "y1": element.bbox.y1,
                        }
                        if element.bbox
                        else None,
                    },
                )

                documents.append(langchain_document)

        logger.info(
            "Created %d LangChain documents",
            len(documents),
        )

        return documents
    
    def create_delta_documents(
        self,
        deltas: list[DeltaEntry],
    ) -> List[Document]:
        """
        Convert DeltaEntry objects into LangChain Documents.
        """

        logger.info(
            "Creating LangChain documents from '%d' delta entries.",
            len(deltas),
        )

        documents: List[Document] = []

        for delta in deltas:

            documents.append(
                Document(
                    page_content=(
                        f"Change Type: {delta.change_type.value}\n"
                        f"Element Type: {delta.element_type.value}\n"
                        f"Page Number: {delta.page_number}\n"
                        f"Description: {delta.description}\n"
                        f"Confidence: {delta.confidence:.2f}"
                    ),
                    metadata={
                        "source": "delta_report",
                        "change_type": delta.change_type.value,
                        "element_type": delta.element_type.value,
                        "page_number": delta.page_number,
                        "confidence": delta.confidence,
                        "bbox": (
                            delta.region.model_dump()
                            if delta.region is not None
                            else None
                        ),
                    },
                )
            )

        logger.info(
            "Created %d LangChain delta documents.",
            len(documents),
        )

        return documents