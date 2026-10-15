"""Abstract base class for text chunking strategies.

Every chunking strategy (fixed, recursive, semantic) implements
this interface. The ChunkerFactory selects the appropriate chunker
based on strategy name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.domain import ChunkRecord
from src.models.enums import ChunkingStrategy


class Chunker(ABC):
    """Base class for all text chunking strategies.

    Subclasses split document text into chunks with metadata,
    using different strategies for different content types.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str,
        metadata: dict | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[ChunkRecord]:
        """Split text into chunks with metadata.

        Args:
            text: The full document text to split.
            document_id: UUID of the parent document.
            source_file: Original filename for metadata.
            metadata: Optional metadata dict with section headings, etc.
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Number of overlapping tokens between chunks.

        Returns:
            List of ChunkRecord objects with complete metadata.
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> ChunkingStrategy:
        """Return the chunking strategy identifier."""
        ...
