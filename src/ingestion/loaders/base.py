"""Abstract base class for document loaders.

Every format-specific loader (Markdown, HTML, PDF, Text) implements
this interface. The LoaderFactory selects the appropriate loader
based on file extension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.domain import ParsedDocument


class DocumentLoader(ABC):
    """Base class for all document format loaders.

    Subclasses must implement load() and supports() to handle
    specific file formats.
    """

    @abstractmethod
    async def load(self, file_path: str) -> ParsedDocument:
        """Parse a document file into structured content.

        Args:
            file_path: Absolute path to the document file.

        Returns:
            ParsedDocument with clean text content, metadata, and sections.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file content cannot be parsed.
        """
        ...

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Check if this loader can handle the given file type.

        Args:
            file_extension: File extension including the dot (e.g., '.md').

        Returns:
            True if this loader supports the extension.
        """
        ...
