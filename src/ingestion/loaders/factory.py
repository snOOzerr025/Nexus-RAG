"""Loader factory — selects the appropriate document loader by file extension.

Centralizes loader selection so callers don't need to know about
specific loader implementations.
"""

from __future__ import annotations

from src.ingestion.loaders.base import DocumentLoader
from src.ingestion.loaders.html_loader import HTMLLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.loaders.text_loader import TextLoader


class LoaderFactory:
    """Factory that selects the correct DocumentLoader for a file extension.

    Usage:
        factory = LoaderFactory()
        loader = factory.get_loader(".md")
        doc = await loader.load("/path/to/file.md")
    """

    def __init__(self) -> None:
        """Initialize with all available loaders."""
        self._loaders: list[DocumentLoader] = [
            MarkdownLoader(),
            HTMLLoader(),
            PDFLoader(),
            TextLoader(),
        ]

    def get_loader(self, file_extension: str) -> DocumentLoader:
        """Get the appropriate loader for a file extension.

        Args:
            file_extension: File extension including the dot (e.g., '.md').

        Returns:
            A DocumentLoader instance that supports the extension.

        Raises:
            ValueError: If no loader supports the given extension.
        """
        ext = file_extension.lower()
        for loader in self._loaders:
            if loader.supports(ext):
                return loader

        supported = self.supported_extensions()
        raise ValueError(
            f"Unsupported file extension: '{file_extension}'. "
            f"Supported extensions: {supported}"
        )

    def supports(self, file_extension: str) -> bool:
        """Check if any loader supports the given extension.

        Args:
            file_extension: File extension including the dot.

        Returns:
            True if a loader exists for this extension.
        """
        ext = file_extension.lower()
        return any(loader.supports(ext) for loader in self._loaders)

    def supported_extensions(self) -> list[str]:
        """Return all supported file extensions."""
        return [".md", ".markdown", ".mkd", ".html", ".htm", ".pdf", ".txt"]
