"""Plain text document loader.

Simplest loader — reads .txt files with encoding detection.
"""

from __future__ import annotations

from pathlib import Path

import chardet

from src.ingestion.loaders.base import DocumentLoader
from src.models.domain import ParsedDocument, Section
from src.utils.logging import get_logger
from src.utils.text_processing import normalize_whitespace

logger = get_logger(__name__)


class TextLoader(DocumentLoader):
    """Loader for plain text (.txt) files."""

    def supports(self, file_extension: str) -> bool:
        """Check if file extension is plain text."""
        return file_extension.lower() == ".txt"

    async def load(self, file_path: str) -> ParsedDocument:
        """Parse a plain text file.

        Reads the file as-is with encoding detection.
        Entire content becomes a single section.

        Args:
            file_path: Path to the .txt file.

        Returns:
            ParsedDocument with the text content and basic metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_bytes = path.read_bytes()

        # Detect encoding
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        raw_text = raw_bytes.decode(encoding, errors="replace")

        logger.debug(
            "loading_text",
            file=path.name,
            size_bytes=len(raw_bytes),
            encoding=encoding,
        )

        clean_text = normalize_whitespace(raw_text)
        line_count = clean_text.count("\n") + 1

        metadata: dict = {
            "source_type": "text",
            "line_count": line_count,
            "encoding": encoding,
        }

        # Single section for the entire document
        sections = [
            Section(
                heading="Document",
                level=1,
                content=clean_text,
                start_char=0,
                end_char=len(clean_text),
            )
        ]

        logger.info(
            "text_loaded",
            file=path.name,
            lines=line_count,
            clean_chars=len(clean_text),
        )

        return ParsedDocument(
            content=clean_text,
            metadata=metadata,
            sections=sections,
        )
