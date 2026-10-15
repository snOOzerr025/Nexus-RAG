"""PDF document loader.

Parses .pdf files using PyPDF for text extraction.
Tracks page numbers for each section of extracted text.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.ingestion.loaders.base import DocumentLoader
from src.models.domain import ParsedDocument, Section
from src.utils.logging import get_logger
from src.utils.text_processing import normalize_whitespace

logger = get_logger(__name__)


class PDFLoader(DocumentLoader):
    """Loader for PDF (.pdf) files."""

    def supports(self, file_extension: str) -> bool:
        """Check if file extension is PDF."""
        return file_extension.lower() == ".pdf"

    async def load(self, file_path: str) -> ParsedDocument:
        """Parse a PDF file into structured content.

        Extracts text page by page, tracking page numbers.
        Each page becomes a section for downstream chunking.

        Args:
            file_path: Path to the .pdf file.

        Returns:
            ParsedDocument with per-page sections and metadata.

        Raises:
            ValueError: If no text could be extracted (possibly a scanned PDF).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.debug("loading_pdf", file=path.name, size_bytes=path.stat().st_size)

        reader = PdfReader(str(path))

        # Extract PDF metadata
        metadata: dict = {
            "source_type": "pdf",
            "page_count": len(reader.pages),
        }

        pdf_meta = reader.metadata
        if pdf_meta:
            if pdf_meta.title:
                metadata["title"] = pdf_meta.title
            if pdf_meta.author:
                metadata["author"] = pdf_meta.author
            if pdf_meta.creator:
                metadata["creator"] = pdf_meta.creator
            if pdf_meta.creation_date:
                metadata["creation_date"] = str(pdf_meta.creation_date)

        # Extract text page by page
        sections: list[Section] = []
        all_text_parts: list[str] = []
        running_pos = 0
        empty_pages = 0

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = normalize_whitespace(page_text)

            if not page_text.strip():
                empty_pages += 1
                continue

            all_text_parts.append(page_text)
            end_pos = running_pos + len(page_text)

            sections.append(
                Section(
                    heading=f"Page {page_num}",
                    level=1,
                    content=page_text,
                    start_char=running_pos,
                    end_char=end_pos,
                )
            )
            running_pos = end_pos + 1  # +1 for newline separator

        if not all_text_parts:
            logger.warning(
                "pdf_no_text_extracted",
                file=path.name,
                pages=len(reader.pages),
                hint="This may be a scanned PDF without OCR text layer.",
            )
            raise ValueError(
                f"No text could be extracted from '{path.name}'. "
                "This may be a scanned PDF that requires OCR."
            )

        if empty_pages > 0:
            metadata["empty_pages"] = empty_pages
            logger.debug(
                "pdf_empty_pages",
                file=path.name,
                empty_pages=empty_pages,
                total_pages=len(reader.pages),
            )

        clean_text = "\n".join(all_text_parts)

        logger.info(
            "pdf_loaded",
            file=path.name,
            pages=len(reader.pages),
            sections=len(sections),
            clean_chars=len(clean_text),
        )

        return ParsedDocument(
            content=clean_text,
            metadata=metadata,
            sections=sections,
        )
