"""HTML document loader.

Parses .html/.htm files using BeautifulSoup. Strips scripts, styles,
and navigation. Extracts title, meta tags, and heading hierarchy.
"""

from __future__ import annotations

from pathlib import Path

import chardet
from bs4 import BeautifulSoup, Tag

from src.ingestion.loaders.base import DocumentLoader
from src.models.domain import ParsedDocument, Section
from src.utils.logging import get_logger
from src.utils.text_processing import normalize_whitespace

logger = get_logger(__name__)

# Tags to remove entirely (content and all)
REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}


class HTMLLoader(DocumentLoader):
    """Loader for HTML (.html, .htm) files."""

    def supports(self, file_extension: str) -> bool:
        """Check if file extension is HTML."""
        return file_extension.lower() in {".html", ".htm"}

    async def load(self, file_path: str) -> ParsedDocument:
        """Parse an HTML file into structured content.

        Extracts:
        - <title> tag content
        - <meta> description and keywords
        - Heading hierarchy (H1-H6)
        - Clean text with tags stripped

        Args:
            file_path: Path to the .html/.htm file.

        Returns:
            ParsedDocument with sections and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_bytes = path.read_bytes()

        # Detect encoding
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        raw_html = raw_bytes.decode(encoding, errors="replace")

        logger.debug(
            "loading_html",
            file=path.name,
            size_bytes=len(raw_bytes),
            encoding=encoding,
        )

        soup = BeautifulSoup(raw_html, "html.parser")

        # Extract metadata from <head>
        metadata: dict = {"source_type": "html"}

        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()

        # Extract meta description and keywords
        for meta in soup.find_all("meta"):
            meta_name = meta.get("name")
            name = (meta_name if isinstance(meta_name, str) else "").lower()
            content = meta.get("content", "")
            if isinstance(content, str):
                if name == "description" and content:
                    metadata["description"] = content
                elif name == "keywords" and content:
                    metadata["keywords"] = [k.strip() for k in content.split(",")]

        # Remove unwanted tags (scripts, styles, navigation)
        for tag_name in REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Extract sections from heading hierarchy
        sections = self._extract_sections(soup)

        # Get clean text content
        clean_text = soup.get_text(separator="\n")
        clean_text = normalize_whitespace(clean_text)

        logger.info(
            "html_loaded",
            file=path.name,
            sections=len(sections),
            clean_chars=len(clean_text),
            title=metadata.get("title", "untitled"),
        )

        return ParsedDocument(
            content=clean_text,
            metadata=metadata,
            sections=sections,
        )

    def _extract_sections(self, soup: BeautifulSoup) -> list[Section]:
        """Extract sections based on H1-H6 heading elements."""
        heading_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        sections: list[Section] = []

        if not heading_tags:
            # No headings — return entire body as one section
            body = soup.find("body")
            text = body.get_text(separator="\n") if body else soup.get_text(separator="\n")
            text = normalize_whitespace(text)
            if text:
                sections.append(
                    Section(
                        heading="Document",
                        level=1,
                        content=text,
                        start_char=0,
                        end_char=len(text),
                    )
                )
            return sections

        running_pos = 0

        for heading_tag in heading_tags:
            if not isinstance(heading_tag, Tag):
                continue

            level = int(heading_tag.name[1])  # h1 -> 1, h2 -> 2, etc.
            heading_text = heading_tag.get_text(strip=True)

            # Collect text between this heading and the next heading
            content_parts: list[str] = []
            sibling = heading_tag.next_sibling
            while sibling:
                if isinstance(sibling, Tag):
                    if sibling.name and sibling.name in {
                        "h1", "h2", "h3", "h4", "h5", "h6"
                    }:
                        break
                    content_parts.append(sibling.get_text(separator="\n"))
                elif isinstance(sibling, str) and sibling.strip():
                    content_parts.append(sibling.strip())
                sibling = sibling.next_sibling

            content = normalize_whitespace("\n".join(content_parts))
            end_pos = running_pos + len(heading_text) + len(content) + 1

            sections.append(
                Section(
                    heading=heading_text,
                    level=level,
                    content=content,
                    start_char=running_pos,
                    end_char=end_pos,
                )
            )
            running_pos = end_pos

        return sections
