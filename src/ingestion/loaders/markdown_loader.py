"""Markdown document loader.

Parses .md files extracting heading hierarchy, frontmatter,
and code blocks. Produces clean plaintext with section metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

import chardet

from src.ingestion.loaders.base import DocumentLoader
from src.models.domain import ParsedDocument, Section
from src.utils.logging import get_logger
from src.utils.text_processing import normalize_whitespace, strip_markdown_formatting

logger = get_logger(__name__)


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown (.md) files."""

    def supports(self, file_extension: str) -> bool:
        """Check if file extension is markdown."""
        return file_extension.lower() in {".md", ".markdown", ".mkd"}

    async def load(self, file_path: str) -> ParsedDocument:
        """Parse a markdown file into structured content.

        Extracts:
        - Heading hierarchy (H1-H6)
        - Frontmatter (YAML between --- delimiters)
        - Code blocks with language tags
        - Clean plaintext without formatting

        Args:
            file_path: Path to the .md file.

        Returns:
            ParsedDocument with sections and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_bytes = path.read_bytes()

        # Detect encoding for non-UTF-8 files
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        raw_text = raw_bytes.decode(encoding, errors="replace")

        logger.debug(
            "loading_markdown",
            file=path.name,
            size_bytes=len(raw_bytes),
            encoding=encoding,
        )

        # Extract frontmatter if present
        metadata: dict = {"source_type": "markdown"}
        frontmatter, content = self._extract_frontmatter(raw_text)
        if frontmatter:
            metadata["frontmatter"] = frontmatter

        # Extract sections from heading hierarchy
        sections = self._extract_sections(content)

        # Extract code block languages for metadata
        code_languages = self._extract_code_languages(content)
        if code_languages:
            metadata["code_languages"] = code_languages

        # Count headings by level
        heading_counts: dict[str, int] = {}
        for section in sections:
            level_key = f"h{section.level}"
            heading_counts[level_key] = heading_counts.get(level_key, 0) + 1
        if heading_counts:
            metadata["heading_counts"] = heading_counts

        # Generate clean plaintext
        clean_text = strip_markdown_formatting(content)
        clean_text = normalize_whitespace(clean_text)

        logger.info(
            "markdown_loaded",
            file=path.name,
            sections=len(sections),
            clean_chars=len(clean_text),
        )

        return ParsedDocument(
            content=clean_text,
            metadata=metadata,
            sections=sections,
        )

    def _extract_frontmatter(self, text: str) -> tuple[dict | None, str]:
        """Extract YAML frontmatter from markdown text.

        Frontmatter is delimited by --- at the start of the file.

        Returns:
            Tuple of (frontmatter dict or None, remaining content).
        """
        pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = pattern.match(text)

        if not match:
            return None, text

        frontmatter_text = match.group(1)
        remaining = text[match.end():]

        # Simple key-value parsing (avoid full YAML dependency)
        frontmatter: dict = {}
        for line in frontmatter_text.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                frontmatter[key.strip()] = value.strip().strip('"').strip("'")

        return frontmatter, remaining

    def _extract_sections(self, text: str) -> list[Section]:
        """Extract sections based on heading markers (# H1, ## H2, etc.)."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))
        sections: list[Section] = []

        if not matches:
            # No headings — entire document is one section
            if text.strip():
                sections.append(
                    Section(
                        heading="Document",
                        level=1,
                        content=text.strip(),
                        start_char=0,
                        end_char=len(text),
                    )
                )
            return sections

        # Content before first heading
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(
                    Section(
                        heading="Preamble",
                        level=0,
                        content=preamble,
                        start_char=0,
                        end_char=matches[0].start(),
                    )
                )

        # Each heading and its content until the next heading
        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2).strip()
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()

            sections.append(
                Section(
                    heading=heading,
                    level=level,
                    content=content,
                    start_char=match.start(),
                    end_char=content_end,
                )
            )

        return sections

    def _extract_code_languages(self, text: str) -> list[str]:
        """Extract programming languages from fenced code blocks."""
        pattern = re.compile(r"```(\w+)")
        languages = list(set(pattern.findall(text)))
        return sorted(languages)
