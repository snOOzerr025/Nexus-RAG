"""Recursive character-based chunking aware of document structure.

Splits by heading hierarchy first (H1 → H2 → H3 → paragraphs),
then falls back to character/sentence boundaries. Preserves
section headings in chunk metadata for better retrieval context.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunkers.base import Chunker
from src.models.domain import ChunkRecord
from src.models.enums import ChunkingStrategy
from src.utils.logging import get_logger
from src.utils.text_processing import extract_markdown_sections
from src.utils.tokenizer import count_tokens

logger = get_logger(__name__)

# Separator hierarchy: try heading-level splits first, then paragraphs, etc.
SEPARATORS = [
    "\n# ",      # H1
    "\n## ",     # H2
    "\n### ",    # H3
    "\n#### ",   # H4
    "\n\n",      # Paragraph break
    "\n",        # Line break
    ". ",        # Sentence boundary
    " ",         # Word boundary
]


class RecursiveChunker(Chunker):
    """Structure-aware recursive chunking using heading hierarchy.

    Uses LangChain's RecursiveCharacterTextSplitter with a separator
    list that prioritizes document structure (headings, paragraphs)
    over arbitrary character boundaries.

    Preserves section headings in chunk metadata so retrieval can
    show users WHERE in the document a chunk came from.
    """

    @property
    def strategy_name(self) -> ChunkingStrategy:
        """Return strategy identifier."""
        return ChunkingStrategy.RECURSIVE

    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str,
        metadata: dict | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[ChunkRecord]:
        """Split text recursively by heading hierarchy.

        Uses LangChain's RecursiveCharacterTextSplitter with custom
        separators that respect markdown/document structure.

        Args:
            text: Full document text (may contain markdown headings).
            document_id: Parent document UUID.
            source_file: Original filename.
            metadata: Optional metadata dict.
            chunk_size: Target tokens per chunk.
            chunk_overlap: Overlapping tokens between chunks.

        Returns:
            List of ChunkRecord objects with section_heading metadata.
        """
        if not text.strip():
            logger.warning("recursive_chunker_empty_text", source_file=source_file)
            return []

        # LangChain splitter uses character count, so approximate
        # token-to-char ratio. For English text, ~4 chars per token.
        chars_per_token = 4
        char_chunk_size = chunk_size * chars_per_token
        char_overlap = chunk_overlap * chars_per_token

        splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=char_chunk_size,
            chunk_overlap=char_overlap,
            length_function=len,  # Character-based for speed, token count added later
            strip_whitespace=True,
        )

        # Extract section structure for heading assignment
        sections = extract_markdown_sections(text)

        # Split the text
        split_texts = splitter.split_text(text)

        chunks: list[ChunkRecord] = []
        last_pos = 0
        for i, chunk_text in enumerate(split_texts):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Determine which section this chunk belongs to by finding its character offset
            chunk_pos = text.find(chunk_text[:50], last_pos)
            if chunk_pos != -1:
                last_pos = chunk_pos
            else:
                chunk_pos = text.find(chunk_text[:50])

            section_heading = self._find_section_heading(chunk_pos, sections)
            token_count = count_tokens(chunk_text)

            chunk = ChunkRecord(
                document_id=document_id,
                content=chunk_text,
                source_file=source_file,
                section_heading=section_heading,
                chunk_index=i,
                chunking_strategy=ChunkingStrategy.RECURSIVE,
                char_count=len(chunk_text),
                token_count=token_count,
            )
            chunks.append(chunk)

        logger.info(
            "recursive_chunking_complete",
            source_file=source_file,
            total_chunks=len(chunks),
            sections_found=len(sections),
            chunk_size=chunk_size,
        )

        return chunks

    def _find_section_heading(
        self, chunk_pos: int, sections: list[dict]
    ) -> str | None:
        """Find which section heading a chunk belongs to by character position.

        Matches chunk start index against section boundaries to determine
        the most specific heading.

        Args:
            chunk_pos: The character index where the chunk starts in the original document.
            sections: List of section dicts from extract_markdown_sections.

        Returns:
            The heading string, or None if no match found.
        """
        if not sections or chunk_pos < 0:
            return None

        best_match: str | None = None
        best_level = -1

        for section in sections:
            start_char = section.get("start_char", 0)
            end_char = section.get("end_char", float('inf'))
            heading = section.get("heading", "")

            if start_char <= chunk_pos < end_char:
                level = section.get("level", 0)
                if level >= best_level:
                    best_match = heading
                    best_level = level

        return best_match
