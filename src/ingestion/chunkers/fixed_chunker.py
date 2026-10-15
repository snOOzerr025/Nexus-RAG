"""Fixed-size token-based chunking with configurable overlap.

The simplest chunking strategy — slides a fixed-size window over
the token sequence with overlap between consecutive chunks.
Good baseline for comparison against structure-aware strategies.
"""

from __future__ import annotations

from src.ingestion.chunkers.base import Chunker
from src.models.domain import ChunkRecord
from src.models.enums import ChunkingStrategy
from src.utils.logging import get_logger
from src.utils.tokenizer import count_tokens, split_by_tokens

logger = get_logger(__name__)


class FixedChunker(Chunker):
    """Fixed-size chunking with token-based splitting and overlap.

    Splits text into chunks of a target token size with configurable
    overlap between consecutive chunks. Uses tiktoken for accurate
    token counting at OpenAI model boundaries.
    """

    @property
    def strategy_name(self) -> ChunkingStrategy:
        """Return strategy identifier."""
        return ChunkingStrategy.FIXED

    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str,
        metadata: dict | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[ChunkRecord]:
        """Split text into fixed-size token chunks with overlap.

        Args:
            text: Full document text.
            document_id: Parent document UUID.
            source_file: Original filename.
            metadata: Optional metadata (section headings not used for fixed).
            chunk_size: Target tokens per chunk (default: 512).
            chunk_overlap: Overlapping tokens between chunks (default: 50).

        Returns:
            List of ChunkRecord objects.
        """
        if not text.strip():
            logger.warning("fixed_chunker_empty_text", source_file=source_file)
            return []

        # Split by tokens using sliding window
        chunk_texts = split_by_tokens(text, chunk_size, chunk_overlap)

        chunks: list[ChunkRecord] = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            token_count = count_tokens(chunk_text)

            chunk = ChunkRecord(
                document_id=document_id,
                content=chunk_text,
                source_file=source_file,
                section_heading=None,  # Fixed chunking doesn't track sections
                chunk_index=i,
                chunking_strategy=ChunkingStrategy.FIXED,
                char_count=len(chunk_text),
                token_count=token_count,
            )
            chunks.append(chunk)

        logger.info(
            "fixed_chunking_complete",
            source_file=source_file,
            total_chunks=len(chunks),
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        return chunks
