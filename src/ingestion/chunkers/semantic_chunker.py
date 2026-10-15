"""Semantic chunking based on embedding similarity between sentences.

The most sophisticated chunking strategy: splits text into sentences,
embeds each one, and groups consecutive sentences into chunks where
they are topically coherent (cosine similarity above a threshold).
When similarity drops below the threshold, a new chunk starts.

More expensive than other strategies (requires embedding each sentence)
but produces the most topically coherent chunks.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from src.ingestion.chunkers.base import Chunker
from src.models.domain import ChunkRecord
from src.models.enums import ChunkingStrategy
from src.utils.logging import get_logger
from src.utils.tokenizer import count_tokens

if TYPE_CHECKING:
    from src.clients.openai_client import OpenAIClient

logger = get_logger(__name__)

# Regex to split text into sentences
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class SemanticChunker(Chunker):
    """Semantic chunking that splits on topic boundaries.

    Algorithm:
    1. Split text into sentences
    2. Embed all sentences (batched for efficiency)
    3. Compute cosine similarity between consecutive sentences
    4. When similarity drops below threshold, start a new chunk
    5. Group sentences between split points into chunks
    6. If any chunk exceeds max size, sub-split with fixed strategy
    """

    def __init__(
        self,
        openai_client: OpenAIClient,
        similarity_threshold: float = 0.5,
    ) -> None:
        """Initialize semantic chunker.

        Args:
            openai_client: OpenAI client for embedding sentences.
            similarity_threshold: Minimum cosine similarity between
                consecutive sentences to stay in the same chunk.
                Lower = larger chunks, higher = smaller chunks.
        """
        self._openai_client = openai_client
        self._threshold = similarity_threshold

    @property
    def strategy_name(self) -> ChunkingStrategy:
        """Return strategy identifier."""
        return ChunkingStrategy.SEMANTIC

    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str,
        metadata: dict | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[ChunkRecord]:
        """Split text into semantically coherent chunks.

        Note: This method is synchronous but internally delegates to the
        async chunk_async(). It safely handles being called both from
        within and outside an existing event loop.

        Args:
            text: Full document text.
            document_id: Parent document UUID.
            source_file: Original filename.
            metadata: Optional metadata dict.
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Not directly used (semantic boundaries determine splits).

        Returns:
            List of ChunkRecord objects.
        """
        import asyncio
        import concurrent.futures

        coro = self.chunk_async(
            text, document_id, source_file, metadata, chunk_size, chunk_overlap
        )

        try:
            asyncio.get_running_loop()
            # Already in an async context — run in a new thread to avoid
            # "asyncio.run() cannot be called from a running event loop"
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run() directly
            return asyncio.run(coro)

    async def chunk_async(
        self,
        text: str,
        document_id: str,
        source_file: str,
        metadata: dict | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[ChunkRecord]:
        """Async version of chunk() — preferred in async contexts.

        Args:
            text: Full document text.
            document_id: Parent document UUID.
            source_file: Original filename.
            metadata: Optional metadata dict.
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Not directly used for semantic splitting.

        Returns:
            List of ChunkRecord objects with semantic chunk boundaries.
        """
        if not text.strip():
            logger.warning("semantic_chunker_empty_text", source_file=source_file)
            return []

        # Step 1: Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            # Single sentence or very short doc — return as one chunk
            token_count = count_tokens(text.strip())
            return [
                ChunkRecord(
                    document_id=document_id,
                    content=text.strip(),
                    source_file=source_file,
                    chunk_index=0,
                    chunking_strategy=ChunkingStrategy.SEMANTIC,
                    char_count=len(text.strip()),
                    token_count=token_count,
                )
            ]

        logger.debug(
            "semantic_chunking_sentences",
            source_file=source_file,
            sentence_count=len(sentences),
        )

        # Step 2: Embed all sentences (batched)
        embeddings = await self._openai_client.embed_batch(sentences)

        # Step 3: Compute similarities between consecutive sentences
        similarities: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Step 4: Find split points (where similarity drops below threshold)
        split_indices: list[int] = []
        for i, sim in enumerate(similarities):
            if sim < self._threshold:
                split_indices.append(i + 1)  # Split AFTER this sentence

        # Step 5: Group sentences into chunks
        chunk_groups: list[list[str]] = []
        start = 0
        for split_idx in split_indices:
            group = sentences[start:split_idx]
            if group:
                chunk_groups.append(group)
            start = split_idx
        # Last group
        if start < len(sentences):
            chunk_groups.append(sentences[start:])

        # Step 6: Build ChunkRecords, sub-splitting oversized chunks
        chunks: list[ChunkRecord] = []
        chunk_index = 0

        for group in chunk_groups:
            group_text = " ".join(group).strip()
            if not group_text:
                continue

            token_count = count_tokens(group_text)

            if token_count > chunk_size * 1.5:
                # Sub-split oversized chunks using simple sentence grouping
                sub_chunks = self._sub_split(group, chunk_size)
                for sub_text in sub_chunks:
                    sub_text = sub_text.strip()
                    if not sub_text:
                        continue
                    chunk = ChunkRecord(
                        document_id=document_id,
                        content=sub_text,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        chunking_strategy=ChunkingStrategy.SEMANTIC,
                        char_count=len(sub_text),
                        token_count=count_tokens(sub_text),
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                chunk = ChunkRecord(
                    document_id=document_id,
                    content=group_text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    chunking_strategy=ChunkingStrategy.SEMANTIC,
                    char_count=len(group_text),
                    token_count=token_count,
                )
                chunks.append(chunk)
                chunk_index += 1

        logger.info(
            "semantic_chunking_complete",
            source_file=source_file,
            total_chunks=len(chunks),
            split_points=len(split_indices),
            threshold=self._threshold,
        )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Uses regex to split on sentence-ending punctuation followed
        by whitespace and a capital letter. Falls back to newlines
        for text without standard punctuation.
        """
        sentences = SENTENCE_PATTERN.split(text)

        # If regex produced very few splits, try splitting on newlines
        if len(sentences) <= 2 and "\n" in text:
            sentences = [s.strip() for s in text.split("\n") if s.strip()]

        # Filter out empty strings
        return [s.strip() for s in sentences if s.strip()]

    def _sub_split(self, sentences: list[str], max_tokens: int) -> list[str]:
        """Split an oversized group of sentences into smaller chunks.

        Groups sentences until the token limit is reached, then starts
        a new group.
        """
        sub_chunks: list[str] = []
        current_group: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = count_tokens(sentence)
            if current_tokens + sent_tokens > max_tokens and current_group:
                sub_chunks.append(" ".join(current_group))
                current_group = []
                current_tokens = 0
            current_group.append(sentence)
            current_tokens += sent_tokens

        if current_group:
            sub_chunks.append(" ".join(current_group))

        return sub_chunks
