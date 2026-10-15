"""Chunker factory — selects the appropriate chunking strategy by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ingestion.chunkers.base import Chunker
from src.ingestion.chunkers.fixed_chunker import FixedChunker
from src.ingestion.chunkers.recursive_chunker import RecursiveChunker
from src.ingestion.chunkers.semantic_chunker import SemanticChunker
from src.models.enums import ChunkingStrategy

if TYPE_CHECKING:
    from src.clients.openai_client import OpenAIClient


class ChunkerFactory:
    """Factory that selects the correct Chunker for a strategy name.

    Usage:
        factory = ChunkerFactory(openai_client=client)
        chunker = factory.get_chunker(ChunkingStrategy.RECURSIVE)
        chunks = chunker.chunk(text, doc_id, filename)
    """

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """Initialize with optional OpenAI client for semantic chunking.

        Args:
            openai_client: Required only if semantic chunking is used.
        """
        self._openai_client = openai_client

    def get_chunker(
        self,
        strategy: ChunkingStrategy | str,
        similarity_threshold: float = 0.5,
    ) -> Chunker:
        """Get the appropriate chunker for a strategy.

        Args:
            strategy: Chunking strategy name or enum.
            similarity_threshold: For semantic chunking only.

        Returns:
            A Chunker instance for the requested strategy.

        Raises:
            ValueError: If strategy is unknown or semantic chunking
                requested without an OpenAI client.
        """
        if isinstance(strategy, str):
            try:
                strategy = ChunkingStrategy(strategy)
            except ValueError as err:
                valid = [s.value for s in ChunkingStrategy]
                raise ValueError(
                    f"Unknown chunking strategy: '{strategy}'. "
                    f"Valid strategies: {valid}"
                ) from err

        if strategy == ChunkingStrategy.FIXED:
            return FixedChunker()

        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker()

        if strategy == ChunkingStrategy.SEMANTIC:
            if self._openai_client is None:
                raise ValueError(
                    "Semantic chunking requires an OpenAI client for embeddings. "
                    "Pass openai_client to ChunkerFactory."
                )
            return SemanticChunker(
                openai_client=self._openai_client,
                similarity_threshold=similarity_threshold,
            )

        raise ValueError(f"Unhandled chunking strategy: {strategy}")
