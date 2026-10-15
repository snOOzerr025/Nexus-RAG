"""Deduplication engine for chunk near-duplicate detection.

Checks new chunks against existing chunks in ChromaDB before insertion.
Prevents the retriever from wasting context window slots on redundant
content when the same information appears in multiple documents.
"""

from __future__ import annotations

from src.clients.chroma_client import ChromaDBClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DeduplicationEngine:
    """Detects and skips near-duplicate chunks before indexing.

    Uses cosine similarity against existing embeddings in ChromaDB.
    A chunk is considered a duplicate if any existing chunk has
    similarity above the configured threshold (default: 0.95).
    """

    def __init__(
        self,
        chroma_client: ChromaDBClient,
        threshold: float = 0.95,
    ) -> None:
        """Initialize deduplication engine.

        Args:
            chroma_client: ChromaDB client for similarity queries.
            threshold: Cosine similarity threshold for duplicate detection.
                Values > this are considered duplicates.
        """
        self._chroma = chroma_client
        self._threshold = threshold

    async def check_duplicate(
        self,
        embedding: list[float],
        chunk_id: str,
    ) -> tuple[bool, str | None]:
        """Check if a chunk is a near-duplicate of an existing chunk.

        Queries ChromaDB for the nearest neighbors and checks if any
        have cosine similarity above the threshold.

        Args:
            embedding: The embedding vector of the new chunk.
            chunk_id: ID of the new chunk (for logging).

        Returns:
            Tuple of (is_duplicate, existing_chunk_id or None).
        """
        # If index is empty, nothing to compare against
        if self._chroma.count == 0:
            return False, None

        # Query for nearest neighbors (small top_k for speed)
        results = self._chroma.query(
            query_embedding=embedding,
            top_k=3,
        )

        if not results["ids"] or not results["ids"][0]:
            return False, None

        # ChromaDB returns distances (cosine distance = 1 - similarity)
        for i, distance in enumerate(results["distances"][0]):
            similarity = 1.0 - distance
            existing_id = results["ids"][0][i]

            if similarity >= self._threshold:
                logger.info(
                    "duplicate_detected",
                    new_chunk_id=chunk_id,
                    existing_chunk_id=existing_id,
                    similarity=round(similarity, 4),
                    threshold=self._threshold,
                )
                return True, existing_id

        return False, None

    async def check_duplicates_batch(
        self,
        embeddings: list[list[float]],
        chunk_ids: list[str],
    ) -> list[bool]:
        """Check a batch of chunks for duplicates.

        Args:
            embeddings: List of embedding vectors.
            chunk_ids: Corresponding chunk IDs.

        Returns:
            List of booleans — True if the chunk is a duplicate.
        """
        results: list[bool] = []
        for embedding, chunk_id in zip(embeddings, chunk_ids, strict=False):
            is_dup, _ = await self.check_duplicate(embedding, chunk_id)
            results.append(is_dup)
        return results
