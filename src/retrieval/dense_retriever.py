"""Dense retriever using ChromaDB and OpenAI embeddings."""

from __future__ import annotations

from typing import Any

from src.clients.chroma_client import ChromaDBClient
from src.models.domain import RetrievedChunk
from src.retrieval.base import Retriever
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever(Retriever):
    """Retrieves chunks by semantic similarity using embeddings.

    Uses OpenAI to embed the query and ChromaDB for vector search.
    Converts cosine distance into a similarity score (0-1).
    """

    def __init__(self, chroma_client: ChromaDBClient, openai_client: Any) -> None:
        """Initialize with required clients."""
        self.chroma_client = chroma_client
        self.openai_client = openai_client

    async def retrieve(
        self, query: str, top_k: int = 10, **kwargs: Any
    ) -> list[RetrievedChunk]:
        """Retrieve most similar chunks for the given query.

        Args:
            query: The search query string.
            top_k: Max number of results.
            **kwargs: Can include 'filters' dict for ChromaDB where clause.

        Returns:
            List of RetrievedChunk objects with `dense_score` populated.
        """
        filters = kwargs.get("filters")

        # 1. Embed the query
        query_embedding = await self.openai_client.embed_single(query)

        # 2. Query ChromaDB
        results = self.chroma_client.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=filters,
        )

        # Chroma query returns lists of lists (since we passed 1 query embedding)
        # Results structure: {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        retrieved_chunks: list[RetrievedChunk] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            logger.info("dense_retrieval_empty", query=query)
            return retrieved_chunks

        # Extract the single query's results
        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # 3. Map to RetrievedChunk objects
        for i in range(len(ids)):
            # Convert cosine distance to cosine similarity (1 - distance)
            distance = distances[i] if i < len(distances) else 0.0
            similarity = max(0.0, 1.0 - distance)

            meta = metadatas[i] or {}

            # Map chunking_strategy string back to enum gracefully
            from src.models.enums import ChunkingStrategy
            strategy_val = meta.get("chunking_strategy", "fixed")
            try:
                strategy = ChunkingStrategy(strategy_val)
            except ValueError:
                strategy = ChunkingStrategy.FIXED

            chunk = RetrievedChunk(
                id=ids[i],
                document_id=str(meta.get("document_id", "")),
                content=documents[i] if i < len(documents) else "",
                source_file=str(meta.get("source_file", "")),
                section_heading=str(meta.get("section_heading", "")) if meta.get("section_heading") else None,
                page_number=int(meta.get("page_number", -1)) if int(meta.get("page_number", -1)) >= 0 else None,
                chunk_index=int(meta.get("chunk_index", 0)),
                chunking_strategy=strategy,
                char_count=int(meta.get("char_count", 0)),
                token_count=int(meta.get("token_count", 0)),
                dense_score=similarity,
            )
            retrieved_chunks.append(chunk)

        # Sort descending by score just to be sure
        retrieved_chunks.sort(key=lambda c: c.dense_score or 0.0, reverse=True)

        logger.info(
            "dense_retrieval_complete",
            query=query,
            retrieved_count=len(retrieved_chunks),
        )

        return retrieved_chunks
