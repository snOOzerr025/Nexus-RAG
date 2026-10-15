"""ChromaDB client wrapper.

Manages a persistent ChromaDB collection for vector storage and retrieval.
Handles collection creation, chunk insertion with metadata, and
similarity queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.models.domain import ChunkRecord
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChromaDBClient:
    """Wrapper around ChromaDB for vector storage and similarity search.

    Uses a persistent file-based ChromaDB instance. The collection
    stores chunk text, embedding vectors, and structured metadata.
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "document_chunks",
    ) -> None:
        """Initialize ChromaDB client with persistent storage.

        Args:
            persist_dir: Directory for ChromaDB persistent storage.
            collection_name: Name of the collection to use.
        """
        self._persist_dir = persist_dir
        self._collection_name = collection_name

        # Ensure persist directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize persistent client
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )

        # Get or create the collection with cosine distance
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "chromadb_initialized",
            persist_dir=persist_dir,
            collection=collection_name,
            existing_count=self._collection.count(),
        )

    @property
    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()

    def add_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with embeddings to the collection.

        Args:
            chunks: List of ChunkRecord objects.
            embeddings: Corresponding embedding vectors (same order).

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.chroma_metadata for chunk in chunks]

        # ChromaDB handles batching internally, but we batch for safety
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self._collection.add(
                ids=ids[i:end],
                embeddings=cast(Any, embeddings[i:end]),
                documents=documents[i:end],
                metadatas=cast(Any, metadatas[i:end]),
            )

        logger.info(
            "chromadb_chunks_added",
            count=len(chunks),
            total=self._collection.count(),
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> dict:
        """Query the collection by embedding similarity.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            where: Optional metadata filter dict.

        Returns:
            ChromaDB query results dict with keys:
            ids, distances, documents, metadatas.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()) if self._collection.count() > 0 else top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        logger.debug(
            "chromadb_query",
            top_k=top_k,
            results_count=len(results["ids"][0]) if results["ids"] else 0,
        )

        return cast(dict, results)

    def get_by_ids(self, ids: list[str]) -> dict:
        """Retrieve chunks by their IDs.

        Args:
            ids: List of chunk UUIDs.

        Returns:
            ChromaDB get results with documents and metadata.
        """
        return cast(
            dict,
            self._collection.get(
                ids=ids,
                include=["documents", "metadatas"],
            ),
        )

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The parent document UUID.
        """
        self._collection.delete(
            where={"document_id": document_id},
        )
        logger.info(
            "chromadb_document_deleted",
            document_id=document_id,
        )

    def clear(self) -> None:
        """Delete all chunks from the collection."""
        # Recreate the collection (fastest way to clear)
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("chromadb_cleared")

    def health_check(self) -> dict:
        """Check ChromaDB health status.

        Returns:
            Dict with status, count, and collection info.
        """
        try:
            count = self._collection.count()
            return {
                "status": "healthy",
                "chunk_count": count,
                "collection": self._collection_name,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
