"""Sparse retriever using BM25 keyword index."""

from __future__ import annotations

from typing import Any

from src.clients.bm25_client import BM25Client
from src.clients.chroma_client import ChromaDBClient
from src.models.domain import RetrievedChunk
from src.retrieval.base import Retriever
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SparseRetriever(Retriever):
    """Retrieves chunks by exact keyword matching using BM25.

    Queries the in-memory BM25 index and retrieves the full chunk
    metadata from ChromaDB. Normalizes scores to 0-1.
    """

    def __init__(self, bm25_client: BM25Client, chroma_client: ChromaDBClient) -> None:
        """Initialize with required clients."""
        self.bm25_client = bm25_client
        self.chroma_client = chroma_client

    async def retrieve(
        self, query: str, top_k: int = 10, **kwargs: Any
    ) -> list[RetrievedChunk]:
        """Retrieve most relevant chunks for the given query using BM25.

        Args:
            query: The search query string.
            top_k: Max number of results.
            **kwargs: Ignored for BM25.

        Returns:
            List of RetrievedChunk objects with `sparse_score` populated.
        """
        # 1. Query BM25 index
        bm25_results = self.bm25_client.query(query_text=query, top_k=top_k)

        if not bm25_results:
            logger.info("sparse_retrieval_empty", query=query)
            return []

        # Extract chunk IDs to fetch full metadata from ChromaDB
        chunk_ids = [res["chunk_id"] for res in bm25_results]

        # Create a mapping of chunk_id to sparse_score
        score_map = {res["chunk_id"]: res["score"] for res in bm25_results}

        # 2. Fetch full metadata from ChromaDB
        chroma_data = self.chroma_client.get_by_ids(ids=chunk_ids)

        if not chroma_data or not chroma_data.get("ids"):
            logger.warning("sparse_retrieval_metadata_missing", chunk_ids=chunk_ids)
            return []

        # 3. Map to RetrievedChunk objects
        retrieved_chunks: list[RetrievedChunk] = []

        c_ids = chroma_data["ids"]
        c_docs = chroma_data.get("documents", [])
        c_metas = chroma_data.get("metadatas", [])

        from src.models.enums import ChunkingStrategy

        for i, c_id in enumerate(c_ids):
            meta = c_metas[i] if i < len(c_metas) and c_metas[i] else {}
            doc = c_docs[i] if i < len(c_docs) and c_docs[i] else ""

            strategy_val = meta.get("chunking_strategy", "fixed")
            try:
                strategy = ChunkingStrategy(strategy_val)
            except ValueError:
                strategy = ChunkingStrategy.FIXED

            chunk = RetrievedChunk(
                id=c_id,
                document_id=str(meta.get("document_id", "")),
                content=doc,
                source_file=str(meta.get("source_file", "")),
                section_heading=str(meta.get("section_heading", "")) if meta.get("section_heading") else None,
                page_number=int(meta.get("page_number", -1)) if int(meta.get("page_number", -1)) >= 0 else None,
                chunk_index=int(meta.get("chunk_index", 0)),
                chunking_strategy=strategy,
                char_count=int(meta.get("char_count", 0)),
                token_count=int(meta.get("token_count", 0)),
                sparse_score=score_map.get(c_id, 0.0),
            )
            retrieved_chunks.append(chunk)

        # Sort descending by sparse_score
        retrieved_chunks.sort(key=lambda c: c.sparse_score or 0.0, reverse=True)

        logger.info(
            "sparse_retrieval_complete",
            query=query,
            retrieved_count=len(retrieved_chunks),
        )

        return retrieved_chunks
