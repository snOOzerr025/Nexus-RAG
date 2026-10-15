"""Base class for all retrievers."""

from abc import ABC, abstractmethod
from typing import Any

from src.models.domain import RetrievedChunk


class Retriever(ABC):
    """Abstract base class for all retrieval methods (dense, sparse, etc.)."""

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 10, **kwargs: Any
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a given query.

        Args:
            query: The search query string.
            top_k: Number of results to return.
            **kwargs: Additional retrieval arguments (e.g., metadata filters).

        Returns:
            List of RetrievedChunk objects, ordered by relevance.
        """
        ...
