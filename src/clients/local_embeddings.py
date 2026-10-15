"""Local embedding client using ChromaDB's built-in ONNX model.

Uses the all-MiniLM-L6-v2 model via ONNX Runtime. Runs entirely
on CPU with no external API calls. Produces 384-dimensional vectors.
"""

from __future__ import annotations

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LocalEmbeddingClient:
    """Local embedding client compatible with the OpenAIClient embedding interface.

    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2 via ONNX)
    to produce 384-dimensional vectors. No API key required.
    """

    def __init__(self) -> None:
        """Initialize the local embedding model."""
        self._ef = DefaultEmbeddingFunction()
        logger.info("local_embeddings_initialized", model="all-MiniLM-L6-v2", dimensions=384)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the local model.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of 384-dimensional embedding vectors.
        """
        if not texts:
            return []

        # DefaultEmbeddingFunction.__call__ returns numpy arrays with np.float32
        # ChromaDB requires plain Python floats, so we explicitly cast each value
        embeddings = self._ef(texts)

        logger.debug("local_embed_complete", count=len(texts))
        return [[float(v) for v in e] for e in embeddings]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Alias for embed_texts."""
        return await self.embed_texts(texts)

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            384-dimensional embedding vector.
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def get_embedding(self, text: str) -> list[float]:
        """Alias for embed_single."""
        return await self.embed_single(text)
