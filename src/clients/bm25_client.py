"""BM25 keyword search index client.

Manages an in-memory BM25 index over document chunks. The index is
serialized to disk via pickle for persistence across restarts.
Stays synchronized with the ChromaDB vector store.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models.domain import ChunkRecord
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Standard English stopwords to filter from BM25 queries
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "shall", "not", "no", "nor", "so", "if", "then", "than", "too",
    "very", "just", "about", "above", "after", "again", "all", "also",
    "am", "any", "as", "because", "before", "between", "both", "during",
    "each", "few", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "into", "its", "itself", "me",
    "more", "most", "my", "myself", "only", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "some", "such",
    "them", "themselves", "there", "these", "they", "those", "through",
    "under", "until", "up", "we", "what", "when", "where", "which",
    "while", "who", "whom", "why", "you", "your", "yours", "yourself",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    Lowercases, splits on whitespace/punctuation, removes stopwords.

    Args:
        text: Text to tokenize.

    Returns:
        List of tokens suitable for BM25.
    """
    # Lowercase and split on non-alphanumeric characters
    tokens = text.lower().split()
    # Clean punctuation from tokens
    tokens = [
        "".join(c for c in token if c.isalnum() or c in "-_.")
        for token in tokens
    ]
    # Remove stopwords and empty tokens
    return [t for t in tokens if t and t not in STOPWORDS]


class BM25Client:
    """In-memory BM25 keyword search index with pickle persistence.

    Maintains a parallel keyword index to the vector store. Captures
    exact term matches that semantic search might miss — critical for
    technical documentation with function names, config keys, and
    error codes.
    """

    def __init__(self, index_path: str = "./data/indexes/bm25_index.pkl") -> None:
        """Initialize BM25 client, loading existing index if available.

        Args:
            index_path: Path to the serialized index file.
        """
        self._index_path = index_path
        self._corpus_tokens: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._chunk_contents: list[str] = []
        self._bm25: BM25Okapi | None = None

        # Try to load existing index
        self._load_from_disk()

    @property
    def count(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._chunk_ids)

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Add chunks to the BM25 index.

        Tokenizes each chunk's content and adds it to the corpus.
        Rebuilds the BM25 model after adding new chunks.

        Args:
            chunks: List of ChunkRecord objects to index.
        """
        if not chunks:
            return

        for chunk in chunks:
            tokens = _tokenize(chunk.content)
            self._corpus_tokens.append(tokens)
            self._chunk_ids.append(chunk.id)
            self._chunk_contents.append(chunk.content)

        # Rebuild BM25 index with updated corpus
        self._rebuild_index()

        logger.info(
            "bm25_chunks_added",
            new_chunks=len(chunks),
            total_chunks=self.count,
        )

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """Query the BM25 index for relevant chunks.

        Args:
            query_text: The search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: chunk_id, content, score.
            Scores are normalized to 0-1 range.
        """
        if self._bm25 is None or self.count == 0:
            logger.warning("bm25_query_empty_index")
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1.0

        results: list[dict] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue  # Skip zero-score results

            results.append({
                "chunk_id": self._chunk_ids[idx],
                "content": self._chunk_contents[idx],
                "score": score / max_score,  # Normalize to 0-1
                "raw_score": score,
            })

        logger.debug(
            "bm25_query",
            query_tokens=len(query_tokens),
            results_count=len(results),
            top_score=results[0]["score"] if results else 0,
        )

        return results

    def delete_by_chunk_ids(self, chunk_ids: set[str]) -> None:
        """Remove specific chunks from the index.

        Args:
            chunk_ids: Set of chunk IDs to remove.
        """
        if not chunk_ids:
            return

        # Filter out the specified chunks
        new_tokens: list[list[str]] = []
        new_ids: list[str] = []
        new_contents: list[str] = []

        for i, cid in enumerate(self._chunk_ids):
            if cid not in chunk_ids:
                new_tokens.append(self._corpus_tokens[i])
                new_ids.append(self._chunk_ids[i])
                new_contents.append(self._chunk_contents[i])

        removed = len(self._chunk_ids) - len(new_ids)
        self._corpus_tokens = new_tokens
        self._chunk_ids = new_ids
        self._chunk_contents = new_contents

        self._rebuild_index()

        logger.info("bm25_chunks_deleted", removed=removed, remaining=self.count)

    def clear(self) -> None:
        """Clear the entire BM25 index."""
        self._corpus_tokens = []
        self._chunk_ids = []
        self._chunk_contents = []
        self._bm25 = None
        self._save_to_disk()
        logger.info("bm25_cleared")

    def save(self) -> None:
        """Persist the index to disk."""
        self._save_to_disk()

    def health_check(self) -> dict:
        """Check BM25 index health status."""
        return {
            "status": "healthy" if self._bm25 is not None or self.count == 0 else "unhealthy",
            "chunk_count": self.count,
            "index_path": self._index_path,
        }

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 model from the current corpus."""
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        """Serialize the index to disk via pickle."""
        path = Path(self._index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "corpus_tokens": self._corpus_tokens,
            "chunk_ids": self._chunk_ids,
            "chunk_contents": self._chunk_contents,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.debug("bm25_saved", path=self._index_path, chunks=self.count)

    def _load_from_disk(self) -> None:
        """Load the index from disk if it exists."""
        path = Path(self._index_path)
        if not path.exists():
            logger.debug("bm25_no_existing_index", path=self._index_path)
            return

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)  # noqa: S301

            self._corpus_tokens = data.get("corpus_tokens", [])
            self._chunk_ids = data.get("chunk_ids", [])
            self._chunk_contents = data.get("chunk_contents", [])

            if self._corpus_tokens:
                self._bm25 = BM25Okapi(self._corpus_tokens)

            logger.info(
                "bm25_loaded_from_disk",
                path=self._index_path,
                chunks=self.count,
            )
        except Exception as e:
            logger.warning(
                "bm25_load_failed",
                path=self._index_path,
                error=str(e),
            )
            # Start with empty index rather than crashing
            self._corpus_tokens = []
            self._chunk_ids = []
            self._chunk_contents = []
            self._bm25 = None
