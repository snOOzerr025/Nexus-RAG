"""Enumerations used across the RAG pipeline."""

from enum import StrEnum


class FileType(StrEnum):
    """Supported document file types."""

    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"


class IngestionStatus(StrEnum):
    """Status of a document ingestion operation."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class ChunkingStrategy(StrEnum):
    """Supported chunking strategies."""
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrievalMode(StrEnum):
    """Available retrieval modes."""
    HYBRID = "hybrid"
    DENSE_ONLY = "dense"
    SPARSE_ONLY = "sparse"
    RERANKED = "reranked"


class VerificationStatus(StrEnum):
    """Citation verification result status."""

    VERIFIED = "verified"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class ConfidenceLevel(StrEnum):
    """Confidence level classification for answers."""

    HIGH = "high"       # overall score >= 0.7
    MEDIUM = "medium"   # overall score 0.4 - 0.7
    LOW = "low"         # overall score < 0.4
