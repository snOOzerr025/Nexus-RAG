"""Pydantic request and response models for the FastAPI layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.models.domain import Citation, ConfidenceScore, RetrievedChunk
from src.models.enums import ChunkingStrategy, FileType, IngestionStatus, RetrievalMode

# ---------------------------------------------------------------------------
# Ask Endpoint Models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Request body for the query / ask endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to answer from the document corpus",
        examples=["How do I authenticate with the API?"],
    )
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID,
        description="Which retrieval strategy to use (hybrid, dense, sparse, reranked)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of final chunks to use for generation",
    )
    chunking_strategy_filter: ChunkingStrategy | None = Field(
        default=None,
        description="Filter chunks by chunking strategy (optional)",
    )
    include_chunks: bool = Field(
        default=True,
        description="Whether to include retrieved chunks in response",
    )


class QueryMetadata(BaseModel):
    """Metadata about query execution and retrieval performance."""

    query_id: str
    query_time_ms: int
    retrieval_mode: RetrievalMode
    model_used: str
    total_chunks_searched: int = 0
    dense_results_count: int = 0
    sparse_results_count: int = 0
    fused_results_count: int = 0
    reranked_results_count: int = 0


class AskResponse(BaseModel):
    """Successful response body for the ask endpoint."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceScore
    retrieved_chunks: list[RetrievedChunk] | None = None
    metadata: QueryMetadata


class LowConfidenceResponse(BaseModel):
    """Structured response when confidence is below threshold."""

    partial_answer: str | None = None
    what_was_found: list[str] = Field(default_factory=list)
    what_was_not_found: list[str] = Field(default_factory=list)
    suggested_documents: list[str] = Field(default_factory=list)
    confidence: ConfidenceScore


# ---------------------------------------------------------------------------
# Ingest Endpoint Models
# ---------------------------------------------------------------------------


class IngestConfig(BaseModel):
    """Configuration options for document ingestion."""

    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.RECURSIVE,
        description="Chunking strategy to use",
    )
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=4000,
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks in tokens",
    )
    dedup_threshold: float = Field(
        default=0.95,
        ge=0.8,
        le=1.0,
        description="Cosine similarity threshold for deduplication",
    )


class DocumentIngestionStatus(BaseModel):
    """Status of individual document ingestion."""

    filename: str
    status: IngestionStatus
    chunks_created: int = 0
    duplicates_skipped: int = 0
    error_message: str | None = None


class IngestionError(BaseModel):
    """Details of an ingestion failure."""

    filename: str
    error_type: str
    error_message: str
    stage: str  # "parsing" | "chunking" | "embedding" | "indexing"


class IngestionResult(BaseModel):
    """Result of a document ingestion operation."""

    ingestion_id: str
    documents_processed: int
    documents_failed: int
    total_chunks_created: int
    duplicates_skipped: int
    processing_time_ms: int
    documents: list[DocumentIngestionStatus]


# ---------------------------------------------------------------------------
# Documents Endpoint Models
# ---------------------------------------------------------------------------


class DocumentSummary(BaseModel):
    """Summary of an indexed document."""

    id: str
    filename: str
    file_type: FileType
    file_size_bytes: int
    chunk_count: int
    chunking_strategy: ChunkingStrategy
    status: IngestionStatus
    ingested_at: datetime


class DocumentListResponse(BaseModel):
    """Response body for listing documents."""

    documents: list[DocumentSummary]
    total_count: int
    total_chunks: int


# ---------------------------------------------------------------------------
# Health Endpoint Models
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: str  # "healthy" | "unhealthy"
    latency_ms: int | None = None
    details: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str
    uptime_seconds: int
    components: dict[str, ComponentHealth]


# ---------------------------------------------------------------------------
# Standard Error Response Model
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standardized API error response."""

    error: str
    error_code: str
    message: str
    details: dict | None = None
    request_id: str
