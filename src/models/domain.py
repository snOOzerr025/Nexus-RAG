"""Core domain models for the RAG pipeline.

These models represent the internal data structures used throughout
the ingestion, retrieval, and generation pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.enums import (
    ChunkingStrategy,
    ConfidenceLevel,
    FileType,
    IngestionStatus,
    VerificationStatus,
)


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a new UUID string."""
    return str(uuid4())


# ---------------------------------------------------------------------------
# Document parsing output
# ---------------------------------------------------------------------------


class Section(BaseModel):
    """A section within a parsed document, identified by its heading."""

    heading: str
    level: int = Field(ge=1, le=6, description="Heading level: 1=H1, 2=H2, etc.")
    content: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class ParsedDocument(BaseModel):
    """Output of a document loader — clean text with structured metadata."""

    content: str = Field(description="Clean plaintext content of the document")
    metadata: dict = Field(
        default_factory=dict,
        description="Format-specific metadata (title, author, page count, etc.)",
    )
    sections: list[Section] = Field(
        default_factory=list,
        description="Extracted sections with heading hierarchy",
    )


# ---------------------------------------------------------------------------
# Document record (persisted in registry)
# ---------------------------------------------------------------------------


class DocumentRecord(BaseModel):
    """Represents an ingested document tracked in the registry."""

    id: str = Field(default_factory=_new_id)
    filename: str
    file_type: FileType
    file_size_bytes: int = Field(ge=0)
    raw_storage_path: str
    processed_storage_path: str
    chunk_count: int = 0
    chunking_strategy: ChunkingStrategy
    status: IngestionStatus = IngestionStatus.PENDING
    metadata: dict = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Chunk record (stored in vector DB + BM25)
# ---------------------------------------------------------------------------


class ChunkRecord(BaseModel):
    """Represents a single chunk of text from a document.

    Stored in both ChromaDB (with embedding) and BM25 index.
    Metadata travels with the chunk throughout the pipeline.
    """

    id: str = Field(default_factory=_new_id)
    document_id: str = Field(description="Foreign key to DocumentRecord.id")
    content: str = Field(description="The actual chunk text")
    source_file: str = Field(description="Original filename")
    section_heading: str | None = None
    page_number: int | None = None
    chunk_index: int = Field(ge=0, description="Position within parent document")
    chunking_strategy: ChunkingStrategy
    char_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def chroma_metadata(self) -> dict:
        """Build metadata dict for ChromaDB storage.

        ChromaDB metadata values must be str, int, float, or bool.
        """
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "section_heading": self.section_heading or "",
            "page_number": self.page_number if self.page_number is not None else -1,
            "chunk_index": self.chunk_index,
            "chunking_strategy": self.chunking_strategy.value,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Retrieved Chunk (used in Phase 2+ retrieval/generation)
# ---------------------------------------------------------------------------


class RetrievedChunk(ChunkRecord):
    """A ChunkRecord augmented with retrieval scores.

    Contains scores from various stages of the retrieval pipeline:
    dense search, sparse search, RRF fusion, and LLM reranking.
    """

    dense_score: float | None = Field(default=None, description="Cosine similarity (0 to 1)")
    sparse_score: float | None = Field(default=None, description="Normalized BM25 score (0 to 1)")
    fused_score: float | None = Field(default=None, description="Reciprocal Rank Fusion score")
    rerank_score: float | None = Field(default=None, description="LLM relevance score (0 to 1)")
    final_rank: int | None = Field(default=None, description="Final 1-based rank after all stages")


# ---------------------------------------------------------------------------
# Generation Layer (used in Phase 3+)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A citation reference in the generated answer."""
    index: int = Field(description="Citation number e.g. 1 for [1]")
    chunk_id: str = Field(description="ID of the source chunk")
    source_file: str = Field(description="Name of the source file")
    section: str | None = Field(default=None, description="Section heading")
    relevant_text: str = Field(description="Excerpt from chunk that supports the claim")
    claim_text: str = Field(description="The claim in the answer this citation supports")
    verification_status: VerificationStatus = Field(default=VerificationStatus.UNSUPPORTED)
    verification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_reasoning: str = Field(default="", description="LLM judge reasoning")


class ConfidenceScore(BaseModel):
    """Composite confidence score for an answer."""
    overall: float = Field(ge=0.0, le=1.0)
    retrieval_relevance: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    answer_completeness: float = Field(ge=0.0, le=1.0)
    level: ConfidenceLevel

    @classmethod
    def compute(
        cls,
        retrieval_relevance: float,
        citation_coverage: float,
        answer_completeness: float,
    ) -> ConfidenceScore:
        overall = (
            0.3 * retrieval_relevance
            + 0.4 * citation_coverage
            + 0.3 * answer_completeness
        )
        if overall >= 0.7:
            level = ConfidenceLevel.HIGH
        elif overall >= 0.4:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        return cls(
            overall=round(overall, 3),
            retrieval_relevance=round(retrieval_relevance, 3),
            citation_coverage=round(citation_coverage, 3),
            answer_completeness=round(answer_completeness, 3),
            level=level,
        )


class GenerationResult(BaseModel):
    """Final result of the RAG pipeline."""
    answer: str = Field(description="The generated answer with inline citations")
    citations: list[Citation] = Field(description="List of verified citations")
    confidence: ConfidenceScore = Field(description="Confidence metrics")
