# src/rag_engine/models/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    chunks_created: int
    processing_time_ms: float
    created_at: datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class SourceDocument(BaseModel):
    content: str
    filename: str
    page: int | None = None
    similarity_score: float
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query_time_ms: float
    tokens_used: int | None = None
    model: str
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Proxy confidence based on source similarity scores"
    )


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    documents_indexed: int
    uptime_seconds: float
    version: str