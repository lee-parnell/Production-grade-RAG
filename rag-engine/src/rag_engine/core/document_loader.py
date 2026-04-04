import hashlib
import time
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag_engine.config import settings
from rag_engine.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoadError(Exception):
    """Raised when document loading fails."""
    pass


class DocumentProcessor:
    """
    Loads, validates, and chunks documents for vector indexing.

    Design decisions:
    - RecursiveCharacterTextSplitter over simple splitting because
      it respects paragraph/sentence boundaries → better retrieval quality.
    - Document hash for deduplication → don't re-index identical content.
    - Metadata enrichment at chunk level → every chunk knows its source.
    """

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def validate_file(self, filename: str, file_size_bytes: int) -> None:
        """Validate before processing. Fail fast."""
        ext = Path(filename).suffix.lower()
        if ext not in settings.supported_formats:
            raise DocumentLoadError(
                f"Unsupported format: {ext}. "
                f"Supported: {settings.supported_formats}"
            )

        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            raise DocumentLoadError(
                f"File too large: {file_size_bytes / 1024 / 1024:.1f}MB. "
                f"Max: {settings.max_upload_size_mb}MB"
            )

    def compute_hash(self, content: bytes) -> str:
        """SHA-256 hash for deduplication."""
        return hashlib.sha256(content).hexdigest()

    def load_and_chunk(
        self, file_path: str, filename: str
    ) -> tuple[list[Document], float]:
        """
        Load document and split into chunks with metadata.

        Returns:
            Tuple of (chunks, processing_time_ms)
        """
        start = time.perf_counter()
        ext = Path(filename).suffix.lower()

        loader_cls = self.LOADER_MAP.get(ext)
        if not loader_cls:
            raise DocumentLoadError(f"No loader for format: {ext}")

        try:
            loader = loader_cls(file_path)
            documents = loader.load()
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            raise DocumentLoadError(f"Failed to parse {filename}: {e}")

        if not documents:
            raise DocumentLoadError(f"No content extracted from {filename}")

        # Enrich metadata BEFORE splitting
        for doc in documents:
            doc.metadata["source_filename"] = filename
            doc.metadata["chunk_size"] = self.chunk_size

        chunks = self.splitter.split_documents(documents)

        # Add chunk-level metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Processed {filename}: {len(documents)} pages → "
            f"{len(chunks)} chunks in {elapsed_ms:.1f}ms"
        )

        return chunks, elapsed_ms