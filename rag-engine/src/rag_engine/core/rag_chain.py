import time
from typing import Optional

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_engine.config import settings
from rag_engine.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


class VectorStoreManager:
    """
    Manages Qdrant vector store lifecycle.

    Design decisions:
    - Lazy initialization: don't connect until first use
    - Health check method: used by /health endpoint
    - Explicit error wrapping: never leak Qdrant internals to API layer
    """

    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            try:
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=10,
                )
                logger.info(f"Connected to Qdrant at {settings.qdrant_url}")
            except Exception as e:
                raise VectorStoreError(f"Failed to connect to Qdrant: {e}")
        return self._client

    @property
    def store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=settings.qdrant_collection_name,
                embedding=self._embeddings,
            )
        return self._vector_store

    def health_check(self) -> dict:
        """Check Qdrant connectivity and collection status."""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            doc_count = 0
            if settings.qdrant_collection_name in collection_names:
                info = self.client.get_collection(
                    settings.qdrant_collection_name
                )
                doc_count = info.points_count or 0

            return {
                "connected": True,
                "documents_indexed": doc_count,
                "collection_exists": (
                    settings.qdrant_collection_name in collection_names
                ),
            }
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return {"connected": False, "error": str(e)}

    def index_documents(
        self, chunks: list[Document]
    ) -> tuple[list[str], float]:
        """
        Index document chunks into Qdrant.

        Returns:
            Tuple of (document_ids, indexing_time_ms)
        """
        start = time.perf_counter()

        try:
            ids = self.store.add_documents(chunks)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"Indexed {len(chunks)} chunks in {elapsed_ms:.1f}ms"
            )
            return ids, elapsed_ms
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise VectorStoreError(f"Failed to index documents: {e}")

    def search(
        self,
        query: str,
        top_k: int = settings.retrieval_top_k,
        score_threshold: float = settings.similarity_threshold,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents with scores.

        Returns:
            List of (document, similarity_score) tuples
        """
        start = time.perf_counter()

        try:
            results = self.store.similarity_search_with_score(
                query, k=top_k
            )

            # Filter by threshold
            filtered = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"Search returned {len(filtered)}/{len(results)} results "
                f"(threshold={score_threshold}) in {elapsed_ms:.1f}ms"
            )

            return filtered
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}")


# Singleton
vector_store_manager = VectorStoreManager()