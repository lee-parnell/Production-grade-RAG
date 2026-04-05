from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
import atexit


class QdrantStorage:
    @staticmethod
    def _default_dim() -> int:
        provider = os.getenv("EMBED_PROVIDER", "openai").lower()
        if provider in ("ollama", "gemini"):
            return 768
        return 3072

    @staticmethod
    def _create_local_client(path: str) -> QdrantClient:
        try:
            return QdrantClient(path=path)
        except RuntimeError as exc:
            # Embedded Qdrant is single-process for a given path.
            if "already accessed by another instance" not in str(exc):
                raise
            fallback_path = f"{path}_pid_{os.getpid()}"
            os.makedirs(fallback_path, exist_ok=True)
            return QdrantClient(path=fallback_path)

    def __init__(self, url=None, collection=None, dim=None):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        collection = collection or os.getenv("QDRANT_COLLECTION", "docs")
        dim = dim or int(os.getenv("EMBED_DIM", str(self._default_dim())))
        local_path = os.getenv("QDRANT_PATH", "qdrant_storage")

        # Prefer remote Qdrant when available, but fall back to embedded/local mode.
        try:
            self.client = QdrantClient(url=url, timeout=30)
            self.client.get_collections()
        except Exception:
            self.client = self._create_local_client(local_path)

        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                with_payload=True,
                limit=top_k,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                with_payload=True,
                limit=top_k,
            )
            results = response.points
        contexts = []
        sources = set()
        records = []

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            score = getattr(r, "score", None)
            if text:
                contexts.append(text)
                sources.add(source)
                records.append(
                    {
                        "text": text,
                        "source": source,
                        "score": float(score) if score is not None else None,
                    }
                )

        return {"contexts": contexts, "sources": list(sources), "records": records}

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


_shared_store: QdrantStorage | None = None


def get_qdrant_storage() -> QdrantStorage:
    global _shared_store
    if _shared_store is None:
        _shared_store = QdrantStorage()
    return _shared_store


@atexit.register
def _close_shared_store() -> None:
    global _shared_store
    if _shared_store is not None:
        _shared_store.close()
        _shared_store = None