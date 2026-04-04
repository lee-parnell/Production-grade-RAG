from rag_engine.core.vector_store import QdrantVectorStore


class FakeClient:
    def __init__(self) -> None:
        self.created = False
        self.upsert_called = False

    def collection_exists(self, _collection: str) -> bool:
        return self.created

    def create_collection(self, **_kwargs):
        self.created = True

    def upsert(self, _collection: str, points):
        self.upsert_called = True
        assert len(points) == 1


def test_vector_store_creates_collection_and_upserts() -> None:
    fake = FakeClient()
    store = QdrantVectorStore(client=fake, collection="docs", dim=3)
    assert fake.created

    store.upsert(ids=["id-1"], vectors=[[0.1, 0.2, 0.3]], payloads=[{"text": "a"}])
    assert fake.upsert_called
