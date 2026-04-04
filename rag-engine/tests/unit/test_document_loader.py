from rag_engine.core.document_loader import chunk_texts


def test_chunk_texts_returns_chunks_for_non_empty_input() -> None:
    chunks = chunk_texts(["Hello world. " * 200])
    assert chunks
    assert isinstance(chunks[0], str)
