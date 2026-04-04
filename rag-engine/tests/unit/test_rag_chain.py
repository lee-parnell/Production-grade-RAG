from rag_engine.models.schemas import RAGQueryResult


def test_rag_query_result_shape() -> None:
    result = RAGQueryResult(answer="ok", sources=["s1"], num_contexts=1)
    assert result.answer == "ok"
    assert result.num_contexts == 1
