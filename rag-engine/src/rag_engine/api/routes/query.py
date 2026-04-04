from fastapi import APIRouter, HTTPException
from rag_engine.models.schemas import QueryRequest, QueryResponse
from rag_engine.core.rag_chain import rag_chain
from rag_engine.core.vector_store import VectorStoreError
from rag_engine.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        response = rag_chain.query(request)
        return response
    except VectorStoreError as e:
        logger.error(f"Vector store error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")