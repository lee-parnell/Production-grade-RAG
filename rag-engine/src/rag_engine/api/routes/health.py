import time
from fastapi import APIRouter
from rag_engine.models.schemas import HealthResponse
from rag_engine.core.vector_store import vector_store_manager

router = APIRouter()
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    qdrant_health = vector_store_manager.health_check()
    return HealthResponse(
        status="healthy" if qdrant_health["connected"] else "degraded",
        qdrant_connected=qdrant_health["connected"],
        documents_indexed=qdrant_health.get("documents_indexed", 0),
        uptime_seconds=round(time.time() - START_TIME, 2),
        version="0.1.0",
    )