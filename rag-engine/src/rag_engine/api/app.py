import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_engine.config import settings
from rag_engine.api.routes import health, documents, query
from rag_engine.utils.logger import get_logger

logger = get_logger(__name__)

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    logger.info("Starting RAG Engine...")
    # Could pre-warm embeddings model here
    yield
    logger.info("Shutting down RAG Engine...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Engine",
        description="Production-grade Retrieval-Augmented Generation API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])

    return app


app = create_app()