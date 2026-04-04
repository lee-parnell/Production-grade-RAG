# RAG Engine

Production-grade Retrieval-Augmented Generation API with document
ingestion, vector search via Qdrant, and LLM-powered answer generation.
Built with FastAPI, LangChain, and OpenAI — containerized with Docker.

## Why This Exists

Most RAG tutorials stop at "it works in a notebook." This project
addresses what happens next:
- What happens when a user uploads a 200MB PDF?   → Input validation + streaming chunking
- What happens when Qdrant goes down?              → Health checks + graceful degradation
- What happens when retrieval quality drops?       → Confidence scoring + similarity thresholds
- What does it cost per query?                     → Token tracking + latency monitoring

## Production Numbers

| Metric              | Value                    |
|---------------------|--------------------------|
| Query latency (p50) | TBD after load test      |
| Query latency (p95) | TBD after load test      |
| Cost per query      | TBD after token tracking |
| Max document size   | 50MB (configurable)      |
| Concurrent users    | TBD after load test      |

## Quick Start

    make docker-up          # Qdrant + API in 30 seconds
    curl localhost:8000/health

## Architecture Decisions

| Decision                    | Constraint              | Trade-off                         |
|-----------------------------|-------------------------|-----------------------------------|
| FastAPI over Streamlit      | Need API consumers      | More setup, but production-ready  |
| OpenAI embeddings           | Quality > cost          | $0.0001/1K tokens vs free local   |
| Qdrant over ChromaDB        | Need persistence + scale| Heavier but battle-tested         |
| Single LLM call per query   | Latency budget <2s      | Less reasoning, faster response   |

## API

    POST /api/v1/documents    Upload and index a document
    POST /api/v1/query        Query indexed documents
    GET  /health              System health check