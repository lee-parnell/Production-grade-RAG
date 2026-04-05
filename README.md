# Production-Grade RAG Engine

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-f90050?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech/)

A production-focused Retrieval-Augmented Generation (RAG) system for PDF ingestion and grounded Q&A.

This repository includes:
- A root application optimized for fast iteration with Streamlit, FastAPI, and Inngest workflow orchestration.
- A packaged implementation under rag-engine with modular API, core services, and tests.

## Why This Project

The goal is practical document intelligence, not notebook demos:
- Multi-provider LLM and embedding support: OpenAI, Gemini, Claude, Ollama, and local fallback paths.
- Durable workflow mode using Inngest step functions.
- Local-first resilience with graceful fallback when remote dependencies are unavailable.
- Source-aware answers with retrieval-backed context.

## High-Level Architecture

```text
PDF Upload -> Chunking -> Embeddings -> Qdrant -> Top-K Retrieval -> LLM Answer
               |                               |
               +--------- Inngest Step Functions ----------+
```

Core design patterns:
- Deterministic IDs for idempotent re-ingestion.
- Provider abstraction via environment configuration.
- Graceful degradation for vector store and model services.
- Preflight diagnostics in UI for developer-friendly troubleshooting.

## Project Layout

```text
.
├── main.py                 # FastAPI + Inngest functions (root app)
├── streamlit_app.py        # Streamlit UI with local fallback + preflight checks
├── data_loader.py          # PDF loading, chunking, embeddings
├── vector_db.py            # Qdrant integration with local fallback
├── custom_types.py         # Pydantic models
├── qdrant_storage/         # Embedded/local Qdrant data path
├── uploads/                # Uploaded PDFs
├── doc.md                  # Extended technical walkthrough
└── rag-engine/             # Packaged production structure
    ├── src/rag_engine/
    ├── tests/
    ├── docker/
    └── pyproject.toml
```

## Quick Start (Windows PowerShell)

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

### 3. Run Streamlit UI (local mode)

```powershell
python -m streamlit run streamlit_app.py
```

Open: http://127.0.0.1:8501

This mode is enough to ingest and query with local fallback behavior.

## Full Workflow Mode (Streamlit + FastAPI + Inngest)

Run each service in a separate terminal from repository root.

### Terminal A: FastAPI backend

```powershell
$env:INNGEST_DEV='1'
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Terminal B: Inngest Dev Server

```powershell
.\.tools\inngest\inngest.exe dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### Terminal C: Streamlit UI

```powershell
python -m streamlit run streamlit_app.py
```

Expected endpoints:
- Streamlit: http://127.0.0.1:8501
- FastAPI Inngest endpoint: http://127.0.0.1:8000/api/inngest
- Inngest Dev Server: http://127.0.0.1:8288

## Configuration

The root app reads configuration from .env.

Important variables:
- LLM_PROVIDER: openai | gemini | claude | ollama | local
- EMBED_PROVIDER: openai | gemini | ollama | local
- OLLAMA_BASE_URL
- OLLAMA_MODEL
- OLLAMA_EMBED_MODEL
- INNGEST_ENABLED: true | false
- INNGEST_DEV: 1 | 0
- INNGEST_API_BASE: usually http://127.0.0.1:8288/v1
- INNGEST_EVENT_API_BASE: usually http://127.0.0.1:8288
- QDRANT_PATH: local embedded storage path
- EMBED_DIM: should match embedding model output dimensions

Note: Keep secrets such as API keys in .env and never commit them.

## Running The Packaged Service (rag-engine)

If you want the package-based API/UI implementation:

```powershell
cd rag-engine
python -m pip install -e .[dev]
python -m uvicorn rag_engine.api.app:app --reload --port 8000
```

Optional commands (from rag-engine):

```powershell
pytest
ruff check src tests
```

## Docker (rag-engine)

From rag-engine directory:

```powershell
docker compose -f docker/docker-compose.yml up --build
```

This launches Qdrant and the packaged API service.

## Troubleshooting

### Streamlit shows Inngest preflight warning
- Verify FastAPI is running on port 8000.
- Verify Inngest Dev Server is running on port 8288.
- Confirm .env has INNGEST_API_BASE and INNGEST_EVENT_API_BASE set to localhost endpoints.

### Qdrant unavailable
- The app will attempt local embedded fallback using QDRANT_PATH.
- Ensure the process can write to qdrant_storage.

### Provider/API errors
- Check provider keys and model names in .env.
- For Ollama, confirm OLLAMA_BASE_URL and model availability.

## Security Notes

- Do not commit .env files with real secrets.
- Use local model providers for sensitive workloads when data residency is required.
- Restrict uploaded document handling to trusted sources and controlled environments.

## Engineering Highlights

- Multi-provider runtime routing for both embeddings and generation.
- Local fallback mode for degraded infrastructure conditions.
- Deterministic ingestion IDs for idempotency.
- Source-aware answers to improve auditability and trust.

## License

Add your preferred license file (MIT, Apache-2.0, etc.) at repository root.

---

Built for real-world RAG operations by Adil Shamim.
