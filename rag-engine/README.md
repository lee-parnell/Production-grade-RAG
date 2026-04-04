# rag-engine

A structured RAG starter project with clear separation of concerns:

- `src/rag_engine/config.py`: application settings and env var loading
- `src/rag_engine/models/schemas.py`: request/response schemas
- `src/rag_engine/core/*`: document loading, embedding, vector store, and RAG pipeline
- `src/rag_engine/api/*`: FastAPI app factory, routes, middleware
- `src/rag_engine/ui/streamlit_app.py`: Streamlit frontend

## Quickstart

```bash
uv sync
uv run uvicorn rag_engine.api.app:app --factory --reload --app-dir src
```

In a second terminal:

```bash
uv run streamlit run src/rag_engine/ui/streamlit_app.py
```

## API

- `GET /health`
- `POST /documents/ingest`
- `POST /query`
