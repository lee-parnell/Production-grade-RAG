# 🚀 Production-Grade RAG Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-f90050?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech/)

A highly resilient, multi-provider Retrieval-Augmented Generation (RAG) architecture. Built for scale, this engine bridges the gap between simple notebook experiments and enterprise-ready document intelligence.

---

## 🌟 Why This Exists

Most RAG tutorials stop at "it works in a notebook." This project is designed to answer real-world engineering challenges:
- **Massive Ingestion:** Input validation and streaming chunking for 200MB+ PDFs.
- **Resilience:** Graceful degradation, local fallback caching, and defensive health checks if the Vector DB goes down.
- **Hardware Agnostic:** Switch seamlessly between OpenAI, Gemini, Claude, or completely local, privacy-first processing with Ollama.
- **Idempotency:** SHA-256 spoof hashing and UUIDv5 chunk deduplication ensure safe, repeatable document ingestion.

## 🏗️ Architecture

The pipeline uses a composable, modular approach emphasizing separation of concerns:
1. **Front Door:** A dual-interface system offering a FastAPI backend for microservices and a Streamlit UI for immediate visual interaction.
2. **Triage & Embedding:** Deterministic chunking (`llama_index`) feeding into your choice of provider embedding models.
3. **Retrieval Store:** Qdrant Vector Database with auto-fallback from remote cloud clusters to local file persistence.
4. **Answer Generation:** Multi-provider LLM routing dynamically maps retrieved context arrays to user queries securely.

## 📊 Production Benchmarks / Constraints

| Metric | Target / Capability |
|--------|---------------------|
| **Embedding Dims** | Auto-scaled: `3072` (OpenAI), `768` (Ollama/Google) |
| **Failover Delay** | `< 1.0s` via graceful degradation handling |
| **Ingest Safety** | Idempotent UUIDv5 deduplication |
| **Chunking Rule** | 1000 characters per chunk, 200 character overlap |
| **Cost Control** | Provider isolation and granular token payload management |

## ⚡ Quick Start

Time to first working query: **~60 seconds**.

### 1. Local Setup
Ensure environment variables are configured. Rename your `.env.example` to `.env` and insert your keys.
```bash
python -m venv .venv
source .venv/bin/activate  # (.venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### 2. The Interactive UI (Streamlit)
Spin up the beautiful frontend interface to interact with documents instantly:
```bash
streamlit run streamlit_app.py
```
*Navigates to `http://localhost:8501`. Comes with built-in health-check preflight displays.*

### 3. The Backend Server (FastAPI)
Run the highly-concurrent REST backend:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```
*Check connection health at `http://localhost:8000/health`.*

## 🛠️ API Surface

The underlying FastAPI engine exposes clean, typed REST endpoints:
- `POST /api/v1/documents` — Stream-upload and securely index a PDF.
- `POST /api/v1/query` — Invoke Qdrant similarity search and LLM completion.
- `GET  /health` — Hardware, workflow, and vector DB connection diagnostics.

---
*Architected for production by Adil Shamim.*