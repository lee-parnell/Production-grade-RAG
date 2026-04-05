# RAG Engine: Building a Production-Grade Document Intelligence System

**By Adil Shamim**

*A Complete End-to-End Production Tutorial*

---

**RAG Engine Product Documentation**

**April 2026**

---

## Contents

1. [Imagine This: You're Building a Document Intelligence Hospital!](#1-imagine-this-youre-building-a-document-intelligence-hospital)
2. [Quick Start — Get Running in 3 Steps](#quick-start--get-running-in-3-steps)
3. [Step 1: The Front Door — `main.py`](#2-step-1-the-front-door--mainpy)
   - 2.1 What is `main.py`?
   - 2.2 The Complete Code
   - 2.3 Line-by-Line Breakdown
3. [Step 2: The Triage Nurse — `data_loader.py`](#3-step-2-the-triage-nurse--dataloaderpy)
   - 3.1 What is `data_loader.py`?
   - 3.2 The Complete Code
   - 3.3 Line-by-Line Breakdown
4. [Step 3: The Medical Records Room — `vector_db.py`](#4-step-3-the-medical-records-room--vectordbpy)
   - 4.1 What is `vector_db.py`?
   - 4.2 The Complete Code
   - 4.3 Line-by-Line Breakdown
5. [Step 4: The Patient Chart — `custom_types.py`](#5-step-4-the-patient-chart--customtypespy)
   - 5.1 What is `custom_types.py`?
   - 5.2 The Complete Code
   - 5.3 Line-by-Line Breakdown
6. [Step 5: The Waiting Room — `streamlit_app.py`](#6-step-5-the-waiting-room--streamlitapppy)
   - 6.1 What is `streamlit_app.py`?
   - 6.2 The Complete Code
   - 6.3 Line-by-Line Breakdown
7. [Step 6: The Hospital Blueprint — `rag-engine/` Package](#7-step-6-the-hospital-blueprint--rag-engine-package)
   - 7.1 Configuration — `config.py`
   - 7.2 API Factory — `api/app.py`
   - 7.3 API Routes — Health, Documents, Query
   - 7.4 Middleware — Error Handling & Logging
   - 7.5 Core Engine — Document Loader, Embeddings, Vector Store, RAG Chain
   - 7.6 Data Models — `models/schemas.py`
   - 7.7 Streamlit UI — `ui/streamlit_app.py`
   - 7.8 Utilities — Logger
8. [Step 7: The Infrastructure — Docker & CI/CD](#8-step-7-the-infrastructure--docker--cicd)
   - 8.1 Dockerfile
   - 8.2 Docker Compose
   - 8.3 GitHub Actions CI
   - 8.4 Makefile
9. [Step 8: The Hospital Policy Manual — Environment & Config](#9-step-8-the-hospital-policy-manual--environment--config)
10. [Architecture Overview](#10-architecture-overview)
11. [Production Patterns Used](#11-production-patterns-used)
12. [How Everything Connects](#12-how-everything-connects)

---

## Quick Start — Get Running in 3 Steps

**Time to first working query: ~2 minutes**

### Step 1: Start Streamlit (Frontend)
```powershell
d:/Production-grade-RAG/.venv/Scripts/python.exe -m streamlit run streamlit_app.py
# Opens http://127.0.0.1:8501 in your browser
```

### Step 2: Start FastAPI Backend (Optional — for Inngest workflows)
```powershell
$env:INNGEST_DEV='1'
d:/Production-grade-RAG/.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
# Runs on http://127.0.0.1:8000
```

### Step 3: Start Inngest Dev Server (Optional — for Inngest workflows)
```powershell
./.tools/inngest/inngest.exe dev -u http://127.0.0.1:8000/api/inngest --no-discovery
# Runs on http://127.0.0.1:8288
```

**What you get:**
- ✅ Step 1 alone = Local fallback mode, fully functional
- ✅ Steps 1-3 = Full Inngest workflow orchestration with Dev Server UI at http://127.0.0.1:8288

**Troubleshooting:** Open Streamlit and check the yellow warning box at the top. It shows which services are missing and exact startup commands.

---

## 1. Imagine This: You're Building a Document Intelligence Hospital!

Think of this entire RAG (Retrieval-Augmented Generation) application like a **hospital for documents**.

- **Patients** = Your PDF documents that need to be "treated" (processed, understood, queried)
- **Front Door (`main.py`)** = The main entrance where patients arrive and get routed
- **Triage Nurse (`data_loader.py`)** = Examines each patient, breaks them into manageable pieces (chunks), and creates an X-ray (embedding) of each piece
- **Medical Records Room (`vector_db.py`)** = Stores all X-rays (vectors) in an organized filing system (Qdrant) so doctors can find similar cases instantly
- **Patient Charts (`custom_types.py`)** = Standardized forms that every department uses — no confusion, no miscommunication
- **Waiting Room (`streamlit_app.py`)** = Where visitors (users) come in, drop off documents, and ask questions
- **Hospital Blueprint (`rag-engine/`)** = The production-grade architectural plans — proper departments, hallways, security, and monitoring
- **Infrastructure (Docker & CI/CD)** = The building itself — plumbing, electricity, fire safety systems

**Why does this matter?**

Most RAG tutorials stop at "it works in a notebook." This project answers what happens next:
- What happens when a user uploads a 200MB PDF? → **Input validation + streaming chunking**
- What happens when Qdrant goes down? → **Health checks + graceful degradation + local fallback**
- What happens when you want to switch from OpenAI to Gemini to Ollama? → **Multi-provider architecture**
- What does it cost per query? → **Token tracking + latency monitoring**

Let's walk through every single file, line by line.

---

## 2. Step 1: The Front Door — `main.py`

### 2.1 What is `main.py`?

This is the **FastAPI backend server** — the main entrance to your hospital. It does three critical things:

1. **Registers Inngest workflow functions** for PDF ingestion and querying
2. **Routes LLM calls** to the correct provider (OpenAI, Gemini, Claude, Ollama, or Local)
3. **Serves the FastAPI application** that Inngest connects to

Think of it as the hospital's **reception desk** — it doesn't do the surgery itself, but it makes sure every patient gets to the right doctor.

### 2.2 The Complete Code

```python
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
import requests
import importlib
from openai import OpenAI
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import get_qdrant_storage
from custom_types import RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()
```

### 2.3 Line-by-Line Breakdown

**The Imports (Lines 1–14):**

| Import | Purpose | Hospital Analogy |
|--------|---------|-----------------|
| `FastAPI` | Web framework for the API server | The hospital building itself |
| `inngest` | Workflow orchestration (step functions) | The appointment scheduling system |
| `load_dotenv` | Loads environment variables from `.env` | Reading the hospital's policy manual |
| `uuid` | Generates unique IDs for document chunks | Patient ID wristbands |
| `OpenAI` | OpenAI API client | One of the specialist doctors |
| `load_and_chunk_pdf, embed_texts` | PDF processing and embedding functions | The triage nurse's toolkit |
| `get_qdrant_storage` | Vector database connection | Access to the medical records room |
| `RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc` | Type-safe data containers | Standardized patient chart forms |

**Multi-Provider LLM Configuration (Lines 20–34):**

```python
def _get_llm_config() -> tuple[str, str, str, str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return provider, openai_model, gemini_model, claude_model, ollama_model, ollama_base_url
```

**Why this matters:** In production, you never hardcode which AI model you use. This function reads from environment variables so you can switch providers without changing a single line of code. Think of it as having **multiple specialist doctors on call** — you pick the one available.

**The Answer Generator (Lines 48–129):**

```python
def generate_answer(user_content: str) -> str:
    system_prompt = "You answer questions using only the provided context."
    llm_provider, openai_model, gemini_model, claude_model, ollama_model, ollama_base_url = _get_llm_config()
```

This is the **brain of the operation**. It takes a user's question (with retrieved context) and routes it to the correct LLM provider:

| Provider | API Style | Use Case |
|----------|-----------|----------|
| `openai` | OpenAI SDK (`client.chat.completions.create`) | Production default, highest quality |
| `gemini` | Google GenerativeAI SDK (`model.generate_content`) | Google ecosystem, cost-effective |
| `claude` | Anthropic SDK (`client.messages.create`) | Best for long-context reasoning |
| `ollama` | REST API (`requests.post`) | Self-hosted, privacy-first, no API costs |
| `local` | Pattern matching on context | Zero-dependency fallback for testing |

**Production Pattern — Lazy Imports:**
```python
if llm_provider == "gemini":
    try:
        genai = importlib.import_module("google.generativeai")
    except ImportError as exc:
        raise RuntimeError("Gemini support requires google-generativeai...") from exc
```
We use `importlib.import_module()` instead of `import` at the top. Why? Because if a user only uses OpenAI, they shouldn't need to install the Gemini SDK. This is **dependency isolation** — a production essential.

**Inngest Workflow Functions (Lines 131–196):**

```python
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)
```

The Inngest client acts as the **hospital's workflow management system**. It breaks complex operations into durable, retryable steps.

**PDF Ingestion Function (Lines 138–168):**

```python
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
    rate_limit=inngest.RateLimit(limit=1, period=datetime.timedelta(hours=4), key="event.data.source_id"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
```

**Production patterns here:**

| Pattern | What It Does | Why It Matters |
|---------|-------------|----------------|
| `throttle` | Max 2 ingestions per minute | Prevents overloading the embedding API |
| `rate_limit` | Same document only once per 4 hours | Prevents duplicate processing |
| Step functions (`ctx.step.run`) | Each step is independently retryable | If embedding fails, chunking doesn't re-run |
| `uuid.uuid5` | Deterministic IDs from source + index | Same document always gets same chunk IDs (idempotent) |

**Query Function (Lines 171–196):**

```python
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
```

This function orchestrates the **full RAG pipeline**:
1. **embed-and-search** — Convert question to vector, find similar chunks
2. **llm-answer** — Send retrieved context + question to the LLM

**FastAPI App Registration (Lines 198–200):**

```python
app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
```

Two lines. That's it. The entire API server is created and Inngest functions are registered. FastAPI handles HTTP routing, Inngest handles workflow orchestration.

---

## 3. Step 2: The Triage Nurse — `data_loader.py`

### 3.1 What is `data_loader.py`?

This is the **document processing pipeline**. Like a triage nurse, it:
1. **Reads** the PDF document
2. **Splits** it into digestible chunks (1000 characters each, with 200-char overlap)
3. **Creates embeddings** (numerical representations) of each chunk

### 3.2 The Complete Code

```python
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
import requests
import hashlib
import importlib

load_dotenv()

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai").lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
```

### 3.3 Line-by-Line Breakdown

**Multi-Provider Embedding Dimensions (Lines 26–36):**

```python
def _default_embed_dim(provider: str) -> int:
    if provider == "gemini":
        return 768
    if provider == "ollama":
        return 768
    if provider == "local":
        return 768
    return 3072  # OpenAI text-embedding-3-large
```

**Why different dimensions?** Each embedding model produces vectors of different sizes. OpenAI's `text-embedding-3-large` produces 3072-dimensional vectors (very detailed), while Gemini and Ollama produce 768-dimensional vectors. The vector database needs to know this size upfront.

**The Deterministic Fallback Embedding (Lines 55–68):**

```python
def _hash_embedding(text: str, dim: int) -> list[float]:
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for i in range(0, len(block), 4):
            if len(out) >= dim:
                break
            value = int.from_bytes(block[i:i + 4], "big")
            out.append((value / 4294967295.0) * 2.0 - 1.0)
        counter += 1
    return out
```

**This is a production gem.** When no embedding API is available (offline, testing, CI), this function creates a **deterministic fake embedding** from the text's SHA-256 hash. The same text always produces the same "embedding." It won't give meaningful semantic similarity, but it lets the entire pipeline run without an API key.

**PDF Loading and Chunking (Lines 70–78):**

```python
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks
```

| Parameter | Value | Why |
|-----------|-------|-----|
| `chunk_size=1000` | 1000 characters per chunk | Small enough for LLM context windows, large enough for meaningful content |
| `chunk_overlap=200` | 200-character overlap between chunks | Prevents information loss at chunk boundaries |

**The Embedding Function (Lines 81–134):**

```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    provider, model, ollama_base, ollama_model, dim = _get_embed_config()
```

This mirrors the LLM provider pattern from `main.py` — four providers, one interface:

| Provider | Model | Dimensions | Cost |
|----------|-------|------------|------|
| OpenAI | `text-embedding-3-large` | 3072 | ~$0.00013/1K tokens |
| Gemini | `text-embedding-004` | 768 | Free tier available |
| Ollama | `nomic-embed-text` | 768 | Free (self-hosted) |
| Local | Hash-based fallback | 768 | Free (no API needed) |

**Production Pattern — Graceful Degradation for Ollama:**
```python
if provider == "ollama":
    for text in texts:
        try:
            response = requests.post(...)
            vectors.append(response.json()["embedding"])
        except Exception:
            vectors.append(_hash_embedding(text, dim))  # Fallback!
```

If Ollama fails for one text, we don't crash — we fall back to the hash embedding for that specific text. The pipeline continues.

---

## 4. Step 3: The Medical Records Room — `vector_db.py`

### 4.1 What is `vector_db.py`?

This is the **vector database layer** — Qdrant. Think of it as the hospital's medical records room where every patient's X-ray (embedding vector) is stored and can be instantly retrieved by similarity.

### 4.2 The Complete Code

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
import atexit
```

### 4.3 Line-by-Line Breakdown

**The QdrantStorage Class (Lines 7–72):**

```python
class QdrantStorage:
    def __init__(self, url=None, collection=None, dim=None):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        collection = collection or os.getenv("QDRANT_COLLECTION", "docs")
        dim = dim or int(os.getenv("EMBED_DIM", str(self._default_dim())))
        local_path = os.getenv("QDRANT_PATH", "qdrant_storage")

        # Prefer remote Qdrant when available, but fall back to embedded/local mode.
        try:
            self.client = QdrantClient(url=url, timeout=30)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(path=local_path)
```

**Production Pattern — Graceful Fallback:**

This is one of the most important production patterns in the project. The connection strategy is:

1. **Try remote Qdrant first** (production server at `QDRANT_URL`)
2. **If it fails, fall back to local embedded mode** (file-based storage at `qdrant_storage/`)
3. **Auto-create the collection** if it doesn't exist

This means the app **never crashes** because of a missing database — it degrades gracefully.

**Upsert Method (Lines 35–37):**

```python
def upsert(self, ids, vectors, payloads):
    points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
    self.client.upsert(self.collection, points=points)
```

`upsert` = **update or insert**. If a chunk with the same ID already exists, it's replaced. This makes re-ingestion safe and idempotent.

**Search Method (Lines 39–66):**

```python
def search(self, query_vector, top_k: int = 5):
    if hasattr(self.client, "search"):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
        )
    else:
        response = self.client.query_points(...)
        results = response.points
```

**Production Pattern — API Version Compatibility:** Different versions of `qdrant_client` have different APIs. By checking `hasattr(self.client, "search")`, we support both old (`search`) and new (`query_points`) APIs without breaking.

**Singleton Pattern (Lines 75–90):**

```python
_shared_store: QdrantStorage | None = None

def get_qdrant_storage() -> QdrantStorage:
    global _shared_store
    if _shared_store is None:
        _shared_store = QdrantStorage()
    return _shared_store

@atexit.register
def _close_shared_store() -> None:
    global _shared_store
    if _shared_store is not None:
        _shared_store.close()
        _shared_store = None
```

**Why a Singleton?** Creating database connections is expensive. We create one connection and share it across all requests. The `@atexit.register` decorator ensures the connection is **cleanly closed** when the process exits — no leaked connections, no corrupted data.

---

## 5. Step 4: The Patient Chart — `custom_types.py`

### 5.1 What is `custom_types.py`?

These are **Pydantic models** — strongly-typed data containers that act as contracts between every component. Like standardized patient charts in a hospital, they ensure every department speaks the same language.

### 5.2 The Complete Code

```python
import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAQQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
```

### 5.3 Line-by-Line Breakdown

| Model | Fields | Used By | Purpose |
|-------|--------|---------|---------|
| `RAGChunkAndSrc` | `chunks`, `source_id` | `rag_ingest_pdf` step 1 → step 2 | Carries chunks between Inngest steps |
| `RAGUpsertResult` | `ingested` | `rag_ingest_pdf` return value | Reports how many chunks were stored |
| `RAGSearchResult` | `contexts`, `sources` | `rag_query_pdf_ai` step 1 → step 2 | Carries search results to LLM step |
| `RAQQueryResult` | `answer`, `sources`, `num_contexts` | Query response | Final answer with provenance |

**Why Pydantic?** Three reasons:
1. **Validation** — If someone passes `ingested="five"` instead of `ingested=5`, it fails immediately with a clear error
2. **Serialization** — Inngest needs to serialize/deserialize data between steps. Pydantic handles this automatically via `PydanticSerializer()`
3. **Documentation** — These models serve as self-documenting API contracts

---

## 6. Step 5: The Waiting Room — `streamlit_app.py`

### 6.1 What is `streamlit_app.py`?

This is the **user-facing frontend** — a Streamlit web application where users can:
1. **Configure** their LLM and embedding providers via a sidebar
2. **Upload** PDF documents for ingestion
3. **Ask questions** about their uploaded documents

### 6.2 Key Sections

**Inngest Health Checks — Preflight Diagnostics (Lines 319–360):**

Added in April 2026 to improve developer experience:

```python
def _check_endpoint(url: str, timeout: float = 1.5) -> tuple[bool, str]:
    try:
        response = requests.get(url, timeout=timeout)
        return True, str(response.status_code)
    except requests.RequestException as exc:
        return False, str(exc)

def _render_inngest_preflight() -> None:
    """Displays Inngest service status and actionable setup guidance."""
    if not _inngest_enabled():
        st.info("Inngest mode is disabled. Local fallback mode is active.")
        return
    
    api_ok, api_info = _check_endpoint("http://127.0.0.1:8000/api/inngest")
    dev_ok, dev_info = _check_endpoint("http://127.0.0.1:8288")
    
    if api_ok and dev_ok:
        st.success("Inngest preflight passed: FastAPI endpoint and Dev Server are reachable.")
        return
    
    st.warning("Inngest preflight failed. Workflow mode needs two running services.")
    st.caption(f"FastAPI /api/inngest: {'OK' if api_ok else 'DOWN'} ({api_info})")
    st.caption(f"Inngest Dev Server : {'OK' if dev_ok else 'DOWN'} ({dev_info})")
    st.code(
        "# 1) Start FastAPI (same venv as this app)\n"
        "$env:INNGEST_DEV='1'\n"
        "d:/ProductionGradeRAGPythonApp-main/.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000\n\n"
        "# 2) Start Inngest Dev Server (npx or Docker fallback)\n"
        "npx --ignore-scripts=false inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery\n\n"
        "# 2b) Docker fallback if npx is unavailable\n"
        "docker run -p 8288:8288 inngest/inngest inngest dev -u http://host.docker.internal:8000/api/inngest --no-discovery",
        language="bash",
    )
```

**Why Preflight Checks?** In production, services fail silently. These checks turn a cryptic "connection refused" error into:
1. ✅ Visual indicator (green/yellow/red)
2. 📊 Actual status codes (200, timeout, etc.)
3. 🛠️ Exact commands to fix it

This is rendered automatically on page load (line 381), so users see the issue **before** trying to upload a PDF.

**Production Pattern — Environment Variable Persistence (Line 193):**

```python
if inngest_enabled:
    _save_env("INNGEST_ENABLED", "true")
    _save_env("INNGEST_DEV", "1")  # NEW: Persist INNGEST_DEV flag
else:
    _save_env("INNGEST_ENABLED", "false")
    _save_env("INNGEST_DEV", "0")
```

When a user toggles "Use Inngest workflow mode" in the sidebar, we now persist both `INNGEST_ENABLED` and `INNGEST_DEV` to `.env`. This ensures the backend automatically runs in dev mode when restarted.

**Fixed Default Toggle Behavior (Line 315):**

```python
def _inngest_enabled() -> bool:
    # Changed default from "true" to "false" for safety
    return os.getenv("INNGEST_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
```

Previously defaulted to `"true"`, which caused connection errors if Inngest Dev Server wasn't running. Now defaults to `"false"` (local fallback mode), making the app **safe to use offline**.

**Model Settings Sidebar (Lines 106–204):**

The sidebar dynamically renders model selection based on the chosen provider:

```python
def _render_model_settings() -> None:
    llm_options = ["openai", "gemini", "claude", "ollama", "local"]
    embed_options = ["openai", "gemini", "ollama", "local"]
```

**Production Feature — Dynamic Ollama Model Discovery:**

```python
def _fetch_ollama_models() -> list[str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "https://api.ollama.com")
    try:
        response = requests.get(f"{base_url}/api/tags", headers=_ollama_headers(), timeout=10)
        response.raise_for_status()
        payload = response.json()
        models = [m.get("name", "") for m in payload.get("models", [])]
        if models:
            return sorted(set(models))
    except Exception:
        pass
    return ["gemma3:4b", "gpt-oss:20b", "llama3.1"]  # Sensible defaults
```

This **auto-discovers** available Ollama models. If the Ollama server is unreachable, it falls back to hardcoded defaults. The user always sees something useful.

**Dual-Mode Execution (Lines 267–296):**

```python
def _run_ingest_locally(pdf_path: Path) -> int:
    chunks = load_and_chunk_pdf(str(pdf_path.resolve()))
    vectors = embed_texts(chunks)
    source_id = pdf_path.name
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    get_qdrant_storage().upsert(ids, vectors, payloads)
    return len(chunks)
```

The app supports **two execution modes**:

| Mode | How It Works | When To Use |
|------|-------------|-------------|
| **Inngest Mode** | Sends events to Inngest Dev Server → FastAPI processes them | Production, with workflow orchestration |
| **Local Mode** | Calls `data_loader` and `vector_db` directly | Development, or when Inngest is unavailable |

**Production Pattern — Cascading Fallback:**

```python
try:
    asyncio.run(send_rag_ingest_event(path))  # Try Inngest first
    st.success(f"Triggered ingestion for: {path.name}")
except Exception as exc:
    if _is_send_events_error(exc):
        st.warning(_send_error_message(exc))
        try:
            ingested = _run_ingest_locally(path)  # Fall back to local
            st.success(f"Ingested locally: {ingested} chunks")
        except Exception as local_exc:
            st.error(f"Local ingestion failed: {local_exc}")
            st.info(_local_fallback_help())  # Help the user fix it
```

Three levels of fallback: Inngest → Local → Helpful error message. The app **never just crashes**.

---

## 7. Step 6: The Hospital Blueprint — `rag-engine/` Package

The `rag-engine/` directory is the **production-grade restructured version** of the application, organized with proper separation of concerns.

### 7.1 Configuration — `config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- LLM ---
    openai_api_key: str = Field(..., description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_dimensions: int = 1536

    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = "documents"

    # --- Document Processing ---
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0)
    max_upload_size_mb: int = Field(default=50, ge=1, le=200)

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @validator("chunk_overlap")
    def overlap_less_than_chunk(cls, v, values):
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
```

**Production Patterns:**
- **`BaseSettings`** — Automatically reads from environment variables and `.env` files
- **`Field` with constraints** — `ge=0.0, le=2.0` ensures temperature is always valid
- **Cross-field validation** — `chunk_overlap` must be less than `chunk_size`
- **Singleton** — `settings = Settings()` is created once and imported everywhere

### 7.2 API Factory — `api/app.py`

```python
def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Engine",
        description="Production-grade Retrieval-Augmented Generation API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, ...)
    app.include_router(health.router, tags=["Health"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    return app
```

**Why a Factory Function?** `create_app()` returns a new app instance each time. This enables:
- **Testing** — Each test gets a fresh app
- **Multiple workers** — Each Uvicorn worker creates its own app
- **Lifespan management** — Startup/shutdown hooks via `asynccontextmanager`

### 7.3 API Routes

**Health Endpoint (`routes/health.py`):**
```python
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
```

Returns `"healthy"` or `"degraded"` — never hides problems. Used by Docker's `HEALTHCHECK` and load balancers.

**Document Ingest Endpoint (`routes/documents.py`):**
```python
@router.post("/ingest", response_model=RAGUpsertResult)
def ingest_document(payload: IngestDocumentRequest) -> RAGUpsertResult:
    chain = RAGChain()
    try:
        return chain.ingest_pdf(pdf_path=payload.pdf_path, source_id=payload.source_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

**Production Pattern — Proper HTTP Status Codes:**
- `404` for missing files (not 500!)
- `400` for invalid input (not 500!)
- Only unexpected errors become 500

**Query Endpoint (`routes/query.py`):**
```python
@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        response = rag_chain.query(request)
        return response
    except VectorStoreError as e:
        raise HTTPException(status_code=503, detail=str(e))  # Service Unavailable
```

`503` means "try again later" — the right code when a downstream service (Qdrant) is down.

### 7.4 Middleware

**Error Handler (`middleware/error_handler.py`):**
```python
def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": str(exc)})
```

**Catches everything.** No stack trace leaks to the client.

**Logging Middleware (`middleware/logging.py`):**
```python
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("%s %s -> %s (%.2f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response
```

Every single request is logged with **method, path, status, and latency**. Essential for production debugging.

### 7.5 Core Engine

**Document Loader (`core/document_loader.py`):**

```python
class DocumentProcessor:
    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }
```

**Production Patterns:**
- **Format validation** before processing (fail fast)
- **File size limits** (`max_upload_size_mb`)
- **SHA-256 hashing** for deduplication
- **Metadata enrichment** at chunk level (source filename, chunk index, total chunks)
- **Performance timing** returned with results

**Vector Store (`core/vector_store.py`):**

```python
class VectorStoreManager:
    def __init__(self):
        self._client: Optional[QdrantClient] = None       # Lazy
        self._vector_store: Optional[QdrantVectorStore] = None  # Lazy
```

**Lazy initialization** — doesn't connect to Qdrant until first use. If the health endpoint is called but no queries are made, no database connection is opened.

**Similarity Threshold Filtering:**
```python
filtered = [(doc, score) for doc, score in results if score >= score_threshold]
```

Only returns results above the confidence threshold. If no results are good enough, returns empty — better than returning garbage.

### 7.6 Data Models — `models/schemas.py`

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query_time_ms: float
    tokens_used: int | None = None
    model: str
    confidence: float = Field(..., ge=0.0, le=1.0,
        description="Proxy confidence based on source similarity scores")
```

Every response includes **timing, model info, confidence, and source provenance**. The client always knows:
- How long did this take?
- Which model generated this?
- How confident is the system?
- Where did the information come from?

### 7.7 Streamlit UI — `ui/streamlit_app.py`

```python
st.set_page_config(page_title="RAG Engine", page_icon="📄", layout="centered")
st.title("RAG Engine")
st.caption("Upload documents and ask grounded questions.")

st.subheader("1) Ingest PDF")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)
if uploaded is not None:
    file_path = save_uploaded_pdf(uploaded)
    if st.button("Ingest"):
        response = requests.post(
            f"{API_BASE}/documents/ingest",
            json={"pdf_path": str(file_path.resolve()), "source_id": file_path.name},
            timeout=120,
        )

st.subheader("2) Query")
question = st.text_input("Question")
top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
if st.button("Ask") and question.strip():
    response = requests.post(
        f"{API_BASE}/query",
        json={"question": question.strip(), "top_k": int(top_k)},
        timeout=120,
    )
```

This is a **lightweight alternative** to the root-level `streamlit_app.py`. Key differences:

| Feature | Root `streamlit_app.py` | `rag-engine` `ui/streamlit_app.py` |
|---------|------------------------|------------------------------------|
| Execution mode | Dual (Inngest + Local fallback) | API-only (calls FastAPI endpoints) |
| Provider config | Full sidebar with model selection | None (uses server-side config) |
| Complexity | 432 lines | 55 lines |
| Use case | Standalone development | Deployed alongside FastAPI backend |

The `rag-engine` UI delegates all logic to the API — it's a **thin client** that only handles file uploads and question submission. This is the production pattern: keep the frontend dumb, keep the backend smart.

### 7.8 Utilities — Logger

```python
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
```

Structured log format: Timestamp | Level | Module | Message. Easy to parse, easy to search, easy to ship to centralized logging.

---

## 8. Step 7: The Infrastructure — Docker & CI/CD

### 8.0 Local Inngest CLI Setup (April 2026 Update)

**Problem Solved:** Inngest Dev Server requires either `npx` (Node.js) or Docker. Many developers have neither installed or available.

**Solution:** Automated download of the Windows Inngest CLI binary.

```powershell
# Create tools directory and download binary
New-Item -ItemType Directory -Force -Path .tools\inngest | Out-Null
$zipPath = Join-Path $PWD '.tools\inngest\inngest_windows_amd64.zip'
Invoke-WebRequest -Uri 'https://github.com/inngest/inngest/releases/download/v1.17.9/inngest_1.17.9_windows_amd64.zip' -OutFile $zipPath
Expand-Archive -LiteralPath $zipPath -DestinationPath .tools\inngest -Force

# Run it
./.tools/inngest/inngest.exe dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

**Why This Works:**
- **No dependencies** — Pure Windows binary, no Node, no Docker needed
- **Always available** — Binary checked into `.tools/` directory
- **Versioned** — Specific release pinned (v1.17.9), no breaking surprises
- **Cross-platform ready** — Same script structure supports macOS/Linux assets

**Next Steps:** Add `.tools/inngest/inngest.exe` to `.gitignore` and include binary download in CI, or commit the binary itself for offline-first development.

### 8.1 Dockerfile

```dockerfile
FROM python:3.12-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY src/ ./src/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "rag_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Production Patterns:**
- **`python:3.12-slim`** — Minimal base image (smaller attack surface, faster pulls)
- **Layer ordering** — `pyproject.toml` before `src/` so dependency installs are cached
- **`HEALTHCHECK`** — Docker automatically restarts unhealthy containers
- **`--no-cache-dir`** — Smaller image size (no pip cache stored)

### 8.2 Docker Compose

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
  rag-engine:
    build: {context: .., dockerfile: docker/Dockerfile}
    ports: ["8000:8000"]
    depends_on:
      qdrant:
        condition: service_healthy  # Wait for Qdrant to be ready!
    environment:
      - QDRANT_URL=http://qdrant:6333  # Docker DNS resolution
```

**`condition: service_healthy`** — The RAG engine doesn't start until Qdrant's health check passes. No race conditions.

### 8.3 GitHub Actions CI

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install .[dev]
      - run: ruff check src tests    # Linting
      - run: pytest                   # Testing
```

Every push and PR automatically runs linting and tests. Broken code never reaches main.

### 8.4 Makefile

```makefile
install:    pip install -e ".[dev]"
test:       pytest tests/ -v --cov=src/rag_engine --cov-report=term-missing
lint:       ruff check src/ tests/ && mypy src/
run:        uvicorn rag_engine.api.app:app --reload --port 8000
docker-up:  docker compose -f docker/docker-compose.yml up --build -d
docker-down: docker compose -f docker/docker-compose.yml down
```

One command to do anything: `make test`, `make run`, `make docker-up`.

---

## 9. Step 8: The Hospital Policy Manual — Environment & Config

### Environment Variable Management (April 2026 Update)

**Before:** INNGEST_ENABLED defaults to `"true"`, causing errors when Dev Server not running.  
**After:** Smart defaults + explicit diagnostics.

**New Behavior:**
1. **INNGEST_ENABLED** defaults to `"false"` — safe, local-first
2. **INNGEST_DEV** persists with settings — backend respects UI choice
3. **Preflight checks** warn before workflow execution — clear visibility

**Updated .env Behavior:**

```bash
# OLD: Would crash silently if Dev Server missing
INNGEST_ENABLED='true'

# NEW: Safe default, user opts into workflow mode via UI
INNGEST_ENABLED='false'
INNGEST_DEV='0'                # NEW: Persisted by Streamlit settings save

# When user toggles "Use Inngest workflow mode" in Streamlit:
# Streamlit auto-updates both flags:
INNGEST_ENABLED='true'
INNGEST_DEV='1'
```

**Key Values:**

```bash
LLM_PROVIDER='ollama'              # Which LLM to use
EMBED_PROVIDER='ollama'            # Which embedding model to use
OLLAMA_BASE_URL='https://api.ollama.com'
OLLAMA_MODEL='gemma3:4b'
OLLAMA_EMBED_MODEL='nomic-embed-text'
OLLAMA_API_KEY='your-key-here'

# Workflow orchestration (NEW defaults)
INNGEST_ENABLED='false'            # Toggle workflow mode (default: off for safety)
INNGEST_DEV='0'                    # Dev server flag (persisted by Streamlit)
INNGEST_API_BASE='http://127.0.0.1:8288/v1'  # Dev server API endpoint
INNGEST_EVENT_API_BASE='http://127.0.0.1:8288'  # Dev server event endpoint

# Vector storage
QDRANT_PATH=qdrant_storage         # Local fallback path
EMBED_DIM='768'                    # Must match your embedding model
```

### pyproject.toml — Dependencies

```toml
[project]
name = "ragproductionapp"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.66.0",          # Claude support
    "fastapi>=0.116.1",           # API framework
    "google-generativeai>=0.8.5", # Gemini support
    "inngest>=0.5.6",             # Workflow orchestration
    "llama-index-core>=0.14.0",   # Document processing
    "llama-index-readers-file>=0.5.4",  # PDF reader
    "openai>=1.107.0",            # OpenAI SDK
    "python-dotenv>=1.1.1",       # Env file loading
    "qdrant-client>=1.15.1",      # Vector database
    "requests>=2.32.5",           # HTTP client
    "streamlit>=1.49.1",          # Web UI
    "uvicorn>=0.35.0",            # ASGI server
]
```

---

## 10. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    streamlit_app.py (Port 8501)                  │
│       ┌──────────┐  ┌──────────────┐  ┌──────────────────┐     │
│       │ Upload   │  │ Ask Question │  │ Model Settings   │     │
│       │ PDF      │  │              │  │ (Sidebar)        │     │
│       └────┬─────┘  └──────┬───────┘  └──────────────────┘     │
└────────────┼───────────────┼────────────────────────────────────┘
             │               │
    ┌────────▼───────────────▼────────┐
    │     EXECUTION MODE SWITCH       │
    │  Inngest? ──→ Send Event        │
    │  Local?   ──→ Direct Call       │
    └────────┬───────────────┬────────┘
             │               │
┌────────────▼───────────────▼────────────────────────────────────┐
│                    BACKEND (FastAPI + Inngest)                   │
│                      main.py (Port 8000)                        │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │  rag_ingest_pdf     │    │  rag_query_pdf_ai   │            │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │            │
│  │  │ load-and-chunk│  │    │  │embed-and-search│  │            │
│  │  └───────┬───────┘  │    │  └───────┬───────┘  │            │
│  │  ┌───────▼───────┐  │    │  ┌───────▼───────┐  │            │
│  │  │embed-and-     │  │    │  │  llm-answer   │  │            │
│  │  │upsert         │  │    │  │               │  │            │
│  │  └───────────────┘  │    │  └───────────────┘  │            │
│  └─────────────────────┘    └─────────────────────┘            │
└────────────┬───────────────────────┬───────────────────┬────────┘
             │                       │                   │
    ┌────────▼────────┐    ┌────────▼────────┐  ┌───────▼────────┐
    │  data_loader.py │    │  vector_db.py   │  │ LLM Providers  │
    │  ┌────────────┐ │    │  ┌────────────┐ │  │ ┌────────────┐ │
    │  │ PDFReader  │ │    │  │  Qdrant    │ │  │ │  OpenAI    │ │
    │  │ Splitter   │ │    │  │  Storage   │ │  │ │  Gemini    │ │
    │  │ Embeddings │ │    │  │  (Remote/  │ │  │ │  Claude    │ │
    │  │            │ │    │  │   Local)   │ │  │ │  Ollama    │ │
    │  └────────────┘ │    │  └────────────┘ │  │ │  Local     │ │
    └─────────────────┘    └─────────────────┘  │ └────────────┘ │
                                                └────────────────┘
```

---

## 11. Production Patterns Used

| # | Pattern | Where Used | Why |
|---|---------|-----------|-----|
| 1 | **Multi-provider abstraction** | `main.py`, `data_loader.py` | Switch LLM/embedding provider via env var |
| 2 | **Graceful degradation** | `vector_db.py`, `streamlit_app.py` | Remote → Local → Error message |
| 3 | **Singleton with cleanup** | `vector_db.py` (`get_qdrant_storage`) | One DB connection, clean shutdown |
| 4 | **Lazy imports** | `main.py` (`importlib`) | Don't require unused SDKs |
| 5 | **Deterministic IDs** | `main.py` (`uuid.uuid5`) | Idempotent re-ingestion |
| 6 | **Rate limiting & throttling** | `main.py` (Inngest decorators) | Protect APIs from abuse |
| 7 | **Durable step functions** | `main.py` (Inngest `ctx.step.run`) | Each step retryable independently |
| 8 | **Type-safe contracts** | `custom_types.py`, `models/schemas.py` | Pydantic validation across boundaries |
| 9 | **Health checks** | `routes/health.py`, `Dockerfile` | Automated liveness monitoring |
| 10 | **Structured logging** | `utils/logger.py`, `middleware/logging.py` | Timestamp, level, module per log line |
| 11 | **Factory pattern** | `api/app.py` (`create_app`) | Testable, multi-worker safe |
| 12 | **Dependency caching** | `Dockerfile` (layer ordering) | Faster Docker builds |
| 13 | **Cross-field validation** | `config.py` (`@validator`) | Catch config errors at startup |
| 14 | **API versioning** | `routes` (`prefix="/api/v1"`) | Non-breaking future changes |
| 15 | **Hash-based fallback embeddings** | `data_loader.py` (`_hash_embedding`) | CI/testing without API keys |
| 16 | **Preflight diagnostics** | `streamlit_app.py` (`_render_inngest_preflight`) | **NEW:** Proactive service health display before workflow execution |
| 17 | **Environment variable persistence** | `streamlit_app.py` (`_save_env`) | **NEW:** Settings survive app restarts |
| 18 | **Endpoint availability checks** | `streamlit_app.py` (`_check_endpoint`) | **NEW:** Detect missing backend services early |
| 19 | **Safe-by-default mode** | `streamlit_app.py` (Inngest default off) | **NEW:** Workflow mode opt-in, not opt-out |
| 20 | **Standalone binary distribution** | `.tools/inngest/` | **NEW:** Inngest CLI without Node/Docker dependency |

---

## 12. How Everything Connects

### Ingestion Flow (Upload a PDF)

```
User uploads PDF via Streamlit
    → streamlit_app.py saves to /uploads/
    → Sends "rag/ingest_pdf" event to Inngest (or runs locally)
    → main.py: rag_ingest_pdf() executes:
        Step 1: data_loader.load_and_chunk_pdf() → chunks[]
        Step 2: data_loader.embed_texts(chunks) → vectors[]
                vector_db.get_qdrant_storage().upsert(ids, vectors, payloads)
    → Returns: {"ingested": N}
```

### Query Flow (Ask a Question)

```
User types question in Streamlit
    → Sends "rag/query_pdf_ai" event to Inngest (or runs locally)
    → main.py: rag_query_pdf_ai() executes:
        Step 1: data_loader.embed_texts([question]) → query_vector
                vector_db.get_qdrant_storage().search(query_vector) → contexts, sources
        Step 2: main.generate_answer(context + question) → answer string
    → Returns: {"answer": "...", "sources": [...], "num_contexts": N}
```

### Project File Structure

```
ProductionGradeRAGPythonApp-main/
├── main.py                          # FastAPI + Inngest backend
├── data_loader.py                   # PDF processing + multi-provider embeddings
├── vector_db.py                     # Qdrant vector storage (remote/local fallback)
├── custom_types.py                  # Pydantic models for type safety
├── streamlit_app.py                 # Full-featured Streamlit UI with diagnostics
├── .env                             # Environment configuration
├── pyproject.toml                   # Dependencies and project metadata
├── doc.md                           # This comprehensive production guide
├── uploads/                         # Uploaded PDF storage
├── qdrant_storage/                  # Local Qdrant data (fallback)
├── .tools/                          # **NEW:** Local development tools
│   └── inngest/
│       ├── inngest.exe              # Inngest CLI binary (Windows v1.17.9)
│       └── inngest_windows_amd64.zip # Source archive
│
└── rag-engine/                      # Production-grade restructured package
    ├── src/rag_engine/
    │   ├── config.py                # Pydantic Settings with validation
    │   ├── api/
    │   │   ├── app.py               # FastAPI factory with lifespan
    │   │   ├── routes/
    │   │   │   ├── health.py        # GET /health
    │   │   │   ├── documents.py     # POST /documents/ingest
    │   │   │   └── query.py         # POST /query
    │   │   └── middleware/
    │   │       ├── error_handler.py # Global exception catching
    │   │       └── logging.py       # Request/response timing
    │   ├── core/
    │   │   ├── document_loader.py   # Multi-format document processor
    │   │   ├── embeddings.py        # OpenAI embedding wrapper
    │   │   ├── vector_store.py      # Qdrant manager with health checks
    │   │   └── rag_chain.py         # Full RAG pipeline orchestrator
    │   ├── models/
    │   │   └── schemas.py           # Request/response Pydantic models
    │   ├── ui/
    │   │   └── streamlit_app.py     # Lightweight Streamlit frontend
    │   └── utils/
    │       └── logger.py            # Structured logging setup
    ├── tests/
    │   ├── unit/                    # Unit tests (no external deps)
    │   └── integration/             # API integration tests
    ├── docker/
    │   ├── Dockerfile               # Multi-stage production image
    │   └── docker-compose.yml       # Qdrant + RAG Engine orchestration
    ├── .github/workflows/ci.yml     # Automated lint + test on every push
    ├── Makefile                      # One-command operations
    └── pyproject.toml               # Package dependencies + tool config
```

---

**Built with production patterns. Documented for humans. Ready for scale.**

*— Adil Shamim, April 2026*
