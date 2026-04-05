import asyncio
import json
import os
from pathlib import Path
import time
import uuid

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv, set_key
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import get_qdrant_storage

load_dotenv()
ENV_FILE = Path(".env").resolve()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

if os.getenv("OLLAMA_API_KEY"):
    os.environ.setdefault("LLM_PROVIDER", "ollama")
    os.environ.setdefault("EMBED_PROVIDER", "ollama")
    os.environ.setdefault("OLLAMA_BASE_URL", "https://api.ollama.com")


def _save_env(key: str, value: str) -> None:
    os.environ[key] = value
    set_key(str(ENV_FILE), key, value)


def _llm_model_key(provider: str) -> str:
    return {
        "openai": "OPENAI_MODEL",
        "gemini": "GEMINI_MODEL",
        "claude": "CLAUDE_MODEL",
        "ollama": "OLLAMA_MODEL",
        "local": "LOCAL_MODEL",
    }.get(provider, "OPENAI_MODEL")


def _embed_model_key(provider: str) -> str:
    return {
        "openai": "EMBED_MODEL",
        "gemini": "GEMINI_EMBED_MODEL",
        "ollama": "OLLAMA_EMBED_MODEL",
        "local": "LOCAL_EMBED_MODEL",
    }.get(provider, "EMBED_MODEL")


def _ollama_headers() -> dict[str, str]:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _fetch_ollama_models() -> list[str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "https://api.ollama.com")
    try:
        response = requests.get(
            f"{base_url}/api/tags",
            headers=_ollama_headers(),
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        models = [m.get("name", "") for m in payload.get("models", [])]
        models = [m for m in models if m]
        if models:
            return sorted(set(models))
    except Exception:
        pass
    return ["gemma3:4b", "gpt-oss:20b", "llama3.1"]


def _llm_models_for(provider: str) -> list[str]:
    if provider == "openai":
        return ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "o3-mini"]
    if provider == "gemini":
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"]
    if provider == "claude":
        return ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"]
    if provider == "ollama":
        return _fetch_ollama_models()
    return ["local-default"]


def _embed_models_for(provider: str) -> list[str]:
    if provider == "openai":
        return ["text-embedding-3-small", "text-embedding-3-large"]
    if provider == "gemini":
        return ["models/text-embedding-004"]
    if provider == "ollama":
        models = _fetch_ollama_models()
        if "nomic-embed-text" not in models:
            models = ["nomic-embed-text", *models]
        return models
    return ["local-hash-v1"]


def _current_provider(value: str, allowed: list[str], default: str) -> str:
    if value in allowed:
        return value
    return default


def _render_model_settings() -> None:
    llm_options = ["openai", "gemini", "claude", "ollama", "local"]
    embed_options = ["openai", "gemini", "ollama", "local"]

    current_llm = _current_provider(os.getenv("LLM_PROVIDER", "ollama").lower(), llm_options, "ollama")
    current_embed = _current_provider(os.getenv("EMBED_PROVIDER", "ollama").lower(), embed_options, "ollama")

    st.sidebar.header("Model Settings")
    llm_provider = st.sidebar.selectbox(
        "LLM provider",
        llm_options,
        index=llm_options.index(current_llm),
        key="settings_llm_provider",
    )
    llm_model_key = _llm_model_key(llm_provider)
    llm_models = _llm_models_for(llm_provider)
    current_llm_model = os.getenv(llm_model_key, llm_models[0] if llm_models else "")
    llm_model_default = current_llm_model if current_llm_model in llm_models else llm_models[0]
    llm_model = st.sidebar.selectbox(
        "LLM model version",
        llm_models,
        index=llm_models.index(llm_model_default),
        key="settings_llm_model",
    )

    embed_provider = st.sidebar.selectbox(
        "Embedding provider",
        embed_options,
        index=embed_options.index(current_embed),
        key="settings_embed_provider",
    )
    embed_model_key = _embed_model_key(embed_provider)
    embed_models = _embed_models_for(embed_provider)
    current_embed_model = os.getenv(embed_model_key, embed_models[0] if embed_models else "")
    embed_model_default = current_embed_model if current_embed_model in embed_models else embed_models[0]
    embed_model = st.sidebar.selectbox(
        "Embedding model version",
        embed_models,
        index=embed_models.index(embed_model_default),
        key="settings_embed_model",
    )

    embed_dim = 768 if embed_provider in {"gemini", "ollama", "local"} else 3072
    st.sidebar.caption(f"Embedding dimension: {embed_dim} (auto)")

    st.sidebar.markdown("API Key Source")
    st.sidebar.caption("You can input and save API keys here.")
    openai_key = st.sidebar.text_input(
        "OPENAI_API_KEY",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        key="settings_openai_key",
    )
    gemini_key = st.sidebar.text_input(
        "GEMINI_API_KEY",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        key="settings_gemini_key",
    )
    claude_key = st.sidebar.text_input(
        "ANTHROPIC_API_KEY",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        key="settings_claude_key",
    )
    ollama_key = st.sidebar.text_input(
        "OLLAMA_API_KEY",
        value=os.getenv("OLLAMA_API_KEY", ""),
        type="password",
        key="settings_ollama_key",
    )

    inngest_enabled = st.sidebar.checkbox(
        "Use Inngest workflow mode",
        value=os.getenv("INNGEST_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        key="settings_inngest_enabled",
    )

    save = st.sidebar.button("Save settings", key="settings_save")

    if save:
        _save_env("LLM_PROVIDER", llm_provider)
        _save_env(llm_model_key, llm_model)
        _save_env("EMBED_PROVIDER", embed_provider)
        _save_env(embed_model_key, embed_model)
        _save_env("EMBED_DIM", str(int(embed_dim)))
        _save_env("INNGEST_ENABLED", "true" if inngest_enabled else "false")
        _save_env("INNGEST_DEV", "1" if inngest_enabled else "0")

        if openai_key:
            _save_env("OPENAI_API_KEY", openai_key)
        if gemini_key:
            _save_env("GEMINI_API_KEY", gemini_key)
        if claude_key:
            _save_env("ANTHROPIC_API_KEY", claude_key)
        if ollama_key:
            _save_env("OLLAMA_API_KEY", ollama_key)

        st.sidebar.success("Settings saved. Applying now...")
        st.rerun()


_render_model_settings()


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
        event_api_base_url=os.getenv("INNGEST_EVENT_API_BASE", "http://127.0.0.1:8288"),
    )


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def _source_hint() -> str | None:
    hint = st.session_state.get("last_uploaded_source")
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    return None


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


async def send_rag_query_event(question: str, top_k: int, source_hint: str | None = None) -> str:
    client = get_inngest_client()
    payload = {
        "question": question,
        "top_k": top_k,
    }
    if source_hint:
        payload["source_hint"] = source_hint
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data=payload,
        )
    )
    return result[0]


def _send_error_message(exc: Exception) -> str:
    return (
        f"Failed to send event to Inngest: {exc}. "
        "Make sure Inngest Dev Server is running and connected to your FastAPI app. "
        "Docs flow: set INNGEST_DEV=1, run FastAPI on http://127.0.0.1:8000, then run Inngest Dev Server "
        "with update URL http://127.0.0.1:8000/api/inngest (or use Docker fallback from docs). "
        "Expected default endpoint: http://127.0.0.1:8288"
    )


def _is_send_events_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "SendEventsError"


def _run_ingest_locally(pdf_path: Path) -> int:
    backend_payload = _post_backend_json(
        "/api/local-ingest",
        {
            "pdf_path": str(pdf_path.resolve()),
            "source_id": pdf_path.name,
        },
        timeout=120,
    )
    if isinstance(backend_payload, dict) and isinstance(backend_payload.get("ingested"), int):
        return int(backend_payload["ingested"])

    chunks = load_and_chunk_pdf(str(pdf_path.resolve()))
    vectors = embed_texts(chunks)
    source_id = pdf_path.name
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    get_qdrant_storage().upsert(ids, vectors, payloads)
    return len(chunks)


def _run_query_locally(question: str, top_k: int, source_hint: str | None = None) -> dict:
    backend_query_payload = {"question": question, "top_k": int(top_k)}
    if source_hint:
        backend_query_payload["source_hint"] = source_hint

    backend_payload = _post_backend_json(
        "/api/local-query-ai",
        backend_query_payload,
        timeout=120,
    )
    if isinstance(backend_payload, dict) and "answer" in backend_payload:
        return {
            "answer": backend_payload.get("answer", ""),
            "sources": backend_payload.get("sources", []),
            "num_contexts": int(backend_payload.get("num_contexts", 0)),
        }

    query_vec = embed_texts([question])[0]
    found = get_qdrant_storage().search(query_vec, top_k)
    context_block = "\n\n".join(f"- {chunk}" for chunk in found["contexts"])
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # Reuse the provider-aware generator configured in main.py.
    from main import generate_answer

    answer = generate_answer(user_content)
    return {
        "answer": answer,
        "sources": found["sources"],
        "num_contexts": len(found["contexts"]),
    }


def _run_query_context_only(question: str, top_k: int, source_hint: str | None = None) -> dict:
    backend_query_payload = {"question": question, "top_k": int(top_k)}
    if source_hint:
        backend_query_payload["source_hint"] = source_hint

    backend_payload = _post_backend_json(
        "/api/local-query-context",
        backend_query_payload,
        timeout=60,
    )
    if isinstance(backend_payload, dict) and "answer" in backend_payload:
        return {
            "answer": backend_payload.get("answer", ""),
            "sources": backend_payload.get("sources", []),
            "num_contexts": int(backend_payload.get("num_contexts", 0)),
        }

    query_vec = embed_texts([question])[0]
    found = get_qdrant_storage().search(query_vec, top_k)
    contexts = found.get("contexts", [])
    if not contexts:
        answer = "I could not find relevant context in indexed documents for this question."
    else:
        answer = f"Based on retrieved context: {contexts[0][:900]}"
    return {
        "answer": answer,
        "sources": found.get("sources", []),
        "num_contexts": len(contexts),
    }


def _local_fallback_help() -> str:
    return (
        "Local fallback requires a configured model provider. "
        "Set EMBED_PROVIDER and LLM_PROVIDER plus required credentials in the Model Settings sidebar. "
        "Examples: OPENAI_API_KEY for openai, or OLLAMA_BASE_URL and OLLAMA_API_KEY for ollama. "
        "Also ensure vector storage is reachable via QDRANT_URL, or set QDRANT_PATH for embedded local storage."
    )


def _fastapi_base_url() -> str:
    return os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _post_backend_json(path: str, payload: dict, timeout: float) -> dict | None:
    try:
        response = requests.post(f"{_fastapi_base_url()}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def _inngest_enabled() -> bool:
    return os.getenv("INNGEST_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _check_endpoint(url: str, timeout: float = 1.5) -> tuple[bool, str]:
    try:
        response = requests.get(url, timeout=timeout)
        return True, str(response.status_code)
    except requests.RequestException as exc:
        return False, str(exc)


def _render_inngest_preflight() -> None:
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
        "python -m uvicorn main:app --host 127.0.0.1 --port 8000\n\n"
        "# 2) Start Inngest Dev Server (npx)\n"
        "npx --ignore-scripts=false inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery\n\n"
        "# 2b) Docker fallback if npx is unavailable\n"
        "docker run -p 8288:8288 inngest/inngest inngest dev -u http://host.docker.internal:8000/api/inngest --no-discovery",
        language="bash",
    )


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        if isinstance(data.get("runs"), list):
            return data["runs"]
    return []


def _normalized_status(status: object) -> str:
    return str(status or "").strip().lower()


def _extract_run_output(run: dict) -> dict:
    output = run.get("output")
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _pick_best_run(runs: list[dict]) -> dict:
    if not runs:
        return {}

    terminal_statuses = {"completed", "succeeded", "success", "finished", "failed", "cancelled", "canceled", "errored", "error"}

    # Prefer run with output first.
    for run in runs:
        if _extract_run_output(run):
            return run

    # Prefer terminal run if available.
    for run in runs:
        if _normalized_status(run.get("status")) in terminal_statuses:
            return run

    # Fallback to first run returned by API.
    return runs[0]


def wait_for_run_output(event_id: str, timeout_s: float | None = None, poll_interval_s: float = 0.5) -> dict:
    if timeout_s is None:
        timeout_s = float(os.getenv("INNGEST_RUN_TIMEOUT_S", "60"))

    start = time.time()
    last_status = None
    poll_count = 0
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = _pick_best_run(runs)
            status = _normalized_status(run.get("status"))
            if status:
                last_status = status

            # Check for failure immediately.
            if status in {"failed", "cancelled", "canceled", "errored", "error"}:
                raise RuntimeError(f"Function run {status}")

            # Some Inngest responses may populate output before terminal status.
            output = _extract_run_output(run)
            if output and isinstance(output, dict) and output.get("answer"):
                return output

            if status in {"completed", "succeeded", "success", "finished"}:
                return output if output else {}

        poll_count += 1
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output after {poll_count} polls (last status: {last_status})")
        time.sleep(poll_interval_s)


st.title("Upload a PDF to Ingest")
_render_inngest_preflight()
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        st.session_state["last_uploaded_source"] = path.name
        if not _inngest_enabled():
            try:
                ingested = _run_ingest_locally(path)
                st.success(f"Ingested locally: {ingested} chunks from {path.name}")
                st.caption("Inngest is disabled by config.")
            except Exception as local_exc:
                st.error(f"Local ingestion failed: {local_exc}")
                st.info(_local_fallback_help())
            st.stop()
        try:
            asyncio.run(send_rag_ingest_event(path))
            time.sleep(0.3)
            st.success(f"Triggered ingestion for: {path.name}")
            st.caption("You can upload another PDF if you like.")
        except Exception as exc:
            if _is_send_events_error(exc):
                st.warning(_send_error_message(exc))
                try:
                    ingested = _run_ingest_locally(path)
                    st.success(f"Inngest unavailable. Ingested locally: {ingested} chunks from {path.name}")
                except Exception as local_exc:
                    st.error(f"Local ingestion failed: {local_exc}")
                    st.info(_local_fallback_help())
            else:
                st.error(f"Unexpected ingestion error: {exc}")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            source_hint = _source_hint()
            if not _inngest_enabled():
                try:
                    output = _run_query_locally(question.strip(), int(top_k), source_hint=source_hint)
                    st.subheader("Answer")
                    st.write(output.get("answer", "") or "(No answer)")
                    sources = output.get("sources", [])
                    if sources:
                        st.caption("Sources")
                        for s in sources:
                            st.write(f"- {s}")
                except Exception as local_exc:
                    st.error(f"Local query failed: {local_exc}")
                    st.info(_local_fallback_help())
                st.stop()
            try:
                event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k), source_hint=source_hint))
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                sources = output.get("sources", [])

                st.subheader("Answer")
                st.write(answer or "(No answer)")
                if sources:
                    st.caption("Sources")
                    for s in sources:
                        st.write(f"- {s}")
            except requests.RequestException as exc:
                st.error(f"Failed to read Inngest run output: {exc}")
            except (TimeoutError, RuntimeError) as exc:
                st.warning(f"{exc}. Falling back to context-only retrieval mode.")
                try:
                    output = _run_query_context_only(question.strip(), int(top_k), source_hint=source_hint)
                    st.subheader("Answer")
                    st.write(output.get("answer", "") or "(No answer)")
                    sources = output.get("sources", [])
                    if sources:
                        st.caption("Sources")
                        for s in sources:
                            st.write(f"- {s}")
                except Exception as local_exc:
                    st.error(f"Context-only fallback failed: {local_exc}")
                    st.info(_local_fallback_help())
            except Exception as exc:
                if _is_send_events_error(exc):
                    st.warning(_send_error_message(exc))
                    try:
                        output = _run_query_locally(question.strip(), int(top_k), source_hint=source_hint)
                        st.subheader("Answer")
                        st.write(output.get("answer", "") or "(No answer)")
                        sources = output.get("sources", [])
                        if sources:
                            st.caption("Sources")
                            for s in sources:
                                st.write(f"- {s}")
                    except Exception as local_exc:
                        st.error(f"Local query failed: {local_exc}")
                        st.info(_local_fallback_help())
                else:
                    st.error(f"Unexpected query error: {exc}")

