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


def _ollama_headers() -> dict[str, str]:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _get_llm_config() -> tuple[str, str, str, str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return provider, openai_model, gemini_model, claude_model, ollama_model, ollama_base_url


def _local_answer(user_content: str) -> str:
    marker = "Context:\n"
    q_marker = "\n\nQuestion:"
    if marker in user_content and q_marker in user_content:
        context = user_content.split(marker, 1)[1].split(q_marker, 1)[0].strip()
        lines = [line.strip("- ").strip() for line in context.splitlines() if line.strip()]
        if lines:
            return f"Based on local mode context: {lines[0]}"
    return "Local mode could not find relevant context."


def generate_answer(user_content: str) -> str:
    system_prompt = "You answer questions using only the provided context."
    llm_provider, openai_model, gemini_model, claude_model, ollama_model, ollama_base_url = _get_llm_config()

    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=openai_model,
            temperature=0.2,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    if llm_provider == "gemini":
        try:
            genai = importlib.import_module("google.generativeai")
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires google-generativeai. Install it with: pip install google-generativeai"
            ) from exc
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for LLM_PROVIDER=gemini")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=gemini_model,
            system_instruction=system_prompt,
        )
        response = model.generate_content(user_content)
        return (response.text or "").strip()

    if llm_provider in ("claude", "anthropic"):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Claude support requires anthropic SDK. Install it with: pip install anthropic"
            ) from exc
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=claude")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=claude_model,
            temperature=0.2,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", "") == "text"]
        return "\n".join(text_blocks).strip()

    if llm_provider == "ollama":
        response = requests.post(
            f"{ollama_base_url}/api/chat",
            headers=_ollama_headers(),
            json={
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "").strip()

    if llm_provider == "local":
        return _local_answer(user_content)

    raise ValueError("Unsupported LLM_PROVIDER. Use one of: openai, gemini, claude, ollama, local")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        get_qdrant_storage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = get_qdrant_storage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    answer = await ctx.step.run("llm-answer", lambda: generate_answer(user_content))
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])