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
import re
from openai import OpenAI
from pydantic import BaseModel
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import get_qdrant_storage
from custom_types import RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc


load_dotenv()

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "from", "with",
    "is", "are", "was", "were", "be", "as", "this", "that", "it", "about", "tell", "me", "all",
    "your", "you", "their", "my", "our", "resume", "pdf",
}


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


def _compact_spaced_letters(text: str) -> str:
    # Converts OCR-style spaced words like "A c h i e v e d" -> "Achieved".
    return re.sub(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), text)


def _normalize_for_match(text: str) -> str:
    compact = _compact_spaced_letters(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", compact).strip()


def _extract_keywords(question: str) -> list[str]:
    tokens = [t for t in _normalize_for_match(question).split() if len(t) >= 3 and t not in _STOPWORDS]
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered_unique.append(token)
    return ordered_unique


def _is_skill_question(question: str) -> bool:
    q = _normalize_for_match(question)
    markers = ("skill", "skills", "tech", "technology", "stack", "tool", "tools", "experience", "expertise")
    return any(marker in q for marker in markers)


def _rank_records(question: str, records: list[dict], top_k: int) -> tuple[list[str], list[str]]:
    if not records:
        return [], []

    keywords = _extract_keywords(question)
    is_skill = _is_skill_question(question)

    def _score(record: dict) -> float:
        text_norm = _normalize_for_match(str(record.get("text", "")))
        overlap = sum(1 for kw in keywords if kw in text_norm)
        skill_bonus = 0
        if is_skill and any(marker in text_norm for marker in ("skill", "skills", "technical", "technology", "stack", "tool", "experience")):
            skill_bonus = 3
        vector_score = float(record.get("score") or 0.0)
        return (overlap * 5.0) + skill_bonus + vector_score

    ranked = sorted(
        list(enumerate(records)),
        key=lambda item: (_score(item[1]), -item[0]),
        reverse=True,
    )

    selected_contexts: list[str] = []
    selected_sources: list[str] = []
    seen_contexts: set[str] = set()

    for _, record in ranked:
        text = _compact_spaced_letters(str(record.get("text", "")).strip())
        source = str(record.get("source", "")).strip()
        if not text or text in seen_contexts:
            continue
        seen_contexts.add(text)
        selected_contexts.append(text)
        if source and source not in selected_sources:
            selected_sources.append(source)
        if len(selected_contexts) >= top_k:
            break

    return selected_contexts, selected_sources


def _evidence_only_answer(question: str, contexts: list[str]) -> str:
    cleaned = [_compact_spaced_letters(c.strip()) for c in contexts if c and c.strip()]
    if not cleaned:
        return "I don't know based on provided documents."

    keywords = _extract_keywords(question)
    is_skill = _is_skill_question(question)
    scored: list[tuple[float, str]] = []
    for chunk in cleaned:
        norm = _normalize_for_match(chunk)
        overlap = sum(1 for kw in keywords if kw in norm)
        marker_bonus = 0
        if is_skill and any(marker in norm for marker in ("skill", "skills", "experience", "python", "machine", "llm", "ai", "tool", "framework")):
            marker_bonus = 1
        scored.append(((overlap * 3.0) + marker_bonus, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    snippets: list[str] = []
    for score, chunk in scored:
        if score <= 0 and snippets:
            continue
        snippet = chunk.replace("\n", " ").strip()[:220]
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= 4:
            break

    if not snippets:
        snippets = [cleaned[0].replace("\n", " ").strip()[:220]]

    bullet_lines = "\n".join(f"- {s}" for s in snippets)
    if is_skill:
        return (
            "I am using strict evidence-only mode. I can only report skills explicitly present in retrieved context:\n"
            f"{bullet_lines}"
        )
    return f"I am using strict evidence-only mode. Relevant evidence:\n{bullet_lines}"


def _local_answer(user_content: str) -> str:
    marker = "Context:\n"
    q_marker = "\n\nQuestion:"
    if marker in user_content and q_marker in user_content:
        context_block = user_content.split(marker, 1)[1].split(q_marker, 1)[0].strip()
        question = user_content.split(q_marker, 1)[1].strip().splitlines()[0] if q_marker in user_content else ""
        lines = [line.strip("- ").strip() for line in context_block.splitlines() if line.strip()]
        return _evidence_only_answer(question, lines)
    return "I don't know based on provided documents."


def generate_answer(user_content: str) -> str:
    system_prompt = (
        "You are a strict retrieval assistant. "
        "Answer ONLY from provided context. "
        "If context does not contain the answer, reply exactly: "
        "I don't know based on provided documents. "
        "Do not invent facts or skills."
    )
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
        try:
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
            content = payload.get("message", {}).get("content", "").strip()
            if content:
                return content
            logging.warning("Ollama returned empty content, using local context fallback.")
        except requests.RequestException as exc:
            logging.warning("Ollama request failed (%s), using local context fallback.", exc)
        except ValueError as exc:
            logging.warning("Ollama response parse failed (%s), using local context fallback.", exc)
        return _local_answer(user_content)

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
    def _search(question: str, top_k: int = 5, source_hint: str | None = None) -> RAGSearchResult:
        found = _search_contexts(question, top_k, source_hint)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    source_hint = ctx.event.data.get("source_hint")

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k, source_hint),
        output_type=RAGSearchResult,
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    def _safe_answer() -> str:
        try:
            return generate_answer(user_content)
        except Exception as exc:
            logging.exception("LLM generation failed, using local context fallback: %s", exc)
            return _local_answer(user_content)

    answer = await ctx.step.run("llm-answer", _safe_answer)
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


class LocalIngestRequest(BaseModel):
    pdf_path: str
    source_id: str | None = None


class LocalQueryRequest(BaseModel):
    question: str
    top_k: int = 5
    source_hint: str | None = None


def _search_contexts(question: str, top_k: int, source_hint: str | None = None) -> dict:
    query_vec = embed_texts([question])[0]
    search_limit = min(max(int(top_k), 12 if _is_skill_question(question) else int(top_k)), 20)
    raw = get_qdrant_storage().search(query_vec, search_limit)
    records = raw.get("records", []) if isinstance(raw, dict) else []

    if source_hint and records:
        source_norm = source_hint.strip().lower()
        source_scoped = [r for r in records if str(r.get("source", "")).strip().lower() == source_norm]
        if source_scoped:
            records = source_scoped

    if records:
        contexts, sources = _rank_records(question, records, int(top_k))
        if contexts:
            return {"contexts": contexts, "sources": sources}

    return {
        "contexts": raw.get("contexts", [])[: int(top_k)] if isinstance(raw, dict) else [],
        "sources": raw.get("sources", []) if isinstance(raw, dict) else [],
    }


def _context_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    return (
        "Use only the following context to answer the question.\n"
        "If answer is not present in context, reply exactly: I don't know based on provided documents.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )


app = FastAPI()


@app.post("/api/local-ingest")
async def local_ingest(payload: LocalIngestRequest):
    source_id = payload.source_id or payload.pdf_path
    chunks = load_and_chunk_pdf(payload.pdf_path)
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    upsert_payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    get_qdrant_storage().upsert(ids, vecs, upsert_payloads)
    return {"ingested": len(chunks), "source_id": source_id}


@app.post("/api/local-query-context")
async def local_query_context(payload: LocalQueryRequest):
    found = _search_contexts(payload.question, int(payload.top_k), payload.source_hint)
    contexts = found.get("contexts", [])
    answer = _evidence_only_answer(payload.question, contexts)
    return {
        "answer": answer,
        "sources": found.get("sources", []),
        "num_contexts": len(contexts),
    }


@app.post("/api/local-query-ai")
async def local_query_ai(payload: LocalQueryRequest):
    found = _search_contexts(payload.question, int(payload.top_k), payload.source_hint)
    contexts = found.get("contexts", [])
    answer = generate_answer(_context_prompt(payload.question, contexts))
    return {
        "answer": answer,
        "sources": found.get("sources", []),
        "num_contexts": len(contexts),
    }

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])