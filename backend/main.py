import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from backend.config import settings
from backend.ingestion.storage import CorpusStorage
from backend.agents.normal_agent import NormalAgent
from backend.agents.deep_research_agent import DeepResearchAgent
from backend.memory.memory_manager import MemoryManager
from backend.guardrails.guardrails import check as guardrails_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vectorless RAG Personal Finance Assistant API",
    description="Hierarchical PageIndex-style retrieval over 28 personal finance books using grok-3",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = CorpusStorage()
memory_manager = MemoryManager()


@app.on_event("startup")
async def startup_event():
    await memory_manager.init()
    logger.info("[Startup] MemoryManager initialised.")

# ── LLM Client Factory ─────────────────────────────────────────────────────────

_PLACEHOLDER = "your_"


def _key_ok(key: str) -> bool:
    """Key is usable if non-empty, not a placeholder, and has reasonable length (>10 chars).
    Accepts both legacy AIza... and new AQ... Google AI Studio key formats.
    """
    return bool(key and not key.strip().startswith(_PLACEHOLDER) and len(key.strip()) > 10)


def _log_env_debug(var_name: str, key_val: str):
    """Debug logging reporting existence, length, and first 5 chars only (never full key)."""
    exists = bool(key_val and len(key_val.strip()) > 0)
    length = len(key_val.strip()) if exists else 0
    prefix = key_val.strip()[:5] if exists else "N/A"
    logger.info(
        f"[ENV DEBUG] {var_name}: exists={exists}, length={length}, prefix={prefix}..."
    )


def _build_provider_fns():
    """Returns (provider_name -> factory_fn) dict. Each fn returns client or None."""

    def _try_gemini():
        key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()
        _log_env_debug("GEMINI_API_KEY", key)
        if not _key_ok(key):
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            env_model = os.getenv("GEMINI_MODEL", settings.GEMINI_MODEL)
            # Try configured model first, followed by known stable aliases
            candidate_models = [env_model]
            for m in ["gemini-flash-latest", "gemini-2.0-flash-lite", "gemini-2.0-flash"]:
                if m not in candidate_models:
                    candidate_models.append(m)

            for model in candidate_models:
                logger.info(
                    f"[LLM] Gemini ({model}) key_prefix={key[:5]}... key_len={len(key)}"
                )
                try:
                    return ChatGoogleGenerativeAI(
                        google_api_key=key,
                        model=model,
                        temperature=0.1,
                        streaming=True,
                    )
                except Exception as me:
                    logger.warning(f"[LLM] Gemini init failed for model '{model}': {me}")
            return None
        except Exception as e:
            logger.warning(f"[LLM] Gemini init failed: {e}")
            return None

    def _try_groq():
        key = os.getenv("GROQ_API_KEY", os.getenv("GROK_API_KEY", "")).strip()
        _log_env_debug("GROQ_API_KEY", key)
        if not _key_ok(key):
            return None
        try:
            from langchain_groq import ChatGroq
            model = os.getenv("GROQ_MODEL", settings.GROQ_MODEL)
            logger.info(
                f"[LLM] Groq ({model}) key_prefix={key[:5]}... key_len={len(key)}"
            )
            return ChatGroq(api_key=key, model_name=model, temperature=0.1)
        except Exception as e:
            logger.warning(f"[LLM] Groq init failed: {e}")
            return None

    def _try_xai():
        key = os.getenv("XAI_API_KEY", "").strip()
        _log_env_debug("XAI_API_KEY", key)
        if not _key_ok(key):
            return None
        try:
            from langchain_openai import ChatOpenAI
            model = os.getenv("GROK_MODEL", settings.GROK_MODEL)
            logger.info(
                f"[LLM] xAI Grok ({model}) key_prefix={key[:5]}... key_len={len(key)}"
            )
            return ChatOpenAI(
                api_key=key,
                model=model,
                base_url=settings.XAI_BASE_URL,
                temperature=0.1,
                streaming=True,
            )
        except Exception as e:
            logger.warning(f"[LLM] xAI Grok init failed: {e}")
            return None

    def _try_openai():
        key = os.getenv("OPENAI_API_KEY", "").strip()
        _log_env_debug("OPENAI_API_KEY", key)
        if not _key_ok(key):
            return None
        try:
            from langchain_openai import ChatOpenAI
            model = os.getenv("OPENAI_MODEL", settings.OPENAI_MODEL)
            logger.info(
                f"[LLM] OpenAI ({model}) key_prefix={key[:5]}... key_len={len(key)}"
            )
            return ChatOpenAI(api_key=key, model=model, temperature=0.1)
        except Exception as e:
            logger.warning(f"[LLM] OpenAI init failed: {e}")
            return None

    return {
        "gemini": _try_gemini,
        "groq": _try_groq,
        "xai": _try_xai,
        "openai": _try_openai,
    }


def _reload_env():
    """Reload .env so freshly pasted keys are picked up without restarting the server."""
    env_path = settings.BASE_DIR / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)


def get_llm_client(skip_providers: list[str] | None = None):
    """
    Returns an LLM client based on environment variables in .env.

    - Dynamically reloads .env on every call so newly pasted API keys work immediately.
    - Provider priority: DEFAULT_PROVIDER first; remaining providers used as fallbacks.
    - skip_providers: list of provider names to skip (e.g. ['gemini'] when Gemini quota
      is exhausted at request time, allowing the caller to retry with a fallback).
    """
    _reload_env()

    provider = os.getenv("DEFAULT_PROVIDER", settings.DEFAULT_PROVIDER).lower()
    _FALLBACK_ORDER = ["gemini", "groq", "xai", "openai"]
    ordered = [provider] + [p for p in _FALLBACK_ORDER if p != provider]

    if skip_providers:
        ordered = [p for p in ordered if p not in skip_providers]

    provider_fns = _build_provider_fns()

    for p in ordered:
        fn = provider_fns.get(p)
        if fn:
            client = fn()
            if client is not None:
                if p != provider:
                    logger.warning(
                        f"[LLM] Primary provider '{provider}' unavailable. "
                        f"Using fallback: '{p}'."
                    )
                return client

    logger.error(
        "[LLM] No usable API key found. "
        "Set GEMINI_API_KEY, GROQ_API_KEY, XAI_API_KEY, or OPENAI_API_KEY in your .env file."
    )
    return None



# ── Request Models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    mode: str = "normal"
    skip_vagueness: bool = True
    use_books: bool = False
    session_id: str = "default"  # unique per browser tab / conversation


class ConfigUpdateRequest(BaseModel):
    num_subqueries: Optional[int] = None
    top_k_rerank: Optional[int] = None
    tree_expansion_breadth: Optional[int] = None
    context_token_budget: Optional[int] = None
    max_books_to_route: Optional[int] = None
    max_chapters_per_book: Optional[int] = None
    max_sections_per_chapter: Optional[int] = None
    token_budget_book_routing: Optional[int] = None
    token_budget_chapter_select: Optional[int] = None
    token_budget_section_select: Optional[int] = None
    token_budget_answer_synthesis: Optional[int] = None
    cross_encoder_enabled: Optional[bool] = None
    deep_research_cross_encoder: Optional[bool] = None
    min_relevance_score: Optional[float] = None
    enable_web_search_fallback: Optional[bool] = None


# ── Health & Corpus Endpoints ──────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    books = storage.get_all_books()
    provider = settings.DEFAULT_PROVIDER.lower()
    # Pick active model name based on provider
    if provider == "gemini":
        active_model = settings.GEMINI_MODEL
    elif provider == "groq":
        active_model = settings.GROQ_MODEL
    elif provider == "xai":
        active_model = settings.GROK_MODEL
    else:
        active_model = settings.OPENAI_MODEL

    gemini_configured = bool(settings.GEMINI_API_KEY and not settings.GEMINI_API_KEY.startswith("your_"))
    groq_configured = bool(settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("your_"))
    tavily_configured = bool(settings.TAVILY_API_KEY and not settings.TAVILY_API_KEY.startswith("your_"))
    return {
        "status": "ok",
        "service": "Vectorless RAG Finance Assistant",
        "model": active_model,
        "provider": provider,
        "corpus_books": len(books),
        "gemini_configured": gemini_configured,
        "groq_configured": groq_configured,
        "tavily_configured": tavily_configured,
    }


@app.get("/api/corpus")
def get_corpus():
    books = storage.get_all_books()
    return {
        "total_books": len(books),
        "books": [
            {
                "book_id": b["node_id"],
                "title": b["title"],
                "stance": b["stance"],
                "topic_tags": b["topic_tags"],
                "summary": b["summary"],
                "audience": b["audience"],
                "total_pages": b["end_page"],
            }
            for b in books
        ],
    }


# ── Config Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return {
        "num_subqueries": settings.NUM_SUBQUERIES,
        "top_k_rerank": settings.TOP_K_RERANK,
        "tree_expansion_breadth": settings.TREE_EXPANSION_BREADTH,
        "max_books_to_route": settings.MAX_BOOKS_TO_ROUTE,
        "max_chapters_per_book": settings.MAX_CHAPTERS_PER_BOOK,
        "max_sections_per_chapter": settings.MAX_SECTIONS_PER_CHAPTER,
        "context_token_budget": settings.CONTEXT_TOKEN_BUDGET_PER_STEP,
        "token_budget_book_routing": settings.TOKEN_BUDGET_BOOK_ROUTING,
        "token_budget_chapter_select": settings.TOKEN_BUDGET_CHAPTER_SELECT,
        "token_budget_section_select": settings.TOKEN_BUDGET_SECTION_SELECT,
        "token_budget_answer_synthesis": settings.TOKEN_BUDGET_ANSWER_SYNTHESIS,
        "cross_encoder_enabled": settings.CROSS_ENCODER_ENABLED,
        "deep_research_cross_encoder": settings.DEEP_RESEARCH_CROSS_ENCODER,
        "cross_encoder_model": settings.CROSS_ENCODER_MODEL,
        "min_relevance_score": settings.MIN_RELEVANCE_SCORE,
        "enable_web_search_fallback": settings.ENABLE_WEB_SEARCH_FALLBACK,
        "default_provider": settings.DEFAULT_PROVIDER,
        "grok_model": settings.GROK_MODEL,
    }


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    if req.num_subqueries is not None:
        settings.NUM_SUBQUERIES = req.num_subqueries
    if req.top_k_rerank is not None:
        settings.TOP_K_RERANK = req.top_k_rerank
    if req.tree_expansion_breadth is not None:
        settings.TREE_EXPANSION_BREADTH = req.tree_expansion_breadth
    if req.context_token_budget is not None:
        settings.CONTEXT_TOKEN_BUDGET_PER_STEP = req.context_token_budget
    if req.max_books_to_route is not None:
        settings.MAX_BOOKS_TO_ROUTE = req.max_books_to_route
    if req.max_chapters_per_book is not None:
        settings.MAX_CHAPTERS_PER_BOOK = req.max_chapters_per_book
    if req.max_sections_per_chapter is not None:
        settings.MAX_SECTIONS_PER_CHAPTER = req.max_sections_per_chapter
    if req.token_budget_book_routing is not None:
        settings.TOKEN_BUDGET_BOOK_ROUTING = req.token_budget_book_routing
    if req.token_budget_chapter_select is not None:
        settings.TOKEN_BUDGET_CHAPTER_SELECT = req.token_budget_chapter_select
    if req.token_budget_section_select is not None:
        settings.TOKEN_BUDGET_SECTION_SELECT = req.token_budget_section_select
    if req.token_budget_answer_synthesis is not None:
        settings.TOKEN_BUDGET_ANSWER_SYNTHESIS = req.token_budget_answer_synthesis
    if req.cross_encoder_enabled is not None:
        settings.CROSS_ENCODER_ENABLED = req.cross_encoder_enabled
    if req.deep_research_cross_encoder is not None:
        settings.DEEP_RESEARCH_CROSS_ENCODER = req.deep_research_cross_encoder
    if req.min_relevance_score is not None:
        settings.MIN_RELEVANCE_SCORE = req.min_relevance_score
    if req.enable_web_search_fallback is not None:
        settings.ENABLE_WEB_SEARCH_FALLBACK = req.enable_web_search_fallback
    return {"status": "updated", "config": get_config()}


# ── Ingestion Endpoints ────────────────────────────────────────────────────────

@app.post("/api/ingest")
def trigger_ingestion():
    from backend.ingestion.seed_all_28 import seed_all
    llm_client = get_llm_client()
    try:
        result = seed_all(llm_client=llm_client)
        return {"status": "success", "message": "Corpus ingestion complete.", "detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ingest/status")
def ingestion_status():
    books = storage.get_all_books()
    books_with_summaries = [b for b in books if b.get("summary") and len(b["summary"]) > 50]
    return {
        "total_books_in_db": len(books),
        "books_with_summaries": len(books_with_summaries),
        "ingestion_complete": len(books_with_summaries) > 0,
        "pdf_dir": str(settings.PDF_DIR),
        "db_path": str(settings.DB_PATH),
    }


# ── Chat Streaming Endpoint ────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Streaming SSE endpoint for real-time progressive response rendering.
    Each yielded line is: data: <JSON>\\n\\n
    Events: status | clarification | traversal | sources | answer_chunk | done | error

    Memory: short-term (last 10 turns) + long-term (LLM-summarised user profile)
    Guardrails: hard-blocked topics + off-topic soft redirect
    """
    llm_client = get_llm_client()
    session_id = req.session_id or "default"

    async def event_generator():
        # ── Guardrails check (fast, no LLM call) ──────────────────────────
        is_blocked, block_msg = guardrails_check(req.query)
        if is_blocked:
            yield f"data: {json.dumps({'event': 'answer_chunk', 'chunk': block_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            return

        yield f"data: {json.dumps({'event': 'status', 'message': f'Initializing {req.mode.upper()} Agent...'})}\n\n"
        await asyncio.sleep(0.05)

        if not llm_client:
            msg = (
                "⚠️ **API Key Notice**: No valid API key detected in `.env`.\n\n"
                "Please open `.env` and paste your key:\n"
                "- `GEMINI_API_KEY=AIzaSy...` (Free at https://aistudio.google.com/apikey)\n"
                "- OR `GROQ_API_KEY=gsk_...` (Free at https://console.groq.com/keys)\n"
                "- Optional for web search: `TAVILY_API_KEY=tvly-...` (Free at https://app.tavily.com/)\n\n"
                "Once saved in `.env`, send your query again — live API calls will execute!"
            )
            yield f"data: {json.dumps({'event': 'answer_chunk', 'chunk': msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            return

        # ── Load memory ────────────────────────────────────────────────────
        conversation_history = await memory_manager.get_short_term(session_id)
        user_profile = await memory_manager.get_long_term(session_id)

        # Persist the user's turn immediately
        await memory_manager.add_turn(session_id, "user", req.query)

        full_assistant_response = []

        try:
            if req.mode == "deep_research":
                agent = DeepResearchAgent(storage, llm_client)
            else:
                agent = NormalAgent(storage, llm_client)

            async for item in agent.run_stream(
                req.query,
                skip_vagueness=req.skip_vagueness,
                use_books=req.use_books,
                conversation_history=conversation_history,
                user_profile=user_profile,
            ):
                # Collect assistant text for memory storage
                if item.get("event") == "answer_chunk":
                    full_assistant_response.append(item.get("chunk", ""))
                yield f"data: {json.dumps(item)}\n\n"
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Chat endpoint error: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            return

        # ── Persist assistant turn + maybe update long-term memory ─────────
        if full_assistant_response:
            assistant_text = "".join(full_assistant_response)
            await memory_manager.add_turn(session_id, "assistant", assistant_text[:1500])
            # Non-blocking: update long-term profile if threshold reached
            asyncio.create_task(
                memory_manager.maybe_update_long_term(session_id, llm_client)
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear short-term memory for a session (e.g. 'New Chat' button)."""
    await memory_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
