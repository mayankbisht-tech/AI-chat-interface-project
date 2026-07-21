import json
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from backend.ingestion.storage import CorpusStorage
from backend.retrieval.vagueness_detector import VaguenessDetector
from backend.retrieval.multi_query import MultiQueryGenerator
from backend.retrieval.tree_traverser import VectorlessTreeTraverser
from backend.retrieval.reranker import get_reranker
from backend.retrieval.relevance_gate import HardRelevanceGate
from backend.tools.tavily_search import TavilySearchTool
from backend.config import settings
from backend.logging_config import setup_logger

logger = setup_logger("NormalAgent")

TEMPORAL_KEYWORDS = [
    "current", "2026", "2025", "2024", "today", "latest", "now",
    "tax rate", "interest rate", "fed", "federal reserve", "inflation rate",
    "bracket", "cpi", "stock price", "market", "rate", "nvidia", "sp500",
]

LATENCY_TARGET_SECONDS = 18


class NormalAgent:
    """
    Normal Agent — Vectorless PageIndex RAG Assistant.

    Execution Pipeline:
    1. Query Received → Check if vagueness clarification is needed.
    2. Step 1: Check if books are needed for the answer (Level 0 routing → tree traversal → reranking).
    3. Step 2: Check for LLM + Tavily for internet search (for current data or corpus gaps).
    4. Step 3: Stream answer in proper paragraph format with strict system message (no references/page numbers).
    """

    def __init__(self, storage: CorpusStorage, llm_client=None):
        self.storage = storage
        self.llm_client = llm_client
        self.vagueness_detector = VaguenessDetector(llm_client)
        self.multi_query_gen = MultiQueryGenerator(llm_client)
        self.traverser = VectorlessTreeTraverser(storage, llm_client)
        self.reranker = get_reranker(
            use_cross_encoder=settings.CROSS_ENCODER_ENABLED,
            llm_client=llm_client,
            top_k=settings.TOP_K_RERANK,
        )
        self.relevance_gate = HardRelevanceGate(
            min_score_threshold=settings.MIN_RELEVANCE_SCORE
        )
        self.tavily_tool = TavilySearchTool()

    async def run_stream(
        self, user_query: str, skip_vagueness: bool = True, use_books: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.time()

        # ── Step 0: Vagueness Evaluation (only if explicitly enabled) ─────────
        if not skip_vagueness:
            v_res = await asyncio.to_thread(
                self.vagueness_detector.evaluate_query, user_query
            )
            if v_res.get("is_vague"):
                yield {
                    "event": "clarification",
                    "questions": v_res.get("questions", []),
                    "reason": v_res.get("reason", ""),
                }
                return

        high_quality_chunks = []
        book_sources = []
        web_results = []

        # ── Step 1: Optional Book Corpus Search ────────────────────────────────
        if use_books:
            yield {
                "event": "status",
                "message": "Step 1: Searching 28 personal finance books for corpus context...",
            }

            sub_queries = await asyncio.to_thread(
                self.multi_query_gen.generate_subqueries, user_query
            )

            retrieval_res = await self.traverser.execute_full_retrieval_async(sub_queries, user_query)
            candidate_books = retrieval_res.get("candidate_books", [])
            traversal_trace = retrieval_res.get("traversal_trace", [])

            yield {
                "event": "traversal",
                "candidate_books": candidate_books,
                "trace": traversal_trace,
            }

            raw_leaf_nodes = retrieval_res.get("leaf_nodes", [])
            top_k_chunks = await asyncio.to_thread(
                self.reranker.rerank_and_filter, raw_leaf_nodes, user_query
            )

            passed_gate, verified_chunks, gate_msg = self.relevance_gate.evaluate_relevance(
                user_query, top_k_chunks
            )

            HIGH_QUALITY_THRESHOLD = 7.0
            high_quality_chunks = [
                c for c in (verified_chunks if passed_gate else [])
                if c.get("relevance_score", 0.0) >= HIGH_QUALITY_THRESHOLD
            ]

            book_sources = [
                {
                    "book_id": c.get("book_id"),
                    "title": c.get("title"),
                    "level": c.get("level"),
                    "pages": f"pp. {c.get('start_page')}-{c.get('end_page')}",
                    "summary": c.get("summary"),
                    "stance": c.get("stance", ""),
                    "relevance_score": c.get("relevance_score", 0.0),
                    "rerank_method": c.get("rerank_method", "llm_pointwise"),
                }
                for c in (verified_chunks if passed_gate else [])
            ]
            yield {"event": "sources", "sources": book_sources, "web_sources": []}

        # ── Step 2: Stream Direct Answer from LLM ────────────────────────────
        yield {"event": "status", "message": "Synthesizing answer with LLM knowledge..."}
        async for chunk_text in self._stream_answer(
            user_query,
            high_quality_chunks,
            web_results,
        ):
            yield {"event": "answer_chunk", "chunk": chunk_text}

        logger.info(f"[NormalAgent] Total pipeline time: {time.time() - start_time:.2f}s")
        yield {"event": "done"}

    async def _stream_answer(
        self,
        query: str,
        book_chunks: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        max_chars = settings.TOKEN_BUDGET_ANSWER_SYNTHESIS * 4

        # Build web search context (primary source of fresh data)
        web_context = ""
        if web_results:
            web_context += "=== LIVE WEB SEARCH RESULTS ===\n"
            for w in web_results:
                snippet = w.get("snippet", "")[:600]
                url = w.get("url", "")
                if snippet.strip():
                    web_context += f"- {snippet}\n"
            web_context = web_context[:max_chars // 2]

        # Build corpus context ONLY from truly high-quality chunks
        # Each chunk already passed the HIGH_QUALITY_THRESHOLD (score >= 7.0)
        # We label them so the LLM can decide if they really address the question
        corpus_context = ""
        if book_chunks:
            corpus_context += "=== BOOK CORPUS EXCERPTS (use ONLY if directly about the user question) ===\n"
            for c in book_chunks:
                score = c.get("relevance_score", 0.0)
                raw = (c.get("raw_text") or c.get("summary", ""))[:600]
                if raw.strip():
                    corpus_context += f"[relevance={score:.1f}] {raw}\n\n"
            corpus_context = corpus_context[:max_chars // 2]

        # Compose context block
        context_block = ""
        if web_context.strip():
            context_block += f"\n\n{web_context}"
        if corpus_context.strip():
            context_block += f"\n\n{corpus_context}"

        context_section = ""
        if context_block.strip():
            context_section = (
                f"\n\n--- SUPPORTING CONTEXT ---"
                f"{context_block}"
                f"\n--- END CONTEXT ---"
            )

        prompt = f"""You are an expert Personal Finance Assistant. Answer the user's question below DIRECTLY and COMPLETELY.

=== STRICT RULES ===
1. READ THE USER QUESTION CAREFULLY. Answer ONLY what was specifically asked.
2. Do NOT include generic financial advice or unrelated paragraphs not directly answering the question.
3. If the Supporting Context contains information directly relevant to the question, use it. If it is about a DIFFERENT topic, IGNORE IT ENTIRELY.
4. Write in clear paragraphs with ## markdown headings. Each paragraph: 3-5 sentences.
5. No bullet-point lists as the primary format. No source citations, no book/author names.
6. End with a ## Summary of 2-3 sentences.
7. If the context is irrelevant or missing, answer from your own expert financial knowledge.

=== USER QUESTION ===
{query}
{context_section}

Now answer the question above directly and thoroughly:
"""

        # ── Execute API call to LLM with fallback support ────────────────────
        failed_provider = None
        if self.llm_client:
            cls_name = type(self.llm_client).__name__.lower()
            if "google" in cls_name or "gemini" in cls_name:
                failed_provider = "gemini"
            elif "groq" in cls_name:
                failed_provider = "groq"
            elif "openai" in cls_name:
                failed_provider = "openai"

        clients_to_try = []
        if self.llm_client:
            clients_to_try.append(self.llm_client)

        if failed_provider:
            from backend.main import get_llm_client
            fallback_client = get_llm_client(skip_providers=[failed_provider])
            if fallback_client:
                logger.info(
                    f"[NormalAgent] Primary provider '{failed_provider}' encountered an error. "
                    f"Adding fallback client '{type(fallback_client).__name__}'."
                )
                clients_to_try.append(fallback_client)

        last_error = None
        for client in clients_to_try:
            if hasattr(client, "astream"):
                try:
                    async for chunk in client.astream(prompt):
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if content:
                            yield content
                    return
                except Exception as e:
                    logger.warning(f"astream failed on {type(client).__name__}: {e}")
                    last_error = e

            if hasattr(client, "stream"):
                try:
                    for chunk in client.stream(prompt):
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if content:
                            yield content
                            await asyncio.sleep(0)
                    return
                except Exception as e:
                    logger.warning(f"stream failed on {type(client).__name__}: {e}")
                    last_error = e

        if not clients_to_try:
            fallback_msg = (
                "⚠️ API Key Error: No active LLM API client. Please ensure a valid API key "
                "(GEMINI_API_KEY, GROQ_API_KEY, XAI_API_KEY, or OPENAI_API_KEY) is set in your .env file."
            )
        else:
            fallback_msg = (
                f"⚠️ API Request Error: Failed to generate response with active LLM clients. "
                f"Details: {last_error if last_error else 'Stream invocation failed'}"
            )
        for word in fallback_msg.split(" "):
            yield word + " "
            await asyncio.sleep(0.01)
