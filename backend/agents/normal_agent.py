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

# Book title keywords — auto-trigger corpus search even when use_books=False
BOOK_TRIGGER_KEYWORDS = [
    "rich dad", "poor dad", "intelligent investor", "psychology of money",
    "simple path to wealth", "i will teach you to be rich", "total money makeover",
    "one up on wall street", "random walk", "little book", "common sense investing",
    "money master", "millionaire next door", "automatic millionaire", "barefoot investor",
    "book on rental", "essays of warren buffett", "warren buffett way", "bogleheads",
    "broke millennial", "financial freedom", "latte factor", "index card",
    "your money or your life", "behavioural investing", "common stocks",
    "think and grow rich", "science of getting rich", "richest man in babylon",
    "wealthy barber", "kiyosaki", "graham", "buffett", "bogle", "ramsey",
    "sethi", "collins", "housel", "lynch", "malkiel", "fisher", "montier",
]

LATENCY_TARGET_SECONDS = 18


class NormalAgent:
    """
    Normal Agent — Vectorless PageIndex RAG Assistant.

    Execution Pipeline:
    1. Query received.
    2. Step 1: Tavily web search (always for temporal queries; optional otherwise).
    3. Step 2: Optional book corpus search (tree traversal + reranking).
    4. Step 3: Stream answer using LLM with web + corpus context.
    5. Step 4: Append a "Book Wisdom" insight drawn from the corpus (chapter/quote hint).
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
        self,
        user_query: str,
        skip_vagueness: bool = True,
        use_books: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.time()

        # ── Step 0: Optional vagueness check ──────────────────────────────
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

        web_results: List[Dict[str, Any]] = []
        high_quality_chunks: List[Dict[str, Any]] = []
        book_sources: List[Dict[str, Any]] = []

        # Auto-enable books if query mentions a known book title/author
        query_lower = user_query.lower()
        if not use_books and any(kw in query_lower for kw in BOOK_TRIGGER_KEYWORDS):
            use_books = True
            logger.info("[NormalAgent] Auto-enabled book corpus search (book title detected).")

        # ── Step 1: Tavily web search ──────────────────────────────────────
        is_temporal = any(kw in user_query.lower() for kw in TEMPORAL_KEYWORDS)
        # Don't web-search for greetings or very short queries
        is_trivial = len(user_query.strip().split()) <= 3
        if not is_trivial and (is_temporal or settings.ENABLE_WEB_SEARCH_FALLBACK):
            reason = "live data needed" if is_temporal else "enriching answer"
            yield {
                "event": "status",
                "message": f"Searching the web for current information ({reason})...",
            }
            web_results = await asyncio.to_thread(self.tavily_tool.search, user_query)

        # ── Step 2: Optional book corpus search ───────────────────────────
        if use_books:
            yield {
                "event": "status",
                "message": "Searching 28 personal finance books for relevant insights...",
            }

            sub_queries = await asyncio.to_thread(
                self.multi_query_gen.generate_subqueries, user_query
            )

            retrieval_res = await self.traverser.execute_full_retrieval_async(
                sub_queries, user_query
            )
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
            yield {"event": "sources", "sources": book_sources, "web_sources": web_results}
        else:
            # Still emit sources for web results
            if web_results:
                yield {"event": "sources", "sources": [], "web_sources": web_results}

        # ── Step 3: Stream main answer ─────────────────────────────────────
        yield {"event": "status", "message": "Composing your answer..."}
        async for chunk_text in self._stream_answer(
            user_query,
            high_quality_chunks,
            web_results,
            conversation_history=conversation_history or [],
            user_profile=user_profile or "",
        ):
            yield {"event": "answer_chunk", "chunk": chunk_text}

        # ── Step 4: Book wisdom snippet (non-blocking, fire-and-collect) ───
        if high_quality_chunks:
            wisdom = await asyncio.to_thread(
                self._build_book_wisdom, user_query, high_quality_chunks
            )
            if wisdom:
                yield {"event": "answer_chunk", "chunk": wisdom}

        logger.info(f"[NormalAgent] Total pipeline time: {time.time() - start_time:.2f}s")
        yield {"event": "done"}

    # ── Main answer streamer ──────────────────────────────────────────────────

    async def _stream_answer(
        self,
        query: str,
        book_chunks: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
        user_profile: str,
    ) -> AsyncGenerator[str, None]:
        max_chars = settings.TOKEN_BUDGET_ANSWER_SYNTHESIS * 4

        # Build web context (primary — freshest data)
        web_context = ""
        if web_results:
            web_context += "=== LIVE WEB SEARCH RESULTS ===\n"
            for w in web_results:
                snippet = w.get("snippet", "")[:600]
                if snippet.strip():
                    web_context += f"- {snippet}\n"
            web_context = web_context[: max_chars // 2]

        # Build corpus context (high-quality chunks only)
        corpus_context = ""
        if book_chunks:
            corpus_context += "=== BOOK CORPUS EXCERPTS (use ONLY if directly about the user question) ===\n"
            for c in book_chunks:
                score = c.get("relevance_score", 0.0)
                raw = (c.get("raw_text") or c.get("summary", ""))[:500]
                if raw.strip():
                    corpus_context += f"[relevance={score:.1f}] {raw}\n\n"
            corpus_context = corpus_context[: max_chars // 2]

        # Build conversation history context (last 6 turns)
        history_context = ""
        if conversation_history:
            recent = conversation_history[-6:]
            history_context = "\n".join(
                f"{t['role'].upper()}: {t['content'][:300]}" for t in recent
            )

        # Only inject long-term profile for substantive questions, not greetings
        is_short_greeting = len(query.strip().split()) <= 3
        profile_section = ""
        if (
            user_profile
            and user_profile.strip()
            and user_profile.strip() != "No profile data yet."
            and not is_short_greeting
        ):
            profile_section = (
                f"\n\n=== USER BACKGROUND (from past sessions — treat as context, not current facts) ===\n"
                f"{user_profile[:400]}\n==="
            )

        # Assemble supporting context block
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

        history_section = ""
        if history_context.strip():
            history_section = f"\n\n--- CONVERSATION HISTORY (most recent) ---\n{history_context}\n---"

        prompt = f"""You are an expert Personal Finance Assistant. Answer the user's question directly and concisely.

RULES:
1. Answer ONLY what was asked. Stay on topic.
2. Use the Conversation History to maintain context — reference prior messages naturally when relevant.
3. The USER BACKGROUND section (if present) is from past sessions. Use it only if the current question is clearly related — never inject names or past topics into greetings or unrelated answers.
4. If Supporting Context is relevant, use it. If not, ignore it.
5. Match length to complexity: short question = short answer, complex question = detailed answer. Do NOT pad with filler.
6. Use ## markdown headings only for longer answers that need structure. Skip headings for short answers.
7. No source citations, no book/author names in the answer.
8. For simple greetings like "hi" or "hello", just greet warmly and offer help — do NOT mention names or topics from past sessions unless the user brings them up.{profile_section}

USER QUESTION: {query}
{history_section}
{context_section}

Answer:
"""

        yield_count = 0
        async for chunk_text in self._stream_from_llm(prompt):
            yield chunk_text
            yield_count += 1

    # ── Book wisdom builder ───────────────────────────────────────────────────

    def _build_book_wisdom(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Picks the single highest-scoring chunk and asks the LLM to craft a short
        'Book Wisdom' callout — a chapter heading / principle / memorable quote-like
        insight relevant to the query.  Returns "" on failure.
        """
        if not chunks or not self.llm_client:
            return ""

        best = max(chunks, key=lambda c: c.get("relevance_score", 0.0))
        book_title = best.get("title", "a personal finance book")
        # Use summary as the text source (raw_text may be very long)
        source_text = (best.get("summary") or best.get("raw_text", ""))[:600]

        if not source_text.strip():
            return ""

        prompt = f"""You are a financial book curator.

The user asked: "{query}"

Here is a relevant excerpt from "{book_title}":
\"\"\"
{source_text}
\"\"\"

Write a short "Book Wisdom" callout (2-3 sentences max) that:
1. Captures the key principle from this excerpt relevant to the user's question.
2. Feels like a memorable, inspiring insight — similar to a chapter heading or key lesson.
3. Does NOT quote the text verbatim.  Paraphrase the idea in your own words.
4. Start with a relevant emoji and the label  📖 **Book Wisdom** on its own line.

Example format:
📖 **Book Wisdom**
*"Pay yourself first — treat saving like a non-negotiable bill to yourself. The Automatic Millionaire argues that automation is the bridge between intention and action."*

Now write the callout for the excerpt above:"""

        try:
            if hasattr(self.llm_client, "invoke"):
                res = self.llm_client.invoke(prompt)
                content = res.content if hasattr(res, "content") else str(res)
                content = content.strip()
                if content:
                    return f"\n\n---\n{content}\n"
        except Exception as e:
            logger.warning(f"[NormalAgent] Book wisdom LLM call failed: {e}")
        return ""

    # ── LLM streaming helper ──────────────────────────────────────────────────

    async def _stream_from_llm(self, prompt: str) -> AsyncGenerator[str, None]:
        """Try primary LLM client, then fall back to alternatives."""
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
                f"⚠️ API Request Error: Failed to generate response. "
                f"Details: {last_error if last_error else 'Stream invocation failed'}"
            )
        for word in fallback_msg.split(" "):
            yield word + " "
            await asyncio.sleep(0.01)
