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

logger = setup_logger("DeepResearchAgent")

TEMPORAL_KEYWORDS = [
    "current", "2026", "2025", "2024", "today", "latest", "now",
    "tax rate", "interest rate", "fed", "federal reserve", "inflation rate",
    "bracket", "cpi", "market", "stock price", "rate", "nvidia", "sp500",
]


class DeepResearchAgent:
    """
    Deep Research Agent — Multi-sub-question iterative retrieval with gap detection.

    Execution Pipeline:
    1. Query Received → Check vagueness.
    2. Decompose into research sub-questions.
    3. Step 1: For each sub-question, check if personal finance books are needed.
    4. Step 2: Check LLM + Tavily for internet search (gap detection & live data).
    5. Step 3: Stream deep report in proper paragraph format with system message (no references/page numbers).
    """

    def __init__(self, storage: CorpusStorage, llm_client=None):
        self.storage = storage
        self.llm_client = llm_client
        self.vagueness_detector = VaguenessDetector(llm_client)
        self.multi_query_gen = MultiQueryGenerator(llm_client)
        self.traverser = VectorlessTreeTraverser(storage, llm_client)
        self.reranker = get_reranker(
            use_cross_encoder=settings.DEEP_RESEARCH_CROSS_ENCODER,
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
        if not skip_vagueness:
            # Offload blocking LLM call to thread pool
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

        yield {"event": "status", "message": "Decomposing query into research sub-questions..."}
        # Decompose query non-blocking
        sub_questions = await asyncio.to_thread(self._decompose_query, user_query)
        yield {
            "event": "status",
            "message": f"Decomposed into {len(sub_questions)} sub-questions. Beginning deep research...",
        }

        research_results: List[Dict[str, Any]] = []
        all_book_sources: List[Dict[str, Any]] = []
        all_web_sources: List[Dict[str, Any]] = []
        all_traversal_traces: List[Dict[str, Any]] = []

        for idx, sq in enumerate(sub_questions, 1):
            yield {
                "event": "status",
                "message": f"[{idx}/{len(sub_questions)}] Step 1: Checking book corpus for \"{sq[:50]}\"...",
            }

            # Non-blocking sub-query generation and async tree traversal
            sq_subqueries = await asyncio.to_thread(
                self.multi_query_gen.generate_subqueries, sq
            )
            ret_res = await self.traverser.execute_full_retrieval_async(sq_subqueries, sq)
            leaf_nodes = ret_res.get("leaf_nodes", [])
            all_traversal_traces.extend(ret_res.get("traversal_trace", []))

            # Reranker is sync — offload to thread pool
            top_chunks = await asyncio.to_thread(
                self.reranker.rerank_and_filter, leaf_nodes, sq
            )
            passed_gate, verified_chunks, gate_msg = self.relevance_gate.evaluate_relevance(
                sq, top_chunks
            )
            if passed_gate:
                all_book_sources.extend(verified_chunks)

            # Step 2: Tavily internet search (non-blocking)
            is_temporal = any(kw in sq.lower() for kw in TEMPORAL_KEYWORDS)
            corpus_insufficient = not passed_gate
            needs_web = is_temporal or corpus_insufficient

            web_results = []
            if needs_web:
                gap_reason = "live market data" if is_temporal else "internet search gap fill"
                yield {
                    "event": "status",
                    "message": f"[{idx}/{len(sub_questions)}] Step 2: Tavily internet search ({gap_reason})...",
                }
                # Tavily is sync HTTP — offload to thread pool
                web_results = await asyncio.to_thread(self.tavily_tool.search, sq)
                all_web_sources.extend(web_results)

            research_results.append({
                "sub_question": sq,
                "book_chunks": verified_chunks if passed_gate else [],
                "web_results": web_results,
                "searched_web": needs_web,
                "corpus_passed": passed_gate,
            })

        # Emit UI Side Panel data
        unique_books_traversed = list({t["book_title"] for t in all_traversal_traces})
        yield {
            "event": "traversal",
            "candidate_books": unique_books_traversed,
            "trace": all_traversal_traces,
        }

        book_source_panel = [
            {
                "book_id": c.get("book_id"),
                "title": c.get("title"),
                "level": c.get("level"),
                "pages": f"pp. {c.get('start_page')}-{c.get('end_page')}",
                "summary": c.get("summary"),
                "stance": c.get("stance", ""),
                "relevance_score": c.get("relevance_score", 0.0),
                "rerank_method": c.get("rerank_method", "cross_encoder"),
            }
            for c in all_book_sources
        ]
        yield {
            "event": "sources",
            "sources": book_source_panel,
            "web_sources": all_web_sources,
        }

        # Step 3: Stream Deep Report in proper paragraph format
        yield {"event": "status", "message": "Step 3: Synthesizing report in proper paragraph format..."}
        async for chunk_text in self._stream_deep_report(user_query, research_results):
            yield {"event": "answer_chunk", "chunk": chunk_text}

        yield {"event": "done"}

    def _decompose_query(self, query: str) -> List[str]:
        prompt = f"""Decompose the following complex financial question into 2 to 4 distinct, focused research sub-questions.

Query: "{query}"

Return STRICTLY a JSON array of strings:
["sub-question 1", "sub-question 2", "sub-question 3"]
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json_list(res)
        if parsed and isinstance(parsed, list) and len(parsed) >= 2:
            return parsed[:4]

        return [
            f"What are the foundational principles and key frameworks for {query}?",
            f"What are the practical action steps and common mistakes to avoid for {query}?",
            f"What do different financial philosophies say about {query}?",
        ]

    async def _stream_deep_report(
        self, query: str, research_results: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        sections_content = ""
        for idx, res in enumerate(research_results, 1):
            sq = res["sub_question"]
            b_chunks = res["book_chunks"]
            w_results = res["web_results"]

            sections_content += f"### Research Section {idx}: {sq}\n\n"

            if b_chunks:
                sections_content += "Book Findings:\n"
                for c in b_chunks[:3]:
                    raw = (c.get("raw_text") or c.get("summary", ""))[:800]
                    sections_content += f"- Content: {raw}\n\n"

            if w_results:
                sections_content += "Internet Web Search Findings (Tavily):\n"
                for w in w_results[:3]:
                    sections_content += f"- Web Content: {w.get('snippet', '')[:500]}\n\n"

        max_chars = settings.TOKEN_BUDGET_ANSWER_SYNTHESIS * 4
        sections_content = sections_content[:max_chars]

        prompt = f"""You are a Senior Financial Analyst producing a Deep Research Report. Answer the user's question DIRECTLY with expert depth.

CRITICAL RULES:
1. Answer the EXACT research question asked. Stay precisely on topic.
2. Write in clear, well-developed PARAGRAPHS with ## markdown section headings.
3. Each paragraph: 3-5 substantive sentences — analytical and thorough, no bullet-point lists.
4. Use the research findings below ONLY where they are directly relevant to the specific question. Discard off-topic content.
5. If research findings are not relevant to a section, draw on your own expert knowledge.
6. DO NOT mention sources, book titles, author names, page numbers, or citations anywhere.
7. End with a ## Executive Summary & Conclusion of 3-4 strong paragraphs distilling all key insights.

User Research Request: "{query}"

Research Findings (use only what is directly relevant):
{sections_content}

Write a comprehensive, on-topic research report:
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
                    f"[DeepResearchAgent] Primary provider '{failed_provider}' encountered an error. "
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

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.llm_client:
            return None
        try:
            if hasattr(self.llm_client, "invoke"):
                res = self.llm_client.invoke(prompt)
                return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.warning(f"DeepResearchAgent LLM call failed: {e}")
        return None

    def _parse_json_list(self, response: Optional[str]) -> Optional[List[str]]:
        if not response:
            return None
        try:
            cleaned = response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            return None
