import json
import logging
import re
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Set
from backend.config import settings
from backend.ingestion.storage import CorpusStorage
from backend.logging_config import setup_logger

logger = setup_logger("VectorlessTreeTraverser")

STOPWORDS = {
    "what", "is", "the", "difference", "between", "an", "and", "a",
    "according", "to", "in", "of", "for", "on", "dad", "rich", "poor",
    "book", "guide", "how", "why", "does", "do", "can", "should", "would",
    "could", "about", "from", "with", "that", "this", "these", "those",
    "are", "was", "were", "has", "have", "had", "not", "but", "also",
}

# Approximate chars-per-token for token-budget truncation
_CHARS_PER_TOKEN = 4


def _cap_text(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    return text[:max_chars] if len(text) > max_chars else text


class VectorlessTreeTraverser:
    """
    Vectorless PageIndex-style RAG retriever.
    
    Recursively navigates LLM-summarized tree nodes without embeddings.
    Each LLM prompt at each level is token-budget-capped to stay within
    grok-3's 131K context window even when traversing many books.
    
    Level 0: Route across all 28 book summaries (budget: TOKEN_BUDGET_BOOK_ROUTING)
    Level 1: Select chapters per book (budget: TOKEN_BUDGET_CHAPTER_SELECT)
    Level 2: Select sections per chapter (budget: TOKEN_BUDGET_SECTION_SELECT)
    Level 3: Select subsections if present (budget: TOKEN_BUDGET_SECTION_SELECT / 2)
    """

    def __init__(self, storage: CorpusStorage, llm_client=None):
        self.storage = storage
        self.llm_client = llm_client
        self.corpus_books = self.storage.get_all_books()
        self._llm_cache: Dict[str, str] = {}  # Cache LLM responses by prompt hash

    def route_books_level0(self, sub_queries: List[str]) -> List[Dict[str, Any]]:
        """
        Level 0: Feed all 28 book summaries into context and let LLM select
        the most relevant books for the given sub-queries.
        Max books returned: settings.MAX_BOOKS_TO_ROUTE (default 4).
        
        Optimized: Batched single LLM call for all sub-queries to minimize latency.
        """
        selected_book_ids: Set[str] = set()

        # Fast path: exact title / author keyword match in sub-queries
        for sq in sub_queries:
            sq_lower = sq.lower()
            for b in self.corpus_books:
                bt_lower = b["title"].lower()
                if bt_lower in sq_lower:
                    selected_book_ids.add(b["node_id"])
                for author_kw in ["rich dad", "bogle", "ramsey", "graham", "buffett",
                                   "kiyosaki", "lynch", "malkiel", "sethi", "collins",
                                   "housel", "bach", "robbins"]:
                    if author_kw in sq_lower and author_kw in bt_lower:
                        selected_book_ids.add(b["node_id"])

        # Build token-budget-capped corpus index string for LLM routing
        chars_budget = settings.TOKEN_BUDGET_BOOK_ROUTING * _CHARS_PER_TOKEN
        books_summary_str = ""
        per_book_budget = max(200, chars_budget // max(len(self.corpus_books), 1))

        for b in self.corpus_books:
            summary_snippet = (b.get("summary") or "")[:per_book_budget]
            books_summary_str += (
                f"[{b['node_id']}] \"{b['title']}\"\n"
                f"Stance: {b.get('stance', 'N/A')[:120]}\n"
                f"Topics: {', '.join(b.get('topic_tags', []))}\n"
                f"Summary: {summary_snippet}\n\n"
            )

        # Batched LLM routing: single call for ALL sub-queries combined
        # This reduces LLM calls from NUM_SUBQUERIES to 1
        combined_query = " | ".join(sub_queries)
        prompt = f"""You are a financial librarian. Select the most relevant books from the catalog for the queries below.

Queries: "{combined_query}"

Book Catalog:
{books_summary_str}

Rules:
- Select 1 to {settings.MAX_BOOKS_TO_ROUTE} books maximum
- If any query explicitly names a book or author, ALWAYS include that book
- Consider topic overlap, author stance, and philosophy alignment
- For multi-topic queries, select books covering each topic

Output ONLY a JSON array of selected book IDs (exactly as shown in brackets):
["book_id_1", "book_id_2"]
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json(res)
        if parsed and isinstance(parsed, list):
            valid_ids = {b["node_id"] for b in self.corpus_books}
            for b_id in parsed:
                if b_id in valid_ids:
                    selected_book_ids.add(b_id)

        # Keyword fallback if LLM routing returned nothing
        if not selected_book_ids:
            q_words_all = set()
            for sq in sub_queries:
                q_words_all.update(
                    w.lower() for w in re.findall(r'\b\w+\b', sq)
                    if len(w) > 3 and w.lower() not in STOPWORDS
                )
            for b in self.corpus_books:
                tags = [t.lower() for t in b.get("topic_tags", [])]
                title_lower = b["title"].lower()
                if any(w in title_lower or w in " ".join(tags) for w in q_words_all):
                    selected_book_ids.add(b["node_id"])

        # Final safety fallback
        if not selected_book_ids and self.corpus_books:
            selected_book_ids = {b["node_id"] for b in self.corpus_books[:3]}

        # Enforce MAX_BOOKS_TO_ROUTE cap
        candidate_ids = list(selected_book_ids)[: settings.MAX_BOOKS_TO_ROUTE]
        candidate_books = [b for b in self.corpus_books if b["node_id"] in candidate_ids]

        logger.info(
            f"[Level 0 Book Routing] Selected {len(candidate_books)} books: "
            f"{[b['title'] for b in candidate_books]}"
        )
        return candidate_books

    def traverse_book_tree(
        self,
        candidate_book: Dict[str, Any],
        query: str,
        breadth: int = settings.TREE_EXPANSION_BREADTH,
    ) -> List[Dict[str, Any]]:
        """
        Lazy recursive tree traversal for a single candidate book.
        Only expands nodes that are selected by the LLM — never loads irrelevant branches.
        """
        book_id = candidate_book["node_id"]
        chapters = self.storage.get_children(book_id)
        if not chapters:
            # Book has no children — use the book node itself as leaf
            return [self.storage.get_node(book_id)]

        # Level 1: Select relevant chapters (budget: TOKEN_BUDGET_CHAPTER_SELECT)
        selected_chapters = self._select_children(
            chapters, query,
            level_name="Chapter",
            breadth=min(breadth, settings.MAX_CHAPTERS_PER_BOOK),
            token_budget=settings.TOKEN_BUDGET_CHAPTER_SELECT,
        )
        leaf_nodes: List[Dict[str, Any]] = []

        for chap in selected_chapters:
            sections = self.storage.get_children(chap["node_id"])
            if not sections:
                # Chapter is a leaf
                leaf_nodes.append(self.storage.get_node(chap["node_id"]))
            else:
                # Level 2: Select relevant sections (budget: TOKEN_BUDGET_SECTION_SELECT)
                selected_sections = self._select_children(
                    sections, query,
                    level_name="Section",
                    breadth=min(breadth, settings.MAX_SECTIONS_PER_CHAPTER),
                    token_budget=settings.TOKEN_BUDGET_SECTION_SELECT,
                )
                for sec in selected_sections:
                    subsections = self.storage.get_children(sec["node_id"])
                    if not subsections:
                        # Section is a leaf
                        leaf_nodes.append(self.storage.get_node(sec["node_id"]))
                    else:
                        # Level 3: Subsections (budget: TOKEN_BUDGET_SECTION_SELECT / 2)
                        selected_sub = self._select_children(
                            subsections, query,
                            level_name="Subsection",
                            breadth=breadth,
                            token_budget=settings.TOKEN_BUDGET_SECTION_SELECT // 2,
                        )
                        for sub in selected_sub:
                            leaf_nodes.append(self.storage.get_node(sub["node_id"]))

        logger.info(
            f"[Lazy Traversal] '{candidate_book['title']}' → {len(leaf_nodes)} leaf nodes."
        )
        return leaf_nodes

    def execute_full_retrieval(
        self, sub_queries: List[str], query: str
    ) -> Dict[str, Any]:
        """
        Full pipeline: Level 0 book routing → recursive lazy tree traversal per book
        → leaf node collection → deduplication across sub-queries.
        """
        candidate_books = self.route_books_level0(sub_queries)

        all_leaf_nodes: List[Dict[str, Any]] = []
        traversal_trace: List[Dict[str, Any]] = []
        seen_node_ids: Set[str] = set()  # Deduplication across sub-queries

        for book in candidate_books:
            leaves = self.traverse_book_tree(book, query)
            for leaf in leaves:
                if leaf and leaf.get("node_id") not in seen_node_ids:
                    seen_node_ids.add(leaf["node_id"])
                    full_leaf = self.storage.get_node(leaf["node_id"])
                    if full_leaf:
                        # Inject book-level context into leaf for reranker
                        full_leaf["book_id"] = book["node_id"]
                        full_leaf["title"] = full_leaf.get("title") or book["title"]
                        full_leaf["stance"] = full_leaf.get("stance") or book.get("stance", "")
                        full_leaf["topic_tags"] = book.get("topic_tags", [])
                        all_leaf_nodes.append(full_leaf)
                        traversal_trace.append({
                            "book_title": book["title"],
                            "leaf_title": leaf.get("title", ""),
                            "level": leaf.get("level", ""),
                            "pages": f"pp. {leaf.get('start_page', '?')}-{leaf.get('end_page', '?')}",
                        })

        logger.info(
            f"[Full Retrieval] {len(candidate_books)} books → "
            f"{len(all_leaf_nodes)} unique leaf nodes."
        )
        return {
            "candidate_books": [b["title"] for b in candidate_books],
            "leaf_nodes": all_leaf_nodes,
            "traversal_trace": traversal_trace,
        }

    def _select_children(
        self,
        children: List[Dict[str, Any]],
        query: str,
        level_name: str,
        breadth: int,
        token_budget: int = settings.TOKEN_BUDGET_CHAPTER_SELECT,
    ) -> List[Dict[str, Any]]:
        """
        Given a list of child nodes and a query, use LLM to select the most relevant ones.
        Falls back to keyword scoring if LLM is unavailable.
        Token budget is enforced on the prompt.
        """
        # If all children fit within breadth, return all (no LLM call needed)
        if len(children) <= breadth:
            return children

        # Fast path: keyword-based selection to avoid LLM call when strong matches exist
        q_words = [
            w.lower() for w in re.findall(r'\b\w+\b', query)
            if len(w) > 2 and w.lower() not in STOPWORDS
        ]
        HIGH_VALUE_TERMS = {
            "asset", "assets", "liability", "liabilities", "cashflow", "cash flow",
            "debt", "tax", "vtsax", "index", "etf", "compound", "retire", "budget",
            "invest", "dividend", "inflation", "rate", "estate", "stock", "bond",
            "snowball", "envelope", "emergency", "income", "expense", "net worth",
        }
        scored_children = []
        for c in children:
            score = 0.0
            c_text = (c["title"] + " " + c.get("summary", "")).lower()
            for w in q_words:
                if w in c_text:
                    score += 10.0 if w in HIGH_VALUE_TERMS else 3.0
                    if w in c["title"].lower():
                        score += 5.0  # Title match bonus
            scored_children.append((score, c))
        scored_children.sort(key=lambda x: x[0], reverse=True)
        
        # If top candidates have strong scores, use keyword selection directly (skip LLM)
        if scored_children and scored_children[0][0] >= 15.0:
            logger.debug(f"[{level_name}] Keyword fast-path: top score {scored_children[0][0]:.1f}")
            return [item[1] for item in scored_children[:breadth]]

        # Build children summary string within token budget
        chars_budget = token_budget * _CHARS_PER_TOKEN
        per_child_budget = max(150, chars_budget // max(len(children), 1))
        children_str = ""
        for c in children:
            summary_snippet = _cap_text(c.get("summary", ""), max_tokens=per_child_budget // _CHARS_PER_TOKEN)
            children_str += (
                f"[{c['node_id']}] {level_name}: \"{c['title']}\"\n"
                f"Summary: {summary_snippet}\n\n"
            )

        prompt = f"""Select up to {breadth} most relevant {level_name} nodes for the query below.

Query: "{query}"

Available {level_name}s:
{children_str}

Output ONLY a JSON array of selected node IDs (exactly as shown in brackets):
["node_id_1", "node_id_2"]
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json(res)
        if parsed and isinstance(parsed, list):
            valid_ids = {c["node_id"] for c in children}
            selected = [c for c in children if c["node_id"] in parsed and c["node_id"] in valid_ids]
            if selected:
                return selected[:breadth]

        # Domain-term weighted keyword fallback
        return [item[1] for item in scored_children[:breadth]]

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Synchronous LLM call — use _call_llm_async from async contexts."""
        if not self.llm_client:
            return None
        # Cache key from prompt hash
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self._llm_cache:
            logger.debug(f"LLM cache hit: {cache_key}")
            return self._llm_cache[cache_key]
        try:
            if hasattr(self.llm_client, "invoke"):
                res = self.llm_client.invoke(prompt)
                result = res.content if hasattr(res, "content") else str(res)
                self._llm_cache[cache_key] = result
                return result
        except Exception as e:
            logger.warning(f"VectorlessTreeTraverser LLM call failed: {e}")
        return None

    async def _call_llm_async(self, prompt: str) -> Optional[str]:
        """Non-blocking async LLM call — offloads invoke() to thread pool."""
        if not self.llm_client:
            return None
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self._llm_cache:
            logger.debug(f"LLM cache hit (async): {cache_key}")
            return self._llm_cache[cache_key]
        try:
            if hasattr(self.llm_client, "ainvoke"):
                res = await self.llm_client.ainvoke(prompt)
                result = res.content if hasattr(res, "content") else str(res)
            elif hasattr(self.llm_client, "invoke"):
                res = await asyncio.to_thread(self.llm_client.invoke, prompt)
                result = res.content if hasattr(res, "content") else str(res)
            else:
                return None
            self._llm_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"VectorlessTreeTraverser async LLM call failed: {e}")
        return None

    async def _select_children_async(
        self,
        children: List[Dict[str, Any]],
        query: str,
        level_name: str,
        breadth: int,
        token_budget: int = settings.TOKEN_BUDGET_CHAPTER_SELECT,
    ) -> List[Dict[str, Any]]:
        """Async version of _select_children — non-blocking LLM call."""
        if len(children) <= breadth:
            return children

        q_words = [
            w.lower() for w in re.findall(r'\b\w+\b', query)
            if len(w) > 2 and w.lower() not in STOPWORDS
        ]
        HIGH_VALUE_TERMS = {
            "asset", "assets", "liability", "liabilities", "cashflow", "cash flow",
            "debt", "tax", "vtsax", "index", "etf", "compound", "retire", "budget",
            "invest", "dividend", "inflation", "rate", "estate", "stock", "bond",
            "snowball", "envelope", "emergency", "income", "expense", "net worth",
        }
        scored_children = []
        for c in children:
            score = 0.0
            c_text = (c["title"] + " " + c.get("summary", "")).lower()
            for w in q_words:
                if w in c_text:
                    score += 10.0 if w in HIGH_VALUE_TERMS else 3.0
                    if w in c["title"].lower():
                        score += 5.0
            scored_children.append((score, c))
        scored_children.sort(key=lambda x: x[0], reverse=True)

        if scored_children and scored_children[0][0] >= 15.0:
            return [item[1] for item in scored_children[:breadth]]

        chars_budget = token_budget * _CHARS_PER_TOKEN
        per_child_budget = max(150, chars_budget // max(len(children), 1))
        children_str = ""
        for c in children:
            summary_snippet = _cap_text(c.get("summary", ""), max_tokens=per_child_budget // _CHARS_PER_TOKEN)
            children_str += (
                f"[{c['node_id']}] {level_name}: \"{c['title']}\"\n"
                f"Summary: {summary_snippet}\n\n"
            )

        prompt = f"""Select up to {breadth} most relevant {level_name} nodes for the query below.

Query: "{query}"

Available {level_name}s:
{children_str}

Output ONLY a JSON array of selected node IDs (exactly as shown in brackets):
["node_id_1", "node_id_2"]
"""
        res = await self._call_llm_async(prompt)
        parsed = self._parse_json(res)
        if parsed and isinstance(parsed, list):
            valid_ids = {c["node_id"] for c in children}
            selected = [c for c in children if c["node_id"] in parsed and c["node_id"] in valid_ids]
            if selected:
                return selected[:breadth]

        return [item[1] for item in scored_children[:breadth]]

    async def traverse_book_tree_async(
        self,
        candidate_book: Dict[str, Any],
        query: str,
        breadth: int = settings.TREE_EXPANSION_BREADTH,
    ) -> List[Dict[str, Any]]:
        """Async version of traverse_book_tree — non-blocking per-book traversal."""
        book_id = candidate_book["node_id"]
        chapters = self.storage.get_children(book_id)
        if not chapters:
            return [self.storage.get_node(book_id)]

        selected_chapters = await self._select_children_async(
            chapters, query,
            level_name="Chapter",
            breadth=min(breadth, settings.MAX_CHAPTERS_PER_BOOK),
            token_budget=settings.TOKEN_BUDGET_CHAPTER_SELECT,
        )
        leaf_nodes: List[Dict[str, Any]] = []

        section_tasks = []
        for chap in selected_chapters:
            sections = self.storage.get_children(chap["node_id"])
            if not sections:
                leaf_nodes.append(self.storage.get_node(chap["node_id"]))
            else:
                section_tasks.append((chap, sections))

        for chap, sections in section_tasks:
            selected_sections = await self._select_children_async(
                sections, query,
                level_name="Section",
                breadth=min(breadth, settings.MAX_SECTIONS_PER_CHAPTER),
                token_budget=settings.TOKEN_BUDGET_SECTION_SELECT,
            )
            for sec in selected_sections:
                subsections = self.storage.get_children(sec["node_id"])
                if not subsections:
                    leaf_nodes.append(self.storage.get_node(sec["node_id"]))
                else:
                    selected_sub = await self._select_children_async(
                        subsections, query,
                        level_name="Subsection",
                        breadth=breadth,
                        token_budget=settings.TOKEN_BUDGET_SECTION_SELECT // 2,
                    )
                    for sub in selected_sub:
                        leaf_nodes.append(self.storage.get_node(sub["node_id"]))

        logger.info(f"[Async Traversal] '{candidate_book['title']}' → {len(leaf_nodes)} leaf nodes.")
        return leaf_nodes

    async def execute_full_retrieval_async(
        self, sub_queries: List[str], query: str
    ) -> Dict[str, Any]:
        """Fully async pipeline: book routing + parallel per-book traversal."""
        # Step 1: Route books (async LLM call)
        candidate_books = await self._route_books_async(sub_queries)

        # Step 2: Traverse all books in parallel
        tasks = [self.traverse_book_tree_async(book, query) for book in candidate_books]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_leaf_nodes: List[Dict[str, Any]] = []
        traversal_trace: List[Dict[str, Any]] = []
        seen_node_ids: Set[str] = set()

        for book, leaves_or_err in zip(candidate_books, results):
            if isinstance(leaves_or_err, Exception):
                logger.warning(f"Traversal error for '{book['title']}': {leaves_or_err}")
                continue
            for leaf in leaves_or_err:
                if leaf and leaf.get("node_id") not in seen_node_ids:
                    seen_node_ids.add(leaf["node_id"])
                    full_leaf = self.storage.get_node(leaf["node_id"])
                    if full_leaf:
                        full_leaf["book_id"] = book["node_id"]
                        full_leaf["title"] = full_leaf.get("title") or book["title"]
                        full_leaf["stance"] = full_leaf.get("stance") or book.get("stance", "")
                        full_leaf["topic_tags"] = book.get("topic_tags", [])
                        all_leaf_nodes.append(full_leaf)
                        traversal_trace.append({
                            "book_title": book["title"],
                            "leaf_title": leaf.get("title", ""),
                            "level": leaf.get("level", ""),
                            "pages": f"pp. {leaf.get('start_page', '?')}-{leaf.get('end_page', '?')}",
                        })

        logger.info(f"[Async Full Retrieval] {len(candidate_books)} books → {len(all_leaf_nodes)} unique leaf nodes.")
        return {
            "candidate_books": [b["title"] for b in candidate_books],
            "leaf_nodes": all_leaf_nodes,
            "traversal_trace": traversal_trace,
        }

    async def _route_books_async(self, sub_queries: List[str]) -> List[Dict[str, Any]]:
        """Async version of route_books_level0."""
        selected_book_ids: Set[str] = set()
        for sq in sub_queries:
            sq_lower = sq.lower()
            for b in self.corpus_books:
                bt_lower = b["title"].lower()
                if bt_lower in sq_lower:
                    selected_book_ids.add(b["node_id"])
                for author_kw in ["rich dad", "bogle", "ramsey", "graham", "buffett",
                                   "kiyosaki", "lynch", "malkiel", "sethi", "collins",
                                   "housel", "bach", "robbins"]:
                    if author_kw in sq_lower and author_kw in bt_lower:
                        selected_book_ids.add(b["node_id"])

        chars_budget = settings.TOKEN_BUDGET_BOOK_ROUTING * _CHARS_PER_TOKEN
        books_summary_str = ""
        per_book_budget = max(200, chars_budget // max(len(self.corpus_books), 1))
        for b in self.corpus_books:
            summary_snippet = (b.get("summary") or "")[:per_book_budget]
            books_summary_str += (
                f"[{b['node_id']}] \"{b['title']}\"\n"
                f"Stance: {b.get('stance', 'N/A')[:120]}\n"
                f"Topics: {', '.join(b.get('topic_tags', []))}\n"
                f"Summary: {summary_snippet}\n\n"
            )

        combined_query = " | ".join(sub_queries)
        prompt = f"""You are a financial librarian. Select the most relevant books from the catalog for the queries below.

Queries: "{combined_query}"

Book Catalog:
{books_summary_str}

Rules:
- Select 1 to {settings.MAX_BOOKS_TO_ROUTE} books maximum
- If any query explicitly names a book or author, ALWAYS include that book
- Consider topic overlap, author stance, and philosophy alignment
- For multi-topic queries, select books covering each topic

Output ONLY a JSON array of selected book IDs (exactly as shown in brackets):
["book_id_1", "book_id_2"]
"""
        res = await self._call_llm_async(prompt)
        parsed = self._parse_json(res)
        if parsed and isinstance(parsed, list):
            valid_ids = {b["node_id"] for b in self.corpus_books}
            for b_id in parsed:
                if b_id in valid_ids:
                    selected_book_ids.add(b_id)

        if not selected_book_ids:
            q_words_all = set()
            for sq in sub_queries:
                q_words_all.update(
                    w.lower() for w in re.findall(r'\b\w+\b', sq)
                    if len(w) > 3 and w.lower() not in STOPWORDS
                )
            for b in self.corpus_books:
                tags = [t.lower() for t in b.get("topic_tags", [])]
                title_lower = b["title"].lower()
                if any(w in title_lower or w in " ".join(tags) for w in q_words_all):
                    selected_book_ids.add(b["node_id"])

        if not selected_book_ids and self.corpus_books:
            selected_book_ids = {b["node_id"] for b in self.corpus_books[:2]}

        candidate_ids = list(selected_book_ids)[:settings.MAX_BOOKS_TO_ROUTE]
        candidate_books = [b for b in self.corpus_books if b["node_id"] in candidate_ids]
        logger.info(f"[Async Level 0] Selected {len(candidate_books)} books: {[b['title'] for b in candidate_books]}")
        return candidate_books

    def _parse_json(self, response: Optional[str]) -> Optional[List[str]]:
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
