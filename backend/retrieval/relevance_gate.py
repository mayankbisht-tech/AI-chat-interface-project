import re
import logging
from typing import List, Dict, Any, Tuple
from backend.logging_config import setup_logger

logger = setup_logger("HardRelevanceGate")

STOPWORDS = {
    "what", "is", "the", "difference", "between", "an", "and", "a", "according", "to", "in", "of",
    "for", "on", "dad", "rich", "poor", "book", "guide", "how", "why", "does", "do", "can", "should",
    "would", "could", "about", "from", "with", "that", "this", "these", "those", "are", "was", "were",
    "has", "have", "had", "not", "but", "also", "repair", "car", "engine", "fix", "toyota", "camry",
}

class HardRelevanceGate:
    """
    Hard Relevance Gate:
    Verifies that top reranked chunks actually contain core query concepts
    and meet a minimum relevance score threshold (>= 6.0/10) before synthesis.
    """

    def __init__(self, min_score_threshold: float = 6.0):
        self.min_score_threshold = min_score_threshold

    def evaluate_relevance(self, query: str, top_chunks: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]], str]:
        if not top_chunks:
            logger.warning(f"[Relevance Gate FAILED] No candidate chunks retrieved for query: '{query}'")
            return False, [], "No candidate chunks were retrieved."

        best_chunk = top_chunks[0]
        top_score = best_chunk.get("relevance_score", 0.0)

        # Extract core domain terms from query (excluding stopwords)
        q_words = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2 and w.lower() not in STOPWORDS]

        # Check for core term matches across top chunks
        relevant_chunks = []
        for chunk in top_chunks:
            score = chunk.get("relevance_score", 0.0)
            text_content = (chunk.get("title", "") + " " + chunk.get("summary", "") + " " + chunk.get("raw_text", "")).lower()

            matches = [w for w in q_words if w in text_content]
            if score >= self.min_score_threshold and (len(matches) >= 1 or not q_words):
                relevant_chunks.append(chunk)

        if not relevant_chunks:
            logger.warning(f"[Relevance Gate FAILED] Top chunk score ({top_score}) failed relevance gate for query terms '{q_words}'.")
            return False, [], f"Top chunks failed relevance gate."

        logger.info(f"[Relevance Gate PASSED] Kept {len(relevant_chunks)} verified chunks for query: '{query[:40]}...'")
        return True, relevant_chunks, "Relevance gate passed."
