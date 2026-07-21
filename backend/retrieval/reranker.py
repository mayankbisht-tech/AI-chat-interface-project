import json
import logging
from typing import List, Dict, Any, Optional
from backend.config import settings
from backend.logging_config import setup_logger

logger = setup_logger("Reranker")

# Approximate chars-per-token for token-budget truncation
_CHARS_PER_TOKEN = 4


def _cap_text(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    return text[:max_chars] if len(text) > max_chars else text


class VectorlessReranker:
    """
    Pointwise LLM relevance reranker (0-10 scale).
    Used for the Normal Agent (fast path — stays under 20s latency target).
    Deduplicates candidates while preserving differing author viewpoints.
    """

    def __init__(self, llm_client=None, top_k: int = settings.TOP_K_RERANK):
        self.llm_client = llm_client
        self.top_k = top_k

    def rerank_and_filter(
        self, candidate_chunks: List[Dict[str, Any]], original_query: str
    ) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []

        # 1. Deduplicate by node_id / book + start_page
        unique_chunks: Dict[str, Dict[str, Any]] = {}
        for c in candidate_chunks:
            key = f"{c.get('book_id', '')}_{c.get('node_id', '')}_{c.get('start_page', 0)}"
            if key not in unique_chunks:
                unique_chunks[key] = c
        chunks_list = list(unique_chunks.values())

        # 2. Batch score ALL chunks in a single LLM call (reduces quota usage from N to 1)
        scored_chunks = self._score_chunks_batch(chunks_list, original_query)

        # 3. Sort strictly by relevance score descending
        scored_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        if scored_chunks:
            logger.info(
                f"[LLM Reranker] Scored {len(scored_chunks)} chunks in 1 batch call. "
                f"Top score: {scored_chunks[0]['relevance_score']:.1f} — "
                f"'{scored_chunks[0].get('title', '')[:40]}'"
            )

        return scored_chunks[: self.top_k]

    def _text_match_score(self, chunk: Dict[str, Any], query: str) -> float:
        """Fast text-matching heuristic fallback — no LLM call."""
        q_words = set(w.lower() for w in query.split() if len(w) > 3)
        text = (
            chunk.get("title", "")
            + " "
            + chunk.get("summary", "")
            + " "
            + chunk.get("raw_text", "")
        ).lower()
        matches = sum(
            3 if (w in chunk.get("title", "").lower() or w in chunk.get("summary", "").lower()) else 1
            for w in q_words
            if w in text
        )
        return min(10.0, float(matches * 2.0))

    def _score_chunks_batch(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Score ALL chunks in a single LLM call (batch scoring).
        Falls back to text-matching if LLM is unavailable or fails.
        This replaces N per-chunk LLM calls with exactly 1 call.
        """
        if not chunks:
            return []

        if not self.llm_client:
            # No LLM — use text matching for all chunks
            for chunk in chunks:
                chunk["relevance_score"] = self._text_match_score(chunk, query)
                chunk["rerank_method"] = "text_match"
            return chunks

        # Build batch prompt listing all chunks
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            summary = chunk.get("summary", "")[:300]
            snippet = (chunk.get("raw_text", "") or "")[:300]
            chunks_text += (
                f"[{i}] Book: {chunk.get('title', 'Unknown')[:60]}\n"
                f"Summary: {summary}\n"
                f"Text: {snippet}\n\n"
            )

        prompt = f"""Rate the relevance of each book excerpt below for answering the user's financial question.
User Question: "{query}"

Score criteria: 9-10=directly answers, 7-8=highly relevant, 5-6=partial, 3-4=tangential, 0-2=off-topic

Excerpts:
{chunks_text}
Output ONLY a JSON array of scores (one float per excerpt, in order):
[7.5, 9.0, 4.0, ...]
"""
        try:
            res = self._call_llm(prompt)
            if res:
                cleaned = res.strip()
                if "```" in cleaned:
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                scores = json.loads(cleaned)
                if isinstance(scores, list) and len(scores) == len(chunks):
                    for chunk, score in zip(chunks, scores):
                        try:
                            chunk["relevance_score"] = min(10.0, max(0.0, float(score)))
                        except (ValueError, TypeError):
                            chunk["relevance_score"] = self._text_match_score(chunk, query)
                        chunk["rerank_method"] = "llm_batch"
                    return chunks
        except Exception as e:
            logger.warning(f"Batch reranker LLM call failed, using text-match fallback: {e}")

        # Fallback: text-match scoring for all chunks
        for chunk in chunks:
            chunk["relevance_score"] = self._text_match_score(chunk, query)
            chunk["rerank_method"] = "text_match"
        return chunks

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.llm_client:
            return None
        try:
            if hasattr(self.llm_client, "invoke"):
                res = self.llm_client.invoke(prompt)
                content = res.content if hasattr(res, "content") else str(res)
                if isinstance(content, list):
                    return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                return str(content)
        except Exception as e:
            logger.warning(f"VectorlessReranker LLM call failed: {e}")
        return None

    def _parse_json(self, response: Optional[str]) -> Optional[Dict[str, Any]]:
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


class CrossEncoderReranker:
    """
    Cross-encoder re-ranker using sentence-transformers (no vector DB).
    Used exclusively at the re-rank stage in Deep Research mode.
    
    The cross-encoder takes (query, passage) pairs and scores them directly,
    providing more accurate relevance than pointwise LLM scoring.
    Note: this is strictly inference-only — no embeddings are stored anywhere.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, lightweight, ~80MB)
    """

    def __init__(
        self,
        model_name: str = settings.CROSS_ENCODER_MODEL,
        top_k: int = settings.TOP_K_RERANK,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"[CrossEncoder] Loading model '{self.model_name}'...")
                self._model = CrossEncoder(self.model_name, max_length=512)
                logger.info(f"[CrossEncoder] Model loaded successfully.")
            except ImportError:
                logger.error(
                    "[CrossEncoder] sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"[CrossEncoder] Failed to load model '{self.model_name}': {e}")
                raise
        return self._model

    def rerank_and_filter(
        self, candidate_chunks: List[Dict[str, Any]], original_query: str
    ) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []

        # 1. Deduplicate by node_id / book + start_page
        unique_chunks: Dict[str, Dict[str, Any]] = {}
        for c in candidate_chunks:
            key = f"{c.get('book_id', '')}_{c.get('node_id', '')}_{c.get('start_page', 0)}"
            if key not in unique_chunks:
                unique_chunks[key] = c
        chunks_list = list(unique_chunks.values())

        # 2. Build (query, passage) pairs for cross-encoder
        # Passage = summary + first 400 chars of raw_text for efficiency
        pairs = []
        for chunk in chunks_list:
            passage = (
                f"{chunk.get('title', '')}. "
                f"{chunk.get('summary', '')} "
                f"{chunk.get('raw_text', '')[:400]}"
            ).strip()
            pairs.append([original_query, passage])

        # 3. Score all pairs in a single batch (efficient)
        try:
            model = self._get_model()
            raw_scores = model.predict(pairs)
        except Exception as e:
            logger.error(f"[CrossEncoder] Prediction failed: {e}. Falling back to text-match scoring.")
            # Fallback: text-match scoring
            raw_scores = []
            q_words = set(w.lower() for w in original_query.split() if len(w) > 3)
            for chunk in chunks_list:
                text = (
                    chunk.get("title", "") + " " + chunk.get("summary", "")
                ).lower()
                score = sum(2.0 for w in q_words if w in text)
                raw_scores.append(score)

        # 4. Normalize scores to 0-10 range
        if len(raw_scores) > 0:
            min_s = float(min(raw_scores))
            max_s = float(max(raw_scores))
            score_range = max_s - min_s if max_s != min_s else 1.0
            normalized_scores = [float(10.0 * (s - min_s) / score_range) for s in raw_scores]
        else:
            normalized_scores = [5.0] * len(chunks_list)

        # 5. Assign scores to chunks
        for chunk, norm_score in zip(chunks_list, normalized_scores):
            chunk["relevance_score"] = round(norm_score, 2)
            chunk["rerank_method"] = "cross_encoder"

        # 6. Sort by score descending — preserving unique books/stances
        chunks_list.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        if chunks_list:
            logger.info(
                f"[CrossEncoder] Reranked {len(chunks_list)} chunks. "
                f"Top score: {chunks_list[0]['relevance_score']:.2f} — "
                f"'{chunks_list[0].get('title', '')[:40]}'"
            )

        return chunks_list[: self.top_k]


def get_reranker(use_cross_encoder: bool = False, llm_client=None, top_k: int = settings.TOP_K_RERANK):
    """
    Factory function: returns the appropriate reranker based on mode.
    - Normal Agent: VectorlessReranker (LLM pointwise, fast)
    - Deep Research Agent: CrossEncoderReranker (sentence-transformers, accurate)
    """
    if use_cross_encoder and settings.CROSS_ENCODER_MODEL:
        try:
            logger.info("[Reranker Factory] Using CrossEncoderReranker for Deep Research mode.")
            return CrossEncoderReranker(model_name=settings.CROSS_ENCODER_MODEL, top_k=top_k)
        except Exception as e:
            logger.warning(
                f"[Reranker Factory] CrossEncoder unavailable ({e}), "
                f"falling back to VectorlessReranker."
            )

    logger.info("[Reranker Factory] Using VectorlessReranker (LLM pointwise).")
    return VectorlessReranker(llm_client=llm_client, top_k=top_k)
