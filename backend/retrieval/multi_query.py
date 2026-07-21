import json
import logging
from typing import List, Optional
from backend.config import settings

logger = logging.getLogger(__name__)

class MultiQueryGenerator:
    """
    Generates 3-5 expanded/reformulated sub-queries representing different angles,
    synonyms, and sub-topics of the original query.
    """

    def __init__(self, llm_client=None, num_queries: int = settings.NUM_SUBQUERIES):
        self.llm_client = llm_client
        self.num_queries = num_queries

    def generate_subqueries(self, original_query: str) -> List[str]:
        prompt = f"""
Given the personal finance user query: "{original_query}"

Generate {self.num_queries} alternative/expanded search queries.
Sub-queries should explore:
1. Core terminology & direct phrasing
2. Practical steps, tactics, or formulas
3. Mindset/behavioral finance principles
4. Differing financial philosophies (e.g. debt payoff vs investing trade-offs)

Return ONLY a JSON array of strings:
["subquery 1", "subquery 2", "subquery 3", "subquery 4"]
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json(res)
        if parsed and isinstance(parsed, list) and len(parsed) > 0:
            return parsed

        # Robust fallback generation if LLM is unavailable
        return [
            original_query,
            f"{original_query} financial rules and strategies",
            f"{original_query} step by step guide and principles",
            f"{original_query} pros cons and trade-offs"
        ][:self.num_queries]

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
            logger.warning(f"MultiQueryGenerator LLM call failed: {e}")
        return None

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
