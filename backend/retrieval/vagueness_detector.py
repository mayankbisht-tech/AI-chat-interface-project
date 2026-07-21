import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class VaguenessDetector:
    """
    Evaluates if user query has sufficient context (goal, amount, timeframe, risk tolerance, age).
    If underspecified, returns 1-2 targeted clarifying questions.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def evaluate_query(self, query: str, context_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        # If user explicitly asks for general guidance or context is already provided
        query_lower = query.lower()
        if any(w in query_lower for w in ["generally", "in general", "overview", "quick summary", "skip"]):
            return {"is_vague": False, "questions": [], "reason": "User requested general answer."}

        prompt = f"""
You are a personal finance assistant. Analyze the following user query:
User Query: "{query}"

Determine if the query is too vague to give customized personal finance advice.
A query is vague if it lacks key context such as:
- Specific financial goal (e.g. buying a house, retirement, paying off debt)
- Time horizon (e.g. 1 year vs 20 years)
- Risk tolerance or current financial situation (e.g. high debt vs large savings)

Output strictly JSON:
{{
  "is_vague": true/false,
  "reason": "short explanation",
  "clarifying_questions": [
    "1-2 specific questions to ask the user to give targeted advice (maximum 2 questions)"
  ]
}}
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json(res)
        if parsed and "is_vague" in parsed:
            return {
                "is_vague": parsed["is_vague"],
                "questions": parsed.get("clarifying_questions", []),
                "reason": parsed.get("reason", "")
            }

        # Rule-based fallback if LLM is unavailable
        words = query.split()
        if len(words) < 5 and not any(char.isdigit() for char in query):
            return {
                "is_vague": True,
                "questions": [
                    "What is your primary financial goal (e.g., debt payoff, home purchase, retirement)?",
                    "What is your investment time horizon and risk tolerance?"
                ],
                "reason": "Query is short and lacks specific goal or timeline."
            }

        return {"is_vague": False, "questions": [], "reason": "Query contains sufficient specificity."}

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
            logger.warning(f"VaguenessDetector LLM call failed: {e}")
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
